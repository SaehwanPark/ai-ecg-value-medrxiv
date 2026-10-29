"""Post-hoc mortality sensitivity using frozen scores and the frozen ECG lexicon.

Decision D1 freezes the development lexicon hash. Decision D3 freezes the
80th-percentile cutoffs on the full paired-valid locked test. A restriction to
`no_overt_major` changes case mix; it does not mediate or explain the original
B−A comparison. `unknown` is not a normal ECG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import polars as pl

from ecg_rep.analysis import representation_gain
from ecg_rep.clinical_setting import complementary_suppress_status, wilson_interval
from ecg_rep.ecg_phenotypes import (
  SUMMARY_LABELS,
  SUMMARY_NEGATIVE,
  SUMMARY_UNKNOWN,
  load_phenotype_dictionary,
  phenotype_frame,
)
from ecg_rep.probe import bootstrap_paired_difference, evaluate_predictions

FROZEN_LEXICON_HASH = "50077ea8055963a6290a7d168fdbbd6c9de3ed9978fe175de0323def0454cd0b"
FROZEN_LEXICON_VERSION = "clinical-review-pending-v0"
HIGH_QUANTILE = 0.8
RISK_GROUPS = ("concordant_low", "a_low_b_high", "a_high_b_low", "concordant_high")
METRIC_COLUMNS = (
  "auroc_a",
  "auroc_b",
  "auprc_a",
  "auprc_b",
  "delta_auroc_b_minus_a",
  "delta_auroc_ci_low",
  "delta_auroc_ci_high",
  "delta_auprc_b_minus_a",
  "delta_auprc_ci_low",
  "delta_auprc_ci_high",
  "brier_a",
  "brier_b",
)


def assert_frozen_lexicon(dictionary_path: Path) -> str:
  """Refuse an outcome join if the lexicon file no longer matches the development freeze."""
  dictionary = load_phenotype_dictionary(dictionary_path)
  if dictionary.version != FROZEN_LEXICON_VERSION or dictionary.rule_hash != FROZEN_LEXICON_HASH:
    raise RuntimeError(
      "The ECG lexicon no longer matches the development freeze "
      f"{FROZEN_LEXICON_VERSION} / {FROZEN_LEXICON_HASH}. "
      "Refusing to join it to mortality."
    )
  return dictionary.rule_hash


def frozen_risk_cuts(
  scores_a: np.ndarray,
  scores_b: np.ndarray,
  high_quantile: float = HIGH_QUANTILE,
) -> tuple[float, float]:
  """Cutoffs from the full comparison population, not from a later slice."""
  if not 0.0 < high_quantile < 1.0:
    raise ValueError("high_quantile must be between 0 and 1.")
  return float(np.quantile(scores_a, high_quantile)), float(np.quantile(scores_b, high_quantile))


def apply_frozen_risk_groups(
  scores_a: np.ndarray,
  scores_b: np.ndarray,
  a_cut: float,
  b_cut: float,
) -> np.ndarray:
  """Label concordance with cutoffs that were fixed outside the slice."""
  if len(scores_a) != len(scores_b):
    raise ValueError("A and B scores must describe the same rows.")
  a_high = np.asarray(scores_a) >= a_cut
  b_high = np.asarray(scores_b) >= b_cut
  labels = np.empty(len(scores_a), dtype=object)
  labels[(~a_high) & (~b_high)] = "concordant_low"
  labels[(~a_high) & b_high] = "a_low_b_high"
  labels[a_high & (~b_high)] = "a_high_b_low"
  labels[a_high & b_high] = "concordant_high"
  return labels.astype(str)


def coalesce_phenotype(summary: str | None) -> str:
  """A missing machine report is unknown, not the absence of an overt phrase."""
  if summary in SUMMARY_LABELS:
    return str(summary)
  return SUMMARY_UNKNOWN


def _hide_until_remainder_is_safe(frame: pl.DataFrame, group_cols: tuple[str, ...]) -> pl.DataFrame:
  """Hide further denominators when the concealed piece of a partition is still under 11.

  A published total minus the visible denominators would otherwise equal that small piece.
  """
  work = frame.with_row_index("_row")
  status = {int(row["_row"]): str(row["suppress_status"]) for row in work.iter_rows(named=True)}
  grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
  for row in work.iter_rows(named=True):
    key = tuple(row[col] for col in group_cols)
    grouped.setdefault(key, []).append(row)
  for rows in grouped.values():
    hidden = [row for row in rows if status[int(row["_row"])] in {"primary", "complement"}]
    shown = [row for row in rows if status[int(row["_row"])] == "ok"]
    hidden_n = sum(int(row["n"]) for row in hidden)
    while hidden and shown and hidden_n < 11:
      extra = min(shown, key=lambda row: int(row["n"]))
      shown.remove(extra)
      status[int(extra["_row"])] = "complement"
      hidden_n += int(extra["n"])
  labels = [status[int(row["_row"])] for row in work.sort("_row").iter_rows(named=True)]
  return frame.with_columns(pl.Series("suppress_status", labels))


def public_suppression(
  frame: pl.DataFrame,
  group_cols: tuple[str, ...],
  metric_cols: tuple[str, ...] = METRIC_COLUMNS,
) -> pl.DataFrame:
  """Blank small cells and one complementary cell so a total cannot recover them."""
  status = complementary_suppress_status(frame, group_cols, "n", "n_events")
  out = _hide_until_remainder_is_safe(frame.with_columns(status), group_cols)
  hide_counts = pl.col("suppress_status").is_in(["primary", "complement"])
  hide_metrics = pl.col("suppress_status") == "primary"
  expressions = [
    pl.when(hide_counts).then(None).otherwise(pl.col("n")).alias("n"),
    pl.when(hide_counts).then(None).otherwise(pl.col("n_events")).alias("n_events"),
    pl.when(hide_counts).then(None).otherwise(pl.col("event_prevalence")).alias("event_prevalence"),
  ]
  for col in ("death_rate", "death_rate_ci_low", "death_rate_ci_high"):
    if col in out.columns:
      expressions.append(pl.when(hide_counts).then(None).otherwise(pl.col(col)).alias(col))
  for col in metric_cols:
    if col in out.columns:
      expressions.append(pl.when(hide_metrics).then(None).otherwise(pl.col(col)).alias(col))
  return out.with_columns(expressions)


def _both_classes(y: np.ndarray) -> bool:
  return len(y) >= 2 and int(np.unique(y).size) >= 2


def performance_row(
  y: np.ndarray,
  scores_a: np.ndarray,
  scores_b: np.ndarray,
  n_bootstraps: int,
  seed: int,
) -> dict[str, Any]:
  """Paired A/B metrics. Empty when the slice has only one outcome class."""
  n = int(len(y))
  events = int(y.sum()) if n else 0
  row: dict[str, Any] = {
    "n": n,
    "n_events": events,
    "event_prevalence": float(y.mean()) if n else None,
    "models_refit": False,
    "n_bootstraps": 0,
  }
  for col in METRIC_COLUMNS:
    row[col] = None
  if not _both_classes(y):
    return row
  metrics_a = evaluate_predictions(y, scores_a)
  metrics_b = evaluate_predictions(y, scores_b)
  delta_auroc = bootstrap_paired_difference(y, scores_a, scores_b, "auroc", n_bootstraps, seed)
  delta_auprc = bootstrap_paired_difference(y, scores_a, scores_b, "auprc", n_bootstraps, seed + 1)
  gain_auroc = representation_gain(y, scores_a, scores_b, metric="auroc")
  gain_auprc = representation_gain(y, scores_a, scores_b, metric="auprc")
  row.update({
    "auroc_a": gain_auroc.score_a,
    "auroc_b": gain_auroc.score_b,
    "auprc_a": gain_auprc.score_a,
    "auprc_b": gain_auprc.score_b,
    "delta_auroc_b_minus_a": delta_auroc.point_estimate,
    "delta_auroc_ci_low": delta_auroc.ci_lower,
    "delta_auroc_ci_high": delta_auroc.ci_upper,
    "delta_auprc_b_minus_a": delta_auprc.point_estimate,
    "delta_auprc_ci_low": delta_auprc.ci_lower,
    "delta_auprc_ci_high": delta_auprc.ci_upper,
    "brier_a": metrics_a.brier,
    "brier_b": metrics_b.brier,
    "n_bootstraps": n_bootstraps,
  })
  return row


def phenotype_performance_table(
  frame: pl.DataFrame,
  score_a: str,
  score_b: str,
  score_source: str,
  n_bootstraps: int,
  seed: int,
) -> pl.DataFrame:
  """A and B on identical rows inside each phenotype summary."""
  rows: list[dict[str, Any]] = []
  for phenotype in SUMMARY_LABELS:
    sub = frame.filter(pl.col("phenotype_summary") == phenotype)
    y = np.asarray(sub["mortality_30d"].to_numpy(), dtype=np.int64)
    row = performance_row(
      y,
      np.asarray(sub[score_a].to_numpy(), dtype=np.float64),
      np.asarray(sub[score_b].to_numpy(), dtype=np.float64),
      n_bootstraps,
      seed,
    )
    row["score_source"] = score_source
    row["phenotype_summary"] = phenotype
    rows.append(row)
  return pl.DataFrame(rows)


def risk_group_table(
  frame: pl.DataFrame,
  group_labels: np.ndarray,
  score_source: str,
  phenotype_summary: str,
) -> pl.DataFrame:
  """Event rates for the four frozen concordance groups."""
  labeled = frame.with_columns(pl.Series("risk_group", group_labels))
  rows: list[dict[str, Any]] = []
  for group in RISK_GROUPS:
    sub = labeled.filter(pl.col("risk_group") == group)
    n = sub.height
    events = int(sub["mortality_30d"].sum()) if n else 0
    low, high = wilson_interval(events, n)
    rows.append({
      "score_source": score_source,
      "phenotype_summary": phenotype_summary,
      "risk_group": group,
      "n": n,
      "n_events": events,
      "event_prevalence": (events / n) if n else None,
      "death_rate": (events / n) if n else None,
      "death_rate_ci_low": low,
      "death_rate_ci_high": high,
    })
  return pl.DataFrame(rows)


def phenotype_enrichment_table(
  frame: pl.DataFrame,
  group_labels: np.ndarray,
  score_source: str,
) -> pl.DataFrame:
  """Phenotype mix inside each frozen concordance group."""
  labeled = frame.with_columns(pl.Series("risk_group", group_labels))
  rows: list[dict[str, Any]] = []
  for group in RISK_GROUPS:
    for phenotype in SUMMARY_LABELS:
      sub = labeled.filter(
        (pl.col("risk_group") == group) & (pl.col("phenotype_summary") == phenotype)
      )
      n = sub.height
      events = int(sub["mortality_30d"].sum()) if n else 0
      rows.append({
        "score_source": score_source,
        "risk_group": group,
        "phenotype_summary": phenotype,
        "n": n,
        "n_events": events,
        "event_prevalence": (events / n) if n else None,
      })
  return pl.DataFrame(rows)


def blank_reconstructable_subset(
  detail: pl.DataFrame,
  reference: pl.DataFrame,
  key_cols: tuple[str, ...],
) -> pl.DataFrame:
  """Blank a visible subset when the published total would reveal a complement under 11.

  ``detail`` rows for one key must partition the ``reference`` row. The visible
  subset is every detail row that still has a count; the complement is the
  reference minus that visible piece.
  """
  ref = {
    tuple(row[col] for col in key_cols): row
    for row in reference.iter_rows(named=True)
  }
  grouped: dict[tuple[Any, ...], list[int]] = {}
  rows = list(detail.iter_rows(named=True))
  for index, row in enumerate(rows):
    grouped.setdefault(tuple(row[col] for col in key_cols), []).append(index)
  blank_events: set[int] = set()
  blank_n: set[int] = set()
  for key, indexes in grouped.items():
    source = ref.get(key)
    if source is None:
      continue
    visible_events = [
      index for index in indexes
      if rows[index].get("n_events") is not None
    ]
    visible_n = [index for index in indexes if rows[index].get("n") is not None]
    if visible_events:
      hidden_events = int(source["n_events"]) - sum(int(rows[index]["n_events"]) for index in visible_events)
      if 0 < hidden_events < 11:
        blank_events.update(visible_events)
    if visible_n:
      hidden_n = int(source["n"]) - sum(int(rows[index]["n"]) for index in visible_n)
      if 0 < hidden_n < 11:
        blank_n.update(visible_n)
  event_fields = [
    col for col in ("n_events", "event_prevalence", "death_rate", "death_rate_ci_low", "death_rate_ci_high")
    if col in detail.columns
  ]
  updated: list[dict[str, Any]] = []
  for index, row in enumerate(rows):
    row = dict(row)
    if index in blank_n and "n" in row:
      row["n"] = None
    if index in blank_events:
      for col in event_fields:
        row[col] = None
    updated.append(row)
  return pl.DataFrame(updated)


def _markdown_table(frame: pl.DataFrame, columns: list[str]) -> str:
  header = "| " + " | ".join(columns) + " |"
  rule = "| " + " | ".join("---" for _ in columns) + " |"
  body: list[str] = []
  for row in frame.select(columns).iter_rows(named=True):
    cells: list[str] = []
    for col in columns:
      value = row[col]
      if value is None:
        cells.append("suppressed")
      elif isinstance(value, float):
        cells.append(f"{value:.3f}")
      else:
        cells.append(str(value))
    body.append("| " + " | ".join(cells) + " |")
  return "\n".join([header, rule, *body])


def render_sensitivity_markdown(
  performance: pl.DataFrame,
  risk_groups: pl.DataFrame,
  enrichment: pl.DataFrame,
  audit: dict[str, Any],
) -> str:
  """Aggregate interpretation. It does not treat phenotype restriction as mediation."""
  context = performance.filter(pl.col("score_source") == "context_adjusted_frozen")
  above: list[str] = []
  includes_zero: list[str] = []
  for row in context.iter_rows(named=True):
    low = row["delta_auroc_ci_low"]
    high = row["delta_auroc_ci_high"]
    if low is None or high is None:
      continue
    label = str(row["phenotype_summary"])
    if low > 0:
      above.append(label)
    elif low <= 0 <= high:
      includes_zero.append(label)
  interval = (
    "Context-adjusted B−A AUROC intervals above zero: "
    + (", ".join(above) if above else "none")
    + ". Intervals that include zero: "
    + (", ".join(includes_zero) if includes_zero else "none")
    + "."
  )
  perf_cols = [
    "score_source", "phenotype_summary", "n", "n_events", "event_prevalence",
    "auroc_a", "auroc_b", "delta_auroc_b_minus_a", "delta_auroc_ci_low", "delta_auroc_ci_high",
    "auprc_a", "auprc_b", "delta_auprc_b_minus_a", "delta_auprc_ci_low", "delta_auprc_ci_high",
  ]
  risk_cols = [
    "score_source", "phenotype_summary", "risk_group", "n", "n_events",
    "death_rate", "death_rate_ci_low", "death_rate_ci_high",
  ]
  enrich_cols = [
    "score_source", "risk_group", "phenotype_summary", "n", "n_events", "event_prevalence",
  ]
  nonmajor = risk_groups.filter(pl.col("phenotype_summary") == SUMMARY_NEGATIVE)
  overall = risk_groups.filter(pl.col("phenotype_summary") == "all_phenotypes")
  return "\n".join([
    "# Mortality sensitivity on frozen scores",
    "",
    "Analysis status: `posthoc_frozen_extension`.",
    f"Lexicon: `{audit['lexicon_version']}`, hash `{audit['lexicon_hash']}`.",
    f"Source SHA: `{audit['source_sha']}`.",
    f"Decision record: `{audit['decision_ids']}`.",
    "",
    "The original mortality probes were not refit. High-risk means at or above the "
    "80th percentile of the full paired-valid locked test for that "
    "score source. The collaborator checkpoint of 14.53%, 8.41%, and 4.22% matches the "
    "ECG-only probabilities; context-adjusted groups are a different partition. "
    "Those cutoffs are reused inside every phenotype slice. "
    f"`{SUMMARY_NEGATIVE}` means the frozen overt phrases were absent. "
    f"`{SUMMARY_UNKNOWN}` includes hedged emergency language and missing reports. "
    "A smaller B−A difference inside a phenotype slice describes that slice. "
    "It is not evidence that overt machine text explains, mediates, or accounts for "
    "the original representation comparison. D-BETA was pretrained on MIMIC-IV-ECG text "
    "and waveforms, so overlap with machine phrases is expected.",
    "",
    f"Locked-test rows: {audit['n_locked_test']:,}. "
    f"Context-adjusted cuts: A {audit['context_a_cut']:.4f}, B {audit['context_b_cut']:.4f}.",
    "",
    interval,
    "",
    "## Phenotype-stratified discrimination",
    "",
    _markdown_table(performance, perf_cols),
    "",
    "## Full locked-test concordance groups",
    "",
    _markdown_table(overall, risk_cols),
    "",
    "## Phenotype mix inside concordance groups",
    "",
    "A suppressed cell has fewer than 11 patients or fewer than 11 deaths, or it is "
    "the complementary cell that would let a published total reveal that count. "
    "The denominator is suppressed together with the event count.",
    "",
    _markdown_table(enrichment, enrich_cols),
    "",
    "## Concordance groups inside no_overt_major",
    "",
    _markdown_table(nonmajor, risk_cols),
    "",
    "This is an in-domain MIMIC description. It is not clinician adjudication of the "
    "tracings and it is not external validation of D-BETA.",
    "",
  ])


def run_mortality_sensitivity(
  predictions_context_path: Path,
  predictions_ecg_path: Path,
  measurements_path: Path,
  dictionary_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, Any]:
  """Join frozen locked-test scores to the frozen lexicon and write aggregate tables."""
  lexicon_hash = assert_frozen_lexicon(dictionary_path)
  dictionary = load_phenotype_dictionary(dictionary_path)
  context = pl.read_parquet(predictions_context_path)
  ecg = pl.read_parquet(predictions_ecg_path)
  study_ids = context.select("study_id").unique()
  if study_ids.height != context.height:
    raise ValueError("Context predictions are not unique by study_id.")
  reports = (
    pl.scan_csv(measurements_path, infer_schema_length=500)
    .select(["study_id", *[f"report_{i}" for i in range(18)]])
    .join(study_ids.lazy(), on="study_id", how="inner")
    .collect()
  )
  phenotypes = phenotype_frame(reports, dictionary).select(
    ["study_id", "phenotype_summary", "phenotype_reason"]
  )
  base = context.join(phenotypes, on="study_id", how="left").with_columns(
    pl.when(pl.col("phenotype_summary").is_in(list(SUMMARY_LABELS)))
    .then(pl.col("phenotype_summary"))
    .otherwise(pl.lit(SUMMARY_UNKNOWN))
    .alias("phenotype_summary")
  )
  if base.height != context.height:
    raise ValueError("Phenotype join changed the locked-test row count.")
  ecg_scores = ecg.select(["study_id", "p_a_ecg", "p_b_ecg"])
  base = base.join(ecg_scores, on="study_id", how="inner")
  if base.height != context.height:
    raise ValueError("ECG-only predictions do not cover the context-adjusted locked test.")

  context_a = np.asarray(base["p_a"].to_numpy(), dtype=np.float64)
  context_b = np.asarray(base["p_b"].to_numpy(), dtype=np.float64)
  ecg_a = np.asarray(base["p_a_ecg"].to_numpy(), dtype=np.float64)
  ecg_b = np.asarray(base["p_b_ecg"].to_numpy(), dtype=np.float64)
  context_cuts = frozen_risk_cuts(context_a, context_b)
  ecg_cuts = frozen_risk_cuts(ecg_a, ecg_b)
  performance = pl.concat([
    phenotype_performance_table(base, "p_a", "p_b", "context_adjusted_frozen", n_bootstraps, seed),
    phenotype_performance_table(base, "p_a_ecg", "p_b_ecg", "ecg_only_frozen", n_bootstraps, seed + 17),
  ])
  risk_frames = [
    risk_group_table(base, apply_frozen_risk_groups(context_a, context_b, *context_cuts), "context_adjusted_frozen", "all_phenotypes"),
    risk_group_table(base, apply_frozen_risk_groups(ecg_a, ecg_b, *ecg_cuts), "ecg_only_frozen", "all_phenotypes"),
  ]
  for phenotype in SUMMARY_LABELS:
    mask = np.asarray(base["phenotype_summary"].to_numpy() == phenotype)
    sliced = base.filter(pl.col("phenotype_summary") == phenotype)
    risk_frames.append(risk_group_table(
      sliced,
      apply_frozen_risk_groups(context_a[mask], context_b[mask], *context_cuts),
      "context_adjusted_frozen",
      phenotype,
    ))
    risk_frames.append(risk_group_table(
      sliced,
      apply_frozen_risk_groups(ecg_a[mask], ecg_b[mask], *ecg_cuts),
      "ecg_only_frozen",
      phenotype,
    ))
  risk_groups = pl.concat(risk_frames)
  enrichment = pl.concat([
    phenotype_enrichment_table(
      base,
      apply_frozen_risk_groups(ecg_a, ecg_b, *ecg_cuts),
      "ecg_only_frozen",
    ),
    phenotype_enrichment_table(
      base,
      apply_frozen_risk_groups(context_a, context_b, *context_cuts),
      "context_adjusted_frozen",
    ),
  ])
  public_performance = public_suppression(performance, ("score_source",))
  public_risk = public_suppression(risk_groups, ("score_source", "phenotype_summary"))
  public_enrichment = public_suppression(enrichment, ("score_source", "risk_group"))
  reference_groups = risk_groups.filter(pl.col("phenotype_summary") == "all_phenotypes")
  public_enrichment = blank_reconstructable_subset(
    public_enrichment, reference_groups, ("score_source", "risk_group"),
  )
  nonmajor = blank_reconstructable_subset(
    public_risk.filter(pl.col("phenotype_summary") == SUMMARY_NEGATIVE),
    reference_groups,
    ("score_source", "risk_group"),
  )
  public_risk = pl.concat([
    public_risk.filter(pl.col("phenotype_summary") != SUMMARY_NEGATIVE),
    nonmajor.select(public_risk.columns),
  ])
  audit = {
    "analysis_status": "posthoc_frozen_extension",
    "decision_ids": "D1, D2, D3, D8",
    "lexicon_version": FROZEN_LEXICON_VERSION,
    "lexicon_hash": lexicon_hash,
    "source_sha": source_sha,
    "n_locked_test": context.height,
    "high_quantile": HIGH_QUANTILE,
    "context_a_cut": context_cuts[0],
    "context_b_cut": context_cuts[1],
    "ecg_a_cut": ecg_cuts[0],
    "ecg_b_cut": ecg_cuts[1],
    "n_bootstraps": n_bootstraps,
    "seed": seed,
    "models_refit": False,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  base.select([
    "study_id", "subject_id", "mortality_30d", "phenotype_summary", "phenotype_reason",
  ]).write_parquet(output_dir / "locked_test_phenotype.parquet")
  (output_dir / "mortality_sensitivity_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  public_performance.write_csv(output_dir / "phenotype_stratified_performance.csv")
  public_risk.write_csv(output_dir / "discordance_by_phenotype.csv")
  public_enrichment.write_csv(output_dir / "phenotype_by_risk_group.csv")
  report = render_sensitivity_markdown(public_performance, public_risk, public_enrichment, audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  tables = report_path.parent / "tables"
  tables.mkdir(parents=True, exist_ok=True)
  public_performance.drop("suppress_status").write_csv(tables / "phenotype_stratified_performance.csv")
  public_risk.filter(pl.col("phenotype_summary") == SUMMARY_NEGATIVE).drop("suppress_status").write_csv(
    tables / "discordance_nonmajor_ecg.csv"
  )
  public_enrichment.drop("suppress_status").write_csv(tables / "phenotype_by_risk_group.csv")
  return audit
