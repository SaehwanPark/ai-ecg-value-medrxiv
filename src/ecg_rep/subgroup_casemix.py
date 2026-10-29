"""Descriptive subgroup metrics and setting-mix reweighting.

Decision D6. Reweighting uses the observed locked-test setting mix. It is not
a causal decomposition and it is not a claim about fairness. A standardized
value is withheld when the retained setting weight is below 0.8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_rep.analysis import assign_age_subgroup, representation_gain
from ecg_rep.mortality_sensitivity import public_suppression
from ecg_rep.probe import bootstrap_paired_difference

NON_AMBIGUOUS_SETTINGS = ("ED", "ICU", "inpatient_non_ICU", "other_or_unlinked")
MIN_RETAINED_WEIGHT = 0.8
MIN_CELL_N = 50
MIN_CELL_EVENTS = 11


def _as_float(value: object) -> float | None:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  return float(value)


def target_setting_weights(settings: np.ndarray) -> dict[str, float]:
  """Share of each non-ambiguous setting in the comparison population."""
  mask = np.isin(settings, NON_AMBIGUOUS_SETTINGS)
  chosen = settings[mask]
  if len(chosen) == 0:
    return {}
  values, counts = np.unique(chosen, return_counts=True)
  total = float(counts.sum())
  return {str(value): float(count) / total for value, count in zip(values, counts, strict=True)}


def direct_standardized_delta(
  y: np.ndarray,
  scores_a: np.ndarray,
  scores_b: np.ndarray,
  settings: np.ndarray,
  weights: dict[str, float],
) -> dict[str, Any]:
  """Reweight setting-specific B−A AUROC to ``weights``.

  Settings without two outcome classes, or below the cell minimum, are left
  out. The result is withheld when the kept weight is under 0.8.
  """
  used: dict[str, float] = {}
  for setting, weight in weights.items():
    if weight <= 0:
      continue
    mask = settings == setting
    y_s = y[mask]
    if int(mask.sum()) < MIN_CELL_N or int(y_s.sum()) < MIN_CELL_EVENTS or len(np.unique(y_s)) < 2:
      continue
    used[setting] = representation_gain(y_s, scores_a[mask], scores_b[mask], metric="auroc").gain
  retained = float(sum(weights[setting] for setting in used))
  if not used or retained < MIN_RETAINED_WEIGHT:
    return {"delta": None, "retained_weight": retained, "settings_used": sorted(used)}
  delta = float(sum(weights[setting] / retained * used[setting] for setting in used))
  return {"delta": delta, "retained_weight": retained, "settings_used": sorted(used)}


def metric_rows(
  frame: pl.DataFrame,
  group_col: str,
  score_a: str,
  score_b: str,
  n_bootstraps: int,
  seed: int,
) -> pl.DataFrame:
  """A, B, and B−A on identical rows within each subgroup level."""
  rows: list[dict[str, Any]] = []
  for level in sorted(set(frame[group_col].to_list())):
    sub = frame.filter(pl.col(group_col) == level)
    y = np.asarray(sub["mortality_30d"].to_numpy(), dtype=np.int64)
    a = np.asarray(sub[score_a].to_numpy(), dtype=np.float64)
    b = np.asarray(sub[score_b].to_numpy(), dtype=np.float64)
    both = len(y) >= 2 and len(np.unique(y)) >= 2
    row: dict[str, Any] = {
      "subgroup_variable": group_col,
      "subgroup_level": str(level),
      "n": int(len(y)),
      "n_events": int(y.sum()) if len(y) else 0,
      "event_prevalence": float(y.mean()) if len(y) else None,
      "auroc_a": None,
      "auroc_b": None,
      "delta_auroc_b_minus_a": None,
      "delta_auroc_ci_low": None,
      "delta_auroc_ci_high": None,
      "auprc_a": None,
      "auprc_b": None,
      "delta_auprc_b_minus_a": None,
      "delta_auprc_ci_low": None,
      "delta_auprc_ci_high": None,
    }
    if both:
      gain_auroc = representation_gain(y, a, b, metric="auroc")
      gain_auprc = representation_gain(y, a, b, metric="auprc")
      boot_auroc = bootstrap_paired_difference(y, a, b, "auroc", n_bootstraps, seed)
      boot_auprc = bootstrap_paired_difference(y, a, b, "auprc", n_bootstraps, seed + 1)
      row.update({
        "auroc_a": gain_auroc.score_a,
        "auroc_b": gain_auroc.score_b,
        "delta_auroc_b_minus_a": boot_auroc.point_estimate,
        "delta_auroc_ci_low": boot_auroc.ci_lower,
        "delta_auroc_ci_high": boot_auroc.ci_upper,
        "auprc_a": gain_auprc.score_a,
        "auprc_b": gain_auprc.score_b,
        "delta_auprc_b_minus_a": boot_auprc.point_estimate,
        "delta_auprc_ci_low": boot_auprc.ci_lower,
        "delta_auprc_ci_high": boot_auprc.ci_upper,
      })
    rows.append(row)
  return pl.DataFrame(rows)


def case_mix_rows(frame: pl.DataFrame, group_col: str) -> pl.DataFrame:
  """Setting composition inside a subgroup. This is description, not a cause."""
  rows: list[dict[str, Any]] = []
  for level in sorted(set(frame[group_col].to_list())):
    sub = frame.filter(pl.col(group_col) == level)
    n = sub.height
    row: dict[str, Any] = {
      "subgroup_variable": group_col,
      "subgroup_level": str(level),
      "n": n,
      "n_events": int(sub["mortality_30d"].sum()) if n else 0,
      "event_prevalence": _as_float(sub["mortality_30d"].mean()) if n else None,
    }
    for setting in (*NON_AMBIGUOUS_SETTINGS, "ambiguous"):
      row[f"pct_{setting}"] = _as_float((sub["setting"] == setting).mean()) if n else None
    rows.append(row)
  return pl.DataFrame(rows)


def standardized_rows(
  frame: pl.DataFrame,
  group_col: str,
  weights: dict[str, float],
  n_bootstraps: int,
  seed: int,
) -> pl.DataFrame:
  """Point estimate plus a patient bootstrap that does not refit models."""
  rows: list[dict[str, Any]] = []
  rng = np.random.default_rng(seed)
  y_all = np.asarray(frame["mortality_30d"].to_numpy(), dtype=np.int64)
  a_all = np.asarray(frame["p_a"].to_numpy(), dtype=np.float64)
  b_all = np.asarray(frame["p_b"].to_numpy(), dtype=np.float64)
  setting_all = np.asarray(frame["setting"].to_numpy())
  labels = np.asarray(frame[group_col].to_numpy())
  for level in sorted(set(labels.tolist())):
    mask = labels == level
    point = direct_standardized_delta(y_all[mask], a_all[mask], b_all[mask], setting_all[mask], weights)
    replicates: list[float] = []
    for _ in range(n_bootstraps):
      idx = rng.choice(np.where(mask)[0], size=int(mask.sum()), replace=True)
      draw = direct_standardized_delta(y_all[idx], a_all[idx], b_all[idx], setting_all[idx], weights)
      if draw["delta"] is not None:
        replicates.append(float(draw["delta"]))
    if point["delta"] is None or len(replicates) < int(0.8 * n_bootstraps):
      low = high = None
      status = "unsupported"
      delta = None
    else:
      low, high = np.quantile(np.asarray(replicates, dtype=np.float64), [0.025, 0.975])
      status = "descriptive_reweighting"
      delta = point["delta"]
    rows.append({
      "subgroup_variable": group_col,
      "subgroup_level": str(level),
      "standardized_delta_auroc": delta,
      "ci_low": None if low is None else float(low),
      "ci_high": None if high is None else float(high),
      "retained_weight": point["retained_weight"],
      "settings_used": ",".join(point["settings_used"]),
      "status": status,
      "n_bootstraps": n_bootstraps,
    })
  return pl.DataFrame(rows)


def write_forest(metrics: pl.DataFrame, output_path: Path) -> None:
  """Error bars for subgroup B−A AUROC. Suppressed intervals are omitted."""
  shown = metrics.filter(pl.col("delta_auroc_ci_low").is_not_null())
  if shown.height == 0:
    return
  labels = [f"{row['subgroup_variable']}: {row['subgroup_level']}" for row in shown.iter_rows(named=True)]
  estimate = np.asarray(shown["delta_auroc_b_minus_a"].to_numpy(), dtype=np.float64)
  low = np.asarray(shown["delta_auroc_ci_low"].to_numpy(), dtype=np.float64)
  high = np.asarray(shown["delta_auroc_ci_high"].to_numpy(), dtype=np.float64)
  y_pos = np.arange(len(labels))
  fig, ax = plt.subplots(figsize=(8, max(3.0, 0.45 * len(labels))))
  ax.errorbar(estimate, y_pos, xerr=[estimate - low, high - estimate], fmt="o", color="#2f5d62")
  ax.axvline(0.0, color="#666666", linewidth=1)
  ax.set_yticks(y_pos, labels)
  ax.set_xlabel("Context-adjusted B−A AUROC")
  ax.set_title("Locked-test subgroup differences")
  fig.tight_layout()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=120)
  plt.close(fig)


def render_subgroup_markdown(
  metrics: pl.DataFrame,
  case_mix: pl.DataFrame,
  standardized: pl.DataFrame,
  audit: dict[str, Any],
) -> str:
  """Narrative that keeps observed differences separate from mechanisms."""
  def table(frame: pl.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in frame.select(columns).iter_rows(named=True):
      cells = []
      for col in columns:
        value = row[col]
        if value is None:
          cells.append("withheld" if col.startswith("standardized") or col.startswith("ci_") else "suppressed")
        elif isinstance(value, float):
          cells.append(f"{value:.3f}")
        else:
          cells.append(str(value))
      body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])

  return "\n".join([
    "# Subgroup case mix",
    "",
    "Analysis status: `posthoc_agent_decision`. Decision D6.",
    f"Source SHA: `{audit['source_sha']}`.",
    "",
    "A and B are the frozen context-adjusted probabilities on the paired-valid locked test. "
    "Probes were not refit. Setting percentages describe where the index ECG was acquired. "
    "A direct standardization reweights setting-specific B−A AUROC to the locked-test mix "
    "of non-ambiguous settings. It is a sensitivity to that observed mix. It does not "
    "identify a cause, and it does not support a claim that a representation is equitable "
    "or inequitable. Cells under 11 are suppressed.",
    "",
    f"Locked-test rows: {audit['n_locked_test']:,}.",
    "",
    "## Discrimination",
    "",
    table(metrics, [
      "subgroup_variable", "subgroup_level", "n", "n_events", "event_prevalence",
      "auroc_a", "auroc_b", "delta_auroc_b_minus_a", "delta_auroc_ci_low", "delta_auroc_ci_high",
      "auprc_a", "auprc_b", "delta_auprc_b_minus_a",
    ]),
    "",
    "## Setting mix",
    "",
    table(case_mix, ["subgroup_variable", "subgroup_level", "n", "pct_ED", "pct_ICU", "pct_inpatient_non_ICU", "pct_other_or_unlinked", "pct_ambiguous"]),
    "",
    "## Setting-reweighted B−A",
    "",
    table(standardized, [
      "subgroup_variable", "subgroup_level", "standardized_delta_auroc", "ci_low", "ci_high",
      "retained_weight", "status",
    ]),
    "",
    "Read A and B separately. A larger B−A in one sex or age band can come from a lower A, "
    "a higher B, or both. Age-band reweighting is withheld when the bootstrap does not "
    "keep at least 80% of the target setting weight. Race and ethnicity rows are descriptive. "
    "A wide interval is not evidence that a representation is equitable or inequitable.",
    "",
    "D-BETA pretraining on MIMIC-IV-ECG is separate from these probe subgroups. "
    "A balanced internal split does not show that the representation was trained on a representative population.",
    "",
  ])


def run_subgroup_casemix(
  predictions_path: Path,
  cohort_path: Path,
  setting_path: Path,
  race_map: pl.DataFrame | None,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = 200,
  seed: int = 42,
) -> dict[str, Any]:
  """Build aggregate subgroup tables from frozen scores."""
  predictions = pl.read_parquet(predictions_path)
  cohort = pl.read_parquet(cohort_path).select(["study_id", "gender", "age_at_ecg"])
  setting = pl.read_parquet(setting_path).select(["study_id", "setting"])
  frame = predictions.join(cohort, on="study_id", how="inner").join(setting, on="study_id", how="inner")
  if frame.height != predictions.height:
    raise ValueError("Subgroup join changed the locked-test row count.")
  ages = assign_age_subgroup(frame["age_at_ecg"].to_numpy())
  frame = frame.with_columns(
    pl.Series("age_group", ages),
    pl.when(pl.col("gender").str.to_uppercase() == "F").then(pl.lit("female"))
    .when(pl.col("gender").str.to_uppercase() == "M").then(pl.lit("male"))
    .otherwise(pl.lit("unknown_sex"))
    .alias("sex"),
  )
  if race_map is not None:
    frame = frame.join(race_map.select(["subject_id", "race_group"]), on="subject_id", how="left").with_columns(
      pl.col("race_group").fill_null("unknown_other")
    )
  else:
    frame = frame.with_columns(pl.lit("unknown_other").alias("race_group"))
  weights = target_setting_weights(np.asarray(frame["setting"].to_numpy()))
  metric_frames = [
    metric_rows(frame, column, "p_a", "p_b", n_bootstraps, seed)
    for column in ("sex", "age_group", "race_group")
  ]
  mix_frames = [case_mix_rows(frame, column) for column in ("sex", "age_group", "race_group")]
  std_frames = [
    standardized_rows(frame, column, weights, n_bootstraps, seed + 10)
    for column in ("sex", "age_group")
  ]
  metrics = public_suppression(pl.concat(metric_frames), ("subgroup_variable",))
  case_mix = public_suppression(pl.concat(mix_frames, how="diagonal"), ("subgroup_variable",))
  standardized = pl.concat(std_frames)
  audit = {
    "decision_ids": "D6",
    "source_sha": source_sha,
    "n_locked_test": predictions.height,
    "n_bootstraps": n_bootstraps,
    "seed": seed,
    "models_refit": False,
    "standardization": "direct setting reweighting to the locked-test non-ambiguous mix",
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "subgroup_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
  report = render_subgroup_markdown(metrics, case_mix, standardized, audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  tables = report_path.parent / "tables"
  tables.mkdir(parents=True, exist_ok=True)
  metrics.drop("suppress_status").write_csv(tables / "subgroup_metric_matrix.csv")
  case_mix.drop("suppress_status").write_csv(tables / "subgroup_case_mix.csv")
  standardized.write_csv(tables / "subgroup_standardized_gains.csv")
  write_forest(metrics, report_path.parent / "subgroup_forest.png")
  return audit
