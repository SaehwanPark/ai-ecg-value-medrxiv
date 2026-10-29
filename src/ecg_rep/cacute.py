"""Same-subset C0 versus Cacute mortality probes.

Decisions D4 and D5 define this block. C0 itself is unchanged. Troponin, lactate,
and ICU vitals are not features. A lab value is usable only when both the chart
time and the stored time are available at the index ECG. Missing labs stay
missingness indicators; their fill values come from the development split only.
The inventory command still refuses a refit. This module is the separate path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import json

import numpy as np
import polars as pl

from ecg_rep.acute_context import LAB_LOOKBACK_HOURS, classify_measurement_timing
from ecg_rep.analysis import (
  DEFAULT_CAPACITY_FRACTIONS,
  fixed_capacity_contrast_rows as _generic_capacity_contrast_rows,
  fixed_capacity_metric_rows,
  representation_gain,
)
from ecg_rep.evaluate import C0_FEATURE_COLS, _join_representation_matrix, fit_and_eval_model
from ecg_rep.probe import ProbeConfig, bootstrap_paired_difference

INCLUDED_SETTINGS = ("ED", "ICU", "inpatient_non_ICU", "other_or_unlinked")
SETTING_INDICATORS = ("setting_ED", "setting_ICU", "setting_inpatient_non_ICU")
LAB_SPECS: dict[str, dict[str, Any]] = {
  "creatinine_blood": {"itemids": [50912], "unit": "mg/dl", "low": 0.2, "high": 20.0},
  "potassium_blood": {"itemids": [50971], "unit": "meq/l", "low": 1.5, "high": 8.0},
  "hemoglobin_blood": {"itemids": [51222], "unit": "g/dl", "low": 3.0, "high": 20.0},
  "wbc_blood": {"itemids": [51301], "unit": "k/ul", "low": 0.2, "high": 100.0},
}


def setting_indicators(setting: pl.Expr) -> list[pl.Expr]:
  """Three indicators. ``other_or_unlinked`` is the reference category."""
  return [
    (setting == "ED").cast(pl.Float64).alias("setting_ED"),
    (setting == "ICU").cast(pl.Float64).alias("setting_ICU"),
    (setting == "inpatient_non_ICU").cast(pl.Float64).alias("setting_inpatient_non_ICU"),
  ]


def latest_plausible_labs(events: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
  """One leak-safe value per study and lab, or a null when none qualifies.

  The plausibility bounds and unit strings are measurement screens from D4.
  They were not chosen from mortality.
  """
  required = {"subject_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"}
  if required - set(events.columns):
    raise ValueError(f"Lab events are missing {sorted(required - set(events.columns))}.")
  pieces: list[pl.DataFrame] = []
  for concept, spec in LAB_SPECS.items():
    subset = events.filter(pl.col("itemid").is_in(spec["itemids"]))
    if subset.height == 0:
      pieces.append(index.select("study_id").with_columns(
        pl.lit(None, dtype=pl.Float64).alias(concept)
      ))
      continue
    labeled = classify_measurement_timing(subset, index, LAB_LOOKBACK_HOURS)
    unit = pl.col("valueuom").fill_null("").str.to_lowercase()
    qualified = labeled.filter(
      (pl.col("timing_class") == "qualified")
      & (unit == spec["unit"])
      & pl.col("valuenum").is_not_null()
      & (pl.col("valuenum") >= spec["low"])
      & (pl.col("valuenum") <= spec["high"])
    ).sort(["study_id", "charttime", "storetime"])
    latest = qualified.group_by("study_id", maintain_order=True).agg(
      pl.col("valuenum").last().alias(concept)
    )
    pieces.append(index.select("study_id").join(latest, on="study_id", how="left"))
  wide = pieces[0]
  for piece in pieces[1:]:
    wide = wide.join(piece, on="study_id", how="left")
  return wide


def impute_development_medians(
  frame: pl.DataFrame,
  value_cols: tuple[str, ...],
) -> tuple[pl.DataFrame, dict[str, float]]:
  """Add missingness flags, then fill with development medians of observed values."""
  dev = frame.filter(pl.col("split") == "dev")
  medians: dict[str, float] = {}
  filled = frame
  for col in value_cols:
    observed = dev.filter(pl.col(col).is_not_null())[col]
    if observed.len() == 0:
      raise ValueError(f"No development observations for {col}; refusing to invent a fill.")
    median_value = observed.median()
    if not isinstance(median_value, (int, float)):
      raise ValueError(f"Development median for {col} is not numeric.")
    medians[col] = float(median_value)
    filled = filled.with_columns(
      pl.col(col).is_null().cast(pl.Float64).alias(f"miss_{col}"),
      pl.col(col).fill_null(medians[col]).alias(col),
    )
  return filled, medians


def select_fixed_subset(frame: pl.DataFrame) -> pl.DataFrame:
  """Drop ambiguous settings so every compared model uses the same patients."""
  if "setting" not in frame.columns:
    raise ValueError("Cacute subset requires a setting column.")
  selected = frame.filter(pl.col("setting").is_in(list(INCLUDED_SETTINGS)))
  if selected.height == 0:
    raise ValueError("The fixed Cacute subset is empty.")
  return selected


def cacute_feature_columns(value_cols: tuple[str, ...] = tuple(LAB_SPECS)) -> list[str]:
  """C0, setting indicators, labs, and lab missingness flags. No troponin column."""
  columns = [
    *C0_FEATURE_COLS,
    *SETTING_INDICATORS,
    *value_cols,
    *[f"miss_{col}" for col in value_cols],
  ]
  if any("troponin" in col or "lactate" in col for col in columns):
    raise ValueError("Cacute must not include troponin or lactate.")
  return columns


def attenuation_row(
  y: np.ndarray,
  c0_a: np.ndarray,
  c0_b: np.ndarray,
  cacute_a: np.ndarray,
  cacute_b: np.ndarray,
  metric: Literal["auroc", "auprc"],
  n_bootstraps: int,
  seed: int,
) -> dict[str, Any]:
  """Paired B−A under C0 and under Cacute, and their difference, on one population."""
  gain_c0 = representation_gain(y, c0_a, c0_b, metric=metric)
  gain_cacute = representation_gain(y, cacute_a, cacute_b, metric=metric)
  boot_c0 = bootstrap_paired_difference(y, c0_a, c0_b, metric, n_bootstraps, seed)
  boot_cacute = bootstrap_paired_difference(y, cacute_a, cacute_b, metric, n_bootstraps, seed + 1)
  # Resample once for the attenuation contrast so the interval matches one patient bootstrap.
  point = gain_c0.gain - gain_cacute.gain
  rng = np.random.default_rng(seed + 2)
  diffs: list[float] = []
  pos = np.where(y == 1)[0]
  neg = np.where(y == 0)[0]
  for _ in range(n_bootstraps):
    idx = np.concatenate([
      rng.choice(pos, size=len(pos), replace=True),
      rng.choice(neg, size=len(neg), replace=True),
    ])
    d0 = representation_gain(y[idx], c0_a[idx], c0_b[idx], metric=metric).gain
    d1 = representation_gain(y[idx], cacute_a[idx], cacute_b[idx], metric=metric).gain
    diffs.append(d0 - d1)
  lo, hi = np.quantile(np.asarray(diffs, dtype=np.float64), [0.025, 0.975])
  return {
    "metric": metric,
    "n": int(len(y)),
    "n_events": int(y.sum()),
    "delta_c0": gain_c0.gain,
    "delta_c0_ci_low": boot_c0.ci_lower,
    "delta_c0_ci_high": boot_c0.ci_upper,
    "score_c0_a": gain_c0.score_a,
    "score_c0_b": gain_c0.score_b,
    "delta_cacute": gain_cacute.gain,
    "delta_cacute_ci_low": boot_cacute.ci_lower,
    "delta_cacute_ci_high": boot_cacute.ci_upper,
    "score_cacute_a": gain_cacute.score_a,
    "score_cacute_b": gain_cacute.score_b,
    "attenuation_c0_minus_cacute": point,
    "attenuation_ci_low": float(lo),
    "attenuation_ci_high": float(hi),
    "n_bootstraps": n_bootstraps,
    "models_refit_inside_bootstrap": False,
  }


#: Context-only model and the optional combined acute-context model (plan A1).
CONTEXT_ONLY_MODEL = "cacute"
COMBINED_MODEL = "cacute_ab"

#: Paired incremental contrasts on the fixed Cacute subset (plan A1).
INCREMENTAL_PAIRS: tuple[tuple[str, str, str], ...] = (
  ("cacute", "cacute_a", "add_A_to_cacute"),
  ("cacute", "cacute_b", "add_B_to_cacute"),
  ("cacute_a", "cacute_b", "B_minus_A_under_cacute"),
  ("cacute_a", "cacute_ab", "add_B_to_cacute_a"),
  ("cacute_b", "cacute_ab", "add_A_to_cacute_b"),
)

#: Models evaluated at fixed review capacity (plan A2).
CAPACITY_MODELS: tuple[str, ...] = ("cacute", "cacute_a", "cacute_b", "cacute_ab")

#: Fixed-capacity paired contrasts: (baseline, comparison, label).
CAPACITY_CONTRAST_PAIRS: tuple[tuple[str, str, str], ...] = (
  ("cacute", "cacute_a", "add_A_to_cacute"),
  ("cacute", "cacute_b", "add_B_to_cacute"),
  ("cacute_a", "cacute_b", "B_minus_A_under_cacute"),
  ("cacute_a", "cacute_ab", "add_B_to_cacute_a"),
  ("cacute_b", "cacute_ab", "add_A_to_cacute_b"),
)


def incremental_contrast_rows(
  y: np.ndarray,
  scores: dict[str, np.ndarray],
  n_bootstraps: int,
  seed: int,
) -> list[dict[str, Any]]:
  """Paired incremental value of adding A or B on top of the acute-context block.

  All contrasts for one metric reuse one bootstrap resample seed so the rows
  are mutually comparable; probes are never refitted inside the bootstrap.
  """
  rows: list[dict[str, Any]] = []
  for metric_index, metric in enumerate(("auroc", "auprc")):
    metric_seed = seed + metric_index
    for base, added, name in INCREMENTAL_PAIRS:
      gain = representation_gain(y, scores[base], scores[added], metric=metric)
      diff = bootstrap_paired_difference(
        y, scores[base], scores[added], metric, n_bootstraps, metric_seed,
      )
      rows.append({
        "contrast": name,
        "metric": metric,
        "n": int(len(y)),
        "n_events": int(y.sum()),
        "score_base": gain.score_a,
        "score_added": gain.score_b,
        "increment": diff.point_estimate,
        "ci_low": diff.ci_lower,
        "ci_high": diff.ci_upper,
        "n_bootstraps": diff.n_bootstraps,
        "seed": diff.seed,
        "models_refit_inside_bootstrap": False,
      })
  return rows


def fixed_capacity_rows(
  y: np.ndarray,
  scores: dict[str, np.ndarray],
  capacity_fractions: tuple[float, ...] = DEFAULT_CAPACITY_FRACTIONS,
) -> list[dict[str, Any]]:
  """Plan-section-12 fixed-capacity evaluation for each acute-context model."""
  return fixed_capacity_metric_rows(y, scores, list(CAPACITY_MODELS), capacity_fractions)


def fixed_capacity_contrast_rows(
  y: np.ndarray,
  scores: dict[str, np.ndarray],
  capacity_fractions: tuple[float, ...] = DEFAULT_CAPACITY_FRACTIONS,
  n_bootstraps: int = 200,
  seed: int = 42,
) -> list[dict[str, Any]]:
  """Paired fixed-capacity differences for the prespecified Cacute contrasts."""
  return _generic_capacity_contrast_rows(
    y, scores, CAPACITY_CONTRAST_PAIRS, capacity_fractions, n_bootstraps, seed,
  )


def prepare_cacute_frame(
  cohort: pl.DataFrame,
  setting: pl.DataFrame,
  labs: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, float]]:
  """Join setting and leak-safe labs, restrict to D5, and impute on development."""
  value_cols = tuple(LAB_SPECS)
  base = cohort.join(setting.select(["study_id", "setting"]), on="study_id", how="inner")
  base = select_fixed_subset(base).with_columns(setting_indicators(pl.col("setting")))
  values = latest_plausible_labs(labs, base.select(["subject_id", "study_id", "ecg_datetime"]))
  base = base.join(values, on="study_id", how="left")
  return impute_development_medians(base, value_cols)


def _aligned_test_scores(
  frame: pl.DataFrame,
  c0_cols: list[str],
  cacute_cols: list[str],
  probes: dict[str, Any],
) -> dict[str, np.ndarray]:
  from ecg_rep.evaluate import _split_xy

  order = frame.filter(pl.col("split") == "test").select("study_id")
  scores: dict[str, np.ndarray] = {}
  y = None
  za_cols = [c for c in frame.columns if c.startswith("za_")]
  zb_cols = [c for c in frame.columns if c.startswith("zb_")]
  for name, cols in (
    ("c0_a", c0_cols + za_cols),
    ("c0_b", c0_cols + zb_cols),
    ("cacute", cacute_cols),
    ("cacute_a", cacute_cols + za_cols),
    ("cacute_b", cacute_cols + zb_cols),
    ("cacute_ab", cacute_cols + za_cols + zb_cols),
  ):
    block = _split_xy(frame, cols, "test")
    if not np.array_equal(block.study_ids, order["study_id"].to_numpy()):
      raise ValueError(f"{name} test rows are not the fixed subset.")
    scores[name] = probes[name].predict_proba(block.x)
    y = block.y
  if y is None:
    raise ValueError("The fixed test subset has no rows.")
  scores["y"] = y
  return scores


def run_cacute_comparison(
  cohort_path: Path,
  c0_path: Path,
  setting_path: Path,
  labevents_path: Path,
  meta_a_path: Path,
  features_a_path: Path,
  meta_b_path: Path,
  embeddings_b_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = 200,
  seed: int = 42,
) -> dict[str, Any]:
  """Refit C0 and Cacute probes on one paired-valid, non-ambiguous population."""
  cohort = pl.read_parquet(cohort_path)
  c0 = pl.read_parquet(c0_path)
  setting = pl.read_parquet(setting_path)
  c0_cols = [col for col in C0_FEATURE_COLS if col in c0.columns]
  if c0_cols != list(C0_FEATURE_COLS):
    raise ValueError("C0 columns do not match the original context block.")
  itemids = [item for spec in LAB_SPECS.values() for item in spec["itemids"]]
  labs = (
    pl.scan_csv(
      labevents_path,
      schema_overrides={
        "subject_id": pl.Int64,
        "itemid": pl.Int64,
        "charttime": pl.String,
        "storetime": pl.String,
        "valuenum": pl.Float64,
        "valueuom": pl.String,
      },
    )
    .filter(pl.col("itemid").is_in(itemids))
    .select(["subject_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"])
    .with_columns(
      pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
      pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    )
    .collect()
  )
  prepared, medians = prepare_cacute_frame(
    cohort.join(c0.select(["subject_id", "study_id", *c0_cols]), on=["subject_id", "study_id"]),
    setting,
    labs,
  )
  meta_a = pl.read_parquet(meta_a_path)
  meta_b = pl.read_parquet(meta_b_path)
  frame = _join_representation_matrix(prepared, meta_a, np.load(features_a_path), "za", "representation_a_valid")
  frame = _join_representation_matrix(frame, meta_b, np.load(embeddings_b_path), "zb", "representation_b_valid")
  frame = frame.filter(
    (pl.col("representation_a_valid") == True)  # noqa: E712
    & (pl.col("representation_b_valid") == True)  # noqa: E712
  )
  cacute_cols = [
    *c0_cols,
    *SETTING_INDICATORS,
    *LAB_SPECS.keys(),
    *[f"miss_{col}" for col in LAB_SPECS],
  ]
  za_cols = [col for col in frame.columns if col.startswith("za_")]
  zb_cols = [col for col in frame.columns if col.startswith("zb_")]
  config = ProbeConfig()
  probes: dict[str, Any] = {}
  model_rows: list[dict[str, Any]] = []
  specs = {
    "c0_a": c0_cols + za_cols,
    "c0_b": c0_cols + zb_cols,
    "cacute": list(cacute_cols),
    "cacute_a": cacute_cols + za_cols,
    "cacute_b": cacute_cols + zb_cols,
    "cacute_ab": cacute_cols + za_cols + zb_cols,
  }
  for name, cols in specs.items():
    result, probe = fit_and_eval_model(frame, cols, name, config=config)
    probes[name] = probe
    metrics = result["test_metrics"]
    model_rows.append({
      "model": name,
      "n_features": result["n_features"],
      "best_c": result["best_c"],
      "n": metrics["n"],
      "n_events": metrics["n_events"],
      "auroc": metrics["auroc"],
      "auprc": metrics["auprc"],
      "log_loss": metrics["log_loss"],
      "brier": metrics["brier"],
      "calibration_intercept": metrics["calibration_intercept"],
      "calibration_slope": metrics["calibration_slope"],
    })
  scores = _aligned_test_scores(frame, c0_cols, cacute_cols, probes)
  comparisons = [
    attenuation_row(scores["y"], scores["c0_a"], scores["c0_b"], scores["cacute_a"], scores["cacute_b"], metric, n_bootstraps, seed)
    for metric in ("auroc", "auprc")
  ]
  increments = incremental_contrast_rows(scores["y"], scores, n_bootstraps, seed)
  capacity_metrics = fixed_capacity_rows(scores["y"], scores)
  capacity_contrasts = fixed_capacity_contrast_rows(
    scores["y"], scores, n_bootstraps=n_bootstraps, seed=seed,
  )
  test_n = int((frame["split"] == "test").sum())
  audit = {
    "decision_ids": "D4, D5",
    "source_sha": source_sha,
    "n_fixed_rows": frame.height,
    "n_fixed_test": test_n,
    "development_medians": medians,
    "excluded_setting": "ambiguous",
    "excluded_features": ["troponin", "lactate", "icu_vitals"],
    "setting_reference": "other_or_unlinked",
    "lab_lookback_hours": LAB_LOOKBACK_HOURS,
    "capacity_fractions": list(DEFAULT_CAPACITY_FRACTIONS),
    "models_refit_inside_bootstrap": False,
    "original_cohort_unchanged": True,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  pl.DataFrame(model_rows).write_csv(output_dir / "fixed_subset_model_comparison.csv")
  pl.DataFrame(comparisons).write_csv(output_dir / "attenuation_comparison.csv")
  pl.DataFrame(increments).write_csv(output_dir / "cacute_incremental_contrasts.csv")
  pl.DataFrame(capacity_metrics).write_csv(output_dir / "fixed_capacity_metrics.csv")
  pl.DataFrame(capacity_contrasts).write_csv(output_dir / "fixed_capacity_contrasts.csv")
  (output_dir / "cacute_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
  report = render_cacute_markdown(
    pl.DataFrame(model_rows),
    pl.DataFrame(comparisons),
    audit,
    pl.DataFrame(increments),
    pl.DataFrame(capacity_metrics),
    pl.DataFrame(capacity_contrasts),
  )
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  tables = report_path.parent / "tables"
  tables.mkdir(parents=True, exist_ok=True)
  pl.DataFrame(model_rows).write_csv(tables / "fixed_subset_model_comparison.csv")
  pl.DataFrame(comparisons).write_csv(tables / "attenuation_comparison.csv")
  pl.DataFrame(increments).write_csv(tables / "cacute_incremental_contrasts.csv")
  pl.DataFrame(capacity_metrics).write_csv(tables / "fixed_capacity_metrics.csv")
  pl.DataFrame(capacity_contrasts).write_csv(tables / "fixed_capacity_contrasts.csv")
  return audit


def render_cacute_markdown(
  models: pl.DataFrame,
  comparisons: pl.DataFrame,
  audit: dict[str, Any],
  increments: pl.DataFrame,
  capacity_metrics: pl.DataFrame,
  capacity_contrasts: pl.DataFrame,
) -> str:
  """Aggregate note. Attenuation is the change in B−A on one fixed subset."""
  def table(frame: pl.DataFrame) -> str:
    columns = frame.columns
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in frame.iter_rows(named=True):
      cells = []
      for col in columns:
        value = row[col]
        cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
      body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])

  capacities = ", ".join(
    f"{100 * float(f):g}%" for f in audit["capacity_fractions"]
  )
  return "\n".join([
    "# Same-subset acute context",
    "",
    "Analysis status: `posthoc_agent_decision`. Decisions D4 and D5.",
    f"Source SHA: `{audit['source_sha']}`.",
    "",
    "C0 is the original age, sex, and 31 prior-admission Elixhauser indicators. "
    "Cacute adds three setting indicators, with other/unlinked as the reference, "
    "and the most recent leak-safe creatinine, potassium, hemoglobin, and WBC "
    "in the 24 hours before the ECG. A stored-after-ECG result is not a feature. "
    "Troponin, lactate, and ICU vitals are omitted. Missing labs use a development "
    "median plus a missingness indicator. Ambiguous settings are excluded, and "
    "every model below is fit on the patients who remain. This is not the original "
    "full-cohort C0 gain, and a smaller B−A after Cacute is not a mechanism.",
    "",
    f"Fixed paired-valid rows: {audit['n_fixed_rows']:,}. "
    f"Fixed locked-test rows: {audit['n_fixed_test']:,}.",
    "",
    "## Models",
    "",
    "`cacute` is the acute-context block alone; `cacute_ab` combines the acute "
    "context with both representations. All models use the same development "
    "standardization, validation log-loss tuning, and locked-test evaluation.",
    "",
    table(models),
    "",
    "## Paired B−A and attenuation",
    "",
    "Attenuation is delta under C0 minus delta under Cacute. Bootstrap intervals "
    "resample the locked test and do not refit the probes.",
    "",
    table(comparisons),
    "",
    "On this fixed subset, context-adjusted B−A AUROC is "
    f"{float(comparisons.filter(pl.col('metric') == 'auroc')['delta_cacute'][0]):.3f} "
    "with Cacute. This answers how different A and B remain after acute information. "
    "It does not answer whether either representation still adds information once "
    "that acute information is available; the next section does.",
    "",
    "## Incremental value over Cacute alone",
    "",
    "Each row is the paired change from adding one block on top of the base model, "
    "with locked-test bootstrap intervals (probes frozen, no refitting).",
    "",
    table(increments),
    "",
    "## Fixed-capacity review at prespecified fractions",
    "",
    f"Top-risk review at {capacities} capacity (plan section 12). Capacity "
    "fractions are fixed review budgets, not clinical probability thresholds. "
    "Captured deaths are event enrichment, not evidence that an intervention "
    "would prevent deaths.",
    "",
    table(capacity_metrics),
    "",
    "### Paired fixed-capacity differences",
    "",
    "Positive deltas favor the comparison model. Bootstrap intervals resample "
    "the locked test with the probes frozen.",
    "",
    table(capacity_contrasts),
    "",
    *boundary_note(models),
    "MIMIC evaluation remains in-domain because D-BETA was pretrained on MIMIC-IV-ECG.",
    "",
  ])


def boundary_note(models: pl.DataFrame) -> list[str]:
  """Flag grid-boundary regularization choices so test-informed grid edits stay visible."""
  from ecg_rep.probe import DEFAULT_C_GRID

  lower, upper = min(DEFAULT_C_GRID), max(DEFAULT_C_GRID)
  boundary_models = [
    str(row["model"])
    for row in models.iter_rows(named=True)
    if row["best_c"] in (lower, upper)
  ]
  if not boundary_models:
    return []
  return [
    "Boundary-selected penalty on the prespecified grid (not extended after the "
    f"locked test was seen): {', '.join(boundary_models)}.",
    "",
  ]
