"""Pre-ECG vital and lab availability inventory.

A measurement is eligible for a ``Cacute`` feature only when the specimen or
chart time is inside the lookback and the result time is at or before the
index ECG. A missing ``storetime`` is not treated as available. This module
counts that availability. Same-subset Cacute probes are fit by ``cacute.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import polars as pl

VITAL_LOOKBACK_HOURS = 6
LAB_LOOKBACK_HOURS = 24
ANALYSIS_STATUS = "inventory_only"
MODEL_REFIT_COMMAND = "clinical-cacute"


def classify_measurement_timing(
  events: pl.DataFrame,
  index: pl.DataFrame,
  lookback_hours: int,
) -> pl.DataFrame:
  """Label each event relative to the index ECG.

  ``qualified`` requires ``charttime`` inside ``(ecg - lookback, ecg]`` and
  ``storetime <= ecg``. Future chart times and results stored after the ECG
  are not qualified.
  """
  required_events = {"subject_id", "charttime", "storetime"}
  required_index = {"subject_id", "study_id", "ecg_datetime"}
  if required_events - set(events.columns) or required_index - set(index.columns):
    raise ValueError("Timing classification is missing subject, chart, store, or ECG columns.")
  joined = events.join(
    index.select(["subject_id", "study_id", "ecg_datetime"]),
    on="subject_id",
    how="inner",
  ).with_columns(
    pl.col("charttime").cast(pl.Datetime("us")),
    pl.col("storetime").cast(pl.Datetime("us")),
    pl.col("ecg_datetime").cast(pl.Datetime("us")),
  )
  window_start = pl.col("ecg_datetime") - pl.duration(hours=lookback_hours)
  return joined.with_columns(
    pl.when(pl.col("charttime").is_null())
    .then(pl.lit("charttime_missing"))
    .when(pl.col("charttime") > pl.col("ecg_datetime"))
    .then(pl.lit("charttime_after_ecg"))
    .when(pl.col("charttime") < window_start)
    .then(pl.lit("outside_lookback"))
    .when(pl.col("storetime").is_null())
    .then(pl.lit("storetime_missing"))
    .when(pl.col("storetime") > pl.col("ecg_datetime"))
    .then(pl.lit("storetime_after_ecg"))
    .otherwise(pl.lit("qualified"))
    .alias("timing_class")
  )


def render_availability_markdown(ledger: pl.DataFrame, audit: dict[str, Any]) -> str:
  """Aggregate availability note. It is not a model comparison."""
  n_dev = int(audit.get("n_development", 0))
  columns = [
    "family", "concept", "n_subjects_qualified", "fraction_of_development",
    "n_subjects_storetime_after_ecg", "n_subjects_any_row",
  ]
  header = "| " + " | ".join(columns) + " |"
  rule = "| " + " | ".join("---" for _ in columns) + " |"
  body: list[str] = []
  for row in ledger.select([col for col in columns if col in ledger.columns]).iter_rows(named=True):
    cells = []
    for col in columns:
      value = row[col]
      if isinstance(value, float):
        cells.append(f"{value:.3f}")
      else:
        cells.append(str(value))
    body.append("| " + " | ".join(cells) + " |")
  lines = [
    "# Pre-ECG acute context availability",
    "",
    f"Analysis status: `{ANALYSIS_STATUS}`. Model refit: `{MODEL_REFIT_COMMAND}` (D4/D5; G3 approved 2026-10-05).",
    f"Inventory SHA: `{audit.get('source_sha', 'unknown')}`.",
    f"Development index ECGs: {n_dev:,}.",
    "",
    "A lab or vital counts as available only when `charttime` is inside the lookback "
    f"({audit.get('lab_lookback_hours', LAB_LOOKBACK_HOURS)} hours for labs, "
    f"{audit.get('vital_lookback_hours', VITAL_LOOKBACK_HOURS)} hours for vitals) and "
    "`storetime` is at or before the index ECG. Missing `storetime` is excluded. "
    "Subject counts are not mutually exclusive: one patient can have both a qualified "
    "result and a later result whose store time is after the ECG. "
    "`n_subjects_any_row` includes measurements outside the lookback.",
    "",
    "MIMIC-IV-ED is not installed, so emergency-department triage vitals are absent. "
    "ICU `chartevents` cover only patients with an ICU chart and do not describe ward "
    "vital availability. Leak-safe ICU vitals within 6 hours are uncommon in this "
    "development split, which is consistent with most index ECGs not occurring in an ICU. "
    "Troponin is inventory only. Troponin I has no leak-safe 24-hour result in this "
    "development count; troponin T is uncommon. Neither assay is a negative AMI label "
    "when missing, and neither may enter an AMI predictor if it is later used to define AMI.",
    "",
    header,
    rule,
    *body,
    "",
    "This inventory reports availability only. Same-subset C0+A, C0+B, Cacute, "
    "Cacute+A, Cacute+B, and Cacute+A+B fits are in `cacute_results.md`.",
    "",
  ]
  return "\n".join(lines)


def _concept_expr(item_map: dict[str, list[int]]) -> pl.Expr:
  expr = pl.lit(None, dtype=pl.String)
  for concept, itemids in item_map.items():
    expr = pl.when(pl.col("itemid").is_in(itemids)).then(pl.lit(concept)).otherwise(expr)
  return expr


def _scan_events(path: Path, itemids: list[int]) -> pl.DataFrame:
  frame = pl.scan_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "itemid": pl.Int64,
      "charttime": pl.String,
      "storetime": pl.String,
    },
  ).filter(pl.col("itemid").is_in(itemids)).select(["subject_id", "itemid", "charttime", "storetime"])
  return frame.with_columns(
    pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
  ).collect()


def inventory_development_availability(
  cohort_path: Path,
  labevents_path: Path,
  chartevents_path: Path,
  lab_map: dict[str, list[int]],
  vital_map: dict[str, list[int]],
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  setting_path: Path | None = None,
) -> dict[str, Any]:
  """Count leak-safe pre-ECG measurements on the development split only."""
  cohort = pl.read_parquet(cohort_path).filter(pl.col("split") == "dev").select(
    ["subject_id", "study_id", "ecg_datetime"]
  )
  lab_ids = [item for ids in lab_map.values() for item in ids]
  vital_ids = [item for ids in vital_map.values() for item in ids]
  labs = _scan_events(labevents_path, lab_ids).with_columns(_concept_expr(lab_map).alias("concept"))
  vitals = _scan_events(chartevents_path, vital_ids).with_columns(_concept_expr(vital_map).alias("concept"))
  labs = labs.join(cohort.select("subject_id").unique(), on="subject_id", how="semi")
  vitals = vitals.join(cohort.select("subject_id").unique(), on="subject_id", how="semi")
  labeled_labs = classify_measurement_timing(labs, cohort, LAB_LOOKBACK_HOURS)
  labeled_vitals = classify_measurement_timing(vitals, cohort, VITAL_LOOKBACK_HOURS)
  ledger = pl.concat([
    availability_by_concept(labeled_labs).with_columns(pl.lit("lab_24h").alias("family")),
    availability_by_concept(labeled_vitals).with_columns(pl.lit("vital_6h").alias("family")),
  ], how="vertical")
  n_dev = cohort.height
  ledger = ledger.with_columns((pl.col("n_subjects_qualified") / n_dev).alias("fraction_of_development"))
  output_dir.mkdir(parents=True, exist_ok=True)
  ledger.write_csv(output_dir / "pre_ecg_availability_ledger.csv")
  if setting_path is not None and setting_path.exists():
    qualified = pl.concat([
      labeled_labs.filter(pl.col("timing_class") == "qualified").select(["study_id", "concept"]),
      labeled_vitals.filter(pl.col("timing_class") == "qualified").select(["study_id", "concept"]),
    ]).unique()
    setting = pl.read_parquet(setting_path).select(["study_id", "setting"])
    by_setting = (
      qualified.join(setting, on="study_id", how="left")
      .group_by(["setting", "concept"]).agg(pl.col("study_id").n_unique().alias("n_qualified"))
    )
    by_setting.write_csv(output_dir / "availability_by_setting.csv")
  audit = {
    "analysis_status": ANALYSIS_STATUS,
    "model_refit": MODEL_REFIT_COMMAND,
    "source_sha": source_sha,
    "n_development": n_dev,
    "lab_lookback_hours": LAB_LOOKBACK_HOURS,
    "vital_lookback_hours": VITAL_LOOKBACK_HOURS,
    "mimic_ed_module": False,
  }
  (output_dir / "acute_context_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
  (report_path.parent / "tables").mkdir(parents=True, exist_ok=True)
  report = render_availability_markdown(ledger, audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  ledger.write_csv(report_path.parent / "tables" / "pre_ecg_availability_ledger.csv")
  return audit


def availability_by_concept(
  classified: pl.DataFrame,
  concept_column: str = "concept",
) -> pl.DataFrame:
  """One row per concept with subject counts. Values are not retained."""
  if concept_column not in classified.columns:
    raise ValueError(f"Missing {concept_column}")
  subjects = classified.select(["study_id", concept_column, "timing_class"]).unique()
  totals = subjects.group_by(concept_column).agg(pl.col("study_id").n_unique().alias("n_subjects_any_row"))
  qualified = (
    subjects.filter(pl.col("timing_class") == "qualified")
    .group_by(concept_column)
    .agg(pl.col("study_id").n_unique().alias("n_subjects_qualified"))
  )
  store_after = (
    subjects.filter(pl.col("timing_class") == "storetime_after_ecg")
    .group_by(concept_column)
    .agg(pl.col("study_id").n_unique().alias("n_subjects_storetime_after_ecg"))
  )
  store_missing = (
    subjects.filter(pl.col("timing_class") == "storetime_missing")
    .group_by(concept_column)
    .agg(pl.col("study_id").n_unique().alias("n_subjects_storetime_missing"))
  )
  return (
    totals.join(qualified, on=concept_column, how="left")
    .join(store_after, on=concept_column, how="left")
    .join(store_missing, on=concept_column, how="left")
    .with_columns(
      pl.col("n_subjects_qualified").fill_null(0),
      pl.col("n_subjects_storetime_after_ecg").fill_null(0),
      pl.col("n_subjects_storetime_missing").fill_null(0),
    )
    .sort(concept_column)
  )
