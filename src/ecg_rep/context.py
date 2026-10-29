"""Pre-index clinical-context availability rules for leakage-safe predictors.

Primary structured context ``C0`` uses age, sex, and Elixhauser indicators derived
only from hospitalizations completed before the index ECG. Index-admission discharge
diagnoses are never treated as pre-index predictors merely because they appear in
billing tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import polars as pl

AvailabilityStatus = Literal["confirmed", "conservatively_excluded", "unknown_timing"]


@dataclass(frozen=True)
class PredictorAvailabilityRule:
  """One predictor family and its temporal availability policy."""

  name: str
  description: str
  status: AvailabilityStatus
  rule: str


PRIMARY_CONTEXT_RULES: tuple[PredictorAvailabilityRule, ...] = (
  PredictorAvailabilityRule(
    name="age_at_ecg",
    description="Age derived from MIMIC anchor_age/anchor_year at ECG acquisition time",
    status="confirmed",
    rule="Available at index ECG; computed as anchor_age + (ecg_year - anchor_year).",
  ),
  PredictorAvailabilityRule(
    name="sex",
    description="Administrative sex/gender recorded in MIMIC patients",
    status="confirmed",
    rule="Treated as known at or before index ECG for primary structured context C0.",
  ),
  PredictorAvailabilityRule(
    name="prior_elixhauser",
    description="ICD-based Elixhauser indicators from completed prior hospitalizations",
    status="confirmed",
    rule=(
      "Use diagnoses only from admissions with disch_dt < index ECG time. "
      "Do not use index-admission discharge diagnoses."
    ),
  ),
  PredictorAvailabilityRule(
    name="index_admission_discharge_diagnoses",
    description="Discharge/billing ICD codes for the admission containing the index ECG",
    status="conservatively_excluded",
    rule=(
      "MIMIC hospital diagnoses lack reliable within-admission timestamps; "
      "exclude from primary C0 even if present in diagnoses_icd."
    ),
  ),
  PredictorAvailabilityRule(
    name="clinical_notes",
    description="Optional note embeddings for secondary context C1",
    status="unknown_timing",
    rule=(
      "Secondary only. Include a note only when its chart time demonstrably precedes "
      "the index ECG; exclude index-ECG interpretation text and post-index notes."
    ),
  ),
  PredictorAvailabilityRule(
    name="race_ethnicity",
    description="Race/ethnicity for subgroup heterogeneity, not default predictors",
    status="confirmed",
    rule="Reserved for RQ4 subgroup analyses unless a separate scientific rationale is locked.",
  ),
)


def prior_completed_admissions(
  admissions_df: pl.DataFrame,
  index_ecg_df: pl.DataFrame,
  subject_id_col: str = "subject_id",
  index_time_col: str = "ecg_datetime",
  discharge_col: str = "disch_dt",
) -> pl.DataFrame:
  """Return admissions completed strictly before each subject's index ECG.

  This is the temporal gate for primary Elixhauser construction. Rows whose discharge
  time is missing or not before the index ECG are dropped.
  """
  if index_time_col not in index_ecg_df.columns:
    raise ValueError(f"Index ECG frame must contain '{index_time_col}'.")
  if discharge_col not in admissions_df.columns:
    raise ValueError(f"Admissions frame must contain '{discharge_col}'.")

  index_times = index_ecg_df.select(subject_id_col, index_time_col)
  joined = admissions_df.join(index_times, on=subject_id_col, how="inner")
  return joined.filter(
    pl.col(discharge_col).is_not_null()
    & (pl.col(discharge_col) < pl.col(index_time_col))
  )


def availability_ledger_markdown(
  rules: tuple[PredictorAvailabilityRule, ...] = PRIMARY_CONTEXT_RULES,
) -> str:
  """Render the predictor availability ledger as Markdown for handoff artifacts."""
  lines = [
    "# Predictor Availability Ledger",
    "",
    "| Predictor | Status | Rule |",
    "| :--- | :--- | :--- |",
  ]
  for rule in rules:
    lines.append(
      f"| `{rule.name}` | `{rule.status}` | {rule.rule} |"
    )
  lines.append("")
  return "\n".join(lines)
