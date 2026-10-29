"""E1: Endpoint, unit, first-ECG index, and censoring feasibility for CODE-15%.

This module builds the single-row-per-patient mortality-annotated first-ECG
index cohort and derives prespecifiable binary horizon labels from the
CODE-15% ``death``/``timey`` fields.

Governance (plan Agent E1, gate G-E1):

- ``timey`` physical units must be resolved from primary documentation **and**
  observed values before any label construction. Only the verified unit
  (``years``) is accepted; anything else fails fast with ``outcome_status=
  unresolved`` semantics.
- Labels are feasibility census quantities. They are **not** used for model
  development, feature selection, or locked-test outcome access.
- Boundary inclusion matches the MIMIC primary estimand: death on day ``h``
  is an event (inclusive upper boundary).
- All outputs are aggregate counts or patient-index assignments persisted
  only in the gitignored artifact tree; nothing patient-level is committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

#: The only verified ``timey`` unit. See ``TIMEY_UNIT_EVIDENCE`` and the
#: endpoint memo for the verification chain before this was accepted.
VERIFIED_TIMEY_UNIT = "years"

#: Julian year used for the years -> days conversion (matches the prespecified
#: 1-year fallback horizon of 365.25 days in the plan).
DAYS_PER_YEAR = 365.25

#: Prespecified horizons in days (30-day primary; 90-day supplementary;
#: 365.25-day separately labeled fallback).
HORIZON_DAYS: dict[str, float] = {
  "30d": 30.0,
  "90d": 90.0,
  "365d": 365.25,
}

#: Operational precision screen from the plan (not a power guarantee).
MIN_EVENTS_GO_NO_GO = 100

LABEL_EVENT = "event"
LABEL_NONEVENT = "nonevent"
LABEL_UNKNOWN = "unknown"


def horizon_label_to_int(frame: Any, horizon: str) -> Any:
  """Return ``frame`` with ``y_{horizon}`` recast to nullable 0/1 ints.

  ``event``→1, ``nonevent``→0, ``unknown``/missing→null. Works on any
  ``polars.DataFrame`` with the E1 label column.
  """
  import polars as pl

  label_col = f"y_{horizon}"
  if frame.schema[label_col] == pl.String:
    frame = frame.with_columns(
      pl.when(pl.col(label_col) == LABEL_EVENT)
      .then(1)
      .when(pl.col(label_col) == LABEL_NONEVENT)
      .then(0)
      .otherwise(None)
      .alias(label_col),
    )
  return frame.with_columns(pl.col(label_col).cast(pl.Int64))


@dataclass(frozen=True)
class IndexCohortResult:
  """Index cohort plus the attrition ledger for its construction."""

  cohort: pl.DataFrame
  attrition: dict[str, int]
  issues: list[str]


def resolve_timey_unit(unit: str | None) -> str:
  """Fail fast on unapproved or unknown ``timey`` units.

  Only the unit verified against primary documentation and observed values
  (``years``) may be used. Passing ``None`` or any other value models the
  ``outcome_status=unresolved`` path required by the plan.
  """
  if unit != VERIFIED_TIMEY_UNIT:
    raise ValueError(
      f"timey unit '{unit}' is not verified/approved; mortality modeling is "
      f"blocked until the endpoint memo (gate G-E1) confirms '{VERIFIED_TIMEY_UNIT}'."
    )
  return VERIFIED_TIMEY_UNIT


def timey_to_days(timey_years: float | np.ndarray) -> float | np.ndarray:
  """Convert verified ``timey`` years to days using the documented formula.

  Conversion formula recorded in the endpoint memo: ``days = timey * 365.25``
  (Julian year), matching the prespecified 1-year fallback horizon.
  """
  return timey_years * DAYS_PER_YEAR


def detect_first_exam_annotation_issues(exams: pl.DataFrame) -> list[str]:
  """Detect duplicate or inconsistent first-exam mortality annotations.

  The release guarantees ``death``/``timey`` on the first exam per patient
  only. Multiple annotated rows per patient, rows with exactly one of the
  two fields present, and annotated rows without a trace file are
  investigated here rather than silently resolved.
  """
  issues: list[str] = []
  annotated = exams.filter(pl.col("death").is_not_null() & pl.col("timey").is_not_null())
  dup_patients = annotated.group_by("patient_id").len().filter(pl.col("len") > 1)
  if dup_patients.height > 0:
    issues.append(
      f"{dup_patients.height} patients have more than one mortality-annotated row "
      f"({int(dup_patients['len'].sum())} annotated rows total)."
    )
  partial = exams.filter(
    (pl.col("death").is_not_null() & pl.col("timey").is_null())
    | (pl.col("death").is_null() & pl.col("timey").is_not_null())
  )
  if partial.height > 0:
    issues.append(
      f"{partial.height} rows have exactly one of death/timey present "
      "(expected: both or neither)."
    )
  untraceable = annotated.filter(pl.col("trace_file").is_null())
  if untraceable.height > 0:
    issues.append(
      f"{untraceable.height} annotated rows have a missing trace_file."
    )
  missing_annotation = exams.filter(
    pl.col("death").is_null() & pl.col("timey").is_null()
  ).group_by("patient_id").len()
  fully_unannotated = missing_annotation.join(
    annotated.select("patient_id").unique(),
    on="patient_id",
    how="anti",
  )
  if fully_unannotated.height > 0:
    issues.append(
      f"{fully_unannotated.height} patients have no mortality-annotated row at all."
    )
  return issues


def build_index_cohort(exams: pl.DataFrame) -> IndexCohortResult:
  """Build the one-row-per-patient mortality-annotated first-ECG index cohort.

  Eligibility: exactly one row per patient with both ``death`` and ``timey``
  present (the release annotates the first exam only). This does **not**
  assume that the minimum ``exam_id`` or CSV row order is chronologically
  first; the release annotation itself defines the index exam.

  The returned cohort carries only the index keys, demographics, and raw
  annotation fields; labels are added separately by
  :func:`attach_horizon_labels`.
  """
  n_raw_rows = int(exams.height)
  n_raw_patients = int(exams["patient_id"].n_unique())

  annotated = exams.filter(
    pl.col("death").is_not_null() & pl.col("timey").is_not_null()
  )
  n_annotated_rows = int(annotated.height)

  dup_counts = annotated.group_by("patient_id").len().filter(pl.col("len") > 1)
  n_duplicate_extra_rows = (
    int(dup_counts["len"].sum()) - int(dup_counts.height) if dup_counts.height > 0 else 0
  )

  # Deterministic reconciliation for duplicate annotations: keep the first
  # row in file order per patient and flag the inconsistency loudly.
  issues = detect_first_exam_annotation_issues(exams)
  cohort = (
    annotated.sort("exam_id")
    .unique(subset="patient_id", keep="first", maintain_order=True)
    .select(["patient_id", "exam_id", "age", "is_male", "death", "timey", "trace_file"])
    .sort("patient_id")
  )

  attrition = {
    "n_raw_exam_rows": n_raw_rows,
    "n_raw_patients": n_raw_patients,
    "n_annotated_rows": n_annotated_rows,
    "n_duplicate_annotation_rows_dropped": n_duplicate_extra_rows,
    "n_unannotated_patients_excluded": n_raw_patients - int(cohort.height),
    "n_index_cohort_patients": int(cohort.height),
  }
  return IndexCohortResult(cohort=cohort, attrition=attrition, issues=issues)


def classify_horizon_label(
  death: bool | None,
  timey_years: float | None,
  horizon_days: float,
) -> str:
  """Classify one patient against a mortality horizon (prespecified rule).

  Rule (plan E1, inclusive boundary consistent with the MIMIC estimand
  ``death on day h included``):

  - ``event``: death confirmed and observed event time is within ``h``
    (``event_days <= h``).
  - ``nonevent``: no death before/at ``h`` and the observed death/censor
    time is at or after ``h`` (``follow_up_days >= h``).
  - ``unknown``: censored before ``h``, missing follow-up, or timing that
    cannot be resolved.

  Examples (h = 30 days):
    censored at 29.9 d -> unknown; survived 30 d -> nonevent;
    death at 30 d -> event; death at 30.1 d -> nonevent (alive through
    day 30; the later death is outside the horizon); missing time -> unknown.
  """
  if timey_years is None:
    return LABEL_UNKNOWN
  observed_days = float(timey_to_days(timey_years))
  if death:
    return LABEL_EVENT if observed_days <= horizon_days else LABEL_NONEVENT
  return LABEL_NONEVENT if observed_days >= horizon_days else LABEL_UNKNOWN


def attach_horizon_labels(
  cohort: pl.DataFrame,
  horizons: dict[str, float] | None = None,
) -> pl.DataFrame:
  """Attach one deterministic label column per horizon to the index cohort."""
  horizons = horizons if horizons is not None else HORIZON_DAYS
  out = cohort
  for name, horizon_days in horizons.items():
    labels = [
      classify_horizon_label(death, timey, horizon_days)
      for death, timey in zip(
        out["death"].to_list(), out["timey"].to_list(), strict=True,
      )
    ]
    out = out.with_columns(pl.Series(f"y_{name}", labels, dtype=pl.String))
  return out


def timey_unit_evidence(annotated: pl.DataFrame) -> dict[str, Any]:
  """Aggregate empirical evidence supporting the verified ``timey`` unit.

  Returned statistics are aggregate-only. The decision chain is documented
  in the endpoint memo:

  1. Official release description: ``timey`` is "the time to the death of
     the patient" or "the follow-up time" (unit named by the companion
     study's survival analyses, which report follow-up in years and
     evaluate 1-year mortality).
  2. Observed range 0-7.68 against a 2010-2016 collection window is only
     consistent with years (days would cap follow-up below 8 days for the
     entire cohort; the maximum observed value exceeds any plausible
     day/month read by orders of magnitude).
  3. Values are not quantized to whole days, whole months, or any coarse
     grid examined, consistent with timestamp-derived decimal years.
  """
  timey = annotated["timey"].to_numpy().astype(float)
  quantization = {}
  for label, unit_years in {
    "day_365.25": 1.0 / 365.25,
    "day_365": 1.0 / 365.0,
    "month": 1.0 / 12.0,
  }.items():
    scaled = timey / unit_years
    resid = np.abs(scaled - np.round(scaled))
    resid = np.minimum(resid, 1.0 - resid)
    quantization[label] = {
      "share_within_1e-6": float(np.mean(resid < 1e-6)),
      "max_residual_units": float(np.max(np.minimum(resid, 0.5))),
    }
  deaths = annotated.filter(pl.col("death"))["timey"].to_numpy().astype(float)
  survivors = annotated.filter(~pl.col("death"))["timey"].to_numpy().astype(float)
  return {
    "verified_unit": VERIFIED_TIMEY_UNIT,
    "conversion_to_days": f"days = timey * {DAYS_PER_YEAR}",
    "n_annotated": int(annotated.height),
    "timey_min_years": float(np.min(timey)),
    "timey_max_years": float(np.max(timey)),
    "timey_median_years": float(np.median(timey)),
    "quantization_tests": quantization,
    "n_deaths": int(len(deaths)),
    "survivor_min_followup_years": float(np.min(survivors)) if len(survivors) else None,
  }


def feasibility_census(cohort: pl.DataFrame) -> dict[str, Any]:
  """Aggregate pre-analysis table for the prespecified horizons.

  Counts events, fully observed negatives, and unknowns per horizon plus
  follow-up quantiles and age/sex summaries of event patients. All
  aggregate-only; safe to commit.
  """
  horizons: dict[str, Any] = {}
  for name in HORIZON_DAYS:
    col = f"y_{name}"
    events = int((cohort[col] == LABEL_EVENT).sum())
    nonevents = int((cohort[col] == LABEL_NONEVENT).sum())
    unknowns = int((cohort[col] == LABEL_UNKNOWN).sum())
    event_rows = cohort.filter(cohort[col] == LABEL_EVENT)
    ages = event_rows["age"].to_numpy()
    male_flags = event_rows["is_male"].to_numpy().astype(float)
    horizons[name] = {
      "n_events": events,
      "n_nonevents_complete_followup": nonevents,
      "n_unknown": unknowns,
      "prevalence_among_known": round(events / (events + nonevents), 6)
      if (events + nonevents) > 0
      else None,
      "event_age_median": float(np.median(ages)) if len(ages) else None,
      "event_share_male": round(float(male_flags.mean()), 6) if len(male_flags) else None,
    }
  followup_days = timey_to_days(cohort["timey"].to_numpy().astype(float))
  quantiles = np.quantile(followup_days, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
  return {
    "n_index_patients": int(cohort.height),
    "followup_days_quantiles": {
      "p0": round(float(quantiles[0]), 2),
      "p5": round(float(quantiles[1]), 2),
      "p25": round(float(quantiles[2]), 2),
      "p50": round(float(quantiles[3]), 2),
      "p75": round(float(quantiles[4]), 2),
      "p95": round(float(quantiles[5]), 2),
      "p100": round(float(quantiles[6]), 2),
    },
    "horizons": horizons,
  }


def build_patient_split(
  cohort: pl.DataFrame,
  seed: int = 1912,
  ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
  stratify_label: str = "y_30d",
) -> pl.DataFrame:
  """Deterministic patient-level 60/20/20 development/validation/test split.

  Stratified by the primary horizon label class (event vs non-event vs
  unknown), defined from the development protocol before any locked-test
  access. One row per patient; assignments are meant to be persisted only
  in the gitignored artifact directory.
  """
  if abs(sum(ratios) - 1.0) > 1e-9:
    raise ValueError("Split ratios must sum to 1.")
  rng = np.random.default_rng(seed)

  patients = cohort.select(["patient_id", stratify_label]).sort("patient_id")
  assignments = np.empty(len(patients), dtype=object)
  # Sorted label classes keep the shared RNG stream deterministic across runs.
  label_classes = sorted(patients[stratify_label].unique().to_list())
  for label_class in label_classes:
    mask = (patients[stratify_label] == label_class).to_numpy()
    class_ids = np.flatnonzero(mask)
    rng.shuffle(class_ids)
    n = len(class_ids)
    n_dev = int(np.floor(ratios[0] * n))
    n_val = int(np.floor(ratios[1] * n))
    assignments[class_ids[:n_dev]] = "dev"
    assignments[class_ids[n_dev:n_dev + n_val]] = "val"
    assignments[class_ids[n_dev + n_val:]] = "test"

  return patients.with_columns(
    pl.Series("split", assignments, dtype=pl.String),
  ).select(["patient_id", "split"])


def go_no_go_decision(
  cohort: pl.DataFrame,
  split: pl.DataFrame,
  horizon: str = "30d",
  min_events: int = MIN_EVENTS_GO_NO_GO,
) -> dict[str, Any]:
  """Apply the prespecified prospective go/no-go rule (plan E1).

  The rule screens the held-out evaluation splits: fewer than
  ``min_events`` confirmed 30-day events in the locked test split forbids
  narrow subgroup/cascade claims and favors descriptive metrics and the
  longer-horizon fallback. Per-split event counts are computed here from
  the cohort/split join; only aggregate counts are returned.
  """
  label_col = f"y_{horizon}"
  joined = split.join(cohort, on="patient_id", how="inner")
  per_split: dict[str, int] = {}
  for split_name in ("dev", "val", "test"):
    per_split[split_name] = int(
      (joined.filter(pl.col("split") == split_name)[label_col] == LABEL_EVENT).sum(),
    )
  test_events = per_split["test"]
  decision = "GO-30d" if test_events >= min_events else "NO-GO-30d"
  return {
    "rule": (
      f"at least {min_events} confirmed {horizon} events in the locked test "
      "split for the primary estimand"
    ),
    "min_events": min_events,
    "decision": decision,
    "per_split_events": per_split,
    "consequence_if_no_go": (
      "favors descriptive metrics and the prespecified longer-horizon "
      "fallback; narrow subgroup/cascade claims are abandoned"
    ),
  }
