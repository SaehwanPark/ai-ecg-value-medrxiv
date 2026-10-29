"""Characterize CODE-15% extraction failures by age, sex, and 1-year mortality.

Audit02 smaller item: the 1,267 recorded extraction failures (0.54% of the
index cohort) could preferentially remove high-risk patients. This module
joins the extraction manifest's failure records to the index cohort by
``exam_id`` and reports aggregate-only summaries. No patient-level values are
emitted; mortality cells below the suppression threshold are withheld,
matching the study's small-cell contract.
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl

#: Small-cell suppression threshold, consistent with the other CODE modules.
MIN_CELL_COUNT = 11

#: Top-level failure categories recorded by the extraction driver.
FAILURE_CATEGORIES: tuple[str, ...] = ("standardize_error", "qc_gate_failed", "ciis_error")

_AGE_STATS = ("mean", "median")


def parse_failure_records(failures: dict[str, str]) -> pl.DataFrame:
  """Parse manifest ``failure`` entries into ``(exam_id, category)`` rows."""
  rows: list[dict[str, Any]] = []
  for key, reason in failures.items():
    if "exam" not in key:
      raise ValueError(f"Malformed failure key: {key!r}")
    exam_id = int(key.rsplit("exam", 1)[-1])
    category = reason.split(":", 1)[0]
    if category not in FAILURE_CATEGORIES:
      raise ValueError(f"Unknown failure category in record: {category!r}")
    rows.append({"exam_id": exam_id, "category": category})
  if not rows:
    raise ValueError("Extraction manifest records no failures.")
  return pl.DataFrame(rows, schema={"exam_id": pl.Int64, "category": pl.String})


def _suppressed(count: int) -> int | str:
  return count if count >= MIN_CELL_COUNT else "suppressed(<11)"


def _group_summary(frame: pl.DataFrame, label_col: str) -> dict[str, Any]:
  label_key = label_col.removeprefix("y_")
  age_mean = cast("float | None", frame["age"].mean())
  age_median = cast("float | None", frame["age"].median())
  n_known = int(frame.filter(pl.col(label_col).is_not_null()).height)
  known = frame.filter(pl.col(label_col).is_not_null())
  deaths = int(known.filter(pl.col(label_col) == 1).height)
  rate = float(deaths / n_known) if n_known and deaths >= MIN_CELL_COUNT else None
  pct_female = (
    round(100.0 * float(frame.filter(~pl.col("is_male")).height) / float(frame.height), 2)
    if frame.height
    else None
  )
  return {
    "n": int(frame.height),
    "age_mean": round(float(age_mean), 2) if age_mean is not None else None,
    "age_median": round(float(age_median), 2) if age_median is not None else None,
    "pct_female": pct_female,
    f"n_known_{label_key}": n_known,
    f"n_deaths_{label_key}": _suppressed(deaths),
    f"mortality_rate_{label_key}": round(rate, 5) if rate is not None else None,
  }


def summarize_failure_demographics(
  failures: dict[str, str],
  cohort: pl.DataFrame,
  horizon: str = "365d",
) -> dict[str, Any]:
  """Aggregate age/sex/mortality characterization of extraction failures.

  ``cohort`` must contain ``exam_id``, ``age``, ``is_male`` and the horizon
  label column, whose ``event``/``nonevent``/``unknown`` values are recast to
  nullable 0/1 ints with the shared E1 label converter. All counts are
  aggregate; mortality cells below ``MIN_CELL_COUNT`` are suppressed.
  """
  from ecg_rep.external.code15_labels import horizon_label_to_int

  label_col = f"y_{horizon}"
  required = {"exam_id", "age", "is_male", label_col}
  missing = required - set(cohort.columns)
  if missing:
    raise ValueError(f"Cohort frame is missing columns: {sorted(missing)}.")
  cohort = horizon_label_to_int(cohort, horizon)
  failure_rows = parse_failure_records(failures)
  n_failure_records = failure_rows.height
  n_duplicate_exam_ids = int(failure_rows["exam_id"].n_unique())

  joined = cohort.select(["exam_id", "age", "is_male", label_col]).join(
    failure_rows, on="exam_id", how="inner",
  )
  unmatched = n_failure_records - joined.height
  if unmatched:
    raise ValueError(
      f"{unmatched} failure exam IDs did not join to the index cohort; refusing to summarize.",
    )

  categories: dict[str, dict[str, Any]] = {}
  for category in FAILURE_CATEGORIES:
    group = joined.filter(pl.col("category") == category)
    if group.height == 0:
      continue
    categories[category] = _group_summary(group, label_col)

  return {
    "method": (
      "extraction-manifest failures joined to the index cohort by exam_id; "
      "aggregate-only summaries, outcome cells below 11 suppressed"
    ),
    "horizon": horizon,
    "label_column": label_col,
    "n_failure_records": n_failure_records,
    "n_distinct_failed_exam_ids": n_duplicate_exam_ids,
    "failure_rate": round(failure_rows.height / cohort.height, 6),
    "by_category": categories,
    "all_failures": _group_summary(joined, label_col),
    "cohort_reference": _group_summary(cohort, label_col),
  }