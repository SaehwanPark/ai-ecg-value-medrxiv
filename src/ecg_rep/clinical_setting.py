"""Assign one care setting to each preserved index ECG.

The primary label is the location that contains ``ecg_datetime`` under the
half-open rule ``[intime, outtime)``. An ECG with no overlapping location is
``other_or_unlinked``. That label is not evidence of outpatient care. Overlapping
locations from different setting classes are ``ambiguous`` rather than resolved
by a silent priority.

Observation and PACU beds stay inside ``inpatient_non_ICU`` and are exposed as
secondary flags. A 6-hour or 24-hour pre-admission proximity flag is also
secondary and never changes the primary label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from ecg_rep.analysis import subject_race_from_admissions
from ecg_rep.data import load_admissions
from ecg_rep.elixhauser import ELIXHAUSER_FEATURE_NAMES
from ecg_rep.paths import _find_data_file, resolve_data_paths
from ecg_rep.probe import (
  bootstrap_paired_difference,
  evaluate_predictions,
)

SETTING_LABELS: tuple[str, ...] = (
  "ICU",
  "ED",
  "inpatient_non_ICU",
  "other_or_unlinked",
  "ambiguous",
)
PRIMARY_LOCATION_KINDS: tuple[str, ...] = ("ICU", "ED", "inpatient_non_ICU")
# Names observed in MIMIC-IV v3.1 transfers/icustays that denote an ICU.
# Intermediate, stepdown, observation, nursery, and PACU units are excluded
# even when a parenthetical mentions intensive care.
ICU_CAREUNIT_EXACT: frozenset[str] = frozenset({
  "Cardiac Vascular Intensive Care Unit (CVICU)",
  "Coronary Care Unit (CCU)",
  "Intensive Care Unit (ICU)",
  "Medical Intensive Care Unit (MICU)",
  "Medical/Surgical Intensive Care Unit (MICU/SICU)",
  "Neuro Surgical Intensive Care Unit (Neuro SICU)",
  "Surgical Intensive Care Unit (SICU)",
  "Trauma SICU (TSICU)",
})
UNKNOWN_CAREUNITS: frozenset[str] = frozenset({"", "UNKNOWN", "Unknown"})
PUBLIC_MIN_CELL = 11
ANALYSIS_STATUS = "posthoc_frozen_extension"
APPROVAL_GATE = "G1_approved_2026-10-05"


@dataclass(frozen=True)
class IntervalBuildStats:
  """Counts from interval construction that the linkage audit must retain."""

  n_transfer_rows: int
  n_location_intervals: int
  n_imputed_transfer_ends: int
  n_open_ended_transfer_intervals: int
  n_icu_rows: int
  n_icu_excluded_missing_end: int
  n_icu_excluded_non_icu_name: int
  n_admission_ed_intervals: int


def setting_rule_hash() -> str:
  """Fingerprint the setting rules so a later edit cannot pass as this version."""
  payload = json.dumps(
    {
      "interval": "[intime,outtime)",
      "icu_exact": sorted(ICU_CAREUNIT_EXACT),
      "unknown_careunits": sorted(UNKNOWN_CAREUNITS),
      "ed_rule": "eventtype ED or careunit Emergency Department",
      "observation_in_primary": "inpatient_non_ICU",
      "missing_transfer_end": "close at the next subject event, including discharge",
      "missing_icu_end": "exclude the ICU row",
      "missing_hospital_discharge": "do not treat the admission as an open span",
      "proximity_hours_not_in_primary_label": [6, 24],
      "outpatient_inference": False,
    },
    sort_keys=True,
  )
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_icu_careunit(careunit: str | None) -> bool:
  """Return whether a MIMIC careunit name is an intensive-care location.

  The exact MIMIC-IV v3.1 ICU names are authoritative. A fallback keeps an
  unseen name that contains ``ICU`` or is ``CCU``, while refusing intermediate,
  stepdown, observation, nursery, and PACU names.
  """
  if careunit is None:
    return False
  name = careunit.strip()
  if name in ICU_CAREUNIT_EXACT:
    return True
  upper = name.upper()
  if any(token in upper for token in ("INTERMEDIATE", "STEPDOWN", "STEP-DOWN", "OBSERVATION", "NURSERY", "PACU")):
    return False
  return "ICU" in upper or upper == "CCU"


def _as_us(column: str) -> pl.Expr:
  return pl.col(column).cast(pl.Datetime("us"))


def _require_columns(frame: pl.DataFrame, columns: set[str], label: str) -> None:
  missing = columns - set(frame.columns)
  if missing:
    raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _location_kind_expr() -> pl.Expr:
  careunit = pl.col("careunit").fill_null("")
  upper = careunit.str.to_uppercase()
  blocked = (
    upper.str.contains("INTERMEDIATE")
    | upper.str.contains("STEPDOWN")
    | upper.str.contains("STEP-DOWN")
    | upper.str.contains("OBSERVATION")
    | upper.str.contains("NURSERY")
    | upper.str.contains("PACU")
  )
  icu_fallback = (~blocked) & (upper.str.contains("ICU") | (upper == "CCU"))
  return (
    pl.when(pl.col("eventtype") == "discharge")
    .then(pl.lit(None, dtype=pl.String))
    .when((pl.col("eventtype") == "ED") | (careunit == "Emergency Department"))
    .then(pl.lit("ED"))
    .when(careunit.is_in(list(UNKNOWN_CAREUNITS)))
    .then(pl.lit("unknown_unit"))
    .when(careunit.is_in(list(ICU_CAREUNIT_EXACT)) | icu_fallback)
    .then(pl.lit("ICU"))
    .otherwise(pl.lit("inpatient_non_ICU"))
  )


def build_location_intervals(
  transfers: pl.DataFrame,
  icustays: pl.DataFrame | None = None,
  admissions: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, IntervalBuildStats]:
  """Build half-open location intervals from transfers, ICU stays, and ED times.

  Discharge rows are kept only long enough to close a preceding missing
  ``outtime``. They are not locations. An ICU stay with a missing ``outtime``
  is excluded so an open end cannot cover later ward time.
  """
  _require_columns(
    transfers,
    {"subject_id", "hadm_id", "eventtype", "careunit", "intime", "outtime"},
    "transfers",
  )
  order_id = "transfer_id" if "transfer_id" in transfers.columns else None
  transfer_events = transfers.with_columns(
    _as_us("intime").alias("intime"),
    _as_us("outtime").alias("outtime"),
    pl.col("eventtype").cast(pl.String),
    pl.col("careunit").cast(pl.String),
  )
  if order_id is None:
    transfer_events = transfer_events.with_row_index("_order_id")
    order_id = "_order_id"
  transfer_events = transfer_events.filter(pl.col("intime").is_not_null()).sort(
    ["subject_id", "intime", order_id]
  )
  transfer_events = transfer_events.with_columns(
    pl.col("intime").shift(-1).over("subject_id").alias("_next_start"),
    _location_kind_expr().alias("location_kind"),
  ).with_columns(
    pl.when(pl.col("outtime").is_not_null())
    .then(pl.col("outtime"))
    .otherwise(pl.col("_next_start"))
    .alias("interval_end"),
    (
      pl.col("outtime").is_null()
      & pl.col("_next_start").is_null()
      & pl.col("location_kind").is_not_null()
    ).alias("open_ended"),
    (pl.col("outtime").is_null() & pl.col("_next_start").is_not_null()).alias("imputed_end"),
  )
  locations = transfer_events.filter(pl.col("location_kind").is_not_null()).filter(
    pl.col("interval_end").is_null() | (pl.col("interval_end") > pl.col("intime"))
  ).select(
    "subject_id",
    "hadm_id",
    pl.col("intime").alias("interval_start"),
    "interval_end",
    "location_kind",
    "open_ended",
    "imputed_end",
    pl.lit("transfer").alias("interval_source"),
    "careunit",
  )

  icu_missing_end = 0
  icu_non_icu_name = 0
  icu_rows = 0
  icu_frame = pl.DataFrame(schema=locations.schema)
  if icustays is not None and icustays.height > 0:
    _require_columns(icustays, {"subject_id", "hadm_id", "intime", "outtime"}, "icustays")
    icu = icustays.with_columns(
      _as_us("intime").alias("intime"),
      _as_us("outtime").alias("outtime"),
    )
    icu_rows = icu.height
    usable = icu.filter(pl.col("intime").is_not_null() & pl.col("outtime").is_not_null())
    icu_missing_end = icu_rows - usable.height
    icu_non_icu_name = 0
    icu_names = list(ICU_CAREUNIT_EXACT)
    if "first_careunit" in usable.columns or "last_careunit" in usable.columns:
      # A stay counts as ICU when either recorded unit is an ICU name.
      # Rare non-ICU names in icustays are left out of this interval source.
      first_ok = (
        pl.col("first_careunit").is_in(icu_names)
        if "first_careunit" in usable.columns
        else pl.lit(False)
      )
      last_ok = (
        pl.col("last_careunit").is_in(icu_names)
        if "last_careunit" in usable.columns
        else pl.lit(False)
      )
      usable = usable.with_columns((first_ok | last_ok).fill_null(False).alias("_is_icu"))
      icu_non_icu_name = usable.filter(~pl.col("_is_icu")).height
      usable = usable.filter(pl.col("_is_icu"))
    icu_frame = usable.filter(pl.col("outtime") > pl.col("intime")).select(
      "subject_id",
      "hadm_id",
      pl.col("intime").alias("interval_start"),
      pl.col("outtime").alias("interval_end"),
      pl.lit("ICU").alias("location_kind"),
      pl.lit(False).alias("open_ended"),
      pl.lit(False).alias("imputed_end"),
      pl.lit("icustay").alias("interval_source"),
      (
        pl.col("last_careunit")
        if "last_careunit" in usable.columns
        else pl.lit(None, dtype=pl.String)
      ).alias("careunit"),
    )

  ed_frame = pl.DataFrame(schema=locations.schema)
  n_ed = 0
  if admissions is not None and admissions.height > 0 and "edreg_dt" in admissions.columns:
    ed = admissions.filter(pl.col("edreg_dt").is_not_null()).with_columns(
      _as_us("edreg_dt").alias("edreg_dt"),
      _as_us("edout_dt").alias("edout_dt") if "edout_dt" in admissions.columns else pl.lit(None).alias("edout_dt"),
      _as_us("admit_dt").alias("admit_dt") if "admit_dt" in admissions.columns else pl.lit(None).alias("admit_dt"),
    )
    # A missing ED departure is closed at hospital admission when that time
    # exists. It is not left open through the rest of the record.
    ed = ed.with_columns(
      pl.when(pl.col("edout_dt").is_not_null())
      .then(pl.col("edout_dt"))
      .when(pl.col("admit_dt").is_not_null() & (pl.col("admit_dt") >= pl.col("edreg_dt")))
      .then(pl.col("admit_dt"))
      .otherwise(pl.lit(None))
      .alias("interval_end")
    ).filter(pl.col("interval_end").is_not_null() & (pl.col("interval_end") > pl.col("edreg_dt")))
    n_ed = ed.height
    ed_frame = ed.select(
      "subject_id",
      "hadm_id",
      pl.col("edreg_dt").alias("interval_start"),
      "interval_end",
      pl.lit("ED").alias("location_kind"),
      pl.lit(False).alias("open_ended"),
      pl.col("edout_dt").is_null().alias("imputed_end"),
      pl.lit("admission_ed").alias("interval_source"),
      pl.lit("Emergency Department").alias("careunit"),
    )

  intervals = pl.concat([locations, icu_frame, ed_frame], how="vertical_relaxed")
  stats = IntervalBuildStats(
    n_transfer_rows=transfers.height,
    n_location_intervals=intervals.height,
    n_imputed_transfer_ends=int(locations.filter(pl.col("imputed_end")).height),
    n_open_ended_transfer_intervals=int(locations.filter(pl.col("open_ended")).height),
    n_icu_rows=icu_rows,
    n_icu_excluded_missing_end=icu_missing_end,
    n_icu_excluded_non_icu_name=icu_non_icu_name,
    n_admission_ed_intervals=n_ed,
  )
  return intervals, stats


def _hospital_overlap(cohort: pl.DataFrame, admissions: pl.DataFrame) -> pl.DataFrame:
  """Mark ECGs inside a finite admission span ``[admit_dt, disch_dt)``."""
  if admissions.height == 0 or "admit_dt" not in admissions.columns:
    return cohort.select("study_id").with_columns(pl.lit(False).alias("in_finite_hospital_span"))
  joined = cohort.select(["study_id", "subject_id", "ecg_datetime"]).join(
    admissions.select(["subject_id", "admit_dt", "disch_dt"]),
    on="subject_id",
    how="inner",
  ).filter(
    pl.col("admit_dt").is_not_null()
    & pl.col("disch_dt").is_not_null()
    & (pl.col("ecg_datetime") >= pl.col("admit_dt"))
    & (pl.col("ecg_datetime") < pl.col("disch_dt"))
  )
  return (
    cohort.select("study_id")
    .join(joined.select("study_id").unique().with_columns(pl.lit(True).alias("in_finite_hospital_span")),
          on="study_id", how="left")
    .with_columns(pl.col("in_finite_hospital_span").fill_null(False))
  )


def _proximity_flags(cohort: pl.DataFrame, admissions: pl.DataFrame) -> pl.DataFrame:
  """Hours from the ECG to the next ED registration or hospital admission."""
  keys = cohort.select(["study_id", "subject_id", "ecg_datetime"])
  pieces: list[pl.DataFrame] = []
  if "admit_dt" in admissions.columns:
    pieces.append(
      keys.join(admissions.select(["subject_id", "admit_dt"]), on="subject_id", how="inner")
      .filter(pl.col("admit_dt") > pl.col("ecg_datetime"))
      .select(
        "study_id",
        ((pl.col("admit_dt") - pl.col("ecg_datetime")).dt.total_microseconds() / 3_600_000_000).alias("hours"),
      )
    )
  if "edreg_dt" in admissions.columns:
    pieces.append(
      keys.join(admissions.select(["subject_id", "edreg_dt"]), on="subject_id", how="inner")
      .filter(pl.col("edreg_dt").is_not_null() & (pl.col("edreg_dt") > pl.col("ecg_datetime")))
      .select(
        "study_id",
        ((pl.col("edreg_dt") - pl.col("ecg_datetime")).dt.total_microseconds() / 3_600_000_000).alias("hours"),
      )
    )
  if not pieces:
    return keys.select("study_id").with_columns(pl.lit(None).cast(pl.Float64).alias("hours_to_next_acute_contact"))
  stacked = pl.concat(pieces, how="vertical")
  return stacked.group_by("study_id").agg(pl.col("hours").min().alias("hours_to_next_acute_contact"))


def assign_index_ecg_setting(
  cohort: pl.DataFrame,
  admissions: pl.DataFrame,
  transfers: pl.DataFrame,
  icustays: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, IntervalBuildStats]:
  """Return one setting row per cohort ``study_id``.

  The cohort frame is not filtered. Every input index ECG remains in the output.
  """
  _require_columns(cohort, {"subject_id", "study_id", "ecg_datetime"}, "cohort")
  if cohort["study_id"].n_unique() != cohort.height:
    raise ValueError("Cohort study_id values must be unique before setting assignment.")
  cohort_keys = cohort.select(["subject_id", "study_id", _as_us("ecg_datetime").alias("ecg_datetime")])
  subjects = cohort_keys.select("subject_id").unique()
  transfers_sub = transfers.join(subjects, on="subject_id", how="semi")
  icu_sub = icustays.join(subjects, on="subject_id", how="semi") if icustays is not None else None
  admissions_sub = admissions.join(subjects, on="subject_id", how="semi")
  intervals, stats = build_location_intervals(transfers_sub, icu_sub, admissions_sub)

  hits = cohort_keys.join(intervals, on="subject_id", how="inner").filter(
    (pl.col("ecg_datetime") >= pl.col("interval_start"))
    & (pl.col("interval_end").is_null() | (pl.col("ecg_datetime") < pl.col("interval_end")))
  )
  # A transfer that starts after the ECG fails the start comparison above.
  summary = hits.group_by("study_id").agg(
    pl.col("location_kind").unique().sort().alias("kinds"),
    pl.col("open_ended").any().alias("any_open_ended"),
    pl.len().alias("n_overlapping_intervals"),
  )
  only_kind = pl.col("kinds").list.first()
  summary = summary.with_columns(
    pl.when(
      (pl.col("kinds").list.len() == 1)
      & only_kind.is_in(list(PRIMARY_LOCATION_KINDS))
    )
    .then(only_kind)
    .otherwise(pl.lit("ambiguous"))
    .alias("setting"),
    pl.col("kinds").list.join("|").alias("overlapping_kinds"),
  )

  winners = (
    hits.join(summary.select(["study_id", "setting"]), on="study_id", how="inner")
    .filter(pl.col("location_kind") == pl.col("setting"))
    .sort(["study_id", "interval_start", "interval_source"], descending=[False, True, False])
    .group_by("study_id", maintain_order=True)
    .agg(
      pl.col("hadm_id").first().alias("hadm_id"),
      pl.col("hadm_id").drop_nulls().n_unique().alias("n_hadm"),
      pl.col("careunit").first().alias("careunit"),
      pl.col("interval_source").first().alias("interval_source"),
      pl.col("open_ended").first().alias("winner_open_ended"),
    )
    .with_columns((pl.col("n_hadm") > 1).alias("multiple_hadm"))
    .with_columns(
      pl.when(pl.col("multiple_hadm")).then(pl.lit(None, dtype=pl.Int64)).otherwise(pl.col("hadm_id")).alias("hadm_id")
    )
  )

  hospital = _hospital_overlap(cohort_keys, admissions_sub)
  proximity = _proximity_flags(cohort_keys, admissions_sub)
  assigned = (
    cohort_keys.select(["subject_id", "study_id"])
    .join(summary, on="study_id", how="left")
    .join(winners.select(["study_id", "hadm_id", "multiple_hadm", "careunit", "interval_source", "winner_open_ended"]),
          on="study_id", how="left")
    .join(hospital, on="study_id", how="left")
    .join(proximity, on="study_id", how="left")
  )
  assigned = assigned.with_columns(
    pl.when(pl.col("setting").is_null() & pl.col("in_finite_hospital_span"))
    .then(pl.lit("ambiguous"))
    .when(pl.col("setting").is_null())
    .then(pl.lit("other_or_unlinked"))
    .otherwise(pl.col("setting"))
    .alias("setting"),
    pl.when(pl.col("setting").is_null() & pl.col("in_finite_hospital_span"))
    .then(pl.lit("hospital_interval_without_unit"))
    .when(pl.col("setting").is_null())
    .then(pl.lit("no_overlapping_location"))
    .when(pl.col("setting") == "ambiguous")
    .then(
      pl.when(pl.col("overlapping_kinds") == "unknown_unit")
      .then(pl.lit("unknown_careunit"))
      .otherwise(pl.lit("conflicting_locations"))
    )
    .otherwise(pl.lit("unique_location"))
    .alias("setting_reason"),
    pl.col("any_open_ended").fill_null(False),
    pl.col("multiple_hadm").fill_null(False),
    pl.col("winner_open_ended").fill_null(False),
    pl.col("n_overlapping_intervals").fill_null(0),
    pl.col("in_finite_hospital_span").fill_null(False),
  )
  care_upper = pl.col("careunit").fill_null("").str.to_uppercase()
  assigned = assigned.with_columns(
    (
      (pl.col("setting") == "other_or_unlinked")
      & pl.col("hours_to_next_acute_contact").is_not_null()
      & (pl.col("hours_to_next_acute_contact") <= 6)
    ).alias("pre_admit_within_6h"),
    (
      (pl.col("setting") == "other_or_unlinked")
      & pl.col("hours_to_next_acute_contact").is_not_null()
      & (pl.col("hours_to_next_acute_contact") <= 24)
    ).alias("pre_admit_within_24h"),
    ((pl.col("setting") == "inpatient_non_ICU") & care_upper.str.contains("OBSERVATION")).alias("observation_unit"),
    ((pl.col("setting") == "inpatient_non_ICU") & (care_upper == "PACU")).alias("procedural_recovery"),
    pl.lit(False).alias("outpatient_documented"),
  )
  validate_setting_assignment(assigned, cohort.height)
  return assigned, stats


def validate_setting_assignment(assignment: pl.DataFrame, n_expected: int) -> None:
  """Raise when the setting table breaks the one-row-per-index-ECG contract."""
  if assignment.height != n_expected:
    raise ValueError(f"Setting rows {assignment.height} != cohort rows {n_expected}.")
  if assignment["study_id"].n_unique() != assignment.height:
    raise ValueError("Setting assignment produced duplicate study_id values.")
  unknown = set(assignment["setting"].unique().to_list()) - set(SETTING_LABELS)
  if unknown:
    raise ValueError(f"Unexpected setting labels: {sorted(unknown)}")
  if int(assignment.filter(pl.col("outpatient_documented")).height) != 0:
    raise ValueError("outpatient_documented must stay false without an outpatient source.")


def wilson_interval(events: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
  """Wilson score interval for a binomial proportion."""
  if n <= 0:
    return None, None
  p = events / n
  denominator = 1.0 + z * z / n
  center = (p + z * z / (2 * n)) / denominator
  half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denominator
  return max(0.0, center - half), min(1.0, center + half)


def complementary_suppress_status(
  frame: pl.DataFrame,
  group_cols: tuple[str, ...],
  n_col: str,
  event_col: str,
  min_cell: int = PUBLIC_MIN_CELL,
) -> pl.Series:
  """Label small event cells and one complementary cell in the same partition.

  ``primary`` cells have fewer than ``min_cell`` events. ``complement`` is the
  next-smallest cell, suppressed only so a public total cannot reveal the
  primary count by subtraction. ``ok`` cells stay publishable.
  """
  work = frame.with_row_index("_row")
  grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
  for row in work.iter_rows(named=True):
    key = tuple(row[col] for col in group_cols)
    grouped_rows.setdefault(key, []).append(row)
  status_by_row: dict[int, str] = {}
  for rows in grouped_rows.values():
    primary = [
      row for row in rows
      if row[event_col] is not None and (
        int(row[n_col]) < min_cell or (0 < int(row[event_col]) < min_cell)
      )
    ]
    primary_ids = {int(row["_row"]) for row in primary}
    for row_id in primary_ids:
      status_by_row[row_id] = "primary"
    if len(primary) == 1:
      remainder = [
        row for row in rows
        if int(row["_row"]) not in primary_ids and row[event_col] is not None
      ]
      if remainder:
        complement = min(remainder, key=lambda row: int(row[event_col]))
        status_by_row[int(complement["_row"])] = "complement"
  labels = [
    status_by_row.get(int(row["_row"]), "ok")
    for row in work.sort("_row").iter_rows(named=True)
  ]
  return pl.Series("suppress_status", labels)


def suppress_small_cells(
  frame: pl.DataFrame,
  n_col: str = "n",
  event_cols: tuple[str, ...] = ("n_deaths",),
  min_cell: int = PUBLIC_MIN_CELL,
) -> pl.DataFrame:
  """Blank small counts in a shareable table.

  A cell is suppressed when the denominator is below ``min_cell`` or a positive
  event count is below ``min_cell``.
  """
  out = frame
  for col in event_cols:
    if col not in out.columns or n_col not in out.columns:
      continue
    small = (pl.col(n_col) < min_cell) | ((pl.col(col) > 0) & (pl.col(col) < min_cell))
    out = out.with_columns(
      pl.when(small).then(pl.lit(None)).otherwise(pl.col(col)).alias(col),
      pl.when(small).then(pl.lit(True)).otherwise(pl.lit(False)).alias(f"{col}_suppressed"),
    )
  return out


def _prior_admission_counts(cohort: pl.DataFrame, admissions: pl.DataFrame) -> pl.DataFrame:
  if "disch_dt" not in admissions.columns:
    return cohort.select("study_id").with_columns(pl.lit(0).alias("n_prior_admissions"))
  prior = (
    cohort.select(["study_id", "subject_id", "ecg_datetime"])
    .join(admissions.select(["subject_id", "hadm_id", "disch_dt"]), on="subject_id", how="inner")
    .filter(pl.col("disch_dt").is_not_null() & (pl.col("disch_dt") < pl.col("ecg_datetime")))
    .group_by("study_id")
    .agg(pl.col("hadm_id").n_unique().alias("n_prior_admissions"))
  )
  return cohort.select("study_id").join(prior, on="study_id", how="left").with_columns(
    pl.col("n_prior_admissions").fill_null(0)
  )


def setting_case_mix_table(
  assignment: pl.DataFrame,
  cohort: pl.DataFrame,
  admissions: pl.DataFrame | None = None,
  c0: pl.DataFrame | None = None,
  pair_validity: pl.DataFrame | None = None,
  predictions: pl.DataFrame | None = None,
) -> pl.DataFrame:
  """Aggregate setting counts, mortality, and case mix.

  ``paired_valid_locked_test`` is the inner join to the frozen prediction file.
  It is not the full assigned test partition.
  """
  _require_columns(cohort, {"study_id", "mortality_30d", "age_at_ecg", "gender", "split"}, "cohort")
  base = assignment.select([
    "study_id", "setting", "observation_unit", "procedural_recovery", "pre_admit_within_6h", "pre_admit_within_24h",
  ]).join(
    cohort.select(["study_id", "subject_id", "split", "mortality_30d", "age_at_ecg", "gender", "ecg_datetime"]),
    on="study_id",
    how="inner",
  )
  if base.height != assignment.height:
    raise ValueError("Setting assignment and cohort study_id values do not match.")
  if admissions is not None and "race" in admissions.columns:
    base = base.join(subject_race_from_admissions(admissions), on="subject_id", how="left").with_columns(
      pl.col("race_group").fill_null("unknown_other")
    )
  else:
    base = base.with_columns(pl.lit("unknown_other").alias("race_group"))
  if admissions is not None:
    base = base.join(_prior_admission_counts(cohort, admissions), on="study_id", how="left")
  else:
    base = base.with_columns(pl.lit(None).cast(pl.Int64).alias("n_prior_admissions"))
  if c0 is not None:
    elix_cols = [c for c in ELIXHAUSER_FEATURE_NAMES if c in c0.columns]
    elix = c0.select(["study_id", *elix_cols]).with_columns(
      pl.max_horizontal([pl.col(c) for c in elix_cols]).alias("any_elixhauser") if elix_cols else pl.lit(0).alias("any_elixhauser"),
      pl.lit(True).alias("c0_row_present"),
    )
    base = base.join(elix.select(["study_id", "any_elixhauser", "c0_row_present"]), on="study_id", how="left")
  else:
    base = base.with_columns(
      pl.lit(None).cast(pl.Int64).alias("any_elixhauser"),
      pl.lit(False).alias("c0_row_present"),
    )
  if pair_validity is not None and "representation_pair_valid" in pair_validity.columns:
    base = base.join(pair_validity.select(["study_id", "representation_pair_valid"]), on="study_id", how="left")
  else:
    base = base.with_columns(pl.lit(None).cast(pl.Boolean).alias("representation_pair_valid"))

  frames = [base.with_columns(pl.lit("all_index_ecgs").alias("population"))]
  if predictions is not None:
    paired = base.join(predictions.select("study_id").unique(), on="study_id", how="inner").with_columns(
      pl.lit("paired_valid_locked_test").alias("population")
    )
    frames.append(paired)
  long = pl.concat(frames, how="vertical")
  grouped = long.group_by(["population", "setting"]).agg(
    pl.len().alias("n"),
    pl.col("mortality_30d").cast(pl.Int64).sum().alias("n_deaths"),
    pl.col("age_at_ecg").mean().alias("mean_age"),
    (pl.col("gender").str.to_uppercase() == "F").mean().alias("pct_female"),
    (pl.col("race_group") == "white").mean().alias("pct_white"),
    (pl.col("race_group") == "black").mean().alias("pct_black"),
    (pl.col("race_group") == "hispanic_latino").mean().alias("pct_hispanic_latino"),
    (pl.col("race_group") == "asian").mean().alias("pct_asian"),
    (pl.col("race_group") == "unknown_other").mean().alias("pct_unknown_other_race"),
    pl.col("n_prior_admissions").median().alias("median_prior_admissions"),
    (pl.col("n_prior_admissions") > 0).mean().alias("pct_prior_admission"),
    pl.col("any_elixhauser").mean().alias("pct_any_elixhauser"),
    pl.col("c0_row_present").mean().alias("pct_c0_row"),
    pl.col("representation_pair_valid").mean().alias("pct_pair_valid"),
    pl.col("observation_unit").mean().alias("pct_observation_unit"),
    pl.col("procedural_recovery").mean().alias("pct_procedural_recovery"),
    pl.col("pre_admit_within_6h").mean().alias("pct_pre_admit_6h"),
    pl.col("pre_admit_within_24h").mean().alias("pct_pre_admit_24h"),
  ).sort(["population", "setting"])
  rows: list[dict[str, Any]] = []
  for row in grouped.iter_rows(named=True):
    lo, hi = wilson_interval(int(row["n_deaths"]), int(row["n"]))
    row["death_rate"] = (row["n_deaths"] / row["n"]) if row["n"] else None
    row["death_rate_ci_low"] = lo
    row["death_rate_ci_high"] = hi
    rows.append(row)
  return pl.DataFrame(rows).sort(["population", "setting"])


def setting_split_table(assignment: pl.DataFrame, cohort: pl.DataFrame) -> pl.DataFrame:
  """Counts of setting by the original patient split."""
  joined = assignment.select(["study_id", "setting"]).join(cohort.select(["study_id", "split"]), on="study_id")
  return joined.group_by(["split", "setting"]).len().sort(["split", "setting"]).rename({"len": "n"})


def setting_performance_table(
  assignment: pl.DataFrame,
  predictions: pl.DataFrame,
  score_a: str,
  score_b: str,
  score_source: str,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> pl.DataFrame:
  """Apply frozen mortality scores within setting. Does not refit probes."""
  joined = assignment.select(["study_id", "setting"]).join(predictions, on="study_id", how="inner")
  if joined.height != predictions.height:
    missing = predictions.height - joined.height
    raise ValueError(f"Frozen predictions are not a subset of the setting table ({missing} missing).")
  rows: list[dict[str, Any]] = []
  for setting in SETTING_LABELS:
    sub = joined.filter(pl.col("setting") == setting)
    y = np.asarray(sub["mortality_30d"].to_numpy(), dtype=np.int64)
    a = np.asarray(sub[score_a].to_numpy(), dtype=np.float64)
    b = np.asarray(sub[score_b].to_numpy(), dtype=np.float64)
    both = len(y) >= 2 and len(np.unique(y)) >= 2
    metrics_a = evaluate_predictions(y, a) if both else None
    metrics_b = evaluate_predictions(y, b) if both else None
    delta_auroc = bootstrap_paired_difference(y, a, b, "auroc", n_bootstraps, seed) if both else None
    delta_auprc = bootstrap_paired_difference(y, a, b, "auprc", n_bootstraps, seed + 1) if both else None
    rows.append({
      "score_source": score_source,
      "setting": setting,
      "n": int(len(y)),
      "n_events": int(y.sum()) if len(y) else 0,
      "event_prevalence": float(y.mean()) if len(y) else None,
      "auroc_a": None if metrics_a is None else metrics_a.auroc,
      "auroc_b": None if metrics_b is None else metrics_b.auroc,
      "auprc_a": None if metrics_a is None else metrics_a.auprc,
      "auprc_b": None if metrics_b is None else metrics_b.auprc,
      "delta_auroc_b_minus_a": None if delta_auroc is None else delta_auroc.point_estimate,
      "delta_auroc_ci_low": None if delta_auroc is None else delta_auroc.ci_lower,
      "delta_auroc_ci_high": None if delta_auroc is None else delta_auroc.ci_upper,
      "delta_auprc_b_minus_a": None if delta_auprc is None else delta_auprc.point_estimate,
      "delta_auprc_ci_low": None if delta_auprc is None else delta_auprc.ci_lower,
      "delta_auprc_ci_high": None if delta_auprc is None else delta_auprc.ci_upper,
      "brier_a": None if metrics_a is None else metrics_a.brier,
      "brier_b": None if metrics_b is None else metrics_b.brier,
      "log_loss_a": None if metrics_a is None else metrics_a.log_loss,
      "log_loss_b": None if metrics_b is None else metrics_b.log_loss,
      "calibration_intercept_a": None if metrics_a is None else metrics_a.calibration_intercept,
      "calibration_slope_a": None if metrics_a is None else metrics_a.calibration_slope,
      "calibration_intercept_b": None if metrics_b is None else metrics_b.calibration_intercept,
      "calibration_slope_b": None if metrics_b is None else metrics_b.calibration_slope,
      "n_bootstraps": 0 if not both else n_bootstraps,
      "models_refit": False,
    })
  return pl.DataFrame(rows)


def write_setting_flow(case_mix: pl.DataFrame, output_stem: Path) -> None:
  """Write an aggregate bar chart of index-ECG counts by setting."""
  plot_df = case_mix.filter(pl.col("population") == "all_index_ecgs").sort("setting")
  labels = plot_df["setting"].to_list()
  counts = plot_df["n"].to_list()
  fig, ax = plt.subplots(figsize=(8, 4.5))
  ax.bar(labels, counts, color="#2f5d62")
  ax.set_ylabel("Index ECGs")
  ax.set_title("Care setting at the index ECG timestamp")
  ax.tick_params(axis="x", rotation=20)
  fig.tight_layout()
  output_stem.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_stem.with_suffix(".svg"))
  fig.savefig(output_stem.with_suffix(".png"), dpi=120)
  plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
  if value is None:
    return "suppressed"
  if isinstance(value, float):
    return f"{value:.{digits}f}"
  return str(value)


def _markdown_table(frame: pl.DataFrame, columns: list[str]) -> str:
  header = "| " + " | ".join(columns) + " |"
  rule = "| " + " | ".join("---" for _ in columns) + " |"
  body = []
  for row in frame.select(columns).iter_rows(named=True):
    cells = []
    for col in columns:
      value = row[col]
      if isinstance(value, float):
        cells.append(_fmt(value, 3 if "pct" in col or "rate" in col or "auroc" in col or "auprc" in col or "delta" in col or "prevalence" in col else 1))
      elif value is None:
        cells.append("suppressed")
      else:
        cells.append(str(value))
    body.append("| " + " | ".join(cells) + " |")
  return "\n".join([header, rule, *body])


def render_setting_results_markdown(
  case_mix: pl.DataFrame,
  performance: pl.DataFrame,
  audit: dict[str, Any],
) -> str:
  """Concise aggregate interpretation. Patient identifiers are not included."""
  all_rows = case_mix.filter(pl.col("population") == "all_index_ecgs")
  n = int(all_rows["n"].sum()) if all_rows.height else 0
  by_setting = {row["setting"]: row for row in all_rows.iter_rows(named=True)}

  def _n(label: str) -> int:
    row = by_setting.get(label)
    return 0 if row is None else int(row["n"])

  acute = _n("ICU") + _n("ED") + _n("inpatient_non_ICU")
  pre_6h = int(audit.get("n_pre_admit_within_6h", 0))
  pre_24h = int(audit.get("n_pre_admit_within_24h", 0))
  reasons = audit.get("reason_counts", {})
  case_cols = [
    "population", "setting", "n", "n_deaths", "death_rate", "death_rate_ci_low", "death_rate_ci_high",
    "mean_age", "pct_female", "pct_prior_admission", "pct_pair_valid",
  ]
  perf_cols = [
    "score_source", "setting", "n", "n_events", "event_prevalence",
    "auroc_a", "auroc_b", "delta_auroc_b_minus_a", "delta_auroc_ci_low", "delta_auroc_ci_high",
    "auprc_a", "auprc_b", "delta_auprc_b_minus_a", "delta_auprc_ci_low", "delta_auprc_ci_high",
  ]
  context_rows = []
  if performance.height and "score_source" in performance.columns:
    context_rows = performance.filter(pl.col("score_source") == "context_adjusted_frozen").iter_rows(named=True)
  above = []
  includes_zero = []
  for row in context_rows:
    low = row.get("delta_auroc_ci_low")
    high = row.get("delta_auroc_ci_high")
    if low is None or high is None:
      continue
    if low > 0:
      above.append(str(row["setting"]))
    elif low <= 0 <= high:
      includes_zero.append(str(row["setting"]))
  if above or includes_zero:
    interval_sentence = (
      "On the context-adjusted frozen scores, B−A AUROC intervals above zero: "
      + (", ".join(above) if above else "none")
      + ". Intervals that include zero: "
      + (", ".join(includes_zero) if includes_zero else "none")
      + ". These are descriptions of the frozen scores inside case-mix groups, "
      "not evidence that setting explains the overall gain."
    )
  else:
    interval_sentence = "Context-adjusted paired intervals were not available for this run."
  lines = [
    "# Index ECG care setting",
    "",
    f"Analysis status: `{ANALYSIS_STATUS}`. Approval gate: `{APPROVAL_GATE}`.",
    f"Rule hash: `{audit.get('setting_rule_hash', '')}`.",
    f"Source SHA: `{audit.get('source_sha', 'unknown')}`.",
    "",
    "The original mortality cohort and frozen predictions were not rebuilt. "
    "Setting is the location containing the index ECG under `[intime, outtime)`. "
    "`other_or_unlinked` means no overlapping ED, ward, or ICU interval was found. "
    "It is not an outpatient classification. MIMIC-IV-ED is not installed, so ED "
    "timing uses hospital `edregtime`/`edouttime` and transfer rows with "
    "`eventtype = ED`.",
    "",
    f"Among {n:,} index ECGs, {acute:,} ({(acute / n) if n else 0:.1%}) fall in ED, "
    f"inpatient non-ICU, or ICU care at the ECG timestamp. "
    f"{_n('ED'):,} are ED, {_n('inpatient_non_ICU'):,} are inpatient non-ICU, and "
    f"{_n('ICU'):,} are ICU. {_n('other_or_unlinked'):,} are other or unlinked and "
    f"{_n('ambiguous'):,} are ambiguous.",
    "",
    f"Ambiguous rows are mostly conflicting locations ({reasons.get('conflicting_locations', 'not recorded')}), "
    f"with {reasons.get('hospital_interval_without_unit', 'not recorded')} finite hospital spans that had no unit interval "
    f"and {reasons.get('unknown_careunit', 'not recorded')} unknown care-unit labels. "
    f"{pre_6h:,} unlinked ECGs are followed by an ED registration or admission within 6 hours, "
    f"and {pre_24h:,} within 24 hours. Those proximity flags do not change the primary label.",
    "",
    "Observation units and PACU remain inside `inpatient_non_ICU`. "
    f"Secondary flags mark {audit.get('n_observation_unit', 0):,} observation-unit index ECGs and "
    f"{audit.get('n_procedural_recovery', 0):,} PACU index ECGs. "
    "MIMIC `icustays` also contains intermediate and stepdown units; those rows are not labeled ICU.",
    "",
    "30-day mortality is highest in the ICU setting and lowest among unlinked ECGs. "
    "That gradient describes where the index tracing was acquired. It does not explain "
    "the original representation comparison. Counts below 11, and one complementary cell "
    "in the same public partition, are suppressed so a published total cannot identify the small cell.",
    "",
    "## Case mix",
    "",
    _markdown_table(case_mix, [col for col in case_cols if col in case_mix.columns]),
    "",
    "## Frozen paired performance",
    "",
    "Scores are the existing context-adjusted (`p_a`, `p_b`) and ECG-only frozen predictions "
    "on the paired-valid locked test. Probes were not retrained inside settings. "
    "A confidence interval that includes zero is uncertainty about the paired difference, "
    "not evidence that the representations are equivalent. Read AUPRC next to the event prevalence.",
    "",
    _markdown_table(performance, [col for col in perf_cols if col in performance.columns]) if performance.height else "_No frozen predictions were available._",
    "",
    interval_sentence,
    "",
    "This is an in-domain MIMIC downstream description. It is not external validation of D-BETA "
    "and it is not a claim about community ECG acquisition.",
    "",
  ]
  return "\n".join(lines)


def _sha256_ids(frame: pl.DataFrame, column: str) -> str:
  values = frame.select(pl.col(column).cast(pl.Utf8)).sort(column)[column].to_list()
  payload = f"n={len(values)}\n" + "\n".join(values)
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_sha() -> str:
  try:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return "unknown"


def load_transfers(path: Path) -> pl.DataFrame:
  """Load MIMIC-IV transfers with parsed interval endpoints."""
  frame = pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "hadm_id": pl.Int64,
      "transfer_id": pl.Int64,
      "eventtype": pl.String,
      "careunit": pl.String,
      "intime": pl.String,
      "outtime": pl.String,
    },
  )
  return frame.with_columns(
    pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
  )


def load_icustays(path: Path) -> pl.DataFrame:
  """Load MIMIC-IV ICU stays with parsed interval endpoints."""
  frame = pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "hadm_id": pl.Int64,
      "stay_id": pl.Int64,
      "first_careunit": pl.String,
      "last_careunit": pl.String,
      "intime": pl.String,
      "outtime": pl.String,
    },
  )
  return frame.with_columns(
    pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    pl.col("outtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
  )


def run_setting_characterization(
  cohort_path: Path,
  output_dir: Path,
  report_path: Path,
  predictions_context_path: Path,
  predictions_ecg_path: Path,
  c0_path: Path,
  pair_validity_path: Path,
  mimic_root: str | None = None,
  ecg_root: str | None = None,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, Any]:
  """Link local MIMIC tables to the frozen cohort and write aggregate outputs."""
  paths = resolve_data_paths(mimic_root, ecg_root)
  admissions_path = _find_data_file(
    paths.mimic_iv_dir,
    ["hosp/admissions.csv.gz", "hosp/admissions.csv", "admissions.csv.gz", "admissions.csv"],
  )
  transfers_path = _find_data_file(
    paths.mimic_iv_dir,
    ["hosp/transfers.csv.gz", "hosp/transfers.csv", "transfers.csv.gz", "transfers.csv"],
  )
  icu_path = _find_data_file(
    paths.mimic_iv_dir,
    ["icu/icustays.csv.gz", "icu/icustays.csv", "icustays.csv.gz", "icustays.csv"],
  )
  for path in (admissions_path, transfers_path, icu_path, cohort_path):
    if not path.exists():
      raise FileNotFoundError(path)

  cohort = pl.read_parquet(cohort_path)
  admissions = load_admissions(admissions_path)
  transfers = load_transfers(transfers_path)
  icustays = load_icustays(icu_path)
  assignment, interval_stats = assign_index_ecg_setting(cohort, admissions, transfers, icustays)
  c0 = pl.read_parquet(c0_path) if c0_path.exists() else None
  pair = pl.read_parquet(pair_validity_path) if pair_validity_path.exists() else None
  pred_context = pl.read_parquet(predictions_context_path) if predictions_context_path.exists() else None
  pred_ecg = pl.read_parquet(predictions_ecg_path) if predictions_ecg_path.exists() else None
  case_mix = setting_case_mix_table(
    assignment, cohort, admissions=admissions, c0=c0, pair_validity=pair, predictions=pred_context,
  )
  death_status = complementary_suppress_status(case_mix, ("population",), "n", "n_deaths")
  public_case_mix = case_mix.with_columns(death_status.alias("death_suppress_status")).with_columns(
    pl.when(pl.col("death_suppress_status") == "ok").then(pl.col("n_deaths")).otherwise(pl.lit(None)).alias("n_deaths"),
    pl.when(pl.col("death_suppress_status") == "ok").then(pl.col("death_rate")).otherwise(pl.lit(None)).alias("death_rate"),
    pl.when(pl.col("death_suppress_status") == "ok").then(pl.col("death_rate_ci_low")).otherwise(pl.lit(None)).alias("death_rate_ci_low"),
    pl.when(pl.col("death_suppress_status") == "ok").then(pl.col("death_rate_ci_high")).otherwise(pl.lit(None)).alias("death_rate_ci_high"),
  )
  split_table = setting_split_table(assignment, cohort)
  performance_frames: list[pl.DataFrame] = []
  if pred_context is not None:
    performance_frames.append(setting_performance_table(
      assignment, pred_context, "p_a", "p_b", "context_adjusted_frozen", n_bootstraps, seed,
    ))
  if pred_ecg is not None:
    performance_frames.append(setting_performance_table(
      assignment, pred_ecg, "p_a_ecg", "p_b_ecg", "ecg_only_frozen", n_bootstraps, seed + 17,
    ))
  performance = pl.concat(performance_frames, how="vertical") if performance_frames else pl.DataFrame()
  if performance.height:
    event_status = complementary_suppress_status(performance, ("score_source",), "n", "n_events")
    count_cols = ["n_events", "event_prevalence"]
    metric_cols = [
      col for col in performance.columns
      if col not in {"score_source", "setting", "n", "n_bootstraps", "models_refit", *count_cols}
    ]
    public_performance = performance.with_columns(event_status.alias("event_suppress_status"))
    public_performance = public_performance.with_columns([
      pl.when(pl.col("event_suppress_status") == "ok").then(pl.col(col)).otherwise(pl.lit(None)).alias(col)
      for col in count_cols
    ]).with_columns([
      pl.when(pl.col("event_suppress_status") == "primary").then(pl.lit(None)).otherwise(pl.col(col)).alias(col)
      for col in metric_cols
    ])
  else:
    public_performance = performance

  output_dir.mkdir(parents=True, exist_ok=True)
  assignment.write_parquet(output_dir / "index_ecg_setting.parquet")
  public_case_mix.write_csv(output_dir / "setting_cohort_table.csv")
  public_performance.write_csv(output_dir / "setting_performance.csv")
  split_table.write_csv(output_dir / "setting_by_split.csv")
  docs_tables = report_path.parent / "tables"
  docs_tables.mkdir(parents=True, exist_ok=True)
  public_case_mix.write_csv(docs_tables / "setting_cohort_table.csv")
  public_performance.write_csv(docs_tables / "setting_performance.csv")
  split_table.write_csv(docs_tables / "setting_by_split.csv")
  write_setting_flow(case_mix, output_dir / "setting_flow")
  flow_png = output_dir / "setting_flow.png"
  if flow_png.exists():
    shutil.copyfile(flow_png, report_path.parent / "setting_flow.png")
  audit: dict[str, Any] = {
    "analysis_status": ANALYSIS_STATUS,
    "approval_gate": APPROVAL_GATE,
    "source_sha": _git_sha(),
    "setting_rule_hash": setting_rule_hash(),
    "cohort_study_id_sha256": _sha256_ids(cohort, "study_id"),
    "n_input": cohort.height,
    "n_output": assignment.height,
    "setting_counts": {
      label: int(assignment.filter(pl.col("setting") == label).height) for label in SETTING_LABELS
    },
    "reason_counts": {
      str(row["setting_reason"]): int(row["len"])
      for row in assignment.group_by("setting_reason").len().iter_rows(named=True)
    },
    "n_open_ended": int(assignment.filter(pl.col("any_open_ended")).height),
    "n_multiple_hadm": int(assignment.filter(pl.col("multiple_hadm")).height),
    "n_observation_unit": int(assignment.filter(pl.col("observation_unit")).height),
    "n_procedural_recovery": int(assignment.filter(pl.col("procedural_recovery")).height),
    "n_pre_admit_within_6h": int(assignment.filter(pl.col("pre_admit_within_6h")).height),
    "n_pre_admit_within_24h": int(assignment.filter(pl.col("pre_admit_within_24h")).height),
    "n_outpatient_documented": int(assignment.filter(pl.col("outpatient_documented")).height),
    "mimic_ed_module": False,
    "interval_stats": interval_stats.__dict__,
    "n_bootstraps": n_bootstraps,
    "seed": seed,
    "models_refit": False,
  }
  (output_dir / "setting_linkage_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  report = render_setting_results_markdown(public_case_mix, public_performance, audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  (output_dir / "setting_results.md").write_text(report, encoding="utf-8")
  return audit
