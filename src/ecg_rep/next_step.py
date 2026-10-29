"""A6: post-ECG next-step evaluation outcomes (MIMIC only).

Decision D12 (proxy clinical decision under D11). Plan:
``docs/plan_20261005_a6a7/a6-next-step-evaluation-plan.md``.

Actions are downstream process outcomes. They are never predictors, and "no action
observed" is not "no action taken": care outside BIDMC, transfers out, goals-of-care
decisions, and death before the window closed all produce an unobserved action.

This module has pure window and eligibility logic plus the development-only
feasibility runner. The feasibility runner never joins scores or mortality.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
import hashlib
import json

import polars as pl
import yaml

from ecg_rep.cacute import latest_plausible_labs

SETTINGS_IN_SCOPE: tuple[str, ...] = ("ED", "inpatient_non_ICU", "other_or_unlinked")
SETTING_ICU = "ICU"
SETTING_AMBIGUOUS = "ambiguous"
STATUS_DRAFT = "draft_pre_feasibility"
STATUS_FROZEN = "frozen_after_feasibility"
BASIC_LAB_CONCEPTS: tuple[str, ...] = ("creatinine", "potassium", "hemoglobin", "wbc")
# Recorded after the development-only feasibility gate; the locked-test command compares against it.
FROZEN_CONFIG_HASH = "e2eaa22e12c1b2bdac240af3c6b4e2e598ae08e439c785e66d79a54be48b70a2"


@dataclass(frozen=True)
class OutcomeSpec:
  """One frozen outcome row."""

  outcome_id: str
  name: str
  window_hours: int
  settings: tuple[str, ...]
  source: str
  min_gap_minutes: int = 0
  primary_population: str | None = None
  stratify_by: str | None = None
  eligibility: str | None = None


@dataclass(frozen=True)
class NextStepConfig:
  """Parsed ``next_step_outcomes_v1.yaml`` with the sha256 of its text."""

  version: str
  status: str
  config_hash: str
  outcomes: tuple[OutcomeSpec, ...]
  gates: dict[str, float]
  lab_items: dict[str, Any]
  poe_orders: dict[str, Any]
  reacquisition_gap_minutes: int
  lab_naive_lookback_hours: int
  prior_troponin_lookback_hours: int
  dropped_cells: tuple[dict[str, str], ...]

  def outcome(self, outcome_id: str) -> OutcomeSpec:
    for spec in self.outcomes:
      if spec.outcome_id == outcome_id:
        return spec
    raise KeyError(outcome_id)

  @property
  def basic_itemids(self) -> list[int]:
    return [int(item) for ids in self.lab_items["basic"].values() for item in ids]

  @property
  def troponin_itemids(self) -> list[int]:
    return [int(item) for item in (*self.lab_items["troponin_t"], *self.lab_items["troponin_i"])]


def load_config(path: Path) -> NextStepConfig:
  """Load and fingerprint the outcome config."""
  text = path.read_text(encoding="utf-8")
  raw = yaml.safe_load(text)
  if not isinstance(raw, dict):
    raise ValueError(f"Outcome config at {path} is not a mapping.")
  outcomes = tuple(
    OutcomeSpec(
      outcome_id=str(item["id"]),
      name=str(item["name"]),
      window_hours=int(item["window_hours"]),
      settings=tuple(str(s) for s in item["settings"]),
      source=str(item["source"]),
      min_gap_minutes=int(item.get("min_gap_minutes", 0)),
      primary_population=item.get("primary_population"),
      stratify_by=item.get("stratify_by"),
      eligibility=item.get("eligibility"),
    )
    for item in raw["outcomes"]
  )
  for spec in outcomes:
    bad = set(spec.settings) - set(SETTINGS_IN_SCOPE)
    if bad:
      raise ValueError(f"Outcome {spec.outcome_id} lists out-of-scope settings {sorted(bad)}.")
  return NextStepConfig(
    version=str(raw["version"]),
    status=str(raw["status"]),
    config_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    outcomes=outcomes,
    gates={key: float(value) for key, value in raw["feasibility_gates"].items()},
    lab_items=dict(raw["lab_items"]),
    poe_orders=dict(raw["poe_cardiac_orders"]),
    reacquisition_gap_minutes=int(raw["reacquisition_gap_minutes"]),
    lab_naive_lookback_hours=int(raw["lab_naive_lookback_hours"]),
    prior_troponin_lookback_hours=int(raw["prior_troponin_lookback_hours"]),
    dropped_cells=tuple(
      {str(k): str(v) for k, v in cell.items()} for cell in (raw.get("dropped_cells") or [])
    ),
  )


def assert_frozen_config(path: Path) -> NextStepConfig:
  """Refuse a locked-test run unless the config is the frozen one."""
  config = load_config(path)
  if config.status != STATUS_FROZEN:
    raise RuntimeError(
      f"Outcome config status is {config.status}; the locked-test run needs {STATUS_FROZEN}."
    )
  if config.config_hash != FROZEN_CONFIG_HASH:
    raise RuntimeError(
      "The next-step outcome config no longer matches the freeze "
      f"{FROZEN_CONFIG_HASH} (current {config.config_hash}). Refusing to run on the locked test."
    )
  return config


def window_flag(
  index: pl.DataFrame,
  events: pl.DataFrame,
  start: timedelta,
  end: timedelta,
  name: str = "action",
) -> pl.DataFrame:
  """Whether any event falls in ``(t0 + start, t0 + end]`` for each index ECG.

  ``index`` needs ``study_id``, ``subject_id``, ``ecg_datetime``; ``events`` needs
  ``subject_id`` and ``event_time``. The lower bound is exclusive, so an event at
  exactly ``t0 + start`` does not count. Missing event times never count.
  """
  if end <= start:
    raise ValueError("Window end must be after its start.")
  idx = index.select(["study_id", "subject_id", "ecg_datetime"]).with_columns(
    pl.col("ecg_datetime").cast(pl.Datetime("us")),
  ).with_columns(
    (pl.col("ecg_datetime") + start).alias("_lo"),
    (pl.col("ecg_datetime") + end).alias("_hi"),
  ).sort("_lo")
  ev = events.select(
    pl.col("subject_id"),
    pl.col("event_time").cast(pl.Datetime("us")).alias("_ev"),
  ).drop_nulls().sort("_ev")
  joined = idx.join_asof(
    ev, left_on="_lo", right_on="_ev", by="subject_id", strategy="forward", allow_exact_matches=False,
    check_sortedness=False,
  )
  return joined.select(
    pl.col("study_id"),
    (pl.col("_ev").is_not_null() & (pl.col("_ev") <= pl.col("_hi"))).alias(name),
  )


def hours(value: float) -> timedelta:
  return timedelta(hours=value)


def eligible_setting(setting: pl.Expr, settings: tuple[str, ...]) -> pl.Expr:
  """ICU and ambiguous settings are never eligible; they are counted separately."""
  return setting.is_in(list(settings))


def exclusion_counts(index: pl.DataFrame) -> dict[str, int]:
  """ICU and ambiguous index ECGs that every outcome excludes."""
  return {
    "n_icu_excluded": int(index.filter(pl.col("setting") == SETTING_ICU).height),
    "n_ambiguous_excluded": int(index.filter(pl.col("setting") == SETTING_AMBIGUOUS).height),
    "n_in_scope": int(index.filter(pl.col("setting").is_in(list(SETTINGS_IN_SCOPE))).height),
  }


def lab_naive_flags(labs: pl.DataFrame, index: pl.DataFrame) -> pl.DataFrame:
  """Lab-naive at t0: none of the four Cacute labs qualifies in the 24 hours before the ECG.

  This is the D4 availability rule: ``charttime`` in ``(ecg - 24h, ecg]``, ``storetime``
  at or before the ECG, expected unit, plausible range.
  """
  wide = latest_plausible_labs(labs, index)
  concepts = [col for col in wide.columns if col != "study_id"]
  return wide.select(
    pl.col("study_id"),
    pl.all_horizontal([pl.col(col).is_null() for col in concepts]).alias("lab_naive"),
  )


def death_time_expr() -> pl.Expr:
  """Death time. A date-only death counts at the end of its day, the conservative direction.

  A hospital death has a timestamp. A date-only death is placed at 23:59:59.999999 so it is
  treated as inside a window only when the whole day fits.
  """
  end_of_day = pl.col("dod_date").cast(pl.Datetime("us")) + pl.duration(days=1) - pl.duration(microseconds=1)
  return pl.coalesce([pl.col("first_hosp_death_dt").cast(pl.Datetime("us")), end_of_day])


def death_in_window_expr(window_hours: int) -> pl.Expr:
  """Died in ``(t0, t0 + window]``."""
  death = death_time_expr()
  t0 = pl.col("ecg_datetime").cast(pl.Datetime("us"))
  return (death.is_not_null() & (death > t0) & (death <= t0 + pl.duration(hours=window_hours))).fill_null(False)


def died_without_action(action: pl.Expr, died: pl.Expr) -> pl.Expr:
  """Bookkeeping column for patients who died inside the window with no observed action."""
  return (~action) & died


def build_outcome_frame(
  index: pl.DataFrame,
  config: NextStepConfig,
  *,
  labs_basic: pl.DataFrame,
  labs_troponin: pl.DataFrame,
  ecg_events: pl.DataFrame,
  admissions: pl.DataFrame,
  ed_arrivals: pl.DataFrame,
  icu_events: pl.DataFrame,
  poe_events: pl.DataFrame,
) -> pl.DataFrame:
  """Eligibility and action flag for every outcome, plus lab-naive and prior-troponin strata.

  ``index`` needs ``study_id``, ``subject_id``, ``ecg_datetime``, ``setting``, ``hadm_id``.
  Event frames use ``subject_id`` and ``event_time`` (labs also need the timing, unit, and
  value columns the D4 rule reads). ``admissions`` needs ``subject_id``, ``hadm_id``,
  ``admittime``, ``dischtime``. No score or mortality column is read.
  """
  base = index.select([
    col for col in ("study_id", "subject_id", "ecg_datetime", "setting", "hadm_id", "split") if col in index.columns
  ]).with_columns(pl.col("ecg_datetime").cast(pl.Datetime("us")))
  frame = base
  basic_events = labs_basic.select(["subject_id", pl.col("charttime").alias("event_time")])
  trop_events = labs_troponin.select(["subject_id", pl.col("charttime").alias("event_time")])

  naive = lab_naive_flags(labs_basic, base.select(["study_id", "subject_id", "ecg_datetime"]))
  frame = frame.join(naive, on="study_id", how="left")
  prior_trop = window_flag(
    base, trop_events, -hours(config.prior_troponin_lookback_hours), timedelta(0), "troponin_in_prior_window",
  )
  frame = frame.join(prior_trop, on="study_id", how="left")

  admit_events = admissions.select(["subject_id", pl.col("admittime").alias("event_time")])
  admission_lookup = admissions.select(["hadm_id", "admittime"]).unique(subset=["hadm_id"])
  # Admission covering t0 or starting within 24 hours (N6 eligibility).
  cover = (
    base.select(["study_id", "subject_id", "ecg_datetime"])
    .with_columns((pl.col("ecg_datetime") + hours(24)).alias("_hi"))
    .sort("_hi")
    .join_asof(
      admissions.select(["subject_id", "hadm_id", "admittime", "dischtime"]).sort("admittime"),
      left_on="_hi", right_on="admittime", by="subject_id", strategy="backward",
      check_sortedness=False,
    )
    .select(
      pl.col("study_id"),
      pl.col("hadm_id").alias("cover_hadm_id"),
      (
        pl.col("hadm_id").is_not_null()
        & (pl.col("dischtime").is_null() | (pl.col("dischtime") >= pl.col("ecg_datetime")))
      ).alias("admission_covering_or_soon"),
    )
  )
  frame = frame.join(cover, on="study_id", how="left")

  for spec in config.outcomes:
    window = hours(spec.window_hours)
    gap = timedelta(minutes=spec.min_gap_minutes)
    elig = eligible_setting(pl.col("setting"), spec.settings)
    if spec.source == "labs_basic":
      flags = window_flag(base, basic_events, gap, window, "_act")
    elif spec.source == "labs_troponin":
      flags = window_flag(base, trop_events, gap, window, "_act")
    elif spec.source == "record_list":
      flags = window_flag(base, ecg_events, gap, window, "_act")
    elif spec.source == "icu_intime":
      flags = window_flag(base, icu_events, gap, window, "_act")
    elif spec.source == "ed_arrival_or_admission":
      flags = window_flag(base, pl.concat([ed_arrivals, admit_events]), gap, window, "_act")
    elif spec.source == "poe_cardiac":
      flags = window_flag(base, poe_events, gap, window, "_act")
    elif spec.source == "ed_interval_admission":
      flags = base.join(admission_lookup, left_on="hadm_id", right_on="hadm_id", how="left").select(
        pl.col("study_id"),
        (
          pl.col("hadm_id").is_not_null()
          & pl.col("admittime").is_not_null()
          & (pl.col("admittime") <= pl.col("ecg_datetime") + window)
        ).alias("_act"),
      )
    else:
      raise ValueError(f"Unknown outcome source {spec.source}.")
    frame = frame.join(flags, on="study_id", how="left").with_columns(
      elig.alias(f"elig_{spec.outcome_id}"),
    )
    if spec.eligibility == "admission_covering_or_starting_within_24h":
      frame = frame.with_columns(
        (pl.col(f"elig_{spec.outcome_id}") & pl.col("admission_covering_or_soon").fill_null(False)).alias(
          f"elig_{spec.outcome_id}"
        )
      )
    frame = frame.with_columns(
      pl.when(pl.col(f"elig_{spec.outcome_id}")).then(pl.col("_act").fill_null(False)).otherwise(None)
      .alias(f"act_{spec.outcome_id}"),
    ).drop("_act")
  return frame


def _rate_row(
  outcome_id: str,
  population: str,
  setting: str,
  sub: pl.DataFrame,
  action_col: str,
  primary: bool,
) -> dict[str, Any]:
  n = sub.height
  events = int(sub[action_col].sum()) if n else 0
  return {
    "outcome_id": outcome_id,
    "population": population,
    "setting": setting,
    "is_primary": primary,
    "eligible_n": n,
    "n_action": events,
    "prevalence": (events / n) if n else None,
  }


def feasibility_table(frame: pl.DataFrame, config: NextStepConfig) -> pl.DataFrame:
  """Eligible n and action prevalence per outcome and setting.

  N1 is reported for the lab-naive primary population and for all eligible ECGs.
  N2 is reported overall and stratified by a troponin in the prior window. No outcome
  from mortality and no score is used.
  """
  rows: list[dict[str, Any]] = []
  for spec in config.outcomes:
    elig = frame.filter(pl.col(f"elig_{spec.outcome_id}"))
    action = f"act_{spec.outcome_id}"
    for setting in spec.settings:
      sub = elig.filter(pl.col("setting") == setting)
      if spec.outcome_id == "N1":
        rows.append(_rate_row("N1", "lab_naive", setting, sub.filter(pl.col("lab_naive")), action, True))
        rows.append(_rate_row("N1", "all_eligible", setting, sub, action, False))
      elif spec.outcome_id == "N2":
        rows.append(_rate_row("N2", "all_eligible", setting, sub, action, True))
        rows.append(_rate_row(
          "N2", "troponin_in_prior_window", setting, sub.filter(pl.col("troponin_in_prior_window")), action, False,
        ))
        rows.append(_rate_row(
          "N2", "no_prior_troponin", setting, sub.filter(~pl.col("troponin_in_prior_window")), action, False,
        ))
      else:
        rows.append(_rate_row(spec.outcome_id, "all_eligible", setting, sub, action, True))
  return pl.DataFrame(rows)


def go_no_go(
  table: pl.DataFrame,
  gates: dict[str, float],
  nonadmitted_ed_n: int,
  poe_coverage: float | None,
) -> pl.DataFrame:
  """Decide each primary outcome by setting cell on development counts only.

  A cell is dropped when eligible n is below the minimum, when prevalence is below the lower
  or above the upper bound, when N4 cannot identify non-admitted ED visits, or when ``poe``
  covers too little of admitted care (N6).
  """
  decisions: list[str] = []
  reasons: list[str] = []
  for row in table.iter_rows(named=True):
    if not row["is_primary"]:
      decisions.append("info")
      reasons.append("")
      continue
    reason = ""
    n = int(row["eligible_n"])
    prevalence = row["prevalence"]
    if n < gates["min_eligible_n"]:
      reason = f"eligible_n_below_{int(gates['min_eligible_n'])}"
    elif prevalence is None or prevalence < gates["min_prevalence"]:
      reason = f"prevalence_below_{gates['min_prevalence']:.0%}"
    elif prevalence > gates["max_prevalence"]:
      reason = f"prevalence_above_{gates['max_prevalence']:.0%}"
    elif row["outcome_id"] == "N4" and nonadmitted_ed_n < gates["min_nonadmitted_ed_n"]:
      reason = "non_admitted_ed_visits_not_identifiable"
    elif row["outcome_id"] == "N6" and (poe_coverage is None or poe_coverage < gates["min_poe_coverage_of_admitted"]):
      reason = f"poe_coverage_below_{gates['min_poe_coverage_of_admitted']:.0%}"
    decisions.append("drop" if reason else "keep")
    reasons.append(reason)
  return table.with_columns(pl.Series("decision", decisions), pl.Series("reason", reasons))


def _fmt(value: Any, digits: int = 3) -> str:
  if value is None:
    return "n/a"
  if isinstance(value, float):
    return f"{value:.{digits}f}"
  return str(value)


def _md_table(frame: pl.DataFrame, columns: list[str]) -> str:
  header = "| " + " | ".join(columns) + " |"
  rule = "| " + " | ".join("---" for _ in columns) + " |"
  body = [
    "| " + " | ".join(_fmt(row[col]) for col in columns) + " |"
    for row in frame.select(columns).iter_rows(named=True)
  ]
  return "\n".join([header, rule, *body])


def render_feasibility_markdown(
  decisions: pl.DataFrame,
  coverage: dict[str, Any],
  audit: dict[str, Any],
) -> str:
  """Development-only feasibility report. It carries no mortality and no A/B score."""
  cols = ["outcome_id", "population", "setting", "eligible_n", "n_action", "prevalence", "decision", "reason"]
  gates = audit["gates"]
  lines = [
    "# A6 next-step evaluation: development feasibility",
    "",
    "Analysis status: `development_only_feasibility`. Proxy clinical decision D12 (governance D11): "
    "outcome set defined by the study team as a proxy clinical decision; clinical review post hoc.",
    f"Config: `{audit['config_version']}`, status `{audit['config_status']}`, sha256 `{audit['config_hash']}`. "
    f"Source SHA: `{audit['source_sha']}`.",
    "",
    "This report uses the development split only. It joins no 30-day mortality and no C0+A or C0+B score, "
    "so the go/no-go decisions below cannot depend on either. The locked-test run (a separate PR) "
    "starts from the frozen config.",
    "",
    f"Development ECGs in the population (paired-valid, non-ambiguous): {audit['n_population']:,}. "
    f"In scope: {audit['n_in_scope']:,}. ICU index ECGs excluded and only counted: {audit['n_icu_excluded']:,}. "
    f"Ambiguous-setting index ECGs excluded (D5): {audit['n_ambiguous_excluded']:,}.",
    "",
    "Actions are downstream process outcomes, never predictors. \"No action observed\" is not "
    "\"no action taken\": care outside BIDMC, transfers out, goals-of-care decisions, and death before the "
    "window closed all produce an unobserved action. Without MIMIC-IV-ED, ED orders that do not become "
    "labs, ECGs, or admissions are not observed, and `poe` covers admitted care only.",
    "",
    "## Go/no-go gates (fixed in the config before any count was seen)",
    "",
    f"Drop a cell if eligible n < {int(gates['min_eligible_n'])} or prevalence < {gates['min_prevalence']:.0%} "
    f"or > {gates['max_prevalence']:.0%}. Drop N4 if fewer than {int(gates['min_nonadmitted_ed_n'])} "
    f"non-admitted ED index ECGs can be identified. Drop N6 if `poe` covers under "
    f"{gates['min_poe_coverage_of_admitted']:.0%} of admitted index encounters.",
    "",
    "## Source coverage checks",
    "",
    f"- Non-admitted ED visits are distinguishable: {coverage['n_ed_nonadmitted']:,} of "
    f"{coverage['n_ed']:,} ED index ECGs have an ED interval with no linked admission.",
    f"- `poe` coverage of admitted index encounters (N6 eligible): {_fmt(coverage['poe_coverage'])} "
    f"({coverage['n_poe_covered']:,} of {coverage['n_admitted_index']:,}).",
    f"- Basic labs with a null `hadm_id` exist for other/unlinked patients: "
    f"{coverage['n_other_basic_null_hadm']:,} of {coverage['n_other_in_scope']:,} other/unlinked index ECGs have one "
    f"in the 24-hour window; any troponin with null `hadm_id` in 12 hours: {coverage['n_other_trop_null_hadm']:,}.",
    f"- Troponin assay use in the N2 window: patients with a troponin T draw {coverage['trop_t_patients']:,}, "
    f"with a troponin I draw {coverage['trop_i_patients']:,}, with both {coverage['trop_both_patients']:,}.",
    "",
    "## Eligible n and action prevalence by outcome and setting",
    "",
    "N1 primary rows are the lab-naive population (no qualifying basic lab in the 24 hours before the ECG); "
    "`all_eligible` is information only. N2 rows stratified by a troponin in the prior 6 hours are information only.",
    "",
    _md_table(decisions, cols),
    "",
    "## Freeze",
    "",
    *(audit["freeze_lines"]),
    "",
  ]
  return "\n".join(lines)


def prepare_admissions(admissions: pl.DataFrame) -> pl.DataFrame:
  """Admission start and end under the column names the window logic expects."""
  return admissions.select(
    pl.col("subject_id"),
    pl.col("hadm_id"),
    pl.col("admit_dt").alias("admittime"),
    pl.col("disch_dt").alias("dischtime"),
  )


def build_lab_cache(
  labevents_path: Path,
  subject_ids: list[int],
  itemids: list[int],
  out_path: Path,
) -> pl.DataFrame:
  """Extract the listed lab items for cohort subjects once. The cache is gitignored."""
  labs = (
    pl.scan_csv(
      labevents_path,
      schema_overrides={
        "subject_id": pl.Int64,
        "hadm_id": pl.Int64,
        "itemid": pl.Int64,
        "charttime": pl.String,
        "storetime": pl.String,
        "valuenum": pl.Float64,
        "valueuom": pl.String,
      },
    )
    .filter(pl.col("itemid").is_in(itemids) & pl.col("subject_id").is_in(subject_ids))
    .select(["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"])
    .with_columns(
      pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
      pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    )
    .collect()
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  labs.write_parquet(out_path)
  return labs


def cardiac_order_expr(config: NextStepConfig) -> pl.Expr:
  """Rows of ``poe`` that are New cardiac workup orders under the frozen subtype list."""
  expr = pl.lit(False)
  for order_type, subtypes in config.poe_orders["order_types"].items():
    expr = expr | ((pl.col("order_type") == order_type) & pl.col("order_subtype").is_in(list(subtypes)))
  return expr & (pl.col("transaction_type") == config.poe_orders["transaction_type"])


def build_poe_cache(
  poe_path: Path,
  subject_ids: list[int],
  config: NextStepConfig,
  orders_out: Path,
  hadm_out: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
  """Cardiac orders and the set of admissions that have any ``poe`` row, for cohort subjects."""
  poe = pl.scan_csv(
    poe_path,
    schema_overrides={
      "subject_id": pl.Int64,
      "hadm_id": pl.Int64,
      "ordertime": pl.String,
      "order_type": pl.String,
      "order_subtype": pl.String,
      "transaction_type": pl.String,
    },
  ).select(["subject_id", "hadm_id", "ordertime", "order_type", "order_subtype", "transaction_type"]).filter(
    pl.col("subject_id").is_in(subject_ids)
  )
  orders = poe.filter(cardiac_order_expr(config)).with_columns(
    pl.col("ordertime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
  ).select(["subject_id", "hadm_id", "ordertime", "order_type", "order_subtype"]).collect()
  hadm = poe.select("hadm_id").drop_nulls().unique().collect()
  orders_out.parent.mkdir(parents=True, exist_ok=True)
  orders.write_parquet(orders_out)
  hadm.write_parquet(hadm_out)
  return orders, hadm


def _load_cached(path: Path) -> pl.DataFrame:
  return pl.read_parquet(path)


@dataclass(frozen=True)
class NextStepSources:
  """Event frames for one set of subjects. Raw events only: no scores, no mortality."""

  labs_basic: pl.DataFrame
  labs_troponin: pl.DataFrame
  ecg_events: pl.DataFrame
  admissions: pl.DataFrame
  ed_arrivals: pl.DataFrame
  icu_events: pl.DataFrame
  poe_events: pl.DataFrame
  poe_hadm: set[int]

  def frame_kwargs(self) -> dict[str, pl.DataFrame]:
    return {
      "labs_basic": self.labs_basic,
      "labs_troponin": self.labs_troponin,
      "ecg_events": self.ecg_events,
      "admissions": self.admissions,
      "ed_arrivals": self.ed_arrivals,
      "icu_events": self.icu_events,
      "poe_events": self.poe_events,
    }


def ensure_caches(
  all_subjects: list[int],
  config: NextStepConfig,
  *,
  labevents_path: Path,
  poe_path: Path,
  cache_dir: Path,
  rebuild: bool = False,
) -> None:
  """Build the gitignored lab and poe caches for every cohort subject once."""
  labs_cache = cache_dir / "labevents_cohort_items.parquet"
  poe_orders_cache = cache_dir / "poe_cardiac_orders.parquet"
  poe_hadm_cache = cache_dir / "poe_hadm_ids.parquet"
  if rebuild or not labs_cache.exists():
    build_lab_cache(labevents_path, all_subjects, [*config.basic_itemids, *config.troponin_itemids], labs_cache)
  if rebuild or not poe_orders_cache.exists() or not poe_hadm_cache.exists():
    build_poe_cache(poe_path, all_subjects, config, poe_orders_cache, poe_hadm_cache)


def load_sources(
  subject_ids: list[int],
  config: NextStepConfig,
  *,
  cache_dir: Path,
  record_list_path: Path,
  transfers_path: Path,
  admissions_df: pl.DataFrame,
  icustays_path: Path,
) -> NextStepSources:
  """Event sources restricted to ``subject_ids`` (one split, so patients never mix splits)."""
  subjects = pl.DataFrame({"subject_id": subject_ids}, schema={"subject_id": pl.Int64})
  labs_all = _load_cached(cache_dir / "labevents_cohort_items.parquet").join(subjects, on="subject_id", how="semi")
  poe_events = _load_cached(cache_dir / "poe_cardiac_orders.parquet").join(subjects, on="subject_id", how="semi").select(
    "subject_id", pl.col("ordertime").alias("event_time"),
  )
  record_list = pl.read_csv(
    record_list_path,
    columns=["subject_id", "study_id", "ecg_time"],
    schema_overrides={"subject_id": pl.Int64, "study_id": pl.Int64, "ecg_time": pl.String},
  ).join(subjects, on="subject_id", how="semi").select(
    "subject_id", pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("event_time"),
  )
  transfers = pl.read_csv(
    transfers_path,
    schema_overrides={"subject_id": pl.Int64, "hadm_id": pl.Int64, "eventtype": pl.String, "intime": pl.String},
  ).join(subjects, on="subject_id", how="semi")
  ed_arrivals = transfers.filter(pl.col("eventtype") == "ED").select(
    "subject_id", pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("event_time"),
  )
  icu = pl.read_csv(
    icustays_path,
    schema_overrides={"subject_id": pl.Int64, "intime": pl.String},
  ).join(subjects, on="subject_id", how="semi").select(
    "subject_id", pl.col("intime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("event_time"),
  )
  return NextStepSources(
    labs_basic=labs_all.filter(pl.col("itemid").is_in(config.basic_itemids)),
    labs_troponin=labs_all.filter(pl.col("itemid").is_in(config.troponin_itemids)),
    ecg_events=record_list,
    admissions=prepare_admissions(admissions_df).join(subjects, on="subject_id", how="semi"),
    ed_arrivals=ed_arrivals,
    icu_events=icu,
    poe_events=poe_events,
    poe_hadm=set(_load_cached(cache_dir / "poe_hadm_ids.parquet")["hadm_id"].to_list()),
  )


def _git_sha() -> str:
  import subprocess

  try:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return "unknown"


def run_next_step_feasibility(
  *,
  cohort_path: Path,
  setting_path: Path,
  pair_validity_path: Path,
  config_path: Path,
  labevents_path: Path,
  poe_path: Path,
  record_list_path: Path,
  transfers_path: Path,
  admissions_df: pl.DataFrame,
  icustays_path: Path,
  cache_dir: Path,
  output_dir: Path,
  report_path: Path,
  split: str = "dev",
  rebuild_cache: bool = False,
) -> dict[str, Any]:
  """Development-only feasibility. Refuses any other split and joins no scores or mortality."""
  if split != "dev":
    raise ValueError(
      "Next-step feasibility uses the development split only so the outcome set cannot be "
      "tuned on validation or test."
    )
  config = load_config(config_path)
  cohort_all = pl.read_parquet(cohort_path).select(["study_id", "subject_id", "ecg_datetime", "split"])
  splits_per_subject = cohort_all.group_by("subject_id").agg(pl.col("split").n_unique().alias("k"))
  if int(splits_per_subject.filter(pl.col("k") > 1).height) > 0:
    raise ValueError("Patients appear in more than one split; refusing a development-only audit.")
  dev = cohort_all.filter(pl.col("split") == "dev")
  setting = pl.read_parquet(setting_path).select(["study_id", "setting", "hadm_id"])
  pair = pl.read_parquet(pair_validity_path).select(["study_id", "representation_pair_valid"])
  index = (
    dev.join(setting, on="study_id", how="left")
    .join(pair, on="study_id", how="left")
    .filter(pl.col("representation_pair_valid") == True)  # noqa: E712
    .drop("representation_pair_valid")
  )
  if index["setting"].null_count():
    raise ValueError("Some development index ECGs have no setting assignment.")
  exclusions = exclusion_counts(index)
  index = index.filter(pl.col("setting") != SETTING_AMBIGUOUS)
  subject_ids = [int(v) for v in index["subject_id"].unique().to_list()]

  # Caches cover all cohort subjects so the locked-test run can reuse them.
  ensure_caches(
    [int(v) for v in cohort_all["subject_id"].unique().to_list()], config,
    labevents_path=labevents_path, poe_path=poe_path, cache_dir=cache_dir, rebuild=rebuild_cache,
  )
  sources = load_sources(
    subject_ids, config,
    cache_dir=cache_dir,
    record_list_path=record_list_path,
    transfers_path=transfers_path,
    admissions_df=admissions_df,
    icustays_path=icustays_path,
  )
  labs_basic = sources.labs_basic
  labs_trop = sources.labs_troponin
  poe_hadm = sources.poe_hadm
  frame = build_outcome_frame(index, config, **sources.frame_kwargs())
  table = feasibility_table(frame, config)

  ed = frame.filter(pl.col("setting") == "ED")
  n_ed_nonadmitted = int(ed.filter(pl.col("hadm_id").is_null()).height)
  n6 = frame.filter(pl.col("elig_N6"))
  n_admitted_index = n6.height
  n_poe_covered = int(n6.filter(pl.col("cover_hadm_id").is_in(list(poe_hadm))).height)
  poe_coverage = (n_poe_covered / n_admitted_index) if n_admitted_index else None
  other = frame.filter(pl.col("setting") == "other_or_unlinked")
  other_idx = other.select(["study_id", "subject_id", "ecg_datetime"])
  basic_null = window_flag(
    other_idx,
    labs_basic.filter(pl.col("hadm_id").is_null()).select("subject_id", pl.col("charttime").alias("event_time")),
    timedelta(0), hours(24), "f",
  )
  trop_null = window_flag(
    other_idx,
    labs_trop.filter(pl.col("hadm_id").is_null()).select("subject_id", pl.col("charttime").alias("event_time")),
    timedelta(0), hours(12), "f",
  )
  scope = frame.filter(pl.col("setting").is_in(list(SETTINGS_IN_SCOPE))).select(["study_id", "subject_id", "ecg_datetime"])

  def _assay_patients(itemids: list[int]) -> pl.DataFrame:
    return window_flag(
      scope,
      labs_trop.filter(pl.col("itemid").is_in(itemids)).select("subject_id", pl.col("charttime").alias("event_time")),
      timedelta(0), hours(12), "f",
    )

  t_flags = _assay_patients([int(v) for v in config.lab_items["troponin_t"]])
  i_flags = _assay_patients([int(v) for v in config.lab_items["troponin_i"]])
  both = t_flags.join(i_flags, on="study_id", suffix="_i")
  coverage = {
    "n_ed": ed.height,
    "n_ed_nonadmitted": n_ed_nonadmitted,
    "n_admitted_index": n_admitted_index,
    "n_poe_covered": n_poe_covered,
    "poe_coverage": poe_coverage,
    "n_other_in_scope": other.height,
    "n_other_basic_null_hadm": int(basic_null["f"].sum()),
    "n_other_trop_null_hadm": int(trop_null["f"].sum()),
    "trop_t_patients": int(t_flags["f"].sum()),
    "trop_i_patients": int(i_flags["f"].sum()),
    "trop_both_patients": int((both["f"] & both["f_i"]).sum()),
  }
  decisions = go_no_go(table, config.gates, n_ed_nonadmitted, poe_coverage)
  dropped = decisions.filter(pl.col("decision") == "drop")
  if dropped.height:
    freeze = [
      f"- Drop {row['outcome_id']} / {row['setting']} ({row['population']}): {row['reason']}."
      for row in dropped.iter_rows(named=True)
    ]
  else:
    freeze = ["- No cell is dropped."]
  if config.status == STATUS_FROZEN:
    freeze += [
      f"- Config frozen: status `{config.status}`, sha256 `{config.config_hash}`, pinned as `FROZEN_CONFIG_HASH`. "
      "The locked-test command (PR A6b) refuses any other hash.",
    ]
  else:
    freeze += [
      "- Record each drop in `dropped_cells` of `configs/next_step_outcomes_v1.yaml`, set "
      "`status: frozen_after_feasibility`, then pin the resulting sha256 in `FROZEN_CONFIG_HASH`.",
    ]
  audit: dict[str, Any] = {
    "analysis_status": "development_only_feasibility",
    "decision_ids": "D8, D11, D12",
    "config_version": config.version,
    "config_status": config.status,
    "config_hash": config.config_hash,
    "source_sha": _git_sha(),
    "split_policy": "development_only",
    "joined_to_mortality": False,
    "joined_to_model_scores": False,
    "n_population": exclusions["n_icu_excluded"] + exclusions["n_in_scope"],
    "gates": config.gates,
    "freeze_lines": freeze,
    "coverage": coverage,
    **exclusions,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "next_step_feasibility_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8",
  )
  decisions.write_csv(output_dir / "next_step_feasibility.csv")
  tables = report_path.parent / "tables"
  tables.mkdir(parents=True, exist_ok=True)
  decisions.write_csv(tables / "next_step_feasibility.csv")
  report = render_feasibility_markdown(decisions, coverage, audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  return audit
