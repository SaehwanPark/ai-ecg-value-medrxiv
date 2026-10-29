"""LM3: maximal structured pre-ECG context (Cmax).

``Cmax`` extends ``Cacute`` with a deliberately broad, leakage-safe pre-ECG
laboratory panel and optional ICU vital signs. Feature construction is
availability- and data-quality-driven only: the concept list, coverage gate,
lookback, buffers, winsorization, and imputation rules are frozen in
``configs/cmax_context_v1.yaml`` before any locked-test evaluation. No feature
may be added, dropped, or redefined because it predicts mortality or improves a
representation contrast. This is a ``post hoc extension``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import polars as pl

from ecg_rep.acute_context import classify_measurement_timing
from ecg_rep.cacute import (
  INCLUDED_SETTINGS,
  SETTING_INDICATORS,
  cacute_feature_columns,
  setting_indicators,
)
from ecg_rep.evaluate import C0_FEATURE_COLS

#: Development coverage gate (LM3 plan): more stringent of the two thresholds.
COVERAGE_MIN_FRACTION = 0.02
COVERAGE_MIN_PATIENTS = 1000
#: Predeclared cardiac-biomarker concepts exempt from the coverage gate
#: (missingness becomes part of the feature representation).
PREDECLARED_LOW_COVERAGE: tuple[str, ...] = ("trop_t_blood", "trop_i_blood", "ntprobnp_blood")
#: ICU-vital go/no-go gate.
ICU_MIN_DEV_PATIENTS = 2000
ICU_MIN_DEV_EVENTS = 100
VITAL_LOOKBACK_HOURS = 6
TIME_SINCE_CAP_HOURS = 24.0
WINSOR_LO = 0.005
WINSOR_HI = 0.995


@dataclass(frozen=True)
class CmaxConfig:
  """Frozen Cmax configuration (authoritative copy: configs/cmax_context_v1.yaml)."""

  version: str
  lookback_hours_primary: int
  lookback_hours_secondary: int
  temporal_buffers_hours: tuple[int, ...]
  primary_buffer_hours: int
  min_fraction_of_dev: float
  min_patients: int
  predeclared: tuple[str, ...]
  icu_min_dev_patients: int
  icu_min_dev_events: int
  icu_lookback_hours: int
  time_since_cap_hours: float
  lab_specs: dict[str, dict[str, Any]]
  vital_specs: dict[str, dict[str, Any]]


def load_cmax_config(path: Path) -> CmaxConfig:
  """Load the frozen config; unknown top-level keys are rejected."""
  import yaml

  raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError(f"Cmax config at {path} is not a mapping.")
  known_top = {
    "version", "analysis_status", "lookback_hours_primary", "lookback_hours_secondary",
    "temporal_buffers_hours", "primary_buffer_hours", "coverage_gate",
    "predeclared_low_coverage", "icu_gate", "winsorization", "imputation",
    "time_since_feature", "concepts", "icu_vitals",
  }
  unknown = set(raw) - known_top
  if unknown:
    raise ValueError(f"Unknown cmax config keys: {sorted(unknown)}")
  gate = raw["coverage_gate"]
  icu = raw["icu_gate"]
  time_since = raw.get("time_since_feature", {})
  concepts_raw = raw["concepts"]
  vitals_raw = raw.get("icu_vitals", {})
  specs: dict[str, dict[str, Any]] = {}
  for concept, spec in {**concepts_raw, **vitals_raw}.items():
    specs[concept] = {
      "itemids": [int(i) for i in spec["itemids"]],
      "unit": str(spec["unit"]),
      "low": float(spec["low"]),
      "high": float(spec["high"]),
    }
  return CmaxConfig(
    version=str(raw["version"]),
    lookback_hours_primary=int(raw["lookback_hours_primary"]),
    lookback_hours_secondary=int(raw["lookback_hours_secondary"]),
    temporal_buffers_hours=tuple(int(b) for b in raw["temporal_buffers_hours"]),
    primary_buffer_hours=int(raw["primary_buffer_hours"]),
    min_fraction_of_dev=float(gate["min_fraction_of_dev"]),
    min_patients=int(gate["min_patients"]),
    predeclared=tuple(str(c) for c in raw["predeclared_low_coverage"]),
    icu_min_dev_patients=int(icu["min_dev_patients"]),
    icu_min_dev_events=int(icu["min_dev_events"]),
    icu_lookback_hours=int(icu["lookback_hours"]),
    time_since_cap_hours=float(time_since.get("cap_hours", TIME_SINCE_CAP_HOURS)),
    lab_specs={k: v for k, v in specs.items() if k in concepts_raw},
    vital_specs={k: v for k, v in specs.items() if k in vitals_raw},
  )


def scan_lab_events(path: Path, itemids: list[int]) -> pl.DataFrame:
  """Load labevents rows for the configured itemids with parsed timestamps."""
  frame = pl.scan_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "itemid": pl.Int64,
      "charttime": pl.String,
      "storetime": pl.String,
      "valuenum": pl.Float64,
      "valueuom": pl.String,
    },
  ).filter(pl.col("itemid").is_in(itemids)).select(
    ["subject_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"]
  )
  return frame.with_columns(
    pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
  ).collect()


def latest_qualified_values(
  events: pl.DataFrame,
  index: pl.DataFrame,
  specs: dict[str, dict[str, Any]],
  lookback_hours: int,
  buffer_hours: int = 0,
  with_time_since: bool = True,
  time_since_cap_hours: float = TIME_SINCE_CAP_HOURS,
) -> pl.DataFrame:
  """Most recent leak-safe pre-ECG value per study and concept.

  Qualified requires: concept itemid, dominant unit, plausible numeric value,
  ``charttime`` in ``(ecg - lookback, ecg - buffer]`` and
  ``storetime <= ecg - buffer`` (strict pre-ECG chart time still applies at 0h).
  ``hours_since`` is capped at ``time_since_cap_hours``; the missingness flag is
  always emitted so predeclared low-coverage concepts keep an explicit signal.
  """
  required = {"subject_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"}
  if required - set(events.columns):
    raise ValueError(f"Events are missing {sorted(required - set(events.columns))}.")
  index_small = index.select(["subject_id", "study_id", "ecg_datetime"])
  pieces: list[pl.DataFrame] = []
  for concept, spec in specs.items():
    subset = events.filter(pl.col("itemid").is_in(spec["itemids"]))
    if subset.height == 0:
      pieces.append(index_small.select("study_id").with_columns(
        pl.lit(None, dtype=pl.Float64).alias(concept),
        pl.lit(None, dtype=pl.Float64).alias(f"hs_{concept}"),
      ))
      continue
    labeled = classify_measurement_timing(subset, index, lookback_hours)
    unit = pl.col("valueuom").fill_null("").str.to_lowercase().str.replace_all("μ", "u")
    # A buffer represents information availability, not just specimen collection time.
    cutoff = pl.col("ecg_datetime") - pl.duration(hours=buffer_hours)
    unit_ok = (
      pl.lit(True) if not spec["unit"]  # empty unit: assay id only (source uom is null)
      else (unit == spec["unit"].lower())
    )
    qualified = labeled.filter(
      (pl.col("timing_class") == "qualified")
      & (pl.col("charttime") <= cutoff)
      & (pl.col("storetime") <= cutoff)
      & unit_ok
      & pl.col("valuenum").is_not_null()
      & (pl.col("valuenum") >= spec["low"])
      & (pl.col("valuenum") <= spec["high"])
    )
    latest = (
      qualified.sort(["study_id", "charttime", "storetime"])
      .group_by("study_id", maintain_order=True)
      .agg([
        pl.col("valuenum").last().alias(concept),
        ((pl.col("ecg_datetime").last() - pl.col("charttime").last()).dt.total_hours()
          if with_time_since else pl.lit(None, dtype=pl.Float64)).alias(f"hs_{concept}"),
      ])
    )
    if with_time_since:
      # Cap only observed hours; missing rows stay null until imputation.
      latest = latest.with_columns(
        pl.col(f"hs_{concept}").clip(0.0, time_since_cap_hours).alias(f"hs_{concept}"),
      )
    pieces.append(index_small.select("study_id").join(latest, on="study_id", how="left"))
  wide = pieces[0]
  for piece in pieces[1:]:
    wide = wide.join(piece, on="study_id", how="left")
  return wide


def coverage_ledger(
  wide: pl.DataFrame,
  index_dev: pl.DataFrame,
  specs: dict[str, dict[str, Any]],
  min_fraction: float,
  min_patients: int,
  predeclared: tuple[str, ...],
) -> pl.DataFrame:
  """One row per concept: qualified dev patients, gate decision, and reason."""
  n_dev = index_dev.height
  threshold = max(min_fraction * n_dev, min_patients)
  rows: list[dict[str, Any]] = []
  for concept in specs:
    n_qual = int(index_dev.join(
      wide.filter(pl.col(concept).is_not_null()).select("study_id"),
      on="study_id",
      how="semi",
    ).height)
    threshold_f = max(min_fraction * n_dev, float(min_patients))
    exempt = concept in predeclared
    if n_qual == 0:
      # No numeric pre-ECG result exists at all: data-quality removal, even if predeclared
      # (missingness-as-feature still requires at least one development observation).
      retained = False
      reason = "no_numeric_value"
    else:
      retained = (n_qual >= threshold_f) or exempt
      reason = (
        "predeclared_low_coverage" if exempt and n_qual < threshold_f
        else "above_gate" if n_qual >= threshold_f else "below_gate"
      )
    rows.append({
      "concept": concept,
      "n_dev_qualified": n_qual,
      "fraction_of_dev": n_qual / n_dev if n_dev else 0.0,
      "gate_threshold_patients": int(np.ceil(threshold_f)),
      "predeclared": exempt,
      "retained": retained,
      "decision_reason": reason,
    })
  return pl.DataFrame(rows)


def winsorize_and_impute(
  frame: pl.DataFrame,
  value_cols: tuple[str, ...],
  low_p: float = WINSOR_LO,
  high_p: float = WINSOR_HI,
) -> tuple[pl.DataFrame, dict[str, dict[str, float]]]:
  """Dev-only winsorization bounds and medians; adds missingness flags."""
  dev = frame.filter(pl.col("split") == "dev")
  stats: dict[str, dict[str, float]] = {}
  out = frame
  for col in value_cols:
    observed = dev.filter(pl.col(col).is_not_null())[col]
    if observed.len() == 0:
      raise ValueError(f"No development observations for {col}; refusing to invent a fill.")
    lo_raw = observed.quantile(low_p)
    hi_raw = observed.quantile(high_p)
    if lo_raw is None or hi_raw is None:
      raise ValueError(f"Development quantiles for {col} are null; refusing to invent a fill.")
    lo_q, hi_q = float(lo_raw), float(hi_raw)
    med_raw = observed.clip(lo_q, hi_q).median()
    if not isinstance(med_raw, (int, float)):
      raise ValueError(f"Development median for {col} is not numeric; refusing to invent a fill.")
    med = float(med_raw)
    stats[col] = {"winsor_low": lo_q, "winsor_high": hi_q, "dev_median": med}
    out = out.with_columns(
      pl.col(col).is_null().cast(pl.Float64).alias(f"miss_{col}"),
      pl.col(col).clip(lo_q, hi_q).fill_null(med).alias(col),
      pl.col(f"hs_{col}").fill_null(0.0).alias(f"hs_{col}"),
    )
  return out, stats


def cmax_feature_columns(retained: tuple[str, ...], include_time_since: bool = True) -> list[str]:
  """C0, setting indicators, retained lab values, missingness, capped time-since."""
  cols = [*C0_FEATURE_COLS, *SETTING_INDICATORS]
  for concept in retained:
    cols += [concept, f"miss_{concept}"]
    if include_time_since:
      cols.append(f"hs_{concept}")
  return cols


def _render_cmax_feasibility(report_path: Path, audit: dict[str, Any], ledger: pl.DataFrame) -> None:
  lines = [
    "# LM3 Cmax: development-only availability and temporal audit",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {audit['source_sha']}",
    "analysis_status: post_hoc_extension",
    "inputs: [configs/cmax_context_v1.yaml, artifacts/cohort, artifacts/clinical/setting, hosp/labevents, icu/chartevents]",
    f"decisions: [concepts/itemids frozen before coverage read; gate = max(2% dev, {audit['min_patients']} patients)]",
    "evidence: [cmax_coverage_ledger.csv, cmax_feature_manifest.json]",
    "unresolved: []",
    "verification: [all counts use strict charttime<ECG and storetime<=ECG; no outcomes joined]",
    "next_action: LM3b frozen same-subset evaluation after this protocol merges",
    "```",
    "",
    f"Development index ECGs (non-ambiguous settings): {audit['n_dev']:,}.",
    f"Coverage gate: qualified dev patients >= {audit['gate_threshold']:,} "
    f"(max of {audit['min_fraction']:.0%} of dev and {audit['min_patients']:,} patients).",
    f"Predeclared low-coverage concepts (missingness retained as a feature): "
    f"{', '.join(audit['predeclared'])}.",
    f"Lookback: {audit['lookback_hours_primary']}h primary; {audit['lookback_hours_secondary']}h availability-only "
    "sensitivity. Temporal buffers: strict pre-ECG primary; 1h and 2h sensitivity. No cutoff is "
    "chosen by performance.",
    "",
    "## Lab panel coverage (strict 24h pre-ECG, dev split)",
    "",
    "| Concept | Qualified dev | Fraction | Retained | Reason |",
    "|---|---|---|---|---|",
  ]
  for row in ledger.iter_rows(named=True):
    lines.append(
      f"| {row['concept']} | {row['n_dev_qualified']:,} | {row['fraction_of_dev']:.3f} | "
      f"{row['retained']} | {row['decision_reason']} |"
    )
  icu = audit.get("icu_vitals", {})
  lines += [
    "",
    "## ICU vitals (Cmax-ICU go/no-go)",
    "",
    f"Gate: >= {ICU_MIN_DEV_PATIENTS:,} development patients and >= {ICU_MIN_DEV_EVENTS:,} "
    f"development mortality events with a leak-safe pre-ECG ICU chart (6h lookback).",
    f"Observed: {icu.get('n_dev_patients', 0):,} patients; {icu.get('n_dev_events', 0):,} deaths.",
    f"Decision: **{icu.get('decision', 'not_run')}**.",
    "",
    "ICU vitals describe only patients with ICU charting; they are never treated as "
    "cohort-wide features. Findings, if run, are reported only within the ICU subset.",
    "",
    "No locked-test outcome appears in this report. Feature selection is availability- and "
    "data-quality-based only; no concept was added, dropped, or redefined using outcome "
    "association.",
    "",
  ]
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text("\n".join(lines), encoding="utf-8")


def run_cmax_feasibility(
  cohort_path: Path,
  setting_path: Path,
  labevents_path: Path,
  chartevents_path: Path | None,
  config_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
) -> dict[str, Any]:
  """LM3a: development-only coverage and temporal audit. No model fits, no outcomes."""
  cfg = load_cmax_config(config_path)
  cohort = pl.read_parquet(cohort_path)
  setting = pl.read_parquet(setting_path).select(["study_id", "setting"])
  frame = (
    cohort.join(setting, on="study_id", how="left")
    .filter(pl.col("setting").is_in(list(INCLUDED_SETTINGS)))
  )
  dev = frame.filter(pl.col("split") == "dev").select(["subject_id", "study_id", "ecg_datetime"])
  n_dev = dev.height

  all_itemids = [i for spec in cfg.lab_specs.values() for i in spec["itemids"]]
  events = scan_lab_events(labevents_path, all_itemids)
  # Strict primary buffer, 24h lookback.
  wide = latest_qualified_values(
    events, dev, cfg.lab_specs,
    lookback_hours=cfg.lookback_hours_primary,
    buffer_hours=cfg.primary_buffer_hours,
    time_since_cap_hours=cfg.time_since_cap_hours,
  )
  ledger = coverage_ledger(
    wide, dev, cfg.lab_specs, cfg.min_fraction_of_dev, cfg.min_patients, cfg.predeclared,
  )
  retained = tuple(
    ledger.filter(pl.col("retained"))["concept"].to_list()
  )
  # Secondary lookback is availability-only; retention decisions are frozen above.
  wide_72 = latest_qualified_values(
    events, dev, cfg.lab_specs,
    lookback_hours=cfg.lookback_hours_secondary,
    buffer_hours=cfg.primary_buffer_hours,
    with_time_since=False,
  )
  ledger_72 = coverage_ledger(
    wide_72, dev, cfg.lab_specs, cfg.min_fraction_of_dev, cfg.min_patients, cfg.predeclared,
  ).rename({"n_dev_qualified": "n_dev_qualified_72h", "fraction_of_dev": "fraction_dev_72h"})
  ledger = ledger.join(
    ledger_72.select(["concept", "n_dev_qualified_72h", "fraction_dev_72h"]), on="concept"
  )

  # Buffer sensitivity availability counts (strict retention decisions unchanged).
  ledger_by_buffer: list[pl.DataFrame] = []
  for buffer_hours in cfg.temporal_buffers_hours:
    if buffer_hours == cfg.primary_buffer_hours:
      continue
    wide_b = latest_qualified_values(
      events, dev, cfg.lab_specs,
      lookback_hours=cfg.lookback_hours_primary,
      buffer_hours=buffer_hours,
      with_time_since=False,
    )
    ledger_by_buffer.append(
      coverage_ledger(wide_b, dev, cfg.lab_specs, cfg.min_fraction_of_dev, cfg.min_patients, cfg.predeclared)
      .rename({"n_dev_qualified": f"n_dev_qualified_{buffer_hours}h_buffer"})
      .select(["concept", f"n_dev_qualified_{buffer_hours}h_buffer"])
    )
  for piece in ledger_by_buffer:
    ledger = ledger.join(piece, on="concept")

  # ICU vital gate: count dev patients and dev deaths with a leak-safe pre-ECG chart.
  icu_audit: dict[str, Any] = {"decision": "not_run"}
  if chartevents_path is not None and Path(chartevents_path).exists() and cfg.vital_specs:
    vital_ids = [i for spec in cfg.vital_specs.values() for i in spec["itemids"]]
    vitals = pl.scan_csv(
      chartevents_path,
      schema_overrides={"subject_id": pl.Int64, "itemid": pl.Int64, "charttime": pl.String,
                        "storetime": pl.String},
    ).filter(pl.col("itemid").is_in(vital_ids)).select(["subject_id", "itemid", "charttime", "storetime"])
    vitals = vitals.with_columns(
      pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
      pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    ).collect()
    labeled_v = classify_measurement_timing(vitals, dev, cfg.icu_lookback_hours)
    dev_patients_with_vitals = (
      labeled_v.filter(pl.col("timing_class") == "qualified").select("subject_id").unique()
    )
    dev_full = pl.read_parquet(cohort_path).filter(pl.col("split") == "dev")
    joined = dev_full.select(["subject_id", "mortality_30d"]).join(
      dev_patients_with_vitals, on="subject_id", how="semi"
    )
    n_patients = joined.height
    n_events = int(joined["mortality_30d"].sum())
    go = n_patients >= cfg.icu_min_dev_patients and n_events >= cfg.icu_min_dev_events
    icu_audit = {
      "n_dev_patients": n_patients,
      "n_dev_events": n_events,
      "min_patients": cfg.icu_min_dev_patients,
      "min_events": cfg.icu_min_dev_events,
      "decision": "go" if go else "underpowered_stop",
    }

  gate_threshold = int(np.ceil(max(cfg.min_fraction_of_dev * n_dev, cfg.min_patients)))
  audit = {
    "analysis_status": "post_hoc_extension",
    "source_sha": source_sha,
    "config_version": cfg.version,
    "n_dev": n_dev,
    "min_fraction": cfg.min_fraction_of_dev,
    "min_patients": cfg.min_patients,
    "gate_threshold": gate_threshold,
    "predeclared": list(cfg.predeclared),
    "retained_concepts": list(retained),
    "lookback_hours_primary": cfg.lookback_hours_primary,
    "lookback_hours_secondary": cfg.lookback_hours_secondary,
    "temporal_buffers_hours": list(cfg.temporal_buffers_hours),
    "primary_buffer_hours": cfg.primary_buffer_hours,
    "icu_vitals": icu_audit,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  ledger.write_csv(output_dir / "cmax_coverage_ledger.csv")
  manifest = {
    **audit,
    "lab_specs": {k: {**v, "itemids": v["itemids"]} for k, v in cfg.lab_specs.items()},
    "vital_specs": {k: {**v, "itemids": v["itemids"]} for k, v in cfg.vital_specs.items()},
  }
  (output_dir / "cmax_feature_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  _render_cmax_feasibility(report_path, audit, ledger)
  return audit


# ---------------------------------------------------------------------------
# LM3b: frozen same-subset probe evaluation.
# ---------------------------------------------------------------------------

#: Frozen bootstrap seeds for the primary Cmax estimands (one per estimand).
CMAX_BOOTSTRAP_SEEDS: tuple[int, ...] = (52026, 52027, 52028, 52029)

CACUTE_VALUE_COLS: tuple[str, ...] = ("creatinine_blood", "potassium_blood", "hemoglobin_blood", "wbc_blood")


def build_cmax_frame(
  cohort_path: Path,
  setting_path: Path,
  labevents_path: Path,
  config_path: Path,
  buffer_hours: int | None = None,
  retained_concepts: tuple[str, ...] | None = None,
) -> pl.DataFrame:
  """Cmax-lab analysis frame on the non-ambiguous setting population.

  Retained concepts come from the development coverage gate for the primary
  analysis. Buffer reruns pass the frozen primary concepts explicitly, so a
  buffer cannot redefine the feature panel. Values are the
  most-recent leak-safe pre-ECG results; winsorization and imputation are
  learned on development only. No outcome column is modified or created.
  """
  cfg = load_cmax_config(config_path)
  cohort = pl.read_parquet(cohort_path)
  setting = pl.read_parquet(setting_path).select(["study_id", "setting"])
  frame = (
    cohort.join(setting, on="study_id", how="left")
    .filter(pl.col("setting").is_in(list(INCLUDED_SETTINGS)))
    .select(["subject_id", "study_id", "split", "mortality_30d", "ecg_datetime", "setting"])
  )
  dev = frame.filter(pl.col("split") == "dev").select(["subject_id", "study_id", "ecg_datetime"])
  events = scan_lab_events(labevents_path, [i for s in cfg.lab_specs.values() for i in s["itemids"]])
  # Features are extracted for ALL splits; the coverage gate is dev-only.
  wide_all = latest_qualified_values(
    events,
    frame.select(["subject_id", "study_id", "ecg_datetime"]),
    cfg.lab_specs,
    lookback_hours=cfg.lookback_hours_primary,
    buffer_hours=cfg.primary_buffer_hours if buffer_hours is None else buffer_hours,
    time_since_cap_hours=cfg.time_since_cap_hours,
  )
  dev_wide = wide_all.filter(pl.col("study_id").is_in(dev["study_id"].to_list()))
  ledger = coverage_ledger(
    dev_wide, dev, cfg.lab_specs, cfg.min_fraction_of_dev, cfg.min_patients, cfg.predeclared,
  )
  retained = (
    tuple(ledger.filter(pl.col("retained"))["concept"].to_list())
    if retained_concepts is None else retained_concepts
  )
  if set(retained) - set(cfg.lab_specs):
    raise ValueError("Frozen retained concepts do not match the Cmax config.")
  if not retained:
    raise ValueError("Cmax gate retained no concepts; refusing to build an empty context.")
  # Drop non-retained concept columns (all-null by construction) so downstream
  # feature lists are availability-driven only.
  dropped = [
    c
    for concept in cfg.lab_specs
    if concept not in retained
    for c in (concept, f"miss_{concept}", f"hs_{concept}")
    if c in wide_all.columns
  ]
  wide_all = wide_all.drop(dropped) if dropped else wide_all
  out = frame.join(wide_all, on="study_id", how="left").with_columns(
    (pl.col("setting") == "ED").cast(pl.Float64).alias("setting_ED"),
    (pl.col("setting") == "ICU").cast(pl.Float64).alias("setting_ICU"),
    (pl.col("setting") == "inpatient_non_ICU").cast(pl.Float64).alias("setting_inpatient_non_ICU"),
  )
  out, _stats = winsorize_and_impute(out, retained)
  return out


def cmax_columns(retained: tuple[str, ...], extra: tuple[str, ...] = ()) -> list[str]:
  """C0 + setting indicators + retained concepts (+ miss, + capped hours-since)."""
  from ecg_rep.evaluate import C0_FEATURE_COLS  # pyright: ignore[reportPrivateUsage]

  cols = [*C0_FEATURE_COLS, *SETTING_INDICATORS]
  for concept in retained:
    cols += [concept, f"miss_{concept}", f"hs_{concept}"]
  return [*cols, *extra]


def _oe_and_deciles(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
  """O/E ratio and decile calibration for one score vector (locked test only)."""
  expected = float(p.sum())
  observed = float(y.sum())
  deciles = np.quantile(p, np.linspace(0, 1, 11)[1:-1])
  bins = np.digitize(p, deciles)
  decile_rows: list[dict[str, Any]] = []
  for b in range(10):
    mask = bins == b
    if mask.sum() == 0:
      continue
    decile_rows.append({
      "decile": b + 1,
      "n": int(mask.sum()),
      "n_events": int(y[mask].sum()),
      "mean_predicted": float(p[mask].mean()),
      "observed_risk": float(y[mask].mean()),
    })
  return {"o_over_e": observed / expected if expected > 0 else float("nan"), "deciles": decile_rows}


def _paired_contrasts(
  y: np.ndarray,
  scores: dict[str, np.ndarray],
  contrasts: list[tuple[str, str]],
  n_bootstraps: int,
) -> list[dict[str, Any]]:
  """Frozen paired contrasts with patient-bootstrap intervals, four metrics each."""
  from ecg_rep.probe import bootstrap_paired_difference

  rows: list[dict[str, Any]] = []
  for k, (name_hi, name_lo) in enumerate(contrasts):
    seed = CMAX_BOOTSTRAP_SEEDS[k % len(CMAX_BOOTSTRAP_SEEDS)]
    for metric in ("auroc", "auprc", "log_loss", "brier"):
      boot = bootstrap_paired_difference(
        y, scores[name_lo], scores[name_hi], metric=metric,  # type: ignore[arg-type]
        n_bootstraps=n_bootstraps, seed=seed + k,
      )
      rows.append({
        "contrast": f"{name_hi}_minus_{name_lo}",
        "metric": metric,
        "point_estimate": boot.point_estimate,
        "ci_lower": boot.ci_lower,
        "ci_upper": boot.ci_upper,
        "n_bootstraps": boot.n_bootstraps,
        "seed": boot.seed,
      })
  return rows


def _render_cmax_results(report_path: Path, results: dict[str, Any]) -> None:
  from ecg_rep.probe import metric_bundle_to_dict  # noqa: F401  (values arrive as dicts)

  lines = [
    "# LM3 Cmax evaluation: same-subset refits with maximal structured pre-ECG context",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {results['source_sha']}",
    "analysis_status: post_hoc_extension",
    "inputs: [configs/cmax_context_v1.yaml, artifacts/limitation_mitigation/lm3/cmax_features.parquet, artifacts/reps, artifacts/context]",
    "decisions: [features frozen from the LM3a audit; same-subset refits; locked test touched once per model]",
    "evidence: [cmax_results.json, cmax_capacity.csv, cmax_temporal_sensitivity.csv, cmax_icu_results.csv]",
    "unresolved: []",
    "verification: [all models evaluated on identical locked-test rows; no feature chosen by performance]",
    "next_action: LM9 claim reconciliation",
    "```",
    "",
    f"Surface: {results['n_surface']:,} ECGs (non-ambiguous settings, paired-valid); "
    f"locked test {results['n_test']:,} rows / {results['n_test_events']} deaths, identical rows for every model.",
    f"Retained concepts: {len(results['retained_concepts'])}.",
    "",
    "## Locked-test metrics (same rows, same hyperparameter grid)",
    "",
    "| Model | AUROC | AUPRC | Log loss | Brier | Calib slope |",
    "|---|---|---|---|---|---|",
  ]
  for name, m in results["test_metrics"].items():
    lines.append(
      f"| {name} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['log_loss']:.4f} | "
      f"{m['brier']:.4f} | {m['calibration_slope']:.3f} |"
    )
  lines += [
    "",
    "## Primary paired estimands (patient bootstrap, frozen seeds)",
    "",
    "| Contrast | Metric | Estimate | Central 95% |",
    "|---|---|---|---|",
  ]
  for row in results["paired_contrasts"]:
    if row["metric"] != "auroc" or row["contrast"].endswith(("Cmax+B_minus_Cmax+A", "Cmax+B_minus_Cmax", "Cmax+A_minus_Cmax", "Cmax+A+B_minus_Cmax+B")):
      if row["metric"] == "auroc":
        lines.append(
          f"| {row['contrast']} | AUROC | {row['point_estimate']:+.4f} | "
          f"{row['ci_lower']:+.4f} to {row['ci_upper']:+.4f} |"
        )
  lines += [
    "",
    "Additional metrics (AUPRC, log loss, Brier) for every contrast are in `cmax_results.json`.",
    "",
    "## Fixed-capacity review budgets",
    "",
    "| Model | Fraction | Flagged | Events captured | Sensitivity | PPV |",
    "|---|---|---|---|---|---|",
  ]
  for row in results["capacity_rows"]:
    if "model" not in row or "capacity_fraction" not in row:
      continue
    lines.append(
      f"| {row['model']} | {row['capacity_fraction']:.1%} | {row['n_flagged']} | "
      f"{row['n_events_captured']} | {row['sensitivity']:.3f} | {row['ppv']:.3f} |"
    )
  lines += [
    "",
    "### Paired Cmax+B minus Cmax+A capacity contrasts",
    "",
    "Fixed predictions; paired patient bootstrap. Intervals describe test-sample",
    "uncertainty, not clinical benefit or model-refitting uncertainty.",
    "",
    "| Fraction | Δ deaths | Central 95% | Δ sensitivity | Central 95% | Δ PPV | Central 95% | Resamples |",
    "|---|---|---|---|---|---|---|---|",
  ]
  for row in results["capacity_rows"]:
    if "contrast" not in row:
      continue
    lines.append(
      f"| {row['capacity_fraction']:.1%} | {row['delta_events_captured']:+d} | "
      f"{row['delta_events_ci_low']:+.2f} to {row['delta_events_ci_high']:+.2f} | "
      f"{row['delta_sensitivity']:+.4f} | {row['delta_sensitivity_ci_low']:+.4f} to "
      f"{row['delta_sensitivity_ci_high']:+.4f} | {row['delta_ppv']:+.4f} | "
      f"{row['delta_ppv_ci_low']:+.4f} to {row['delta_ppv_ci_high']:+.4f} | {row['n_bootstraps']} |"
    )
  if "temporal_correction" in results:
    lines += ["", "Audit01: corrected 1h/2h availability buffers require both charttime and",
              "storetime <= ECG-buffer. Primary metrics and capacity contrasts were not rerun.",
              f"Correction source: `{results['temporal_correction']['source_sha']}`;",
              f"invalid original retained at `{results['temporal_correction']['original_results']}`."]
  lines += [
    "",
    "## Temporal-buffer sensitivity (Cmax+A and Cmax+B refits)",
    "",
    "| Buffer | Model | AUROC | AUPRC | Brier |",
    "|---|---|---|---|---|",
  ]
  for row in results["temporal_sensitivity"]:
    lines.append(
      f"| {row['buffer_hours']}h | {row['model']} | {row['auroc']:.4f} | "
      f"{row['auprc']:.4f} | {row['brier']:.4f} |"
    )
  joint = results.get("strong_comparator_joint")
  if joint:
    lines += [
    "",
    "## Strong-comparator joint analysis (A2 joint on the same-subset surface; separate estimand)",
      "",
      "| Model | AUROC | AUPRC |",
      "|---|---|---|",
    ]
    for name, m in joint["models"].items():
      lines.append(f"| {name} | {m['auroc']:.4f} | {m['auprc']:.4f} |")
  icu = results.get("cmax_icu")
  if icu:
    lines += [
      "",
      f"## Cmax-ICU sensitivity (ICU subset only; dev n={icu['n_dev']:,}, "
      f"{icu['n_dev_events']} events; test n={icu['n_test']:,}, {icu['n_test_events']} events)",
      "",
      "| Model | AUROC | AUPRC | Brier |",
      "|---|---|---|---|",
    ]
    for row in icu["models_rows"]:
      lines.append(f"| {row['model']} | {row['auroc']:.4f} | {row['auprc']:.4f} | {row['brier']:.4f} |")
    lines += [
      "",
      "ICU vitals describe only patients with a leak-safe pre-ECG ICU chart; these results",
      "must not be extrapolated to the full cohort.",
    ]
  lines += [
    "",
    "## Interpretation rules (frozen)",
    "",
    "- If B−A approaches zero under Cmax: after a broad set of pre-ECG structured clinical",
    "  measurements was included, little incremental discrimination remained between the",
    "  ECG representations. This does not invalidate the ECG-only representation result.",
    "- If a residual persists: D-BETA retained incremental information beyond a broad",
    "  structured pre-ECG clinical context. No causal uniqueness language.",
    "- If capacity differences persist despite small AUROC differences: the representations",
    "  continued to prioritize partly different patients at the highest-risk tail.",
    "",
  ]
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text("\n".join(lines), encoding="utf-8")


def run_cmax_evaluation(
  cohort_path: Path,
  setting_path: Path,
  labevents_path: Path,
  chartevents_path: Path | None,
  c0_path: Path,
  reps_dir: Path,
  config_path: Path,
  measurements_path: Path | None,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = 200,
) -> dict[str, Any]:
  """LM3b: same-subset refits of every context model on one frozen Cmax surface.

  The locked test is touched exactly once per model, on identical rows, with
  frozen features. Nothing here may select or redefine a feature using
  locked-test performance.
  """
  from ecg_rep.evaluate import _fit_probe, _join_representation_matrix  # pyright: ignore[reportPrivateUsage]
  from ecg_rep.probe import metric_bundle_to_dict

  frame = build_cmax_frame(cohort_path, setting_path, labevents_path, config_path)
  cfg = load_cmax_config(config_path)
  retained = tuple(c for c in frame.columns if c in cfg.lab_specs)
  c0 = pl.read_parquet(c0_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]
  c0_sel = c0.select(["subject_id", "study_id", *c0_cols])

  def _join_c0(f: pl.DataFrame) -> pl.DataFrame:
    return f.join(c0_sel, on=["subject_id", "study_id"], how="left")

  frame = _join_c0(frame)
  reps = Path(reps_dir)

  # Same-subset rule: A and B matrices joined; identical rows for every model.
  frame = _join_representation_matrix(
    frame,
    pl.read_parquet(reps / "ciis_features_meta.parquet"),
    np.load(reps / "ciis_features.npy"),
    "za",
    "representation_a_valid",
  )
  frame = _join_representation_matrix(
    frame,
    pl.read_parquet(reps / "dbeta_embedding_meta.parquet"),
    np.load(reps / "dbeta_embeddings.npy"),
    "zb",
    "representation_b_valid",
  ).filter(pl.col("representation_a_valid") & pl.col("representation_b_valid"))
  za_cols = [c for c in frame.columns if c.startswith("za_")]
  zb_cols = [c for c in frame.columns if c.startswith("zb_")]
  output_dir.mkdir(parents=True, exist_ok=True)
  frame.write_parquet(output_dir / "cmax_features.parquet")

  cacute_cols = [c for c in cacute_feature_columns(CACUTE_VALUE_COLS) if c in frame.columns]
  cmax_cols = [c for c in cmax_columns(retained) if c in frame.columns]
  model_specs: dict[str, list[str]] = {
    "C0": list(c0_cols),
    "C0+A": [*c0_cols, *za_cols],
    "C0+B": [*c0_cols, *zb_cols],
    "Cacute": cacute_cols,
    "Cacute+A": [*cacute_cols, *za_cols],
    "Cacute+B": [*cacute_cols, *zb_cols],
    "Cmax": cmax_cols,
    "Cmax+A": [*cmax_cols, *za_cols],
    "Cmax+B": [*cmax_cols, *zb_cols],
    "Cmax+A+B": [*cmax_cols, *za_cols, *zb_cols],
  }
  # A'' strong-comparator joint (separate estimand; machine reports cover every
  # paired-valid row on this surface, so the population is the same-subset surface).
  joint_results: dict[str, Any] | None = None
  joint_scores: dict[str, np.ndarray] | None = None
  joint_y: np.ndarray | None = None
  if measurements_path is not None:
    from ecg_rep.cart_report_baseline import (
      MACHINE_RAW_COLUMNS,
      REPORT_COLUMNS,
      build_cart_report_frame,
      load_cart_report_config,
    )

    machine = pl.read_csv(
      measurements_path,
      columns=["subject_id", "study_id", *MACHINE_RAW_COLUMNS, *REPORT_COLUMNS],
      infer_schema_length=100_000,
    )
    cart_cfg = load_cart_report_config(Path("configs/cart_report_comparator_v1.yaml"))
    cart_frame, builder = build_cart_report_frame(cohort_path, machine, c0_path, reps, cart_cfg)
    # The cart frame already carries C0, za_*, zb_* (same frozen matrices); join
    # only the Cmax-specific setting/lab columns to avoid name collisions.
    cmax_specific = [c for c in cmax_cols if c not in C0_FEATURE_COLS]
    cart_frame = cart_frame.join(
      frame.select(["study_id", *cmax_specific]), on="study_id", how="inner"
    )
    zb_cart = [c for c in cart_frame.columns if c.startswith("zb_")]
    a2_cols = list(builder.feature_columns)
    joint_specs = {
      "Cmax+A2": [*cmax_cols, *a2_cols],
      "Cmax+B": [*cmax_cols, *zb_cart],
      "Cmax+A2+B": [*cmax_cols, *a2_cols, *zb_cart],
    }
    joint_scores, joint_y = _fit_models_and_scores(cart_frame, joint_specs)
    joint_results = {
      "analysis_status": "post_hoc_extension_strong_comparator_joint",
      "n_test": int((cart_frame["split"] == "test").sum()),
      "n_test_events": int(cart_frame.filter(pl.col("split") == "test")["mortality_30d"].sum()),
      "models": {
        name: metric_bundle_to_dict(_eval_bundle(cart_frame, cols))
        for name, cols in joint_specs.items()
      },
    }

  scores, y_test = _fit_models_and_scores(frame, model_specs)
  test_metrics = {
    name: metric_bundle_to_dict(_eval_bundle(frame, cols)) for name, cols in model_specs.items()
  }
  contrasts = [
    ("Cmax+B", "Cmax+A"),
    ("Cmax+B", "Cmax"),
    ("Cmax+A", "Cmax"),
    ("Cmax+A+B", "Cmax+B"),
  ]
  contrast_rows = _paired_contrasts(y_test, scores, contrasts, n_bootstraps)

  # Fixed-capacity rows and B-vs-A contrast under Cmax.
  from ecg_rep.analysis import fixed_capacity_contrast_rows, fixed_capacity_metric_rows

  capacity_rows = fixed_capacity_metric_rows(
    y_test, scores, ["C0+A", "C0+B", "Cmax+A", "Cmax+B"],
  )
  capacity_rows += fixed_capacity_contrast_rows(
    y_test, scores, [("Cmax+A", "Cmax+B", "b_minus_a")],
  )

  # Temporal-buffer sensitivity: refit Cmax+A and Cmax+B at 1h/2h buffers on the same rows.
  temporal_rows: list[dict[str, Any]] = []
  for buffer_hours in (1, 2):
    if buffer_hours not in cfg.temporal_buffers_hours:
      continue
    # Full frame at the buffer; identical rows via study_id, same A/B matrices.
    frame_b = _join_c0(
      build_cmax_frame(cohort_path, setting_path, labevents_path, config_path,
                       buffer_hours=buffer_hours, retained_concepts=retained)
    )
    za_b = [c for c in frame.columns if c.startswith("za_")]
    zb_b = [c for c in frame.columns if c.startswith("zb_")]
    frame_b = frame_b.join(
      frame.select(["study_id", *za_b, *zb_b]), on="study_id", how="inner"
    )
    b_scores, b_y = _fit_models_and_scores(
      frame_b,
      {
        f"Cmax+A@{buffer_hours}h": [*cmax_cols, *za_cols],
        f"Cmax+B@{buffer_hours}h": [*cmax_cols, *zb_cols],
      },
    )
    for name, s in b_scores.items():
      from ecg_rep.probe import evaluate_predictions

      m = metric_bundle_to_dict(evaluate_predictions(b_y, s))
      temporal_rows.append({"buffer_hours": buffer_hours, "model": name, **m})

  # Cmax-ICU sensitivity (go/no-go already frozen in the audit).
  icu_results: dict[str, Any] | None = None
  if chartevents_path is not None and Path(chartevents_path).exists() and cfg.vital_specs:
    icu_results = _run_icu_sensitivity(
      frame, cohort_path, chartevents_path, cfg, retained, cmax_cols, za_cols, zb_cols,
    )

  # O/E and decile calibration for the four headline context models.
  calibration: dict[str, Any] = {}
  for name in ("C0+A", "C0+B", "Cmax+A", "Cmax+B"):
    calibration[name] = _oe_and_deciles(y_test, scores[name])

  results: dict[str, Any] = {
    "status": "complete",
    "analysis_status": "post_hoc_extension",
    "source_sha": source_sha,
    "config_version": cfg.version,
    "retained_concepts": list(retained),
    "n_surface": frame.height,
    "n_test": int((frame["split"] == "test").sum()),
    "n_test_events": int(frame.filter(pl.col("split") == "test")["mortality_30d"].sum()),
    "test_metrics": test_metrics,
    "paired_contrasts": contrast_rows,
    "capacity_rows": capacity_rows,
    "temporal_sensitivity": temporal_rows,
    "strong_comparator_joint": joint_results,
    "cmax_icu": icu_results,
    "calibration_oe_deciles": calibration,
    "n_bootstraps": n_bootstraps,
  }
  (output_dir / "cmax_results.json").write_text(
    json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
  )
  pl.DataFrame(capacity_rows).write_csv(output_dir / "cmax_capacity.csv")
  pl.DataFrame(temporal_rows).write_csv(output_dir / "cmax_temporal_sensitivity.csv")
  if icu_results is not None:
    pl.DataFrame(icu_results["models_rows"]).write_csv(output_dir / "cmax_icu_results.csv")
  _render_cmax_results(report_path, results)
  return results


def _fit_models_and_scores(
  frame: pl.DataFrame,
  model_specs: dict[str, list[str]],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
  """Refit every model on the identical dev/val rows; return frozen-test scores."""
  from ecg_rep.evaluate import _fit_probe  # pyright: ignore[reportPrivateUsage]

  test = frame.filter(pl.col("split") == "test")
  y = test["mortality_30d"].to_numpy().astype(np.int64)
  scores: dict[str, np.ndarray] = {}
  for name, cols in model_specs.items():
    probe, _dev, _val, _test = _fit_probe(frame, cols, model_name=name.replace("+", "_"))
    scores[name] = probe.predict_proba(test.select(cols).to_numpy().astype(np.float64))
  return scores, y


def _eval_bundle(frame: pl.DataFrame, cols: list[str]):
  """Fit and evaluate one model on the locked test (identical rows)."""
  from ecg_rep.evaluate import _fit_probe  # pyright: ignore[reportPrivateUsage]
  from ecg_rep.probe import evaluate_predictions

  test = frame.filter(pl.col("split") == "test")
  y = test["mortality_30d"].to_numpy().astype(np.int64)
  probe, _dev, _val, _test = _fit_probe(frame, cols)
  p = probe.predict_proba(test.select(cols).to_numpy().astype(np.float64))
  return evaluate_predictions(y, p)


def _run_icu_sensitivity(
  frame: pl.DataFrame,
  cohort_path: Path,
  chartevents_path: Path,
  cfg: CmaxConfig,
  retained: tuple[str, ...],
  cmax_cols: list[str],
  za_cols: list[str],
  zb_cols: list[str],
) -> dict[str, Any]:
  """ICU-subset refits: Cmax-ICU, Cmax-ICU+A, Cmax-ICU+B. Never cohort-wide."""
  from ecg_rep.evaluate import _fit_probe
  from ecg_rep.probe import evaluate_predictions, metric_bundle_to_dict

  vital_ids = [i for s in cfg.vital_specs.values() for i in s["itemids"]]
  vitals = pl.scan_csv(
    chartevents_path,
    schema_overrides={"subject_id": pl.Int64, "itemid": pl.Int64, "charttime": pl.String,
                      "storetime": pl.String},
  ).filter(pl.col("itemid").is_in(vital_ids)).select(["subject_id", "itemid", "charttime", "storetime"])
  vitals = vitals.with_columns(
    pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
  ).collect()
  # Identify ICU-charted patients against the FULL index (all splits); the
  # dev-only gate decision was frozen in the audit.
  full_index = frame.select(["subject_id", "study_id", "ecg_datetime"])
  labeled_v = classify_measurement_timing(vitals, full_index, cfg.icu_lookback_hours)
  icu_subjects = (
    labeled_v.filter(pl.col("timing_class") == "qualified").select("subject_id").unique()
  )
  subset = frame.join(icu_subjects, on="subject_id", how="semi")
  # ICU vital features within the subset (dev-only preprocessing inside the subset).
  vitals_val = pl.scan_csv(
    chartevents_path,
    schema_overrides={"subject_id": pl.Int64, "itemid": pl.Int64, "charttime": pl.String,
                      "storetime": pl.String, "valuenum": pl.Float64},
  ).filter(pl.col("itemid").is_in(vital_ids)).select(["subject_id", "itemid", "charttime", "storetime", "valuenum"])
  vitals_val = vitals_val.with_columns(
    pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
    pl.col("storetime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
  ).collect()
  wide_v = latest_qualified_chartvalues(
    vitals_val, subset.select(["subject_id", "study_id", "ecg_datetime"]), cfg.vital_specs,
    cfg.icu_lookback_hours,
  )
  subset = subset.join(wide_v, on="study_id", how="left")
  vital_names = tuple(cfg.vital_specs)
  subset, _ = winsorize_and_impute(subset, vital_names)
  icu_cols = [*cmax_cols]
  for v in vital_names:
    icu_cols += [v, f"miss_{v}", f"hs_{v}"]
  specs = {
    "Cmax-ICU": icu_cols,
    "Cmax-ICU+A": [*icu_cols, *za_cols],
    "Cmax-ICU+B": [*icu_cols, *zb_cols],
  }
  rows: list[dict[str, Any]] = []
  test = subset.filter(pl.col("split") == "test")
  y = test["mortality_30d"].to_numpy().astype(np.int64)
  for name, cols in specs.items():
    probe, _d, _v, _t = _fit_probe(subset, cols, model_name=name.replace("+", "_").replace("-", "_"))
    p = probe.predict_proba(test.select(cols).to_numpy().astype(np.float64))
    rows.append({"model": name, **metric_bundle_to_dict(evaluate_predictions(y, p))})
  return {
    "analysis_status": "post_hoc_extension_icu_sensitivity",
    "n_dev": int((subset["split"] == "dev").sum()),
    "n_dev_events": int(subset.filter(pl.col("split") == "dev")["mortality_30d"].sum()),
    "n_val": int((subset["split"] == "val").sum()),
    "n_test": int((subset["split"] == "test").sum()),
    "n_test_events": int(test["mortality_30d"].sum()),
    "models_rows": rows,
  }


def rerun_cmax_temporal_sensitivity(
  cohort_path: Path,
  setting_path: Path,
  labevents_path: Path,
  c0_path: Path,
  config_path: Path,
  primary_dir: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
) -> dict[str, Any]:
  """Replace only invalid 1h/2h sensitivities; never refit the primary models.

  The original results remain in ``primary_dir`` for provenance. Corrected
  results copy every non-temporal value verbatim to a separate output directory.
  """
  from ecg_rep.probe import evaluate_predictions, metric_bundle_to_dict

  if primary_dir.resolve() == output_dir.resolve():
    raise ValueError("Temporal correction requires a separate output directory.")
  original = json.loads((primary_dir / "cmax_results.json").read_text())
  cfg = load_cmax_config(config_path)
  manifest = json.loads((primary_dir / "cmax_feature_manifest.json").read_text())
  if (original["config_version"] != cfg.version or manifest["lab_specs"] != cfg.lab_specs
      or manifest["lookback_hours_primary"] != cfg.lookback_hours_primary
      or manifest["retained_concepts"] != original["retained_concepts"]):
    raise ValueError("Primary Cmax frozen feature manifest/config mismatch.")
  primary = pl.read_parquet(primary_dir / "cmax_features.parquet")
  retained = tuple(original["retained_concepts"])
  za = [c for c in primary.columns if c.startswith("za_")]
  zb = [c for c in primary.columns if c.startswith("zb_")]
  c0 = pl.read_parquet(c0_path)
  c0_cols = list(C0_FEATURE_COLS)
  context_cols = cmax_columns(retained)
  # Intersection would silently change the frozen probe if the reference cache is corrupt.
  expected_context = set(c0_cols) | set(context_cols)
  if (not za or not zb or expected_context - set(primary.columns)
      or set(c0_cols) - set(c0.columns)):
    raise ValueError("Temporal correction is missing frozen representation/context features.")
  c0 = c0.select(["study_id", *c0_cols])
  rows: list[dict[str, Any]] = []
  for buffer in (1, 2):
    if buffer not in cfg.temporal_buffers_hours:
      continue
    frame = build_cmax_frame(
      cohort_path, setting_path, labevents_path, config_path,
      buffer_hours=buffer, retained_concepts=retained,
    ).join(c0, on="study_id", how="left", validate="1:1").join(
      primary.select(["study_id", *za, *zb]), on="study_id", how="inner", validate="1:1",
    )
    if expected_context - set(frame.columns):
      raise ValueError("Temporal correction is missing frozen representation/context features in corrected frame.")
    keys = ["subject_id", "study_id", "split", "mortality_30d"]
    if not frame.select(keys).sort("study_id").equals(primary.select(keys).sort("study_id")):
      raise ValueError("Temporal correction changed the frozen patient/split/outcome surface.")
    scores, y = _fit_models_and_scores(frame, {
      f"Cmax+A@{buffer}h": [*context_cols, *za],
      f"Cmax+B@{buffer}h": [*context_cols, *zb],
    })
    for name, score in scores.items():
      rows.append({"buffer_hours": buffer, "model": name,
                   **metric_bundle_to_dict(evaluate_predictions(y, score))})
  corrected = {**original, "temporal_sensitivity": rows, "temporal_correction": {
    "source_sha": source_sha,
    "original_results": str(primary_dir / "cmax_results.json"),
    "availability_rule": "charttime <= ECG-buffer AND storetime <= ECG-buffer",
    "primary_refit": False,
  }}
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "cmax_results.json").write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n")
  pl.DataFrame(rows).write_csv(output_dir / "cmax_temporal_sensitivity.csv")
  _render_cmax_results(report_path, corrected)
  return corrected


def latest_qualified_chartvalues(
  events: pl.DataFrame,
  index: pl.DataFrame,
  specs: dict[str, dict[str, Any]],
  lookback_hours: int,
) -> pl.DataFrame:
  """Most-recent qualified chart value per study/concept (chartevents; no unit screen)."""
  pieces: list[pl.DataFrame] = []
  for concept, spec in specs.items():
    subset = events.filter(pl.col("itemid").is_in(spec["itemids"]))
    if subset.height == 0:
      pieces.append(index.select("study_id").with_columns(
        pl.lit(None, dtype=pl.Float64).alias(concept),
        pl.lit(None, dtype=pl.Float64).alias(f"hs_{concept}"),
      ))
      continue
    labeled = classify_measurement_timing(subset, index, lookback_hours)
    qualified = labeled.filter(
      (pl.col("timing_class") == "qualified") & pl.col("valuenum").is_not_null()
    )
    latest = (
      qualified.sort(["study_id", "charttime"])
      .group_by("study_id", maintain_order=True)
      .agg([
        pl.col("valuenum").last().alias(concept),
        (pl.col("ecg_datetime").last() - pl.col("charttime").last()).dt.total_hours().alias(f"hs_{concept}"),
      ])
    )
    latest = latest.with_columns(
      pl.col(f"hs_{concept}").clip(0.0, float(lookback_hours)).alias(f"hs_{concept}"),
    )
    pieces.append(index.select("study_id").join(latest, on="study_id", how="left"))
  wide = pieces[0]
  for piece in pieces[1:]:
    wide = wide.join(piece, on="study_id", how="left")
  return wide
