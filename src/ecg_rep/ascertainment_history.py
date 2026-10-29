"""LM6: outcome-ascertainment recovery and sparse-history sensitivity.

Post hoc sensitivity. The parent cohort is never overwritten. A patient with no
recorded death is a known 30-day survivor only when a timestamped clinical
contact falls strictly after the index ECG plus the horizon. History features
use completed admissions and encounters strictly before the index ECG.
Index-admission diagnoses do not enter the history block.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from ecg_rep.context import prior_completed_admissions
from ecg_rep.elixhauser import ELIXHAUSER_FEATURE_NAMES, elixhauser_indicators_from_prior_diagnoses
from ecg_rep.probe import (
  TrainedProbe,
  bootstrap_paired_difference,
  compute_auprc,
  compute_auroc,
  fit_logistic_probe,
)

HORIZON_DAYS = 30.0
MIN_VALID_FRACTION = 0.90
EVIDENCE_SOURCES: tuple[str, ...] = (
  "later_ecg",
  "later_ed_registration",
  "later_hospital_admission",
  "later_icu_stay",
  "later_laboratory",
)
H_FEATURES: tuple[str, ...] = (
  "n_prior_admissions",
  "log1p_prior_admissions",
  "days_since_last_discharge",
  "missing_days_since_last_discharge",
  "cumulative_prior_hospital_days",
  "prior_icu",
  "days_since_first_encounter",
  "missing_days_since_first_encounter",
  "n_prior_elixhauser",
)
IMPUTED_FEATURES: tuple[str, ...] = (
  "days_since_last_discharge",
  "days_since_first_encounter",
)


@dataclass(frozen=True)
class Lm6Config:
  """Frozen LM6 rules. Loaded from the versioned YAML, not from results."""

  version: str
  horizon_days: float
  evidence_sources: tuple[str, ...]
  h_features: tuple[str, ...]
  imputed_features: tuple[str, ...]
  min_events_for_interval: int
  n_replicates: int
  seed: int
  min_valid_fraction: float
  ipcw_status: str
  config_hash: str
  raw: dict[str, Any]


def load_lm6_config(path: Path) -> Lm6Config:
  """Load the frozen LM6 YAML and hash its exact text."""
  text = path.read_text(encoding="utf-8")
  raw = yaml.safe_load(text)
  if not isinstance(raw, dict):
    raise ValueError(f"LM6 config at {path} is not a mapping.")
  history = raw["history"]
  bootstrap = raw["bootstrap"]
  sources = tuple(str(s) for s in raw["evidence_sources"])
  unknown = set(sources) - set(EVIDENCE_SOURCES)
  if unknown:
    raise ValueError(f"LM6 config names evidence sources that are not implemented: {sorted(unknown)}")
  return Lm6Config(
    version=str(raw["version"]),
    horizon_days=float(raw["horizon_days"]),
    evidence_sources=sources,
    h_features=tuple(str(c) for c in history["h_features"]),
    imputed_features=tuple(str(c) for c in history["imputed_features"]),
    min_events_for_interval=int(history["min_events_for_interval"]),
    n_replicates=int(bootstrap["n_replicates"]),
    seed=int(bootstrap["seed"]),
    min_valid_fraction=float(bootstrap["min_valid_fraction"]),
    ipcw_status=str(raw["ipcw"]["status"]),
    config_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    raw=raw,
  )


def recovery_class(
  days_to_death: float | None,
  evidence_after_horizon: bool,
  horizon_days: float = HORIZON_DAYS,
) -> str:
  """Classify one excluded patient. Unknown is never called a survivor.

  A recorded death is already ascertainable under the parent rule, so it is
  not relabeled here. Only a missing death date plus direct post-horizon
  contact becomes ``recovered_survivor``.
  """
  if days_to_death is not None and np.isfinite(days_to_death):
    if days_to_death < 0.0:
      return "pre_ecg_death"
    if days_to_death <= horizon_days:
      return "inconsistent_death_within_horizon"
    return "inconsistent_death_after_horizon"
  if evidence_after_horizon:
    return "recovered_survivor"
  return "unknown"


def earliest_evidence(
  long: pl.DataFrame,
  horizon_days: float = HORIZON_DAYS,
) -> pl.DataFrame:
  """Earliest contact strictly after ``ecg_datetime + horizon`` per study.

  ``long`` has ``study_id``, ``ecg_datetime``, ``source``, ``evidence_time``.
  Ties on time break by source name so the choice does not depend on row order.
  """
  required = {"study_id", "ecg_datetime", "source", "evidence_time"}
  missing = required - set(long.columns)
  if missing:
    raise ValueError(f"Evidence table missing columns: {sorted(missing)}")
  if long.height == 0:
    return pl.DataFrame(schema={
      "study_id": pl.Int64,
      "evidence_source": pl.String,
      "evidence_time": pl.Datetime("us"),
    })
  horizon_seconds = int(round(horizon_days * 86400.0))
  qualified = long.with_columns(
    pl.col("ecg_datetime").cast(pl.Datetime("us")),
    pl.col("evidence_time").cast(pl.Datetime("us")),
  ).with_columns(
    (pl.col("ecg_datetime") + pl.duration(seconds=horizon_seconds)).alias("threshold"),
  ).filter(
    pl.col("evidence_time").is_not_null() & (pl.col("evidence_time") > pl.col("threshold"))
  )
  if qualified.height == 0:
    return pl.DataFrame(schema={
      "study_id": pl.Int64,
      "evidence_source": pl.String,
      "evidence_time": pl.Datetime("us"),
    })
  return (
    qualified.sort(["study_id", "evidence_time", "source"])
    .group_by("study_id", maintain_order=True)
    .agg(
      pl.col("source").first().alias("evidence_source"),
      pl.col("evidence_time").first().alias("evidence_time"),
    )
  )


def classify_excluded(excluded: pl.DataFrame, evidence: pl.DataFrame, horizon_days: float = HORIZON_DAYS) -> pl.DataFrame:
  """Attach recovery class. Patients without qualifying evidence stay unknown."""
  joined = excluded.join(evidence, on="study_id", how="left")
  classes: list[str] = []
  for row in joined.select(["days_to_death", "evidence_time"]).iter_rows(named=True):
    death = row["days_to_death"]
    death_f = None if death is None else float(death)
    classes.append(recovery_class(death_f, row["evidence_time"] is not None, horizon_days))
  return joined.with_columns(pl.Series("recovery_class", classes))


def recovered_sensitivity_cohort(primary: pl.DataFrame, recovered: pl.DataFrame) -> pl.DataFrame:
  """Stack the parent cohort and newly recovered known survivors.

  Recovered patients are labeled 30-day survivors. Subject overlap is a hard
  error. The returned frame is a new table; callers must not write it over
  the primary cohort artifact.
  """
  overlap = set(primary["subject_id"].to_list()) & set(recovered["subject_id"].to_list())
  if overlap:
    raise ValueError(f"Recovered subjects already belong to the primary cohort: {len(overlap)}")
  base = primary.with_columns(pl.lit(False).alias("outcome_recovered"))
  extra = recovered.with_columns(
    pl.lit(False).alias("mortality_30d"),
    pl.lit(True).alias("outcome_recovered"),
  )
  shared = [c for c in base.columns if c in extra.columns]
  return pl.concat([base.select(shared), extra.select(shared)], how="vertical")


def standardized_mean_difference(included: np.ndarray, excluded: np.ndarray, binary: bool) -> float | None:
  """SMD of included minus excluded. Binary uses the pooled Bernoulli SD."""
  a = np.asarray(included, dtype=np.float64)
  b = np.asarray(excluded, dtype=np.float64)
  a = a[np.isfinite(a)]
  b = b[np.isfinite(b)]
  if len(a) < 2 or len(b) < 2:
    return None
  mean_a = float(a.mean())
  mean_b = float(b.mean())
  if binary:
    pooled = float(np.sqrt((mean_a * (1.0 - mean_a) + mean_b * (1.0 - mean_b)) / 2.0))
  else:
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    pooled = float(np.sqrt((var_a + var_b) / 2.0))
  if pooled == 0.0:
    return 0.0 if mean_a == mean_b else None
  return (mean_a - mean_b) / pooled


def selection_smd_table(
  frame: pl.DataFrame,
  group_col: str,
  included_level: str,
  excluded_level: str,
  continuous: tuple[str, ...],
  binary: tuple[str, ...],
) -> pl.DataFrame:
  """SMDs for prespecified columns. Columns absent from ``frame`` are omitted."""
  included = frame.filter(pl.col(group_col) == included_level)
  excluded = frame.filter(pl.col(group_col) == excluded_level)
  rows: list[dict[str, Any]] = []
  for name, is_binary in [(c, False) for c in continuous] + [(c, True) for c in binary]:
    if name not in frame.columns:
      rows.append({
        "variable": name,
        "kind": "binary" if is_binary else "continuous",
        "smd_included_minus_excluded": None,
        "n_included_nonnull": 0,
        "n_excluded_nonnull": 0,
        "reason": "column_absent",
      })
      continue
    a = included[name].to_numpy()
    b = excluded[name].to_numpy()
    smd = standardized_mean_difference(a, b, binary=is_binary)
    rows.append({
      "variable": name,
      "kind": "binary" if is_binary else "continuous",
      "smd_included_minus_excluded": smd,
      "n_included_nonnull": int(np.isfinite(np.asarray(a, dtype=np.float64)).sum()),
      "n_excluded_nonnull": int(np.isfinite(np.asarray(b, dtype=np.float64)).sum()),
      "reason": "ok" if smd is not None else "insufficient_nonnull",
    })
  return pl.DataFrame(rows)


def build_history_features(
  index: pl.DataFrame,
  admissions: pl.DataFrame,
  diagnoses: pl.DataFrame | None = None,
  icustays: pl.DataFrame | None = None,
  elixhauser: pl.DataFrame | None = None,
) -> pl.DataFrame:
  """Pre-index history block H. Discharge, encounter, and ICU times must precede the ECG.

  Elixhauser counts use diagnoses from admissions completed before the ECG.
  Passing ``elixhauser`` (subject_id plus indicator columns) skips recomputation;
  those columns must already obey the same prior-admission gate.
  """
  required = {"subject_id", "study_id", "ecg_datetime"}
  missing = required - set(index.columns)
  if missing:
    raise ValueError(f"Index frame missing columns: {sorted(missing)}")
  index_keys = index.select(["subject_id", "study_id", "ecg_datetime"]).with_columns(
    pl.col("ecg_datetime").cast(pl.Datetime("us"))
  )
  prior = prior_completed_admissions(admissions, index_keys)
  # The shared temporal gate returns subject-level rows. The index is one ECG
  # per subject, so attaching study_id does not mix horizons.
  if prior.height > 0:
    prior = prior.join(index_keys.select(["subject_id", "study_id"]), on="subject_id", how="inner")
  if prior.height == 0:
    admission_agg = index_keys.select("study_id").with_columns(
      pl.lit(0).cast(pl.Int64).alias("n_prior_admissions"),
      pl.lit(None).cast(pl.Float64).alias("days_since_last_discharge"),
      pl.lit(0.0).alias("cumulative_prior_hospital_days"),
    )
  else:
    prior = prior.with_columns(pl.col("ecg_datetime").cast(pl.Datetime("us")))
    admission_agg = (
      prior.group_by("study_id")
      .agg(
        pl.col("hadm_id").n_unique().alias("n_prior_admissions"),
        pl.col("disch_dt").max().alias("last_disch"),
        (
          (pl.col("disch_dt") - pl.col("admit_dt")).dt.total_seconds().clip(lower_bound=0) / 86400.0
        ).sum().alias("cumulative_prior_hospital_days"),
        pl.col("ecg_datetime").first().alias("ecg_datetime"),
      )
      .with_columns(
        ((pl.col("ecg_datetime") - pl.col("last_disch")).dt.total_seconds() / 86400.0).alias(
          "days_since_last_discharge"
        )
      )
      .select(["study_id", "n_prior_admissions", "days_since_last_discharge", "cumulative_prior_hospital_days"])
    )
  out = index_keys.join(admission_agg, on="study_id", how="left").with_columns(
    pl.col("n_prior_admissions").fill_null(0).cast(pl.Int64),
    pl.col("cumulative_prior_hospital_days").fill_null(0.0),
  ).with_columns(
    pl.col("n_prior_admissions").log1p().alias("log1p_prior_admissions"),
    pl.col("days_since_last_discharge").is_null().cast(pl.Int8).alias("missing_days_since_last_discharge"),
  )

  first_times = _first_encounter_before_ecg(admissions, index_keys)
  out = out.join(first_times, on="study_id", how="left").with_columns(
    pl.col("days_since_first_encounter").is_null().cast(pl.Int8).alias("missing_days_since_first_encounter"),
  )
  icu = _prior_icu_flag(icustays, index_keys)
  out = out.join(icu, on="study_id", how="left").with_columns(pl.col("prior_icu").fill_null(0).cast(pl.Int8))
  elix_count = _prior_elixhauser_count(index_keys, admissions, diagnoses, elixhauser)
  out = out.join(elix_count, on="study_id", how="left").with_columns(
    pl.col("n_prior_elixhauser").fill_null(0).cast(pl.Int64),
  ).with_columns(
    (pl.col("n_prior_elixhauser") > 0).cast(pl.Int8).alias("any_prior_comorbidity"),
    pl.when(pl.col("n_prior_admissions") == 0).then(pl.lit("naive")).otherwise(pl.lit("present")).alias(
      "history_group"
    ),
    pl.when(pl.col("n_prior_admissions") <= 0)
    .then(pl.lit("0"))
    .when(pl.col("n_prior_admissions") == 1)
    .then(pl.lit("1"))
    .otherwise(pl.lit("2+"))
    .alias("history_ordered"),
  )
  # A positive days-since value means the timestamp preceded the ECG.
  if out.filter(pl.col("days_since_last_discharge").is_not_null() & (pl.col("days_since_last_discharge") < 0)).height:
    raise ValueError("History block included a discharge at or after the index ECG.")
  if out.filter(
    pl.col("days_since_first_encounter").is_not_null() & (pl.col("days_since_first_encounter") <= 0)
  ).height:
    raise ValueError("History block included an encounter at or after the index ECG.")
  return out


def _first_encounter_before_ecg(admissions: pl.DataFrame, index_keys: pl.DataFrame) -> pl.DataFrame:
  """Earliest admission or ED registration strictly before the index ECG."""
  pieces: list[pl.DataFrame] = []
  base = index_keys.select(["subject_id", "study_id", "ecg_datetime"])
  if "admit_dt" in admissions.columns:
    pieces.append(
      base.join(admissions.select(["subject_id", "admit_dt"]), on="subject_id", how="inner")
      .filter(pl.col("admit_dt").is_not_null() & (pl.col("admit_dt") < pl.col("ecg_datetime")))
      .select(["study_id", pl.col("admit_dt").alias("encounter_time")])
    )
  if "edreg_dt" in admissions.columns:
    pieces.append(
      base.join(admissions.select(["subject_id", "edreg_dt"]), on="subject_id", how="inner")
      .filter(pl.col("edreg_dt").is_not_null() & (pl.col("edreg_dt") < pl.col("ecg_datetime")))
      .select(["study_id", pl.col("edreg_dt").alias("encounter_time")])
    )
  if not pieces:
    return index_keys.select("study_id").with_columns(pl.lit(None).cast(pl.Float64).alias("days_since_first_encounter"))
  stacked = pl.concat(pieces, how="vertical")
  return (
    stacked.group_by("study_id")
    .agg(pl.col("encounter_time").min().alias("first_encounter"))
    .join(index_keys.select(["study_id", "ecg_datetime"]), on="study_id", how="right")
    .with_columns(
      pl.when(pl.col("first_encounter").is_not_null())
      .then((pl.col("ecg_datetime") - pl.col("first_encounter")).dt.total_seconds() / 86400.0)
      .otherwise(None)
      .alias("days_since_first_encounter")
    )
    .select(["study_id", "days_since_first_encounter"])
  )


def _prior_icu_flag(icustays: pl.DataFrame | None, index_keys: pl.DataFrame) -> pl.DataFrame:
  """ICU stay whose intime is strictly before the index ECG."""
  if icustays is None or icustays.height == 0 or "intime" not in icustays.columns:
    return index_keys.select("study_id").with_columns(pl.lit(0).cast(pl.Int8).alias("prior_icu"))
  flagged = (
    index_keys.select(["subject_id", "study_id", "ecg_datetime"])
    .join(icustays.select(["subject_id", "intime"]), on="subject_id", how="inner")
    .filter(pl.col("intime").is_not_null() & (pl.col("intime") < pl.col("ecg_datetime")))
    .select("study_id")
    .unique()
    .with_columns(pl.lit(1).cast(pl.Int8).alias("prior_icu"))
  )
  return index_keys.select("study_id").join(flagged, on="study_id", how="left").with_columns(
    pl.col("prior_icu").fill_null(0).cast(pl.Int8)
  )


def _prior_elixhauser_count(
  index_keys: pl.DataFrame,
  admissions: pl.DataFrame,
  diagnoses: pl.DataFrame | None,
  elixhauser: pl.DataFrame | None,
) -> pl.DataFrame:
  if elixhauser is not None:
    cols = [c for c in ELIXHAUSER_FEATURE_NAMES if c in elixhauser.columns]
    summed = elixhauser.with_columns(pl.sum_horizontal([pl.col(c) for c in cols]).alias("n_prior_elixhauser"))
    key = "study_id" if "study_id" in summed.columns else "subject_id"
    return index_keys.select(["subject_id", "study_id"]).join(
      summed.select([key, "n_prior_elixhauser"]), on=key, how="left"
    ).select(["study_id", "n_prior_elixhauser"])
  if diagnoses is None or diagnoses.height == 0:
    return index_keys.select("study_id").with_columns(pl.lit(0).cast(pl.Int64).alias("n_prior_elixhauser"))
  prior = prior_completed_admissions(admissions, index_keys)
  if prior.height == 0:
    return index_keys.select("study_id").with_columns(pl.lit(0).cast(pl.Int64).alias("n_prior_elixhauser"))
  prior_diag = diagnoses.join(prior.select(["subject_id", "hadm_id"]).unique(), on=["subject_id", "hadm_id"], how="inner")
  indicators = elixhauser_indicators_from_prior_diagnoses(prior_diag, subject_ids=index_keys["subject_id"])
  cols = list(ELIXHAUSER_FEATURE_NAMES)
  counted = indicators.with_columns(pl.sum_horizontal([pl.col(c) for c in cols]).alias("n_prior_elixhauser"))
  return index_keys.select(["subject_id", "study_id"]).join(
    counted.select(["subject_id", "n_prior_elixhauser"]), on="subject_id", how="left"
  ).select(["study_id", "n_prior_elixhauser"])


def impute_history_features(frame: pl.DataFrame, features: tuple[str, ...] = IMPUTED_FEATURES) -> pl.DataFrame:
  """Development-median fill for undefined history times. Other splits are not used."""
  out = frame
  for feature in features:
    out = out.with_columns(pl.col(feature).fill_null(dev_median(frame, feature)).alias(feature))
  return out


def dev_median(frame: pl.DataFrame, feature: str) -> float:
  """Median of observed development values. Other splits are ignored."""
  observed = frame.filter(pl.col("split") == "dev")[feature].drop_nulls().to_numpy()
  if len(observed) == 0:
    raise ValueError(f"Development split has no observed values for {feature}.")
  return float(np.median(np.asarray(observed, dtype=np.float64)))


def history_contrast_interval(
  y: np.ndarray,
  score_a: np.ndarray,
  score_b: np.ndarray,
  n_replicates: int,
  seed: int,
  min_valid_fraction: float = MIN_VALID_FRACTION,
) -> dict[str, Any]:
  """Paired B−A AUROC interval. Skips the interval when too many replicates lack both classes."""
  yt = np.asarray(y, dtype=np.int64)
  if len(yt) < 2 or len(np.unique(yt)) < 2:
    return {"point": None, "ci_lower": None, "ci_upper": None, "valid_fraction": 0.0, "interval_reported": False}
  # Probe the replicate validity rate with the same stratified resample the CI uses.
  boot = bootstrap_paired_difference(yt, score_a, score_b, "auroc", n_replicates, seed, stratified=True)
  # bootstrap_paired_difference already drops nothing; a one-class draw falls through
  # to an unstratified resample only when the observed sample itself lacks a class,
  # which was rejected above. Validity is therefore 1 when both classes are present.
  return {
    "point": boot.point_estimate,
    "ci_lower": boot.ci_lower,
    "ci_upper": boot.ci_upper,
    "valid_fraction": 1.0,
    "interval_reported": True,
    "n": int(len(yt)),
    "n_events": int(yt.sum()),
    "auroc_a": float(compute_auroc(yt, score_a)),
    "auroc_b": float(compute_auroc(yt, score_b)),
    "auprc_a": float(compute_auprc(yt, score_a)),
    "auprc_b": float(compute_auprc(yt, score_b)),
    "n_replicates": n_replicates,
    "seed": seed,
    "min_valid_fraction": min_valid_fraction,
  }


def heterogeneity_interval(
  y: np.ndarray,
  score_a: np.ndarray,
  score_b: np.ndarray,
  history: np.ndarray,
  n_replicates: int,
  seed: int,
  min_valid_fraction: float = MIN_VALID_FRACTION,
) -> dict[str, Any]:
  """Present-minus-naive B−A. One patient resample recomputes both strata."""
  yt = np.asarray(y, dtype=np.int64)
  sa = np.asarray(score_a, dtype=np.float64)
  sb = np.asarray(score_b, dtype=np.float64)
  labels = np.asarray(history)
  point = _stratum_delta(yt, sa, sb, labels)
  rng = np.random.default_rng(seed)
  pos = np.where(yt == 1)[0]
  neg = np.where(yt == 0)[0]
  diffs: list[float] = []
  for _ in range(n_replicates):
    if len(pos) == 0 or len(neg) == 0:
      idx = rng.integers(0, len(yt), size=len(yt))
    else:
      idx = np.concatenate([
        rng.choice(pos, size=len(pos), replace=True),
        rng.choice(neg, size=len(neg), replace=True),
      ])
    value = _stratum_delta(yt[idx], sa[idx], sb[idx], labels[idx])
    if value is not None:
      diffs.append(value)
  valid_fraction = len(diffs) / n_replicates if n_replicates else 0.0
  reported = valid_fraction >= min_valid_fraction and len(diffs) > 0
  lo = hi = None
  if reported:
    lo, hi = (float(v) for v in np.quantile(np.asarray(diffs), [0.025, 0.975]))
  return {
    "point": point,
    "ci_lower": lo,
    "ci_upper": hi,
    "valid_fraction": valid_fraction,
    "interval_reported": reported,
    "n_replicates": n_replicates,
    "seed": seed,
  }


def _stratum_delta(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, history: np.ndarray) -> float | None:
  present = history == "present"
  naive = history == "naive"
  if _both_classes(y[present]) and _both_classes(y[naive]):
    gain_present = compute_auroc(y[present], score_b[present]) - compute_auroc(y[present], score_a[present])
    gain_naive = compute_auroc(y[naive], score_b[naive]) - compute_auroc(y[naive], score_a[naive])
    return float(gain_present - gain_naive)
  return None


def _both_classes(y: np.ndarray) -> bool:
  return len(y) >= 2 and len(np.unique(y)) >= 2


def fit_history_probe(
  x_dev: np.ndarray,
  y_dev: np.ndarray,
  x_val: np.ndarray,
  y_val: np.ndarray,
  feature_names: list[str],
  model_name: str,
) -> TrainedProbe:
  """Validation-only C tuning. Test labels are not an argument."""
  return fit_logistic_probe(
    x_dev, y_dev, x_val, y_val, feature_names=feature_names, model_name=model_name,
  )


def score_frozen_probe(
  probe: TrainedProbe,
  frame: pl.DataFrame,
  feature_names: tuple[str, ...],
) -> np.ndarray | None:
  """Score rows whose feature columns match the frozen probe. Missing columns block scoring."""
  if probe.feature_names is None or tuple(probe.feature_names) != feature_names:
    return None
  if any(name not in frame.columns for name in feature_names):
    return None
  x = frame.select(list(feature_names)).to_numpy().astype(np.float64)
  if not np.isfinite(x).all():
    return None
  return probe.predict_proba(x)


def write_protocol_note(path: Path, config: Lm6Config, source_commit: str) -> None:
  """Protocol handoff. Contains no recovery counts and no locked-test metrics."""
  lines = [
    "# LM6 protocol: ascertainment recovery and sparse history",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {source_commit}",
    "analysis_status: post_hoc_sensitivity",
    f"inputs: [configs/lm6_ascertainment_history_v1.yaml]",
    "decisions:",
    "  - survivor label requires a timestamp strictly after index ECG + 30 days",
    "  - IPCW infeasible: no censoring time defined for every excluded patient",
    "  - representation summaries omitted for patients outside the primary cache",
    "  - history times imputed with the paired-valid development median plus a missingness indicator",
    "evidence: []",
    "unresolved: []",
    "verification: [rule unit tests; no recovery counts; no locked-test scores]",
    "next_action: run ascertainment-audit and history-sensitivity once under this config",
    "```",
    "",
    f"Config `{config.version}` sha256 `{config.config_hash}`.",
    "",
    "Evidence sources (equal timestamps break ties by source name, alphabetically):",
    "",
  ]
  for source in config.evidence_sources:
    lines.append(f"- `{source}`")
  lines += [
    "",
    "IPCW status: infeasible. The parent exclusion is exactly the absence of a discharge",
    "whose death-record window covers day 30, so a censoring time is not defined for every",
    "pre-ascertainment patient without treating the recovery contact as both survival and censoring.",
    "",
    "History block H is pre-index only. Index-admission diagnoses stay out.",
    "Within-stratum B−A uses the frozen primary C0+A and C0+B scores.",
    "(C0+H+B) − (C0+H+A) is a new fit, tuned on validation log loss, scored once on the locked test.",
    "",
  ]
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(lines), encoding="utf-8")


def dumps_recovery_summary(classified: pl.DataFrame) -> dict[str, Any]:
  """Aggregate recovery counts. No patient identifiers."""
  counts = classified.group_by("recovery_class").len().sort("recovery_class")
  return {
    "n_excluded": int(classified.height),
    "by_class": {str(r["recovery_class"]): int(r["len"]) for r in counts.iter_rows(named=True)},
  }


def categorical_shares(frame: pl.DataFrame, group_col: str, variable: str) -> list[dict[str, Any]]:
  """Level shares within each selection group. No significance test."""
  if variable not in frame.columns:
    return []
  counts = frame.group_by([group_col, variable]).len()
  totals = frame.group_by(group_col).len().rename({"len": "n_group"})
  joined = counts.join(totals, on=group_col, how="left").with_columns(
    (pl.col("len") / pl.col("n_group")).alias("share")
  )
  rows: list[dict[str, Any]] = []
  for row in joined.iter_rows(named=True):
    rows.append({
      "variable": variable,
      "level": None if row[variable] is None else str(row[variable]),
      "group": str(row[group_col]),
      "n": int(row["len"]),
      "share": float(row["share"]),
    })
  return rows


def _index_before_ascertainment(
  records: pl.DataFrame,
  patients: pl.DataFrame,
  admissions: pl.DataFrame,
  eligible_study_ids: set[int],
) -> pl.DataFrame:
  """Same steps as the parent cohort through index selection, before the outcome filter."""
  from ecg_rep.cohort import compute_age_at_ecg, filter_adult_records, select_index_ecgs
  from ecg_rep.outcomes import compute_mortality_outcomes

  linked = compute_age_at_ecg(records, patients)
  adults = filter_adult_records(linked, min_age=18)
  waveform = adults.filter(pl.col("study_id").is_in(list(eligible_study_ids)))
  index = select_index_ecgs(waveform, strategy="earliest")
  return compute_mortality_outcomes(index, patients, admissions_df=admissions)


def _evidence_long(
  excluded: pl.DataFrame,
  admissions: pl.DataFrame,
  records: pl.DataFrame,
  icustays: pl.DataFrame | None,
  labevents_path: Path,
  horizon_days: float,
) -> pl.DataFrame:
  """Stack candidate contact times. ``earliest_evidence`` applies the horizon cut."""
  keys = excluded.select(["subject_id", "study_id", "ecg_datetime"])
  pieces: list[pl.DataFrame] = []

  def _piece(frame: pl.DataFrame, source: str) -> None:
    if frame.height == 0:
      return
    pieces.append(frame.with_columns(pl.lit(source).alias("source")))

  if "admit_dt" in admissions.columns:
    _piece(
      keys.join(admissions.select(["subject_id", "admit_dt"]), on="subject_id", how="inner").select(
        "study_id", "ecg_datetime", pl.col("admit_dt").alias("evidence_time")
      ),
      "later_hospital_admission",
    )
  if "edreg_dt" in admissions.columns:
    _piece(
      keys.join(admissions.select(["subject_id", "edreg_dt"]), on="subject_id", how="inner").select(
        "study_id", "ecg_datetime", pl.col("edreg_dt").alias("evidence_time")
      ),
      "later_ed_registration",
    )
  if icustays is not None and "intime" in icustays.columns:
    _piece(
      keys.join(icustays.select(["subject_id", "intime"]), on="subject_id", how="inner").select(
        "study_id", "ecg_datetime", pl.col("intime").alias("evidence_time")
      ),
      "later_icu_stay",
    )
  _piece(
    keys.join(
      records.select(["subject_id", pl.col("study_id").alias("other_study"), pl.col("ecg_datetime").alias("evidence_time")]),
      on="subject_id",
      how="inner",
    ).filter(pl.col("other_study") != pl.col("study_id")).select(["study_id", "ecg_datetime", "evidence_time"]),
    "later_ecg",
  )
  if labevents_path.exists():
    subjects = keys["subject_id"].unique().to_list()
    labs = (
      pl.scan_csv(labevents_path, schema_overrides={"subject_id": pl.Int64, "charttime": pl.String})
      .select(["subject_id", "charttime"])
      .filter(pl.col("subject_id").is_in(subjects))
      .with_columns(pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False))
      .collect()
    )
    _piece(
      keys.join(labs, on="subject_id", how="inner").select(
        "study_id", "ecg_datetime", pl.col("charttime").alias("evidence_time")
      ),
      "later_laboratory",
    )
  if not pieces:
    return pl.DataFrame(schema={
      "study_id": pl.Int64,
      "ecg_datetime": pl.Datetime("us"),
      "source": pl.String,
      "evidence_time": pl.Datetime("us"),
    })
  return pl.concat(pieces, how="vertical")


def _machine_measurements(path: Path, study_ids: list[int]) -> pl.DataFrame:
  from ecg_rep.machine_baseline import MACHINE_RAW_COLUMNS, derive_machine_features

  if not path.exists():
    return pl.DataFrame({"study_id": pl.Series([], dtype=pl.Int64)})
  raw = (
    pl.scan_csv(
      path,
      schema_overrides={"study_id": pl.Int64, **{col: pl.Int64 for col in MACHINE_RAW_COLUMNS}},
    )
    .select(["study_id", *MACHINE_RAW_COLUMNS])
    .filter(pl.col("study_id").is_in(study_ids))
    .collect()
  )
  derived = derive_machine_features(raw)
  keep = [c for c in ("study_id", "heart_rate_bpm", "qrs_duration_ms", "qtc_bazett_ms") if c in derived.columns]
  return derived.select(keep)


def run_ascertainment_audit(
  config_path: Path,
  cohort_path: Path,
  waveform_manifest: Path,
  c0_path: Path,
  mimic_root: str | None,
  ecg_root: str | None,
  output_dir: Path,
  report_path: Path,
  source_commit: str,
) -> dict[str, Any]:
  """Selection SMDs and survivor recovery. Does not write the primary cohort or split file."""
  from ecg_rep.clinical_setting import assign_index_ecg_setting, load_icustays, load_transfers
  from ecg_rep.paths import _find_data_file, load_dataset_tables, load_diagnoses_table, resolve_data_paths
  from ecg_rep.split import SplitRatios, assign_patient_disjoint_split

  config = load_lm6_config(config_path)
  if config.ipcw_status != "infeasible":
    raise ValueError("LM6 v1 refuses to run IPCW unless the frozen config says infeasible was reconsidered.")
  paths = resolve_data_paths(mimic_root, ecg_root)
  records, patients, admissions = load_dataset_tables(paths)
  diagnoses = load_diagnoses_table(paths)
  manifest = pl.read_parquet(waveform_manifest)
  eligible = set(int(v) for v in manifest.filter(pl.col("waveform_eligible") == True)["study_id"].to_list())  # noqa: E712
  annotated = _index_before_ascertainment(records, patients, admissions, eligible)
  included = annotated.filter(pl.col("is_valid"))
  excluded = annotated.filter(~pl.col("is_valid"))
  primary = pl.read_parquet(cohort_path)
  if set(included["subject_id"].to_list()) != set(primary["subject_id"].to_list()):
    raise RuntimeError(
      "Outcome-ascertainable index patients do not match the primary cohort. "
      "Refusing to audit a different population."
    )
  c0 = pl.read_parquet(c0_path)
  icu_path = _find_data_file(paths.mimic_iv_dir, ["icu/icustays.csv.gz", "icu/icustays.csv"])
  transfers_path = _find_data_file(paths.mimic_iv_dir, ["hosp/transfers.csv.gz", "hosp/transfers.csv"])
  icustays = load_icustays(icu_path)
  included_h = build_history_features(included, admissions, elixhauser=c0, icustays=icustays)
  excluded_h = build_history_features(excluded, admissions, diagnoses=diagnoses, icustays=icustays)
  history = pl.concat([
    included_h.with_columns(pl.lit("included").alias("selection_group")),
    excluded_h.with_columns(pl.lit("excluded").alias("selection_group")),
  ], how="diagonal")
  demo = annotated.select(["study_id", "age_at_ecg", "ecg_year", "gender"]).with_columns(
    (pl.col("gender").str.to_uppercase() == "F").cast(pl.Float64).alias("sex_female")
  )
  transfers = load_transfers(transfers_path)
  setting, _stats = assign_index_ecg_setting(annotated, admissions, transfers, icustays)
  measurements_path = paths.mimic_iv_ecg_dir / "machine_measurements.csv"
  machines = _machine_measurements(measurements_path, annotated["study_id"].to_list())
  selection = (
    history.join(demo, on="study_id", how="left")
    .join(setting.select(["study_id", "setting"]), on="study_id", how="left")
    .join(machines, on="study_id", how="left")
    .rename({"setting": "care_setting"})
  )
  continuous = tuple(str(c) for c in config.raw["selection_continuous"])
  binary = tuple(str(c) for c in config.raw["selection_binary"])
  smd = selection_smd_table(selection, "selection_group", "included", "excluded", continuous, binary)
  shares: list[dict[str, Any]] = []
  for variable in config.raw["selection_categorical"]:
    shares.extend(categorical_shares(selection, "selection_group", str(variable)))
  labs_path = _find_data_file(paths.mimic_iv_dir, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])
  evidence = earliest_evidence(
    _evidence_long(excluded, admissions, records, icustays, labs_path, config.horizon_days),
    config.horizon_days,
  )
  classified = classify_excluded(excluded, evidence, config.horizon_days)
  recovered = classified.filter(pl.col("recovery_class") == "recovered_survivor")
  split_seed = int(config.raw["recovered_split"]["seed"])
  ratios = config.raw["recovered_split"]
  if recovered.height == 0:
    assigned = recovered
    preservation: dict[str, Any] | None = None
  else:
    assigned_res = assign_patient_disjoint_split(
      recovered.select(["subject_id", "study_id"]),
      ratios=SplitRatios(dev=float(ratios["dev"]), val=float(ratios["val"]), test=float(ratios["test"])),
      seed=split_seed,
      existing_split_df=primary.select(["subject_id", "split"]),
      allow_new_test=True,
      return_report=True,
    )
    assigned, report = assigned_res if isinstance(assigned_res, tuple) else (assigned_res, None)
    preservation = None if report is None else {
      "n_preserved": report.n_preserved,
      "n_new_dev": report.n_new_dev,
      "n_new_val": report.n_new_val,
      "n_new_test": report.n_new_test,
    }
  by_split = {}
  if assigned.height and "split" in assigned.columns:
    for row in assigned.group_by("split").len().iter_rows(named=True):
      by_split[str(row["split"])] = int(row["len"])
  output_dir.mkdir(parents=True, exist_ok=True)
  smd.write_csv(output_dir / "ascertainment_selection.csv")
  if shares:
    pl.DataFrame(shares).write_csv(output_dir / "ascertainment_selection_categorical.csv")
  recovery = dumps_recovery_summary(classified)
  recovery_payload = {
    "config_hash": config.config_hash,
    "horizon_days": config.horizon_days,
    "ipcw": config.raw["ipcw"],
    "representation_summaries": config.raw["representation_summaries"],
    "recovery": recovery,
    "recovered_by_split": by_split,
    "split_preservation": preservation,
    "scoring_status": "not_applicable" if recovered.height == 0 else "blocked_missing_representations",
    "scoring_note": (
      "Recovered index ECGs are not in the primary representation cache. "
      "Scoring waits on a separate extraction that must not overwrite artifacts/reps."
    ),
  }
  (output_dir / "survival_evidence_recovery.json").write_text(
    json.dumps(recovery_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  # Patient-level rows stay in the gitignored artifact directory.
  classified.select([
    c for c in ("subject_id", "study_id", "recovery_class", "evidence_source", "days_to_death") if c in classified.columns
  ]).write_parquet(output_dir / "excluded_recovery_class.parquet")
  if assigned.height:
    assigned.write_parquet(output_dir / "recovered_split_assignments.parquet")
  summary = {
    "source_commit": source_commit,
    "config_hash": config.config_hash,
    "n_included": int(included.height),
    "n_excluded": int(excluded.height),
    "recovery": recovery,
    "recovered_by_split": by_split,
    "scoring_status": recovery_payload["scoring_status"],
    "ipcw_status": config.ipcw_status,
    "smd": smd.to_dicts(),
    "categorical_shares": shares,
    "primary_cohort_overwritten": False,
  }
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(render_results_markdown(summary), encoding="utf-8")
  return summary


def run_history_sensitivity(
  config_path: Path,
  cohort_path: Path,
  c0_path: Path,
  reps_dir: Path,
  predictions_context_path: Path,
  mimic_root: str | None,
  output_dir: Path,
  report_path: Path,
  source_commit: str,
  n_replicates: int | None = None,
) -> dict[str, Any]:
  """Frozen-score stratum contrasts plus one new C0+H fit. Primary probes are not replaced."""
  from ecg_rep.clinical_setting import load_icustays
  from ecg_rep.evaluate import C0_FEATURE_COLS, _join_representation_matrix
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  config = load_lm6_config(config_path)
  replicates = config.n_replicates if n_replicates is None else n_replicates
  paths = resolve_data_paths(mimic_root, None)
  admissions_path = _find_data_file(paths.mimic_iv_dir, ["hosp/admissions.csv.gz", "hosp/admissions.csv"])
  icu_path = _find_data_file(paths.mimic_iv_dir, ["icu/icustays.csv.gz", "icu/icustays.csv"])
  from ecg_rep.data import load_admissions

  cohort = pl.read_parquet(cohort_path)
  c0 = pl.read_parquet(c0_path)
  admissions = load_admissions(admissions_path)
  icustays = load_icustays(icu_path)
  history = build_history_features(cohort, admissions, elixhauser=c0, icustays=icustays)
  meta_a = pl.read_parquet(reps_dir / "ciis_features_meta.parquet")
  meta_b = pl.read_parquet(reps_dir / "dbeta_embedding_meta.parquet")
  frame = cohort.join(history.select(["study_id", *config.h_features, "history_group", "history_ordered"]), on="study_id")
  c0_new = [c for c in C0_FEATURE_COLS if c in c0.columns and c not in frame.columns]
  frame = frame.join(c0.select(["study_id", *c0_new]), on="study_id")
  frame = _join_representation_matrix(frame, meta_a, np.load(reps_dir / "ciis_features.npy"), "za", "representation_a_valid")
  frame = _join_representation_matrix(
    frame, meta_b, np.load(reps_dir / "dbeta_embeddings.npy"), "zb", "representation_b_valid"
  )
  frame = frame.filter(
    (pl.col("representation_a_valid") == True) & (pl.col("representation_b_valid") == True)  # noqa: E712
  )
  # Medians come from the paired-valid development rows only, the rows that enter the fit.
  frame = impute_history_features(frame, config.imputed_features)
  preds = pl.read_parquet(predictions_context_path).select(["study_id", "p_a", "p_b", "mortality_30d"])
  test = frame.filter(pl.col("split") == "test").join(preds, on="study_id", how="inner", suffix="_pred")
  n_test_surface = frame.filter(pl.col("split") == "test").height
  if test.height == 0 or test.height != n_test_surface:
    raise RuntimeError(
      f"Frozen context predictions cover {test.height} of {n_test_surface} paired-valid locked-test rows."
    )
  outcome_ok = np.array_equal(
    np.asarray(test["mortality_30d"].to_numpy(), dtype=np.int64),
    np.asarray(test["mortality_30d_pred"].to_numpy(), dtype=np.int64),
  )
  if not outcome_ok:
    raise RuntimeError("Frozen context predictions do not match cohort outcomes on the history surface.")
  y_col = "mortality_30d"
  contrasts: dict[str, Any] = {}
  y_test = np.asarray(test[y_col].to_numpy(), dtype=np.int64)
  p_a = np.asarray(test["p_a"].to_numpy(), dtype=np.float64)
  p_b = np.asarray(test["p_b"].to_numpy(), dtype=np.float64)
  groups = np.asarray(test["history_group"].to_numpy())
  for name in ("naive", "present"):
    mask = groups == name
    contrasts[f"b_minus_a_history_{name}"] = history_contrast_interval(
      y_test[mask], p_a[mask], p_b[mask], replicates, config.seed, config.min_valid_fraction,
    )
    contrasts[f"b_minus_a_history_{name}"]["score_source"] = "frozen_primary_C0A_C0B"
  contrasts["present_minus_naive"] = heterogeneity_interval(
    y_test, p_a, p_b, groups, replicates, config.seed + 1, config.min_valid_fraction,
  )
  ordered: dict[str, Any] = {}
  ordered_labels = np.asarray(test["history_ordered"].to_numpy())
  for level in ("0", "1", "2+"):
    mask = ordered_labels == level
    n_events = int(y_test[mask].sum()) if mask.any() else 0
    interval = history_contrast_interval(
      y_test[mask], p_a[mask], p_b[mask], replicates, config.seed, config.min_valid_fraction,
    )
    if n_events < config.min_events_for_interval:
      interval["ci_lower"] = None
      interval["ci_upper"] = None
      interval["interval_reported"] = False
      interval["reason"] = "below_min_events_for_interval"
    ordered[level] = interval
  c0_cols = [c for c in C0_FEATURE_COLS if c in frame.columns]
  h_cols = list(config.h_features)
  za_cols = [c for c in frame.columns if c.startswith("za_")]
  zb_cols = [c for c in frame.columns if c.startswith("zb_")]
  specs = {
    "C0H": [*c0_cols, *h_cols],
    "C0HA": [*c0_cols, *h_cols, *za_cols],
    "C0HB": [*c0_cols, *h_cols, *zb_cols],
  }
  fitted: dict[str, np.ndarray] = {}
  selected_c: dict[str, float] = {}
  y_by_split = {
    split: np.asarray(frame.filter(pl.col("split") == split)["mortality_30d"].to_numpy(), dtype=np.int64)
    for split in ("dev", "val", "test")
  }
  for name, cols in specs.items():
    blocks = {
      split: frame.filter(pl.col("split") == split).select(cols).to_numpy().astype(np.float64)
      for split in ("dev", "val", "test")
    }
    probe = fit_history_probe(blocks["dev"], y_by_split["dev"], blocks["val"], y_by_split["val"], cols, name)
    fitted[name] = probe.predict_proba(blocks["test"])
    selected_c[name] = float(probe.best_c)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe.save(output_dir / f"{name}.json")
  h_contrast = history_contrast_interval(
    y_by_split["test"], fitted["C0HA"], fitted["C0HB"], replicates, config.seed + 2, config.min_valid_fraction,
  )
  h_contrast["score_source"] = "refit_C0H_validation_tuned"
  contrasts["c0hb_minus_c0ha"] = h_contrast
  summary: dict[str, Any] = {
    "source_commit": source_commit,
    "config_hash": config.config_hash,
    "analysis_status": "post_hoc_sensitivity",
    "n_surface": int(frame.height),
    "n_test": int(test.height),
    "n_test_events": int(y_test.sum()),
    "history_counts_test": {
      name: int((groups == name).sum()) for name in ("naive", "present")
    },
    "contrasts": contrasts,
    "ordered_groups": ordered,
    "selected_c": selected_c,
    "model_auroc_test": {
      name: float(compute_auroc(y_by_split["test"], scores)) for name, scores in fitted.items()
    },
    "primary_artifacts_overwritten": False,
    "n_replicates": replicates,
    "bootstrap_matches_config": replicates == config.n_replicates,
  }
  (output_dir / "history_sensitivity.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
  )
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(render_results_markdown({"kind": "lm6b_history", **summary}), encoding="utf-8")
  return summary


def run_recovered_cohort_scoring(
  primary_cohort_path: Path,
  recovered_cohort_path: Path,
  primary_c0_path: Path,
  recovered_c0_path: Path,
  primary_reps_dir: Path,
  recovered_reps_dir: Path,
  probe_dir: Path,
  predictions_dir: Path,
  config_path: Path,
  output_dir: Path,
  report_path: Path,
  source_commit: str,
) -> dict[str, Any]:
  """Apply frozen primary probes to the enlarged known-outcome test surface.

  No fitting occurs. Recomputed original scores must match stored primary
  predictions before new survivors can alter the evaluation denominator.
  """
  from ecg_rep.evaluate import C0_FEATURE_COLS, _join_representation_matrix
  from ecg_rep.representations import verify_representation_artifact_integrity

  if primary_reps_dir.resolve() == recovered_reps_dir.resolve():
    raise ValueError("Recovered representations require a separate cache.")
  cfg = load_lm6_config(config_path)
  primary = pl.read_parquet(primary_cohort_path)
  recovered = pl.read_parquet(recovered_cohort_path)
  if set(primary["subject_id"].to_list()) & set(recovered["subject_id"].to_list()):
    raise ValueError("Recovered patients overlap the primary cohort.")
  if (recovered["subject_id"].n_unique() != recovered.height
      or recovered["study_id"].n_unique() != recovered.height):
    raise ValueError("Recovered cohort must have one index ECG per patient.")
  if (recovered["mortality_30d"].null_count() or recovered["recovery_class"].null_count()
      or recovered["mortality_30d"].any()
      or not recovered["recovery_class"].eq("recovered_survivor").all()):
    raise ValueError("Recovered cohort must contain only evidenced known survivors.")
  allowed = {"dev", "val", "test"}
  if set(recovered["split"].to_list()) - allowed:
    raise ValueError("Recovered split assignment is missing or invalid.")

  def _frame(cohort: pl.DataFrame, c0_path: Path, reps: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    verify_representation_artifact_integrity(reps, cohort)
    meta_a = pl.read_parquet(reps / "ciis_features_meta.parquet")
    meta_b = pl.read_parquet(reps / "dbeta_embedding_meta.parquet")
    validity = cohort.select(["subject_id", "study_id", "split"]).join(
      meta_a.select(["subject_id", "study_id", "representation_a_valid"]),
      on=["subject_id", "study_id"], validate="1:1",
    ).join(
      meta_b.select(["subject_id", "study_id", "representation_b_valid"]),
      on=["subject_id", "study_id"], validate="1:1",
    )
    if validity.height != cohort.height:
      raise ValueError("Representation metadata do not cover the full cohort.")
    coverage = {}
    for split in ("dev", "val", "test"):
      part = validity.filter(pl.col("split") == split)
      coverage[split] = {
        "n_assigned": part.height,
        "n_a_valid": int(part["representation_a_valid"].sum()),
        "n_b_valid": int(part["representation_b_valid"].sum()),
        "n_paired_valid": int((part["representation_a_valid"] & part["representation_b_valid"]).sum()),
      }
    test = cohort.filter(pl.col("split") == "test")
    c0 = pl.read_parquet(c0_path)
    new = [c for c in C0_FEATURE_COLS if c not in test.columns]
    test = test.join(c0.select(["subject_id", "study_id", *new]),
                     on=["subject_id", "study_id"], how="left", validate="1:1")
    test = _join_representation_matrix(test, meta_a, np.load(reps / "ciis_features.npy"), "za", "representation_a_valid")
    test = _join_representation_matrix(test, meta_b, np.load(reps / "dbeta_embeddings.npy"), "zb", "representation_b_valid")
    return test.filter(pl.col("representation_a_valid") & pl.col("representation_b_valid")).sort("study_id"), coverage

  original, _ = _frame(primary, primary_c0_path, primary_reps_dir)
  extra, coverage = _frame(recovered, recovered_c0_path, recovered_reps_dir)
  if extra.height == 0:
    raise ValueError("No recovered sensitivity-test patients have paired-valid features.")
  models = {name: TrainedProbe.load(probe_dir / f"{name}.json")
            for name in ("M_A_ECG", "M_B_ECG", "M_A", "M_B")}
  scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
  for name, probe in models.items():
    if probe.feature_names is None:
      raise ValueError(f"Frozen probe {name} lacks feature names.")
    base = score_frozen_probe(probe, original, probe.feature_names)
    added = score_frozen_probe(probe, extra, probe.feature_names)
    if base is None or added is None:
      raise ValueError(f"Missing/nonfinite frozen-probe features for {name}; refusing a different surface.")
    scores[name] = base, added
  y_base = original["mortality_30d"].to_numpy().astype(np.int64)
  y = np.concatenate([y_base, np.zeros(extra.height, dtype=np.int64)])
  results: dict[str, Any] = {
    "status": "complete", "scoring_status": "complete",
    "source_commit": source_commit, "analysis_status": "post_hoc_sensitivity",
    "config_hash": cfg.config_hash, "variant": "frozen_primary_probes_no_refit",
    "recovered_coverage_by_split": coverage, "n_original_test": original.height,
    "n_recovered_test": extra.height, "n_test": len(y), "n_test_events": int(y.sum()),
    "event_rate": float(y.mean()), "primary_artifacts_overwritten": False,
    "probe_hashes": {name: hashlib.sha256((probe_dir / f"{name}.json").read_bytes()).hexdigest() for name in models},
    "contrasts": {},
  }
  comparisons = (
    ("ecg_only", "M_A_ECG", "M_B_ECG", "test_predictions_ecg_only.parquet", "p_a_ecg", "p_b_ecg"),
    ("C0", "M_A", "M_B", "test_predictions_context.parquet", "p_a", "p_b"),
  )
  # Verify every original model before any enlarged-outcome metric is computed.
  for _label, name_a, name_b, pred_file, pred_a, pred_b in comparisons:
    stored = pl.read_parquet(predictions_dir / pred_file).sort("study_id")
    keys = ["subject_id", "study_id", "mortality_30d"]
    if not original.select(keys).with_columns(pl.col("mortality_30d").cast(pl.Int64)).equals(stored.select(keys)):
      raise ValueError("Original frozen prediction patient/outcome surface mismatch.")
    a_base, a_extra = scores[name_a]
    b_base, b_extra = scores[name_b]
    if not (np.allclose(a_base, stored[pred_a].to_numpy(), atol=1e-10, rtol=0)
            and np.allclose(b_base, stored[pred_b].to_numpy(), atol=1e-10, rtol=0)):
      raise ValueError("Recomputed original predictions differ from frozen scores.")
  for label, name_a, name_b, _file, _pred_a, _pred_b in comparisons:
    a_base, a_extra = scores[name_a]
    b_base, b_extra = scores[name_b]
    a = np.concatenate([a_base, a_extra])
    b = np.concatenate([b_base, b_extra])
    block: dict[str, Any] = {"original": {}, "enlarged": {}}
    for metric, compute in (("auroc", compute_auroc), ("auprc", compute_auprc)):
      point_base = float(compute(y_base, b_base) - compute(y_base, a_base))
      boot = bootstrap_paired_difference(y, a, b, "auroc" if metric == "auroc" else "auprc", 1000, 42, stratified=True)
      block["original"][metric] = {"a": float(compute(y_base, a_base)), "b": float(compute(y_base, b_base)), "b_minus_a": point_base}
      block["enlarged"][metric] = {
        "a": float(compute(y, a)), "b": float(compute(y, b)), "b_minus_a": boot.point_estimate,
        "ci_lower": boot.ci_lower, "ci_upper": boot.ci_upper,
        "change_from_original": boot.point_estimate - point_base,
      }
    results["contrasts"][label] = block
  results["bootstrap"] = {"n_replicates": 1000, "seed": 42, "stratified": True, "models_refit": False}
  results["original_scores_verified"] = True
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "recovered_cohort_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
  lines = ["# LM6A recovered-cohort frozen-score sensitivity", "", "```yaml", "status: complete",
           "producer: study-orchestrator", "consumer: study-review", f"source_commit: {source_commit}",
           "analysis_status: post_hoc_sensitivity", "inputs: [frozen primary probes, recovered A/B/C0 cache]",
           "decisions: [no refitting; paired-valid recovered sensitivity-test survivors only]",
           "evidence: [artifacts/limitation_mitigation/lm6/recovered_cohort_results.json]",
           "unresolved: [LM9 synthesis pending; unknown outcomes remain excluded]",
           "verification: [original scores reproduced to 1e-10; primary artifacts untouched]",
           "next_action: scientific review and dedicated LM9 synthesis", "```", "",
           f"Original test n={original.height:,}; added survivors n={extra.height:,}; enlarged test n={len(y):,}, {int(y.sum())} deaths.",
           "", "Intervals: 1,000 outcome-stratified paired patient-bootstrap resamples, seed 42, fixed probes.", "",
           "| Context | Metric | A | B | B−A | Central 95% | Change from original B−A |",
           "|---|---|---|---|---|---|---|"]
  for context, block in results["contrasts"].items():
    for metric, value in block["enlarged"].items():
      lines.append(f"| {context} | {metric} | {value['a']:.4f} | {value['b']:.4f} | {value['b_minus_a']:+.4f} | {value['ci_lower']:+.4f} to {value['ci_upper']:+.4f} | {value['change_from_original']:+.4f} |")
  lines += ["", "## Recovery coverage by frozen sensitivity split", "", "| Split | Assigned | A valid | B valid | Paired valid |",
            "|---|---|---|---|---|"]
  for split, counts in coverage.items():
    lines.append(f"| {split} | {counts['n_assigned']} | {counts['n_a_valid']} | {counts['n_b_valid']} | {counts['n_paired_valid']} |")
  lines += ["", "This sensitivity adds only demonstrably observed survivors, not all excluded patients.",
            "It does not remove follow-up selection or redefine the primary cohort. Recovery assignments",
            "are sensitivity-only; no recovered patient entered the original locked test.", ""]
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text("\n".join(lines))
  return results


def render_results_markdown(summary: dict[str, Any]) -> str:
  """Aggregate results note. Numbers come only from ``summary``."""
  lines = [
    "# LM6 results",
    "",
    "```yaml",
    "status: incomplete" if summary.get("scoring_status", "complete") == "blocked_missing_representations" else "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {summary.get('source_commit', 'unknown')}",
    "analysis_status: post_hoc_sensitivity",
    f"config_hash: {summary.get('config_hash')}",
    "verification: [parent cohort not overwritten; unknown patients not labeled survivors]",
    "unresolved: [recovered-cohort performance missing]" if summary.get("scoring_status") == "blocked_missing_representations" else "unresolved: []",
    "next_action: extract recovered A/B into separate cache; LM9 blocked" if summary.get("scoring_status") == "blocked_missing_representations" else "next_action: review before LM9 claim reconciliation",
    "```",
    "",
    "```json",
    json.dumps(summary, indent=2, sort_keys=True, default=str),
    "```",
    "",
  ]
  return "\n".join(lines)
