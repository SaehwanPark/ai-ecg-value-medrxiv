"""LM7: simultaneous bands and max-T pairwise intervals for existing subgroups.

Reporting sensitivity only. Levels, scores, and point estimates are the
prespecified context-adjusted B−A contrasts. No new subgroup is created.
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

from ecg_rep.analysis import (
  DEFAULT_AGE_BINS,
  assign_age_subgroup,
  representation_gain,
  subject_race_from_admissions,
)
from ecg_rep.probe import compute_auroc

SEX_LEVELS: tuple[str, ...] = ("F", "M")
AGE_LEVELS: tuple[str, ...] = tuple(name for _lo, _hi, name in DEFAULT_AGE_BINS)
RACE_LEVELS: tuple[str, ...] = ("asian", "black", "hispanic_latino", "unknown_other", "white")
AGE_REFERENCE = "65-79"
MIN_VALID_FRACTION = 0.90
MIN_PAIR_EVENTS = 11
N_REPLICATES = 1000
SEED = 72026


@dataclass(frozen=True)
class Lm7Config:
  version: str
  n_replicates: int
  seed: int
  min_valid_fraction: float
  age_reference: str
  min_pair_events: int
  config_hash: str


def load_lm7_config(path: Path) -> Lm7Config:
  text = path.read_text(encoding="utf-8")
  raw = yaml.safe_load(text)
  if not isinstance(raw, dict):
    raise ValueError(f"LM7 config at {path} is not a mapping.")
  if tuple(raw["levels"]["sex"]) != SEX_LEVELS:
    raise ValueError("LM7 sex levels do not match the prespecified set.")
  if tuple(raw["levels"]["age"]) != AGE_LEVELS:
    raise ValueError("LM7 age levels do not match the prespecified bins.")
  if tuple(raw["levels"]["race_ethnicity"]) != RACE_LEVELS:
    raise ValueError("LM7 race/ethnicity levels do not match the prespecified set.")
  return Lm7Config(
    version=str(raw["version"]),
    n_replicates=int(raw["bootstrap"]["n_replicates"]),
    seed=int(raw["bootstrap"]["seed"]),
    min_valid_fraction=float(raw["bootstrap"]["min_valid_fraction"]),
    age_reference=str(raw["pairwise"]["age"]["reference"]),
    min_pair_events=int(raw["pairwise"]["race_ethnicity"]["min_events_each_level"]),
    config_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
  )


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  """Same outcome-stratified patient resample as the parent subgroup bootstrap."""
  pos = np.where(y == 1)[0]
  neg = np.where(y == 0)[0]
  if len(pos) == 0 or len(neg) == 0:
    return rng.integers(0, len(y), size=len(y))
  return np.concatenate([
    rng.choice(pos, size=len(pos), replace=True),
    rng.choice(neg, size=len(neg), replace=True),
  ])


def level_gain(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, labels: np.ndarray, level: str) -> float:
  """B−A AUROC on one level. NaN when the level does not contain both classes."""
  mask = labels == level
  y_level = y[mask]
  if len(y_level) < 2 or len(np.unique(y_level)) < 2:
    return float("nan")
  return float(representation_gain(y_level, score_a[mask], score_b[mask], metric="auroc").gain)


def bootstrap_level_estimates(
  y: np.ndarray,
  score_a: np.ndarray,
  score_b: np.ndarray,
  labels: np.ndarray,
  levels: tuple[str, ...],
  n_replicates: int,
  seed: int,
) -> dict[str, np.ndarray]:
  """One shared stratified resample per replicate, then a gain for every level."""
  yt = np.asarray(y, dtype=np.int64)
  sa = np.asarray(score_a, dtype=np.float64)
  sb = np.asarray(score_b, dtype=np.float64)
  lab = np.asarray(labels)
  out = {level: np.full(n_replicates, np.nan) for level in levels}
  rng = np.random.default_rng(seed)
  for i in range(n_replicates):
    idx = _stratified_indices(yt, rng)
    y_b, a_b, b_b, lab_b = yt[idx], sa[idx], sb[idx], lab[idx]
    for level in levels:
      mask = lab_b == level
      y_level = y_b[mask]
      if len(y_level) < 2 or len(np.unique(y_level)) < 2:
        continue
      out[level][i] = float(compute_auroc(y_level, b_b[mask]) - compute_auroc(y_level, a_b[mask]))
  return out


def simultaneous_bands(
  levels: tuple[str, ...],
  points: dict[str, float],
  replicates: dict[str, np.ndarray],
  min_valid_fraction: float = MIN_VALID_FRACTION,
) -> list[dict[str, Any]]:
  """Unstandardized simultaneous 95% bands from the max absolute deviation.

  A replicate enters the max statistic only when every band-eligible level is
  valid in that replicate. Levels below the valid-fraction cutoff keep their
  point estimate and receive no band.
  """
  valid_fraction = {
    level: float(np.isfinite(replicates[level]).mean()) if len(replicates[level]) else 0.0
    for level in levels
  }
  eligible = [
    level for level in levels
    if valid_fraction[level] >= min_valid_fraction and np.isfinite(points[level])
  ]
  crit: float | None = None
  n_max = 0
  if eligible:
    n = len(replicates[eligible[0]])
    maxima: list[float] = []
    for i in range(n):
      if not all(np.isfinite(replicates[level][i]) for level in eligible):
        continue
      maxima.append(max(abs(float(replicates[level][i]) - points[level]) for level in eligible))
    n_max = len(maxima)
    if maxima:
      crit = float(np.quantile(np.asarray(maxima), 0.95))
  rows: list[dict[str, Any]] = []
  for level in levels:
    point = points[level]
    band = level in eligible and crit is not None
    rows.append({
      "level": level,
      "point": None if not np.isfinite(point) else float(point),
      "simultaneous_low": None if not band else float(point - crit),  # type: ignore[operator]
      "simultaneous_high": None if not band else float(point + crit),  # type: ignore[operator]
      "valid_fraction": valid_fraction[level],
      "band_reported": band,
      "label": "simultaneous_95" if band else "underpowered",
      "n_max_replicates": n_max,
      "critical_value": crit,
    })
  return rows


def pairwise_contrasts(
  dimension: str,
  event_counts: dict[str, int],
  age_reference: str = AGE_REFERENCE,
  min_pair_events: int = MIN_PAIR_EVENTS,
) -> list[tuple[str, str, str]]:
  """Prespecified pairs. Race pairs require the event minimum on both levels."""
  if dimension == "sex":
    return [("F_minus_M", "F", "M")]
  if dimension == "age":
    return [
      (f"{level}_minus_{age_reference}", level, age_reference)
      for level in AGE_LEVELS
      if level != age_reference
    ]
  if dimension == "race_ethnicity":
    eligible = [level for level in RACE_LEVELS if event_counts.get(level, 0) >= min_pair_events]
    pairs: list[tuple[str, str, str]] = []
    for i, left in enumerate(eligible):
      for right in eligible[i + 1 :]:
        pairs.append((f"{left}_minus_{right}", left, right))
    return pairs
  raise ValueError(f"Unknown subgroup dimension: {dimension}")


def max_t_intervals(
  pairs: list[tuple[str, str, str]],
  points: dict[str, float],
  replicates: dict[str, np.ndarray],
  min_valid_fraction: float = MIN_VALID_FRACTION,
) -> list[dict[str, Any]]:
  """Simultaneous intervals for pairwise gain differences. Not multiplicity-adjusted p-values."""
  if not pairs:
    return []
  n = len(next(iter(replicates.values())))
  contrast_points: dict[str, float] = {}
  contrast_reps: dict[str, np.ndarray] = {}
  for name, left, right in pairs:
    contrast_points[name] = points[left] - points[right]
    contrast_reps[name] = replicates[left] - replicates[right]
  bands = simultaneous_bands(
    tuple(name for name, _left, _right in pairs),
    contrast_points,
    contrast_reps,
    min_valid_fraction,
  )
  by_name = {row["level"]: row for row in bands}
  rows: list[dict[str, Any]] = []
  for name, left, right in pairs:
    band = by_name[name]
    rows.append({
      "contrast": name,
      "level_left": left,
      "level_right": right,
      "point": band["point"],
      "adjusted_low": band["simultaneous_low"],
      "adjusted_high": band["simultaneous_high"],
      "interval_reported": band["band_reported"],
      "valid_fraction": band["valid_fraction"],
      "n_replicates_nominal": n,
    })
  return rows


def omnibus_range(points: dict[str, float], levels: tuple[str, ...]) -> dict[str, Any]:
  """Descriptive range of level estimates. Not a hypothesis test."""
  finite = [points[level] for level in levels if np.isfinite(points[level])]
  if not finite:
    return {"range": None, "min": None, "max": None, "n_levels": 0}
  return {
    "range": float(max(finite) - min(finite)),
    "min": float(min(finite)),
    "max": float(max(finite)),
    "n_levels": len(finite),
  }


def _event_counts(y: np.ndarray, labels: np.ndarray, levels: tuple[str, ...]) -> dict[str, int]:
  return {level: int(y[labels == level].sum()) for level in levels}


def run_dimension(
  y: np.ndarray,
  score_a: np.ndarray,
  score_b: np.ndarray,
  labels: np.ndarray,
  dimension: str,
  levels: tuple[str, ...],
  config: Lm7Config,
) -> dict[str, Any]:
  """Point estimates, simultaneous bands, max-T pairs, and the descriptive range."""
  points = {level: level_gain(y, score_a, score_b, labels, level) for level in levels}
  replicates = bootstrap_level_estimates(
    y, score_a, score_b, labels, levels, config.n_replicates, config.seed,
  )
  counts = _event_counts(y, labels, levels)
  ns = {level: int((labels == level).sum()) for level in levels}
  bands = simultaneous_bands(levels, points, replicates, config.min_valid_fraction)
  for row in bands:
    row["dimension"] = dimension
    row["n"] = ns[row["level"]]
    row["n_events"] = counts[row["level"]]
  pairs = pairwise_contrasts(dimension, counts, config.age_reference, config.min_pair_events)
  adjusted = max_t_intervals(pairs, points, replicates, config.min_valid_fraction)
  for row in adjusted:
    row["dimension"] = dimension
  return {
    "bands": bands,
    "pairwise": adjusted,
    "omnibus_range": omnibus_range(points, levels),
    "event_counts": counts,
  }


def run_subgroup_multiplicity(
  cohort_path: Path,
  predictions_context_path: Path,
  admissions_path: Path | None,
  config_path: Path,
  output_dir: Path,
  report_path: Path,
  source_commit: str,
) -> dict[str, Any]:
  """Context-adjusted locked-test B−A only. Does not refit probes or add levels."""
  from ecg_rep.data import load_admissions

  config = load_lm7_config(config_path)
  cohort = pl.read_parquet(cohort_path).select(["subject_id", "gender", "age_at_ecg"])
  preds = pl.read_parquet(predictions_context_path)
  joined = preds.join(cohort, on="subject_id", how="inner")
  y = np.asarray(joined["mortality_30d"].to_numpy(), dtype=np.int64)
  score_a = np.asarray(joined["p_a"].to_numpy(), dtype=np.float64)
  score_b = np.asarray(joined["p_b"].to_numpy(), dtype=np.float64)
  sex = np.asarray(joined["gender"].fill_null("U").to_numpy())
  age = assign_age_subgroup(joined["age_at_ecg"].to_numpy().astype(np.float64))
  dimensions: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {
    "sex": (sex, SEX_LEVELS),
    "age": (age, AGE_LEVELS),
  }
  if admissions_path is not None and admissions_path.exists():
    race = subject_race_from_admissions(load_admissions(admissions_path))
    with_race = joined.select("subject_id").join(race, on="subject_id", how="left")
    dimensions["race_ethnicity"] = (
      np.asarray(with_race["race_group"].fill_null("unknown_other").to_numpy()),
      RACE_LEVELS,
    )
  results = {
    name: run_dimension(y, score_a, score_b, labels, name, levels, config)
    for name, (labels, levels) in dimensions.items()
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  band_rows = [row for dim in results.values() for row in dim["bands"]]
  pair_rows = [row for dim in results.values() for row in dim["pairwise"]]
  pl.DataFrame(band_rows).write_csv(output_dir / "subgroup_simultaneous_intervals.csv")
  if pair_rows:
    pl.DataFrame(pair_rows).write_csv(output_dir / "subgroup_pairwise_adjusted.csv")
  summary = {
    "source_commit": source_commit,
    "config_hash": config.config_hash,
    "analysis_status": "post_hoc_reporting_sensitivity",
    "n": int(len(y)),
    "n_events": int(y.sum()),
    "score_source": "frozen_primary_context_adjusted",
    "case_mix_standardization": "not_recomputed",
    "dimensions": {
      name: {"omnibus_range": dim["omnibus_range"], "event_counts": dim["event_counts"]}
      for name, dim in results.items()
    },
    "primary_artifacts_overwritten": False,
  }
  (output_dir / "subgroup_multiplicity_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(_render(summary, results, config), encoding="utf-8")
  return summary


def _render(summary: dict[str, Any], results: dict[str, Any], config: Lm7Config) -> str:
  lines = [
    "# LM7 subgroup multiplicity",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {summary['source_commit']}",
    "analysis_status: post_hoc_reporting_sensitivity",
    f"config_hash: {config.config_hash}",
    "verification: [existing levels only; point estimates are the prespecified B-A gains; case-mix standardization not mixed in]",
    "next_action: LM9 claim reconciliation",
    "```",
    "",
    "Simultaneous bands are unstandardized max-deviation intervals within each dimension.",
    "They are not a replacement for the prespecified point estimates and they are not case-mix standardized.",
    "",
  ]
  for name, dim in results.items():
    lines.append(f"## {name}")
    lines.append("")
    lines.append("| Level | n | Events | B−A | Simultaneous low | Simultaneous high | Label |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in dim["bands"]:
      point = "NA" if row["point"] is None else f"{row['point']:+.4f}"
      lo = "NA" if row["simultaneous_low"] is None else f"{row['simultaneous_low']:+.4f}"
      hi = "NA" if row["simultaneous_high"] is None else f"{row['simultaneous_high']:+.4f}"
      lines.append(
        f"| {row['level']} | {row['n']} | {row['n_events']} | {point} | {lo} | {hi} | {row['label']} |"
      )
    lines.append("")
    span = dim["omnibus_range"]
    lines.append(
      f"Descriptive range of level estimates: {span['range']}. Not a hypothesis test."
    )
    lines.append("")
    if dim["pairwise"]:
      lines.append("| Contrast | Point | Adjusted low | Adjusted high |")
      lines.append("|---|---|---|---|")
      for row in dim["pairwise"]:
        point = "NA" if row["point"] is None else f"{row['point']:+.4f}"
        lo = "NA" if row["adjusted_low"] is None else f"{row['adjusted_low']:+.4f}"
        hi = "NA" if row["adjusted_high"] is None else f"{row['adjusted_high']:+.4f}"
        lines.append(f"| {row['contrast']} | {point} | {lo} | {hi} |")
      lines.append("")
  return "\n".join(lines)
