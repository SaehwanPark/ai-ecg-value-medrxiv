"""External measurement validity of the frozen CIIS extractor on PTB-XL against GE 12SL.

MIMIC-IV-ECG ships only global machine intervals, so the per-lead morphology that
Representation A depends on has no in-domain reference. PTB-XL+ supplies GE
Marquette 12SL per-lead measurements for the public PTB-XL 500 Hz recordings, which
lets the unchanged MIMIC extractor be compared with a commercial reference on
public data. Results describe the extractor, not the MIMIC cohort, and are
reported as a measurement-validity audit rather than outcome validation.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score

from ecg_rep.data import read_ecg_waveform
from ecg_rep.probe import bootstrap_paired_difference, compute_auroc
from ecg_rep.scoring.traditional import (
  CIIS_FEATURE_NAMES,
  CIIS_PHYSIOLOGIC_BOUNDS,
  CIISMeasurements,
  compute_ciis_from_measurements,
  score_ecg_waveform,
  validate_ciis_measurements_against_reference,
)

PTBXL_LEAD_ALIASES: dict[str, str] = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

CIIS_ITEM_NAMES: tuple[str, ...] = (
  "item1_avl_q_duration",
  "item2_avl_t_amplitude",
  "item3_avr_r_amplitude",
  "item4_avr_t_amplitude",
  "item5_lead2_avf_qr_ratio",
  "item6_lead3_avl_q_duration",
  "item7_lead3_t_amplitude",
  "item8_v1_t_amplitude",
  "item9_v2_r_amplitude",
  "item10_v2_t_amplitude",
  "item11_v3_qr_ratio",
  "item12_v5_s_amplitude",
)

CIIS_CATEGORY_ORDER: tuple[str, ...] = ("normal", "borderline", "possible_injury", "probable_infarction")

# Codable-deflection threshold shared with the extractor's Q/R ratio rule.
_MIN_DEFLECTION_MV = 0.025


def harmonize_lead_names(names: Sequence[str]) -> list[str]:
  """Map PTB-XL upper-case limb-lead labels onto the extractor's canonical names."""
  return [PTBXL_LEAD_ALIASES.get(name, name) for name in names]


def _qr_ratio(q_amp_mv: float, r_amp_mv: float) -> float:
  q = abs(q_amp_mv)
  if r_amp_mv >= _MIN_DEFLECTION_MV:
    return min(10.0, q / r_amp_mv)
  return 10.0 if q >= _MIN_DEFLECTION_MV else 0.0


def reference_ciis_measurements(row: Mapping[str, Any]) -> dict[str, float]:
  """Translate one PTB-XL+ 12SL record into the 15 CIIS measurements.

  12SL reports Q, S, and negative-T amplitudes as signed (negative) millivolts, so
  magnitudes are taken; ``Max_*_Amp`` columns are unsigned microvolts and are not
  used. Inverted-lead quantities follow the extractor's definitions: the -aVR R wave
  is the larger of the aVR Q (QS complexes) and S magnitudes, the -aVR positive T
  is the negative T of aVR, and the -aVL Q wave is an initial aVL R wave when aVL has
  no Q wave. Values outside the extractor's physiologic bounds (12SL sentinels such
  as -219 mV) become NaN rather than being clipped.
  """

  def g(name: str) -> float:
    value = row.get(name)
    return float(value) if value is not None else float("nan")

  avl_has_q = g("Q_Dur_aVL") > 0.0
  inv_avl_q = 0.0 if avl_has_q or g("R_Amp_aVL") < _MIN_DEFLECTION_MV else g("R_Dur_aVL")
  values = {
    "avl_q_duration_ms": g("Q_Dur_aVL"),
    "avl_t_pos_mv": max(g("T+_Amp_aVL"), 0.0),
    "avl_t_neg_mv": abs(min(g("T-_Amp_aVL"), 0.0)),
    "inv_avr_r_mv": max(abs(min(g("Q_Amp_aVR"), 0.0)), abs(min(g("S_Amp_aVR"), 0.0))),
    "inv_avr_t_pos_mv": abs(min(g("T-_Amp_aVR"), 0.0)),
    "lead2_qr_ratio": _qr_ratio(g("Q_Amp_II"), g("R_Amp_II")),
    "avf_qr_ratio": _qr_ratio(g("Q_Amp_aVF"), g("R_Amp_aVF")),
    "lead3_q_duration_ms": g("Q_Dur_III"),
    "inv_avl_q_duration_ms": inv_avl_q,
    "lead3_t_neg_mv": abs(min(g("T-_Amp_III"), 0.0)),
    "v1_t_pos_mv": max(g("T+_Amp_V1"), 0.0),
    "v2_r_mv": g("R_Amp_V2"),
    "v2_t_neg_mv": abs(min(g("T-_Amp_V2"), 0.0)),
    "v3_qr_ratio": _qr_ratio(g("Q_Amp_V3"), g("R_Amp_V3")),
    "v5_s_mv": abs(min(g("S_Amp_V5"), 0.0)),
  }
  out: dict[str, float] = {}
  for name, value in values.items():
    lo, hi = CIIS_PHYSIOLOGIC_BOUNDS[name]
    out[name] = value if np.isfinite(value) and lo <= value <= hi else float("nan")
  return out


def score_measurement_row(row: Mapping[str, float]) -> dict[str, Any] | None:
  """Score a measurement mapping with the canonical CIIS; ``None`` if any input is missing."""
  if any(not np.isfinite(row[name]) for name in CIIS_FEATURE_NAMES):
    return None
  result = compute_ciis_from_measurements(CIISMeasurements(**{n: float(row[n]) for n in CIIS_FEATURE_NAMES}))
  assert result.item_scores is not None and result.category is not None
  return {
    "ciis_total_score": result.total_score,
    "ciis_category": result.category.value,
    **{name: float(v) for name, v in asdict(result.item_scores).items()},
  }


def mi_labels(database: pl.DataFrame, statements: pl.DataFrame) -> pl.DataFrame:
  """Label PTB-XL records with any MI-class SCP code (PTB-XL superclass aggregation)."""
  mi_codes = set(
    statements.filter(pl.col("diagnostic_class") == "MI").get_column("code").to_list()
  )
  rows = []
  for ecg_id, codes in zip(database["ecg_id"].to_list(), database["scp_codes"].to_list()):
    parsed: dict[str, float] = ast.literal_eval(codes)
    rows.append({
      "ecg_id": int(ecg_id),
      "mi": int(any(code in mi_codes for code in parsed)),
      "norm": int("NORM" in parsed),
    })
  return pl.DataFrame(rows)


def item_agreement(ours: pl.DataFrame, ref: pl.DataFrame) -> dict[str, Any]:
  """Per-item firing rates and agreement between extractor-based and 12SL-based CIIS.

  Item 3 is continuous (-1 point per mm of -aVR R), so it is summarized by
  correlation; every other item takes a small set of point values and is
  summarized by exact agreement and Cohen's kappa on "points differ from the
  item's modal 12SL value".
  """
  joined = ours.join(ref, on="ecg_id", suffix="_ref")
  out: dict[str, Any] = {"n": joined.height, "items": {}}
  for item in CIIS_ITEM_NAMES:
    a = joined[item].to_numpy().astype(np.float64)
    b = joined[f"{item}_ref"].to_numpy().astype(np.float64)
    if item == "item3_avr_r_amplitude":
      out["items"][item] = {
        "mean_points_extractor": float(a.mean()),
        "mean_points_12sl": float(b.mean()),
        "pearson_r": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None,
      }
      continue
    values, counts = np.unique(b, return_counts=True)
    modal = float(values[np.argmax(counts)])
    fire_a = (a != modal).astype(int)
    fire_b = (b != modal).astype(int)
    kappa = (
      float(cohen_kappa_score(fire_a, fire_b))
      if len(np.unique(fire_a)) > 1 or len(np.unique(fire_b)) > 1
      else None
    )
    out["items"][item] = {
      "modal_12sl_points": modal,
      "non_modal_rate_extractor": float(fire_a.mean()),
      "non_modal_rate_12sl": float(fire_b.mean()),
      "exact_point_agreement": float(np.mean(a == b)),
      "cohen_kappa_non_modal": kappa,
    }
  return out


def category_agreement(ours: pl.DataFrame, ref: pl.DataFrame) -> dict[str, Any]:
  """Total-score correlation, category confusion matrix, and quadratic-weighted kappa."""
  joined = ours.join(ref, on="ecg_id", suffix="_ref")
  rank = {c: i for i, c in enumerate(CIIS_CATEGORY_ORDER)}
  a = [rank[c] for c in joined["ciis_category"].to_list()]
  b = [rank[c] for c in joined["ciis_category_ref"].to_list()]
  matrix = np.zeros((4, 4), dtype=int)
  for i, j in zip(a, b):
    matrix[i, j] += 1
  sa = joined["ciis_total_score"].to_numpy().astype(np.float64)
  sb = joined["ciis_total_score_ref"].to_numpy().astype(np.float64)
  return {
    "n": joined.height,
    "total_score_pearson_r": float(np.corrcoef(sa, sb)[0, 1]),
    "total_score_mean_extractor": float(sa.mean()),
    "total_score_mean_12sl": float(sb.mean()),
    "category_exact_agreement": float(np.trace(matrix) / matrix.sum()),
    "category_quadratic_kappa": float(cohen_kappa_score(a, b, weights="quadratic")),
    "confusion_rows_extractor_cols_12sl": {
      CIIS_CATEGORY_ORDER[i]: {CIIS_CATEGORY_ORDER[j]: int(matrix[i, j]) for j in range(4)}
      for i in range(4)
    },
    "category_distribution_extractor": {c: int(sum(1 for x in a if x == rank[c])) for c in CIIS_CATEGORY_ORDER},
    "category_distribution_12sl": {c: int(sum(1 for x in b if x == rank[c])) for c in CIIS_CATEGORY_ORDER},
  }


def mi_discrimination(
  ours: pl.DataFrame,
  ref: pl.DataFrame,
  labels: pl.DataFrame,
  n_bootstraps: int = 1000,
  seed: int = 2026,
) -> dict[str, Any]:
  """AUROC of the CIIS total for PTB-XL MI labels, extractor vs 12SL inputs, paired."""
  joined = (
    ours.select(["ecg_id", "ciis_total_score"])
    .join(ref.select(["ecg_id", pl.col("ciis_total_score").alias("ciis_total_score_ref")]), on="ecg_id")
    .join(labels, on="ecg_id")
  )
  y = joined["mi"].to_numpy().astype(np.int64)
  sa = joined["ciis_total_score"].to_numpy().astype(np.float64)
  sb = joined["ciis_total_score_ref"].to_numpy().astype(np.float64)
  diff = bootstrap_paired_difference(y, sa, sb, "auroc", n_bootstraps=n_bootstraps, seed=seed)
  return {
    "n": int(len(y)),
    "n_mi": int(y.sum()),
    "auroc_extractor": compute_auroc(y, sa),
    "auroc_12sl": compute_auroc(y, sb),
    "auroc_12sl_minus_extractor": asdict(diff),
  }


def parse_sample_ids(text: str) -> list[int]:
  """Read ``ecg_id``s from a list of PTB-XL ``records500/NNNNN/NNNNN_hr`` paths or bare ids."""
  ids = []
  for token in text.split():
    stem = token.rsplit("/", 1)[-1].removesuffix("_hr")
    ids.append(int(stem))
  return ids


def extract_ptbxl_measurements(
  records_root: Path, database: pl.DataFrame, ecg_ids: Sequence[int]
) -> pl.DataFrame:
  """Run the unchanged MIMIC CIIS extractor on PTB-XL 500 Hz records."""
  by_id = dict(zip(database["ecg_id"].to_list(), database["filename_hr"].to_list()))
  rows: list[dict[str, Any]] = []
  for ecg_id in ecg_ids:
    record = records_root / str(by_id[ecg_id])
    if not (record.with_suffix(".dat").exists() and record.with_suffix(".hea").exists()):
      rows.append({"ecg_id": int(ecg_id), "is_valid": False, "error": "record_missing"})
      continue
    signal, names, fs = read_ecg_waveform(record)
    result = score_ecg_waveform(signal, harmonize_lead_names(names), fs)
    if not result.is_valid or result.measurements is None:
      rows.append({"ecg_id": int(ecg_id), "is_valid": False, "error": result.error_message})
      continue
    rows.append({
      "ecg_id": int(ecg_id),
      "is_valid": True,
      "error": None,
      **{n: float(getattr(result.measurements, n)) for n in CIIS_FEATURE_NAMES},
    })
  return pl.DataFrame(rows, infer_schema_length=None)


def _scored_frame(measurements: pl.DataFrame) -> pl.DataFrame:
  rows = []
  for row in measurements.iter_rows(named=True):
    scored = score_measurement_row(row)
    if scored is not None:
      rows.append({"ecg_id": int(row["ecg_id"]), **scored})
  return pl.DataFrame(rows)


def _finite(value: float) -> float | None:
  return float(value) if np.isfinite(value) else None


def run_ptbxl_measurement_validity(
  records_root: Path,
  database_path: Path,
  statements_path: Path,
  features_12sl_path: Path,
  sample_ids_path: Path,
  output_dir: Path,
  tables_dir: Path,
  n_bootstraps: int = 1000,
) -> dict[str, Any]:
  """Compare extractor measurements, CIIS items, categories, and MI AUROC with 12SL."""
  database = pl.read_csv(database_path, infer_schema_length=None)
  statements = pl.read_csv(statements_path, infer_schema_length=None).rename({"": "code"})
  sl = pl.read_csv(features_12sl_path, infer_schema_length=None)
  ecg_ids = parse_sample_ids(sample_ids_path.read_text(encoding="utf-8"))

  ours_raw = extract_ptbxl_measurements(records_root, database, ecg_ids)
  ours = ours_raw.filter(pl.col("is_valid")).drop(["is_valid", "error"])
  sl_sample = sl.filter(pl.col("ecg_id").cast(pl.Int64).is_in(ecg_ids))
  ref = pl.DataFrame([
    {"ecg_id": int(row["ecg_id"]), **reference_ciis_measurements(row)}
    for row in sl_sample.iter_rows(named=True)
  ])

  feature_agreement = validate_ciis_measurements_against_reference(ours, ref, id_col="ecg_id")
  joined = ours.join(ref, on="ecg_id", suffix="_ref")
  feature_rows = []
  for name in CIIS_FEATURE_NAMES:
    stats = feature_agreement["features"].get(name)
    sub = joined.filter(pl.col(name).is_not_nan() & pl.col(f"{name}_ref").is_not_nan())
    x = sub[name].to_numpy().astype(np.float64)
    y = sub[f"{name}_ref"].to_numpy().astype(np.float64)
    feature_rows.append({
      "feature": name,
      "n": int(len(x)),
      "pearson_r": None if stats is None else stats["pearson_r"],
      "spearman_rho": None if stats is None else stats["spearman_rho"],
      "mae": None if stats is None else stats["mae"],
      "bias_extractor_minus_12sl": None if stats is None else stats["bias"],
      "loa_lower": None if stats is None else stats["bias"] - 1.96 * stats["error_std"],
      "loa_upper": None if stats is None else stats["bias"] + 1.96 * stats["error_std"],
      "median_extractor": _finite(float(np.median(x))) if len(x) else None,
      "median_12sl": _finite(float(np.median(y))) if len(y) else None,
      "zero_rate_extractor": float(np.mean(x == 0.0)) if len(x) else None,
      "zero_rate_12sl": float(np.mean(y == 0.0)) if len(y) else None,
    })

  ours_scored = _scored_frame(ours)
  ref_scored = _scored_frame(ref)
  labels = mi_labels(database, statements)
  results: dict[str, Any] = {
    "status": "complete",
    "reference": "PTB-XL+ GE 12SL per-lead measurements (PTB-XL 500 Hz records)",
    "n_sampled": len(ecg_ids),
    "n_extractor_valid": ours.height,
    "n_reference_complete": ref_scored.height,
    "extractor_failures": ours_raw.filter(~pl.col("is_valid")).group_by("error").len().to_dicts(),
    "features": feature_rows,
    "items": item_agreement(ours_scored, ref_scored),
    "categories": category_agreement(ours_scored, ref_scored),
    "mi_discrimination": mi_discrimination(ours_scored, ref_scored, labels, n_bootstraps=n_bootstraps),
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  tables_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "ptbxl_measurement_validity.json").write_text(
    json.dumps(results, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
  )
  pl.DataFrame(feature_rows).write_csv(tables_dir / "supp_ptbxl_feature_agreement.csv")
  item_rows = [{"item": k, **v} for k, v in results["items"]["items"].items()]
  pl.DataFrame(item_rows, infer_schema_length=None).write_csv(tables_dir / "supp_ptbxl_item_agreement.csv")
  return results
