"""Post hoc conventional-ECG baseline A' from cart global machine measurements.

Representation A (CIIS morphology) has poor per-lead measurement agreement for its
Q-wave and Q/R inputs, so the manuscript needs a second conventional comparator that
does not depend on the in-repo extractor. MIMIC-IV-ECG ships the acquiring cart's
global fiducials and axes; A' turns them into the interval, rate, and axis summary
clinicians read off every ECG. A' is fitted with the same ridge-logistic probe,
development-only preprocessing, validation tuning, and paired-valid locked test as
A and B. It was added after the locked test had been opened and is reported as a
post hoc sensitivity comparator, never as a pre-specified representation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecg_rep.evaluate import (  # pyright: ignore[reportPrivateUsage]
  C0_FEATURE_COLS,
  _fit_probe,  # pyright: ignore[reportPrivateUsage]
  _join_representation_matrix,  # pyright: ignore[reportPrivateUsage]
)
from ecg_rep.probe import bootstrap_paired_difference, evaluate_predictions, metric_bundle_to_dict

MACHINE_RAW_COLUMNS: tuple[str, ...] = (
  "rr_interval",
  "p_onset",
  "p_end",
  "qrs_onset",
  "qrs_end",
  "t_end",
  "p_axis",
  "qrs_axis",
  "t_axis",
)

# Plausibility windows (ms, degrees). Cart sentinels (29999, 32767, 65535, -32768)
# and implausible fiducial differences fall outside and become missing.
_PLAUSIBLE: dict[str, tuple[float, float]] = {
  "heart_rate_bpm": (20.0, 300.0),
  "pr_interval_ms": (60.0, 400.0),
  "p_duration_ms": (20.0, 250.0),
  "qrs_duration_ms": (40.0, 250.0),
  "qt_interval_ms": (200.0, 700.0),
  "qtc_bazett_ms": (250.0, 750.0),
  "p_axis_deg": (-180.0, 360.0),
  "qrs_axis_deg": (-180.0, 360.0),
  "t_axis_deg": (-180.0, 360.0),
}

MACHINE_BASE_FEATURES: tuple[str, ...] = (
  "heart_rate_bpm",
  "pr_interval_ms",
  "p_duration_ms",
  "qrs_duration_ms",
  "qt_interval_ms",
  "qtc_bazett_ms",
  "qrs_t_angle_deg",
  "p_axis_sin",
  "p_axis_cos",
  "qrs_axis_sin",
  "qrs_axis_cos",
  "t_axis_sin",
  "t_axis_cos",
)

# One indicator per measured quantity, not per encoded column: sin/cos of one axis
# share their missingness, and the QRS-T angle is missing whenever either axis is.
MACHINE_MISSING_SOURCES: tuple[str, ...] = (
  "heart_rate_bpm",
  "pr_interval_ms",
  "p_duration_ms",
  "qrs_duration_ms",
  "qt_interval_ms",
  "p_axis_deg",
  "qrs_axis_deg",
  "t_axis_deg",
)


def _masked(expr: pl.Expr, name: str) -> pl.Expr:
  lo, hi = _PLAUSIBLE[name]
  return pl.when(expr.is_between(lo, hi)).then(expr).otherwise(None).alias(name)


def derive_machine_features(raw: pl.DataFrame) -> pl.DataFrame:
  """Derive rate, interval, QTc, axis, and QRS-T angle features; implausible values become null.

  Axes are circular, so a linear probe receives their sine and cosine. The frontal
  QRS-T angle is the absolute axis difference folded into [0, 180] degrees.
  """
  rr = pl.col("rr_interval").cast(pl.Float64)
  qrs_on = pl.col("qrs_onset").cast(pl.Float64)
  rr_ok = pl.when(rr.is_between(200.0, 3000.0)).then(rr).otherwise(None)
  step = raw.with_columns(
    _masked(60000.0 / rr_ok, "heart_rate_bpm"),
    _masked(qrs_on - pl.col("p_onset"), "pr_interval_ms"),
    _masked(pl.col("p_end").cast(pl.Float64) - pl.col("p_onset"), "p_duration_ms"),
    _masked(pl.col("qrs_end").cast(pl.Float64) - qrs_on, "qrs_duration_ms"),
    _masked(pl.col("t_end").cast(pl.Float64) - qrs_on, "qt_interval_ms"),
    _masked(pl.col("p_axis").cast(pl.Float64), "p_axis_deg"),
    _masked(pl.col("qrs_axis").cast(pl.Float64), "qrs_axis_deg"),
    _masked(pl.col("t_axis").cast(pl.Float64), "t_axis_deg"),
    rr_ok.alias("_rr_ok"),
  )
  angle = (pl.col("qrs_axis_deg") - pl.col("t_axis_deg")).abs() % 360.0
  return step.with_columns(
    _masked(pl.col("qt_interval_ms") / (pl.col("_rr_ok") / 1000.0).sqrt(), "qtc_bazett_ms"),
    pl.when(angle > 180.0).then(360.0 - angle).otherwise(angle).alias("qrs_t_angle_deg"),
    *[
      f(pl.col(f"{axis}_deg").radians()).alias(f"{axis}_{suffix}")
      for axis in ("p_axis", "qrs_axis", "t_axis")
      for suffix, f in (("sin", pl.Expr.sin), ("cos", pl.Expr.cos))
    ],
  ).drop("_rr_ok")


def impute_with_dev_medians(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
  """Add missing indicators, then fill each feature with its development-split median."""
  dev = frame.filter(pl.col("split") == "dev")
  medians = {name: float(dev[name].median() or 0.0) for name in MACHINE_BASE_FEATURES}  # pyright: ignore[reportArgumentType]
  with_flags = frame.with_columns(
    pl.col(src).is_null().cast(pl.Float64).alias(f"missing_{src}") for src in MACHINE_MISSING_SOURCES
  )
  return with_flags.with_columns(pl.col(n).fill_null(medians[n]) for n in MACHINE_BASE_FEATURES), medians


def machine_feature_columns() -> list[str]:
  """Ordered A' design columns: 13 derived features and 8 missing indicators."""
  return [*MACHINE_BASE_FEATURES, *(f"missing_{s}" for s in MACHINE_MISSING_SOURCES)]


def _contrast(y: np.ndarray, ref: np.ndarray, new: np.ndarray, n_bootstraps: int) -> dict[str, Any]:
  return {
    m: asdict(bootstrap_paired_difference(y, ref, new, metric=m, n_bootstraps=n_bootstraps, seed=42))
    for m in ("auroc", "auprc")
  }


def run_machine_baseline(
  cohort_path: Path,
  machine_measurements_path: Path,
  c0_path: Path,
  reps_dir: Path,
  reference_ecg_only_path: Path,
  reference_context_path: Path,
  output_dir: Path,
  tables_dir: Path,
  n_bootstraps: int = 1000,
) -> dict[str, Any]:
  """Fit A' probes on the paired-valid cohort and contrast them with frozen A and B.

  Models: ECG-only ``A'``, ``A+A'``, ``B+A'``; context-adjusted ``C0+A'``, ``C0+A+A'``,
  ``C0+B+A'``. Contrasts use the frozen locked-test ``M_A``/``M_B`` predictions so the
  pre-specified estimates are never refit.
  """
  cohort = pl.read_parquet(cohort_path)
  paired = pl.read_parquet(reps_dir / "representation_pair_validity.parquet").filter(
    pl.col("representation_pair_valid")
  )
  machine = pl.read_csv(
    machine_measurements_path,
    columns=["subject_id", "study_id", *MACHINE_RAW_COLUMNS],
    infer_schema_length=100_000,
  )
  c0 = pl.read_parquet(c0_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]
  frame = (
    cohort.join(paired.select(["subject_id", "study_id"]), on=["subject_id", "study_id"])
    .join(machine, on=["subject_id", "study_id"], how="left")
    .join(c0.select(["subject_id", "study_id", *c0_cols]), on=["subject_id", "study_id"])
  )
  frame, medians = impute_with_dev_medians(derive_machine_features(frame))
  frame = _join_representation_matrix(
    frame,
    pl.read_parquet(reps_dir / "ciis_features_meta.parquet"),
    np.load(reps_dir / "ciis_features.npy"),
    "za",
    "representation_a_valid",
  )
  frame = _join_representation_matrix(
    frame,
    pl.read_parquet(reps_dir / "dbeta_embedding_meta.parquet"),
    np.load(reps_dir / "dbeta_embeddings.npy"),
    "zb",
    "representation_b_valid",
  ).filter(pl.col("representation_a_valid") & pl.col("representation_b_valid"))
  za = [c for c in frame.columns if c.startswith("za_")]
  zb = [c for c in frame.columns if c.startswith("zb_")]
  ap = machine_feature_columns()

  specs = {
    "M_AP_ECG": ap,
    "M_A_AP_ECG": za + ap,
    "M_B_AP_ECG": zb + ap,
    "M_AP": c0_cols + ap,
    "M_A_AP": c0_cols + za + ap,
    "M_B_AP": c0_cols + zb + ap,
  }
  test_ids = None
  preds: dict[str, np.ndarray] = {}
  models: dict[str, Any] = {}
  for name, cols in specs.items():
    probe, _dev, _val, test = _fit_probe(frame, cols, model_name=name)
    p = probe.predict_proba(test.x)
    if test_ids is None:
      test_ids = pl.DataFrame({"subject_id": test.subject_ids, "study_id": test.study_ids, "y": test.y})
    preds[name] = p
    models[name] = {
      "n_features": len(cols),
      "best_c": probe.best_c,
      "test_metrics": metric_bundle_to_dict(evaluate_predictions(test.y, p)),
    }
  assert test_ids is not None
  joined = (
    test_ids.with_columns(*(pl.Series(k, v) for k, v in preds.items()))
    .join(pl.read_parquet(reference_ecg_only_path).select(["subject_id", "study_id", "p_a_ecg", "p_b_ecg"]), on=["subject_id", "study_id"])
    .join(pl.read_parquet(reference_context_path).select(["subject_id", "study_id", "p_c", "p_a", "p_b"]), on=["subject_id", "study_id"])
  )
  if joined.height != test_ids.height:
    raise ValueError(f"Frozen predictions cover {joined.height} of {test_ids.height} locked-test rows.")

  def col(name: str) -> np.ndarray:
    return joined[name].to_numpy().astype(np.float64)

  y = joined["y"].to_numpy().astype(np.int64)
  contrasts = {
    "ecg_only_ap_minus_a": _contrast(y, col("p_a_ecg"), col("M_AP_ECG"), n_bootstraps),
    "ecg_only_b_minus_ap": _contrast(y, col("M_AP_ECG"), col("p_b_ecg"), n_bootstraps),
    "ecg_only_a_ap_minus_ap": _contrast(y, col("M_AP_ECG"), col("M_A_AP_ECG"), n_bootstraps),
    "ecg_only_b_ap_minus_b": _contrast(y, col("p_b_ecg"), col("M_B_AP_ECG"), n_bootstraps),
    "context_ap_minus_c0": _contrast(y, col("p_c"), col("M_AP"), n_bootstraps),
    "context_ap_minus_a": _contrast(y, col("p_a"), col("M_AP"), n_bootstraps),
    "context_b_minus_ap": _contrast(y, col("M_AP"), col("p_b"), n_bootstraps),
    "context_a_ap_minus_ap": _contrast(y, col("M_AP"), col("M_A_AP"), n_bootstraps),
    "context_b_ap_minus_b": _contrast(y, col("p_b"), col("M_B_AP"), n_bootstraps),
  }
  missing_rates = {
    src: float(frame.filter(pl.col("split") == "test")[f"missing_{src}"].mean() or 0.0)  # pyright: ignore[reportArgumentType]
    for src in MACHINE_MISSING_SOURCES
  }
  results: dict[str, Any] = {
    "status": "complete",
    "post_hoc": True,
    "representation": "A_prime_cart_global_measurements",
    "n_test": int(len(y)),
    "n_test_events": int(y.sum()),
    "dev_medians": medians,
    "test_missing_rates": missing_rates,
    "models": models,
    "contrasts": contrasts,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  tables_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "machine_baseline_results.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  pl.DataFrame([
    {
      "contrast": name,
      **{f"{m}_{k}": v[m][k] for m in ("auroc", "auprc") for k in ("point_estimate", "ci_lower", "ci_upper")},
    }
    for name, v in contrasts.items()
  ]).write_csv(tables_dir / "supp_machine_baseline_contrasts.csv")
  pl.DataFrame([
    {"model": name, "n_features": m["n_features"], "best_c": m["best_c"], **{
      k: m["test_metrics"][k] for k in ("auroc", "auprc", "brier", "calibration_slope") if k in m["test_metrics"]
    }}
    for name, m in models.items()
  ]).write_csv(tables_dir / "supp_machine_baseline_models.csv")
  return results
