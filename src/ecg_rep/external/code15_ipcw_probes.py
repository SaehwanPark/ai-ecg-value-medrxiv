"""Audit02 P1-3: IPCW-weighted fixed-horizon probe fitting on CODE-15%.

The primary E4 probes are trained only among patients with known 1-year binary
outcomes (complete-case fitting); the D4 IPCW sensitivity corrects the
*evaluation* but keeps that fitting restriction. A reviewer can reasonably ask
whether excluding the ~3.8% censored patients during model fitting changes the
representation comparison. This module closes that question with a cheap,
post hoc sensitivity: ridge-logistic probes refit with inverse-probability-of-
censoring case weights on the same fixed-horizon labels, then evaluated with
the same cumulative/dynamic IPCW AUROC on all extracted test rows.

Governance:

- This is an audit-requested post hoc sensitivity; the locked test had already
  been evaluated. Nothing here restores confirmatory status.
- The censoring Kaplan–Meier is fitted on all extracted development+validation
  rows (identical reference to the D4 evaluation sensitivity), held fixed for
  fitting weights, evaluation weights, and bootstrap resamples.
- Case weights are ``1/G(T_i)`` under the scikit-survival parity convention;
  known controls have unit weight. Development-only standardization is kept;
  the C grid, solver, and tuning protocol mirror the MIMIC/E4 specification
  with weighted validation log-loss for ``C`` selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ecg_rep.external.code15_labels import DAYS_PER_YEAR, HORIZON_DAYS, horizon_label_to_int
from ecg_rep.external.code15_probes import (
  FIT_SPECS,
  assemble_fit_matrix,
)
from ecg_rep.external.code15_survival import (
  CensoringDistribution,
  _case_control_masks,
  censoring_sensitivity_report,
  fit_censoring_distribution,
)
from ecg_rep.probe import DEFAULT_C_GRID, TrainedProbe

#: Fit names refit with IPCW case weights (the contrast-relevant E4 fits).
IPCW_FIT_SPECS: tuple[str, ...] = ("A", "B", "A+B", "C_demo+A", "C_demo+B")

#: Contrast name -> (base fit, added fit) evaluated on the weighted probes.
IPCW_CONTRASTS: dict[str, tuple[str, str]] = {
  "B_minus_A": ("A", "B"),
  "C_demo_B_minus_C_demo_A": ("C_demo+A", "C_demo+B"),
}


def ipcw_case_weights(
  times: npt.NDArray[np.floating],
  event_observed: npt.NDArray[np.bool_],
  horizon: float,
  censoring_distribution: CensoringDistribution,
) -> npt.NDArray[np.float64]:
  """Return IPCW fitting weights on known-horizon-label rows.

  Cases (event at or before ``horizon``) get ``1/G(T_i)`` under the
  scikit-survival parity convention; known controls get unit weight. The
  masks reuse the evaluation label contract so fitting and evaluation
  populations are defined exactly once.
  """
  cases, controls = _case_control_masks(
    np.asarray(times, dtype=np.float64),
    np.asarray(event_observed, dtype=bool),
    horizon,
  )
  known = cases | controls
  if not np.all(known):
    raise ValueError("All fitting rows must be horizon cases or known controls.")
  weights = np.ones(len(times), dtype=np.float64)
  g_case = censoring_distribution.case_weight_survival(np.asarray(times, dtype=np.float64)[cases])
  if np.any(g_case <= 0):
    raise ValueError("Censoring survival is zero at one or more case times.")
  weights[cases] = 1.0 / g_case
  return weights


def _weighted_log_loss(
  y_true: npt.NDArray[np.integer],
  prob: npt.NDArray[np.floating],
  weights: npt.NDArray[np.floating],
) -> float:
  eps = np.finfo(np.float64).eps
  p = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
  y = np.asarray(y_true, dtype=np.float64)
  losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
  return float(np.average(losses, weights=weights))


def fit_weighted_logistic_probe(
  x_dev: npt.NDArray[np.float64],
  y_dev: npt.NDArray[np.int64],
  w_dev: npt.NDArray[np.floating],
  x_val: npt.NDArray[np.float64],
  y_val: npt.NDArray[np.int64],
  w_val: npt.NDArray[np.floating],
  c_grid: tuple[float, ...] = DEFAULT_C_GRID,
  seed: int = 42,
  max_iter: int = 2000,
) -> TrainedProbe:
  """Fit a ridge-logistic probe with IPCW sample weights (dev fit, val tuning).

  Standardization is learned on ``x_dev`` alone (unsupervised, unweighted).
  ``C`` is selected by weighted validation log-loss, mirroring the E4 protocol
  with the IPCW objective.
  """
  from sklearn.linear_model import LogisticRegression
  from sklearn.preprocessing import StandardScaler

  from ecg_rep.probe import compute_auroc, compute_brier_score, ProbeValidationStep

  scaler = StandardScaler()
  x_dev_fit = np.asarray(scaler.fit_transform(np.asarray(x_dev, dtype=np.float64)), dtype=np.float64)
  x_val_fit = np.asarray(scaler.transform(np.asarray(x_val, dtype=np.float64)), dtype=np.float64)
  y_dev_arr = np.asarray(y_dev, dtype=np.int64)
  y_val_arr = np.asarray(y_val, dtype=np.int64)
  w_dev_arr = np.asarray(w_dev, dtype=np.float64)
  w_val_arr = np.asarray(w_val, dtype=np.float64)
  if np.any(w_dev_arr <= 0) or np.any(w_val_arr <= 0):
    raise ValueError("IPCW fitting weights must be strictly positive.")

  history: list[ProbeValidationStep] = []
  best_c = c_grid[0]
  best_loss = float("inf")
  best_coef: npt.NDArray[np.float64] | None = None
  best_intercept = 0.0
  for c_value in c_grid:
    model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=max_iter, random_state=seed)
    model.fit(x_dev_fit, y_dev_arr, sample_weight=w_dev_arr)
    val_prob = np.asarray(model.predict_proba(x_val_fit)[:, 1], dtype=np.float64)
    step = ProbeValidationStep(
      c_value=float(c_value),
      val_log_loss=_weighted_log_loss(y_val_arr, val_prob, w_val_arr),
      val_auroc=compute_auroc(y_val_arr, val_prob),
      val_brier=compute_brier_score(y_val_arr, val_prob),
    )
    history.append(step)
    if step.val_log_loss < best_loss:
      best_loss = step.val_log_loss
      best_c = float(c_value)
      best_coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
      best_intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])
  if best_coef is None:
    raise RuntimeError("IPCW-weighted probe fitting produced no coefficients.")
  means = tuple(float(v) for v in np.asarray(scaler.mean_, dtype=np.float64).reshape(-1))
  scales = tuple(float(v) for v in np.asarray(scaler.scale_, dtype=np.float64).reshape(-1))
  return TrainedProbe(
    coefficients=tuple(float(v) for v in best_coef),
    intercept=best_intercept,
    best_c=best_c,
    tuning_history=tuple(history),
    feature_means=means,
    feature_scales=scales,
    standardize=True,
  )


def run_code15_ipcw_fit_sensitivity(
  features_a: npt.NDArray[np.float64],
  features_b: npt.NDArray[np.float64],
  meta: Any,
  horizon: str = "365d",
  seed: int = 42,
  n_bootstraps: int = 1000,
) -> dict[str, Any]:
  """Fit IPCW-weighted probes and evaluate them with cumulative/dynamic AUROC."""
  import polars as pl

  label_col = f"y_{horizon}"
  required = {"patient_id", "split", "age", "is_male", "death", "timey", label_col}
  missing = required - set(meta.columns)
  if missing:
    raise ValueError(f"Meta frame is missing columns: {sorted(missing)}.")
  meta = horizon_label_to_int(meta, horizon)

  splits = meta["split"].to_list()
  dev_val_all_mask = np.asarray([s in ("dev", "val") for s in splits], dtype=bool)
  test_all_mask = np.asarray([s == "test" for s in splits], dtype=bool)

  ref_times = np.asarray(meta.filter(pl.Series(dev_val_all_mask))["timey"].to_list(), dtype=np.float64) * DAYS_PER_YEAR
  ref_events = np.asarray(meta.filter(pl.Series(dev_val_all_mask))["death"].to_list(), dtype=bool)
  censoring = fit_censoring_distribution(ref_times, ref_events)

  # IPCW-weighted fitting on dev and val rows with known horizon labels.
  weights_report: dict[str, Any] = {}
  split_stores: dict[str, tuple[dict[str, npt.NDArray[np.float64]], npt.NDArray[np.int64], npt.NDArray[np.float64]]] = {}
  for split_name in ("dev", "val"):
    split_mask = np.asarray([s == split_name for s in splits], dtype=bool)
    sub = meta.filter(pl.Series(split_mask))
    labels = sub[label_col].to_list()
    known_mask = np.asarray([v is not None for v in labels], dtype=bool)
    sub_known = sub.filter(pl.Series(known_mask))
    y = np.asarray(sub_known[label_col].to_list(), dtype=np.int64)
    times = np.asarray(sub_known["timey"].to_list(), dtype=np.float64) * DAYS_PER_YEAR
    events = np.asarray(sub_known["death"].to_list(), dtype=bool)
    weights = ipcw_case_weights(times, events, HORIZON_DAYS[horizon], censoring)
    cases, controls = _case_control_masks(times, events, HORIZON_DAYS[horizon])
    weights_report[split_name] = {
      "n_rows": int(split_mask.sum()),
      f"n_dropped_censored_before_{horizon}": int((~known_mask).sum()),
      "n_fit_rows": int(known_mask.sum()),
      "n_cases": int(cases.sum()),
      "n_controls": int(controls.sum()),
      "case_weight_min": float(weights[cases].min()) if cases.any() else None,
      "case_weight_median": float(np.median(weights[cases])) if cases.any() else None,
      "case_weight_max": float(weights[cases].max()) if cases.any() else None,
    }
    fa_known = np.asarray(features_a, dtype=np.float64)[split_mask][known_mask]
    fb_known = np.asarray(features_b, dtype=np.float64)[split_mask][known_mask]
    demo_known = np.asarray(
      list(zip(sub_known["age"].to_list(), sub_known["is_male"].to_list(), strict=True)),
      dtype=np.float64,
    )
    split_stores[split_name] = (
      {
        fit_name: assemble_fit_matrix(fit_name, fa_known, fb_known, demo_known)
        for fit_name in IPCW_FIT_SPECS
      },
      y,
      weights,
    )

  (dev_matrices, y_dev, w_dev) = split_stores["dev"]
  (val_matrices, y_val, w_val) = split_stores["val"]
  unknown = set(IPCW_FIT_SPECS) - set(FIT_SPECS)
  if unknown:
    raise ValueError(f"Unknown IPCW fit names: {sorted(unknown)}.")
  probes: dict[str, TrainedProbe] = {}
  for fit_name in IPCW_FIT_SPECS:
    probes[fit_name] = fit_weighted_logistic_probe(
      dev_matrices[fit_name], y_dev, w_dev,
      val_matrices[fit_name], y_val, w_val,
      seed=seed,
    )
  tuning = {
    name: {
      "best_c": probe.best_c,
      "val_log_loss_at_best_c": (
        min(step.val_log_loss for step in probe.tuning_history)
        if probe.tuning_history
        else None
      ),
    }
    for name, probe in probes.items()
  }

  # Score every extracted test row (including censored-before-horizon rows).
  test_meta = meta.filter(pl.Series(test_all_mask))
  fa_test = np.asarray(features_a, dtype=np.float64)[test_all_mask]
  fb_test = np.asarray(features_b, dtype=np.float64)[test_all_mask]
  demo_test = np.asarray(
    list(zip(test_meta["age"].to_list(), test_meta["is_male"].to_list(), strict=True)),
    dtype=np.float64,
  )
  scores_test_all: dict[str, npt.NDArray[np.float64]] = {}
  for fit_name, probe in probes.items():
    matrix = assemble_fit_matrix(fit_name, fa_test, fb_test, demo_test)
    scores_test_all[fit_name] = probe.predict_proba(matrix)

  sensitivity = censoring_sensitivity_report(
    reference_times=ref_times,
    reference_events=ref_events,
    test_times=np.asarray(test_meta["timey"].to_list(), dtype=np.float64) * DAYS_PER_YEAR,
    test_events=np.asarray(test_meta["death"].to_list(), dtype=bool),
    risk_scores=scores_test_all,
    contrasts=IPCW_CONTRASTS,
    horizon=HORIZON_DAYS[horizon],
    n_bootstraps=n_bootstraps,
    seed=seed,
  )

  return {
    "method": (
      "IPCW-weighted fixed-horizon ridge-logistic probes (audit02 P1-3 post hoc "
      "sensitivity); censoring KM from all extracted dev+val rows, held fixed for "
      "fitting weights, evaluation, and bootstrap"
    ),
    "horizon": horizon,
    "fit_specs": list(IPCW_FIT_SPECS),
    "fitting": weights_report,
    "censoring_reference": {
      "n_reference": censoring.n_reference,
      "n_censored_reference": censoring.n_censored,
      "estimator": (
        "reverse Kaplan-Meier; tied true events removed from the risk denominator "
        "(scikit-survival reverse=True convention, parity tested)"
      ),
    },
    "tuning": tuning,
    "ipcw_sensitivity": sensitivity,
  }