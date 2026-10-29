"""Censoring-aware fixed-horizon discrimination sensitivities for CODE-15%."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class CensoringDistribution:
  """Kaplan–Meier estimate of the censoring survival function."""

  censoring_times: npt.NDArray[np.float64]
  survival_after: npt.NDArray[np.float64]
  n_reference: int
  n_censored: int
  max_follow_up: float

  def case_weight_survival(self, times: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Return the censoring survival used for IPCW case weights.

    Matches the scikit-survival ``CensoringDistributionEstimator.predict_proba``
    convention: the left limit of the censoring survival at each time, except
    at times exactly equal to a reference censoring time, where the post-tie
    value is used.
    """
    values = np.asarray(times, dtype=np.float64)
    n_times = len(self.censoring_times)
    positions = np.searchsorted(self.censoring_times, values, side="left")
    result = np.ones(values.shape, dtype=np.float64)
    if n_times == 0:
      return result
    observed = positions > 0
    result[observed] = self.survival_after[positions[observed] - 1]
    exact = (positions < n_times) & (self.censoring_times[np.minimum(positions, n_times - 1)] == values)
    result[exact] = self.survival_after[positions[exact]]
    return result


def fit_censoring_distribution(
  times: npt.NDArray[np.floating],
  event_observed: npt.NDArray[np.bool_],
) -> CensoringDistribution:
  """Fit the censoring KM curve on development/validation follow-up only."""
  observed_times = np.asarray(times, dtype=np.float64).ravel()
  observed_events = np.asarray(event_observed, dtype=bool).ravel()
  if len(observed_times) != len(observed_events) or len(observed_times) < 2:
    raise ValueError("Censoring reference times/events must have equal length >= 2.")
  if not np.all(np.isfinite(observed_times)) or np.any(observed_times < 0):
    raise ValueError("Censoring reference times must be finite and non-negative.")

  censoring = ~observed_events
  unique_times, counts = np.unique(observed_times[censoring], return_counts=True)
  if not len(unique_times):
    return CensoringDistribution(
      censoring_times=unique_times.astype(np.float64),
      survival_after=np.empty(0, dtype=np.float64),
      n_reference=len(observed_times),
      n_censored=0,
      max_follow_up=float(np.max(observed_times)),
    )

  sorted_times = np.sort(observed_times)
  # Reverse Kaplan-Meier convention (scikit-survival kaplan_meier_estimator
  # with ``reverse=True``): at a time with tied true events and censorings,
  # the true events leave the risk set before the censorings at that time
  # are processed, so the risk denominator counts only the censorings at the
  # tied time plus observations still under follow-up strictly after it.
  n_after_t = len(sorted_times) - np.searchsorted(sorted_times, unique_times, side="right")
  n_at_risk = counts + n_after_t
  survival_after = np.cumprod(1.0 - counts / n_at_risk, dtype=np.float64)
  return CensoringDistribution(
    censoring_times=np.asarray(unique_times, dtype=np.float64),
    survival_after=survival_after,
    n_reference=len(observed_times),
    n_censored=int(censoring.sum()),
    max_follow_up=float(np.max(observed_times)),
  )


def _case_control_masks(
  times: npt.NDArray[np.float64],
  event_observed: npt.NDArray[np.bool_],
  horizon: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
  cases = event_observed & (times <= horizon)
  # Match the established label contract: a non-event observed exactly at the
  # horizon is known to be event-free through that horizon.
  controls = (times > horizon) | (~event_observed & (times == horizon))
  return cases, controls


def cumulative_dynamic_ipcw_auc(
  times: npt.NDArray[np.floating],
  event_observed: npt.NDArray[np.bool_],
  risk_scores: npt.NDArray[np.floating],
  horizon: float,
  censoring_distribution: CensoringDistribution,
) -> tuple[float, dict[str, int]]:
  """Estimate cumulative/dynamic AUROC with inverse censoring weights.

  Cumulative cases are events at or before ``horizon``. Dynamic controls are
  those known to be event-free through the horizon. Patients censored before
  the horizon contribute neither class. Case weights are ``1/G(T_i)`` evaluated
  with the scikit-survival parity convention (post-tie value at exact ties with
  reference censoring times); controls have unit weight, matching the
  cumulative/dynamic IPCW estimator. The censoring curve is a reverse
  Kaplan-Meier estimate whose risk denominator excludes true events at tied
  times, matching ``sksurv.nonparametric.kaplan_meier_estimator(reverse=True)``.
  """
  observed_times = np.asarray(times, dtype=np.float64).ravel()
  observed_events = np.asarray(event_observed, dtype=bool).ravel()
  scores = np.asarray(risk_scores, dtype=np.float64).ravel()
  if not (len(observed_times) == len(observed_events) == len(scores)):
    raise ValueError("Times, event indicators, and scores must have equal length.")
  if (
    len(observed_times) == 0
    or not np.all(np.isfinite(observed_times))
    or np.any(observed_times < 0)
  ):
    raise ValueError("Evaluation times must be a non-empty finite, non-negative array.")
  if not np.all(np.isfinite(scores)):
    raise ValueError("Risk scores must all be finite.")

  cases, controls = _case_control_masks(observed_times, observed_events, horizon)
  known = cases | controls
  n_cases = int(cases.sum())
  n_controls = int(controls.sum())
  n_censored_before_horizon = int((~known).sum())
  if n_cases == 0 or n_controls == 0:
    raise ValueError("IPCW AUROC requires at least one case and one control.")
  if float(np.max(observed_times[cases])) > censoring_distribution.max_follow_up:
    raise ValueError("Censoring reference follow-up does not cover all IPCW case times.")

  g_case = censoring_distribution.case_weight_survival(observed_times[cases])
  if np.any(g_case <= 0):
    raise ValueError("Censoring survival is zero before at least one case time.")
  weights = np.ones(int(known.sum()), dtype=np.float64)
  weights[cases[known]] = 1.0 / g_case
  labels = cases[known].astype(np.int8)
  auc = float(roc_auc_score(labels, scores[known], sample_weight=weights))
  return auc, {
    "n_test": int(len(observed_times)),
    "n_cases": n_cases,
    "n_controls": n_controls,
    "n_censored_before_horizon": n_censored_before_horizon,
  }


def _auc_summary(values: list[float]) -> dict[str, float | None]:
  if not values:
    return {"mean": None, "ci_lower": None, "ci_upper": None}
  arr = np.asarray(values, dtype=np.float64)
  return {
    "mean": float(arr.mean()),
    "ci_lower": float(np.percentile(arr, 2.5)),
    "ci_upper": float(np.percentile(arr, 97.5)),
  }


def censoring_sensitivity_report(
  reference_times: npt.NDArray[np.floating],
  reference_events: npt.NDArray[np.bool_],
  test_times: npt.NDArray[np.floating],
  test_events: npt.NDArray[np.bool_],
  risk_scores: dict[str, npt.NDArray[np.float64]],
  contrasts: dict[str, tuple[str, str]],
  horizon: float,
  n_bootstraps: int = 1000,
  seed: int = 1912,
) -> dict[str, Any]:
  """Compare IPCW AUROC with the complete-case binary AUROC.

  ``contrasts`` maps a result name to ``(base, added)`` score names; each
  reported difference is ``AUROC(added) - AUROC(base)``. The development/
  validation censoring KM is fixed across all patient-bootstrap replicates.
  """
  ref_times = np.asarray(reference_times, dtype=np.float64).ravel()
  ref_events = np.asarray(reference_events, dtype=bool).ravel()
  times = np.asarray(test_times, dtype=np.float64).ravel()
  events = np.asarray(test_events, dtype=bool).ravel()
  if len(ref_times) != len(ref_events) or len(times) != len(events):
    raise ValueError("Reference/test times and event indicators must be paired.")
  if not np.isfinite(horizon) or horizon <= 0:
    raise ValueError("The IPCW horizon must be finite and positive.")
  if n_bootstraps < 1:
    raise ValueError("At least one bootstrap replicate is required.")
  score_arrays = {name: np.asarray(values, dtype=np.float64).ravel() for name, values in risk_scores.items()}
  if any(len(values) != len(times) for values in score_arrays.values()):
    raise ValueError("Every risk-score vector must match the test row count.")
  unknown_names = {
    name for pair in contrasts.values() for name in pair if name not in score_arrays
  }
  if unknown_names:
    raise ValueError(f"Contrasts reference unknown scores: {sorted(unknown_names)}.")

  censoring = fit_censoring_distribution(ref_times, ref_events)
  cases, controls = _case_control_masks(times, events, horizon)
  known = cases | controls
  labels_cc = cases[known].astype(np.int8)
  if labels_cc.size == 0 or labels_cc.min() == labels_cc.max():
    raise ValueError("Complete-case sensitivity requires both cases and controls.")

  model_metrics: dict[str, dict[str, float | int]] = {}
  ipcw_values: dict[str, float] = {}
  complete_case_values: dict[str, float] = {}
  counts: dict[str, int] | None = None
  for name, scores in score_arrays.items():
    ipcw_auc, counts = cumulative_dynamic_ipcw_auc(
      times, events, scores, horizon, censoring,
    )
    complete_case_auc = float(roc_auc_score(labels_cc, scores[known]))
    ipcw_values[name] = ipcw_auc
    complete_case_values[name] = complete_case_auc
    model_metrics[name] = {
      "ipcw_auroc": ipcw_auc,
      "complete_case_auroc": complete_case_auc,
      "difference_ipcw_minus_complete_case": ipcw_auc - complete_case_auc,
    }

  rng = np.random.default_rng(seed)
  case_indices = np.flatnonzero(cases)
  control_indices = np.flatnonzero(controls)
  labels_b = np.concatenate(
    (
      np.ones(len(case_indices), dtype=np.int8),
      np.zeros(len(control_indices), dtype=np.int8),
    ),
  )
  bootstrap_values: dict[str, dict[str, list[float]]] = {
    name: {"ipcw": [], "complete_case": [], "shift": []} for name in contrasts
  }
  skipped = 0
  for _ in range(n_bootstraps):
    sampled_cases = rng.choice(case_indices, size=len(case_indices), replace=True)
    sampled_controls = rng.choice(control_indices, size=len(control_indices), replace=True)
    idx = np.concatenate((sampled_cases, sampled_controls))
    g_case_b = censoring.case_weight_survival(times[sampled_cases])
    if np.any(g_case_b <= 0):
      skipped += 1
      continue
    case_weights_b = np.concatenate(
      (1.0 / g_case_b, np.ones(len(sampled_controls), dtype=np.float64)),
    )

    for contrast_name, (base_name, added_name) in contrasts.items():
      base = score_arrays[base_name][idx]
      added = score_arrays[added_name][idx]
      base_ipcw = float(
        roc_auc_score(labels_b, base, sample_weight=case_weights_b),
      )
      added_ipcw = float(
        roc_auc_score(labels_b, added, sample_weight=case_weights_b),
      )
      base_cc = float(roc_auc_score(labels_b, base))
      added_cc = float(roc_auc_score(labels_b, added))
      ipcw_delta = added_ipcw - base_ipcw
      cc_delta = added_cc - base_cc
      bootstrap_values[contrast_name]["ipcw"].append(ipcw_delta)
      bootstrap_values[contrast_name]["complete_case"].append(cc_delta)
      bootstrap_values[contrast_name]["shift"].append(ipcw_delta - cc_delta)

  contrast_results: dict[str, Any] = {}
  for name, (base_name, added_name) in contrasts.items():
    ipcw_point = ipcw_values[added_name] - ipcw_values[base_name]
    cc_point = complete_case_values[added_name] - complete_case_values[base_name]
    contrast_results[name] = {
      "base_score": base_name,
      "added_score": added_name,
      "ipcw_delta_auroc": {
        "point_estimate": ipcw_point,
        **_auc_summary(bootstrap_values[name]["ipcw"]),
      },
      "complete_case_delta_auroc": {
        "point_estimate": cc_point,
        **_auc_summary(bootstrap_values[name]["complete_case"]),
      },
      "delta_shift_ipcw_minus_complete_case": {
        "point_estimate": ipcw_point - cc_point,
        **_auc_summary(bootstrap_values[name]["shift"]),
      },
    }

  return {
    "method": "cumulative_dynamic_auroc_ipcw_case_weights",
    "horizon": horizon,
    "censoring_reference": "development+validation reverse Kaplan-Meier, held fixed during bootstrap",
    "censoring_estimator": (
      "reverse Kaplan-Meier; risk denominator at tied times excludes true events "
      "(scikit-survival kaplan_meier_estimator reverse=True convention)"
    ),
    "censoring_assumption": "independent censoring estimated without conditioning on features",
    "bootstrap_method": "paired patient bootstrap stratified on observed horizon cases and controls",
    "n_reference": censoring.n_reference,
    "n_censored_reference": censoring.n_censored,
    "models": model_metrics,
    "test_population": counts,
    "contrasts": contrast_results,
    "n_bootstraps": n_bootstraps,
    "n_bootstraps_used": n_bootstraps - skipped,
    "n_skipped_degenerate": skipped,
    "seed": seed,
  }
