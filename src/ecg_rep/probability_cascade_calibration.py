"""Exploratory calibration and decision analysis for probability cascades.

Calibration maps are fitted on a declared calibration split and applied unchanged
to assessment data. Raw cascade mixtures remain separately labeled diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class LogisticRiskMap:
  """A logistic calibration map on the log-odds of one frozen score."""

  method: Literal["logistic", "intercept_only"]
  intercept: float
  slope: float
  n_fit: int
  events_fit: int

  def predict(self, probabilities: npt.ArrayLike) -> npt.NDArray[np.float64]:
    values = _validated_probabilities(probabilities)
    logits = logit(np.clip(values, 1e-8, 1.0 - 1e-8))
    return np.asarray(expit(self.intercept + self.slope * logits), dtype=np.float64)

  def to_dict(self) -> dict[str, object]:
    return asdict(self)


def _validated_binary_outcomes(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
  raw = np.asarray(values)
  if raw.ndim != 1 or len(raw) == 0 or not np.isin(raw, (0, 1)).all():
    raise ValueError("Outcomes must be a non-empty one-dimensional binary array.")
  return np.asarray(raw, dtype=np.int64)


def _validated_probabilities(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
  p = np.asarray(values, dtype=np.float64)
  if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all():
    raise ValueError("Probabilities must be a non-empty one-dimensional finite array.")
  if ((p < 0.0) | (p > 1.0)).any():
    raise ValueError("Probabilities must lie in [0, 1].")
  return p


def _aligned_arrays(y_true: npt.ArrayLike, probabilities: npt.ArrayLike) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
  y = _validated_binary_outcomes(y_true)
  p = _validated_probabilities(probabilities)
  if len(y) != len(p):
    raise ValueError("Outcome and probability rows must align.")
  return y, p


def fit_logistic_risk_map(
  y_true: npt.ArrayLike,
  probabilities: npt.ArrayLike,
  *,
  minimum_slope_events: int = 1,
) -> LogisticRiskMap:
  """Fit ``Y ~ logit(p)`` or an intercept-only map using fit-split labels."""
  y, p = _aligned_arrays(y_true, probabilities)
  events = int(y.sum())
  if events == 0 or events == len(y):
    raise ValueError("Calibration fitting requires both outcome classes.")
  logits = logit(np.clip(p, 1e-8, 1.0 - 1e-8))
  if events < minimum_slope_events or (len(y) - events) < minimum_slope_events:
    prevalence = float(y.mean())
    root = cast(float, brentq(lambda delta: float(expit(logits + delta).mean()) - prevalence, -40.0, 40.0))
    offset = float(root)
    return LogisticRiskMap("intercept_only", offset, 1.0, len(y), events)

  model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
  model.fit(logits.reshape(-1, 1), y)
  return LogisticRiskMap(
    "logistic",
    float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0]),
    float(np.asarray(model.coef_, dtype=np.float64)[0, 0]),
    len(y),
    events,
  )


def fit_branch_risk_maps(
  y_true: npt.ArrayLike,
  p_a: npt.ArrayLike,
  p_b: npt.ArrayLike,
  route_to_b: npt.ArrayLike,
  *,
  minimum_slope_events: int = 50,
  minimum_branch_events: int = 10,
) -> tuple[LogisticRiskMap, LogisticRiskMap, dict[str, int]]:
  """Fit low-A and routed-B maps; sparse branches fall back to a global map."""
  y, pa = _aligned_arrays(y_true, p_a)
  _y_b, pb = _aligned_arrays(y_true, p_b)
  route = np.asarray(route_to_b, dtype=bool)
  if len(route) != len(y):
    raise ValueError("Route mask must align with calibration rows.")
  raw = np.where(route, pb, pa)
  global_map = fit_logistic_risk_map(y, raw, minimum_slope_events=minimum_slope_events)

  low = ~route
  high = route
  low_events = int(y[low].sum())
  high_events = int(y[high].sum())
  support = {
    "n_low_a": int(low.sum()),
    "events_low_a": low_events,
    "n_routed_b": int(high.sum()),
    "events_routed_b": high_events,
  }

  def fit_or_fallback(mask: npt.NDArray[np.bool_], score: npt.NDArray[np.float64], events: int) -> LogisticRiskMap:
    if int(mask.sum()) == 0 or events < minimum_branch_events or events == int(mask.sum()):
      return global_map
    return fit_logistic_risk_map(y[mask], score[mask], minimum_slope_events=minimum_slope_events)

  return fit_or_fallback(low, pa, low_events), fit_or_fallback(high, pb, high_events), support


def apply_branch_risk_maps(
  p_a: npt.ArrayLike,
  p_b: npt.ArrayLike,
  route_to_b: npt.ArrayLike,
  low_map: LogisticRiskMap,
  high_map: LogisticRiskMap,
) -> npt.NDArray[np.float64]:
  """Apply branch maps to offline full-cohort score arrays."""
  pa = _validated_probabilities(p_a)
  pb = _validated_probabilities(p_b)
  route = np.asarray(route_to_b, dtype=bool)
  if len(pa) != len(pb) or len(pa) != len(route):
    raise ValueError("A, B, and route arrays must align.")
  return apply_routed_branch_risk_maps(pa, pb[route], route, low_map, high_map)


def apply_routed_branch_risk_maps(
  p_a: npt.ArrayLike,
  p_b_routed: npt.ArrayLike,
  route_to_b: npt.ArrayLike,
  low_map: LogisticRiskMap,
  high_map: LogisticRiskMap,
) -> npt.NDArray[np.float64]:
  """Apply branch maps when B scores exist only for routed patients.

  ``p_b_routed`` follows the order of ``np.flatnonzero(route_to_b)`` so a
  selective system does not need to manufacture scores for off-route patients.
  """
  pa = _validated_probabilities(p_a)
  route = np.asarray(route_to_b, dtype=bool)
  if route.ndim != 1 or len(pa) != len(route):
    raise ValueError("A probabilities and route mask must align.")
  pb = np.asarray(p_b_routed, dtype=np.float64)
  if pb.ndim != 1 or len(pb) != int(route.sum()):
    raise ValueError("B probabilities must contain exactly one value per routed patient.")
  if not np.isfinite(pb).all() or ((pb < 0.0) | (pb > 1.0)).any():
    raise ValueError("Routed B probabilities must be finite and lie in [0, 1].")
  output = low_map.predict(pa)
  if route.any():
    output[route] = high_map.predict(pb)
  return output


def calibration_metrics(
  y_true: npt.ArrayLike,
  probabilities: npt.ArrayLike,
  *,
  bins: int = 10,
) -> tuple[dict[str, float | int | None], list[dict[str, float | int]]]:
  """Return calibration summaries and equal-count reliability bins."""
  y, p = _aligned_arrays(y_true, probabilities)
  clipped = np.clip(p, 1e-8, 1.0 - 1e-8)
  observed_events = int(y.sum())
  expected_events = float(p.sum())
  logit_p = logit(clipped)
  if 0 < observed_events < len(y):
    slope_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    slope_model.fit(logit_p.reshape(-1, 1), y)
    calibration_intercept = float(np.asarray(slope_model.intercept_, dtype=np.float64).reshape(-1)[0])
    calibration_slope = float(np.asarray(slope_model.coef_, dtype=np.float64)[0, 0])
    prevalence = float(y.mean())
    root = cast(float, brentq(lambda delta: float(expit(logit_p + delta).mean()) - prevalence, -40.0, 40.0))
    intercept_only = float(root)
  else:
    calibration_intercept = None
    calibration_slope = None
    intercept_only = None
  metrics: dict[str, float | int | None] = {
    "n": len(y),
    "events": observed_events,
    "observed_event_rate": float(y.mean()),
    "expected_events": expected_events,
    "observed_expected_ratio": observed_events / expected_events if expected_events > 0 else None,
    "calibration_in_the_large": intercept_only,
    "calibration_intercept": calibration_intercept,
    "calibration_slope": calibration_slope,
    "brier": float(brier_score_loss(y, p)),
    "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
  }
  order = np.argsort(p, kind="stable")
  bin_indices = np.array_split(order, min(max(1, bins), len(y)))
  rows: list[dict[str, float | int]] = []
  for bin_id, indices in enumerate(bin_indices, start=1):
    rows.append({
      "bin": bin_id,
      "n": int(len(indices)),
      "mean_predicted": float(p[indices].mean()),
      "observed_rate": float(y[indices].mean()),
    })
  return metrics, rows


def decision_curve_net_benefit(
  y_true: npt.ArrayLike,
  probabilities: npt.ArrayLike,
  threshold: float,
) -> float:
  """Compute whole-cohort decision-curve net benefit at one action threshold."""
  y, p = _aligned_arrays(y_true, probabilities)
  if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
    raise ValueError("Decision threshold must lie strictly between 0 and 1.")
  action = p >= threshold
  true_positive = int(np.sum(action & (y == 1)))
  false_positive = int(np.sum(action & (y == 0)))
  odds = threshold / (1.0 - threshold)
  return float(true_positive / len(y) - false_positive / len(y) * odds)


def dca_reference_net_benefit(y_true: npt.ArrayLike, threshold: float) -> tuple[float, float]:
  """Return treat-none and treat-all references for one whole cohort."""
  y = _validated_binary_outcomes(y_true)
  if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
    raise ValueError("Decision threshold must lie strictly between 0 and 1.")
  treat_none = 0.0
  prevalence = float(y.mean())
  treat_all = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
  return treat_none, float(treat_all)

