"""Scientific estimands for CIIS vs D-BETA representation comparisons.

Implements the research-plan analysis surface for incremental value, attenuation,
alignment, residual-risk gradients, capacity, and selective A→B deployment.
Measured results require locked probe predictions; definitions remain valid without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import spearmanr

from ecg_rep.probe import (
  compute_auprc,
  compute_auroc,
  compute_binary_log_loss,
  compute_brier_score,
)

PerformanceMetric = Literal["auroc", "auprc", "log_loss", "brier"]
DEFAULT_CAPACITY_FRACTIONS: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10, 0.20)
DEFAULT_AGE_BINS: tuple[tuple[float, float, str], ...] = (
  (18.0, 50.0, "18-49"),
  (50.0, 65.0, "50-64"),
  (65.0, 80.0, "65-79"),
  (80.0, 200.0, "80+"),
)


@dataclass(frozen=True)
class RepresentationGain:
  """Paired gain of representation B over A for one performance measure."""

  metric: PerformanceMetric
  gain: float
  score_a: float
  score_b: float


@dataclass(frozen=True)
class ContextAttenuation:
  """Descriptive attenuation of B-over-A gain after adding shared clinical context."""

  metric: PerformanceMetric
  ecg_only_gain: float
  context_adjusted_gain: float
  attenuation: float


@dataclass(frozen=True)
class BidirectionalIncrement:
  """Asymmetric information increments when adding the other representation."""

  metric: PerformanceMetric
  add_b_to_a: float
  add_a_to_b: float
  add_b_to_a_ci_lower: float | None = None
  add_b_to_a_ci_upper: float | None = None
  add_a_to_b_ci_lower: float | None = None
  add_a_to_b_ci_upper: float | None = None
  interpretation_note: str = (
    "CIIS-derived features appear largely predictively redundant conditional on "
    "D-BETA under the prespecified ridge-logistic probe."
  )


@dataclass(frozen=True)
class AlignmentResult:
  """Spearman alignment of total risk or ECG-specific contributions."""

  spearman_rho: float
  n: int


@dataclass(frozen=True)
class ResidualRiskStratum:
  """Outcome gradient of one model within a risk stratum of the other."""

  stratum_label: str
  n: int
  n_events: int
  event_rate: float
  within_model_low_rate: float
  within_model_high_rate: float


@dataclass(frozen=True)
class DiscordantGroup:
  """Clinically interpretable concordance/discordance quadrant."""

  label: str
  n: int
  n_events: int
  event_rate: float


@dataclass(frozen=True)
class CapacityResult:
  """Fixed-capacity top-risk evaluation (not a probability threshold)."""

  capacity_fraction: float
  n_flagged: int
  n_events_captured: int
  sensitivity: float
  ppv: float
  false_positives_per_event: float | None
  number_needed_to_evaluate: float | None


@dataclass(frozen=True)
class CapacityComparisonResult:
  """Comparative event capture under fixed clinical management capacity."""

  capacity_fraction: float
  capacity_a: CapacityResult
  capacity_b: CapacityResult
  delta_events_captured: int
  delta_sensitivity: float
  delta_ppv: float
  delta_events_ci_lower: float | None = None
  delta_events_ci_upper: float | None = None
  delta_sensitivity_ci_lower: float | None = None
  delta_sensitivity_ci_upper: float | None = None
  delta_ppv_ci_lower: float | None = None
  delta_ppv_ci_upper: float | None = None


@dataclass(frozen=True)
class CascadePolicyResult:
  """Selective A→B deployment under a D-BETA utilization budget."""

  utilization_fraction: float
  n_total: int
  n_sent_to_b: int
  n_events_total: int
  n_events_captured: int
  n_events_missed_vs_full_b: int
  relative_event_capture_ratio: float
  fraction_of_full_b_events_recovered: float  # alias for backwards compatibility
  capacity_fraction: float = 0.05
  n_flagged_at_capacity: int = 0
  sensitivity_at_capacity: float = float("nan")
  ppv_at_capacity: float = float("nan")
  number_needed_to_evaluate_at_capacity: float | None = None


@dataclass(frozen=True)
class SubgroupGain:
  """Heterogeneity target: B-over-A gain within a prespecified subgroup."""

  subgroup_variable: str
  subgroup_level: str
  metric: PerformanceMetric
  gain: float
  n: int
  n_events: int
  ci_lower: float | None = None
  ci_upper: float | None = None


@dataclass(frozen=True)
class SubgroupContrast:
  """Contrast difference between two subgroup representation gains (Delta_s1 - Delta_s2)."""

  subgroup_variable: str
  level_a: str
  level_b: str
  metric: PerformanceMetric
  contrast: float
  ci_lower: float | None = None
  ci_upper: float | None = None
  n_bootstraps: int = 1000


def _metric_value(
  metric: PerformanceMetric,
  y_true: npt.NDArray[np.int64],
  scores: npt.NDArray[np.float64],
) -> float:
  fn = {
    "auroc": compute_auroc,
    "auprc": compute_auprc,
    "log_loss": compute_binary_log_loss,
    "brier": compute_brier_score,
  }[metric]
  return float(fn(y_true, scores))


def representation_gain(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  metric: PerformanceMetric = "auroc",
) -> RepresentationGain:
  """Compute Δ = P(B) - P(A) on the same patients."""
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  score_a = _metric_value(metric, yt, sa)
  score_b = _metric_value(metric, yt, sb)
  # For proper scoring rules, lower is better; report B-minus-A still, with sign interpreted by caller.
  return RepresentationGain(metric=metric, gain=score_b - score_a, score_a=score_a, score_b=score_b)


def context_attenuation(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  a_ecg: npt.NDArray[np.float64] | Sequence[float],
  b_ecg: npt.NDArray[np.float64] | Sequence[float],
  a_ctx: npt.NDArray[np.float64] | Sequence[float],
  b_ctx: npt.NDArray[np.float64] | Sequence[float],
  metric: PerformanceMetric = "auroc",
) -> ContextAttenuation:
  """Descriptive attenuation = ECG-only gain − context-adjusted gain."""
  ecg = representation_gain(y_true, a_ecg, b_ecg, metric=metric)
  ctx = representation_gain(y_true, a_ctx, b_ctx, metric=metric)
  return ContextAttenuation(
    metric=metric,
    ecg_only_gain=ecg.gain,
    context_adjusted_gain=ctx.gain,
    attenuation=ecg.gain - ctx.gain,
  )


def _stratified_bootstrap_indices(
  y: npt.NDArray[np.int64],
  rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
  """Generate bootstrap sample indices stratified by binary outcome y."""
  pos_idx = np.where(y == 1)[0]
  neg_idx = np.where(y == 0)[0]
  if len(pos_idx) == 0 or len(neg_idx) == 0:
    return rng.integers(0, len(y), size=len(y))
  boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
  boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
  return np.concatenate([boot_pos, boot_neg])


def _stratified_contrast_bootstrap_indices(
  y: npt.NDArray[np.int64],
  labels: npt.NDArray[np.str_],
  level_a: str,
  level_b: str,
  rng: np.random.Generator,
) -> npt.NDArray[np.int64]:
  """Generate bootstrap indices stratified by (outcome, group) for contrast estimation."""
  strata_indices: list[npt.NDArray[np.int64]] = []
  for lvl in (level_a, level_b):
    for out_val in (0, 1):
      idx = np.where((labels == lvl) & (y == out_val))[0]
      if len(idx) > 0:
        strata_indices.append(rng.choice(idx, size=len(idx), replace=True))
  other_idx = np.where((labels != level_a) & (labels != level_b))[0]
  if len(other_idx) > 0:
    strata_indices.append(rng.choice(other_idx, size=len(other_idx), replace=True))
  return np.concatenate(strata_indices) if strata_indices else rng.integers(0, len(y), size=len(y))


def bidirectional_increments(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  scores_ab: npt.NDArray[np.float64] | Sequence[float],
  metric: PerformanceMetric = "auroc",
  n_bootstraps: int = 0,
  seed: int = 42,
) -> BidirectionalIncrement:
  """Increments from M_A→M_AB and M_B→M_AB with optional bootstrap uncertainty."""
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  sab = np.asarray(scores_ab, dtype=np.float64)
  a = _metric_value(metric, yt, sa)
  b = _metric_value(metric, yt, sb)
  ab = _metric_value(metric, yt, sab)
  inc_b_to_a = ab - a
  inc_a_to_b = ab - b

  ci_b_to_a_lo = None
  ci_b_to_a_hi = None
  ci_a_to_b_lo = None
  ci_a_to_b_hi = None

  if n_bootstraps > 0 and len(yt) >= 5 and len(np.unique(yt)) > 1:
    rng = np.random.default_rng(seed)
    diffs_b_to_a = np.zeros(n_bootstraps, dtype=np.float64)
    diffs_a_to_b = np.zeros(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
      idx = _stratified_bootstrap_indices(yt, rng)
      yt_b = yt[idx]
      ab_b = _metric_value(metric, yt_b, sab[idx])
      a_b = _metric_value(metric, yt_b, sa[idx])
      b_b = _metric_value(metric, yt_b, sb[idx])
      diffs_b_to_a[i] = ab_b - a_b
      diffs_a_to_b[i] = ab_b - b_b

    ci_b_to_a_lo = float(np.percentile(diffs_b_to_a, 2.5))
    ci_b_to_a_hi = float(np.percentile(diffs_b_to_a, 97.5))
    ci_a_to_b_lo = float(np.percentile(diffs_a_to_b, 2.5))
    ci_a_to_b_hi = float(np.percentile(diffs_a_to_b, 97.5))

  return BidirectionalIncrement(
    metric=metric,
    add_b_to_a=inc_b_to_a,
    add_a_to_b=inc_a_to_b,
    add_b_to_a_ci_lower=ci_b_to_a_lo,
    add_b_to_a_ci_upper=ci_b_to_a_hi,
    add_a_to_b_ci_lower=ci_a_to_b_lo,
    add_a_to_b_ci_upper=ci_a_to_b_hi,
  )



def spearman_alignment(
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
) -> AlignmentResult:
  """Spearman correlation of two score vectors (total or ECG-specific)."""
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  if len(sa) != len(sb):
    raise ValueError("Alignment requires equal-length score vectors.")
  if len(sa) < 2:
    return AlignmentResult(spearman_rho=float("nan"), n=len(sa))
  rho, _ = spearmanr(sa, sb)
  return AlignmentResult(spearman_rho=float(rho), n=len(sa))


def residual_risk_gradient(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  outer_scores: npt.NDArray[np.float64] | Sequence[float],
  inner_scores: npt.NDArray[np.float64] | Sequence[float],
  n_strata: int = 4,
) -> tuple[ResidualRiskStratum, ...]:
  """Event-rate gradient of ``inner`` within quantile strata of ``outer``."""
  yt = np.asarray(y_true, dtype=np.int64)
  outer = np.asarray(outer_scores, dtype=np.float64)
  inner = np.asarray(inner_scores, dtype=np.float64)
  if n_strata < 2:
    raise ValueError("n_strata must be at least 2.")
  quantiles = np.quantile(outer, np.linspace(0.0, 1.0, n_strata + 1))
  strata: list[ResidualRiskStratum] = []
  for i in range(n_strata):
    lo, hi = quantiles[i], quantiles[i + 1]
    if i == n_strata - 1:
      mask = (outer >= lo) & (outer <= hi)
    else:
      mask = (outer >= lo) & (outer < hi)
    if not np.any(mask):
      continue
    y_s = yt[mask]
    inner_s = inner[mask]
    mid = float(np.median(inner_s))
    low_mask = inner_s <= mid
    high_mask = inner_s > mid
    low_rate = float(np.mean(y_s[low_mask])) if np.any(low_mask) else float("nan")
    high_rate = float(np.mean(y_s[high_mask])) if np.any(high_mask) else float("nan")
    strata.append(
      ResidualRiskStratum(
        stratum_label=f"Q{i + 1}",
        n=int(mask.sum()),
        n_events=int(y_s.sum()),
        event_rate=float(np.mean(y_s)),
        within_model_low_rate=low_rate,
        within_model_high_rate=high_rate,
      )
    )
  return tuple(strata)


def discordance_groups(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  high_quantile: float = 0.8,
) -> tuple[DiscordantGroup, ...]:
  """Concordant/discordant groups using high-risk quantile cutpoints."""
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  a_cut = float(np.quantile(sa, high_quantile))
  b_cut = float(np.quantile(sb, high_quantile))
  a_high = sa >= a_cut
  b_high = sb >= b_cut
  definitions = (
    ("concordant_low", (~a_high) & (~b_high)),
    ("a_low_b_high", (~a_high) & b_high),
    ("a_high_b_low", a_high & (~b_high)),
    ("concordant_high", a_high & b_high),
  )
  groups: list[DiscordantGroup] = []
  for label, mask in definitions:
    y_g = yt[mask]
    n = int(mask.sum())
    events = int(y_g.sum()) if n else 0
    groups.append(
      DiscordantGroup(
        label=label,
        n=n,
        n_events=events,
        event_rate=float(events / n) if n else float("nan"),
      )
    )
  return tuple(groups)


def capacity_evaluation(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores: npt.NDArray[np.float64] | Sequence[float],
  capacity_fraction: float,
) -> CapacityResult:
  """Top ``capacity_fraction`` predicted-risk patients under fixed capacity."""
  if not 0.0 < capacity_fraction <= 1.0:
    raise ValueError("capacity_fraction must be in (0, 1].")
  yt = np.asarray(y_true, dtype=np.int64)
  sc = np.asarray(scores, dtype=np.float64)
  n = len(yt)
  n_flagged = max(1, int(round(n * capacity_fraction)))
  order = np.argsort(-sc)
  flagged = order[:n_flagged]
  captured = int(yt[flagged].sum())
  total_events = int(yt.sum())
  fp = n_flagged - captured
  sens = float(captured / total_events) if total_events else float("nan")
  ppv = float(captured / n_flagged) if n_flagged else float("nan")
  return CapacityResult(
    capacity_fraction=capacity_fraction,
    n_flagged=n_flagged,
    n_events_captured=captured,
    sensitivity=sens,
    ppv=ppv,
    false_positives_per_event=(float(fp / captured) if captured else None),
    number_needed_to_evaluate=(float(n_flagged / captured) if captured else None),
  )


def capacity_comparison(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  capacity_fraction: float,
  n_bootstraps: int = 0,
  seed: int = 42,
) -> CapacityComparisonResult:
  """Compare top ``capacity_fraction`` predicted-risk capture between representations A and B.

  Computes capacity_a, capacity_b, paired event difference (B - A), sensitivity difference,
  and PPV difference, with optional stratified bootstrap confidence intervals.
  """
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)

  cap_a = capacity_evaluation(yt, sa, capacity_fraction)
  cap_b = capacity_evaluation(yt, sb, capacity_fraction)

  delta_events = cap_b.n_events_captured - cap_a.n_events_captured
  delta_sens = cap_b.sensitivity - cap_a.sensitivity
  delta_ppv = cap_b.ppv - cap_a.ppv

  ci_events_lo = None
  ci_events_hi = None
  ci_sens_lo = None
  ci_sens_hi = None
  ci_ppv_lo = None
  ci_ppv_hi = None

  if n_bootstraps > 0 and len(yt) >= 10 and len(np.unique(yt)) > 1:
    rng = np.random.default_rng(seed)
    diff_events_arr = np.zeros(n_bootstraps, dtype=np.float64)
    diff_sens_arr = np.zeros(n_bootstraps, dtype=np.float64)
    diff_ppv_arr = np.zeros(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
      idx = _stratified_bootstrap_indices(yt, rng)
      yt_b = yt[idx]
      c_a = capacity_evaluation(yt_b, sa[idx], capacity_fraction)
      c_b = capacity_evaluation(yt_b, sb[idx], capacity_fraction)
      diff_events_arr[i] = float(c_b.n_events_captured - c_a.n_events_captured)
      diff_sens_arr[i] = float(c_b.sensitivity - c_a.sensitivity)
      diff_ppv_arr[i] = float(c_b.ppv - c_a.ppv)

    ci_events_lo = float(np.percentile(diff_events_arr, 2.5))
    ci_events_hi = float(np.percentile(diff_events_arr, 97.5))
    ci_sens_lo = float(np.percentile(diff_sens_arr, 2.5))
    ci_sens_hi = float(np.percentile(diff_sens_arr, 97.5))
    ci_ppv_lo = float(np.percentile(diff_ppv_arr, 2.5))
    ci_ppv_hi = float(np.percentile(diff_ppv_arr, 97.5))

  return CapacityComparisonResult(
    capacity_fraction=capacity_fraction,
    capacity_a=cap_a,
    capacity_b=cap_b,
    delta_events_captured=delta_events,
    delta_sensitivity=delta_sens,
    delta_ppv=delta_ppv,
    delta_events_ci_lower=ci_events_lo,
    delta_events_ci_upper=ci_events_hi,
    delta_sensitivity_ci_lower=ci_sens_lo,
    delta_sensitivity_ci_upper=ci_sens_hi,
    delta_ppv_ci_lower=ci_ppv_lo,
    delta_ppv_ci_upper=ci_ppv_hi,
  )


def risk_stratified_incremental_value(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  conventional_risk: npt.NDArray[np.float64] | Sequence[float],
  cutoffs: Sequence[float],
  capacity_fraction: float = 0.05,
  top_fraction: float = 0.20,
  n_bootstraps: int = 0,
  seed: int = 42,
) -> list[dict[str, Any]]:
  """Post hoc conditional incremental value of B over A within conventional-risk strata.

  Round 05 (audit 2026-10-16, finding 1): strata are defined by validation-derived
  cutoffs on the conventional model's (C0 + A) predicted risk and never use test
  outcomes. Within each stratum this reports discrimination (supporting measure),
  top-quintile reclassification cells with observed mortality, and how each model's
  GLOBAL fixed-capacity review list distributes across strata (additional/missed
  death identities). All thresholds are outcome-free.
  """
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  rc = np.asarray(conventional_risk, dtype=np.float64)
  edges = [-np.inf, *cutoffs, np.inf]
  n = len(yt)
  n_flagged = max(1, int(round(n * capacity_fraction)))
  list_a = np.argsort(-sa)[:n_flagged]
  list_b = np.argsort(-sb)[:n_flagged]
  in_list_a = np.zeros(n, dtype=bool)
  in_list_a[list_a] = True
  in_list_b = np.zeros(n, dtype=bool)
  in_list_b[list_b] = True

  rows: list[dict[str, Any]] = []
  labels = ("lower", "intermediate", "higher")
  for i, (label, lo, hi) in enumerate(zip(labels, edges[:-1], edges[1:])):
    mask = (rc > lo) & (rc <= hi) if i > 0 else (rc <= hi)
    idx = np.where(mask)[0]
    if len(idx) == 0:
      continue
    ys, as_, bs_, cs_ = yt[idx], sa[idx], sb[idx], rc[idx]
    n_ev = int(ys.sum())
    auroc_a = compute_auroc(ys, as_)
    auroc_b = compute_auroc(ys, bs_)
    delta = auroc_b - auroc_a
    ci_lo = ci_hi = None
    n_used = 0
    if n_bootstraps > 0 and len(ys) >= 10 and 0 < n_ev < len(ys):
      rng = np.random.default_rng(seed)
      diffs = np.empty(n_bootstraps, dtype=np.float64)
      for b in range(n_bootstraps):
        bidx = _stratified_bootstrap_indices(ys, rng)
        diffs[b] = compute_auroc(ys[bidx], bs_[bidx]) - compute_auroc(ys[bidx], as_[bidx])
      ci_lo = float(np.percentile(diffs, 2.5))
      ci_hi = float(np.percentile(diffs, 97.5))
      n_used = n_bootstraps
    # Within-stratum top-quintile flags per model (outcome-free thresholds).
    thr_a = np.quantile(as_, 1.0 - top_fraction)
    thr_b = np.quantile(bs_, 1.0 - top_fraction)
    top_a = as_ >= thr_a
    top_b = bs_ >= thr_b
    cells: dict[str, Any] = {}
    for name, (m1, m2) in {
      "a_top_b_top": (top_a, top_b),
      "a_top_b_not": (top_a, ~top_b),
      "a_not_b_top": (~top_a, top_b),
      "a_not_b_not": (~top_a, ~top_b),
    }.items():
      sel = m1 & m2
      cells[name] = {
        "n": int(sel.sum()),
        "n_events": int(ys[sel].sum()),
        "observed_mortality": float(ys[sel].mean()) if sel.any() else None,
      }
    # Global fixed-capacity list movement restricted to this stratum.
    sel_a = in_list_a[idx]
    sel_b = in_list_b[idx]
    rows.append({
      "stratum": label,
      "cutoff_lower": None if i == 0 else float(edges[i]),
      "cutoff_upper": None if i == len(labels) - 1 else float(edges[i + 1]),
      "n": int(len(idx)),
      "n_events": n_ev,
      "observed_mortality": float(ys.mean()),
      "median_conventional_risk": float(np.median(cs_)),
      "auroc_a": auroc_a,
      "auroc_b": auroc_b,
      "delta_auroc": delta,
      "delta_ci_lower": ci_lo,
      "delta_ci_upper": ci_hi,
      "n_bootstraps_used": n_used,
      "seed": seed if n_used else None,
      "reclassification_top_quintile": cells,
      "capacity_list": {
        "capacity_fraction": capacity_fraction,
        "flagged_by_a": int(sel_a.sum()),
        "flagged_by_b": int(sel_b.sum()),
        "events_captured_by_a": int(ys[sel_a].sum()),
        "events_captured_by_b": int(ys[sel_b].sum()),
        "events_b_only": int(ys[sel_b & ~sel_a].sum()),
        "events_a_only": int(ys[sel_a & ~sel_b].sum()),
        "patients_b_only": int((sel_b & ~sel_a).sum()),
        "patients_a_only": int((sel_a & ~sel_b).sum()),
      },
      "analysis_status": "post_hoc",
    })
  return rows


def fixed_capacity_metric_rows(
  y_true: npt.NDArray[np.int64],
  scores_by_model: dict[str, npt.NDArray[np.float64]],
  models: Sequence[str],
  capacity_fractions: tuple[float, ...] = DEFAULT_CAPACITY_FRACTIONS,
) -> list[dict[str, Any]]:
  """Plan-section-12 fixed-capacity evaluation for each named model.

  ``scores_by_model`` must contain locked-test predictions for every named
  model; capacity fractions are fixed review budgets, not probability thresholds.
  """
  rows: list[dict[str, Any]] = []
  for model in models:
    if model not in scores_by_model:
      raise ValueError(f"No scores provided for capacity model '{model}'.")
    for fraction in capacity_fractions:
      result = capacity_evaluation(y_true, scores_by_model[model], fraction)
      rows.append({
        "model": model,
        "capacity_fraction": result.capacity_fraction,
        "n_flagged": result.n_flagged,
        "n_events_captured": result.n_events_captured,
        "n_events_total": int(y_true.sum()),
        "sensitivity": result.sensitivity,
        "ppv": result.ppv,
        "false_positives_per_event": result.false_positives_per_event,
        "number_needed_to_evaluate": result.number_needed_to_evaluate,
      })
  return rows


def fixed_capacity_contrast_rows(
  y_true: npt.NDArray[np.int64],
  scores_by_model: dict[str, npt.NDArray[np.float64]],
  contrast_pairs: Sequence[tuple[str, str, str]],
  capacity_fractions: tuple[float, ...] = DEFAULT_CAPACITY_FRACTIONS,
  n_bootstraps: int = 200,
  seed: int = 42,
) -> list[dict[str, Any]]:
  """Paired fixed-capacity differences with stratified bootstrap intervals."""
  rows: list[dict[str, Any]] = []
  for base, comparison, name in contrast_pairs:
    if base not in scores_by_model or comparison not in scores_by_model:
      raise ValueError(f"Contrast '{name}' references unknown model scores.")
    for fraction_index, fraction in enumerate(capacity_fractions):
      result = capacity_comparison(
        y_true, scores_by_model[base], scores_by_model[comparison], fraction,
        n_bootstraps=n_bootstraps, seed=seed + fraction_index,
      )
      rows.append({
        "contrast": name,
        "capacity_fraction": fraction,
        "events_base": result.capacity_a.n_events_captured,
        "events_comparison": result.capacity_b.n_events_captured,
        "delta_events_captured": result.delta_events_captured,
        "delta_events_ci_low": result.delta_events_ci_lower,
        "delta_events_ci_high": result.delta_events_ci_upper,
        "delta_sensitivity": result.delta_sensitivity,
        "delta_sensitivity_ci_low": result.delta_sensitivity_ci_lower,
        "delta_sensitivity_ci_high": result.delta_sensitivity_ci_upper,
        "delta_ppv": result.delta_ppv,
        "delta_ppv_ci_low": result.delta_ppv_ci_lower,
        "delta_ppv_ci_high": result.delta_ppv_ci_upper,
        "n_bootstraps": n_bootstraps,
        "models_refit_inside_bootstrap": False,
      })
  return rows


def selective_a_to_b_cascade(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  utilization_fraction: float,
  capacity_fraction: float = 0.05,
) -> CascadePolicyResult:
  """Gate D-BETA to the top A-risk fraction, then flag top capacity under blended scores.

  Patients not sent to B keep score A; patients sent to B use score B. Full-B
  capacity capture is the reference for missed events.
  """
  if not 0.0 < utilization_fraction <= 1.0:
    raise ValueError("utilization_fraction must be in (0, 1].")
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  n = len(yt)
  n_send = max(1, int(round(n * utilization_fraction)))
  send_idx = np.argsort(-sa)[:n_send]
  send_mask = np.zeros(n, dtype=bool)
  send_mask[send_idx] = True
  blended = np.where(send_mask, sb, sa)

  n_flagged = max(1, int(round(n * capacity_fraction)))
  full_b_flagged = set(np.argsort(-sb)[:n_flagged])
  cascade_flagged = set(np.argsort(-blended)[:n_flagged])

  event_indices = set(np.where(yt == 1)[0])
  events_full_b = full_b_flagged.intersection(event_indices)
  events_cascade = cascade_flagged.intersection(event_indices)

  # Event-set difference: deaths flagged by full B but missed by cascade policy
  # | E_{flagged by full B} \ E_{flagged by cascade} |
  events_missed_vs_full_b = events_full_b - events_cascade
  missed_count = len(events_missed_vs_full_b)

  n_full_b_events = len(events_full_b)
  n_cascade_events = len(events_cascade)
  relative_ratio = (
    float(n_cascade_events / n_full_b_events)
    if n_full_b_events > 0
    else float("nan")
  )
  events_recovered = events_full_b.intersection(events_cascade)
  fraction_recovered = (
    float(len(events_recovered) / n_full_b_events)
    if n_full_b_events > 0
    else float("nan")
  )

  # Harmonized cascade reporting (plan A4): the same final-capacity metrics the
  # fixed-capacity table reports, attached to each cascade operating point.
  nne = float(n_flagged / n_cascade_events) if n_cascade_events else None
  return CascadePolicyResult(
    utilization_fraction=utilization_fraction,
    n_total=n,
    n_sent_to_b=n_send,
    n_events_total=int(yt.sum()),
    n_events_captured=n_cascade_events,
    n_events_missed_vs_full_b=missed_count,
    relative_event_capture_ratio=relative_ratio,
    fraction_of_full_b_events_recovered=fraction_recovered,
    capacity_fraction=capacity_fraction,
    n_flagged_at_capacity=n_flagged,
    sensitivity_at_capacity=float(n_cascade_events / yt.sum()) if yt.sum() else float("nan"),
    ppv_at_capacity=float(n_cascade_events / n_flagged) if n_flagged else float("nan"),
    number_needed_to_evaluate_at_capacity=nne,
  )


def assign_age_subgroup(
  ages: npt.NDArray[np.float64] | Sequence[float],
  bins: tuple[tuple[float, float, str], ...] = DEFAULT_AGE_BINS,
) -> npt.NDArray[np.str_]:
  """Map ages into prespecified closed-open bins ``[lo, hi)``."""
  age_arr = np.asarray(ages, dtype=np.float64)
  labels = np.full(len(age_arr), "unknown", dtype=object)
  for lo, hi, name in bins:
    mask = (age_arr >= lo) & (age_arr < hi)
    labels[mask] = name
  return labels.astype(str)


def collapse_race_ethnicity(
  raw: npt.NDArray[np.str_] | npt.NDArray[np.object_] | Sequence[str | None],
) -> npt.NDArray[np.str_]:
  """Collapse MIMIC admission race strings to a limited subgroup set."""
  out: list[str] = []
  for value in raw:
    s = str(value or "").strip().upper()
    if not s or s in {"UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"}:
      out.append("unknown_other")
    elif "BLACK" in s or "AFRICAN" in s:
      out.append("black")
    elif "HISPANIC" in s or "LATINO" in s or "SOUTH AMERICAN" in s:
      out.append("hispanic_latino")
    elif "ASIAN" in s:
      out.append("asian")
    elif s.startswith("WHITE") or "PORTUGUESE" in s:
      out.append("white")
    else:
      out.append("unknown_other")
  return np.asarray(out, dtype=np.str_)


def subject_race_from_admissions(admissions_df: pl.DataFrame) -> pl.DataFrame:
  """Earliest available recorded admission race per subject (may postdate the index ECG); descriptive subgroup labeling only, never a predictor."""
  if "race" not in admissions_df.columns:
    raise ValueError("admissions_df missing race column")
  ordered = admissions_df.filter(pl.col("race").is_not_null())
  if "admittime" in ordered.columns:
    ordered = ordered.sort(["subject_id", "admittime"])
  grouped = ordered.group_by("subject_id").agg(pl.col("race").first().alias("race_raw"))
  groups = collapse_race_ethnicity(grouped["race_raw"].to_list())
  return grouped.with_columns(pl.Series("race_group", groups))


def subgroup_representation_gains(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  subgroup_labels: npt.NDArray[np.str_] | Sequence[str],
  subgroup_variable: str,
  metric: PerformanceMetric = "auroc",
  n_bootstraps: int = 0,
  seed: int = 42,
) -> tuple[SubgroupGain, ...]:
  """Estimate B-over-A gain within each subgroup level with optional bootstrap uncertainty."""
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  labels = np.asarray(subgroup_labels)
  gains: list[SubgroupGain] = []
  for level in sorted(set(labels.tolist())):
    mask = labels == level
    n_sub = int(mask.sum())
    if n_sub < 2:
      continue
    yt_sub = yt[mask]
    sa_sub = sa[mask]
    sb_sub = sb[mask]
    g = representation_gain(yt_sub, sa_sub, sb_sub, metric=metric)

    ci_lo = None
    ci_hi = None
    if n_bootstraps > 0 and n_sub >= 5 and len(np.unique(yt_sub)) > 1:
      rng = np.random.default_rng(seed)
      boot_diffs = np.zeros(n_bootstraps, dtype=np.float64)
      for b_i in range(n_bootstraps):
        b_idx = _stratified_bootstrap_indices(yt_sub, rng)
        yt_b = yt_sub[b_idx]
        sa_b = sa_sub[b_idx]
        sb_b = sb_sub[b_idx]
        g_b = representation_gain(yt_b, sa_b, sb_b, metric=metric)
        boot_diffs[b_i] = g_b.gain
      ci_lo = float(np.percentile(boot_diffs, 2.5))
      ci_hi = float(np.percentile(boot_diffs, 97.5))

    gains.append(
      SubgroupGain(
        subgroup_variable=subgroup_variable,
        subgroup_level=str(level),
        metric=metric,
        gain=g.gain,
        n=n_sub,
        n_events=int(yt_sub.sum()),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
      )
    )
  return tuple(gains)


def subgroup_representation_contrasts(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  scores_a: npt.NDArray[np.float64] | Sequence[float],
  scores_b: npt.NDArray[np.float64] | Sequence[float],
  subgroup_labels: npt.NDArray[np.str_] | Sequence[str],
  subgroup_variable: str,
  level_a: str,
  level_b: str,
  metric: PerformanceMetric = "auroc",
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> SubgroupContrast:
  """Estimate paired contrast difference Delta_{level_a} - Delta_{level_b} with bootstrap uncertainty."""
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  labels = np.asarray(subgroup_labels)

  mask_a = labels == level_a
  mask_b = labels == level_b

  gain_a = (
    representation_gain(yt[mask_a], sa[mask_a], sb[mask_a], metric=metric).gain
    if int(mask_a.sum()) >= 2 and len(np.unique(yt[mask_a])) > 1
    else 0.0
  )
  gain_b = (
    representation_gain(yt[mask_b], sa[mask_b], sb[mask_b], metric=metric).gain
    if int(mask_b.sum()) >= 2 and len(np.unique(yt[mask_b])) > 1
    else 0.0
  )
  point_contrast = gain_a - gain_b

  ci_lo = None
  ci_hi = None
  if (
    n_bootstraps > 0
    and int(mask_a.sum()) >= 2
    and int(mask_b.sum()) >= 2
    and len(np.unique(yt[mask_a])) > 1
    and len(np.unique(yt[mask_b])) > 1
  ):
    rng = np.random.default_rng(seed)
    contrast_diffs = np.zeros(n_bootstraps, dtype=np.float64)
    for b_i in range(n_bootstraps):
      b_idx = _stratified_contrast_bootstrap_indices(yt, labels, level_a, level_b, rng)
      yt_s = yt[b_idx]
      sa_s = sa[b_idx]
      sb_s = sb[b_idx]
      lab_s = labels[b_idx]

      ma = lab_s == level_a
      mb = lab_s == level_b
      ga = representation_gain(yt_s[ma], sa_s[ma], sb_s[ma], metric=metric).gain
      gb = representation_gain(yt_s[mb], sa_s[mb], sb_s[mb], metric=metric).gain
      contrast_diffs[b_i] = ga - gb

    ci_lo = float(np.percentile(contrast_diffs, 2.5))
    ci_hi = float(np.percentile(contrast_diffs, 97.5))

  return SubgroupContrast(
    subgroup_variable=subgroup_variable,
    level_a=level_a,
    level_b=level_b,
    metric=metric,
    contrast=point_contrast,
    ci_lower=ci_lo,
    ci_upper=ci_hi,
    n_bootstraps=n_bootstraps,
  )

