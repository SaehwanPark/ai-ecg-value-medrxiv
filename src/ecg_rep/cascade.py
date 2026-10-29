"""Shared selective A→B cascade policy machinery (MIMIC and CODE-15%).

Extracted from ``external/code15_deployment.py`` so the MIMIC audit-response
sensitivity (rank-normalized cascade beside the mixed-score reference, with
matched random-routing controls) can reuse the exact same policy definitions
and harmonized reporting fields as CODE-15% (collaborator item A4; audit
2026-10-05 item 1).

Policies
--------
- The **reference mixed-score policy** (route top A-risk ``u`` to B, blend
  raw probabilities) is preserved — including its known probability/rank
  scale-mismatch pathology — never reselected from locked-test results.
- A **rank-normalized policy** maps A and B scores through ECDFs fitted on
  development/validation rows only before blending.
- Pure-A, pure-B and seeded random-routing references.

Metrics distinguish total events captured, event-identity overlap with
full-B capture, identity recovery, relative capture ratio (may exceed
100%), full-B events missed, and the final-capacity sensitivity/PPV/NNE.
Captured deaths are *event enrichment*, never proven prevention or
treatment benefit. No clinical-safety or cost-effectiveness claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

#: Event counts below this value are suppressed in public outputs.
MIN_CELL_COUNT = 11

#: B evaluation budgets (fractions of the cohort routed to D-BETA scoring).
UTILIZATION_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 1.0)

#: Primary clinical review capacity and predeclared sensitivity capacities.
PRIMARY_CAPACITY = 0.05
SENSITIVITY_CAPACITIES: tuple[float, ...] = (0.025, 0.10)


class ECDFRankCalibrator:
  """Monotone rank map fitted on development/validation scores only."""

  def __init__(self) -> None:
    self._sorted: npt.NDArray[np.float64] | None = None

  def fit(self, scores: npt.NDArray[np.floating]) -> "ECDFRankCalibrator":
    arr = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    if len(arr) < 2:
      raise ValueError("Rank calibration needs at least 2 development scores.")
    self._sorted = arr
    return self

  def transform(self, scores: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Map scores to their dev/val percentile, clipped to (0, 1) exclusive."""
    if self._sorted is None:
      raise RuntimeError("ECDFRankCalibrator must be fitted before transform.")
    x = np.asarray(scores, dtype=np.float64).ravel()
    ranks = (np.searchsorted(self._sorted, x, side="left") + 0.5) / len(self._sorted)
    return np.clip(ranks, 0.5 / len(self._sorted), 1.0 - 0.5 / len(self._sorted))


def cascade_events_captured(
  y: npt.NDArray[np.integer],
  blended: npt.NDArray[np.floating],
  capacity_fraction: float,
) -> tuple[set[int], set[int]]:
  """Return (flagged indices, flagged event indices) for one blended ranking."""
  yt = np.asarray(y, dtype=np.int64)
  n_flagged = max(1, int(round(len(yt) * capacity_fraction)))
  flagged = set(np.argsort(-np.asarray(blended), kind="stable")[:n_flagged])
  return flagged, flagged & set(np.where(yt == 1)[0])


def evaluate_cascade_policy(
  y: npt.NDArray[np.integer],
  scores_a: npt.NDArray[np.floating],
  scores_b: npt.NDArray[np.floating],
  blended: npt.NDArray[np.floating],
  n_b_evaluations: int,
  capacity_fraction: float,
  label: str,
  utilization: float,
  min_cell: int = MIN_CELL_COUNT,
) -> dict[str, Any]:
  """One operating point of one prespecified cascade policy."""
  yt = np.asarray(y, dtype=np.int64)
  n_flagged = max(1, int(round(len(yt) * capacity_fraction)))
  _, full_b_events = cascade_events_captured(yt, scores_b, capacity_fraction)
  _, cascade_events = cascade_events_captured(yt, blended, capacity_fraction)
  n_cascade = len(cascade_events)
  shared_event_ids = cascade_events & full_b_events
  relative = float(n_cascade / len(full_b_events)) if full_b_events else float("nan")

  events = int(yt.sum())
  # Harmonized cascade reporting (plan A4): final-capacity sensitivity/PPV/NNE
  # beside the identity metrics. Suppressed together with event counts because
  # a published PPV or NNE with a known n_flagged would round-trip a
  # suppressed small event count.
  events_public = events >= min_cell
  captured = len(cascade_events)
  suppressed = "suppressed(<11)"
  ppv = float(captured / n_flagged) if (events_public and n_flagged) else suppressed
  sensitivity = float(captured / events) if events_public and events else suppressed
  nne = float(n_flagged / captured) if events_public and captured else suppressed
  return {
    "policy": label,
    "utilization": utilization,
    "capacity_fraction": capacity_fraction,
    "n_b_evaluations": n_b_evaluations,
    "n_flagged": n_flagged,
    "n_flagged_at_capacity": n_flagged,
    "events_captured": n_cascade if events_public else suppressed,
    "events_captured_full_b": len(full_b_events) if events_public else suppressed,
    "events_also_captured_by_full_b": len(shared_event_ids),
    "recovery_fraction": float(len(shared_event_ids) / len(full_b_events)) if full_b_events else None,
    "relative_capture_ratio": relative,
    "full_b_events_missed": len(full_b_events - cascade_events) if events_public else suppressed,
    "events_total": events if events_public else suppressed,
    "sensitivity_at_capacity": sensitivity,
    "ppv_at_capacity": ppv,
    "number_needed_to_evaluate_at_capacity": nne,
  }


def random_routing_blend(
  scores_a: npt.NDArray[np.floating],
  scores_b: npt.NDArray[np.floating],
  utilization: float,
  seed: int = 1912,
) -> tuple[npt.NDArray[np.floating], int]:
  """Blend rank-matched scores after routing exactly the requested number to B."""
  rng = np.random.default_rng(seed)
  n = len(scores_a)
  n_send = max(1, int(round(n * utilization)))
  if len(scores_b) != n:
    raise ValueError("A and B scores must have the same number of rows.")
  send = np.zeros(n, dtype=bool)
  send[rng.choice(n, size=n_send, replace=False)] = True
  blended = np.where(send, scores_b, scores_a)
  return blended, n_send


def random_routing_seed_stability(
  y: npt.NDArray[np.integer],
  rank_scores_a: npt.NDArray[np.floating],
  rank_scores_b: npt.NDArray[np.floating],
  utilizations: tuple[float, ...] = UTILIZATION_FRACTIONS,
  capacity_fractions: tuple[float, ...] = (PRIMARY_CAPACITY, *SENSITIVITY_CAPACITIES),
  n_seeds: int = 200,
  base_seed: int = 1912,
) -> dict[str, Any]:
  """Summarize the matched random-routing control across many routing seeds.

  A single fixed seed is reproducible but arbitrary; this re-runs the exactly
  matched random-routing control over ``n_seeds`` deterministic seeds and
  reports the spread of its event-capture metrics next to the risk-guided
  (rank-normalized) policy at the same utilizations and capacities. Scores
  must already be rank-transformed with the dev/val-fitted calibrators.
  """
  if n_seeds < 2:
    raise ValueError("Seed stability needs at least two routing seeds.")
  yt = np.asarray(y, dtype=np.int64)
  ra = np.asarray(rank_scores_a, dtype=np.float64)
  rb = np.asarray(rank_scores_b, dtype=np.float64)
  if len(yt) != len(ra) or len(yt) != len(rb):
    raise ValueError("Labels and rank-transformed score arrays must be aligned.")
  n = len(yt)
  seeds = [base_seed + i for i in range(n_seeds)]
  utilizations_out: list[dict[str, Any]] = []
  for u in utilizations:
    n_send = max(1, int(round(n * u)))
    send_mask = np.zeros(n, dtype=bool)
    send_mask[np.argsort(-ra, kind="stable")[:n_send]] = True
    for cap in capacity_fractions:
      random_rows: list[dict[str, Any]] = []
      guided_row = evaluate_cascade_policy(
        yt, ra, rb, np.where(send_mask, rb, ra), n_send, cap, "rank_normalized", u,
      )
      for seed in seeds:
        blended, n_rand = random_routing_blend(ra, rb, u, seed=seed)
        random_rows.append(
          evaluate_cascade_policy(yt, ra, rb, blended, n_rand, cap, "random_routing", u),
        )
      captures = [r["events_captured"] for r in random_rows]
      recoveries = [r["recovery_fraction"] for r in random_rows if r["recovery_fraction"] is not None]
      relatives = [r["relative_capture_ratio"] for r in random_rows]
      utilizations_out.append(
        {
          "utilization": u,
          "capacity_fraction": cap,
          "n_seeds": n_seeds,
          "base_seed": base_seed,
          "random_events_captured": {
            "min": min(captures),
            "median": float(np.median(captures)),
            "mean": float(np.mean(captures)),
            "max": max(captures),
          },
          "random_identity_recovery": {
            "min": float(np.min(recoveries)),
            "median": float(np.median(recoveries)),
            "mean": float(np.mean(recoveries)),
            "max": float(np.max(recoveries)),
          },
          "random_relative_capture_mean": float(np.mean(relatives)),
          "rank_normalized_reference": {
            "events_captured": guided_row["events_captured"],
            "recovery_fraction": guided_row["recovery_fraction"],
            "relative_capture_ratio": guided_row["relative_capture_ratio"],
          },
          "n_seeds_exceeding_rank_normalized_events": int(
            sum(c > guided_row["events_captured"] for c in captures),
          )
          if isinstance(guided_row["events_captured"], int)
          else None,
        },
      )
  return {
    "method": "exactly matched random-routing control re-run over deterministic seeds",
    "score_blending": "dev/val-fitted ECDF rank transform, identical to rank_normalized",
    "n_seeds": n_seeds,
    "base_seed": base_seed,
    "utilizations": utilizations_out,
  }


def cascade_policy_comparison(
  y: npt.NDArray[np.integer],
  scores_a: npt.NDArray[np.floating],
  scores_b: npt.NDArray[np.floating],
  calibration_scores_a: npt.NDArray[np.floating] | None = None,
  calibration_scores_b: npt.NDArray[np.floating] | None = None,
  utilization_fractions: tuple[float, ...] = UTILIZATION_FRACTIONS,
  capacity_fractions: tuple[float, ...] = (PRIMARY_CAPACITY, *SENSITIVITY_CAPACITIES),
  routing_seed: int = 1912,
  rank_calibrator_a: ECDFRankCalibrator | None = None,
  rank_calibrator_b: ECDFRankCalibrator | None = None,
) -> list[dict[str, Any]]:
  """Compare all prespecified cascade policies on one evaluation set.

  Rank calibration (when ``calibration_scores_a/b`` are provided, typically
  the development/validation rows) is fitted only on those rows and applied
  unchanged to the evaluated rows. Risk-guided and random routing use the
  same rank-normalized blend and exactly matched B evaluation counts. The
  raw probability-scale blend is retained as a separate reference policy.
  """
  yt = np.asarray(y, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)

  if (rank_calibrator_a is None) != (rank_calibrator_b is None):
    raise ValueError("Both fitted rank calibrators must be provided together.")
  if rank_calibrator_a is not None and rank_calibrator_b is not None:
    if calibration_scores_a is not None or calibration_scores_b is not None:
      raise ValueError("Pass fitted calibrators or calibration scores, not both.")
    ra, rb = rank_calibrator_a.transform(sa), rank_calibrator_b.transform(sb)
  elif calibration_scores_a is not None and calibration_scores_b is not None:
    calib_a = ECDFRankCalibrator().fit(calibration_scores_a)
    calib_b = ECDFRankCalibrator().fit(calibration_scores_b)
    ra, rb = calib_a.transform(sa), calib_b.transform(sb)
  elif calibration_scores_a is None and calibration_scores_b is None:
    ra, rb = sa, sb
  else:
    raise ValueError("Both A and B calibration score arrays must be provided together.")

  n = len(yt)
  rows: list[dict[str, Any]] = []
  for cap in capacity_fractions:
    # Pure references (routing-independent).
    rows.append(evaluate_cascade_policy(yt, sa, sb, sa, 0, cap, "pure_A", 0.0))
    rows.append(evaluate_cascade_policy(yt, sa, sb, sb, n, cap, "pure_B", 1.0))
    for u in utilization_fractions:
      n_send = max(1, int(round(n * u)))
      # (i) Reference mixed-score policy (probability-scale blend).
      send_mask = np.zeros(n, dtype=bool)
      send_mask[np.argsort(-sa, kind="stable")[:n_send]] = True
      blended_mixed = np.where(send_mask, sb, sa)
      rows.append(
        evaluate_cascade_policy(yt, sa, sb, blended_mixed, n_send, cap, "mixed_reference", u),
      )
      # (ii) Rank-normalized blend (dev/val-fitted ECDFs).
      blended_rank = np.where(send_mask, rb, ra)
      rows.append(
        evaluate_cascade_policy(yt, sa, sb, blended_rank, n_send, cap, "rank_normalized", u),
      )
      # (iv) Random-routing efficiency control, matched to rank-normalized
      # scores and exactly the risk-guided policy's B evaluation count.
      blended_rand, n_rand = random_routing_blend(ra, rb, u, seed=routing_seed)
      rows.append(
        evaluate_cascade_policy(yt, sa, sb, blended_rand, n_rand, cap, "random_routing", u),
      )
  return rows


def policy_agreement(
  y: npt.NDArray[np.integer],
  scores_a: npt.NDArray[np.floating],
  scores_b: npt.NDArray[np.floating],
  calibration_scores_a: npt.NDArray[np.floating],
  calibration_scores_b: npt.NDArray[np.floating],
  utilization_fractions: tuple[float, ...] = UTILIZATION_FRACTIONS,
  capacity_fraction: float = PRIMARY_CAPACITY,
  routing_seed: int = 1912,
) -> list[dict[str, Any]]:
  """Directly compare the *selected patients* of the mixed vs rank-normalized policies.

  Aggregate summaries (total captured events, overlap with full-B identities)
  can be identical while the two policies flag different patients. This audits
  the audit-20261005/audit02 request: per utilization, at one review capacity,
  report the selected-patient overlap, captured-death identity overlap between
  the two policies, and the deaths each policy flags that the other does not.
  """
  yt = np.asarray(y, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  calib_a = ECDFRankCalibrator().fit(calibration_scores_a)
  calib_b = ECDFRankCalibrator().fit(calibration_scores_b)
  ra, rb = calib_a.transform(sa), calib_b.transform(sb)

  n = len(yt)
  n_flagged = max(1, int(round(n * capacity_fraction)))
  event_indices = set(np.where(yt == 1)[0])
  rows: list[dict[str, Any]] = []
  for u in utilization_fractions:
    n_send = max(1, int(round(n * u)))
    send_mask = np.zeros(n, dtype=bool)
    send_mask[np.argsort(-sa, kind="stable")[:n_send]] = True
    blended_mixed = np.where(send_mask, sb, sa)
    blended_rank = np.where(send_mask, rb, ra)
    mixed_flagged = set(np.argsort(-blended_mixed, kind="stable")[:n_flagged])
    rank_flagged = set(np.argsort(-blended_rank, kind="stable")[:n_flagged])
    mixed_events = mixed_flagged & event_indices
    rank_events = rank_flagged & event_indices
    rows.append(
      {
        "utilization": u,
        "capacity_fraction": capacity_fraction,
        "n_flagged": n_flagged,
        "selected_patient_overlap": len(mixed_flagged & rank_flagged),
        "selected_patient_overlap_fraction": float(len(mixed_flagged & rank_flagged) / n_flagged),
        "captured_events_mixed": len(mixed_events),
        "captured_events_rank": len(rank_events),
        "captured_death_overlap": len(mixed_events & rank_events),
        "captured_death_overlap_fraction": (
          float(len(mixed_events & rank_events) / len(mixed_events)) if mixed_events else float("nan")
        ),
        "mixed_only_deaths": len(mixed_events - rank_events),
        "rank_only_deaths": len(rank_events - mixed_events),
      },
    )
  return rows


def bootstrap_primary_operating_point(
  y: npt.NDArray[np.integer],
  scores_a: npt.NDArray[np.floating],
  scores_b: npt.NDArray[np.floating],
  rank_calibrator_a: ECDFRankCalibrator,
  rank_calibrator_b: ECDFRankCalibrator,
  n_bootstraps: int = 1000,
  seed: int = 1912,
  utilization: float = 0.10,
  capacity_fraction: float = PRIMARY_CAPACITY,
) -> dict[str, Any]:
  """Bootstrap policies with rank calibrators fixed from development/validation."""
  yt = np.asarray(y, dtype=np.int64)
  sa = np.asarray(scores_a, dtype=np.float64)
  sb = np.asarray(scores_b, dtype=np.float64)
  rng = np.random.default_rng(seed)
  n = len(yt)
  diffs: list[float] = []
  skipped = 0
  for _ in range(n_bootstraps):
    idx = rng.integers(0, n, size=n)
    y_b = yt[idx]
    if y_b.min() == y_b.max():
      skipped += 1
      continue
    n_send = max(1, int(round(n * utilization)))
    send_mask = np.zeros(n, dtype=bool)
    send_mask[np.argsort(-sa[idx], kind="stable")[:n_send]] = True
    blended_mixed = np.where(send_mask, sb[idx], sa[idx])
    blended_rank = np.where(
      send_mask,
      rank_calibrator_b.transform(sb[idx]),
      rank_calibrator_a.transform(sa[idx]),
    )
    _, events_mixed = cascade_events_captured(y_b, blended_mixed, capacity_fraction)
    _, events_rank = cascade_events_captured(y_b, blended_rank, capacity_fraction)
    diffs.append(float(len(events_rank) - len(events_mixed)))
  arr = np.asarray(diffs, dtype=np.float64)
  return {
    "policy_contrast": "rank_normalized_minus_mixed_reference_events",
    "utilization": utilization,
    "capacity_fraction": capacity_fraction,
    "mean": float(arr.mean()) if len(diffs) else None,
    "lo95": float(np.percentile(arr, 2.5)) if len(diffs) else None,
    "hi95": float(np.percentile(arr, 97.5)) if len(diffs) else None,
    "n_bootstraps": n_bootstraps,
    "n_skipped_degenerate": skipped,
    "seed": seed,
  }
