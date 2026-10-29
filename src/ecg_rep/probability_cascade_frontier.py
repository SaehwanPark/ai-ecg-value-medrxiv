"""Patient-level capacity and identity frontiers for probability routing."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from ecg_rep.probability_cascade import ThresholdFit, fit_reference_cutoff
from ecg_rep.probability_cascade_calibration import (
  apply_routed_branch_risk_maps,
  fit_branch_risk_maps,
)


def top_capacity_mask(
  scores: npt.ArrayLike,
  subject_ids: npt.ArrayLike,
  study_ids: npt.ArrayLike,
  capacity: float,
) -> npt.NDArray[np.bool_]:
  """Select the fixed-capacity highest scores with ECG keys as tie-breakers."""
  values = np.asarray(scores, dtype=np.float64)
  subjects = np.asarray(subject_ids, dtype=np.int64)
  studies = np.asarray(study_ids, dtype=np.int64)
  if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
    raise ValueError("Scores must be a non-empty, one-dimensional finite array.")
  if len(subjects) != len(values) or len(studies) != len(values):
    raise ValueError("Score and ECG-key arrays must align.")
  if not np.isfinite(capacity) or not 0.0 < capacity <= 1.0:
    raise ValueError("Review capacity must lie in (0, 1].")
  n_selected = max(1, int(round(len(values) * capacity)))
  order = np.lexsort((studies, subjects, -values))
  selected = np.zeros(len(values), dtype=bool)
  selected[order[:n_selected]] = True
  return selected


def deterministic_random_route(
  subject_ids: npt.ArrayLike,
  study_ids: npt.ArrayLike,
  utilization: float,
  *,
  seed: int,
) -> npt.NDArray[np.bool_]:
  """Create patient-stable random routes independent of outcomes and scores."""
  subjects = np.asarray(subject_ids, dtype=np.int64)
  studies = np.asarray(study_ids, dtype=np.int64)
  if len(subjects) != len(studies):
    raise ValueError("Patient and ECG key arrays must align.")
  if not np.isfinite(utilization) or not 0.0 <= utilization <= 1.0:
    raise ValueError("Random-routing utilization must lie in [0, 1].")
  threshold = int(utilization * (1 << 64))
  selected = np.zeros(len(subjects), dtype=bool)
  for index, (subject, study) in enumerate(zip(subjects, studies, strict=True)):
    token = f"{seed}\0{subject}\0{study}".encode("utf-8")
    draw = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    selected[index] = draw < threshold
  return selected


def route_high_a_plus_predicted_discordance(
  p_a: npt.ArrayLike,
  predicted_b_minus_a: npt.ArrayLike,
  tau_high_a: float,
  tau_discordance: float,
) -> npt.NDArray[np.bool_]:
  """Route high-A patients plus low-A patients with predicted B-A uplift.

  The inputs are available before Stage B: A risk and an A-feature-only
  surrogate prediction. Deliberately, this API accepts no Stage-B scores.
  """
  a = np.asarray(p_a, dtype=np.float64)
  gap = np.asarray(predicted_b_minus_a, dtype=np.float64)
  if a.ndim != 1 or gap.ndim != 1 or len(a) != len(gap):
    raise ValueError("A risk and predicted disagreement arrays must align.")
  if not np.isfinite(a).all() or ((a < 0.0) | (a > 1.0)).any():
    raise ValueError("A risk probabilities must be finite and lie in [0, 1].")
  if not np.isfinite(gap).all():
    raise ValueError("Predicted disagreement values must be finite.")
  if not np.isfinite(tau_high_a) or not 0.0 <= tau_high_a <= 1.0:
    raise ValueError("The high-A risk cutoff must lie in [0, 1].")
  if not np.isfinite(tau_discordance):
    raise ValueError("The predicted-disagreement cutoff must be finite.")
  high_a = a >= tau_high_a
  return high_a | ((~high_a) & (gap >= tau_discordance))


def branch_selection_diagnostics(
  route_to_b: npt.ArrayLike,
  final_scores: npt.ArrayLike,
  selected: npt.ArrayLike,
) -> dict[str, float | int | None]:
  """Summarize selected counts and within-branch score boundaries."""
  route = np.asarray(route_to_b, dtype=bool)
  scores = np.asarray(final_scores, dtype=np.float64)
  review = np.asarray(selected, dtype=bool)
  if route.ndim != 1 or scores.ndim != 1 or review.ndim != 1:
    raise ValueError("Route, final-score, and selection arrays must be one-dimensional.")
  if len(route) != len(scores) or len(route) != len(review):
    raise ValueError("Route, final-score, and selection arrays must align.")
  if not np.isfinite(scores).all():
    raise ValueError("Final scores must be finite.")

  result: dict[str, float | int | None] = {}
  for branch_name, branch_mask in (("routed_b", route), ("retained_a", ~route)):
    selected_scores = scores[branch_mask & review]
    unselected_scores = scores[branch_mask & ~review]
    result[f"n_reviewed_{branch_name}"] = int(len(selected_scores))
    result[f"min_selected_score_{branch_name}"] = (
      float(selected_scores.min()) if len(selected_scores) else None
    )
    result[f"max_unselected_score_{branch_name}"] = (
      float(unselected_scores.max()) if len(unselected_scores) else None
    )
  return result


def _event_keys(mask: npt.NDArray[np.bool_], y: npt.NDArray[np.int64], subjects: npt.NDArray[np.int64], studies: npt.NDArray[np.int64]) -> set[tuple[int, int]]:
  return {
    (int(subjects[index]), int(studies[index]))
    for index in np.flatnonzero(mask & (y == 1))
  }


def _metric_row(
  *,
  routing_method: str,
  final_score_method: str,
  route_mask: npt.NDArray[np.bool_],
  final_scores: npt.NDArray[np.float64],
  p_a: npt.NDArray[np.float64],
  p_b: npt.NDArray[np.float64],
  y: npt.NDArray[np.int64],
  subjects: npt.NDArray[np.int64],
  studies: npt.NDArray[np.int64],
  capacity: float,
  target_utilization: float | None,
  tau_r: float | None,
  reference_fit: ThresholdFit | None,
  full_b_event_keys: set[tuple[int, int]],
  full_b_review_mask: npt.NDArray[np.bool_],
) -> tuple[dict[str, Any], dict[str, Any]]:
  selected = top_capacity_mask(final_scores, subjects, studies, capacity)
  full_b_keys = _event_keys(full_b_review_mask, y, subjects, studies)
  if full_b_keys != full_b_event_keys:
    raise ValueError("Full-B reference event identities do not match the paired assessment records.")
  selected_event_keys = _event_keys(selected, y, subjects, studies)
  recovered = full_b_event_keys & selected_event_keys
  missed = full_b_event_keys - selected_event_keys
  row_by_key = {
    (int(subject), int(study)): index
    for index, (subject, study) in enumerate(zip(subjects, studies, strict=True))
  }
  gate_missed = {
    key for key in missed
    if not route_mask[row_by_key[key]]
  }
  routed_downgraded = {
    key for key in missed
    if route_mask[row_by_key[key]]
    and p_b[row_by_key[key]] < p_a[row_by_key[key]]
  }
  routed_competition_missed = missed - gate_missed - routed_downgraded
  tp = int(np.sum(selected & (y == 1)))
  fp = int(np.sum(selected & (y == 0)))
  n_events = int(y.sum())
  row = {
    "routing_method": routing_method,
    "final_score_method": final_score_method,
    "target_b_utilization": target_utilization,
    "tau_r": tau_r,
    "reference_split": reference_fit.reference_split if reference_fit else None,
    "reference_expected_utilization": target_utilization,
    "reference_realized_utilization": reference_fit.realized_reference_utilization if reference_fit else None,
    "reference_n_routed": reference_fit.n_routed_reference if reference_fit else None,
    "reference_n": reference_fit.n_reference if reference_fit else None,
    "reference_ties_at_tau": reference_fit.n_tied_at_threshold if reference_fit else None,
    "n_patients": len(y),
    "n_events_total": n_events,
    "n_routed_to_b": int(route_mask.sum()),
    "realized_b_utilization": float(route_mask.mean()),
    "b_runtime_measured": False,
    "b_runtime_note": "B scores were cached for offline counterfactual evaluation; this fraction is a policy route rate, not measured compute savings.",
    "review_capacity": capacity,
    "n_flagged": int(selected.sum()),
    "true_positives": tp,
    "false_positives": fp,
    "sensitivity": tp / n_events if n_events else None,
    "ppv": tp / int(selected.sum()) if selected.any() else None,
    "number_needed_to_evaluate": int(selected.sum()) / tp if tp else None,
    "events_missed_at_capacity": n_events - tp,
    "full_b_top_capacity_events": len(full_b_event_keys),
    "full_b_event_identities_recovered": len(recovered),
    "full_b_identity_recovery": len(recovered) / len(full_b_event_keys) if full_b_event_keys else None,
    "full_b_event_identities_missed": len(missed),
    "cascade_only_event_identities": len(selected_event_keys - full_b_event_keys),
  }
  gate = {
    "routing_method": routing_method,
    "final_score_method": final_score_method,
    "target_b_utilization": target_utilization,
    "tau_r": tau_r,
    "n_total_events_not_routed": int(np.sum((~route_mask) & (y == 1))),
    "n_full_b_top_capacity_events": len(full_b_event_keys),
    "n_full_b_top_capacity_events_not_routed": len(gate_missed),
    "n_full_b_top_capacity_events_routed_downgraded_by_b": len(routed_downgraded),
    "n_full_b_top_capacity_events_routed_missed_at_final_topk": len(routed_competition_missed),
    "n_full_b_top_capacity_events_recovered": len(recovered),
    "n_full_b_top_capacity_events_missed": len(missed),
    "miss_decomposition_reconciles": len(gate_missed) + len(routed_downgraded) + len(routed_competition_missed) == len(missed),
  }
  return row, gate


def evaluate_frontier(
  reference: pl.DataFrame,
  assessment: pl.DataFrame,
  *,
  targets: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40),
  review_capacities: tuple[float, ...] = (0.05,),
  seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[float, ThresholdFit]]:
  """Compare numeric, legacy, random, A-only, and B-all policies on paired rows."""
  required = {"subject_id", "study_id", "split", "p_a_ecg", "p_b_ecg", "mortality_30d"}
  for name, frame in (("reference", reference), ("assessment", assessment)):
    missing = required - set(frame.columns)
    if missing:
      raise ValueError(f"{name} frame missing columns: {sorted(missing)}")
    if frame.is_empty():
      raise ValueError(f"{name} frame must not be empty.")
  if set(reference.get_column("split").unique().to_list()) != {"val"}:
    raise ValueError("Reference scores must contain validation patients only.")
  if set(assessment.get_column("split").unique().to_list()) != {"test"}:
    raise ValueError("Assessment scores must contain the post-hoc test split only.")
  shared_patients = set(reference.get_column("subject_id").to_list()) & set(assessment.get_column("subject_id").to_list())
  if shared_patients:
    raise ValueError("Reference and assessment splits must be patient-disjoint.")

  reference_scores = np.asarray(reference.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  if not targets or not review_capacities:
    raise ValueError("At least one route target and review capacity are required.")
  fits = {
    target: fit_reference_cutoff(reference_scores.tolist(), target, "val")
    for target in targets
  }
  test = assessment.sort(["subject_id", "study_id"])
  y = np.asarray(test.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pa = np.asarray(test.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb = np.asarray(test.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  subjects = np.asarray(test.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(test.get_column("study_id").to_numpy(), dtype=np.int64)
  if len(np.unique(subjects)) != len(subjects) or len(np.unique(studies)) != len(studies):
    raise ValueError("Assessment rows must contain one unique index ECG per patient.")
  if not np.isin(y, (0, 1)).all():
    raise ValueError("Assessment mortality labels must be binary.")
  all_true_events = int(y.sum())

  rows: list[dict[str, Any]] = []
  gate_rows: list[dict[str, Any]] = []
  overlap_rows: list[dict[str, Any]] = []
  for capacity in review_capacities:
    full_b_review = top_capacity_mask(pb, subjects, studies, capacity)
    full_b_event_keys = _event_keys(full_b_review, y, subjects, studies)
    overlap_rows.append({
      "review_capacity": capacity,
      "full_b_top_capacity_events": len(full_b_event_keys),
      "full_b_top_capacity_patients": int(full_b_review.sum()),
      "total_mortality_events": all_true_events,
      "assessment_patients": len(y),
    })
    policies: list[tuple[str, str, npt.NDArray[np.bool_], npt.NDArray[np.float64], float | None, float | None, ThresholdFit | None]] = [
      ("A_only", "p_a_ecg", np.zeros(len(y), dtype=bool), pa, None, None, None),
      ("B_all", "p_b_ecg", np.ones(len(y), dtype=bool), pb, 1.0, None, None),
    ]
    for target, fit in fits.items():
      route = pa >= fit.tau_r
      policies.append((
        "fixed_numeric_A_risk", "raw_selective_mixture", route,
        np.where(route, pb, pa), target, fit.tau_r, fit,
      ))
      legacy_route = np.zeros(len(y), dtype=bool)
      selected = np.lexsort((studies, subjects, -pa))[: max(1, int(round(len(y) * target)))]
      legacy_route[selected] = True
      policies.append((
        "legacy_top_u", "raw_selective_mixture", legacy_route,
        np.where(legacy_route, pb, pa), target, None, None,
      ))
      random_route = deterministic_random_route(subjects, studies, target, seed=seed)
      policies.append((
        "matched_random", "raw_selective_mixture", random_route,
        np.where(random_route, pb, pa), target, None, None,
      ))

    for routing_method, score_method, route, final_score, target, tau, fit in policies:
      row, gate = _metric_row(
        routing_method=routing_method,
        final_score_method=score_method,
        route_mask=route,
        final_scores=final_score,
        p_a=pa,
        p_b=pb,
        y=y,
        subjects=subjects,
        studies=studies,
        capacity=capacity,
        target_utilization=target,
        tau_r=tau,
        reference_fit=fit,
        full_b_event_keys=full_b_event_keys,
        full_b_review_mask=full_b_review,
      )
      row["n_events_total"] = all_true_events
      rows.append(row)
      if routing_method not in ("A_only", "B_all"):
        gate_rows.append(gate)

  return pl.DataFrame(rows), pl.DataFrame(overlap_rows), pl.DataFrame(gate_rows), fits


def bootstrap_primary_frontier(
  assessment: pl.DataFrame,
  tau_r: float,
  *,
  capacity: float = 0.05,
  n_bootstraps: int = 200,
  seed: int = 42,
) -> pl.DataFrame:
  """Paired patient bootstrap for 20%-route identity and event-capture contrasts."""
  if n_bootstraps < 1:
    raise ValueError("n_bootstraps must be positive.")
  frame = assessment.sort(["subject_id", "study_id"])
  y = np.asarray(frame.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pa = np.asarray(frame.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb = np.asarray(frame.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  subjects = np.asarray(frame.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(frame.get_column("study_id").to_numpy(), dtype=np.int64)
  route = pa >= tau_r
  mixture = np.where(route, pb, pa)
  full_b_selected = top_capacity_mask(pb, subjects, studies, capacity)
  cascade_selected = top_capacity_mask(mixture, subjects, studies, capacity)
  full_b_events = full_b_selected & (y == 1)
  cascade_events = cascade_selected & (y == 1)
  point_delta_capture = int(cascade_events.sum() - full_b_events.sum())
  point_identity = (
    float(np.sum(full_b_events & cascade_events) / full_b_events.sum())
    if full_b_events.any()
    else None
  )
  rng = np.random.default_rng(seed)
  delta_capture: list[float] = []
  identity: list[float] = []
  for _ in range(n_bootstraps):
    sample = rng.integers(0, len(y), size=len(y))
    ys, pas, pbs, qs = y[sample], pa[sample], pb[sample], mixture[sample]
    ss, es = subjects[sample], studies[sample]
    # A patient sampled more than once becomes multiple bootstrap units. Use the
    # draw position only after score and ECG keys to break ties between copies.
    draw_positions = np.arange(len(sample), dtype=np.int64)
    n_selected = max(1, int(round(len(sample) * capacity)))
    order_b = np.lexsort((draw_positions, es, ss, -pbs))
    order_q = np.lexsort((draw_positions, es, ss, -qs))
    selected_b = np.zeros(len(sample), dtype=bool)
    selected_q = np.zeros(len(sample), dtype=bool)
    selected_b[order_b[:n_selected]] = True
    selected_q[order_q[:n_selected]] = True
    b_events = selected_b & (ys == 1)
    q_events = selected_q & (ys == 1)
    delta_capture.append(float(q_events.sum() - b_events.sum()))
    identity.append(float(np.sum(b_events & q_events) / b_events.sum()) if b_events.any() else float("nan"))
  finite_identity = np.asarray(identity, dtype=np.float64)
  finite_identity = finite_identity[np.isfinite(finite_identity)]
  return pl.DataFrame([
    {
      "metric": "cascade_minus_B_all_event_count_at_capacity",
      "point_estimate": point_delta_capture,
      "ci_lower": float(np.quantile(delta_capture, 0.025)),
      "ci_upper": float(np.quantile(delta_capture, 0.975)),
      "n_bootstraps": n_bootstraps,
      "seed": seed,
    },
    {
      "metric": "full_B_event_identity_recovery",
      "point_estimate": point_identity,
      "ci_lower": float(np.quantile(finite_identity, 0.025)) if len(finite_identity) else None,
      "ci_upper": float(np.quantile(finite_identity, 0.975)) if len(finite_identity) else None,
      "n_bootstraps": int(len(finite_identity)),
      "seed": seed,
    },
  ])


def bootstrap_policy_pair_contrast(
  assessment: pl.DataFrame,
  baseline_scores: npt.ArrayLike,
  candidate_scores: npt.ArrayLike,
  *,
  capacity: float = 0.05,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> pl.DataFrame:
  """Bootstrap candidate-minus-baseline capacity and identity contrasts.

  Resamples paired patients with replacement. Repeated draws are separate
  bootstrap units, with draw position used only after score and ECG keys to
  break ties. Full-B identity recovery uses the same sampled B reference list
  for both policies in each replicate.
  """
  if n_bootstraps < 1:
    raise ValueError("n_bootstraps must be positive.")
  if not np.isfinite(capacity) or not 0.0 < capacity <= 1.0:
    raise ValueError("Review capacity must lie in (0, 1].")
  required = {"subject_id", "study_id", "p_b_ecg", "mortality_30d"}
  missing = required - set(assessment.columns)
  if missing:
    raise ValueError(f"Assessment frame missing columns: {sorted(missing)}")
  frame = assessment.sort(["subject_id", "study_id"])
  y = np.asarray(frame.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pb = np.asarray(frame.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  subjects = np.asarray(frame.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(frame.get_column("study_id").to_numpy(), dtype=np.int64)
  baseline = np.asarray(baseline_scores, dtype=np.float64)
  candidate = np.asarray(candidate_scores, dtype=np.float64)
  if y.ndim != 1 or not len(y) or not np.isin(y, (0, 1)).all():
    raise ValueError("Assessment outcomes must be a non-empty binary array.")
  if any(values.shape != y.shape for values in (pb, baseline, candidate)):
    raise ValueError("B, baseline, and candidate score arrays must align with outcomes.")
  if any(not np.isfinite(values).all() for values in (pb, baseline, candidate)):
    raise ValueError("B, baseline, and candidate scores must be finite.")
  if len(np.unique(subjects)) != len(y) or len(np.unique(studies)) != len(y):
    raise ValueError("Assessment rows must have unique patient and ECG keys.")

  n_selected = max(1, int(round(len(y) * capacity)))

  def choose(scores: np.ndarray, sample: np.ndarray, draw_positions: np.ndarray) -> np.ndarray:
    order = np.lexsort((draw_positions, studies[sample], subjects[sample], -scores[sample]))
    selected = np.zeros(len(sample), dtype=bool)
    selected[order[:n_selected]] = True
    return selected

  def metrics_for(
    sampled_y: np.ndarray,
    selected_b: np.ndarray,
    selected_baseline: np.ndarray,
    selected_candidate: np.ndarray,
  ) -> tuple[float, float, float]:
    events_b = selected_b & (sampled_y == 1)
    events_baseline = selected_baseline & (sampled_y == 1)
    events_candidate = selected_candidate & (sampled_y == 1)
    denominator = int(events_b.sum())
    recovery_baseline = float(np.sum(events_b & events_baseline) / denominator) if denominator else float("nan")
    recovery_candidate = float(np.sum(events_b & events_candidate) / denominator) if denominator else float("nan")
    return (
      float(events_candidate.sum() - events_baseline.sum()),
      recovery_candidate - recovery_baseline,
      float(np.sum(selected_baseline & selected_candidate) / n_selected),
    )

  point_positions = np.arange(len(y), dtype=np.int64)
  point_values = metrics_for(
    y,
    choose(pb, point_positions, point_positions),
    choose(baseline, point_positions, point_positions),
    choose(candidate, point_positions, point_positions),
  )
  replicate_values: list[list[float]] = [[], [], []]
  rng = np.random.default_rng(seed)
  for _ in range(n_bootstraps):
    sample = rng.integers(0, len(y), size=len(y))
    positions = np.arange(len(sample), dtype=np.int64)
    values = metrics_for(
      y[sample],
      choose(pb, sample, positions),
      choose(baseline, sample, positions),
      choose(candidate, sample, positions),
    )
    for metric_values, value in zip(replicate_values, values, strict=True):
      metric_values.append(value)

  metric_names = (
    "events_captured_delta_candidate_minus_baseline",
    "full_b_identity_recovery_delta_candidate_minus_baseline",
    "selected_list_overlap_fraction",
  )
  rows: list[dict[str, Any]] = []
  for name, point, values in zip(metric_names, point_values, replicate_values, strict=True):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    rows.append({
      "metric": name,
      "point_estimate": point,
      "ci_lower": float(np.quantile(finite, 0.025)) if len(finite) else None,
      "ci_upper": float(np.quantile(finite, 0.975)) if len(finite) else None,
      "n_bootstraps": int(len(finite)),
      "seed": seed,
    })
  return pl.DataFrame(rows)


def evaluate_branch_calibrated_frontier(
  reference: pl.DataFrame,
  assessment: pl.DataFrame,
  *,
  targets: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40),
  capacity: float = 0.05,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[float, dict[str, Any]]]:
  """Evaluate validation-fitted branch maps at fixed top-capacity review.

  The returned overlap table compares selected patients under raw and
  branch-calibrated final risks. Returned score arrays are private assessment
  data intended for paired uncertainty calculations, not publication.
  """
  required = {"subject_id", "study_id", "split", "p_a_ecg", "p_b_ecg", "mortality_30d"}
  for name, frame in (("reference", reference), ("assessment", assessment)):
    missing = required - set(frame.columns)
    if missing:
      raise ValueError(f"{name} frame missing columns: {sorted(missing)}")
    if frame.is_empty():
      raise ValueError(f"{name} frame must not be empty.")
  if set(reference.get_column("split").unique().to_list()) != {"val"}:
    raise ValueError("Branch maps must be fit on validation patients only.")
  if set(assessment.get_column("split").unique().to_list()) != {"test"}:
    raise ValueError("Branch frontier assessment must use test patients only.")
  if set(reference.get_column("subject_id").to_list()) & set(assessment.get_column("subject_id").to_list()):
    raise ValueError("Reference and assessment patients must be disjoint.")

  val = reference.sort(["subject_id", "study_id"])
  test = assessment.sort(["subject_id", "study_id"])
  y_val = np.asarray(val.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pa_val = np.asarray(val.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb_val = np.asarray(val.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  y = np.asarray(test.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pa = np.asarray(test.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb = np.asarray(test.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  subjects = np.asarray(test.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(test.get_column("study_id").to_numpy(), dtype=np.int64)
  if not np.isin(y_val, (0, 1)).all() or not np.isin(y, (0, 1)).all():
    raise ValueError("Reference and assessment outcomes must be binary.")
  if len(np.unique(subjects)) != len(subjects) or len(np.unique(studies)) != len(studies):
    raise ValueError("Assessment rows must contain unique patient and ECG keys.")

  fits = {
    target: fit_reference_cutoff(pa_val.tolist(), target, "val")
    for target in targets
  }
  full_b_review = top_capacity_mask(pb, subjects, studies, capacity)
  full_b_event_keys = _event_keys(full_b_review, y, subjects, studies)
  rows: list[dict[str, Any]] = []
  overlap_rows: list[dict[str, Any]] = []
  scores_by_target: dict[float, dict[str, Any]] = {}

  for target, fit in fits.items():
    route_val = pa_val >= fit.tau_r
    low_map, high_map, support = fit_branch_risk_maps(
      y_val, pa_val, pb_val, route_val,
      minimum_slope_events=50,
      minimum_branch_events=10,
    )
    route = pa >= fit.tau_r
    raw = np.where(route, pb, pa)
    branch = apply_routed_branch_risk_maps(pa, pb[route], route, low_map, high_map)
    low_test = low_map.predict(pa)
    routed_high_test = high_map.predict(pb[route])
    mapped_b_for_decomposition = np.full(len(y), np.nan, dtype=np.float64)
    mapped_b_for_decomposition[route] = routed_high_test

    common = {
      "routing_method": "fixed_numeric_A_risk",
      "target_b_utilization": target,
      "tau_r": fit.tau_r,
      "reference_split": "val",
      "reference_realized_utilization": fit.realized_reference_utilization,
      "reference_n_routed": fit.n_routed_reference,
      "reference_n": fit.n_reference,
      "reference_ties_at_tau": fit.n_tied_at_threshold,
      "branch_fit_n_low_a": support["n_low_a"],
      "branch_fit_events_low_a": support["events_low_a"],
      "branch_fit_n_routed_b": support["n_routed_b"],
      "branch_fit_events_routed_b": support["events_routed_b"],
    }
    raw_row, _ = _metric_row(
      routing_method="fixed_numeric_A_risk",
      final_score_method="raw_selective_mixture",
      route_mask=route,
      final_scores=raw,
      p_a=pa,
      p_b=pb,
      y=y,
      subjects=subjects,
      studies=studies,
      capacity=capacity,
      target_utilization=target,
      tau_r=fit.tau_r,
      reference_fit=fit,
      full_b_event_keys=full_b_event_keys,
      full_b_review_mask=full_b_review,
    )
    branch_row, branch_gate = _metric_row(
      routing_method="fixed_numeric_A_risk",
      final_score_method="branch_recalibrated",
      route_mask=route,
      final_scores=branch,
      p_a=low_test,
      p_b=mapped_b_for_decomposition,
      y=y,
      subjects=subjects,
      studies=studies,
      capacity=capacity,
      target_utilization=target,
      tau_r=fit.tau_r,
      reference_fit=fit,
      full_b_event_keys=full_b_event_keys,
      full_b_review_mask=full_b_review,
    )
    rows.extend([{**raw_row}, {**branch_row, **common}])

    raw_selected = top_capacity_mask(raw, subjects, studies, capacity)
    branch_selected = top_capacity_mask(branch, subjects, studies, capacity)
    raw_keys = {
      (int(subjects[i]), int(studies[i])) for i in np.flatnonzero(raw_selected)
    }
    branch_keys = {
      (int(subjects[i]), int(studies[i])) for i in np.flatnonzero(branch_selected)
    }
    intersection = len(raw_keys & branch_keys)
    union = len(raw_keys | branch_keys)
    overlap_rows.append({
      **common,
      "review_capacity": capacity,
      "n_raw_selected": len(raw_keys),
      "n_branch_selected": len(branch_keys),
      "n_selected_patient_overlap": intersection,
      "fraction_of_capacity_overlapping": intersection / len(raw_keys) if raw_keys else None,
      "selected_patient_jaccard": intersection / union if union else None,
      "raw_events_captured": raw_row["true_positives"],
      "branch_events_captured": branch_row["true_positives"],
      "delta_events_branch_minus_raw": branch_row["true_positives"] - raw_row["true_positives"],
      "raw_full_b_identity_recovery": raw_row["full_b_identity_recovery"],
      "branch_full_b_identity_recovery": branch_row["full_b_identity_recovery"],
      "delta_full_b_identity_recovery_branch_minus_raw": (
        branch_row["full_b_identity_recovery"] - raw_row["full_b_identity_recovery"]
        if raw_row["full_b_identity_recovery"] is not None
        and branch_row["full_b_identity_recovery"] is not None else None
      ),
    })
    scores_by_target[target] = {
      "tau_r": fit.tau_r,
      "route": route,
      "raw": raw,
      "branch": branch,
      "full_b": pb,
      "low_map": low_map,
      "high_map": high_map,
      "gate": branch_gate,
    }

  return pl.DataFrame(rows), pl.DataFrame(overlap_rows), scores_by_target


def bootstrap_route_pair_contrast(
  assessment: pl.DataFrame,
  score_pairs: dict[str, tuple[npt.ArrayLike, npt.ArrayLike]],
  *,
  capacity: float = 0.05,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> pl.DataFrame:
  """Bootstrap 40%-minus-20% capture and identity plus set overlap.

  Each named pair maps to (20%-route score, 40%-route score). This returns the
  paired event-count and full-B identity-recovery contrasts and the overlap
  fraction between the two selected patient sets, using draw positions as
  unique bootstrap identities when a patient is sampled more than once.
  """
  if n_bootstraps < 1:
    raise ValueError("n_bootstraps must be positive.")
  if not 0.0 < capacity <= 1.0:
    raise ValueError("Review capacity must lie in (0, 1].")
  frame = assessment.sort(["subject_id", "study_id"])
  y = np.asarray(frame.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pb = np.asarray(frame.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  subjects = np.asarray(frame.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(frame.get_column("study_id").to_numpy(), dtype=np.int64)
  if not np.isin(y, (0, 1)).all():
    raise ValueError("Assessment outcomes must be binary.")
  if not score_pairs:
    raise ValueError("At least one 20%-versus-40% score pair is required.")
  pairs = {
    name: (np.asarray(s20, dtype=np.float64), np.asarray(s40, dtype=np.float64))
    for name, (s20, s40) in score_pairs.items()
  }
  if any(a.shape != y.shape or b.shape != y.shape or not np.isfinite(a).all() or not np.isfinite(b).all() for a, b in pairs.values()):
    raise ValueError("Every paired score vector must be finite and align with assessment rows.")

  n_selected = max(1, int(round(len(y) * capacity)))
  rng = np.random.default_rng(seed)
  samples: dict[str, dict[str, list[float]]] = {
    name: {
      "event_capture_delta_40_minus_20": [],
      "full_b_identity_recovery_delta_40_minus_20": [],
      "selected_patient_overlap_fraction_20_vs_40": [],
    }
    for name in pairs
  }

  def choose(scores: np.ndarray, sample: np.ndarray, positions: np.ndarray) -> np.ndarray:
    order = np.lexsort((positions, studies[sample], subjects[sample], -scores[sample]))
    selected = np.zeros(len(sample), dtype=bool)
    selected[order[:n_selected]] = True
    return selected

  for _ in range(n_bootstraps):
    sample = rng.integers(0, len(y), size=len(y))
    positions = np.arange(len(sample), dtype=np.int64)
    ys = y[sample]
    selected_b = choose(pb, sample, positions)
    events_b = selected_b & (ys == 1)
    denominator = int(events_b.sum())
    for name, (score20, score40) in pairs.items():
      selected20 = choose(score20, sample, positions)
      selected40 = choose(score40, sample, positions)
      events20 = selected20 & (ys == 1)
      events40 = selected40 & (ys == 1)
      recovery20 = float(np.sum(events_b & events20) / denominator) if denominator else float("nan")
      recovery40 = float(np.sum(events_b & events40) / denominator) if denominator else float("nan")
      samples[name]["event_capture_delta_40_minus_20"].append(float(events40.sum() - events20.sum()))
      samples[name]["full_b_identity_recovery_delta_40_minus_20"].append(recovery40 - recovery20)
      samples[name]["selected_patient_overlap_fraction_20_vs_40"].append(
        float(np.sum(selected20 & selected40) / n_selected)
      )

  def point_metric(name: str, metric: str) -> float:
    score20, score40 = pairs[name]
    selected20 = top_capacity_mask(score20, subjects, studies, capacity)
    selected40 = top_capacity_mask(score40, subjects, studies, capacity)
    selected_b = top_capacity_mask(pb, subjects, studies, capacity)
    events20 = selected20 & (y == 1)
    events40 = selected40 & (y == 1)
    events_b = selected_b & (y == 1)
    if metric == "event_capture_delta_40_minus_20":
      return float(events40.sum() - events20.sum())
    if metric == "full_b_identity_recovery_delta_40_minus_20":
      if not events_b.any():
        return float("nan")
      return float(np.sum(events_b & events40) / events_b.sum() - np.sum(events_b & events20) / events_b.sum())
    return float(np.sum(selected20 & selected40) / max(1, int(selected20.sum())))

  rows: list[dict[str, Any]] = []
  for name, metrics in samples.items():
    for metric, values in metrics.items():
      values_array = np.asarray(values, dtype=np.float64)
      values_array = values_array[np.isfinite(values_array)]
      rows.append({
        "score_method": name,
        "metric": metric,
        "point_estimate": point_metric(name, metric),
        "ci_lower": float(np.quantile(values_array, 0.025)) if len(values_array) else None,
        "ci_upper": float(np.quantile(values_array, 0.975)) if len(values_array) else None,
        "n_bootstraps": int(len(values_array)),
        "seed": seed,
      })
  return pl.DataFrame(rows)

