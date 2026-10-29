"""Run the focused round-02 probability-cascade audit analyses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ecg_rep.probability_cascade import fit_reference_cutoff
from ecg_rep.probability_cascade_audit import (
  _as_binary,
  _capacity_metrics,
  _keys_equal,
  _load_frozen_scores,
  _load_paired_feature_index,
  _selected_overlap,
  _sha256_file,
  _write_private_csv,
  _write_private_json,
  _write_public_csv,
)
from ecg_rep.probability_cascade_calibration import (
  apply_routed_branch_risk_maps,
  fit_branch_risk_maps,
)
from ecg_rep.probability_cascade_frontier import (
  bootstrap_policy_pair_contrast,
  bootstrap_primary_frontier,
  branch_selection_diagnostics,
  evaluate_branch_calibrated_frontier,
  route_high_a_plus_predicted_discordance,
  top_capacity_mask,
)
from ecg_rep.probe import ProbeConfig, TrainedProbe, fit_logistic_probe


def _cross_fitted_probability(
  matrix: npt.NDArray[np.floating[Any]],
  feature_rows: npt.NDArray[np.int64],
  labels: npt.NDArray[np.int64],
  dev_indices: npt.NDArray[np.int64],
  *,
  c_value: float,
  seed: int,
  n_folds: int,
  model_name: str,
) -> npt.NDArray[np.float64]:
  """Generate development-only out-of-fold scores with fixed validation C."""
  if n_folds < 2 or len(dev_indices) < n_folds:
    raise ValueError("Cross-fitting requires at least two folds and rows per fold.")
  splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
  output = np.empty(len(dev_indices), dtype=np.float64)
  fixed_config = ProbeConfig(regularization_c_grid=(c_value,))
  for train_pos, held_pos in splitter.split(dev_indices):
    train_rows = feature_rows[dev_indices[train_pos]]
    held_rows = feature_rows[dev_indices[held_pos]]
    x_train = np.asarray(matrix[train_rows], dtype=np.float64)
    x_held = np.asarray(matrix[held_rows], dtype=np.float64)
    probe = fit_logistic_probe(
      x_train,
      labels[dev_indices[train_pos]],
      x_held,
      labels[dev_indices[held_pos]],
      config=fixed_config,
      model_name=f"{model_name}_crossfit",
    )
    output[held_pos] = probe.predict_proba(x_held)
  if not np.isfinite(output).all() or ((output < 0.0) | (output > 1.0)).any():
    raise ValueError(f"Cross-fitted {model_name} probabilities are invalid.")
  return output


def _branch_composition_rows(
  validation: pl.DataFrame,
  assessment: pl.DataFrame,
  capacity: float,
) -> pl.DataFrame:
  _, _, scores_by_target = evaluate_branch_calibrated_frontier(
    validation,
    assessment,
    targets=(0.05, 0.10, 0.20, 0.40),
    capacity=capacity,
  )
  test = assessment.sort(["subject_id", "study_id"])
  subjects = np.asarray(test.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(test.get_column("study_id").to_numpy(), dtype=np.int64)
  rows: list[dict[str, Any]] = []
  for target, policy_scores in scores_by_target.items():
    route = np.asarray(policy_scores["route"], dtype=bool)
    for score_method in ("raw", "branch"):
      final_scores = np.asarray(policy_scores[score_method], dtype=np.float64)
      selected = top_capacity_mask(final_scores, subjects, studies, capacity)
      rows.append({
        "routing_method": "fixed_numeric_A_risk",
        "target_b_utilization": target,
        "tau_r": float(policy_scores["tau_r"]),
        "final_score_method": "raw_selective_mixture" if score_method == "raw" else "branch_recalibrated",
        "n_test_routed_b": int(route.sum()),
        "n_test_retained_a": int((~route).sum()),
        "n_reviewed_total": int(selected.sum()),
        **branch_selection_diagnostics(route, final_scores, selected),
      })
  return pl.DataFrame(rows)


def _load_internal_resplit(
  config: dict[str, Any],
  source_audit_dir: Path,
) -> tuple[pl.DataFrame, npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
  inputs = config["inputs"]
  paired, matrix_a, matrix_b = _load_paired_feature_index(inputs)
  scores = pl.read_parquet(source_audit_dir / "internal_resplit_scores.parquet")
  _keys_equal(paired, scores, "round-01 internal resplit scores")
  score_columns = scores.select([
    "subject_id", "study_id", "split", "p_a_ecg", "p_b_ecg"
  ])
  aligned = paired.join(score_columns, on=["subject_id", "study_id"], how="inner")
  if aligned.height != paired.height:
    raise ValueError("Internal resplit scores do not cover every paired-valid patient.")
  return aligned.sort(["subject_id", "study_id"]), matrix_a, matrix_b


def _hybrid_policy_outputs(
  config: dict[str, Any],
  source_audit_dir: Path,
  private_run_dir: Path,
  public_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
  aligned, matrix_a, matrix_b = _load_internal_resplit(config, source_audit_dir)
  y = _as_binary(aligned.get_column("mortality_30d").to_numpy(), "internal-resplit outcomes")
  split_labels = np.asarray(aligned.get_column("split").to_numpy(), dtype=str)
  row_a = np.asarray(aligned.get_column("a_row").to_numpy(), dtype=np.int64)
  row_b = np.asarray(aligned.get_column("b_row").to_numpy(), dtype=np.int64)
  indices = {
    name: np.flatnonzero(split_labels == name).astype(np.int64)
    for name in ("dev", "val", "calibration_fit", "calibration_check", "test")
  }
  if any(len(index) == 0 for index in indices.values()):
    raise ValueError("Round-01 internal resplit is missing a required partition.")
  dev, val = indices["dev"], indices["val"]
  pa = np.asarray(aligned.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb = np.asarray(aligned.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  probe_dir = source_audit_dir / "refit_models"
  probe_a = TrainedProbe.load(probe_dir / "M_A_ECG.json")
  probe_b = TrainedProbe.load(probe_dir / "M_B_ECG.json")

  crossfit_seed = int(config["crossfit_seed"])
  n_folds = int(config["crossfit_folds"])
  p_a_oof = _cross_fitted_probability(
    matrix_a, row_a, y, dev, c_value=probe_a.best_c, seed=crossfit_seed,
    n_folds=n_folds, model_name="M_A_ECG",
  )
  p_b_oof = _cross_fitted_probability(
    matrix_b, row_b, y, dev, c_value=probe_b.best_c, seed=crossfit_seed,
    n_folds=n_folds, model_name="M_B_ECG",
  )
  delta_oof = p_b_oof - p_a_oof
  delta_val = pb[val] - pa[val]

  x_dev = np.asarray(matrix_a[row_a[dev]], dtype=np.float64)
  x_val = np.asarray(matrix_a[row_a[val]], dtype=np.float64)
  feature_scaler = StandardScaler().fit(x_dev)
  z_dev = np.asarray(feature_scaler.transform(x_dev), dtype=np.float64)
  z_val = np.asarray(feature_scaler.transform(x_val), dtype=np.float64)
  best_alpha: float | None = None
  best_mse = float("inf")
  surrogate: Ridge | None = None
  alpha_rows: list[dict[str, float]] = []
  for raw_alpha in config["surrogate_alphas"]:
    alpha = float(raw_alpha)
    candidate = Ridge(alpha=alpha)
    candidate.fit(z_dev, delta_oof)
    prediction = np.asarray(candidate.predict(z_val), dtype=np.float64)
    mse = float(mean_squared_error(delta_val, prediction))
    alpha_rows.append({"alpha": alpha, "validation_mse": mse})
    if mse < best_mse:
      best_alpha, best_mse, surrogate = alpha, mse, candidate
  if best_alpha is None or surrogate is None:
    raise ValueError("No valid discordance-surrogate regularization value was configured.")
  tau_a = float(fit_reference_cutoff(
    pa[val].tolist(), float(config["high_a_utilization"]), "val"
  ).tau_r)
  high_a_val = pa[val] >= tau_a
  extra_target_in_low_a = (
    float(config["additional_discordance_utilization"])
    / (1.0 - float(config["high_a_utilization"]))
  )
  delta_hat_val = np.asarray(surrogate.predict(z_val), dtype=np.float64)
  low_a_gap_val = delta_hat_val[~high_a_val]
  if len(low_a_gap_val) == 0 or not np.isfinite(low_a_gap_val).all():
    raise ValueError("Validation low-A patients must have finite predicted disagreement scores.")
  tau_delta = float(np.quantile(
    low_a_gap_val, 1.0 - extra_target_in_low_a, method="linear"
  ))
  extra_validation_utilization = float(np.mean(low_a_gap_val >= tau_delta))

  def predict_delta(rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    x = np.asarray(matrix_a[row_a[rows]], dtype=np.float64)
    z = np.asarray(feature_scaler.transform(x), dtype=np.float64)
    prediction = np.asarray(surrogate.predict(z), dtype=np.float64)
    if not np.isfinite(prediction).all():
      raise ValueError("Discordance surrogate returned non-finite values.")
    return prediction

  delta_by_split = {
    name: predict_delta(partition)
    for name, partition in indices.items()
  }
  route_by_split = {
    name: route_high_a_plus_predicted_discordance(
      pa[partition], delta_by_split[name], tau_a, tau_delta
    )
    for name, partition in indices.items()
  }
  baseline_tau = float(fit_reference_cutoff(pa[val].tolist(), 0.20, "val").tau_r)
  baseline_routes = {
    name: pa[partition] >= baseline_tau
    for name, partition in indices.items()
  }

  cal = indices["calibration_fit"]
  low_base, high_base, support_base = fit_branch_risk_maps(
    y[cal], pa[cal], pb[cal], baseline_routes["calibration_fit"],
    minimum_slope_events=50,
    minimum_branch_events=10,
  )
  low_hybrid, high_hybrid, support_hybrid = fit_branch_risk_maps(
    y[cal], pa[cal], pb[cal], route_by_split["calibration_fit"],
    minimum_slope_events=50,
    minimum_branch_events=10,
  )
  test = aligned.filter(pl.col("split") == "test").sort(["subject_id", "study_id"])
  test_idx = indices["test"]
  test_pa, test_pb = pa[test_idx], pb[test_idx]
  test_ids = np.asarray(test.get_column("subject_id").to_numpy(), dtype=np.int64)
  test_ecgs = np.asarray(test.get_column("study_id").to_numpy(), dtype=np.int64)
  route_hybrid = route_by_split["test"]
  route_base = baseline_routes["test"]
  raw_hybrid = np.where(route_hybrid, test_pb, test_pa)
  raw_base = np.where(route_base, test_pb, test_pa)
  mapped_hybrid = apply_routed_branch_risk_maps(
    test_pa, test_pb[route_hybrid], route_hybrid, low_hybrid, high_hybrid
  )
  mapped_base = apply_routed_branch_risk_maps(
    test_pa, test_pb[route_base], route_base, low_base, high_base
  )

  metric_rows: list[dict[str, Any]] = []
  selected_by_policy: dict[str, npt.NDArray[np.bool_]] = {}
  for policy_name, route, raw_score, mapped_score, val_route in (
    ("numeric_A_top20", route_base, raw_base, mapped_base, baseline_routes["val"]),
    ("high_A15_plus_predicted_discordance5", route_hybrid, raw_hybrid, mapped_hybrid, route_by_split["val"]),
  ):
    reference_utilization = float(np.mean(val_route))
    for score_method, score in (("raw_selective_mixture", raw_score), ("branch_recalibrated", mapped_score)):
      row, selected = _capacity_metrics(
        policy=policy_name,
        score_method=score_method,
        frame=test,
        route=route,
        score=score,
        full_b_probability=test_pb,
        capacity=float(config["review_capacity"]),
        reference_utilization=reference_utilization,
      )
      metric_rows.append(row)
      selected_by_policy[f"{policy_name}:{score_method}"] = selected
  metrics_frame = pl.DataFrame(metric_rows)
  _write_private_csv(metrics_frame, private_run_dir / "hybrid_discordance_policy.csv")
  _write_public_csv(metrics_frame, public_dir / "hybrid_discordance_policy.csv")

  baseline_raw_selected = selected_by_policy["numeric_A_top20:raw_selective_mixture"]
  baseline_branch_selected = selected_by_policy["numeric_A_top20:branch_recalibrated"]
  hybrid_raw_selected = selected_by_policy["high_A15_plus_predicted_discordance5:raw_selective_mixture"]
  hybrid_branch_selected = selected_by_policy["high_A15_plus_predicted_discordance5:branch_recalibrated"]
  overlap_frame = pl.DataFrame([
    {
      "candidate_policy": "high_A15_plus_predicted_discordance5",
      "final_score_method": "raw_selective_mixture",
      **_selected_overlap(baseline_raw_selected, hybrid_raw_selected),
    },
    {
      "candidate_policy": "high_A15_plus_predicted_discordance5",
      "final_score_method": "branch_recalibrated",
      **_selected_overlap(baseline_branch_selected, hybrid_branch_selected),
    },
  ])
  _write_private_csv(overlap_frame, private_run_dir / "hybrid_discordance_overlap.csv")
  _write_public_csv(overlap_frame, public_dir / "hybrid_discordance_overlap.csv")

  bootstrap_rows: list[dict[str, Any]] = []
  for score_method, baseline_score, hybrid_score in (
    ("raw_selective_mixture", raw_base, raw_hybrid),
    ("branch_recalibrated", mapped_base, mapped_hybrid),
  ):
    pair = bootstrap_policy_pair_contrast(
      test,
      baseline_score,
      hybrid_score,
      capacity=float(config["review_capacity"]),
      n_bootstraps=int(config["bootstrap_replicates"]),
      seed=int(config["bootstrap_seed"]),
    ).with_columns(pl.lit(score_method).alias("score_method"))
    bootstrap_rows.extend(pair.to_dicts())
  hybrid_bootstrap = pl.DataFrame(bootstrap_rows)
  _write_private_csv(hybrid_bootstrap, private_run_dir / "hybrid_discordance_bootstrap.csv")
  _write_public_csv(hybrid_bootstrap, public_dir / "hybrid_discordance_bootstrap.csv")

  internal_benchmark = bootstrap_primary_frontier(
    test,
    baseline_tau,
    capacity=float(config["review_capacity"]),
    n_bootstraps=int(config["bootstrap_replicates"]),
    seed=int(config["bootstrap_seed"]),
  )
  _write_private_csv(internal_benchmark, private_run_dir / "internal_resplit_benchmark_bootstrap.csv")
  _write_public_csv(internal_benchmark, public_dir / "internal_resplit_benchmark_bootstrap.csv")

  composition_rows = [
    {
      "routing_method": "numeric_A_top20",
      "target_b_utilization": 0.20,
      "tau_r": baseline_tau,
      "final_score_method": "raw_selective_mixture",
      "n_test_routed_b": int(route_base.sum()),
      "n_test_retained_a": int((~route_base).sum()),
      "n_reviewed_total": int(baseline_raw_selected.sum()),
      **branch_selection_diagnostics(route_base, raw_base, baseline_raw_selected),
    },
    {
      "routing_method": "numeric_A_top20",
      "target_b_utilization": 0.20,
      "tau_r": baseline_tau,
      "final_score_method": "branch_recalibrated",
      "n_test_routed_b": int(route_base.sum()),
      "n_test_retained_a": int((~route_base).sum()),
      "n_reviewed_total": int(baseline_branch_selected.sum()),
      **branch_selection_diagnostics(route_base, mapped_base, baseline_branch_selected),
    },
    {
      "routing_method": "high_A15_plus_predicted_discordance5",
      "target_b_utilization": 0.20,
      "tau_r": None,
      "final_score_method": "raw_selective_mixture",
      "n_test_routed_b": int(route_hybrid.sum()),
      "n_test_retained_a": int((~route_hybrid).sum()),
      "n_reviewed_total": int(hybrid_raw_selected.sum()),
      **branch_selection_diagnostics(route_hybrid, raw_hybrid, hybrid_raw_selected),
    },
    {
      "routing_method": "high_A15_plus_predicted_discordance5",
      "target_b_utilization": 0.20,
      "tau_r": None,
      "final_score_method": "branch_recalibrated",
      "n_test_routed_b": int(route_hybrid.sum()),
      "n_test_retained_a": int((~route_hybrid).sum()),
      "n_reviewed_total": int(hybrid_branch_selected.sum()),
      **branch_selection_diagnostics(route_hybrid, mapped_hybrid, hybrid_branch_selected),
    },
  ]
  composition_frame = pl.DataFrame(composition_rows)
  _write_private_csv(composition_frame, private_run_dir / "hybrid_branch_composition.csv")
  _write_public_csv(composition_frame, public_dir / "hybrid_branch_composition.csv")

  cutoffs_payload = {
    "policy": "route if p_A >= tau_A15 or predicted_p_B_minus_p_A >= tau_gap among patients below tau_A15",
    "routing_decision_inputs": ["p_a_ecg", "surrogate_prediction_from_A_features"],
    "stage_b_probability_required_for_individual_route_decision": False,
    "high_a_target_utilization": float(config["high_a_utilization"]),
    "additional_discordance_target_utilization": float(config["additional_discordance_utilization"]),
    "tau_a15": tau_a,
    "tau_predicted_b_minus_a": tau_delta,
    "tau_numeric_a20": baseline_tau,
    "validation_b_utilization_hybrid": float(route_by_split["val"].mean()),
    "validation_extra_discordance_utilization_within_low_a": extra_validation_utilization,
    "assessment_b_utilization_hybrid": float(route_hybrid.mean()),
    "surrogate": {
      "model": "standardized_ridge_regression",
      "training_partition": "development out-of-fold risk differences p_B - p_A",
      "crossfit_folds": n_folds,
      "crossfit_seed": crossfit_seed,
      "alpha_grid_validation_mse": alpha_rows,
      "selected_alpha": best_alpha,
      "validation_mse": best_mse,
    },
    "branch_calibration_fit_partition": "calibration_fit",
    "branch_calibration_support": support_hybrid,
    "baseline_branch_calibration_support": support_base,
  }
  _write_private_json(private_run_dir / "alternative_policy_cutoffs.json", cutoffs_payload)
  (public_dir / "alternative_policy_cutoffs.json").write_text(
    json.dumps(cutoffs_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  private_surrogate = {
    "alpha": best_alpha,
    "intercept": float(surrogate.intercept_),
    "coefficients": [float(value) for value in np.asarray(surrogate.coef_).reshape(-1)],
    "feature_mean": [float(value) for value in np.asarray(feature_scaler.mean_).reshape(-1)],
    "feature_scale": [float(value) for value in np.asarray(feature_scaler.scale_).reshape(-1)],
    "crossfit_folds": n_folds,
    "crossfit_seed": crossfit_seed,
  }
  _write_private_json(private_run_dir / "surrogate_model.json", private_surrogate)

  return {
    "n_paired_valid": aligned.height,
    "split_counts": {
      name: {"n": len(partition), "events": int(y[partition].sum())}
      for name, partition in indices.items()
    },
    "best_c": {"M_A_ECG": probe_a.best_c, "M_B_ECG": probe_b.best_c},
    "hybrid_tau_a": tau_a,
    "hybrid_tau_predicted_gap": tau_delta,
    "hybrid_validation_b_utilization": float(route_by_split["val"].mean()),
    "hybrid_validation_extra_discordance_utilization_within_low_a": extra_validation_utilization,
    "hybrid_test_b_utilization": float(route_hybrid.mean()),
    "surrogate_alpha": best_alpha,
    "surrogate_validation_mse": best_mse,
    "hybrid_policy_rows": metrics_frame.height,
    "hybrid_bootstrap_rows": hybrid_bootstrap.height,
    "internal_benchmark_bootstrap_rows": internal_benchmark.height,
  }, [
    "hybrid_discordance_policy.csv",
    "hybrid_discordance_overlap.csv",
    "hybrid_discordance_bootstrap.csv",
    "internal_resplit_benchmark_bootstrap.csv",
    "hybrid_branch_composition.csv",
    "alternative_policy_cutoffs.json",
  ]


def _publish_staged_public_outputs(
  staging_dir: Path,
  public_dir: Path,
  output_names: list[str],
) -> None:
  """Publish complete aggregates and leave no stale completion marker on error."""
  public_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = public_dir / "manifest.json"
  manifest_path.unlink(missing_ok=True)
  try:
    for name in output_names:
      os.replace(staging_dir / name, public_dir / name)
    os.replace(staging_dir / "manifest.json", manifest_path)
  except BaseException:
    manifest_path.unlink(missing_ok=True)
    raise


def run_probability_cascade_audit_round02(config_path: Path) -> dict[str, Any]:
  """Run branch-composition, hybrid-routing, and resplit-bootstrap analyses."""
  config_path = config_path.resolve()
  config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
  if not isinstance(config, dict) or config.get("version") != 1:
    raise ValueError("Round-02 audit runner requires a version-1 YAML mapping.")
  run_id = str(config.get("run_id", ""))
  if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_id):
    raise ValueError("run_id contains invalid characters.")
  private_root = Path("artifacts/probability_cascade/audit_round02").resolve()
  output_root = Path(config.get("output_root", private_root)).resolve()
  if not output_root.is_relative_to(private_root):
    raise ValueError("Private audit outputs must stay under artifacts/probability_cascade/audit_round02/.")
  public_dir = Path(config.get("public_output_root", "docs/probability_cascade/audit_round02")).resolve()
  if public_dir != Path("docs/probability_cascade/audit_round02").resolve():
    raise ValueError("Round-02 public outputs must use docs/probability_cascade/audit_round02/.")
  private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
  private_run_dir = output_root / run_id
  private_run_dir.mkdir(mode=0o700, exist_ok=False)
  staging_public_dir = private_run_dir / "public_outputs"
  staging_public_dir.mkdir(mode=0o700)

  source_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
  ).stdout.strip()
  source_v2_dir = Path(config["source_v2_run_dir"])
  source_audit_dir = Path(config["source_audit_run_dir"])
  v2_manifest = json.loads((source_v2_dir / "manifest.json").read_text(encoding="utf-8"))
  round01_manifest = json.loads((source_audit_dir / "manifest.json").read_text(encoding="utf-8"))
  if v2_manifest.get("status") != "complete" or round01_manifest.get("status") != "complete":
    raise ValueError("Both round-01 source runs must have complete private manifests.")
  if v2_manifest.get("run_id") != config["source_v2_run_id"]:
    raise ValueError("V2 source run ID does not match the requested run.")
  if round01_manifest.get("run_id") != config["source_audit_run_id"]:
    raise ValueError("Round-01 source run ID does not match the requested run.")

  manifest: dict[str, Any] = {
    "status": "running",
    "run_id": run_id,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": source_commit,
    "config_sha256": _sha256_file(config_path),
    "analysis_status": "post_hoc_existing_test_and_internal_resplit",
    "source_v2_run_id": config["source_v2_run_id"],
    "source_audit_run_id": config["source_audit_run_id"],
    "private_output_directory": str(private_run_dir),
    "public_outputs": [],
  }
  _write_private_json(private_run_dir / "manifest.json", manifest)
  try:
    inputs = config["inputs"]
    all_scores, validation, assessment = _load_frozen_scores(source_v2_dir, Path(inputs["cohort_path"]))
    composition = _branch_composition_rows(
      validation, assessment, float(config["review_capacity"])
    )
    _write_private_csv(composition, private_run_dir / "branch_review_composition.csv")
    _write_public_csv(composition, staging_public_dir / "branch_review_composition.csv")
    resplit_result, resplit_outputs = _hybrid_policy_outputs(
      config, source_audit_dir, private_run_dir, staging_public_dir
    )
    output_names = ["branch_review_composition.csv", *resplit_outputs]
    if any(not (staging_public_dir / name).is_file() for name in output_names):
      raise FileNotFoundError("An expected round-02 aggregate output was not written.")
    source_hashes = {
      "v2_scores.parquet": _sha256_file(source_v2_dir / "scores.parquet"),
      "v2_manifest.json": _sha256_file(source_v2_dir / "manifest.json"),
      "round01_manifest.json": _sha256_file(source_audit_dir / "manifest.json"),
      "round01_internal_resplit_scores.parquet": _sha256_file(source_audit_dir / "internal_resplit_scores.parquet"),
      "round01_probe_a.json": _sha256_file(source_audit_dir / "refit_models/M_A_ECG.json"),
      "round01_probe_b.json": _sha256_file(source_audit_dir / "refit_models/M_B_ECG.json"),
      "primary_cohort.parquet": _sha256_file(Path(inputs["cohort_path"])),
      "ciis_features.npy": _sha256_file(Path(inputs["features_a_path"])),
      "dbeta_embeddings.npy": _sha256_file(Path(inputs["embeddings_b_path"])),
      "ciis_metadata.parquet": _sha256_file(Path(inputs["meta_a_path"])),
      "dbeta_metadata.parquet": _sha256_file(Path(inputs["meta_b_path"])),
      "pair_validity.parquet": _sha256_file(Path(inputs["pair_validity_path"])),
    }
    manifest.update({
      "status": "complete",
      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
      "existing_test_analysis": {
        "analysis_status": "post_hoc_exploratory",
        "branch_composition_rows": composition.height,
        "n_paired_scores": all_scores.height,
        "n_assessment": assessment.height,
        "n_assessment_events": int(assessment.get_column("mortality_30d").sum()),
      },
      "internal_resplit_analysis": resplit_result,
      "source_artifact_sha256_by_filename": source_hashes,
      "hybrid_route_definition": "15% high A risk plus 5% predicted positive p_B-p_A gap among lower-A patients; surrogate is trained on development out-of-fold scores and takes A features only at inference",
      "internal_resplit_scope": "The cohort has been previously analyzed; this is an internal resplit, not temporal, external, or prospective confirmation.",
      "privacy": "Public outputs are aggregate tables, policy cutoffs, and a sanitized manifest; row-level predictions, keys, fold scores, and surrogate parameters stay private.",
      "public_outputs": output_names + ["manifest.json"],
    })
    _write_private_json(private_run_dir / "manifest.json", manifest)
    public_manifest = {
      key: value for key, value in manifest.items()
      if key not in {"private_output_directory", "software"}
    }
    (staging_public_dir / "manifest.json").write_text(
      json.dumps(public_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _publish_staged_public_outputs(staging_public_dir, public_dir, output_names)
    staging_public_dir.rmdir()
    return manifest
  except BaseException as exc:
    manifest.update({
      "status": "incomplete",
      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
      "failure_type": type(exc).__name__,
      "failure_message": str(exc),
    })
    _write_private_json(private_run_dir / "manifest.json", manifest)
    raise


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  args = parser.parse_args()
  result = run_probability_cascade_audit_round02(args.config)
  print(json.dumps({
    "status": result["status"],
    "run_id": result["run_id"],
    "source_commit": result["source_commit"],
    "existing_test_analysis": result["existing_test_analysis"],
    "internal_resplit_analysis": result["internal_resplit_analysis"],
    "public_outputs": result["public_outputs"],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
