"""Reproducible end-to-end runner for the v2 ECG probability cascade."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from ecg_rep.probability_cascade import (
  RoutingPolicy,
  StageAScore,
  StageBScore,
  fit_reference_cutoff,
  replay_scores,
)
from ecg_rep.probability_cascade_calibration import (
  apply_routed_branch_risk_maps,
  calibration_metrics,
  decision_curve_net_benefit,
  dca_reference_net_benefit,
  fit_branch_risk_maps,
  fit_logistic_risk_map,
)
from ecg_rep.probability_cascade_frontier import (
  bootstrap_primary_frontier,
  evaluate_frontier,
  top_capacity_mask,
)
from ecg_rep.probe import TrainedProbe
from ecg_rep.representations import verify_representation_artifact_integrity


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _sha256_key(dataset: str, subject_id: int, study_id: int) -> str:
  """Hash a private ECG key for A/B row agreement; this is not a waveform hash."""
  return hashlib.sha256(f"{dataset}\0{subject_id}\0{study_id}".encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  with path.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
  path.chmod(0o600)


def _write_private_frame(frame: pl.DataFrame, path: Path) -> None:
  frame.write_parquet(path)
  path.chmod(0o600)


def _write_private_csv(frame: pl.DataFrame, path: Path) -> None:
  frame.write_csv(path)
  path.chmod(0o600)


def _validate_unique_keys(frame: pl.DataFrame, source: str) -> None:
  if frame.select(
    pl.col("subject_id").is_null().any() | pl.col("study_id").is_null().any()
  ).item():
    raise ValueError(f"{source} contains missing ECG keys.")
  if frame.get_column("subject_id").n_unique() != len(frame):
    raise ValueError(f"{source} must contain one index ECG per patient.")
  if frame.get_column("study_id").n_unique() != len(frame):
    raise ValueError(f"{source} must contain one row per ECG study ID.")


def _check_key_coverage(reference: pl.DataFrame, candidate: pl.DataFrame, source: str) -> None:
  keys = ["subject_id", "study_id"]
  missing = reference.select(keys).join(candidate.select(keys), on=keys, how="anti").height
  extra = candidate.select(keys).join(reference.select(keys), on=keys, how="anti").height
  if missing or extra:
    raise ValueError(f"{source} key mismatch: {missing} cohort keys missing and {extra} extra keys.")


def _batched_probe_scores(
  probe: TrainedProbe,
  matrix: np.ndarray,
  valid_indices: np.ndarray,
  *,
  batch_size: int = 2048,
) -> np.ndarray:
  if matrix.ndim != 2 or matrix.shape[1] != len(probe.coefficients):
    raise ValueError("Representation dimensions do not match frozen probe coefficients.")
  scores = np.empty(len(valid_indices), dtype=np.float64)
  for start in range(0, len(valid_indices), batch_size):
    indices = valid_indices[start : start + batch_size]
    features = np.asarray(matrix[indices], dtype=np.float64)
    if not np.isfinite(features).all():
      raise ValueError("A row marked representation-valid contains a non-finite feature.")
    batch_scores = probe.predict_proba(features)
    if not np.isfinite(batch_scores).all() or ((batch_scores < 0.0) | (batch_scores > 1.0)).any():
      raise ValueError("Frozen probe returned a non-finite or out-of-range probability.")
    scores[start : start + len(indices)] = batch_scores
  return scores


def _load_and_score_inputs(
  config: dict[str, Any],
  run_dir: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
  inputs = config["inputs"]
  cohort_path = Path(inputs["cohort_path"])
  split_path = Path(inputs["split_path"])
  reps_dir = Path(inputs["reps_dir"])
  models_dir = Path(inputs["models_dir"])
  meta_a_path = Path(inputs["meta_a_path"])
  features_a_path = Path(inputs["features_a_path"])
  meta_b_path = Path(inputs["meta_b_path"])
  embeddings_b_path = Path(inputs["embeddings_b_path"])
  pair_validity_path = Path(inputs["pair_validity_path"])
  model_a_path = models_dir / "M_A_ECG.joblib"
  model_b_path = models_dir / "M_B_ECG.joblib"
  for path in (
    cohort_path, split_path, meta_a_path, features_a_path, meta_b_path,
    embeddings_b_path, pair_validity_path, model_a_path, model_b_path,
  ):
    if not path.exists():
      raise FileNotFoundError(f"Required probability-cascade input is missing: {path}")

  cohort_keys = pl.scan_parquet(cohort_path).select(
    ["subject_id", "study_id", "split"]
  ).collect()
  _validate_unique_keys(cohort_keys, "primary cohort")
  if set(cohort_keys.get_column("split").unique().to_list()) != {"dev", "val", "test"}:
    raise ValueError("Cohort must contain exactly dev, val, and test split assignments.")
  if cohort_keys.group_by("subject_id").agg(pl.col("split").n_unique()).filter(pl.col("split") != 1).height:
    raise ValueError("A patient occurs in more than one cohort split.")

  split_file = pl.read_csv(split_path)
  split_file = split_file.select(["subject_id", "study_id", "split"])
  _validate_unique_keys(split_file, "split assignment file")
  _check_key_coverage(cohort_keys, split_file, "split assignment file")
  mismatched_splits = cohort_keys.join(
    split_file.rename({"split": "split_file"}), on=["subject_id", "study_id"]
  ).filter(pl.col("split") != pl.col("split_file")).height
  if mismatched_splits:
    raise ValueError(f"Split file disagrees with the cohort for {mismatched_splits} ECGs.")

  verify_representation_artifact_integrity(reps_dir, cohort_keys, check_b=True)
  meta_a = pl.read_parquet(meta_a_path)
  meta_b = pl.read_parquet(meta_b_path)
  pair = pl.read_parquet(pair_validity_path)
  for frame, name in ((meta_a, "CIIS metadata"), (meta_b, "D-BETA metadata"), (pair, "pair-validity table")):
    _validate_unique_keys(frame, name)
    _check_key_coverage(cohort_keys, frame, name)

  valid_a = meta_a.get_column("representation_a_valid")
  valid_b = meta_b.get_column("representation_b_valid")
  if valid_a.null_count() or valid_b.null_count():
    raise ValueError("Representation validity flags may not be missing.")
  pair_check = meta_a.select(["subject_id", "study_id", "representation_a_valid"]).join(
    meta_b.select(["subject_id", "study_id", "representation_b_valid"]),
    on=["subject_id", "study_id"],
  ).join(
    pair.select(["subject_id", "study_id", "representation_pair_valid"]),
    on=["subject_id", "study_id"],
  )
  pair_mismatches = pair_check.filter(
    (pl.col("representation_a_valid") & pl.col("representation_b_valid"))
    != pl.col("representation_pair_valid")
  ).height
  if pair_mismatches:
    raise ValueError(f"Pair-validity artifact disagrees with A/B flags for {pair_mismatches} ECGs.")

  matrix_a = np.load(features_a_path, mmap_mode="r")
  matrix_b = np.load(embeddings_b_path, mmap_mode="r")
  if matrix_a.shape[0] != len(meta_a) or matrix_b.shape[0] != len(meta_b):
    raise ValueError("Representation metadata and matrix row counts differ.")
  if matrix_a.shape[1] != 15 or matrix_b.shape[1] != 768:
    raise ValueError("Expected the primary 15-feature CIIS and frozen 768-dimensional D-BETA inputs.")
  probe_a = TrainedProbe.load(model_a_path)
  probe_b = TrainedProbe.load(model_b_path)
  if len(probe_a.coefficients) != matrix_a.shape[1] or len(probe_b.coefficients) != matrix_b.shape[1]:
    raise ValueError("Frozen probe dimensions do not match their representation matrices.")
  if probe_a.feature_names is not None and probe_a.feature_names != tuple(f"za_{i}" for i in range(matrix_a.shape[1])):
    raise ValueError("Frozen CIIS probe feature order does not match the cached CIIS matrix contract.")
  if probe_b.feature_names is not None and probe_b.feature_names != tuple(f"zb_{i}" for i in range(matrix_b.shape[1])):
    raise ValueError("Frozen D-BETA probe feature order does not match the cached embedding matrix contract.")

  indices_a = np.flatnonzero(valid_a.to_numpy())
  indices_b = np.flatnonzero(valid_b.to_numpy())
  scores_a = _batched_probe_scores(probe_a, matrix_a, indices_a)
  scores_b = _batched_probe_scores(probe_b, matrix_b, indices_b)
  scores_a_full = np.full(len(meta_a), np.nan, dtype=np.float64)
  scores_a_full[indices_a] = scores_a
  scores_b_full = np.full(len(meta_b), np.nan, dtype=np.float64)
  scores_b_full[indices_b] = scores_b
  scored_a = meta_a.select(["subject_id", "study_id"]).with_columns(pl.Series("p_a_ecg", scores_a_full))
  scored_b = meta_b.select(["subject_id", "study_id"]).with_columns(pl.Series("p_b_ecg", scores_b_full))
  paired = cohort_keys.join(scored_a, on=["subject_id", "study_id"]).join(
    scored_b, on=["subject_id", "study_id"]
  ).filter(pl.col("p_a_ecg").is_finite() & pl.col("p_b_ecg").is_finite())
  pair_expected = pair.filter(pl.col("representation_pair_valid")).height
  if len(paired) != pair_expected:
    raise ValueError(f"Scored paired rows ({len(paired)}) differ from pair-valid records ({pair_expected}).")
  _validate_unique_keys(paired, "paired scored cohort")
  id_hashes = [
    _sha256_key("mimic", int(subject), int(study))
    for subject, study in zip(paired.get_column("subject_id"), paired.get_column("study_id"), strict=True)
  ]
  paired = paired.with_columns(pl.Series("input_sha256", id_hashes)).sort(["subject_id", "study_id"])
  score_frame = paired.select([
    "subject_id", "study_id", "split", "p_a_ecg", "p_b_ecg", "input_sha256"
  ])

  score_path = run_dir / "scores.parquet"
  _write_private_frame(score_frame, score_path)
  score_split_hashes: dict[str, str] = {}
  for split_name in ("dev", "val", "test"):
    split_out = run_dir / f"scores_{split_name}.parquet"
    _write_private_frame(score_frame.filter(pl.col("split") == split_name), split_out)
    score_split_hashes[split_name] = _sha256_file(split_out)

  prov_a = json.loads((reps_dir / "ciis_features_provenance.json").read_text(encoding="utf-8"))
  prov_b = json.loads((reps_dir / "dbeta_embedding_provenance.json").read_text(encoding="utf-8"))
  schema_b_hash = hashlib.sha256(json.dumps({
    "representation": "dbeta_embedding",
    "dimension": matrix_b.shape[1],
    "model_name": prov_b.get("model_name"),
    "model_revision": prov_b.get("model_revision"),
    "preprocess": prov_b.get("preprocess"),
  }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
  input_manifest = {
    str(path): _sha256_file(path) for path in (
      cohort_path, split_path, meta_a_path, features_a_path, meta_b_path,
      embeddings_b_path, pair_validity_path, model_a_path, model_b_path,
      reps_dir / "ciis_features_provenance.json", reps_dir / "dbeta_embedding_provenance.json",
    )
  }
  summary = {
    "n_cohort": len(cohort_keys),
    "n_ciis_valid": int(valid_a.sum()),
    "n_dbeta_valid": int(valid_b.sum()),
    "n_pair_valid": pair_expected,
    "n_unpaired_or_invalid": len(cohort_keys) - pair_expected,
    "split_counts": {
      split_name: int(score_frame.filter(pl.col("split") == split_name).height)
      for split_name in ("dev", "val", "test")
    },
    "source_hashes": input_manifest,
    "score_table_sha256": _sha256_file(score_path),
    "score_table_sha256_by_split": score_split_hashes,
    "model_sha256": {"M_A_ECG": input_manifest[str(model_a_path)], "M_B_ECG": input_manifest[str(model_b_path)]},
    "feature_schema_sha256": {"M_A_ECG": prov_a["feature_schema_hash"], "M_B_ECG": schema_b_hash},
    "dbeta_model": {
      "name": prov_b.get("model_name"),
      "revision": prov_b.get("model_revision"),
      "frozen_encoder": prov_b.get("frozen_encoder"),
      "dimension": prov_b.get("dimension"),
    },
    "input_hash_semantics": "SHA-256 of dataset ID plus subject_id/study_id; aligns A/B rows but is not a waveform content checksum.",
  }
  return paired, summary


def _fit_policies(
  reference: pl.DataFrame,
  *,
  targets: list[float],
  model_hashes: dict[str, str],
  feature_hashes: dict[str, str],
  reference_sha256: str,
) -> list[tuple[dict[str, Any], RoutingPolicy]]:
  pa = np.asarray(reference.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  result: list[tuple[dict[str, Any], RoutingPolicy]] = []
  for target in targets:
    fit = fit_reference_cutoff(pa.tolist(), target, "val", reference_source_sha256=reference_sha256)
    policy = RoutingPolicy(
      version="probability_cascade_v1",
      model_scope="ecg_only",
      reference_dataset="mimic",
      outcome_horizon_days=30,
      route_mode="threshold",
      tau_r=fit.tau_r,
      inclusive=True,
      model_hash_a=model_hashes["M_A_ECG"],
      model_hash_b=model_hashes["M_B_ECG"],
      feature_schema_hash_a=feature_hashes["M_A_ECG"],
      feature_schema_hash_b=feature_hashes["M_B_ECG"],
      reference_split="val",
      reference_source_sha256=reference_sha256,
      target_utilization=target,
    )
    result.append((asdict(fit) | {"policy_id": policy.policy_id, "target_utilization": target}, policy))
  return result


def _replay_policies(
  score_frame: pl.DataFrame,
  policies: list[tuple[dict[str, Any], RoutingPolicy]],
  run_dir: Path,
) -> None:
  route_rows: list[dict[str, Any]] = []
  b_reference_rows: list[dict[str, Any]] = []
  ordered = score_frame.sort(["subject_id", "study_id"])
  for fit_payload, policy in policies:
    scores_a = [
      StageAScore(
        probability=float(row["p_a_ecg"]),
        model_scope=policy.model_scope,
        dataset_id=policy.reference_dataset,
        outcome_horizon_days=policy.outcome_horizon_days,
        model_hash=policy.model_hash_a,
        feature_schema_hash=policy.feature_schema_hash_a,
        input_sha256=str(row["input_sha256"]),
        subject_id=int(row["subject_id"]),
        study_id=int(row["study_id"]),
      )
      for row in ordered.iter_rows(named=True)
    ]
    scores_b = [
      StageBScore(
        probability=float(row["p_b_ecg"]),
        model_scope=policy.model_scope,
        dataset_id=policy.reference_dataset,
        outcome_horizon_days=policy.outcome_horizon_days,
        model_hash=policy.model_hash_b,
        feature_schema_hash=policy.feature_schema_hash_b,
        input_sha256=str(row["input_sha256"]),
        subject_id=int(row["subject_id"]),
        study_id=int(row["study_id"]),
      )
      for row in ordered.iter_rows(named=True)
    ]
    result = replay_scores(scores_a, policy, scores_b, allow_precomputed_b_reference=True)
    split_values = ordered.get_column("split").to_list()
    for split_name, replay_row in zip(split_values, result.rows, strict=True):
      route_rows.append({
        **asdict(replay_row),
        "split": split_name,
        "target_utilization": policy.target_utilization,
        "tau_r": policy.tau_r,
        "analysis_status": "post_hoc_exploratory",
      })
    if result.counterfactual_b_reference is not None:
      for split_name, audit_row in zip(split_values, result.counterfactual_b_reference, strict=True):
        b_reference_rows.append({**asdict(audit_row), "split": split_name})

  route_frame = pl.DataFrame(route_rows)
  _write_private_frame(route_frame, run_dir / "routed_predictions.parquet")
  _write_private_frame(pl.DataFrame(b_reference_rows), run_dir / "counterfactual_b_reference.parquet")


def _calibration_outputs(
  outcomes: pl.DataFrame,
  policies: list[tuple[dict[str, Any], RoutingPolicy]],
  run_dir: Path,
  thresholds: list[float],
  bootstrap_replicates: int,
  seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[float, dict[str, np.ndarray]]]:
  val = outcomes.filter(pl.col("split") == "val").sort(["subject_id", "study_id"])
  test = outcomes.filter(pl.col("split") == "test").sort(["subject_id", "study_id"])
  y_val = np.asarray(val.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  y_test = np.asarray(test.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pa_val = np.asarray(val.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb_val = np.asarray(val.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  pa_test = np.asarray(test.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb_test = np.asarray(test.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  map_a = fit_logistic_risk_map(y_val, pa_val, minimum_slope_events=50)
  map_b = fit_logistic_risk_map(y_val, pb_val, minimum_slope_events=50)
  test_scores: dict[float, dict[str, np.ndarray]] = {}
  metric_rows: list[dict[str, Any]] = []
  bin_rows: list[dict[str, Any]] = []
  model_dir = run_dir / "final_risk_models"
  model_dir.mkdir(mode=0o700)
  _write_json(model_dir / "M_A_ECG_global.json", {"fit_split": "val", "risk_map": map_a.to_dict()})
  _write_json(model_dir / "M_B_ECG_global.json", {"fit_split": "val", "risk_map": map_b.to_dict()})

  def record_metrics(split_name: str, model_name: str, probabilities: np.ndarray, y: np.ndarray) -> None:
    metrics, bins = calibration_metrics(y, probabilities)
    metric_rows.append({"split": split_name, "model": model_name, **metrics})
    if split_name == "test_post_hoc_exploratory":
      bin_rows.extend({"model": model_name, **row} for row in bins)

  record_metrics("val_fit_apparent", "A_raw", pa_val, y_val)
  record_metrics("val_fit_apparent", "B_all_raw", pb_val, y_val)
  record_metrics("val_fit_apparent", "A_global_map", map_a.predict(pa_val), y_val)
  record_metrics("val_fit_apparent", "B_global_map", map_b.predict(pb_val), y_val)
  record_metrics("test_post_hoc_exploratory", "A_raw", pa_test, y_test)
  record_metrics("test_post_hoc_exploratory", "B_all_raw", pb_test, y_test)
  record_metrics("test_post_hoc_exploratory", "A_global_map", map_a.predict(pa_test), y_test)
  record_metrics("test_post_hoc_exploratory", "B_global_map", map_b.predict(pb_test), y_test)

  for fit_payload, policy in policies:
    if policy.target_utilization is None or policy.tau_r is None:
      raise ValueError("Calibration requires numeric routing policies with a target and threshold.")
    target = float(policy.target_utilization)
    tau_r = float(policy.tau_r)
    route_val = pa_val >= tau_r
    route_test = pa_test >= tau_r
    raw_val = np.where(route_val, pb_val, pa_val)
    raw_test = np.where(route_test, pb_test, pa_test)
    global_map = fit_logistic_risk_map(y_val, raw_val, minimum_slope_events=50)
    low_map, high_map, support = fit_branch_risk_maps(
      y_val, pa_val, pb_val, route_val,
      minimum_slope_events=50,
      minimum_branch_events=10,
    )
    global_test = global_map.predict(raw_test)
    branch_test = apply_routed_branch_risk_maps(
      pa_test, pb_test[route_test], route_test, low_map, high_map
    )
    test_scores[target] = {
      "A_raw": pa_test,
      "B_all_raw": pb_test,
      "A_global_map": map_a.predict(pa_test),
      "B_global_map": map_b.predict(pb_test),
      "cascade_raw": raw_test,
      "cascade_global": global_test,
      "cascade_branch": branch_test,
      "route_test": route_test.astype(np.float64),
    }
    record_metrics("val_fit_apparent", f"cascade_{int(target * 100):02d}_raw", raw_val, y_val)
    record_metrics("val_fit_apparent", f"cascade_{int(target * 100):02d}_global", global_map.predict(raw_val), y_val)
    record_metrics(
      "val_fit_apparent",
      f"cascade_{int(target * 100):02d}_branch",
      apply_routed_branch_risk_maps(pa_val, pb_val[route_val], route_val, low_map, high_map),
      y_val,
    )
    record_metrics("test_post_hoc_exploratory", f"cascade_{int(target * 100):02d}_raw", raw_test, y_test)
    record_metrics("test_post_hoc_exploratory", f"cascade_{int(target * 100):02d}_global", global_test, y_test)
    record_metrics("test_post_hoc_exploratory", f"cascade_{int(target * 100):02d}_branch", branch_test, y_test)
    prefix = f"cascade_{int(target * 100):02d}"
    _write_json(model_dir / f"{prefix}_global.json", {
      "fit_split": "val",
      "policy_id": policy.policy_id,
      "risk_map": global_map.to_dict(),
    })
    _write_json(model_dir / f"{prefix}_branch.json", {
      "fit_split": "val",
      "policy_id": policy.policy_id,
      "branch_support": support,
      "minimum_slope_events": 50,
      "minimum_branch_events": 10,
      "low_a_map": low_map.to_dict(),
      "routed_b_map": high_map.to_dict(),
    })

  metric_frame = pl.DataFrame(metric_rows)
  bin_frame = pl.DataFrame(bin_rows)
  _write_private_csv(metric_frame, run_dir / "calibration_metrics.csv")
  _write_private_csv(bin_frame, run_dir / "calibration_bins.csv")

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
  target = min(test_scores, key=lambda value: abs(value - 0.20))
  for model_name in (
    "A_raw", "B_all_raw", f"cascade_{int(target * 100):02d}_raw",
    f"cascade_{int(target * 100):02d}_global", f"cascade_{int(target * 100):02d}_branch",
  ):
    selected_bins = [row for row in bin_rows if row["model"] == model_name]
    if selected_bins:
      axes[0].plot(
        [row["mean_predicted"] for row in selected_bins],
        [row["observed_rate"] for row in selected_bins],
        marker="o", linewidth=1.4, label=model_name,
      )
  axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
  axes[0].set(xlabel="Mean predicted risk", ylabel="Observed event rate", title="Post-hoc test reliability")
  axes[0].legend(fontsize=7)
  selected_metrics = metric_frame.filter(
    (pl.col("split") == "test_post_hoc_exploratory")
    & pl.col("model").is_in([f"cascade_{int(target * 100):02d}_raw", f"cascade_{int(target * 100):02d}_global", f"cascade_{int(target * 100):02d}_branch"])
  )
  axes[1].bar(selected_metrics.get_column("model").to_list(), selected_metrics.get_column("brier").to_list(), color=["#6b7280", "#2563eb", "#10b981"])
  axes[1].set(ylabel="Brier score", title=f"Cascade candidates at {target:.0%} reference routing")
  axes[1].tick_params(axis="x", labelrotation=25, labelsize=8)
  fig.savefig(run_dir / "calibration_curves.png", dpi=160)
  plt.close(fig)
  (run_dir / "calibration_curves.png").chmod(0o600)

  decisions = (
    "# Calibration decisions\n\n"
    "Status: diagnostic / exploratory; no calibration PASS is claimed. Frozen base probes were not refit. "
    "Risk maps were fit on validation outcomes; validation was also used historically to select each probe's regularization, "
    "so the fit-split diagnostics are apparent and the calibration maps are not independent of base-model tuning. "
    "The original MIMIC test was previously examined, so test rows are reported as post-hoc exploratory.\n\n"
    "For each numeric A-risk route, the runner fit a global intercept/slope logistic map and branch-aware A/B maps. "
    "Branch slopes require at least 50 events and 50 non-events; branches with fewer than 10 events use the global map, "
    "and intermediate support uses intercept-only recalibration. This support rule is a practical exploratory fallback.\n\n"
    "The global map is the simple reference candidate; branch maps and raw mixtures remain sensitivity analyses. "
    "Selection is not based on test metrics. Weak absolute-risk calibration changes the DCA interpretation but does not affect the count-based frontier.\n"
  )
  (run_dir / "calibration_decisions.md").write_text(decisions, encoding="utf-8")
  (run_dir / "calibration_decisions.md").chmod(0o600)
  return metric_frame, bin_frame, pl.DataFrame(), test_scores


def _dca_outputs(
  test_outcomes: pl.DataFrame,
  test_scores: dict[float, dict[str, np.ndarray]],
  policies: list[tuple[dict[str, Any], RoutingPolicy]],
  thresholds: list[float],
  bootstrap_replicates: int,
  seed: int,
  run_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
  y = np.asarray(test_outcomes.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  records: list[dict[str, Any]] = []
  contrasts: list[dict[str, Any]] = []
  for fit_payload, policy in policies:
    if policy.target_utilization is None:
      raise ValueError("Decision analysis requires a numeric routing target.")
    target = float(policy.target_utilization)
    candidates = test_scores[target]
    methods = {
      "A_only_raw": candidates["A_raw"],
      "A_only_global": candidates["A_global_map"],
      "B_all_raw": candidates["B_all_raw"],
      "B_all_global": candidates["B_global_map"],
      "cascade_raw_diagnostic": candidates["cascade_raw"],
      "cascade_global": candidates["cascade_global"],
      "cascade_branch": candidates["cascade_branch"],
    }
    cascade_utilization = float(candidates["route_test"].mean())
    for method, probabilities in methods.items():
      if method.startswith("A_only"):
        method_utilization = 0.0
      elif method.startswith("B_all"):
        method_utilization = 1.0
      else:
        method_utilization = cascade_utilization
      for threshold in thresholds:
        nb = decision_curve_net_benefit(y, probabilities, threshold)
        treat_none, treat_all = dca_reference_net_benefit(y, threshold)
        records.append({
          "target_b_utilization": target,
          "realized_b_utilization_test": method_utilization,
          "method": method,
          "action_threshold": threshold,
          "net_benefit": nb,
          "treat_none_net_benefit": treat_none,
          "treat_all_net_benefit": treat_all,
          "n_assessment": len(y),
          "events": int(y.sum()),
          "analysis_status": "post_hoc_exploratory_hypothetical_action",
        })
    for method in ("cascade_raw_diagnostic", "cascade_global", "cascade_branch"):
      for baseline in _dca_baselines(method):
        for threshold in thresholds:
          contrast = decision_curve_net_benefit(y, methods[method], threshold) - decision_curve_net_benefit(y, methods[baseline], threshold)
          contrasts.append({
            "target_b_utilization": target,
            "method": method,
            "baseline": baseline,
            "action_threshold": threshold,
            "net_benefit_difference": contrast,
            "ci_lower": None,
            "ci_upper": None,
            "n_bootstraps": 0,
          })

  # Paired patient bootstrap is limited to the 20% route frontier for a first pass.
  primary_target = min(test_scores, key=lambda value: abs(value - 0.20))
  if bootstrap_replicates > 0:
    rng = np.random.default_rng(seed)
    methods = test_scores[primary_target]
    baseline_scores = {
      "A_only_raw": methods["A_raw"],
      "B_all_raw": methods["B_all_raw"],
      "A_only_global": methods["A_global_map"],
      "B_all_global": methods["B_global_map"],
    }
    candidate_scores = {
      "cascade_raw_diagnostic": methods["cascade_raw"],
      "cascade_global": methods["cascade_global"],
      "cascade_branch": methods["cascade_branch"],
    }
    differences: dict[tuple[str, str, float], list[float]] = {}
    for method in candidate_scores:
      for baseline in _dca_baselines(method):
        for threshold in thresholds:
          differences[(method, baseline, threshold)] = []
    for _ in range(bootstrap_replicates):
      sample = rng.integers(0, len(y), size=len(y))
      y_sample = y[sample]
      for threshold in thresholds:
        for method, probabilities in candidate_scores.items():
          cascade_nb = decision_curve_net_benefit(y_sample, probabilities[sample], threshold)
          for baseline in _dca_baselines(method):
            differences[(method, baseline, threshold)].append(
              cascade_nb - decision_curve_net_benefit(y_sample, baseline_scores[baseline][sample], threshold)
            )
    for row in contrasts:
      if row["target_b_utilization"] != primary_target:
        continue
      values = differences[(row["method"], row["baseline"], row["action_threshold"])]
      row["ci_lower"] = float(np.quantile(values, 0.025))
      row["ci_upper"] = float(np.quantile(values, 0.975))
      row["n_bootstraps"] = bootstrap_replicates
      row["bootstrap_seed"] = seed

  dca_frame = pl.DataFrame(records)
  contrast_frame = pl.DataFrame(contrasts)
  _write_private_csv(dca_frame, run_dir / "dca_scenarios.csv")
  _write_private_csv(contrast_frame, run_dir / "dca_contrasts.csv")

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
  primary_target = min(test_scores, key=lambda value: abs(value - 0.20))
  primary_rows = dca_frame.filter(pl.col("target_b_utilization") == primary_target)
  for method in (
    "A_only_global", "B_all_global", "cascade_global", "cascade_branch", "cascade_raw_diagnostic"
  ):
    subset = primary_rows.filter(pl.col("method") == method).sort("action_threshold")
    axes[0].plot(subset.get_column("action_threshold"), subset.get_column("net_benefit"), marker="o", label=method)
  references = primary_rows.filter(pl.col("method") == "A_only_raw").sort("action_threshold")
  axes[0].plot(references.get_column("action_threshold"), references.get_column("treat_all_net_benefit"), linestyle="--", color="black", label="treat all")
  axes[0].axhline(0.0, linestyle=":", color="gray", label="treat none")
  axes[0].set(xlabel="Hypothetical review threshold", ylabel="Net benefit per patient", title="Whole-cohort DCA scenario")
  axes[0].legend(fontsize=7)
  for threshold in (0.05, 0.10):
    subset = dca_frame.filter(
      (pl.col("action_threshold") == threshold)
      & pl.col("method").is_in(["A_only_global", "B_all_global", "cascade_global"])
    ).sort("realized_b_utilization_test")
    axes[1].plot(
      subset.get_column("realized_b_utilization_test"),
      subset.get_column("net_benefit"), marker="o", label=f"global risk map, threshold {threshold:.0%}",
    )
  axes[1].set(xlabel="Observed Stage B route fraction", ylabel="Net benefit per patient", title="DCA by resource use")
  axes[1].legend(fontsize=7)
  fig.savefig(run_dir / "dca_vs_utilization.png", dpi=160)
  plt.close(fig)
  (run_dir / "dca_vs_utilization.png").chmod(0o600)
  return dca_frame, contrast_frame


def _dca_baselines(method: str) -> tuple[str, str]:
  """Pair raw cascades with raw probes and calibrated cascades with calibrated probes."""
  if method == "cascade_raw_diagnostic":
    return "A_only_raw", "B_all_raw"
  if method in ("cascade_global", "cascade_branch"):
    return "A_only_global", "B_all_global"
  raise ValueError(f"No paired DCA baselines are defined for {method!r}.")


def _frontier_outputs(
  reference: pl.DataFrame,
  assessment: pl.DataFrame,
  policies: list[tuple[dict[str, Any], RoutingPolicy]],
  run_dir: Path,
  seed: int,
  bootstrap_replicates: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
  if any(policy.target_utilization is None for _, policy in policies):
    raise ValueError("Frontier analysis requires numeric routing targets.")
  targets = tuple(float(policy.target_utilization) for _, policy in policies if policy.target_utilization is not None)
  summary, overlap, gate_misses, _fits = evaluate_frontier(
    reference,
    assessment,
    targets=targets,
    review_capacities=(0.05,),
    seed=seed,
  )
  identity_columns = [
    "routing_method", "final_score_method", "target_b_utilization", "tau_r",
    "review_capacity", "full_b_top_capacity_events", "full_b_event_identities_recovered",
    "full_b_identity_recovery", "full_b_event_identities_missed", "cascade_only_event_identities",
  ]
  identity = summary.select(identity_columns)
  _write_private_csv(summary, run_dir / "frontier_long.csv")
  _write_private_csv(identity, run_dir / "identity_overlap.csv")
  _write_private_csv(gate_misses, run_dir / "gate_miss_counts.csv")
  _write_private_csv(summary, run_dir / "routing_summary.csv")
  _write_private_csv(overlap, run_dir / "identity_reference.csv")

  ordered_assessment = assessment.sort(["subject_id", "study_id"])
  y_assessment = np.asarray(ordered_assessment.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pa_assessment = np.asarray(ordered_assessment.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb_assessment = np.asarray(ordered_assessment.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  subjects_assessment = np.asarray(ordered_assessment.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies_assessment = np.asarray(ordered_assessment.get_column("study_id").to_numpy(), dtype=np.int64)
  full_b_review = top_capacity_mask(
    pb_assessment, subjects_assessment, studies_assessment, 0.05
  )
  full_b_events = full_b_review & (y_assessment == 1)
  risk_rows: list[dict[str, Any]] = []
  for _fit_payload, policy in policies:
    if policy.target_utilization is None or policy.tau_r is None:
      raise ValueError("Gate-miss score summaries require numeric routing policies.")
    routed_events = full_b_events & (pa_assessment >= policy.tau_r)
    gate_missed_events = full_b_events & (pa_assessment < policy.tau_r)
    gate_scores_a = pa_assessment[gate_missed_events]
    gate_scores_b = pb_assessment[gate_missed_events]
    retained_scores_a = pa_assessment[routed_events]
    retained_scores_b = pb_assessment[routed_events]
    risk_row: dict[str, Any] = {
      "target_b_utilization": policy.target_utilization,
      "tau_r": policy.tau_r,
      "n_full_b_top_capacity_events": int(full_b_events.sum()),
      "n_full_b_events_routed": int(routed_events.sum()),
      "n_full_b_events_missed_at_a_gate": int(gate_missed_events.sum()),
      "n_gate_misses_with_p_a_at_or_above_tau": int(np.sum(gate_scores_a >= policy.tau_r)),
    }
    for name, values in (
      ("gate_missed_p_a", gate_scores_a),
      ("gate_missed_p_b", gate_scores_b),
      ("routed_p_a", retained_scores_a),
      ("routed_p_b", retained_scores_b),
    ):
      for statistic, value in (
        ("min", np.min(values) if len(values) else None),
        ("median", np.median(values) if len(values) else None),
        ("mean", np.mean(values) if len(values) else None),
        ("max", np.max(values) if len(values) else None),
      ):
        risk_row[f"{name}_{statistic}"] = float(value) if value is not None else None
    risk_rows.append(risk_row)
  gate_risk = pl.DataFrame(risk_rows)
  _write_private_csv(gate_risk, run_dir / "gate_miss_risk_summary.csv")

  primary = min(targets, key=lambda value: abs(value - 0.20))
  fit = next(item for _, item in policies if item.target_utilization is not None and abs(item.target_utilization - primary) < 1e-8)
  if fit.tau_r is None:
    raise ValueError("Primary frontier bootstrap requires a numeric routing threshold.")
  bootstrap = bootstrap_primary_frontier(
    assessment, fit.tau_r, n_bootstraps=bootstrap_replicates, seed=seed
  )
  _write_private_csv(bootstrap, run_dir / "frontier_bootstrap.csv")

  reference_y = np.asarray(reference.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  assessment_y = np.asarray(assessment.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  reference_pa = np.asarray(reference.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  reference_pb = np.asarray(reference.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  assessment_pa = np.asarray(assessment.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  assessment_pb = np.asarray(assessment.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  drift_rows: list[dict[str, Any]] = []
  for _fit_payload, policy in policies:
    if policy.target_utilization is None or policy.tau_r is None:
      raise ValueError("Case-mix drift analysis requires numeric routing policies.")
    reference_route = reference_pa >= policy.tau_r
    assessment_route = assessment_pa >= policy.tau_r
    reference_utilization = float(reference_route.mean())
    assessment_utilization = float(assessment_route.mean())
    reference_rate = float(reference_y.mean())
    assessment_rate = float(assessment_y.mean())
    drift_rows.append({
      "reference_split": "val",
      "assessment_split": "test_post_hoc_exploratory",
      "target_b_utilization": policy.target_utilization,
      "tau_r": policy.tau_r,
      "n_reference": len(reference_y),
      "events_reference": int(reference_y.sum()),
      "mortality_rate_reference": reference_rate,
      "mean_p_a_reference": float(reference_pa.mean()),
      "mean_p_b_reference": float(reference_pb.mean()),
      "reference_b_route_count": int(reference_route.sum()),
      "reference_b_route_fraction": reference_utilization,
      "n_assessment": len(assessment_y),
      "events_assessment": int(assessment_y.sum()),
      "mortality_rate_assessment": assessment_rate,
      "mortality_rate_difference_test_minus_val": assessment_rate - reference_rate,
      "mean_p_a_assessment": float(assessment_pa.mean()),
      "mean_p_b_assessment": float(assessment_pb.mean()),
      "assessment_b_route_count": int(assessment_route.sum()),
      "assessment_b_route_fraction": assessment_utilization,
      "route_fraction_difference_test_minus_val": assessment_utilization - reference_utilization,
    })
  drift_frame = pl.DataFrame(drift_rows)
  _write_private_csv(drift_frame, run_dir / "case_mix_threshold_drift.csv")

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
  numeric = summary.filter(pl.col("routing_method").is_in(["A_only", "fixed_numeric_A_risk", "B_all"]))
  for method, group in numeric.group_by("routing_method", maintain_order=True):
    label = method[0] if isinstance(method, tuple) else method
    group = group.sort("realized_b_utilization")
    axes[0].plot(group.get_column("realized_b_utilization"), group.get_column("sensitivity"), marker="o", label=label)
    axes[1].plot(group.get_column("realized_b_utilization"), group.get_column("full_b_identity_recovery"), marker="o", label=label)
  axes[0].set(xlabel="Observed Stage B route fraction", ylabel="Sensitivity at 5% review capacity", title="Total event capture")
  axes[1].set(xlabel="Observed Stage B route fraction", ylabel="Full-B event identity recovery", title="Identity frontier")
  axes[0].legend(fontsize=8)
  axes[1].legend(fontsize=8)
  fig.savefig(run_dir / "frontier.png", dpi=160)
  plt.close(fig)
  (run_dir / "frontier.png").chmod(0o600)

  # Summarize the previously noted 20% versus 40% non-monotonic pattern.
  numeric_rows = summary.filter(pl.col("routing_method") == "fixed_numeric_A_risk")
  row20 = numeric_rows.filter((pl.col("target_b_utilization") - 0.20).abs() < 1e-8).row(0, named=True)
  row40 = numeric_rows.filter((pl.col("target_b_utilization") - 0.40).abs() < 1e-8).row(0, named=True)
  ordered_assessment = assessment.sort(["subject_id", "study_id"])
  y = np.asarray(ordered_assessment.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  pa = np.asarray(ordered_assessment.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb = np.asarray(ordered_assessment.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  subjects = np.asarray(ordered_assessment.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(ordered_assessment.get_column("study_id").to_numpy(), dtype=np.int64)
  policy20 = next(item for _, item in policies if item.target_utilization is not None and abs(item.target_utilization - 0.20) < 1e-8)
  policy40 = next(item for _, item in policies if item.target_utilization is not None and abs(item.target_utilization - 0.40) < 1e-8)
  if policy20.tau_r is None or policy40.tau_r is None or policy40.tau_r > policy20.tau_r:
    raise ValueError("The 20%-to-40% routing sensitivity requires nested numeric risk cutoffs.")
  route20 = pa >= policy20.tau_r
  route40 = pa >= policy40.tau_r
  if (route20 & ~route40).any():
    raise ValueError("The 40% route must include every patient routed under the 20% cutoff.")
  cascade20 = np.where(route20, pb, pa)
  cascade40 = np.where(route40, pb, pa)
  selected20 = top_capacity_mask(cascade20, subjects, studies, 0.05) & (y == 1)
  selected40 = top_capacity_mask(cascade40, subjects, studies, 0.05) & (y == 1)
  event_keys20 = {
    (int(subjects[index]), int(studies[index])) for index in np.flatnonzero(selected20)
  }
  event_keys40 = {
    (int(subjects[index]), int(studies[index])) for index in np.flatnonzero(selected40)
  }
  lost_at_40 = event_keys20 - event_keys40
  gained_at_40 = event_keys40 - event_keys20
  row_by_key = {
    (int(subject), int(study)): index
    for index, (subject, study) in enumerate(zip(subjects, studies, strict=True))
  }
  newly_routed_band = route40 & ~route20
  lost_newly_routed = [key for key in lost_at_40 if newly_routed_band[row_by_key[key]]]
  gained_newly_routed = [key for key in gained_at_40 if newly_routed_band[row_by_key[key]]]
  robustness = pl.DataFrame([{
    "comparison": "numeric_40pct_vs_20pct_reference",
    "reference_fraction_20": 0.20,
    "reference_fraction_40": 0.40,
    "tau_r_20": row20["tau_r"],
    "tau_r_40": row40["tau_r"],
    "realized_b_utilization_20": row20["realized_b_utilization"],
    "realized_b_utilization_40": row40["realized_b_utilization"],
    "delta_sensitivity_at_5pct": row40["sensitivity"] - row20["sensitivity"],
    "delta_full_b_identity_recovery": row40["full_b_identity_recovery"] - row20["full_b_identity_recovery"],
    "delta_events_captured": row40["true_positives"] - row20["true_positives"],
    "n_events_captured_20pct": len(event_keys20),
    "n_events_captured_40pct": len(event_keys40),
    "n_events_captured_by_both": len(event_keys20 & event_keys40),
    "n_events_lost_at_40pct": len(lost_at_40),
    "n_events_gained_at_40pct": len(gained_at_40),
    "n_newly_routed_patients_20_to_40": int(newly_routed_band.sum()),
    "n_newly_routed_events_20_to_40": int(np.sum(newly_routed_band & (y == 1))),
    "n_20pct_only_events_newly_routed_at_40pct": len(lost_newly_routed),
    "n_lost_newly_routed_events_with_b_below_a": sum(
      pb[row_by_key[key]] < pa[row_by_key[key]] for key in lost_newly_routed
    ),
    "n_40pct_only_events_newly_routed_at_40pct": len(gained_newly_routed),
    "n_gained_newly_routed_events_with_b_above_a": sum(
      pb[row_by_key[key]] > pa[row_by_key[key]] for key in gained_newly_routed
    ),
  }])
  _write_private_csv(robustness, run_dir / "robustness_summary.csv")
  return summary, identity, gate_misses, bootstrap, drift_frame, gate_risk


def run_probability_cascade(config_path: Path) -> dict[str, Any]:
  """Run score generation, numeric routing, frontier, calibration, and DCA."""
  config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
  if not isinstance(config, dict) or config.get("version") != 2:
    raise ValueError("Probability-cascade runner requires a version-2 YAML mapping.")
  if config.get("research_mode") != "exploratory":
    raise ValueError("Set research_mode: exploratory for this research-only run.")
  if config.get("analysis_status") not in (
    "post_hoc_exploratory", "internal_resplit", "new_temporal_evaluation", "external_validation"
  ):
    raise ValueError("analysis_status must identify the evidence class.")
  study = config.get("study")
  routing = config.get("routing")
  assessment = config.get("assessment")
  dca_config = config.get("decision_analysis")
  if not isinstance(study, dict):
    raise ValueError("Config study section must be a mapping.")
  if not isinstance(routing, dict):
    raise ValueError("Config routing section must be a mapping.")
  if not isinstance(assessment, dict):
    raise ValueError("Config assessment section must be a mapping.")
  if not isinstance(dca_config, dict):
    raise ValueError("Config must provide study, routing, assessment, and decision_analysis mappings.")
  if study.get("cohort") != "mimic" or study.get("horizon_days") != 30 or study.get("primary_model_pair") != "ECG_only_A_vs_B":
    raise ValueError("Primary runner is scoped to ECG-only MIMIC 30-day mortality.")
  if assessment.get("split") != "test" or not study.get("previous_test_inspected"):
    raise ValueError("This run replays the previously inspected MIMIC test and must be post hoc exploratory.")
  run_id = str(config.get("run_id", ""))
  if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", run_id):
    raise ValueError("run_id must contain only letters, digits, dot, underscore, or hyphen.")
  output_root = Path(config.get("output_root", "artifacts/probability_cascade/v2")).resolve()
  private_root = Path("artifacts/probability_cascade/v2").resolve()
  if not output_root.is_relative_to(private_root):
    raise ValueError("Patient-level outputs must remain under artifacts/probability_cascade/v2/.")
  output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
  run_dir = output_root / run_id
  run_dir.mkdir(mode=0o700, exist_ok=False)

  manifest: dict[str, Any] = {
    "status": "running",
    "run_id": run_id,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
    "config_path": str(config_path),
    "config_sha256": _sha256_file(config_path),
    "analysis_status": config["analysis_status"],
    "research_mode": config["research_mode"],
    "study": {"cohort": "mimic", "outcome": "mortality_30d", "horizon_days": 30, "model_scope": "ecg_only"},
    "split_design": {"source": "existing frozen patient-level dev/val/test assignments", "reference_split": routing.get("reference_split"), "assessment_split": "test"},
    "random_seed": int(assessment.get("seed", 42)),
    "software": {"python": sys.version, "platform": platform.platform()},
    "outputs": {},
  }
  try:
    paired, score_summary = _load_and_score_inputs(config, run_dir)
    reference = paired.filter(pl.col("split") == routing.get("reference_split"))
    if routing.get("reference_split") != "val":
      raise ValueError("Current v2 cascade reference must be the frozen-probe validation split.")

    model_hashes = score_summary["model_sha256"]
    feature_hashes = score_summary["feature_schema_sha256"]
    reference_path = run_dir / "scores_val.parquet"
    policies = _fit_policies(
      reference,
      targets=[float(value) for value in routing.get("reference_target_b_fractions", [])],
      model_hashes=model_hashes,
      feature_hashes=feature_hashes,
      reference_sha256=_sha256_file(reference_path),
    )
    policy_records = [
      {**fit, "policy": policy.to_dict(), "model_hash_a": policy.model_hash_a, "model_hash_b": policy.model_hash_b}
      for fit, policy in policies
    ]
    _write_json(run_dir / "routing_policies.json", {
      "status": "fitted_on_validation_A_scores_only",
      "reference_split": "val",
      "reference_score_table_sha256": _sha256_file(reference_path),
      "policies": policy_records,
    })
    _replay_policies(paired.select([
      "subject_id", "study_id", "split", "p_a_ecg", "p_b_ecg", "input_sha256"
    ]), policies, run_dir)

    # Outcome columns are joined only after thresholds and route assignments are frozen.
    reference_labels = pl.scan_parquet(config["inputs"]["cohort_path"]).select(
      ["subject_id", "study_id", "mortality_30d"]
    ).collect()
    paired_labels = paired.join(reference_labels, on=["subject_id", "study_id"], how="inner")
    outcomes = paired_labels.join(
      paired.select(["subject_id", "study_id", "split", "p_a_ecg", "p_b_ecg", "input_sha256"]),
      on=["subject_id", "study_id", "split"],
    ).with_columns(pl.col("mortality_30d").cast(pl.Int64, strict=True))
    if outcomes.get_column("mortality_30d").null_count() or not outcomes.get_column("mortality_30d").is_in([0, 1]).all():
      raise ValueError("Paired assessment outcomes must be non-missing binary 30-day mortality labels.")
    _validate_unique_keys(outcomes, "paired outcome table")

    reference_for_frontier = paired_labels.filter(pl.col("split") == "val").select([
      "subject_id", "study_id", "split", "p_a_ecg", "p_b_ecg", "mortality_30d"
    ])
    assessment_for_frontier = paired_labels.filter(pl.col("split") == "test").select([
      "subject_id", "study_id", "split", "p_a_ecg", "p_b_ecg", "mortality_30d"
    ])
    summary, identity, gate_misses, frontier_bootstrap, case_mix_drift, gate_risk = _frontier_outputs(
      reference_for_frontier,
      assessment_for_frontier,
      policies,
      run_dir,
      int(assessment.get("seed", 42)),
      int(assessment.get("bootstrap_replicates_first_pass", 200)),
    )

    calibration_metrics_frame, calibration_bins, _, test_scores = _calibration_outputs(
      outcomes,
      policies,
      run_dir,
      [float(value) for value in dca_config.get("thresholds", [])],
      int(assessment.get("bootstrap_replicates_first_pass", 200)),
      int(assessment.get("seed", 42)),
    )
    test_outcomes = outcomes.filter(pl.col("split") == "test").sort(["subject_id", "study_id"])
    dca_frame, dca_contrasts = _dca_outputs(
      test_outcomes,
      test_scores,
      policies,
      [float(value) for value in dca_config.get("thresholds", [])],
      int(assessment.get("bootstrap_replicates_first_pass", 200)),
      int(assessment.get("seed", 42)),
      run_dir,
    )
    manifest.update({
      "status": "complete",
      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
      "cohort_and_score_audit": score_summary,
      "routing_reference": {"split": "val", "thresholds": [fit for fit, _policy in policies]},
      "assessment": {
        "split": "test",
        "n_patients": len(test_outcomes),
        "n_events": int(test_outcomes.get_column("mortality_30d").sum()),
        "analysis_status": "post_hoc_exploratory",
        "previously_examined": True,
      },
      "interpretation": {
        "mimic_claim_scope": "in-domain downstream evaluation; D-BETA was pretrained on MIMIC-IV-ECG",
        "resource": "routed fraction is a policy assignment; offline cached-B replay does not measure runtime compute savings",
        "dca_action": "hypothetical ECG-time chart review/reassessment; not clinician-endorsed care",
        "calibration": "exploratory maps fitted on validation; no calibration PASS is claimed",
      },
      "headline_counts": {
        "frontier_rows": len(summary),
        "identity_rows": len(identity),
        "gate_miss_rows": len(gate_misses),
        "calibration_rows": len(calibration_metrics_frame),
        "calibration_bins": len(calibration_bins),
        "dca_rows": len(dca_frame),
        "dca_contrasts": len(dca_contrasts),
        "frontier_bootstrap_rows": len(frontier_bootstrap),
        "case_mix_threshold_drift_rows": len(case_mix_drift),
        "gate_miss_risk_rows": len(gate_risk),
      },
      "outputs": {
        name: str(run_dir / name) for name in (
          "scores.parquet", "scores_dev.parquet", "scores_val.parquet", "scores_test.parquet",
          "routing_policies.json", "routed_predictions.parquet", "counterfactual_b_reference.parquet",
          "routing_summary.csv", "frontier_long.csv", "identity_overlap.csv", "gate_miss_counts.csv",
          "identity_reference.csv",
          "gate_miss_risk_summary.csv",
          "case_mix_threshold_drift.csv",
          "frontier_bootstrap.csv", "robustness_summary.csv", "calibration_metrics.csv",
          "calibration_bins.csv", "calibration_curves.png", "calibration_decisions.md",
          "dca_scenarios.csv", "dca_contrasts.csv", "dca_vs_utilization.png", "final_risk_models",
        )
      },
    })
    _write_json(run_dir / "manifest.json", manifest)
    return manifest
  except BaseException as exc:
    manifest.update({
      "status": "incomplete",
      "completed_at_utc": datetime.now(timezone.utc).isoformat(),
      "failure_type": type(exc).__name__,
      "failure_message": str(exc),
    })
    _write_json(run_dir / "manifest.json", manifest)
    raise

