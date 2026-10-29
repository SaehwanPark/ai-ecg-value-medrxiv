"""Reproduce the round-01 probability-cascade audit extensions.

Patient-level tables and refit probes stay in the ignored artifact tree. Only
aggregate CSVs and a sanitized manifest are written below the public report
directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import yaml

from ecg_rep.probability_cascade import fit_reference_cutoff
from ecg_rep.probability_cascade_calibration import (
  LogisticRiskMap,
  apply_routed_branch_risk_maps,
  calibration_metrics,
  fit_branch_risk_maps,
  fit_logistic_risk_map,
)
from ecg_rep.probability_cascade_frontier import (
  bootstrap_route_pair_contrast,
  evaluate_branch_calibrated_frontier,
  top_capacity_mask,
)
from ecg_rep.probe import ProbeConfig, TrainedProbe, fit_logistic_probe
from ecg_rep.representations import verify_representation_artifact_integrity


_FORBIDDEN_PUBLIC_COLUMNS = {
  "subject_id", "study_id", "hadm_id", "patient_id", "ecg_id",
  "file_name", "path", "input_sha256",
}
_INTERNAL_SPLIT_FRACTIONS = {
  "dev": 0.50,
  "val": 0.15,
  "calibration_fit": 0.10,
  "calibration_check": 0.10,
  "test": 0.15,
}


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  path.chmod(0o600)


def _write_private_csv(frame: pl.DataFrame, path: Path) -> None:
  frame.write_csv(path)
  path.chmod(0o600)


def _write_private_parquet(frame: pl.DataFrame, path: Path) -> None:
  frame.write_parquet(path)
  path.chmod(0o600)


def _write_public_csv(frame: pl.DataFrame, path: Path) -> None:
  forbidden = _FORBIDDEN_PUBLIC_COLUMNS & {column.lower() for column in frame.columns}
  forbidden |= {
    column.lower() for column in frame.columns
    if column.lower().endswith("_subject_id") or column.lower().endswith("_study_id")
  }
  if forbidden:
    raise ValueError(f"Refusing to publish direct-identifier columns: {sorted(forbidden)}")
  frame.write_csv(path)


def _keys_equal(left: pl.DataFrame, right: pl.DataFrame, source: str) -> None:
  keys = ["subject_id", "study_id"]
  missing = left.select(keys).join(right.select(keys), on=keys, how="anti").height
  extra = right.select(keys).join(left.select(keys), on=keys, how="anti").height
  if missing or extra:
    raise ValueError(f"{source} key mismatch: {missing} missing and {extra} extra.")


def _as_binary(values: npt.ArrayLike, source: str) -> npt.NDArray[np.int64]:
  raw = np.asarray(values)
  if raw.ndim != 1 or not np.isin(raw, (False, True, 0, 1)).all():
    raise ValueError(f"{source} must be a one-dimensional binary array.")
  return np.asarray(raw, dtype=np.int64)


def _event_keys(
  selected: npt.NDArray[np.bool_],
  y: npt.NDArray[np.int64],
  subjects: npt.NDArray[np.int64],
  studies: npt.NDArray[np.int64],
) -> set[tuple[int, int]]:
  return {
    (int(subjects[index]), int(studies[index]))
    for index in np.flatnonzero(selected & (y == 1))
  }


def _capacity_metrics(
  *,
  policy: str,
  score_method: str,
  frame: pl.DataFrame,
  route: npt.NDArray[np.bool_],
  score: npt.NDArray[np.float64],
  full_b_probability: npt.NDArray[np.float64],
  capacity: float,
  reference_utilization: float | None,
) -> tuple[dict[str, Any], npt.NDArray[np.bool_]]:
  ordered = frame.sort(["subject_id", "study_id"])
  y = _as_binary(ordered.get_column("mortality_30d").to_numpy(), "mortality_30d")
  subjects = np.asarray(ordered.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(ordered.get_column("study_id").to_numpy(), dtype=np.int64)
  if len(route) != len(y) or len(score) != len(y) or len(full_b_probability) != len(y):
    raise ValueError("Policy arrays must align with the assessment rows.")
  selected = top_capacity_mask(score, subjects, studies, capacity)
  full_b_selected = top_capacity_mask(full_b_probability, subjects, studies, capacity)
  selected_events = _event_keys(selected, y, subjects, studies)
  full_b_events = _event_keys(full_b_selected, y, subjects, studies)
  recovered = selected_events & full_b_events
  tp = int(np.sum(selected & (y == 1)))
  row = {
    "routing_method": policy,
    "final_score_method": score_method,
    "n_patients": len(y),
    "n_events_total": int(y.sum()),
    "reference_b_utilization": reference_utilization,
    "test_b_utilization": float(route.mean()),
    "review_capacity": capacity,
    "n_selected": int(selected.sum()),
    "events_captured": tp,
    "sensitivity": tp / int(y.sum()) if int(y.sum()) else None,
    "ppv": tp / int(selected.sum()) if selected.any() else None,
    "full_b_top_capacity_events": len(full_b_events),
    "full_b_event_identities_recovered": len(recovered),
    "full_b_identity_recovery": len(recovered) / len(full_b_events) if full_b_events else None,
    "full_b_event_identities_missed": len(full_b_events - selected_events),
  }
  return row, selected


def _selected_overlap(
  left: npt.NDArray[np.bool_], right: npt.NDArray[np.bool_]
) -> dict[str, float | int | None]:
  intersection = int(np.sum(left & right))
  union = int(np.sum(left | right))
  n_left = int(left.sum())
  return {
    "n_selected_patient_overlap": intersection,
    "fraction_of_capacity_overlapping": intersection / n_left if n_left else None,
    "selected_patient_jaccard": intersection / union if union else None,
  }


def _load_frozen_scores(
  source_run_dir: Path,
  cohort_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
  scores = pl.read_parquet(source_run_dir / "scores.parquet").sort(["subject_id", "study_id"])
  cohort = pl.read_parquet(cohort_path).select(["subject_id", "study_id", "mortality_30d"])
  if cohort.get_column("subject_id").n_unique() != len(cohort):
    raise ValueError("Source cohort must contain one index ECG per patient.")
  joined = scores.join(cohort, on=["subject_id", "study_id"], how="left")
  if len(joined) != len(scores) or joined.get_column("mortality_30d").null_count():
    raise ValueError("Frozen score keys do not have complete aligned outcomes.")
  joined = joined.with_columns(pl.col("mortality_30d").cast(pl.Int64, strict=True))
  labels = _as_binary(joined.get_column("mortality_30d").to_numpy(), "mortality_30d")
  if len(labels) != joined.height:
    raise ValueError("Outcome conversion changed the paired score row count.")
  return joined, joined.filter(pl.col("split") == "val"), joined.filter(pl.col("split") == "test")


def _existing_run_outputs(
  config: dict[str, Any],
  private_run_dir: Path,
  public_dir: Path,
) -> dict[str, Any]:
  source_run_dir = Path(config["source_run_dir"])
  cohort_path = Path(config["inputs"]["cohort_path"])
  if not source_run_dir.is_dir():
    raise FileNotFoundError(f"Source v2 run directory is missing: {source_run_dir}")
  source_manifest = json.loads((source_run_dir / "manifest.json").read_text(encoding="utf-8"))
  if source_manifest.get("status") != "complete" or source_manifest.get("run_id") != config["source_run_id"]:
    raise ValueError("The selected source run manifest is incomplete or has a different run ID.")
  all_scores, val, test = _load_frozen_scores(source_run_dir, cohort_path)
  capacity = float(config["review_capacity"])
  targets = tuple(float(value) for value in config["routing_targets"])
  if targets != (0.05, 0.10, 0.20, 0.40):
    raise ValueError("Round-01 audit expects numeric route targets 5%, 10%, 20%, and 40%.")

  calibrated, overlap, scores = evaluate_branch_calibrated_frontier(
    val, test, targets=targets, capacity=capacity
  )
  raw_score_pairs = {
    "raw_selective_mixture": (scores[0.20]["raw"], scores[0.40]["raw"]),
    "branch_recalibrated": (scores[0.20]["branch"], scores[0.40]["branch"]),
  }
  uncertainty = bootstrap_route_pair_contrast(
    test,
    raw_score_pairs,  # type: ignore[arg-type]
    capacity=capacity,
    n_bootstraps=int(config["bootstrap_replicates"]),
    seed=int(config["seed"]),
  )
  _write_private_csv(calibrated, private_run_dir / "branch_calibrated_frontier.csv")
  _write_private_csv(overlap, private_run_dir / "branch_score_overlap.csv")
  _write_private_csv(uncertainty, private_run_dir / "twenty_forty_bootstrap.csv")
  _write_public_csv(calibrated, public_dir / "branch_calibrated_frontier.csv")
  _write_public_csv(overlap, public_dir / "branch_score_overlap.csv")
  _write_public_csv(uncertainty, public_dir / "twenty_forty_bootstrap.csv")

  primary_scores = all_scores.sort(["subject_id", "study_id"])
  val_scores = primary_scores.filter(pl.col("split") == "val").sort(["subject_id", "study_id"])
  test_scores = primary_scores.filter(pl.col("split") == "test").sort(["subject_id", "study_id"])
  pa_val = np.asarray(val_scores.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pa_test = np.asarray(test_scores.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb_test = np.asarray(test_scores.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  y_val = _as_binary(val_scores.get_column("mortality_30d").to_numpy(), "validation outcomes")
  y_test = _as_binary(test_scores.get_column("mortality_30d").to_numpy(), "test outcomes")
  subjects = np.asarray(test_scores.get_column("subject_id").to_numpy(), dtype=np.int64)
  studies = np.asarray(test_scores.get_column("study_id").to_numpy(), dtype=np.int64)
  numeric_fit = fit_reference_cutoff(pa_val.tolist(), 0.20, "val")
  q_low, q_high = np.quantile(pa_val, [0.40, 0.60], method="linear")
  routes = {
    "numeric_A_top20": (pa_val >= numeric_fit.tau_r, pa_test >= numeric_fit.tau_r),
    "middle_A_risk_band_p40_p60": (
      (pa_val >= q_low) & (pa_val < q_high),
      (pa_test >= q_low) & (pa_test < q_high),
    ),
  }
  numeric_raw_test = np.where(routes["numeric_A_top20"][1], pb_test, pa_test)
  numeric20_selected = top_capacity_mask(numeric_raw_test, subjects, studies, capacity)
  alternative_rows: list[dict[str, Any]] = []
  alternative_overlap_rows: list[dict[str, Any]] = []
  policy_records: list[dict[str, Any]] = []

  for name, (route_val, route_test) in routes.items():
    low_map, high_map, support = fit_branch_risk_maps(
      y_val,
      pa_val,
      np.asarray(val_scores.get_column("p_b_ecg").to_numpy(), dtype=np.float64),
      route_val,
      minimum_slope_events=50,
      minimum_branch_events=10,
    )
    route_util_val = float(route_val.mean())
    route_util_test = float(route_test.mean())
    raw_test = np.where(route_test, pb_test, pa_test)
    branch_test = apply_routed_branch_risk_maps(
      pa_test, pb_test[route_test], route_test, low_map, high_map
    )
    raw_row, raw_selected = _capacity_metrics(
      policy=name, score_method="raw_selective_mixture", frame=test_scores,
      route=route_test, score=raw_test, full_b_probability=pb_test,
      capacity=capacity, reference_utilization=route_util_val,
    )
    branch_row, branch_selected = _capacity_metrics(
      policy=name, score_method="branch_recalibrated", frame=test_scores,
      route=route_test, score=branch_test, full_b_probability=pb_test,
      capacity=capacity, reference_utilization=route_util_val,
    )
    if name == "numeric_A_top20":
      route_metadata = {
        "route_cutoff_definition": "p_a_ecg >= tau_r",
        "route_cutoff_low": float(numeric_fit.tau_r),
        "route_cutoff_high": None,
      }
    else:
      route_metadata = {
        "route_cutoff_definition": "q40 <= p_a_ecg < q60",
        "route_cutoff_low": float(q_low),
        "route_cutoff_high": float(q_high),
      }
    raw_row.update(route_metadata)
    branch_row.update(route_metadata)
    alternative_rows.extend([raw_row, branch_row])
    baseline_raw_selected = numeric20_selected
    baseline_branch = np.asarray(scores[0.20]["branch"], dtype=np.float64)
    baseline_branch_selected = top_capacity_mask(baseline_branch, subjects, studies, capacity)
    alternative_overlap_rows.extend([
      {
        "routing_method": name,
        "final_score_method": "raw_selective_mixture",
        **_selected_overlap(baseline_raw_selected, raw_selected),
      },
      {
        "routing_method": name,
        "final_score_method": "branch_recalibrated",
        **_selected_overlap(baseline_branch_selected, branch_selected),
      },
    ])
    policy_records.append({
      "routing_method": name,
      "fit_split": "val",
      "n_val": len(pa_val),
      "val_b_utilization": route_util_val,
      "test_b_utilization": route_util_test,
      "cutoff_low": float(q_low) if name == "middle_A_risk_band_p40_p60" else None,
      "cutoff_high": float(q_high) if name == "middle_A_risk_band_p40_p60" else None,
      "tau_r": float(numeric_fit.tau_r) if name == "numeric_A_top20" else None,
      "branch_fit_support": support,
    })

  alternative_frame = pl.DataFrame(alternative_rows)
  alternative_overlap = pl.DataFrame(alternative_overlap_rows)
  _write_private_csv(alternative_frame, private_run_dir / "alternative_routing.csv")
  _write_private_csv(alternative_overlap, private_run_dir / "alternative_routing_overlap.csv")
  _write_private_json(private_run_dir / "alternative_routing_policies.json", {"policies": policy_records})
  _write_public_csv(alternative_frame, public_dir / "alternative_routing.csv")
  _write_public_csv(alternative_overlap, public_dir / "alternative_routing_overlap.csv")
  (public_dir / "alternative_routing_policies.json").write_text(
    json.dumps({"policies": policy_records}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )

  return {
    "source_run_id": config["source_run_id"],
    "source_scores_sha256": _sha256_file(source_run_dir / "scores.parquet"),
    "source_manifest_sha256": _sha256_file(source_run_dir / "manifest.json"),
    "n_paired_scores": all_scores.height,
    "n_validation": val.height,
    "n_test": test.height,
    "n_test_events": int(y_test.sum()),
    "branch_frontier_rows": calibrated.height,
    "branch_overlap_rows": overlap.height,
    "twenty_forty_bootstrap_rows": uncertainty.height,
    "alternative_routing_rows": alternative_frame.height,
    "alternative_routing_overlap_rows": alternative_overlap.height,
    "analysis_status": "post_hoc_exploratory",
  }


def _patient_disjoint_split(
  subject_ids: npt.ArrayLike, seed: int
) -> npt.NDArray[np.str_]:
  """Assign unique patients independently of outcomes to disjoint partitions."""
  subjects = np.asarray(subject_ids, dtype=np.int64)
  if subjects.ndim != 1 or len(np.unique(subjects)) != len(subjects):
    raise ValueError("Internal split requires one unique row per patient.")
  split_names = tuple(_INTERNAL_SPLIT_FRACTIONS)
  fractions = np.asarray(list(_INTERNAL_SPLIT_FRACTIONS.values()), dtype=np.float64)
  if not np.isclose(float(fractions.sum()), 1.0):
    raise ValueError("Internal split fractions must sum to one.")
  rng = np.random.default_rng(seed)
  indices = rng.permutation(len(subjects))
  boundaries = np.r_[0, np.rint(np.cumsum(fractions) * len(subjects)).astype(np.int64)]
  boundaries[-1] = len(subjects)
  split = np.full(len(subjects), "", dtype="<U32")
  for split_index, name in enumerate(split_names):
    split[indices[boundaries[split_index] : boundaries[split_index + 1]]] = name
  return split


def _predict_probe_rows(
  probe: TrainedProbe,
  matrix: npt.NDArray[np.floating[Any]],
  row_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
  scores = np.empty(len(row_indices), dtype=np.float64)
  for start in range(0, len(row_indices), 2048):
    batch_indices = row_indices[start : start + 2048]
    batch = np.asarray(matrix[batch_indices], dtype=np.float64)
    if not np.isfinite(batch).all():
      raise ValueError("A paired-valid feature row contains non-finite values.")
    batch_scores = probe.predict_proba(batch)
    if not np.isfinite(batch_scores).all() or ((batch_scores < 0.0) | (batch_scores > 1.0)).any():
      raise ValueError("A refit probe returned invalid probabilities.")
    scores[start : start + len(batch_indices)] = batch_scores
  return scores


def _load_paired_feature_index(
  inputs: dict[str, Any],
) -> tuple[pl.DataFrame, npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
  cohort = pl.read_parquet(inputs["cohort_path"]).select([
    "subject_id", "study_id", "mortality_30d"
  ])
  if cohort.get_column("subject_id").n_unique() != len(cohort):
    raise ValueError("The primary cohort must contain one index ECG per patient.")
  if cohort.get_column("mortality_30d").null_count():
    raise ValueError("The internal resplit requires complete 30-day outcomes.")
  cohort = cohort.with_columns(pl.col("mortality_30d").cast(pl.Int64, strict=True))
  if not cohort.get_column("mortality_30d").is_in([0, 1]).all():
    raise ValueError("The internal resplit requires binary 30-day outcomes.")
  verify_representation_artifact_integrity(
    Path(inputs["reps_dir"]), cohort.select(["subject_id", "study_id"]), check_b=True
  )

  meta_a = pl.read_parquet(inputs["meta_a_path"]).with_row_index("a_row")
  meta_b = pl.read_parquet(inputs["meta_b_path"]).with_row_index("b_row")
  pair = pl.read_parquet(inputs["pair_validity_path"])
  for frame, name in ((meta_a, "CIIS metadata"), (meta_b, "D-BETA metadata"), (pair, "pair validity")):
    _keys_equal(cohort, frame, name)

  joined = (
    cohort.join(meta_a.select(["subject_id", "study_id", "a_row", "representation_a_valid"]),
                on=["subject_id", "study_id"], how="inner")
    .join(meta_b.select(["subject_id", "study_id", "b_row", "representation_b_valid"]),
          on=["subject_id", "study_id"], how="inner")
    .join(pair.select(["subject_id", "study_id", "representation_pair_valid"]),
          on=["subject_id", "study_id"], how="inner")
  )
  pair_mismatch = joined.filter(
    (pl.col("representation_a_valid") & pl.col("representation_b_valid"))
    != pl.col("representation_pair_valid")
  )
  if pair_mismatch.height:
    raise ValueError("Pair-validity flags disagree with A/B representation validity.")
  paired = joined.filter(pl.col("representation_pair_valid")).sort(["subject_id", "study_id"])
  if paired.get_column("subject_id").n_unique() != len(paired):
    raise ValueError("Paired-valid rows must contain one ECG per patient.")

  matrix_a = np.load(inputs["features_a_path"], mmap_mode="r")
  matrix_b = np.load(inputs["embeddings_b_path"], mmap_mode="r")
  if matrix_a.ndim != 2 or matrix_b.ndim != 2:
    raise ValueError("Representation matrices must be two-dimensional.")
  if matrix_a.shape != (len(meta_a), 15) or matrix_b.shape != (len(meta_b), 768):
    raise ValueError("Representation row counts or dimensions do not match metadata.")
  return paired, matrix_a, matrix_b


def _internal_resplit_outputs(
  config: dict[str, Any],
  private_run_dir: Path,
  public_dir: Path,
) -> dict[str, Any]:
  inputs = config["inputs"]
  paired, matrix_a, matrix_b = _load_paired_feature_index(inputs)
  y = _as_binary(paired.get_column("mortality_30d").to_numpy(), "paired outcomes")
  split_labels = _patient_disjoint_split(
    paired.get_column("subject_id").to_numpy(), int(config["internal_split_seed"])
  )
  paired = paired.with_columns(pl.Series("split", split_labels))
  row_a = np.asarray(paired.get_column("a_row").to_numpy(), dtype=np.int64)
  row_b = np.asarray(paired.get_column("b_row").to_numpy(), dtype=np.int64)
  split_indices = {
    name: np.flatnonzero(split_labels == name).astype(np.int64)
    for name in _INTERNAL_SPLIT_FRACTIONS
  }
  for name, indices in split_indices.items():
    if not len(indices):
      raise ValueError(f"Internal split {name} is empty.")

  dev = split_indices["dev"]
  val = split_indices["val"]
  y_dev = y[dev]
  y_val = y[val]
  config_probe = ProbeConfig()
  x_dev_a = np.asarray(matrix_a[row_a[dev]], dtype=np.float64)
  x_val_a = np.asarray(matrix_a[row_a[val]], dtype=np.float64)
  probe_a = fit_logistic_probe(
    x_dev_a, y_dev, x_val_a, y_val, config=config_probe,
    feature_names=tuple(f"za_{index}" for index in range(matrix_a.shape[1])),
    model_name="M_A_ECG_round01_internal_resplit",
  )
  del x_dev_a, x_val_a

  x_dev_b = np.asarray(matrix_b[row_b[dev]], dtype=np.float64)
  x_val_b = np.asarray(matrix_b[row_b[val]], dtype=np.float64)
  probe_b = fit_logistic_probe(
    x_dev_b, y_dev, x_val_b, y_val, config=config_probe,
    feature_names=tuple(f"zb_{index}" for index in range(matrix_b.shape[1])),
    model_name="M_B_ECG_round01_internal_resplit",
  )
  del x_dev_b, x_val_b
  model_dir = private_run_dir / "refit_models"
  model_dir.mkdir(mode=0o700)
  probe_a.save(model_dir / "M_A_ECG.json")
  probe_b.save(model_dir / "M_B_ECG.json")
  (model_dir / "M_A_ECG.json").chmod(0o600)
  (model_dir / "M_B_ECG.json").chmod(0o600)

  p_a = np.empty(len(paired), dtype=np.float64)
  p_b = np.empty(len(paired), dtype=np.float64)
  for indices in split_indices.values():
    p_a[indices] = _predict_probe_rows(probe_a, matrix_a, row_a[indices])
    p_b[indices] = _predict_probe_rows(probe_b, matrix_b, row_b[indices])
  if not np.isfinite(p_a).all() or not np.isfinite(p_b).all():
    raise ValueError("Internal resplit probes failed to score all paired-valid patients.")

  raw_scores = pl.DataFrame({
    "subject_id": paired.get_column("subject_id"),
    "study_id": paired.get_column("study_id"),
    "split": split_labels,
    "mortality_30d": y,
    "p_a_ecg": p_a,
    "p_b_ecg": p_b,
  })
  _write_private_parquet(raw_scores, private_run_dir / "internal_resplit_scores.parquet")
  val_scores = raw_scores.filter(pl.col("split") == "val").sort(["subject_id", "study_id"])
  cal_fit = raw_scores.filter(pl.col("split") == "calibration_fit").sort(["subject_id", "study_id"])
  cal_check = raw_scores.filter(pl.col("split") == "calibration_check").sort(["subject_id", "study_id"])
  test = raw_scores.filter(pl.col("split") == "test").sort(["subject_id", "study_id"])

  pa_val = np.asarray(val_scores.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  pb_val = np.asarray(val_scores.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  y_val = _as_binary(val_scores.get_column("mortality_30d").to_numpy(), "validation outcomes")
  route_fit = fit_reference_cutoff(pa_val.tolist(), float(config["target_b_utilization"]), "val")
  tau_r = float(route_fit.tau_r)

  def apply_maps(frame: pl.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    pa = np.asarray(frame.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
    pb = np.asarray(frame.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
    route = pa >= tau_r
    return {
      "A_raw": pa,
      "A_global_calibrated": map_a.predict(pa),
      "B_all_raw": pb,
      "B_global_calibrated": map_b.predict(pb),
      "cascade_raw": np.where(route, pb, pa),
      "cascade_global_calibrated": global_map.predict(np.where(route, pb, pa)),
      "cascade_branch_calibrated": apply_routed_branch_risk_maps(
        pa, pb[route], route, low_map, high_map
      ),
    }, route

  cf_pa = np.asarray(cal_fit.get_column("p_a_ecg").to_numpy(), dtype=np.float64)
  cf_pb = np.asarray(cal_fit.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  cf_y = _as_binary(cal_fit.get_column("mortality_30d").to_numpy(), "calibration-fit outcomes")
  cf_route = cf_pa >= tau_r
  cf_raw = np.where(cf_route, cf_pb, cf_pa)
  map_a = fit_logistic_risk_map(cf_y, cf_pa, minimum_slope_events=50)
  map_b = fit_logistic_risk_map(cf_y, cf_pb, minimum_slope_events=50)
  global_map = fit_logistic_risk_map(cf_y, cf_raw, minimum_slope_events=50)
  low_map, high_map, support = fit_branch_risk_maps(
    cf_y, cf_pa, cf_pb, cf_route,
    minimum_slope_events=50,
    minimum_branch_events=10,
  )
  _write_private_json(private_run_dir / "calibration_maps.json", {
    "fit_split": "calibration_fit",
    "route_reference_split": "val",
    "tau_r": tau_r,
    "branch_support": support,
    "A_global": map_a.to_dict(),
    "B_global": map_b.to_dict(),
    "cascade_global": global_map.to_dict(),
    "low_a_branch": low_map.to_dict(),
    "routed_b_branch": high_map.to_dict(),
  })

  calibration_rows: list[dict[str, Any]] = []
  for split_name, frame in (("calibration_check", cal_check), ("test_internal_resplit", test)):
    score_methods, _route = apply_maps(frame)
    y_split = _as_binary(frame.get_column("mortality_30d").to_numpy(), f"{split_name} outcomes")
    for score_name, probability in score_methods.items():
      metrics, _bins = calibration_metrics(y_split, probability)
      calibration_rows.append({"split": split_name, "score_method": score_name, **metrics})
  calibration_frame = pl.DataFrame(calibration_rows)
  _write_private_csv(calibration_frame, private_run_dir / "internal_resplit_calibration.csv")
  _write_public_csv(calibration_frame, public_dir / "internal_resplit_calibration.csv")

  test_scores, route_test = apply_maps(test)
  y_test = _as_binary(test.get_column("mortality_30d").to_numpy(), "internal test outcomes")
  p_b_test = np.asarray(test.get_column("p_b_ecg").to_numpy(), dtype=np.float64)
  internal_frontier_rows: list[dict[str, Any]] = []
  selections: dict[str, np.ndarray] = {}
  for score_name in ("A_raw", "B_all_raw", "cascade_raw", "cascade_branch_calibrated"):
    if score_name == "A_raw":
      route = np.zeros(len(test), dtype=bool)
      reference_utilization = 0.0
    elif score_name == "B_all_raw":
      route = np.ones(len(test), dtype=bool)
      reference_utilization = 1.0
    else:
      route = route_test
      reference_utilization = float(np.mean(pa_val >= tau_r))
    row, selected = _capacity_metrics(
      policy=("numeric_A_risk_20pct" if score_name.startswith("cascade") else score_name),
      score_method=score_name,
      frame=test,
      route=route,
      score=np.asarray(test_scores[score_name], dtype=np.float64),
      full_b_probability=p_b_test,
      capacity=float(config["review_capacity"]),
      reference_utilization=reference_utilization,
    )
    row.update({"tau_r": tau_r if score_name.startswith("cascade") else None})
    internal_frontier_rows.append(row)
    selections[score_name] = selected
  internal_frontier = pl.DataFrame(internal_frontier_rows)
  internal_overlap = pl.DataFrame([{
    "comparison": "cascade_branch_calibrated_vs_raw",
    **_selected_overlap(selections["cascade_raw"], selections["cascade_branch_calibrated"]),
    "n_raw_events": internal_frontier.filter(pl.col("final_score_method") == "cascade_raw").item(0, "events_captured"),
    "n_branch_events": internal_frontier.filter(pl.col("final_score_method") == "cascade_branch_calibrated").item(0, "events_captured"),
  }])
  _write_private_csv(internal_frontier, private_run_dir / "internal_resplit_frontier.csv")
  _write_private_csv(internal_overlap, private_run_dir / "internal_resplit_score_overlap.csv")
  _write_public_csv(internal_frontier, public_dir / "internal_resplit_frontier.csv")
  _write_public_csv(internal_overlap, public_dir / "internal_resplit_score_overlap.csv")

  split_counts = {
    name: {
      "n": int(np.sum(split_labels == name)),
      "events": int(y[split_labels == name].sum()),
    }
    for name in _INTERNAL_SPLIT_FRACTIONS
  }
  model_payload = {
    "M_A_ECG": probe_a.to_dict(),
    "M_B_ECG": probe_b.to_dict(),
  }
  _write_private_json(private_run_dir / "refit_model_metadata.json", {
    "M_A_ECG": {"best_c": probe_a.best_c, "model_sha256": hashlib.sha256(json.dumps(model_payload["M_A_ECG"], sort_keys=True).encode()).hexdigest()},
    "M_B_ECG": {"best_c": probe_b.best_c, "model_sha256": hashlib.sha256(json.dumps(model_payload["M_B_ECG"], sort_keys=True).encode()).hexdigest()},
    "probe_config": {
      "c_grid": list(config_probe.regularization_c_grid),
      "max_iter": config_probe.max_iter,
      "solver": config_probe.solver,
      "standardize": config_probe.standardize,
    },
  })
  return {
    "analysis_status": "internal_resplit",
    "n_paired_valid": len(paired),
    "split_counts": split_counts,
    "route_threshold": tau_r,
    "reference_route_utilization": float(np.mean(pa_val >= tau_r)),
    "test_route_utilization": float(route_test.mean()),
    "calibration_fit_support": support,
    "calibration_rows": calibration_frame.height,
    "frontier_rows": internal_frontier.height,
    "selected_set_overlap_rows": internal_overlap.height,
    "probe_best_c": {"M_A_ECG": probe_a.best_c, "M_B_ECG": probe_b.best_c},
  }


def run_probability_cascade_audit(config_path: Path) -> dict[str, Any]:
  """Run the aggregate-only audit extension and patient-disjoint sensitivity."""
  config_path = config_path.resolve()
  config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
  if not isinstance(config, dict) or config.get("version") != 1:
    raise ValueError("Audit runner requires a version-1 YAML mapping.")
  run_id = str(config.get("run_id", ""))
  if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_id):
    raise ValueError("run_id must contain only letters, digits, dot, underscore, or hyphen.")
  output_root = Path(config.get("output_root", "artifacts/probability_cascade/audit_round01")).resolve()
  private_root = Path("artifacts/probability_cascade/audit_round01").resolve()
  if not output_root.is_relative_to(private_root):
    raise ValueError("Patient-level audit outputs must stay under artifacts/probability_cascade/audit_round01/.")
  public_dir = Path(config.get("public_output_root", "docs/probability_cascade/audit_round01")).resolve()
  if public_dir != Path("docs/probability_cascade/audit_round01").resolve():
    raise ValueError("Public audit outputs must use docs/probability_cascade/audit_round01/.")
  private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
  private_run_dir = output_root / run_id
  private_run_dir.mkdir(mode=0o700, exist_ok=False)
  public_dir.mkdir(parents=True, exist_ok=True)

  source_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
  ).stdout.strip()
  manifest: dict[str, Any] = {
    "status": "running",
    "run_id": run_id,
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": source_commit,
    "config_sha256": _sha256_file(config_path),
    "analysis_status": "post_hoc_exploratory_and_internal_resplit",
    "source_run_id": config.get("source_run_id"),
    "software": {"python": sys.version, "platform": platform.platform()},
    "private_output_directory": str(private_run_dir),
    "public_outputs": [],
  }
  _write_private_json(private_run_dir / "manifest.json", manifest)
  try:
    existing = _existing_run_outputs(config, private_run_dir, public_dir)
    resplit = _internal_resplit_outputs(config, private_run_dir, public_dir)
    output_names = [
      "branch_calibrated_frontier.csv",
      "branch_score_overlap.csv",
      "twenty_forty_bootstrap.csv",
      "alternative_routing.csv",
      "alternative_routing_overlap.csv",
      "alternative_routing_policies.json",
      "internal_resplit_calibration.csv",
      "internal_resplit_frontier.csv",
      "internal_resplit_score_overlap.csv",
    ]
    if any(not (public_dir / name).is_file() for name in output_names):
      raise FileNotFoundError("An expected aggregate audit table was not written.")
    inputs = config["inputs"]
    source_run_dir = Path(config["source_run_dir"])
    source_hashes = {
      "source_scores.parquet": _sha256_file(source_run_dir / "scores.parquet"),
      "source_run_manifest.json": _sha256_file(source_run_dir / "manifest.json"),
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
      "existing_run_analysis": existing,
      "internal_resplit_analysis": resplit,
      "source_artifact_sha256_by_filename": source_hashes,
      "internal_split": {
        "unit": "one earliest ECG per patient; paired-valid rows only",
        "seed": int(config["internal_split_seed"]),
        "fractions": _INTERNAL_SPLIT_FRACTIONS,
        "roles": ["dev", "val", "calibration_fit", "calibration_check", "test"],
      "assignment": "randomized by unique subject_id only; mortality labels are not used to assign split membership",
        "analysis_status": "internal_resplit",
      },
      "post_outcome_exploration": {
        "existing_test_used_for": [
          "branch-calibrated frontier and route-set overlap",
          "paired 20%-versus-40% bootstrap",
          "alternative routing policy assessment",
        ],
        "new_split_calibration_check_used_for_model_selection": False,
        "new_split_test_used_for_cutoff_or_calibration_fit": False,
        "scope": "The source cohort and outcomes were previously studied; the new patient-disjoint split is an internal resplit, not independent prospective or external confirmation.",
      },
      "public_outputs": output_names + ["manifest.json"],
      "privacy": "Only aggregate rows are public; patient-level scores, keys, split assignments, features, and refit models remain under the ignored private run directory.",
    })
    _write_private_json(private_run_dir / "manifest.json", manifest)
    public_manifest = {
      key: value for key, value in manifest.items()
      if key not in {"private_output_directory", "software"}
    }
    (public_dir / "manifest.json").write_text(
      json.dumps(public_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
  result = run_probability_cascade_audit(args.config)
  print(json.dumps({
    "status": result["status"],
    "run_id": result["run_id"],
    "existing_run_analysis": result["existing_run_analysis"],
    "internal_resplit_analysis": result["internal_resplit_analysis"],
    "public_outputs": result["public_outputs"],
  }, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
