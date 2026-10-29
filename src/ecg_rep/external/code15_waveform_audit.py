"""Post hoc audit of CODE-15% waveform characteristics and extraction QC.

The scan uses HDF5 exam IDs to characterize every index-cohort waveform.
Aggregate extraction-failure counts come from the existing per-exam manifest;
the locked-test comparison uses the unchanged frozen E3 probes and only
complete one-year labels. No exam or patient identifiers are returned.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

ACTIVE_ATOL = 1e-6
MIN_EVENTS = 11

DURATION_GROUPS = (
  "2934_active_samples",
  "4096_active_samples",
  "other_active_length",
  "no_active_region",
)
PADDING_GROUPS = (
  "581_each_side",
  "none",
  "symmetric_other",
  "asymmetric_or_other",
  "no_active_region",
)
JOINT_GROUPS = (
  "2934_active_with_581_each_side",
  "4096_active_unpadded",
  "other_active_and_padding",
  "no_active_region",
)


def classify_waveform_region(
  trace: npt.NDArray[np.floating],
  atol: float = ACTIVE_ATOL,
) -> dict[str, Any]:
  """Summarize active duration and padding using the extraction threshold."""
  active = np.any(np.abs(np.asarray(trace)) > atol, axis=1)
  positions = np.flatnonzero(active)
  if positions.size == 0:
    return {
      "active_samples": 0,
      "pad_start_samples": int(trace.shape[0]),
      "pad_end_samples": int(trace.shape[0]),
      "active_duration_seconds": 0.0,
      "duration_group": "no_active_region",
      "padding_group": "no_active_region",
      "joint_group": "no_active_region",
    }

  start = int(positions[0])
  end = int(positions[-1]) + 1
  active_samples = end - start
  pad_end = int(trace.shape[0] - end)
  if active_samples == 2934:
    duration_group = "2934_active_samples"
  elif active_samples == 4096:
    duration_group = "4096_active_samples"
  else:
    duration_group = "other_active_length"

  if start == 581 and pad_end == 581:
    padding_group = "581_each_side"
  elif start == 0 and pad_end == 0:
    padding_group = "none"
  elif start == pad_end:
    padding_group = "symmetric_other"
  else:
    padding_group = "asymmetric_or_other"

  if active_samples == 2934 and start == 581 and pad_end == 581:
    joint_group = "2934_active_with_581_each_side"
  elif active_samples == 4096 and start == 0 and pad_end == 0:
    joint_group = "4096_active_unpadded"
  else:
    joint_group = "other_active_and_padding"

  return {
    "active_samples": active_samples,
    "pad_start_samples": start,
    "pad_end_samples": pad_end,
    "active_duration_seconds": round(active_samples / 400.0, 4),
    "duration_group": duration_group,
    "padding_group": padding_group,
    "joint_group": joint_group,
  }


def classify_extraction_failure(error: str) -> str:
  """Collapse an extraction manifest error to a safe aggregate category."""
  if error.startswith("standardize_error:"):
    if "no detectable active region" in error:
      return "standardize:no_active_region"
    return "standardize:other"
  if error.startswith("qc_gate_failed:"):
    flags = set(error.partition(":")[2].split(","))
    flags.discard("active_length_uncertain")
    return "qc:" + "+".join(sorted(flags))
  if error.startswith("ciis_error:"):
    message = error.partition(":")[2]
    if "Insufficient R-peaks" in message:
      return "ciis:insufficient_r_peaks"
    if "No valid complete beats" in message:
      return "ciis:no_valid_complete_beats"
    return "ciis:other"
  if error.startswith("dbeta_"):
    return "dbeta:" + error.partition(":")[0]
  return "other"


def _natural_part_key(path: Path) -> tuple[int, str]:
  match = re.search(r"exams_part(\d+)\.hdf5$", path.name)
  return (int(match.group(1)), path.name) if match else (2**31 - 1, path.name)


def _resample_class_indices(y: npt.NDArray[np.int64], rng: np.random.Generator) -> npt.NDArray[np.int64]:
  positives = np.flatnonzero(y == 1)
  negatives = np.flatnonzero(y == 0)
  return np.concatenate(
    (
      rng.choice(positives, size=len(positives), replace=True),
      rng.choice(negatives, size=len(negatives), replace=True),
    ),
  )


def _group_delta_bootstrap(
  y: npt.NDArray[np.int64],
  p_a: npt.NDArray[np.float64],
  p_b: npt.NDArray[np.float64],
  indices: npt.NDArray[np.int64],
  n_bootstraps: int,
  seed: int,
) -> dict[str, Any]:
  from sklearn.metrics import roc_auc_score

  y_g = y[indices]
  if int(y_g.sum()) < MIN_EVENTS or int(len(y_g) - y_g.sum()) < MIN_EVENTS:
    return {
      "point_estimate": None,
      "ci_lower": None,
      "ci_upper": None,
      "n_bootstraps": n_bootstraps,
      "n_bootstraps_used": 0,
      "status": "suppressed_or_insufficient_events",
    }
  point = float(roc_auc_score(y_g, p_b[indices]) - roc_auc_score(y_g, p_a[indices]))
  rng = np.random.default_rng(seed)
  values = np.empty(n_bootstraps, dtype=np.float64)
  for replicate in range(n_bootstraps):
    sampled = _resample_class_indices(y_g, rng)
    original_indices = indices[sampled]
    values[replicate] = (
      roc_auc_score(y[original_indices], p_b[original_indices])
      - roc_auc_score(y[original_indices], p_a[original_indices])
    )
  return {
    "point_estimate": point,
    "ci_lower": float(np.percentile(values, 2.5)),
    "ci_upper": float(np.percentile(values, 97.5)),
    "n_bootstraps": n_bootstraps,
    "n_bootstraps_used": n_bootstraps,
    "status": "estimated",
  }


def _interaction_bootstrap(
  y: npt.NDArray[np.int64],
  p_a: npt.NDArray[np.float64],
  p_b: npt.NDArray[np.float64],
  left_indices: npt.NDArray[np.int64],
  right_indices: npt.NDArray[np.int64],
  n_bootstraps: int,
  seed: int,
) -> dict[str, Any]:
  from sklearn.metrics import roc_auc_score

  left_y, right_y = y[left_indices], y[right_indices]
  if min(
    int(left_y.sum()), int(len(left_y) - left_y.sum()),
    int(right_y.sum()), int(len(right_y) - right_y.sum()),
  ) < MIN_EVENTS:
    return {
      "contrast": "2934_with_581_each_side_minus_4096_unpadded_delta_auroc",
      "interaction_estimate": None,
      "ci_lower": None,
      "ci_upper": None,
      "n_bootstraps": n_bootstraps,
      "n_bootstraps_used": 0,
      "status": "suppressed_or_insufficient_events",
    }

  left_delta = float(
    roc_auc_score(left_y, p_b[left_indices]) - roc_auc_score(left_y, p_a[left_indices]),
  )
  right_delta = float(
    roc_auc_score(right_y, p_b[right_indices]) - roc_auc_score(right_y, p_a[right_indices]),
  )
  rng = np.random.default_rng(seed)
  values = np.empty(n_bootstraps, dtype=np.float64)
  for replicate in range(n_bootstraps):
    left_sample = left_indices[_resample_class_indices(left_y, rng)]
    right_sample = right_indices[_resample_class_indices(right_y, rng)]
    left_boot_delta = (
      roc_auc_score(y[left_sample], p_b[left_sample])
      - roc_auc_score(y[left_sample], p_a[left_sample])
    )
    right_boot_delta = (
      roc_auc_score(y[right_sample], p_b[right_sample])
      - roc_auc_score(y[right_sample], p_a[right_sample])
    )
    values[replicate] = left_boot_delta - right_boot_delta
  return {
    "contrast": "2934_with_581_each_side_minus_4096_unpadded_delta_auroc",
    "interaction_estimate": left_delta - right_delta,
    "ci_lower": float(np.percentile(values, 2.5)),
    "ci_upper": float(np.percentile(values, 97.5)),
    "n_bootstraps": n_bootstraps,
    "n_bootstraps_used": n_bootstraps,
    "status": "estimated",
  }


def _performance_by_group(
  y: npt.NDArray[np.int64],
  p_a: npt.NDArray[np.float64],
  p_b: npt.NDArray[np.float64],
  groups: npt.NDArray[np.str_],
  group_order: tuple[str, ...],
  n_bootstraps: int,
  seed: int,
) -> list[dict[str, Any]]:
  from sklearn.metrics import roc_auc_score

  rows: list[dict[str, Any]] = []
  for group_index, group in enumerate(group_order):
    indices = np.flatnonzero(groups == group)
    if len(indices) == 0:
      continue
    y_g = y[indices]
    events = int(y_g.sum())
    controls = int(len(y_g) - events)
    estimable = events >= MIN_EVENTS and controls >= MIN_EVENTS
    row: dict[str, Any] = {
      "group": group,
      "n_complete_case": int(len(indices)),
      "n_events": events if events >= MIN_EVENTS else "suppressed(<11)",
      "n_controls": controls if controls >= MIN_EVENTS else "suppressed(<11)",
      "auroc_a": float(roc_auc_score(y_g, p_a[indices])) if estimable else None,
      "auroc_b": float(roc_auc_score(y_g, p_b[indices])) if estimable else None,
    }
    row["delta_b_minus_a"] = _group_delta_bootstrap(
      y, p_a, p_b, indices, n_bootstraps, seed + group_index,
    )
    rows.append(row)
  return rows


def _distribution_by_group(
  traits_by_exam_id: dict[int, dict[str, Any]],
  group_field: str,
  group_order: tuple[str, ...],
  extracted_ids: set[int],
  failure_by_exam_id: dict[int, str],
) -> list[dict[str, Any]]:
  counts: dict[str, Counter[str]] = defaultdict(Counter)
  for exam_id, trait in traits_by_exam_id.items():
    group = str(trait[group_field])
    counts[group]["n_index_cohort"] += 1
    if exam_id in extracted_ids:
      counts[group]["n_extracted"] += 1
    else:
      counts[group]["n_extraction_failures"] += 1
      counts[group][f"failure:{failure_by_exam_id.get(exam_id, 'unclassified')}"] += 1

  rows: list[dict[str, Any]] = []
  for group in group_order:
    counter = counts[group]
    n_total = int(counter["n_index_cohort"])
    n_failed = int(counter["n_extraction_failures"])
    n_extracted = int(counter["n_extracted"])
    failure_categories = {
      key.removeprefix("failure:"): int(value)
      for key, value in sorted(counter.items())
      if key.startswith("failure:")
    }
    rows.append(
      {
        "group": group,
        "n_index_cohort": n_total,
        "n_extracted": n_extracted,
        "n_extraction_failures": n_failed,
        "extraction_failure_rate": (n_failed / n_total) if n_total else None,
        "failure_categories": failure_categories,
      },
    )
  return rows


def audit_code15_waveforms(
  data_dir: Path,
  features_dir: Path,
  cohort_path: Path,
  split_path: Path,
  model_dir: Path,
  baseline_manifest_path: Path,
  n_bootstraps: int = 1000,
  seed: int = 20261002,
  batch_size: int = 256,
  progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
  """Scan the index cohort waveforms and compare frozen E3 scores by traits."""
  import h5py
  import polars as pl

  from ecg_rep.external.code15_transport import load_frozen_probes, transport_scores
  from ecg_rep.external.code15_waveforms import resolve_hdf5_keys

  if n_bootstraps < 1:
    raise ValueError("At least one waveform-stratified bootstrap replicate is required.")
  if batch_size < 1:
    raise ValueError("Waveform scan batch size must be at least one.")
  data_dir = Path(data_dir)
  features_dir = Path(features_dir)
  cohort_path = Path(cohort_path)
  split_path = Path(split_path)
  baseline_manifest_path = Path(baseline_manifest_path)
  if not baseline_manifest_path.exists():
    raise FileNotFoundError(f"Frozen-probe baseline manifest not found: {baseline_manifest_path}")
  manifest = json.loads((features_dir / "extraction_manifest.json").read_text(encoding="utf-8"))

  cohort = pl.read_parquet(cohort_path).select(
    ["exam_id", "patient_id", "trace_file", "y_365d"],
  )
  if cohort["exam_id"].n_unique() != cohort.height:
    raise ValueError("Index cohort contains duplicate exam IDs.")
  cohort_ids = set(int(v) for v in cohort["exam_id"].to_list())
  target_ids_by_file: dict[str, set[int]] = defaultdict(set)
  expected_file_by_id: dict[int, str] = {}
  for exam_id, trace_file in cohort.select(["exam_id", "trace_file"]).iter_rows():
    exam_id_int = int(exam_id)
    trace_file_name = str(trace_file)
    target_ids_by_file[trace_file_name].add(exam_id_int)
    expected_file_by_id[exam_id_int] = trace_file_name

  meta = pl.read_parquet(features_dir / "extracted_meta.parquet").select(
    ["exam_id", "patient_id"],
  )
  extracted_ids = set(int(v) for v in meta["exam_id"].to_list())
  failure_by_exam_id: dict[int, str] = {}
  unparsed_failures = 0
  for manifest_key, error in manifest.get("failures", {}).items():
    match = re.search(r"part\d+/exam(\d+)$", str(manifest_key))
    if match is None:
      unparsed_failures += 1
      continue
    failure_by_exam_id[int(match.group(1))] = classify_extraction_failure(str(error))
  failed_ids = set(failure_by_exam_id)
  if extracted_ids & failed_ids:
    raise ValueError("Extraction metadata and failure manifest overlap by exam ID.")
  if extracted_ids | failed_ids != cohort_ids:
    raise ValueError(
      "Extracted metadata plus failure manifest do not cover the index cohort: "
      f"cohort={len(cohort_ids)}, extracted={len(extracted_ids)}, failures={len(failed_ids)}."
    )

  hdf5_paths = sorted(data_dir.glob("exams_part*.hdf5"), key=_natural_part_key)
  if not hdf5_paths:
    raise FileNotFoundError(f"No exams_part*.hdf5 files found in '{data_dir}'.")
  if set(target_ids_by_file) != {path.name for path in hdf5_paths if path.name in target_ids_by_file}:
    missing_files = sorted(set(target_ids_by_file) - {path.name for path in hdf5_paths})
    if missing_files:
      raise FileNotFoundError(f"Index cohort references {len(missing_files)} missing waveform partitions.")

  traits_by_exam_id: dict[int, dict[str, Any]] = {}
  seen_source_ids: set[int] = set()
  source_duplicate_ids: set[int] = set()
  n_source_duplicate_occurrences = 0
  seen_cohort_source_ids: set[int] = set()
  n_cohort_source_duplicate_occurrences = 0
  n_cohort_ids_in_noncanonical_partition = 0
  n_source_rows = 0
  for part_index, path in enumerate(hdf5_paths, start=1):
    expected_ids = target_ids_by_file.get(path.name, set())
    with h5py.File(path, "r") as handle:
      trace_key, exam_key = resolve_hdf5_keys(handle)
      ids_dataset = handle[exam_key]
      trace_dataset = handle[trace_key]
      if not isinstance(ids_dataset, h5py.Dataset):
        raise TypeError(f"'{exam_key}' is not an HDF5 dataset in '{path.name}'.")
      if not isinstance(trace_dataset, h5py.Dataset):
        raise TypeError(f"'{trace_key}' is not an HDF5 dataset in '{path.name}'.")
      exam_ids = np.asarray(ids_dataset[()], dtype=np.int64).ravel()
      if trace_dataset.shape[0] != len(exam_ids):
        raise ValueError(f"Waveform and exam-ID row counts differ in '{path.name}'.")
      if trace_dataset.ndim != 3 or trace_dataset.shape[1:] != (4096, 12):
        raise ValueError(f"Unexpected trace tensor shape in '{path.name}': {trace_dataset.shape}.")

      for exam_id in exam_ids.tolist():
        exam_id_int = int(exam_id)
        if exam_id_int in seen_source_ids:
          n_source_duplicate_occurrences += 1
          source_duplicate_ids.add(exam_id_int)
        seen_source_ids.add(exam_id_int)
        if exam_id_int in cohort_ids:
          if exam_id_int in seen_cohort_source_ids:
            n_cohort_source_duplicate_occurrences += 1
          seen_cohort_source_ids.add(exam_id_int)
          if path.name != expected_file_by_id[exam_id_int]:
            n_cohort_ids_in_noncanonical_partition += 1

      for start in range(0, len(exam_ids), batch_size):
        stop = min(start + batch_size, len(exam_ids))
        batch_ids = exam_ids[start:stop]
        offsets = [i for i, value in enumerate(batch_ids.tolist()) if int(value) in expected_ids]
        if not offsets:
          continue
        batch = np.asarray(trace_dataset[start:stop])
        selected = batch[np.asarray(offsets, dtype=np.int64)]
        active_mask = np.any(np.abs(selected) > ACTIVE_ATOL, axis=2)
        has_active = active_mask.any(axis=1)
        starts = active_mask.argmax(axis=1)
        ends = active_mask.shape[1] - np.argmax(active_mask[:, ::-1], axis=1)
        for j, offset in enumerate(offsets):
          exam_id = int(batch_ids[offset])
          if has_active[j]:
            active_start = int(starts[j])
            active_end = int(ends[j])
            active_len = active_end - active_start
            pad_end = int(selected.shape[1] - active_end)
            trait = classify_waveform_region(
              np.asarray(selected[j, active_start:active_end, :]),
            )
            # The active segment alone does not retain source padding; restore
            # the source-derived leading/trailing counts and classifications.
            trait["pad_start_samples"] = active_start
            trait["pad_end_samples"] = pad_end
            if active_start == 581 and pad_end == 581:
              trait["padding_group"] = "581_each_side"
            elif active_start == 0 and pad_end == 0:
              trait["padding_group"] = "none"
            elif active_start == pad_end:
              trait["padding_group"] = "symmetric_other"
            else:
              trait["padding_group"] = "asymmetric_or_other"
            if active_len == 2934 and active_start == 581 and pad_end == 581:
              trait["joint_group"] = "2934_active_with_581_each_side"
            elif active_len == 4096 and active_start == 0 and pad_end == 0:
              trait["joint_group"] = "4096_active_unpadded"
            else:
              trait["joint_group"] = "other_active_and_padding"
          else:
            trait = classify_waveform_region(selected[j])
          traits_by_exam_id[exam_id] = trait
      n_source_rows += len(exam_ids)
    if progress is not None:
      progress(
        f"scanned waveform partition {part_index}/{len(hdf5_paths)} "
        f"({len(traits_by_exam_id):,} index-cohort traces characterized)"
      )

  if set(traits_by_exam_id) != cohort_ids:
    raise ValueError(
      "HDF5 exam-ID scan did not match the full index cohort: "
      f"cohort={len(cohort_ids)}, scanned={len(traits_by_exam_id)}."
    )
  n_cohort_ids_missing_from_source = len(cohort_ids - seen_cohort_source_ids)
  if (
    n_cohort_source_duplicate_occurrences
    or n_cohort_ids_in_noncanonical_partition
    or n_cohort_ids_missing_from_source
  ):
    raise ValueError(
      "Index cohort HDF5 identity check failed: "
      f"duplicates={n_cohort_source_duplicate_occurrences}, "
      f"wrong_partition={n_cohort_ids_in_noncanonical_partition}, "
      f"missing={n_cohort_ids_missing_from_source}."
    )

  duration_distribution = _distribution_by_group(
    traits_by_exam_id, "duration_group", DURATION_GROUPS, extracted_ids, failure_by_exam_id,
  )
  padding_distribution = _distribution_by_group(
    traits_by_exam_id, "padding_group", PADDING_GROUPS, extracted_ids, failure_by_exam_id,
  )
  joint_distribution = _distribution_by_group(
    traits_by_exam_id, "joint_group", JOINT_GROUPS, extracted_ids, failure_by_exam_id,
  )

  failure_counts = Counter(failure_by_exam_id.values())
  if unparsed_failures:
    failure_counts["unparsed_manifest_failure"] += unparsed_failures

  # Join labels and patient-level splits without changing the saved feature
  # row association verified by the independent D1 alignment audit.
  indexed_meta = meta.with_row_index("_feature_row")
  split = pl.read_csv(split_path).select(["patient_id", "split"])
  eval_frame = (
    indexed_meta.join(cohort.select(["exam_id", "patient_id", "y_365d"]),
                      on=["exam_id", "patient_id"], how="left", validate="1:1")
    .join(split, on="patient_id", how="left", validate="m:1")
    .sort("_feature_row")
  )
  if eval_frame["y_365d"].null_count() or eval_frame["split"].null_count():
    raise ValueError("Some extracted feature rows lack a cohort label or patient split.")

  extracted_test = eval_frame.filter(pl.col("split") == "test")
  complete_test = extracted_test.filter(pl.col("y_365d").is_in(["event", "nonevent"]))
  row_positions = np.asarray(complete_test["_feature_row"].to_numpy(), dtype=np.int64)
  y = np.asarray([1 if label == "event" else 0 for label in complete_test["y_365d"].to_list()],
                 dtype=np.int64)
  exam_ids_test = [int(value) for value in complete_test["exam_id"].to_list()]
  duration_groups = np.asarray(
    [traits_by_exam_id[exam_id]["duration_group"] for exam_id in exam_ids_test], dtype=str,
  )
  padding_groups = np.asarray(
    [traits_by_exam_id[exam_id]["padding_group"] for exam_id in exam_ids_test], dtype=str,
  )
  joint_groups = np.asarray(
    [traits_by_exam_id[exam_id]["joint_group"] for exam_id in exam_ids_test], dtype=str,
  )

  features_a = np.load(features_dir / "ciis_features.npy", mmap_mode="r")
  features_b = np.load(features_dir / "dbeta_embeddings.npy", mmap_mode="r")
  if max(row_positions, default=-1) >= len(features_a) or max(row_positions, default=-1) >= len(features_b):
    raise ValueError("Locked-test feature row index exceeds the saved feature arrays.")
  probes = load_frozen_probes(Path(model_dir), Path(baseline_manifest_path))
  p_a, p_b = transport_scores(probes, features_a[row_positions], features_b[row_positions])
  p_a = np.asarray(p_a, dtype=np.float64)
  p_b = np.asarray(p_b, dtype=np.float64)
  from sklearn.metrics import roc_auc_score

  if y.min() == y.max():
    raise ValueError("Locked-test complete-case waveform audit is single-class.")
  overall = {
    "n_locked_test_complete_case": int(len(y)),
    "n_events": int(y.sum()),
    "auroc_a": float(roc_auc_score(y, p_a)),
    "auroc_b": float(roc_auc_score(y, p_b)),
    "delta_b_minus_a": float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
  }
  duration_performance = _performance_by_group(
    y, p_a, p_b, duration_groups, DURATION_GROUPS, n_bootstraps, seed,
  )
  padding_performance = _performance_by_group(
    y, p_a, p_b, padding_groups, PADDING_GROUPS, n_bootstraps, seed + 100,
  )
  joint_performance = _performance_by_group(
    y, p_a, p_b, joint_groups, JOINT_GROUPS, n_bootstraps, seed + 200,
  )
  interaction = _interaction_bootstrap(
    y, p_a, p_b,
    np.flatnonzero(joint_groups == "2934_active_with_581_each_side"),
    np.flatnonzero(joint_groups == "4096_active_unpadded"),
    n_bootstraps,
    seed + 300,
  )

  # Keep one-year censoring visible by waveform group. The performance
  # estimates above remain the existing complete-case E3 binary estimand.
  test_labels = extracted_test["y_365d"].to_list()
  test_exam_ids = [int(value) for value in extracted_test["exam_id"].to_list()]
  censoring_by_joint_group: dict[str, dict[str, int]] = {}
  for group in JOINT_GROUPS:
    group_pairs = [
      (exam_id, label) for exam_id, label in zip(test_exam_ids, test_labels, strict=True)
      if traits_by_exam_id[exam_id]["joint_group"] == group
    ]
    censoring_by_joint_group[group] = {
      "n_locked_test_extracted": len(group_pairs),
      "n_censored_before_365d": sum(label == "unknown" for _, label in group_pairs),
      "n_complete_case": sum(label in ("event", "nonevent") for _, label in group_pairs),
    }

  return {
    "schema_version": "code15_waveform_audit_v1",
    "status": "passed",
    "scan_contract": "HDF5 exam_id -> cohort trace_file/exam_id; active region threshold |amplitude| > 1e-6",
    "evaluation_contract": (
      "post hoc exploratory complete-case 365d locked-test comparison of unchanged frozen MIMIC E3 probes; "
      "1000 paired, class-stratified bootstrap replicates by group; test outcomes were previously exposed"
    ),
    "interpretation_limit": (
      "The dominant waveform groups couple active duration and padding pattern (2934 samples with 581+581 "
      "padding versus 4096 samples without padding); subgroup contrasts cannot isolate either attribute and "
      "are not causal. QC-failed exams have no E3 scores, so their failure reasons are summarized as extraction "
      "attrition and discrimination is estimated only among successfully extracted exams."
    ),
    "scan_counts": {
      "n_index_cohort": len(cohort_ids),
      "n_extracted_feature_rows": len(extracted_ids),
      "n_manifest_extraction_failures": len(failed_ids),
      "n_source_hdf5_rows_scanned": n_source_rows,
      "n_index_cohort_traces_characterized": len(traits_by_exam_id),
      "n_source_unique_duplicate_exam_ids": len(source_duplicate_ids),
      "n_source_exam_id_duplicate_occurrences": n_source_duplicate_occurrences,
      "n_index_cohort_exam_id_duplicate_occurrences": n_cohort_source_duplicate_occurrences,
      "n_index_cohort_ids_in_noncanonical_partition": n_cohort_ids_in_noncanonical_partition,
      "n_index_cohort_ids_missing_from_source": n_cohort_ids_missing_from_source,
    },
    "failure_categories": {category: int(count) for category, count in sorted(failure_counts.items())},
    "distribution_by_active_duration": duration_distribution,
    "distribution_by_padding_pattern": padding_distribution,
    "distribution_by_joint_waveform_pattern": joint_distribution,
    "locked_test_complete_case_overall": overall,
    "locked_test_performance_by_active_duration": duration_performance,
    "locked_test_performance_by_padding_pattern": padding_performance,
    "locked_test_performance_by_joint_waveform_pattern": joint_performance,
    "locked_test_duration_padding_interaction": interaction,
    "locked_test_followup_by_joint_waveform_pattern": censoring_by_joint_group,
  }
