"""Independent exam-ID keyed audit of saved CODE-15% feature alignment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
  match = re.search(r"checkpoint_part(\d+)\.npz$", path.name)
  return (int(match.group(1)), path.name) if match else (2**31 - 1, path.name)


def audit_code15_feature_alignment(
  features_dir: Path,
  cohort_path: Path,
) -> dict[str, Any]:
  """Compare partition checkpoint features to saved rows by exam ID.

  This independently verifies the alignment repair without relying on saved
  row order: each checkpoint exam ID is looked up in the extracted metadata,
  its A/B feature rows are compared exactly, and the metadata patient key is
  joined to the canonical index cohort by exam ID.
  """
  import polars as pl

  features_dir = Path(features_dir)
  checkpoint_paths = sorted(
    features_dir.glob("checkpoint_part*.npz"),
    key=_checkpoint_sort_key,
  )
  if not checkpoint_paths:
    raise FileNotFoundError(f"No partition checkpoints found in '{features_dir}'.")

  meta = pl.read_parquet(features_dir / "extracted_meta.parquet").select(
    ["exam_id", "patient_id"],
  ).sort("exam_id")
  saved_exam_ids = np.asarray(meta["exam_id"].to_numpy(), dtype=np.int64)
  saved_exam_id_duplicates = int(len(saved_exam_ids) - len(np.unique(saved_exam_ids)))
  features_a = np.load(features_dir / "ciis_features.npy", mmap_mode="r")
  features_b = np.load(features_dir / "dbeta_embeddings.npy", mmap_mode="r")
  if len(features_a) != len(saved_exam_ids) or len(features_b) != len(saved_exam_ids):
    raise ValueError(
      "Saved feature arrays and extracted metadata have different row counts: "
      f"A={len(features_a)}, B={len(features_b)}, meta={len(saved_exam_ids)}."
    )

  n_checkpoint_rows = 0
  n_checkpoint_duplicates = 0
  n_checkpoint_ids_missing_from_meta = 0
  n_a_exact_rows = 0
  n_b_exact_rows = 0
  seen_ids: set[int] = set()

  for checkpoint_path in checkpoint_paths:
    with np.load(checkpoint_path) as checkpoint:
      exam_ids = np.asarray(checkpoint["exam_ids"], dtype=np.int64).ravel()
      checkpoint_a = np.asarray(checkpoint["features_a"])
      checkpoint_b = np.asarray(checkpoint["features_b"])
      if len(exam_ids) != len(checkpoint_a) or len(exam_ids) != len(checkpoint_b):
        raise ValueError(f"Ragged exam/features arrays in '{checkpoint_path.name}'.")
      if checkpoint_a.shape[1:] != features_a.shape[1:] or checkpoint_b.shape[1:] != features_b.shape[1:]:
        raise ValueError(f"Feature dimensions differ in '{checkpoint_path.name}'.")

      for exam_id in exam_ids.tolist():
        exam_id_int = int(exam_id)
        if exam_id_int in seen_ids:
          n_checkpoint_duplicates += 1
        seen_ids.add(exam_id_int)

      positions = np.searchsorted(saved_exam_ids, exam_ids)
      in_bounds = positions < len(saved_exam_ids)
      matched = np.zeros(len(exam_ids), dtype=bool)
      if len(saved_exam_ids):
        matched[in_bounds] = saved_exam_ids[positions[in_bounds]] == exam_ids[in_bounds]
      n_checkpoint_ids_missing_from_meta += int((~matched).sum())
      if np.any(matched):
        saved_positions = positions[matched]
        n_a_exact_rows += int(
          np.all(features_a[saved_positions] == checkpoint_a[matched], axis=1).sum(),
        )
        n_b_exact_rows += int(
          np.all(features_b[saved_positions] == checkpoint_b[matched], axis=1).sum(),
        )
      n_checkpoint_rows += len(exam_ids)

  cohort = pl.read_parquet(cohort_path).select(
    ["exam_id", pl.col("patient_id").alias("cohort_patient_id")],
  )
  cohort_exam_id_duplicates = int(cohort.height - cohort["exam_id"].n_unique())
  n_unmatched_cohort_rows = len(meta)
  n_patient_id_mismatches = len(meta)
  if cohort_exam_id_duplicates == 0:
    joined = meta.join(cohort, on="exam_id", how="left", validate="1:1")
    n_unmatched_cohort_rows = int(joined["cohort_patient_id"].null_count())
    n_patient_id_mismatches = int(
      joined.filter(
        pl.col("cohort_patient_id").is_not_null()
        & (pl.col("patient_id") != pl.col("cohort_patient_id")),
      ).height,
    )

  n_meta_ids_missing_from_checkpoints = sum(
    int(exam_id) not in seen_ids for exam_id in saved_exam_ids.tolist()
  )
  n_unique_checkpoint_ids = len(seen_ids)
  status = "passed" if (
    saved_exam_id_duplicates == 0
    and cohort_exam_id_duplicates == 0
    and n_checkpoint_rows == len(saved_exam_ids)
    and n_checkpoint_duplicates == 0
    and n_checkpoint_ids_missing_from_meta == 0
    and n_meta_ids_missing_from_checkpoints == 0
    and n_a_exact_rows == len(saved_exam_ids)
    and n_b_exact_rows == len(saved_exam_ids)
    and n_unmatched_cohort_rows == 0
    and n_patient_id_mismatches == 0
  ) else "failed"

  return {
    "schema_version": "code15_alignment_audit_v1",
    "status": status,
    "join_contract": "checkpoint exam_id -> extracted_meta exam_id -> cohort exam_id/patient_id",
    "feature_comparison": "exact numeric row equality after exam_id lookup",
    "n_checkpoint_parts": len(checkpoint_paths),
    "n_checkpoint_rows": n_checkpoint_rows,
    "n_unique_checkpoint_exam_ids": n_unique_checkpoint_ids,
    "n_checkpoint_duplicate_exam_id_occurrences": n_checkpoint_duplicates,
    "n_extracted_meta_rows": len(saved_exam_ids),
    "n_extracted_meta_exam_id_duplicates": saved_exam_id_duplicates,
    "n_checkpoint_exam_ids_missing_from_meta": n_checkpoint_ids_missing_from_meta,
    "n_meta_exam_ids_missing_from_checkpoints": n_meta_ids_missing_from_checkpoints,
    "n_ciis_rows_exactly_matched": n_a_exact_rows,
    "n_dbeta_rows_exactly_matched": n_b_exact_rows,
    "n_cohort_exam_id_duplicates": cohort_exam_id_duplicates,
    "n_meta_rows_without_cohort_match": n_unmatched_cohort_rows,
    "n_patient_id_mismatches_after_exam_id_join": n_patient_id_mismatches,
  }
