"""Shared CIIS-feature and frozen D-BETA representation extraction.

Both paths consume the same index ECG waveform identifiers. Outcome labels never
enter preprocessing. Private embeddings and waveforms stay outside version control.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import polars as pl

from ecg_rep.data import (
  DEFAULT_SAMPLING_RATE_HZ,
  DEFAULT_SIGNAL_LENGTH_SAMPLES,
  WaveformEligibilityCriteria,
  read_ecg_header,
  read_ecg_waveform,
  validate_waveform_eligibility,
)
from ecg_rep.scoring.dbeta import (
  DEFAULT_DBETA_EMBEDDING_DIM,
  DEFAULT_DBETA_MODEL_NAME,
  DEFAULT_DBETA_REVISION,
  DbetaAdapter,
  DbetaConfig,
  load_dbeta_model,
)
from ecg_rep.scoring.traditional import (
  CIIS_FEATURE_NAMES,
  measurements_to_feature_vector,
  score_ecg_waveform,
)

from collections.abc import Sequence

REPRESENTATION_A_NAME = "ciis_features"
REPRESENTATION_B_NAME = "dbeta_embedding"
REPRESENTATION_A_DIM = len(CIIS_FEATURE_NAMES)
REPRESENTATION_B_DIM = DEFAULT_DBETA_EMBEDDING_DIM

CIIS_EXTRACTOR_VERSION: str = "ciis_morphology_v2.1_pr_jpoint_tend_25uv"
CIIS_SCHEMA_VERSION: str = "ciis_15_continuous_measurements_v1"


def compute_feature_schema_hash(feature_names: Sequence[str]) -> str:
  """Cryptographic SHA-256 fingerprint of the ordered feature column names."""
  payload = json.dumps(list(feature_names), sort_keys=True).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


CIIS_FEATURE_SCHEMA_HASH: str = compute_feature_schema_hash(CIIS_FEATURE_NAMES)


def get_git_commit_hash() -> str:
  """Retrieve current git commit SHA if available, or 'unknown'."""
  try:
    import subprocess
    res = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      capture_output=True,
      text=True,
      timeout=2,
      check=False,
    )
    if res.returncode == 0 and res.stdout.strip():
      return res.stdout.strip()
  except Exception:
    pass
  return "unknown"


@dataclass(frozen=True)
class RepresentationProvenance:
  """Inspectable identity for a cached representation artifact."""

  representation: str
  dimension: int
  code_version: str
  model_name: str | None
  model_revision: str | None
  frozen_encoder: bool
  preprocess: dict[str, Any]
  cohort_hash: str
  n_records: int
  n_valid: int
  hardware: str
  elapsed_seconds: float
  extractor_version: str | None = None
  feature_schema_hash: str | None = None
  git_commit: str | None = None


def cohort_identity_hash(cohort_df: pl.DataFrame) -> str:
  """Stable hash over ordered subject_id/study_id pairs for cache keys."""
  payload = b"".join(
    f"{row['subject_id']}:{row['study_id']}\n".encode("utf-8")
    for row in cohort_df.select(["subject_id", "study_id"]).iter_rows(named=True)
  )
  return hashlib.sha256(payload).hexdigest()


def _resolve_record_path(
  relative_path: str,
  ecg_data_dir: Path | None,
) -> Path:
  return ecg_data_dir / relative_path if ecg_data_dir is not None else Path(relative_path)


def _process_single_ciis_record(
  item: tuple[int, str, bool],
) -> tuple[int, bool, npt.NDArray[np.float64] | None, float | None, str | None, str | None]:
  idx, rec_path_str, enforce_eligibility = item
  criteria = WaveformEligibilityCriteria()
  rec_path = Path(rec_path_str)
  try:
    if enforce_eligibility:
      eligible, reason = validate_waveform_eligibility(read_ecg_header(rec_path), criteria)
      if not eligible:
        raise ValueError(reason or "Waveform failed eligibility checks")
    signal_array, lead_names, fs = read_ecg_waveform(rec_path)
    result = score_ecg_waveform(signal_array, lead_names, fs)
    if not result.is_valid or result.measurements is None:
      raise ValueError(result.error_message or "CIIS extraction failed")
    feat = measurements_to_feature_vector(result.measurements)
    cat = result.category.value if result.category is not None else None
    return idx, True, feat, result.total_score, cat, None
  except Exception as exc:  # noqa: BLE001
    return idx, False, None, None, None, str(exc)


def extract_ciis_feature_cohort(
  cohort_df: pl.DataFrame,
  ecg_data_dir: Path | str | None = None,
  relative_path_col: str = "path",
  enforce_waveform_eligibility: bool = True,
  progress_every: int = 1000,
  checkpoint_path: Path | str | None = None,
  checkpoint_interval: int = 5000,
  num_workers: int = 1,
) -> tuple[pl.DataFrame, npt.NDArray[np.float64], RepresentationProvenance]:
  """Extract CIIS-derived feature vectors ``Z_A`` for each index ECG.

  Returns metadata (validity/errors/canonical score), a ``(n, 15)`` feature matrix,
  and provenance for caching. Optional ``checkpoint_path`` enables resume.
  """
  data_dir = Path(ecg_data_dir) if ecg_data_dir is not None else None
  criteria = WaveformEligibilityCriteria()
  n = len(cohort_df)
  features = np.full((n, REPRESENTATION_A_DIM), np.nan, dtype=np.float64)
  subject_ids = [int(v) for v in cohort_df["subject_id"].to_list()]
  study_ids = [int(v) for v in cohort_df["study_id"].to_list()]
  scores: list[float | None] = [None] * n
  categories: list[str | None] = [None] * n
  valid_flags: list[bool] = [False] * n
  errors: list[str | None] = [None] * n
  started = time.perf_counter()
  cohort_hash = cohort_identity_hash(cohort_df)
  start_idx = 0

  ckpt_p = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
  if ckpt_p is not None and ckpt_p.exists():
    with np.load(ckpt_p, allow_pickle=True) as ckpt:
      ckpt_cohort_hash = str(ckpt.get("cohort_hash", ""))
      ckpt_schema_hash = str(ckpt.get("feature_schema_hash", ""))
      ckpt_extractor_ver = str(ckpt.get("extractor_version", ""))
      ckpt_rep_ver = str(ckpt.get("representation_version", ""))
      ckpt_features = ckpt.get("features")
      ckpt_dim = (
        ckpt_features.shape[1]
        if ckpt_features is not None and ckpt_features.ndim == 2
        else 0
      )

      is_valid_checkpoint = (
        ckpt_cohort_hash == cohort_hash
        and ckpt_schema_hash == CIIS_FEATURE_SCHEMA_HASH
        and ckpt_extractor_ver == CIIS_EXTRACTOR_VERSION
        and ckpt_rep_ver == REPRESENTATION_A_NAME
        and ckpt_dim == REPRESENTATION_A_DIM
      )

      if is_valid_checkpoint:
        saved = int(ckpt["processed_count"])
        features[:saved] = ckpt["features"][:saved]
        scores[:saved] = ckpt["scores"].tolist()[:saved]
        categories[:saved] = ckpt["categories"].tolist()[:saved]
        valid_flags[:saved] = ckpt["valid_flags"].tolist()[:saved]
        errors[:saved] = ckpt["errors"].tolist()[:saved]
        start_idx = saved
        print(
          f"[CIIS] Resumed validated checkpoint at {start_idx}/{n} (extractor: {CIIS_EXTRACTOR_VERSION})",
          flush=True,
        )
      else:
        mismatches: list[str] = []
        if ckpt_cohort_hash != cohort_hash:
          mismatches.append(f"cohort_hash ({ckpt_cohort_hash[:8]} vs {cohort_hash[:8]})")
        if ckpt_schema_hash != CIIS_FEATURE_SCHEMA_HASH:
          mismatches.append(f"schema_hash ({ckpt_schema_hash[:8]} vs {CIIS_FEATURE_SCHEMA_HASH[:8]})")
        if ckpt_extractor_ver != CIIS_EXTRACTOR_VERSION:
          mismatches.append(f"extractor_version ({ckpt_extractor_ver} vs {CIIS_EXTRACTOR_VERSION})")
        if ckpt_rep_ver != REPRESENTATION_A_NAME:
          mismatches.append(f"representation ({ckpt_rep_ver} vs {REPRESENTATION_A_NAME})")
        if ckpt_dim != REPRESENTATION_A_DIM:
          mismatches.append(f"dimension ({ckpt_dim} vs {REPRESENTATION_A_DIM})")
        print(
          f"[CIIS] Checkpoint provenance mismatch ({', '.join(mismatches)}); invalidating stale checkpoint and restarting from 0.",
          flush=True,
        )

  rows = list(cohort_df.iter_rows(named=True))
  if num_workers > 1 and start_idx < n:
    from concurrent.futures import ProcessPoolExecutor

    chunk_size = checkpoint_interval if checkpoint_interval > 0 else 5000
    for chunk_start in range(start_idx, n, chunk_size):
      chunk_end = min(chunk_start + chunk_size, n)
      chunk_items = [
        (
          i,
          str(_resolve_record_path(str(rows[i].get(relative_path_col, "")), data_dir)),
          enforce_waveform_eligibility,
        )
        for i in range(chunk_start, chunk_end)
      ]
      with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, valid_ok, feat_vec, score_val, cat_val, err_msg in executor.map(
          _process_single_ciis_record, chunk_items, chunksize=32
        ):
          if valid_ok and feat_vec is not None:
            features[idx] = feat_vec
            scores[idx] = score_val
            categories[idx] = cat_val
            valid_flags[idx] = True
            errors[idx] = None
          else:
            scores[idx] = None
            categories[idx] = None
            valid_flags[idx] = False
            errors[idx] = err_msg

      done = chunk_end
      elapsed = time.perf_counter() - started
      rate = (done - start_idx) / elapsed if elapsed > 0 else 0.0
      print(f"[CIIS] {done}/{n} ({100.0 * done / n:.1f}%) {rate:.1f} rec/s", flush=True)
      if ckpt_p is not None and checkpoint_interval > 0:
        ckpt_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
          ckpt_p,
          cohort_hash=cohort_hash,
          processed_count=done,
          features=features,
          scores=np.asarray(scores, dtype=object),
          categories=np.asarray(categories, dtype=object),
          valid_flags=np.asarray(valid_flags, dtype=bool),
          errors=np.asarray(errors, dtype=object),
          feature_names=np.asarray(CIIS_FEATURE_NAMES, dtype=object),
          feature_schema_hash=CIIS_FEATURE_SCHEMA_HASH,
          representation_version=REPRESENTATION_A_NAME,
          extractor_version=CIIS_EXTRACTOR_VERSION,
          git_commit=get_git_commit_hash(),
        )
  else:
    for idx in range(start_idx, n):
      row = rows[idx]
      rec_path = _resolve_record_path(str(row.get(relative_path_col, "")), data_dir)
      try:
        if enforce_waveform_eligibility:
          eligible, reason = validate_waveform_eligibility(read_ecg_header(rec_path), criteria)
          if not eligible:
            raise ValueError(reason or "Waveform failed eligibility checks")
        signal_array, lead_names, fs = read_ecg_waveform(rec_path)
        result = score_ecg_waveform(signal_array, lead_names, fs)
        if not result.is_valid or result.measurements is None:
          raise ValueError(result.error_message or "CIIS extraction failed")
        features[idx] = measurements_to_feature_vector(result.measurements)
        scores[idx] = result.total_score
        categories[idx] = result.category.value if result.category is not None else None
        valid_flags[idx] = True
        errors[idx] = None
      except Exception as exc:  # noqa: BLE001 - per-record extraction must not abort cohort
        scores[idx] = None
        categories[idx] = None
        valid_flags[idx] = False
        errors[idx] = str(exc)

      done = idx + 1
      if progress_every > 0 and done % progress_every == 0:
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"[CIIS] {done}/{n} ({100.0 * done / n:.1f}%) {rate:.1f} rec/s", flush=True)
      if ckpt_p is not None and checkpoint_interval > 0 and done % checkpoint_interval == 0:
        ckpt_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
          ckpt_p,
          cohort_hash=cohort_hash,
          processed_count=done,
          features=features,
          scores=np.asarray(scores, dtype=object),
          categories=np.asarray(categories, dtype=object),
          valid_flags=np.asarray(valid_flags, dtype=bool),
          errors=np.asarray(errors, dtype=object),
          feature_names=np.asarray(CIIS_FEATURE_NAMES, dtype=object),
          feature_schema_hash=CIIS_FEATURE_SCHEMA_HASH,
          representation_version=REPRESENTATION_A_NAME,
          extractor_version=CIIS_EXTRACTOR_VERSION,
          git_commit=get_git_commit_hash(),
        )

  if ckpt_p is not None and checkpoint_interval > 0 and n > 0:
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      ckpt_p,
      cohort_hash=cohort_hash,
      processed_count=n,
      features=features,
      scores=np.asarray(scores, dtype=object),
      categories=np.asarray(categories, dtype=object),
      valid_flags=np.asarray(valid_flags, dtype=bool),
      errors=np.asarray(errors, dtype=object),
      feature_names=np.asarray(CIIS_FEATURE_NAMES, dtype=object),
      feature_schema_hash=CIIS_FEATURE_SCHEMA_HASH,
      representation_version=REPRESENTATION_A_NAME,
      extractor_version=CIIS_EXTRACTOR_VERSION,
      git_commit=get_git_commit_hash(),
    )

  meta = pl.DataFrame(
    {
      "subject_id": subject_ids,
      "study_id": study_ids,
      "ciis_total_score": scores,
      "ciis_category": categories,
      "representation_a_valid": valid_flags,
      "representation_a_error": errors,
    },
    schema={
      "subject_id": pl.Int64,
      "study_id": pl.Int64,
      "ciis_total_score": pl.Float64,
      "ciis_category": pl.String,
      "representation_a_valid": pl.Boolean,
      "representation_a_error": pl.String,
    },
  )
  elapsed = time.perf_counter() - started
  provenance = RepresentationProvenance(
    representation=REPRESENTATION_A_NAME,
    dimension=REPRESENTATION_A_DIM,
    code_version="ecg_rep.scoring.traditional",
    model_name=None,
    model_revision=None,
    frozen_encoder=True,
    preprocess={
      "waveform_eligibility": asdict(criteria) if enforce_waveform_eligibility else None,
      "feature_names": list(CIIS_FEATURE_NAMES),
    },
    cohort_hash=cohort_hash,
    n_records=n,
    n_valid=int(sum(valid_flags)),
    hardware="cpu",
    elapsed_seconds=elapsed,
    extractor_version=CIIS_EXTRACTOR_VERSION,
    feature_schema_hash=CIIS_FEATURE_SCHEMA_HASH,
    git_commit=get_git_commit_hash(),
  )
  return meta, features, provenance


def extract_dbeta_embedding_cohort(
  cohort_df: pl.DataFrame,
  adapter: DbetaAdapter | None = None,
  ecg_data_dir: Path | str | None = None,
  relative_path_col: str = "path",
  batch_size: int = 32,
  device: str = "cpu",
  enforce_waveform_eligibility: bool = True,
  progress_every: int = 1000,
  checkpoint_path: Path | str | None = None,
  checkpoint_interval: int = 5000,
  random_weights: bool = False,
  random_seed: int = 42,
) -> tuple[pl.DataFrame, npt.NDArray[np.float64], RepresentationProvenance]:
  """Extract frozen D-BETA embeddings ``Z_B`` for each index ECG.

  Optional ``checkpoint_path`` enables resume across interrupted embedding runs.
  """
  data_dir = Path(ecg_data_dir) if ecg_data_dir is not None else None
  criteria = WaveformEligibilityCriteria()
  if adapter is None:
    config = DbetaConfig(
      device=device,
      batch_size=batch_size,
      random_weights=random_weights,
      random_seed=random_seed,
    )
    model = load_dbeta_model(
      model_name=config.model_name,
      revision=config.model_version or DEFAULT_DBETA_REVISION,
      device=config.device,
      trust_remote_code=config.trust_remote_code,
      local_checkpoint_path=config.local_checkpoint_path,
      random_weights=random_weights,
      seed=random_seed,
    )
    active_adapter: DbetaAdapter = DbetaAdapter(config=config, model=model)
  else:
    active_adapter = adapter

  n = len(cohort_df)
  embeddings = np.full((n, active_adapter.config.embedding_dim), np.nan, dtype=np.float64)
  valid_flags: list[bool] = [False] * n
  errors: list[str | None] = [None] * n
  subject_ids = [int(v) for v in cohort_df["subject_id"].to_list()]
  study_ids = [int(v) for v in cohort_df["study_id"].to_list()]
  started = time.perf_counter()
  cohort_hash = cohort_identity_hash(cohort_df)
  start_idx = 0
  prior_elapsed = 0.0

  is_random = bool(random_weights or (getattr(active_adapter.config, "random_weights", False) is True))
  rep_name = f"dbeta_random_seed{random_seed}" if is_random else REPRESENTATION_B_NAME
  expected_emb_dim = int(active_adapter.config.embedding_dim)

  ckpt_p = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
  if ckpt_p is not None and ckpt_p.exists():
    with np.load(ckpt_p, allow_pickle=True) as ckpt:
      ckpt_cohort_hash = str(ckpt.get("cohort_hash", ""))
      ckpt_model_name = str(ckpt.get("model_name", ""))
      ckpt_model_rev = str(ckpt.get("model_revision", ""))
      ckpt_is_random = bool(ckpt.get("random_weights", False))
      ckpt_seed = int(ckpt.get("random_seed", 0)) if ckpt_is_random else None
      ckpt_emb = ckpt.get("embeddings")
      ckpt_dim = ckpt_emb.shape[1] if ckpt_emb is not None and ckpt_emb.ndim == 2 else 0

      is_valid_dbeta_ckpt = (
        ckpt_cohort_hash == cohort_hash
        and (ckpt_model_name == DEFAULT_DBETA_MODEL_NAME if not is_random else bool(ckpt_model_name))
        and (ckpt_model_rev == DEFAULT_DBETA_REVISION if not is_random else True)
        and ckpt_is_random == is_random
        and (not is_random or ckpt_seed == random_seed)
        and ckpt_dim == expected_emb_dim
      )

      if is_valid_dbeta_ckpt:
        saved = int(ckpt["processed_count"])
        embeddings[:saved] = ckpt["embeddings"][:saved]
        valid_flags[:saved] = ckpt["valid_flags"].tolist()[:saved]
        errors[:saved] = ckpt["errors"].tolist()[:saved]
        prior_elapsed = float(ckpt.get("elapsed_seconds", 0.0))
        start_idx = saved
        print(f"[D-BETA] Resumed validated checkpoint at {start_idx}/{n}", flush=True)
      else:
        print("[D-BETA] Checkpoint provenance mismatch; invalidating stale checkpoint and restarting from 0.", flush=True)

  if start_idx >= n:
    print(f"[D-BETA] Checkpoint already complete ({n} records)", flush=True)
  processed = start_idx
  batch_records: list[tuple[npt.NDArray[np.float64], list[str], int]] = []
  batch_indices: list[int] = []

  def _write_checkpoint(done: int) -> None:
    if ckpt_p is None or checkpoint_interval <= 0:
      return
    if done % checkpoint_interval != 0 and done != n:
      return
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    elapsed_so_far = prior_elapsed + (time.perf_counter() - started)
    np.savez_compressed(
      ckpt_p,
      cohort_hash=cohort_hash,
      processed_count=done,
      embeddings=embeddings,
      valid_flags=np.asarray(valid_flags, dtype=bool),
      errors=np.asarray(errors, dtype=object),
      elapsed_seconds=elapsed_so_far,
      model_name=DEFAULT_DBETA_MODEL_NAME,
      model_revision=DEFAULT_DBETA_REVISION,
      embedding_dim=expected_emb_dim,
      random_weights=is_random,
      random_seed=int(random_seed) if is_random else 0,
      git_commit=get_git_commit_hash(),
    )

  def flush_batch() -> None:
    nonlocal processed
    if not batch_records:
      return
    batch_result = active_adapter.embed_batch(batch_records)
    batch_embeddings = batch_result.embeddings
    if batch_embeddings is None:
      raise RuntimeError("D-BETA batch result missing embeddings array")
    for local_i, global_i in enumerate(batch_indices):
      if batch_result.valid_mask[local_i]:
        embeddings[global_i] = batch_embeddings[local_i]
        valid_flags[global_i] = True
        errors[global_i] = None
      else:
        valid_flags[global_i] = False
        reason = (
          batch_result.failure_reasons[local_i]
          if batch_result.failure_reasons
          else "D-BETA embed failed"
        )
        errors[global_i] = reason
    processed += len(batch_indices)
    if progress_every > 0 and (
      processed % progress_every < batch_size or processed == n
    ):
      elapsed = prior_elapsed + (time.perf_counter() - started)
      rate = processed / elapsed if elapsed > 0 else 0.0
      print(
        f"[D-BETA] {processed}/{n} ({100.0 * processed / n:.1f}%) {rate:.1f} rec/s",
        flush=True,
      )
    _write_checkpoint(processed)
    batch_records.clear()
    batch_indices.clear()

  rows = list(cohort_df.iter_rows(named=True))
  for idx in range(start_idx, n):
    row = rows[idx]
    rec_path = _resolve_record_path(str(row.get(relative_path_col, "")), data_dir)
    try:
      if enforce_waveform_eligibility:
        eligible, reason = validate_waveform_eligibility(read_ecg_header(rec_path), criteria)
        if not eligible:
          raise ValueError(reason or "Waveform failed eligibility checks")
      signal_array, lead_names, fs = read_ecg_waveform(rec_path)
      batch_records.append((signal_array, list(lead_names), int(fs)))
      batch_indices.append(idx)
      if len(batch_records) >= batch_size:
        flush_batch()
    except Exception as exc:  # noqa: BLE001
      valid_flags[idx] = False
      errors[idx] = str(exc)
      processed += 1
      if progress_every > 0 and processed % progress_every == 0:
        elapsed = prior_elapsed + (time.perf_counter() - started)
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(
          f"[D-BETA] {processed}/{n} ({100.0 * processed / n:.1f}%) {rate:.1f} rec/s",
          flush=True,
        )
      _write_checkpoint(processed)

  flush_batch()
  if ckpt_p is not None and checkpoint_interval > 0:
    # Force a final checkpoint even if n is not a multiple of interval.
    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      ckpt_p,
      cohort_hash=cohort_hash,
      processed_count=n,
      embeddings=embeddings,
      valid_flags=np.asarray(valid_flags, dtype=bool),
      errors=np.asarray(errors, dtype=object),
      elapsed_seconds=prior_elapsed + (time.perf_counter() - started),
      model_name=DEFAULT_DBETA_MODEL_NAME,
      model_revision=DEFAULT_DBETA_REVISION,
      embedding_dim=expected_emb_dim,
      random_weights=is_random,
      random_seed=int(random_seed) if is_random else 0,
      git_commit=get_git_commit_hash(),
    )
  if progress_every > 0:
    print(f"[D-BETA] finished {n} records", flush=True)
  elapsed = prior_elapsed + (time.perf_counter() - started)
  meta_dict: dict[str, Any] = {
    "subject_id": subject_ids,
    "study_id": study_ids,
    "representation_b_valid": valid_flags,
    "representation_b_error": errors,
  }
  meta_schema: dict[str, Any] = {
    "subject_id": pl.Int64,
    "study_id": pl.Int64,
    "representation_b_valid": pl.Boolean,
    "representation_b_error": pl.String,
  }
  if is_random:
    meta_dict["representation_b_random_valid"] = valid_flags
    meta_schema["representation_b_random_valid"] = pl.Boolean

  meta = pl.DataFrame(meta_dict, schema=meta_schema)
  provenance = RepresentationProvenance(
    representation=rep_name,
    dimension=int(active_adapter.config.embedding_dim),
    code_version="ecg_rep.scoring.dbeta",
    model_name=DEFAULT_DBETA_MODEL_NAME,
    model_revision=DEFAULT_DBETA_REVISION,
    frozen_encoder=True,
    preprocess={
      "waveform_eligibility": asdict(criteria) if enforce_waveform_eligibility else None,
      "target_fs": DEFAULT_SAMPLING_RATE_HZ,
      "target_sig_len": DEFAULT_SIGNAL_LENGTH_SAMPLES,
      "device": device,
      "batch_size": batch_size,
      "random_weights": is_random,
      "random_seed": int(random_seed) if is_random else None,
    },
    cohort_hash=cohort_hash,
    n_records=n,
    n_valid=int(sum(valid_flags)),
    hardware=device,
    elapsed_seconds=elapsed,
    git_commit=get_git_commit_hash(),
  )
  return meta, embeddings, provenance


def join_representation_validity(
  meta_a: pl.DataFrame,
  meta_b: pl.DataFrame,
) -> pl.DataFrame:
  """Mark patients eligible for paired A/B comparison (both representations valid)."""
  joined = meta_a.join(
    meta_b.select(["subject_id", "study_id", "representation_b_valid", "representation_b_error"]),
    on=["subject_id", "study_id"],
    how="inner",
  )
  return joined.with_columns(
    (pl.col("representation_a_valid") & pl.col("representation_b_valid")).alias(
      "representation_pair_valid"
    )
  )


def save_representation_artifacts(
  output_dir: Path | str,
  meta: pl.DataFrame,
  features_or_embeddings: npt.NDArray[np.float64],
  provenance: RepresentationProvenance,
  array_name: str,
) -> None:
  """Write gitignored metadata, array, and provenance JSON for one representation."""
  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)
  meta.write_parquet(out / f"{provenance.representation}_meta.parquet")
  np.save(out / f"{array_name}.npy", features_or_embeddings)
  (out / f"{provenance.representation}_provenance.json").write_text(
    json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )


def verify_representation_artifact_integrity(
  reps_dir: Path | str,
  cohort: pl.DataFrame,
  check_b: bool = True,
  enforce_strict_extractor_version: bool = True,
) -> None:
  """Verify cryptographic and scientific provenance of representation artifacts before evaluation.

  Strictly checks:
  1. CIIS: representation == REPRESENTATION_A_NAME, dimension == 15, feature names match CIIS_FEATURE_NAMES,
     feature_schema_hash matches CIIS_FEATURE_SCHEMA_HASH (must exist and not be empty),
     extractor version matches CIIS_EXTRACTOR_VERSION (must exist and not be empty),
     and cohort hash matches.
  2. D-BETA (if check_b): dbeta_embedding_provenance.json exists, dimension == 768, frozen_encoder == True,
     model_name == DEFAULT_DBETA_MODEL_NAME, model_revision == DEFAULT_DBETA_REVISION,
     and cohort hash matches.

  Git commit hash is recorded in provenance metadata for reproducibility tracking and post-run auditing.
  Raises ValueError on any provenance violation to prevent silent evaluation corruption.
  """
  r_dir = Path(reps_dir)
  expected_cohort_hash = cohort_identity_hash(cohort)

  prov_a_path = r_dir / "ciis_features_provenance.json"
  if not prov_a_path.exists():
    raise ValueError(
      f"Missing CIIS provenance JSON at '{prov_a_path}'. Artifact provenance verification required."
    )
  prov_a = json.loads(prov_a_path.read_text(encoding="utf-8"))
  if prov_a.get("representation") != REPRESENTATION_A_NAME:
    raise ValueError(
      f"CIIS representation mismatch: expected '{REPRESENTATION_A_NAME}', got '{prov_a.get('representation')}'."
    )
  if prov_a.get("dimension") != REPRESENTATION_A_DIM:
    raise ValueError(
      f"CIIS artifact dimension mismatch: expected {REPRESENTATION_A_DIM}, got {prov_a.get('dimension')}."
    )
  if prov_a.get("cohort_hash") != expected_cohort_hash:
    raise ValueError(
      f"CIIS cohort hash mismatch: artifact ({prov_a.get('cohort_hash')}) != current cohort ({expected_cohort_hash})."
    )
  feats_in_prov = prov_a.get("preprocess", {}).get("feature_names")
  if feats_in_prov != list(CIIS_FEATURE_NAMES):
    raise ValueError(
      f"CIIS feature schema mismatch in provenance: {feats_in_prov} != {list(CIIS_FEATURE_NAMES)}."
    )
  schema_hash_in_prov = prov_a.get("feature_schema_hash")
  if not schema_hash_in_prov or schema_hash_in_prov != CIIS_FEATURE_SCHEMA_HASH:
    raise ValueError(
      f"CIIS feature schema hash missing or mismatched: '{schema_hash_in_prov}' != expected '{CIIS_FEATURE_SCHEMA_HASH}'."
    )
  ext_ver = prov_a.get("extractor_version")
  if enforce_strict_extractor_version and (not ext_ver or ext_ver != CIIS_EXTRACTOR_VERSION):
    raise ValueError(
      f"CIIS extractor version missing or mismatched: '{ext_ver}', expected current '{CIIS_EXTRACTOR_VERSION}'."
    )

  if check_b:
    prov_b_path = r_dir / "dbeta_embedding_provenance.json"
    if not prov_b_path.exists():
      raise ValueError(
        f"Missing D-BETA provenance JSON at '{prov_b_path}'. Artifact provenance verification required."
      )
    prov_b = json.loads(prov_b_path.read_text(encoding="utf-8"))
    if prov_b.get("cohort_hash") != expected_cohort_hash:
      raise ValueError(
        f"D-BETA cohort hash mismatch: artifact ({prov_b.get('cohort_hash')}) != current cohort ({expected_cohort_hash})."
      )
    if prov_b.get("dimension") != REPRESENTATION_B_DIM:
      raise ValueError(
        f"D-BETA dimension mismatch: expected {REPRESENTATION_B_DIM}, got {prov_b.get('dimension')}."
      )
    if prov_b.get("frozen_encoder") is not True:
      raise ValueError("D-BETA encoder must be frozen (frozen_encoder must be True).")
    is_random = bool(prov_b.get("preprocess", {}).get("random_weights", False))
    model_name = prov_b.get("model_name")
    if not is_random and model_name != DEFAULT_DBETA_MODEL_NAME:
      raise ValueError(
        f"D-BETA model name mismatch: expected '{DEFAULT_DBETA_MODEL_NAME}', got '{model_name}'."
      )
    model_rev = prov_b.get("model_revision") or prov_b.get("model_version")
    if not is_random and model_rev != DEFAULT_DBETA_REVISION:
      raise ValueError(
        f"D-BETA model revision mismatch: expected '{DEFAULT_DBETA_REVISION}', got '{model_rev}'."
      )
