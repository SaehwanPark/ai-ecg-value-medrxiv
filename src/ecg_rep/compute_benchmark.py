"""Controlled extraction-cost benchmark for CIIS (A) and D-BETA (B) on real cohort ECGs.

The earlier cost figures came from full-cohort extraction logs on different hardware
paths (CIIS in a CPU process pool, D-BETA on Apple MPS), so they could not support a
like-for-like statement. This benchmark times both extractors on the same fixed
sample of cohort ECGs, after the waveforms are already in memory, with warm-up and
repeated timed passes, and runs D-BETA on both CPU and MPS. Each pathway runs in a
fresh spawned process so that its peak resident memory is its own.

The sample is drawn from identifiers and file paths only; outcome columns are never
read. Timings describe one workstation and are workload arithmetic, not costs.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from ecg_rep.data import read_ecg_waveform
from ecg_rep.scoring.traditional import score_ecg_waveform

Record = tuple[npt.NDArray[np.float64], list[str], int]


def sample_benchmark_rows(cohort: pl.DataFrame, n: int = 1024, seed: int = 2026) -> pl.DataFrame:
  """Outcome-blind seeded sample of (subject_id, study_id, path) rows."""
  ids = cohort.select(["subject_id", "study_id", "path"]).sort(["subject_id", "study_id"])
  return ids.sample(n=min(n, ids.height), seed=seed, shuffle=True)


def _peak_rss_bytes() -> int:
  # ru_maxrss is bytes on macOS and kilobytes on Linux.
  peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return int(peak if sys.platform == "darwin" else peak * 1024)


def _summary(per_ecg_ms: list[float]) -> dict[str, float]:
  return {
    "median_ms_per_ecg": statistics.median(per_ecg_ms),
    "min_ms_per_ecg": min(per_ecg_ms),
    "max_ms_per_ecg": max(per_ecg_ms),
  }


def _load_records(paths: list[str]) -> list[Record]:
  return [read_ecg_waveform(Path(p)) for p in paths]


def _ciis_worker(paths: list[str], repeats: int, warmup: int) -> dict[str, Any]:
  records = _load_records(paths)
  for sig, leads, fs in records[:warmup]:
    score_ecg_waveform(sig, leads, fs)
  per_ecg: list[float] = []
  n_valid = 0
  for _ in range(repeats):
    start = time.perf_counter()
    n_valid = sum(score_ecg_waveform(sig, leads, fs).is_valid for sig, leads, fs in records)
    per_ecg.append(1000.0 * (time.perf_counter() - start) / len(records))
  return {
    "pathway": "ciis_cpu_single_process",
    "n_ecgs": len(records),
    "n_valid": int(n_valid),
    "repeats_ms_per_ecg": per_ecg,
    **_summary(per_ecg),
    "peak_rss_bytes": _peak_rss_bytes(),
  }


def _dbeta_worker(paths: list[str], device: str, batch_size: int, repeats: int, warmup_batches: int) -> dict[str, Any]:
  import torch

  from ecg_rep.scoring.dbeta import DbetaAdapter, DbetaConfig

  torch.manual_seed(0)
  records = _load_records(paths)
  adapter = DbetaAdapter(DbetaConfig(device=device, batch_size=batch_size))
  model = adapter._ensure_model_loaded()  # pyright: ignore[reportPrivateUsage]
  n_params = int(sum(p.numel() for p in model.parameters()))

  def sync() -> None:
    if device == "mps":
      torch.mps.synchronize()

  for i in range(warmup_batches):
    adapter.embed_batch(records[i * batch_size : (i + 1) * batch_size])
  sync()
  per_ecg: list[float] = []
  n_valid = 0
  for _ in range(repeats):
    start = time.perf_counter()
    result = adapter.embed_batch(records)
    sync()
    per_ecg.append(1000.0 * (time.perf_counter() - start) / len(records))
    n_valid = result.count_valid
  out: dict[str, Any] = {
    "pathway": f"dbeta_{device}_batch{batch_size}",
    "device": device,
    "batch_size": batch_size,
    "torch_threads": torch.get_num_threads(),
    "n_ecgs": len(records),
    "n_valid": int(n_valid),
    "repeats_ms_per_ecg": per_ecg,
    **_summary(per_ecg),
    "peak_rss_bytes": _peak_rss_bytes(),
    "n_parameters": n_params,
    "fp32_weight_bytes": n_params * 4,
  }
  if device == "mps":
    out["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
  return out


def _io_worker(paths: list[str]) -> dict[str, Any]:
  start = time.perf_counter()
  records = _load_records(paths)
  elapsed = time.perf_counter() - start
  return {
    "pathway": "wfdb_read_local_disk",
    "n_ecgs": len(records),
    "median_ms_per_ecg": 1000.0 * elapsed / len(records),
    "note": "single cold-to-warm pass; OS file cache state not controlled",
  }


DOWNSTREAM_MODELS: tuple[str, ...] = ("A_ecg_only", "B_ecg_only", "C0_plus_A", "C0_plus_B")


def _downstream_worker(
  cohort_path: str, c0_path: str, reps_dir: str, model: str, fit_repeats: int, predict_repeats: int
) -> dict[str, Any]:
  from ecg_rep.evaluate import C0_FEATURE_COLS, _join_representation_matrix, _split_xy  # pyright: ignore[reportPrivateUsage]
  from ecg_rep.probe import fit_logistic_probe

  reps = Path(reps_dir)
  c0 = pl.read_parquet(c0_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]
  frame = pl.read_parquet(cohort_path).join(
    c0.select(["subject_id", "study_id", *c0_cols]), on=["subject_id", "study_id"]
  )
  frame = _join_representation_matrix(
    frame, pl.read_parquet(reps / "ciis_features_meta.parquet"), np.load(reps / "ciis_features.npy"),
    "za", "representation_a_valid",
  )
  frame = _join_representation_matrix(
    frame, pl.read_parquet(reps / "dbeta_embedding_meta.parquet"), np.load(reps / "dbeta_embeddings.npy"),
    "zb", "representation_b_valid",
  ).filter(pl.col("representation_a_valid") & pl.col("representation_b_valid"))
  za = [c for c in frame.columns if c.startswith("za_")]
  zb = [c for c in frame.columns if c.startswith("zb_")]
  cols = {"A_ecg_only": za, "B_ecg_only": zb, "C0_plus_A": c0_cols + za, "C0_plus_B": c0_cols + zb}[model]
  dev, val, test = (_split_xy(frame, cols, s) for s in ("dev", "val", "test"))
  del frame
  rss_after_load = _peak_rss_bytes()

  fit_s: list[float] = []
  probe = None
  for _ in range(fit_repeats):
    start = time.perf_counter()
    probe = fit_logistic_probe(dev.x, dev.y, val.x, val.y)
    fit_s.append(time.perf_counter() - start)
  assert probe is not None
  probe.predict_proba(test.x)
  predict_us: list[float] = []
  for _ in range(predict_repeats):
    start = time.perf_counter()
    probe.predict_proba(test.x)
    predict_us.append(1e6 * (time.perf_counter() - start) / test.x.shape[0])
  return {
    "pathway": f"probe_{model}",
    "n_features": len(cols),
    "n_dev": int(dev.x.shape[0]),
    "n_val": int(val.x.shape[0]),
    "n_test": int(test.x.shape[0]),
    "n_c_grid": 8,
    "fit_seconds_repeats": fit_s,
    "fit_seconds_median": statistics.median(fit_s),
    "best_c": probe.best_c,
    "predict_us_per_ecg_repeats": predict_us,
    "predict_us_per_ecg_median": statistics.median(predict_us),
    "predict_ecgs_per_second": 1e6 / statistics.median(predict_us),
    "peak_rss_after_load_bytes": rss_after_load,
    "peak_rss_bytes": _peak_rss_bytes(),
    "dev_matrix_float64_bytes": int(dev.x.nbytes),
  }


def run_downstream_benchmark(
  cohort_path: Path,
  c0_path: Path,
  reps_dir: Path,
  output_dir: Path,
  tables_dir: Path,
  fit_repeats: int = 3,
  predict_repeats: int = 20,
) -> dict[str, Any]:
  """Time full probe tuning (development fit, validation C grid) and locked-test prediction."""
  pathways = [
    _run_isolated(_downstream_worker, str(cohort_path), str(c0_path), str(reps_dir), m, fit_repeats, predict_repeats)
    for m in DOWNSTREAM_MODELS
  ]
  results: dict[str, Any] = {
    "status": "complete",
    "protocol": {
      "fit_repeats": fit_repeats,
      "predict_repeats": predict_repeats,
      "fit_scope": "development standardization plus 8-value C grid fit and validation log-loss selection",
      "predict_scope": "frozen probe on the paired-valid locked-test matrix after one warm-up call; no metrics computed",
      "cohort": "paired-valid (A and B valid)",
    },
    "hardware": {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version()},
    "pathways": pathways,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  tables_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "downstream_benchmark.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
  pl.DataFrame([
    {
      "pathway": p["pathway"],
      "n_features": p["n_features"],
      "n_dev": p["n_dev"],
      "fit_seconds_median": p["fit_seconds_median"],
      "fit_seconds_min": min(p["fit_seconds_repeats"]),
      "fit_seconds_max": max(p["fit_seconds_repeats"]),
      "predict_us_per_ecg_median": p["predict_us_per_ecg_median"],
      "predict_ecgs_per_second": p["predict_ecgs_per_second"],
      "peak_rss_mb": p["peak_rss_bytes"] / 2**20,
    }
    for p in pathways
  ]).write_csv(tables_dir / "supp_downstream_benchmark.csv")
  return results


def _run_isolated(fn: Any, *args: Any) -> dict[str, Any]:
  ctx = mp.get_context("spawn")
  with ctx.Pool(1) as pool:
    return pool.apply(fn, args)


def run_compute_benchmark(
  cohort_path: Path,
  ecg_root: Path,
  output_dir: Path,
  tables_dir: Path,
  n_ecgs: int = 1024,
  seed: int = 2026,
  repeats: int = 5,
  warmup_batches: int = 2,
  batch_size: int = 64,
  devices: tuple[str, ...] = ("cpu", "mps"),
) -> dict[str, Any]:
  """Time CIIS and D-BETA on the same in-memory ECG sample and write a summary table."""
  rows = sample_benchmark_rows(pl.read_parquet(cohort_path), n=n_ecgs, seed=seed)
  paths = [str(ecg_root / p) for p in rows["path"].to_list()]
  pathways = [
    _run_isolated(_io_worker, paths),
    _run_isolated(_ciis_worker, paths, repeats, warmup_batches * batch_size),
    *(_run_isolated(_dbeta_worker, paths, d, batch_size, repeats, warmup_batches) for d in devices),
  ]
  ciis = next(p for p in pathways if p["pathway"].startswith("ciis"))
  for p in pathways:
    if p["pathway"].startswith("dbeta"):
      p["ratio_to_ciis_median"] = p["median_ms_per_ecg"] / ciis["median_ms_per_ecg"]
  storage = {
    "a_float32_bytes_per_ecg": 15 * 4,
    "b_float32_bytes_per_ecg": 768 * 4,
  }
  results: dict[str, Any] = {
    "status": "complete",
    "sample": {"n_ecgs": len(paths), "seed": seed, "outcome_columns_read": False},
    "protocol": {
      "repeats": repeats,
      "warmup_batches": warmup_batches,
      "batch_size": batch_size,
      "timing_scope": "waveform already in memory; includes preprocessing and feature/embedding computation",
    },
    "hardware": {
      "machine": platform.machine(),
      "platform": platform.platform(),
      "processor": platform.processor(),
      "python": platform.python_version(),
    },
    "pathways": pathways,
    "storage": storage,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  tables_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "compute_benchmark.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
  pl.DataFrame([
    {
      "pathway": p["pathway"],
      "n_ecgs": p["n_ecgs"],
      "median_ms_per_ecg": p["median_ms_per_ecg"],
      "min_ms_per_ecg": p.get("min_ms_per_ecg"),
      "max_ms_per_ecg": p.get("max_ms_per_ecg"),
      "ratio_to_ciis_median": p.get("ratio_to_ciis_median"),
      "peak_rss_mb": p["peak_rss_bytes"] / 2**20 if "peak_rss_bytes" in p else None,
      "n_parameters": p.get("n_parameters"),
    }
    for p in pathways
  ]).write_csv(tables_dir / "supp_compute_benchmark.csv")
  return results
