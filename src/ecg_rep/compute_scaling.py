"""LM8: equal-resource throughput scaling on one workstation.

CIIS is timed with process workers. D-BETA CPU is timed with an explicit
intra-op thread count and inter-op threads fixed at 1. Those are different
mechanisms, so the table does not reduce them to one ratio. A ratio is emitted
only when both sides name the same mechanism and the same allocation.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import polars as pl

from ecg_rep.compute_benchmark import (
  _load_records,
  _peak_rss_bytes,
  _run_isolated,
  sample_benchmark_rows,
)
from ecg_rep.data import read_ecg_waveform
from ecg_rep.scoring.traditional import score_ecg_waveform

CPU_ALLOCATIONS: tuple[int, ...] = (1, 2, 4, 8, 12)
THREAD_GRID_BATCH_SIZE = 32
CPU_BATCH_GRID_THREADS = 8
CPU_BATCH_SIZES: tuple[int, ...] = (1, 8, 32, 64)
MPS_BATCH_CANDIDATES: tuple[int, ...] = (1, 8, 32, 64, 128)
INTEROP_THREADS = 1
WARMUP_PASSES = 2
TIMED_REPEATS = 5
SAMPLE_N = 1024
SAMPLE_SEED = 2026


def freeze_batch_sizes(candidates: tuple[int, ...], fits: dict[int, bool]) -> tuple[int, ...]:
  """Keep batch sizes that the smoke run could allocate. Order stays frozen."""
  missing = [size for size in candidates if size not in fits]
  if missing:
    raise ValueError(f"Smoke result missing batch sizes: {missing}")
  return tuple(size for size in candidates if fits[size])


def matched_ratio(left: dict[str, Any], right: dict[str, Any]) -> float | None:
  """Throughput ratio only when the resource mechanism and allocation match."""
  if left.get("phase") != "warm" or right.get("phase") != "warm":
    return None
  if left.get("resource_mechanism") != right.get("resource_mechanism"):
    return None
  if left.get("resource_allocation") != right.get("resource_allocation"):
    return None
  if left.get("batch_size") != right.get("batch_size"):
    return None
  denom = float(right["ecgs_per_second"])
  if denom <= 0:
    return None
  return float(left["ecgs_per_second"]) / denom


def _chunks(paths: list[str], n_workers: int) -> list[list[str]]:
  if n_workers < 1:
    raise ValueError("n_workers must be positive")
  size = math.ceil(len(paths) / n_workers)
  return [paths[i : i + size] for i in range(0, len(paths), size)]


def _score_records(records: list[Any]) -> None:
  for sig, leads, fs in records:
    score_ecg_waveform(sig, leads, fs)


def _ciis_proc(
  paths: list[str],
  repeats: int,
  warmup: int,
  loaded: Any,
  warmed: Any,
  release_warmup: Any,
  release_timing: Any,
  out: Any,
) -> None:
  """Load, warm up, then time. Two events keep the timed passes aligned."""
  records = [read_ecg_waveform(Path(p)) for p in paths]
  loaded.set()
  release_warmup.wait()
  for _ in range(warmup):
    _score_records(records)
  warmed.set()
  release_timing.wait()
  elapsed: list[float] = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    _score_records(records)
    elapsed.append(time.perf_counter() - t0)
  out.put({"elapsed_s": elapsed, "n": len(records), "peak_rss_bytes": _peak_rss_bytes()})


def time_ciis_workers(paths: list[str], n_workers: int, repeats: int = TIMED_REPEATS, warmup: int = WARMUP_PASSES) -> dict[str, Any]:
  """Makespan of aligned passes. Load and warmup stay outside the timed section."""
  ctx = mp.get_context("spawn")
  chunks = _chunks(paths, n_workers)
  release_warmup = ctx.Event()
  release_timing = ctx.Event()
  out = ctx.Queue()
  procs = []
  loaded_flags = []
  warmed_flags = []
  for chunk in chunks:
    loaded = ctx.Event()
    warmed = ctx.Event()
    proc = ctx.Process(
      target=_ciis_proc,
      args=(chunk, repeats, warmup, loaded, warmed, release_warmup, release_timing, out),
    )
    proc.start()
    procs.append(proc)
    loaded_flags.append(loaded)
    warmed_flags.append(warmed)
  for flag in loaded_flags:
    flag.wait()
  release_warmup.set()
  for flag in warmed_flags:
    flag.wait()
  release_timing.set()
  reports = [out.get() for _ in chunks]
  for proc in procs:
    proc.join()
  per_repeat = [max(report["elapsed_s"][i] for report in reports) for i in range(repeats)]
  n = len(paths)
  return _timing_row(
    pathway="ciis_cpu_process_workers",
    mechanism="process_workers",
    allocation=n_workers,
    batch_size=None,
    phase="warm",
    per_ecg_ms=[1000.0 * t / n for t in per_repeat],
    batch_latency_ms=[1000.0 * t for t in per_repeat],
    peak_rss_bytes=max(int(report["peak_rss_bytes"]) for report in reports),
    extra={"interop_threads": None, "n_ecgs": n},
  )


def _timing_row(
  pathway: str,
  mechanism: str,
  allocation: int | None,
  batch_size: int | None,
  phase: str,
  per_ecg_ms: list[float],
  batch_latency_ms: list[float],
  peak_rss_bytes: int,
  extra: dict[str, Any],
) -> dict[str, Any]:
  ordered = sorted(per_ecg_ms)
  median = statistics.median(ordered)
  n = int(extra.get("n_ecgs") or 0)
  return {
    "pathway": pathway,
    "resource_mechanism": mechanism,
    "resource_allocation": allocation,
    "batch_size": batch_size,
    "phase": phase,
    "median_ms_per_ecg": median,
    "p95_ms_per_ecg": _percentile(ordered, 0.95),
    "min_ms_per_ecg": ordered[0],
    "max_ms_per_ecg": ordered[-1],
    "ecgs_per_second": (1000.0 / median) if median > 0 else None,
    "median_batch_latency_ms": sorted(batch_latency_ms)[len(batch_latency_ms) // 2] if batch_latency_ms else None,
    # Worker high-water marks are not additive simultaneous process memory.
    "peak_rss_bytes": peak_rss_bytes if mechanism != "process_workers" else None,
    "max_worker_peak_rss_bytes": peak_rss_bytes if mechanism == "process_workers" else None,
    "interop_threads": extra.get("interop_threads"),
    "n_ecgs": n,
    "outcome_columns_read": False,
    "parent_wall_s": extra.get("parent_wall_s"),
  }


def apply_cpu_threads(threads: int, interop: int = INTEROP_THREADS) -> dict[str, int]:
  """Set intra-op threads and record the inter-op policy. Call before model load."""
  import torch

  torch.set_num_threads(threads)
  try:
    torch.set_num_interop_threads(interop)
  except RuntimeError:
    # Already set in this process; the recorded policy is still the frozen value.
    pass
  return {"intra_op_threads": threads, "interop_threads": interop, "torch_get_num_threads": torch.get_num_threads()}


def _dbeta_scaling_worker(
  paths: list[str],
  device: str,
  threads: int,
  batch_size: int,
  repeats: int,
  warmup: int,
  smoke_only: bool,
) -> dict[str, Any]:
  import torch

  from ecg_rep.scoring.dbeta import DbetaAdapter, DbetaConfig

  thread_info = apply_cpu_threads(threads) if device == "cpu" else {"intra_op_threads": None, "interop_threads": None}
  records = _load_records(paths)
  t_load = time.perf_counter()
  adapter = DbetaAdapter(DbetaConfig(device=device, batch_size=batch_size))
  adapter._ensure_model_loaded()  # pyright: ignore[reportPrivateUsage]
  load_s = time.perf_counter() - t_load
  if smoke_only:
    adapter.embed_batch(records[:batch_size])
    return {"fits": True, "batch_size": batch_size, "device": device, "model_load_s": load_s}

  def sync() -> None:
    if device == "mps":
      torch.mps.synchronize()

  t_first = time.perf_counter()
  adapter.embed_batch(records[:batch_size])
  sync()
  first_s = time.perf_counter() - t_first
  for i in range(warmup):
    adapter.embed_batch(records[i * batch_size : (i + 1) * batch_size] or records[:batch_size])
  sync()
  per_ecg: list[float] = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    adapter.embed_batch(records)
    sync()
    per_ecg.append(1000.0 * (time.perf_counter() - t0) / len(records))
  row = _timing_row(
    pathway=f"dbeta_{device}",
    mechanism="mps" if device == "mps" else "cpu_threads",
    allocation=threads if device == "cpu" else None,
    batch_size=batch_size,
    phase="warm",
    per_ecg_ms=per_ecg,
    batch_latency_ms=per_ecg,
    peak_rss_bytes=_peak_rss_bytes(),
    extra={"interop_threads": thread_info.get("interop_threads"), "n_ecgs": len(records)},
  )
  row["model_load_s"] = load_s
  row["first_batch_s"] = first_s
  if device == "mps":
    row["mps_driver_allocated_bytes"] = int(torch.mps.driver_allocated_memory())
  return row


def _percentile(ordered: list[float], q: float) -> float:
  if len(ordered) == 1:
    return ordered[0]
  pos = q * (len(ordered) - 1)
  lo = math.floor(pos)
  hi = math.ceil(pos)
  if lo == hi:
    return ordered[lo]
  return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cold_phase_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
  """Split model-load and first-batch times out of a warm D-BETA result."""
  load_s = result.pop("model_load_s", None)
  first_s = result.pop("first_batch_s", None)
  rows = [result]
  n = max(int(result["n_ecgs"]), 1)
  if load_s is not None:
    load_ms = 1000.0 * float(load_s)
    rows.append(_phase_copy(result, "cold_model_load", load_ms / n, load_ms, None))
  if first_s is not None:
    batch = int(result["batch_size"] or 1)
    first_ms = 1000.0 * float(first_s)
    rows.append(_phase_copy(result, "cold_first_batch", first_ms / batch, first_ms, batch / float(first_s)))
  return rows


def _phase_copy(
  warm: dict[str, Any], phase: str, ms_per_ecg: float, latency_ms: float, throughput: float | None,
) -> dict[str, Any]:
  row = {key: value for key, value in warm.items() if key != "mps_driver_allocated_bytes"}
  row["phase"] = phase
  row["median_ms_per_ecg"] = ms_per_ecg
  row["p95_ms_per_ecg"] = ms_per_ecg
  row["min_ms_per_ecg"] = ms_per_ecg
  row["max_ms_per_ecg"] = ms_per_ecg
  row["ecgs_per_second"] = throughput
  row["median_batch_latency_ms"] = latency_ms
  return row


def waveform_paths(relative: list[str], ecg_root: Path) -> list[str]:
  """Resolve cohort-relative waveform paths. Absolute paths are left alone."""
  resolved: list[str] = []
  for raw in relative:
    path = Path(raw)
    resolved.append(str(path if path.is_absolute() else ecg_root / path))
  return resolved


def is_allocation_failure(exc: BaseException) -> bool:
  """True only for device memory failures. Missing files are not a batch-size decision."""
  name = type(exc).__name__
  if name in {"OutOfMemoryError", "MemoryError"}:
    return True
  text = str(exc).lower()
  return "out of memory" in text or "mps backend out of memory" in text


def smoke_fits(paths: list[str], device: str, batch_size: int) -> bool:
  """One forward pass. Only an allocation failure drops the batch size."""
  try:
    result = _run_isolated(_dbeta_scaling_worker, paths[: max(batch_size, 1)], device, 1, batch_size, 1, 0, True)
  except Exception as exc:  # noqa: BLE001 - re-raised unless it is an allocation failure
    if is_allocation_failure(exc):
      return False
    raise
  return bool(result.get("fits"))


def resolve_sample(cohort: pl.DataFrame, manifest_path: Path) -> pl.DataFrame:
  """Reuse a recorded sample. Otherwise draw the outcome-blind 1024-row sample."""
  allowed = {"subject_id", "study_id", "path"}
  if manifest_path.exists():
    prior = json.loads(manifest_path.read_text())
    ids = pl.DataFrame(prior["sample_ids"]).select(["subject_id", "study_id"])
    sampled = ids.join(cohort.select(["subject_id", "study_id", "path"]), on=["subject_id", "study_id"], how="inner")
    if sampled.height != ids.height:
      raise ValueError("Recorded benchmark IDs are not all present in the cohort")
  else:
    sampled = sample_benchmark_rows(cohort, n=SAMPLE_N, seed=SAMPLE_SEED)
  if set(sampled.columns) - allowed:
    raise ValueError(f"Benchmark sample read columns outside {sorted(allowed)}")
  return sampled.select(["subject_id", "study_id", "path"])


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_compute_scaling(
  cohort_path: Path,
  ecg_root: Path,
  output_dir: Path,
  report_path: Path,
  *,
  skip_timing: bool = False,
) -> dict[str, Any]:
  """Smoke-freeze batch sizes, then time the frozen grid. Does not read outcomes."""
  cohort = pl.read_parquet(cohort_path)
  manifest_path = output_dir / "compute_scaling_manifest.json"
  sampled = resolve_sample(cohort, manifest_path)
  paths = waveform_paths([str(p) for p in sampled["path"].to_list()], ecg_root)
  base = {
    "analysis_status": "post_hoc_measurement",
    "sample_n": sampled.height,
    "sample_seed": SAMPLE_SEED,
    "sample_ids": sampled.select(["subject_id", "study_id"]).to_dicts(),
    "outcome_columns_read": False,
    "cpu_allocations": list(CPU_ALLOCATIONS),
    "thread_grid_batch_size": THREAD_GRID_BATCH_SIZE,
    "cpu_batch_grid_threads": CPU_BATCH_GRID_THREADS,
    "interop_threads": INTEROP_THREADS,
    "warmup_passes": WARMUP_PASSES,
    "timed_repeats": TIMED_REPEATS,
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "ciis_timing": "max worker pass time after a shared start; load and warmup excluded",
    "dbeta_timing": "five full-sample passes after model load, one untimed first batch, and two warmup batches",
    "timed_evaluation": "not_started",
  }
  _write_manifest(manifest_path, base)

  cpu_fits = {size: smoke_fits(paths, "cpu", size) for size in CPU_BATCH_SIZES}
  mps_fits = {size: smoke_fits(paths, "mps", size) for size in MPS_BATCH_CANDIDATES}
  cpu_batches = freeze_batch_sizes(CPU_BATCH_SIZES, cpu_fits)
  mps_batches = freeze_batch_sizes(MPS_BATCH_CANDIDATES, mps_fits)
  base["cpu_batch_sizes_frozen"] = list(cpu_batches)
  base["mps_batch_sizes_frozen"] = list(mps_batches)
  base["cpu_smoke_fits"] = {str(k): v for k, v in cpu_fits.items()}
  base["mps_smoke_fits"] = {str(k): v for k, v in mps_fits.items()}
  base["ecg_root"] = str(ecg_root)
  if THREAD_GRID_BATCH_SIZE not in cpu_batches:
    base["timed_evaluation"] = "invalid_thread_grid_batch"
    _write_manifest(manifest_path, base)
    raise RuntimeError(
      f"Thread-grid batch size {THREAD_GRID_BATCH_SIZE} did not fit; "
      "the frozen comparison is invalid and was not retimed at another batch size"
    )
  base["timed_evaluation"] = "frozen_not_timed" if skip_timing else "in_progress"
  _write_manifest(manifest_path, base)
  if skip_timing:
    return base

  rows: list[dict[str, Any]] = []
  for workers in CPU_ALLOCATIONS:
    rows.append(time_ciis_workers(paths, workers))
  for threads in CPU_ALLOCATIONS:
    result = _run_isolated(
      _dbeta_scaling_worker, paths, "cpu", threads, THREAD_GRID_BATCH_SIZE, TIMED_REPEATS, WARMUP_PASSES, False,
    )
    rows.extend(cold_phase_rows(result))
  for batch_size in cpu_batches:
    if batch_size == THREAD_GRID_BATCH_SIZE:
      continue
    result = _run_isolated(
      _dbeta_scaling_worker, paths, "cpu", CPU_BATCH_GRID_THREADS, batch_size, TIMED_REPEATS, WARMUP_PASSES, False,
    )
    rows.extend(cold_phase_rows(result))
  for batch_size in mps_batches:
    result = _run_isolated(
      _dbeta_scaling_worker, paths, "mps", 0, batch_size, TIMED_REPEATS, WARMUP_PASSES, False,
    )
    rows.extend(cold_phase_rows(result))

  for row in rows:
    row.pop("parent_wall_s", None)
    row.pop("mps_driver_allocated_bytes", None)
  output_dir.mkdir(parents=True, exist_ok=True)
  pl.DataFrame([{col: row.get(col) for col in SCALING_COLUMNS} for row in rows]).write_csv(
    output_dir / "compute_scaling.csv"
  )
  base["timed_evaluation"] = "complete"
  base["n_rows"] = len(rows)
  _write_manifest(manifest_path, base)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(_render_report(rows, base))
  return base


SCALING_COLUMNS: tuple[str, ...] = (
  "pathway",
  "resource_mechanism",
  "resource_allocation",
  "batch_size",
  "phase",
  "median_ms_per_ecg",
  "p95_ms_per_ecg",
  "min_ms_per_ecg",
  "max_ms_per_ecg",
  "ecgs_per_second",
  "median_batch_latency_ms",
  "peak_rss_bytes",
  "max_worker_peak_rss_bytes",
  "interop_threads",
  "n_ecgs",
  "outcome_columns_read",
)


def _render_report(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
  warm = [row for row in rows if row["phase"] == "warm"]
  lines = [
    "---",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    "analysis_status: post_hoc_measurement",
    f"source_commit: {manifest.get('source_commit', 'unknown')}",
    "---",
    "",
    "# LM8 equal-resource compute scaling",
    "",
    "CIIS rows use process workers. D-BETA CPU rows use intra-op threads with inter-op threads fixed at "
    f"{manifest['interop_threads']}. Those mechanisms are not one ratio.",
    "CIIS memory is `max_worker_peak_rss_bytes`: the largest worker high-water mark, not total",
    "process-pool RSS. D-BETA `peak_rss_bytes` is single-process RSS; these memory scopes are not comparable.",
    "",
    f"Sample n={manifest['sample_n']}, seed={manifest['sample_seed']}. Outcome columns were not read.",
    f"Frozen after smoke, before timed repeats: CPU batches {manifest.get('cpu_batch_sizes_frozen')}, "
    f"MPS batches {manifest.get('mps_batch_sizes_frozen')}.",
    "",
    "| pathway | mechanism | allocation | batch | median ms/ECG | min | max | ECGs/s |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  for row in warm:
    lines.append(
      f"| {row['pathway']} | {row['resource_mechanism']} | {row['resource_allocation']} | "
      f"{row['batch_size']} | {row['median_ms_per_ecg']:.4f} | {row['min_ms_per_ecg']:.4f} | "
      f"{row['max_ms_per_ecg']:.4f} | {row['ecgs_per_second']:.3f} |"
    )
  lines += [
    "", "## Reading", "",
    "This is one workstation. Warm rows exclude model load and untimed first-batch/warmup work.",
    "CIIS loads records and warms two full passes per worker; D-BETA warms two batches.",
    "Only contrasts within the same resource mechanism describe scaling; there is no cross-mechanism speed ratio.",
    "The batch-size comparisons are descriptive within the frozen grid, not a universal deployment optimum.",
    "Cold load and first-batch rows are in the CSV and are not warm throughput.", "",
  ]
  return "\n".join(lines) + "\n"
