"""RQ5 computational cost summaries for representation extraction and probes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import time

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ExtractionCost:
  """One-time representation extraction cost under fixed hardware/software."""

  representation: str
  n_records: int
  n_valid: int
  wall_seconds: float
  throughput_ecg_per_s: float
  per_ecg_latency_ms: float
  storage_bytes: int
  hardware: str
  notes: str = ""


@dataclass(frozen=True)
class DownstreamCost:
  """Downstream probe fit and inference cost (separate from extraction)."""

  model_name: str
  n_features: int
  n_train: int
  n_predict: int
  fit_wall_seconds: float
  predict_wall_seconds: float
  predict_throughput_per_s: float
  per_patient_predict_latency_ms: float


def extraction_cost_from_provenance(
  provenance: dict[str, Any],
  array_path: Path | None = None,
) -> ExtractionCost:
  """Build an ExtractionCost from a saved provenance JSON (+ optional array size)."""
  n = int(provenance.get("n_records", 0))
  elapsed = float(provenance.get("elapsed_seconds", 0.0))
  throughput = (n / elapsed) if elapsed > 0 else float("nan")
  latency_ms = (1000.0 * elapsed / n) if n > 0 else float("nan")
  storage = int(array_path.stat().st_size) if array_path is not None and array_path.exists() else 0
  return ExtractionCost(
    representation=str(provenance.get("representation", "unknown")),
    n_records=n,
    n_valid=int(provenance.get("n_valid", 0)),
    wall_seconds=elapsed,
    throughput_ecg_per_s=throughput,
    per_ecg_latency_ms=latency_ms,
    storage_bytes=storage,
    hardware=str(provenance.get("hardware", "unknown")),
    notes="Excludes D-BETA pretraining; one-time extraction only.",
  )


def load_extraction_costs(
  reps_dir: Path,
  names: tuple[tuple[str, str], ...] = (
    ("ciis_features", "ciis_features.npy"),
    ("dbeta_embedding", "dbeta_embeddings.npy"),
  ),
) -> list[ExtractionCost]:
  """Load extraction costs for available representation provenance files."""
  out: list[ExtractionCost] = []
  for stem, array_name in names:
    prov_path = reps_dir / f"{stem}_provenance.json"
    if not prov_path.exists():
      continue
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    out.append(extraction_cost_from_provenance(provenance, reps_dir / array_name))
  return out


def time_probe_downstream(
  fit_fn: Any,
  predict_fn: Any,
  model_name: str,
  n_features: int,
  n_train: int,
  n_predict: int,
) -> DownstreamCost:
  """Time a zero-arg fit callable and a zero-arg predict callable after warm-up."""
  predict_fn()  # warm-up
  t0 = time.perf_counter()
  fit_fn()
  fit_s = time.perf_counter() - t0
  t1 = time.perf_counter()
  predict_fn()
  pred_s = time.perf_counter() - t1
  throughput = (n_predict / pred_s) if pred_s > 0 else float("nan")
  latency_ms = (1000.0 * pred_s / n_predict) if n_predict > 0 else float("nan")
  return DownstreamCost(
    model_name=model_name,
    n_features=n_features,
    n_train=n_train,
    n_predict=n_predict,
    fit_wall_seconds=fit_s,
    predict_wall_seconds=pred_s,
    predict_throughput_per_s=throughput,
    per_patient_predict_latency_ms=latency_ms,
  )


def incremental_extraction_cost(
  cost_a: ExtractionCost,
  cost_b: ExtractionCost,
) -> dict[str, Any]:
  """Incremental B-over-A extraction cost ratios/deltas for RQ5 tables."""
  hw_matched = cost_a.hardware == cost_b.hardware
  scenario = "controlled_hardware_benchmark" if hw_matched else "optimized_deployment_benchmark"
  if hw_matched:
    note = f"Controlled hardware benchmark: matched hardware ({cost_a.hardware})."
  else:
    note = (
      f"Heterogeneous hardware benchmark: CIIS on {cost_a.hardware} vs D-BETA on {cost_b.hardware}. "
      "Wall-time ratios reflect deployment paths rather than controlled resource equivalence."
    )

  return {
    "hardware_matched": hw_matched,
    "hardware_a": cost_a.hardware,
    "hardware_b": cost_b.hardware,
    "scenario": scenario,
    "comparison_note": note,
    "delta_wall_seconds": cost_b.wall_seconds - cost_a.wall_seconds,
    "wall_ratio_b_over_a": (
      cost_b.wall_seconds / cost_a.wall_seconds
      if cost_a.wall_seconds > 0
      else float("nan")
    ),
    "delta_storage_bytes": float(cost_b.storage_bytes - cost_a.storage_bytes),
    "storage_ratio_b_over_a": (
      cost_b.storage_bytes / cost_a.storage_bytes
      if cost_a.storage_bytes > 0
      else float("nan")
    ),
    "throughput_ratio_a_over_b": (
      cost_a.throughput_ecg_per_s / cost_b.throughput_ecg_per_s
      if cost_b.throughput_ecg_per_s > 0
      else float("nan")
    ),
  }


def costs_to_jsonable(costs: list[ExtractionCost] | list[DownstreamCost]) -> list[dict[str, Any]]:
  """Serialize cost dataclasses for JSON reports."""
  return [asdict(c) for c in costs]


def array_nbytes(arr: npt.NDArray[Any]) -> int:
  """In-memory size of a NumPy array."""
  return int(arr.nbytes)
