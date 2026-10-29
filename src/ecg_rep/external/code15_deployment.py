"""E6: Capacity policies, selective-inference stability and compute on CODE-15%.

Reimplements the selective A→B cascade comparison under the plan E6 mandate:

- The **reference mixed-score policy** (route top A-risk ``u`` to B, blend
    raw probabilities) is preserved — including its known probability/rank
    scale-mismatch pathology — never reselected from CODE test results.
- A **rank-normalized policy** maps A and B scores through ECDFs fitted on
    development/validation rows only before blending.
- Pure-A, pure-B and seeded random-routing references.
- Primary operating curve: B evaluation budgets 5/10/20/40/100% at top 5%
    clinical review; sensitivity capacities 2.5% and 10% (predeclared).
- Metrics distinguish total events captured, event-identity overlap with
    full-B capture, identity recovery, relative capture ratio (may exceed
    100%), full-B events missed, and incremental B evaluations.
- Captured deaths are *event enrichment*, never proven prevention or
    treatment benefit. No clinical-safety or cost-effectiveness claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ecg_rep.cascade import (
  ECDFRankCalibrator,
  MIN_CELL_COUNT,
  PRIMARY_CAPACITY,
  SENSITIVITY_CAPACITIES,
  UTILIZATION_FRACTIONS,
  bootstrap_primary_operating_point,
  cascade_events_captured,
  cascade_policy_comparison,
  evaluate_cascade_policy,
  random_routing_blend,
  random_routing_seed_stability,
)
from ecg_rep.external.code15_transport import MIN_CELL_COUNT as _TRANSPORT_MIN_CELL_COUNT

# The cascade module defines the suppression threshold locally so that core
# modules do not depend on external modules; CODE-15% keeps its historical
# definition in code15_transport. They must stay identical.
if MIN_CELL_COUNT != _TRANSPORT_MIN_CELL_COUNT:
  raise RuntimeError(
    "MIN_CELL_COUNT in ecg_rep.cascade and ecg_rep.external.code15_transport diverged; "
    "reconcile the suppression threshold before publishing cascade metrics."
  )

__all__ = [
  "ECDFRankCalibrator",
  "MIN_CELL_COUNT",
  "PRIMARY_CAPACITY",
  "SENSITIVITY_CAPACITIES",
  "UTILIZATION_FRACTIONS",
  "benchmark_extraction_throughput",
  "bootstrap_primary_operating_point",
  "cascade_events_captured",
  "cascade_policy_comparison",
  "evaluate_cascade_policy",
  "random_routing_blend",
  "random_routing_seed_stability",
]


def benchmark_extraction_throughput(
  n_exams: int = 32,
  seed: int = 1912,
) -> dict[str, Any]:
  """Measure CIIS (A) vs D-BETA (B) wall time on synthetic 4096x12 traces.

  D-BETA timing is only meaningful with real accelerator/model access; when
  the frozen checkpoint is unavailable this reports the CIIS path and marks
  the D-BETA row ``unavailable`` rather than simulating it. Simulated
  inference volume must never be presented as deployment cost.
  """
  import time

  from ecg_rep.external.code15_waveforms import extract_ciis_from_standardized, standardize_code15_trace

  rng = np.random.default_rng(seed)
  trace = rng.normal(0, 0.1, (4096, 12)).astype(np.float32) * 0.5
  start = time.perf_counter()
  for _ in range(n_exams):
    standardized = standardize_code15_trace(trace)
    extract_ciis_from_standardized(standardized)
  ciis_seconds = time.perf_counter() - start
  return {
    "n_exams": n_exams,
    "ciis_seconds_total": round(ciis_seconds, 4),
    "ciis_ms_per_exam": round(1000 * ciis_seconds / n_exams, 3),
    "dbeta_ms_per_exam": "unavailable_without_model",
    "note": "CIIS timing is a measured CPU wall time on synthetic traces; "
    "D-BETA timing requires the frozen checkpoint and is reported from the "
    "extraction logs, never simulated.",
  }
