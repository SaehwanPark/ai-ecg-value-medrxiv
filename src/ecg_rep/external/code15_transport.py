"""E3: Frozen MIMIC-trained probe transport onto CODE-15% (no CODE fitting).

Scores the **unchanged** MIMIC-fitted ``M_A_ECG``/``M_B_ECG`` probes on
extracted CODE-15% representations and computes the prespecified paired
discrimination, calibration, and fixed-capacity metrics.

Governance (plan Agent E3):

- No CODE-specific model fitting, calibration, threshold tuning, or
  outcome-dependent preprocessing. Probe files must hash-match the E0
  baseline before use.
- Per the delegated G-E1 decision (dual-track): the 30-day track is
  **descriptive** (wide-uncertainty caveats, no subgroup/cascade claims);
  the 1-year (365.25-day) track is the primary estimand and is labeled a
  **different-horizon association** for the frozen 30-day-trained scores —
  never same-endpoint external validation.
- Censored-before-horizon patients are excluded from the binary analysis
  with a transparent attrition count. When a patient split is available, an
  audit-requested IPCW discrimination sensitivity is reported separately on
  the locked test; see D4 for its assumptions and post hoc status.
- Aggregate metrics only are returned/committed; patient-level predictions
  stay in the gitignored artifact tree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ecg_rep.external.code15_labels import DAYS_PER_YEAR, HORIZON_DAYS, horizon_label_to_int
from ecg_rep.probe import TrainedProbe

#: Frozen probe artifacts (ECG-only MIMIC models) transported unchanged.
TRANSPORT_MODEL_FILES: dict[str, str] = {
  "M_A_ECG": "models/M_A_ECG.json",
  "M_B_ECG": "models/M_B_ECG.json",
}

#: Expected input dimensionalities enforcing the 15 A / 768 B contract.
EXPECTED_FEATURE_DIMS: dict[str, int] = {"M_A_ECG": 15, "M_B_ECG": 768}

#: Fixed clinical-review capacities for the capacity table.
CAPACITY_FRACTIONS: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10, 0.20)

#: Cell-suppression floor for published event counts (repository convention).
MIN_CELL_COUNT = 11


def sha256_file(path: Path) -> str:
  """Return the SHA256 hex digest of a file read in bounded chunks."""
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    while True:
      chunk = handle.read(1 << 20)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def load_frozen_probes(
  model_dir: Path,
  baseline_manifest_path: Path | None = None,
) -> dict[str, TrainedProbe]:
  """Load the unchanged frozen MIMIC ECG-only probes with hash verification.

  Fails fast when a probe file is missing, its hash differs from the E0
  baseline manifest (any change is a protocol violation), or its input
  dimensionality does not match the 15 A / 768 B contract.
  """
  model_dir = Path(model_dir)
  baseline: dict[str, Any] | None = None
  if baseline_manifest_path is not None and Path(baseline_manifest_path).exists():
    loaded = json.loads(Path(baseline_manifest_path).read_text(encoding="utf-8"))
    baseline = {name: entry for name, entry in loaded.get("files", {}).items()}

  probes: dict[str, TrainedProbe] = {}
  for model_name, rel in TRANSPORT_MODEL_FILES.items():
    path = model_dir / rel
    if not path.exists():
      raise FileNotFoundError(f"Frozen probe artifact not found: {path}")
    if baseline is not None:
      entry = baseline.get(rel)
      if entry is None or entry.get("sha256") is None:
        raise ValueError(f"E0 baseline is missing a hash for '{rel}'.")
      if sha256_file(path) != entry["sha256"]:
        raise RuntimeError(
          f"Frozen probe file '{path}' changed since the E0 baseline; "
          "transport of modified models is forbidden."
        )
    probe = TrainedProbe.load(path)
    if probe.feature_means is not None and len(probe.feature_means) != EXPECTED_FEATURE_DIMS[model_name]:
      raise ValueError(
        f"Frozen probe '{model_name}' expects {len(probe.feature_means)} features; "
        f"expected {EXPECTED_FEATURE_DIMS[model_name]}."
      )
    if len(probe.coefficients) != EXPECTED_FEATURE_DIMS[model_name]:
      raise ValueError(
        f"Frozen probe '{model_name}' has {len(probe.coefficients)} coefficients; "
        f"expected {EXPECTED_FEATURE_DIMS[model_name]}."
      )
    probes[model_name] = probe
  return probes


def transport_scores(
  probes: dict[str, TrainedProbe],
  features_a: npt.NDArray[np.floating],
  features_b: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
  """Score features with the unchanged frozen probes (A and B)."""
  if features_a.shape[1] != EXPECTED_FEATURE_DIMS["M_A_ECG"]:
    raise ValueError(f"A features must have 15 columns, got {features_a.shape}.")
  if features_b.shape[1] != EXPECTED_FEATURE_DIMS["M_B_ECG"]:
    raise ValueError(f"B features must have 768 columns, got {features_b.shape}.")
  p_a = probes["M_A_ECG"].predict_proba(np.asarray(features_a, dtype=np.float64))
  p_b = probes["M_B_ECG"].predict_proba(np.asarray(features_b, dtype=np.float64))
  return p_a, p_b


def calibration_intercept_slope(
  y: npt.NDArray[np.integer],
  p: npt.NDArray[np.floating],
) -> dict[str, float | None]:
  """Calibration intercept and slope via logistic recalibration of the logit.

  Follows the standard convention: fit ``y ~ a + b * logit(p)``; ``b=1,
  a=0`` is perfect calibration. Returns ``None`` values for degenerate
  single-class samples.
  """
  from sklearn.linear_model import LogisticRegression

  y = np.asarray(y).astype(int)
  if y.min() == y.max():
    return {"intercept": None, "slope": None}
  p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
  logit = np.log(p / (1 - p)).reshape(-1, 1)
  model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
  model.fit(logit, y)
  return {
    "intercept": float(np.asarray(model.intercept_).ravel()[0]),
    "slope": float(np.asarray(model.coef_).ravel()[0]),
  }


def binary_metrics(
  y: npt.NDArray[np.integer],
  p: npt.NDArray[np.floating],
) -> dict[str, float | None]:
  """Discrimination/calibration metrics for one frozen score on one horizon."""
  y = np.asarray(y).astype(int)
  p = np.asarray(p, dtype=np.float64)
  n_events = int(y.sum())
  single_class = y.min() == y.max()
  metrics: dict[str, float | None] = {
    "n": int(len(y)),
    "n_events": n_events,
    "prevalence": float(y.mean()) if len(y) else None,
    "auroc": None if single_class else float(roc_auc_score(y, p)),
    "auprc": float(average_precision_score(y, p)) if n_events > 0 else None,
    "brier": float(np.mean((p - y) ** 2)),
    "log_loss": float(log_loss(y, p, labels=[0, 1])),
  }
  metrics.update(calibration_intercept_slope(y, p))
  return metrics


def paired_bootstrap_delta(
  y: npt.NDArray[np.integer],
  p_a: npt.NDArray[np.floating],
  p_b: npt.NDArray[np.floating],
  n_bootstraps: int = 1000,
  seed: int = 1912,
) -> dict[str, Any]:
  """Patient-paired bootstrap CIs for the B−A AUROC/AUPRC differences."""
  y = np.asarray(y).astype(int)
  rng = np.random.default_rng(seed)
  n = len(y)
  deltas_auroc: list[float] = []
  deltas_auprc: list[float] = []
  skipped = 0
  for _ in range(n_bootstraps):
    idx = rng.integers(0, n, size=n)
    y_b = y[idx]
    if y_b.min() == y_b.max():
      skipped += 1
      continue
    a = roc_auc_score(y_b, p_a[idx])
    b = roc_auc_score(y_b, p_b[idx])
    deltas_auroc.append(b - a)
    deltas_auprc.append(average_precision_score(y_b, p_b[idx]) - average_precision_score(y_b, p_a[idx]))

  def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
      return {"mean": None, "lo95": None, "hi95": None}
    arr = np.asarray(values)
    return {
      "mean": float(arr.mean()),
      "lo95": float(np.percentile(arr, 2.5)),
      "hi95": float(np.percentile(arr, 97.5)),
    }

  return {
    "delta_auroc_b_minus_a": _summary(deltas_auroc),
    "delta_auprc_b_minus_a": _summary(deltas_auprc),
    "n_bootstraps": n_bootstraps,
    "n_skipped_degenerate": skipped,
    "seed": seed,
  }


def fixed_capacity_policies(
  y: npt.NDArray[np.integer],
  p: npt.NDArray[np.floating],
  fractions: tuple[float, ...] = CAPACITY_FRACTIONS,
  min_cell: int = MIN_CELL_COUNT,
) -> list[dict[str, Any]]:
  """Rank-based fixed-capacity review table with small-cell suppression.

  Deaths captured is *event enrichment*, not treatment benefit. Event
  counts below ``min_cell`` are suppressed per repository convention.
  """
  y = np.asarray(y).astype(int)
  p = np.asarray(p, dtype=np.float64)
  n = len(y)
  total_events = int(y.sum())
  order = np.argsort(-p, kind="stable")
  rows: list[dict[str, Any]] = []
  for frac in fractions:
    k = max(1, int(np.ceil(frac * n)))
    flagged = order[:k]
    deaths = int(y[flagged].sum())
    ppv = deaths / k
    row: dict[str, Any] = {
      "capacity_fraction": frac,
      "n_reviewed": k,
      "deaths_captured": deaths if deaths >= min_cell else "suppressed(<11)",
      "sensitivity": round(deaths / total_events, 4) if total_events else None,
      "ppv": round(float(ppv), 4),
      "nne": round(float(1 / ppv), 1) if ppv > 0 else None,
      "total_events": total_events if total_events >= min_cell else "suppressed(<11)",
    }
    rows.append(row)
  return rows


@dataclass(frozen=True)
class TransportTrack:
  """One estimand track (horizon + population) of the transport evaluation."""

  track_name: str
  horizon_label: str
  estimand_label: str
  n: int
  n_events: int
  metrics_a: dict[str, Any]
  metrics_b: dict[str, Any]
  delta: dict[str, Any]
  capacity_a: list[dict[str, Any]]
  capacity_b: list[dict[str, Any]]


def run_transport_track(
  y: npt.NDArray[np.integer],
  p_a: npt.NDArray[np.floating],
  p_b: npt.NDArray[np.floating],
  track_name: str,
  horizon_label: str,
  estimand_label: str,
  n_bootstraps: int = 1000,
  seed: int = 1912,
) -> TransportTrack:
  """Evaluate the frozen transport on one labeled track."""
  y = np.asarray(y).astype(int)
  if y.min() == y.max():
    raise ValueError(
      f"Track '{track_name}' is single-class; cannot evaluate transport."
    )
  return TransportTrack(
    track_name=track_name,
    horizon_label=horizon_label,
    estimand_label=estimand_label,
    n=int(len(y)),
    n_events=int(y.sum()),
    metrics_a=binary_metrics(y, p_a),
    metrics_b=binary_metrics(y, p_b),
    delta=paired_bootstrap_delta(y, p_a, p_b, n_bootstraps=n_bootstraps, seed=seed),
    capacity_a=fixed_capacity_policies(y, p_a),
    capacity_b=fixed_capacity_policies(y, p_b),
  )


def transport_track_to_dict(track: TransportTrack) -> dict[str, Any]:
  """Serialize a transport track (nested dataclasses/dicts) for JSON."""
  return {
    "track_name": track.track_name,
    "horizon_label": track.horizon_label,
    "estimand_label": track.estimand_label,
    "n": track.n,
    "n_events": track.n_events,
    "metrics_a": track.metrics_a,
    "metrics_b": track.metrics_b,
    "delta": track.delta,
    "capacity_a": track.capacity_a,
    "capacity_b": track.capacity_b,
  }


def evaluate_transport(
  features_dir: Path,
  cohort_path: Path,
  model_dir: Path,
  baseline_manifest_path: Path | None = None,
  split_path: Path | None = None,
  n_bootstraps: int = 1000,
  seed: int = 1912,
) -> dict[str, Any]:
  """Join extracted CODE-15% features with labels and run all E3 tracks.

  Per the signed G-E1 dual-track decision:

  - ``full_cohort_1yr``: primary estimand — frozen 30-day-trained scores
      evaluated against the 1-year (365.25-day) endpoint, labeled a
      **different-horizon association**; censored-before-horizon patients
      are excluded with a transparent attrition count.
  - ``full_cohort_30d``: descriptive only (wide-uncertainty caveats).
  - ``locked_test_1yr``: secondary confirmation on the prespecified
      patient-level held-out split (no fitting anywhere in this track).

  Requires both feature matrices to be complete for every meta row (the
  extraction driver excludes failed exams from the saved arrays).
  """
  import polars as pl

  features_dir = Path(features_dir)
  features_a = np.load(features_dir / "ciis_features.npy")
  features_b = np.load(features_dir / "dbeta_embeddings.npy")
  meta = pl.read_parquet(features_dir / "extracted_meta.parquet").sort("exam_id")
  if len(meta) != len(features_a) or len(meta) != len(features_b):
    raise ValueError(
      f"Feature/meta mismatch: meta={len(meta)}, A={len(features_a)}, B={len(features_b)}."
    )

  probes = load_frozen_probes(Path(model_dir), baseline_manifest_path)
  p_a, p_b = transport_scores(probes, features_a, features_b)

  cohort = pl.read_parquet(cohort_path)
  label_cols = [c for c in ("y_30d", "y_365d") if c in cohort.columns]
  missing = {"exam_id", "patient_id", "death", "timey", *label_cols} - set(cohort.columns)
  if missing:
    raise ValueError(f"Cohort parquet is missing columns: {sorted(missing)}.")
  for col in label_cols:
    horizon = col.removeprefix("y_")
    cohort = horizon_label_to_int(cohort, horizon)
  merged = meta.select(["exam_id", "patient_id"]).join(
    cohort.select(["exam_id", "patient_id", "death", "timey", *label_cols]),
    on=["exam_id", "patient_id"],
    how="left",
    validate="1:1",
  )
  n_unmatched = int(merged[label_cols[0]].null_count())
  if n_unmatched:
    raise ValueError(f"{n_unmatched} extracted exams have no cohort label row.")

  tracks: list[dict[str, Any]] = []
  attrition: dict[str, Any] = {}
  ipcw_sensitivity: dict[str, Any] | None = None

  # Full-cohort tracks (primary 1-year; descriptive 30-day).
  full_mask = np.ones(len(meta), dtype=bool)
  for horizon, estimand in (
    ("365d", "different-horizon association (frozen 30d-trained scores vs 1-year endpoint)"),
    ("30d", "descriptive only; wide-uncertainty caveats; no subgroup/cascade claims"),
  ):
    y_all = merged[f"y_{horizon}"].to_list()
    keep_idx = np.asarray([i for i, v in enumerate(y_all) if v is not None], dtype=int)
    y_arr = np.asarray([v for v in y_all if v is not None], dtype=np.int64)
    track_name = f"full_cohort_{horizon}"
    attrition[track_name] = {
      "n_extracted": int(len(meta)),
      "n_labeled_complete": int(len(y_arr)),
      "n_dropped_unknown": int(len(meta) - len(y_arr)),
    }
    tracks.append(
      transport_track_to_dict(
        run_transport_track(
          y_arr, p_a[keep_idx], p_b[keep_idx], track_name, horizon, estimand,
          n_bootstraps=n_bootstraps, seed=seed,
        )
      )
    )

  # Locked-test confirmation track (1-year) when split assignments exist.
  if split_path is not None and Path(split_path).exists():
    split = pl.read_csv(Path(split_path))
    split_map = dict(
      zip(split["patient_id"].to_list(), split["split"].to_list(), strict=True),
    )
    labels_365 = merged["y_365d"].to_list()
    pids = merged["patient_id"].to_list()
    test_assigned_mask = np.asarray(
      [split_map.get(pid) == "test" for pid in pids],
      dtype=bool,
    )
    test_complete_mask = test_assigned_mask & np.asarray(
      [v is not None for v in labels_365],
      dtype=bool,
    )
    y_arr = np.asarray(
      [v for v, keep in zip(labels_365, test_complete_mask, strict=True) if keep],
      dtype=np.int64,
    )
    # The complete-case mask preserves exam order and is distinct from the
    # extracted-test mask so censoring and extraction loss remain visible.
    keep_idx = np.flatnonzero(test_complete_mask)
    p_a_test = p_a[keep_idx]
    p_b_test = p_b[keep_idx]
    track_name = "locked_test_365d"
    n_split_assigned_test = sum(split == "test" for split in split_map.values())
    n_extracted_test = int(test_assigned_mask.sum())
    n_complete_test = int(test_complete_mask.sum())
    attrition[track_name] = {
      "n_split_assigned": int(n_split_assigned_test),
      "n_extracted": n_extracted_test,
      "n_labeled_complete": n_complete_test,
      "n_dropped_unknown": int(n_extracted_test - n_complete_test),
      "n_not_extracted": int(n_split_assigned_test - n_extracted_test),
    }
    tracks.append(
      transport_track_to_dict(
        run_transport_track(
          y_arr, p_a_test, p_b_test, track_name, "365d",
          "secondary confirmation on prespecified held-out patients",
          n_bootstraps=n_bootstraps, seed=seed,
        )
      )
    )

    reference_mask = np.asarray(
      [split_map.get(pid) in ("dev", "val") for pid in pids],
      dtype=bool,
    )
    test_times_days = np.asarray(merged["timey"].to_list(), dtype=np.float64) * DAYS_PER_YEAR
    events = np.asarray(merged["death"].to_list(), dtype=bool)
    from ecg_rep.external.code15_survival import censoring_sensitivity_report

    ipcw_sensitivity = censoring_sensitivity_report(
      reference_times=test_times_days[reference_mask],
      reference_events=events[reference_mask],
      test_times=test_times_days[test_assigned_mask],
      test_events=events[test_assigned_mask],
      risk_scores={"A": p_a[test_assigned_mask], "B": p_b[test_assigned_mask]},
      contrasts={"B_minus_A": ("A", "B")},
      horizon=HORIZON_DAYS["365d"],
      n_bootstraps=n_bootstraps,
      seed=seed,
    )

  return {
    "schema_version": "code15_transport_v2",
    "estimand_note": (
      "Per delegated G-E1: the 1-year track is the primary estimand and is a "
      "different-horizon association for the frozen 30-day-trained MIMIC "
      "scores; it is never same-endpoint external validation. The 30-day "
      "track is descriptive only."
    ),
    "n_extracted": int(len(meta)),
    "attrition": attrition,
    "tracks": tracks,
    "ipcw_sensitivity": ipcw_sensitivity,
  }
