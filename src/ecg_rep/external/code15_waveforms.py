"""E2: CODE-15% waveform reader, calibration, and preflight QC.

Replays the frozen MIMIC representation extractors (15 CIIS features and the
768-dim D-BETA embedding) on CODE-15% waveforms **without fitting anything**.

Locked extraction policy (per the plan, Agent E2):

- CODE-15% HDF5 stores padded ``4096``-sample traces at 400 Hz; 7 s traces
  carry 648 leading + 648 trailing zeros, 10 s traces 48 + 48. The **active
  region is trimmed before any resampling** so Fourier resampling never runs
  over padded zero edges and CIIS morphology is never computed on padding.
- After trimming, the active signal is resampled 400 -> 500 Hz with the
  locked zero-phase pipeline and padding is applied **at the end** to the
  D-BETA target length (5000 samples), matching the frozen MIMIC
  preprocessing exactly.
- CIIS morphology is extracted on the trimmed active signal at 500 Hz (its
  native post-resample length), never on the end padding.
- Amplitude units must pass a sample-scale check before bulk processing;
  the source release documents mV.
- The QC gate is deterministic and development-only; preflight examples are
  sampled without reference to outcomes or risk scores.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ecg_rep.scoring.preprocess import (
  CropAlign,
  WaveformPadMode,
  pad_or_crop_waveform,
  resample_waveform,
)
from ecg_rep.scoring.traditional import (
  CIIS_FEATURE_NAMES,
  extract_ciis_measurements_from_waveform,
)

logger = logging.getLogger(__name__)

#: Fixed lead axis order of the released CODE-15% ``tracings`` dataset.
CODE15_LEAD_ORDER: tuple[str, ...] = (
  "DI", "DII", "DIII", "AVR", "AVL", "AVF",
  "V1", "V2", "V3", "V4", "V5", "V6",
)

#: Mapping from CODE-15% lead axis names to the canonical clinical names
#: used by the frozen MIMIC extractors.
CODE15_LEAD_TO_CANONICAL: dict[str, str] = {
  "DI": "I",
  "DII": "II",
  "DIII": "III",
  "AVR": "aVR",
  "AVL": "aVL",
  "AVF": "aVF",
  "V1": "V1",
  "V2": "V2",
  "V3": "V3",
  "V4": "V4",
  "V5": "V5",
  "V6": "V6",
}

CANONICAL_LEADS_FROM_CODE15: tuple[str, ...] = tuple(
  CODE15_LEAD_TO_CANONICAL[name] for name in CODE15_LEAD_ORDER
)

CODE15_SOURCE_FS = 400
CODE15_PADDED_SAMPLES = 4096
CODE15_NATIVE_DURATIONS_SECONDS: dict[int, int] = {7: 2800, 10: 4000}
#: Symmetric padding sample counts by native duration from the release
#: description. **Empirical note (E2 preflight, 2026-10-01):** real traces
#: overwhelmingly show active lengths of 2934 samples (~7.3 s) with 581+581
#: symmetric padding, or full 4096-sample (~10.2 s) traces with no padding —
#: the release's 2800/648 description is approximate. Active lengths are
#: therefore detected per trace, logged, and never assumed.
CODE15_SYMMETRIC_PAD_SAMPLES: dict[int, int] = {2800: 648, 4000: 48}

D_BETA_TARGET_FS = 500
D_BETA_TARGET_SAMPLES = 5000

#: Acceptable per-lead amplitude range (mV) for the sample-scale check;
#: CODE-15% is documented in mV and typical clinical QRS amplitudes fall
#: well inside this band.
AMPLITUDE_SCALE_PLAUSIBLE_MV = (0.02, 15.0)

HDF5_EXAM_ID_KEY_CANDIDATES = ("exam_id", "id_exam")
HDF5_TRACINGS_KEY_CANDIDATES = ("tracings", "signal")


@dataclass(frozen=True)
class StandardizedCode15Trace:
  """A trimmed/resampled CODE-15% trace plus its active-region audit."""

  active_500hz: npt.NDArray[np.float64]  # (active_samples_500, 12) mV
  padded_5000: npt.NDArray[np.float64]  # (5000, 12) mV, D-BETA input
  active_samples_at_400: int
  pad_start_samples: int
  pad_end_samples: int
  active_duration_seconds: float  # active_samples_at_400 / 400 (empirical)
  padding_symmetric: bool
  active_length_uncertain: bool


def resolve_hdf5_keys(handle: Any) -> tuple[str, str]:
  """Resolve ``(tracings_key, exam_id_key)`` from an open HDF5 file.

  The Zenodo prose and snippet disagree on naming; both candidate families
  are probed and the actual keys are returned (never hard-coded by callers).
  """
  available = set(handle.keys())

  def _find(candidates: tuple[str, ...], what: str) -> str:
    for key in candidates:
      if key in available:
        return key
    raise KeyError(f"Could not resolve {what} dataset; found keys: {sorted(available)}")

  return _find(HDF5_TRACINGS_KEY_CANDIDATES, "tracings"), _find(
    HDF5_EXAM_ID_KEY_CANDIDATES, "exam_id",
  )


def build_exam_id_row_index(partition_path: Path) -> dict[int, int]:
  """Map every ``exam_id`` in an HDF5 partition to its row position.

  The join is defined on the ``exam_id`` dataset values, never on CSV row
  order or positional coincidence.
  """
  import h5py

  partition_path = Path(partition_path)
  if not partition_path.exists():
    raise FileNotFoundError(f"HDF5 partition not found: {partition_path}")
  with h5py.File(partition_path, "r") as handle:
    _, exam_key = resolve_hdf5_keys(handle)
    exam_dataset = handle[exam_key]
    if not isinstance(exam_dataset, h5py.Dataset):
      raise TypeError(f"'{exam_key}' is not an HDF5 dataset in '{partition_path.name}'.")
    exam_ids = exam_dataset[()]
  index: dict[int, int] = {}
  for row, exam_id in enumerate(np.asarray(exam_ids).tolist()):
    exam_id_int = int(exam_id)
    if exam_id_int in index:
      raise ValueError(
        f"Duplicate exam_id {exam_id_int} in partition '{partition_path.name}' "
        "(row-position join would be ambiguous)."
      )
    index[exam_id_int] = row
  return index


def read_exam_trace(partition_path: Path, row: int) -> npt.NDArray[np.float32]:
  """Lazily read one padded ``(4096, 12)`` trace from an HDF5 partition."""
  import h5py

  with h5py.File(partition_path, "r") as handle:
    trace_key, _ = resolve_hdf5_keys(handle)
    trace_dataset = handle[trace_key]
    if not isinstance(trace_dataset, h5py.Dataset):
      raise TypeError(f"'{trace_key}' is not an HDF5 dataset in '{partition_path.name}'.")
    trace = trace_dataset[row, :, :]
  return np.asarray(trace, dtype=np.float32)


def detect_active_region(
  trace: npt.NDArray[np.float32],
  atol: float = 1e-6,
) -> tuple[int, int, bool]:
  """Detect the nonzero active region of a padded ``(4096, 12)`` trace.

  Returns ``(start, end_exclusive, uncertain)`` where ``uncertain`` marks
  traces whose active length does not sit on the released 7 s/10 s grid
  (2800/4000 samples with symmetric padding) — such traces are logged, and
  the detection distinguishes real signal from padding so morphology never
  runs over padding.
  """
  active_mask = np.any(np.abs(trace) > atol, axis=1)
  nonzero = np.flatnonzero(active_mask)
  if len(nonzero) == 0:
    return 0, 0, True
  start, end = int(nonzero[0]), int(nonzero[-1]) + 1
  active_len = end - start
  if active_len in CODE15_SYMMETRIC_PAD_SAMPLES:
    pad = CODE15_SYMMETRIC_PAD_SAMPLES[active_len]
    uncertain = start != pad  # asymmetric padding violates the release spec
    on_release_grid = active_len in set(CODE15_NATIVE_DURATIONS_SECONDS.values())
    return start, end, bool(uncertain) or not on_release_grid
  return start, end, True


def resample_active_region(
  active: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """Resample a trimmed active CODE-15% region from 400 Hz to 500 Hz.

  Uses the locked zero-phase Fourier resampler from the frozen MIMIC
  preprocessing; trimming before this step guarantees resampling never
  spans padded zero edges.
  """
  return resample_waveform(np.asarray(active, dtype=np.float64), CODE15_SOURCE_FS, D_BETA_TARGET_FS)


def standardize_code15_trace(trace: npt.NDArray[np.float32]) -> StandardizedCode15Trace:
  """Trim padding, reorder leads, resample to 500 Hz, and pad to 5000.

  Two views are returned:

  - ``active_500hz``: the trimmed active signal at 500 Hz for CIIS
    morphology (no padding inside).
  - ``padded_5000``: the frozen D-BETA input — active signal resampled to
    500 Hz then end-padded with zeros to the locked 5000-sample length,
    exactly as the MIMIC preprocessing treats native 5000-sample traces.
  """
  trace = np.asarray(trace, dtype=np.float32)
  if trace.shape != (CODE15_PADDED_SAMPLES, len(CODE15_LEAD_ORDER)):
    raise ValueError(
      f"Expected CODE-15% trace shape ({CODE15_PADDED_SAMPLES}, {len(CODE15_LEAD_ORDER)}), "
      f"got {trace.shape}."
    )

  start, end, uncertain = detect_active_region(trace)
  if end <= start:
    raise ValueError("Trace has no detectable active region (all padding).")
  active_400 = trace[start:end, :].astype(np.float64)

  active_500 = resample_active_region(active_400)
  padded = pad_or_crop_waveform(
    active_500,
    target_length=D_BETA_TARGET_SAMPLES,
    pad_mode=WaveformPadMode.CONSTANT,
    crop_align=CropAlign.START,
    fill_value=0.0,
  )

  active_len = end - start
  pad_end = CODE15_PADDED_SAMPLES - end
  symmetric = start == pad_end
  return StandardizedCode15Trace(
    active_500hz=active_500,
    padded_5000=padded,
    active_samples_at_400=active_len,
    pad_start_samples=start,
    pad_end_samples=pad_end,
    active_duration_seconds=round(active_len / CODE15_SOURCE_FS, 4),
    padding_symmetric=symmetric,
    active_length_uncertain=uncertain or not symmetric,
  )


def canonical_leads_for_ciis() -> tuple[str, ...]:
  """Lead names (canonical spelling) attached to the CIIS extraction view."""
  return CANONICAL_LEADS_FROM_CODE15


def extract_ciis_from_standardized(
  standardized: StandardizedCode15Trace,
) -> dict[str, float]:
  """Replay the frozen 15-feature CIIS extractor on the active signal.

  The extractor is never fitted; it runs on the trimmed, resampled active
  region only, with the canonical lead spelling expected by the frozen
  MIMIC implementation.
  """
  measurements = extract_ciis_measurements_from_waveform(
    standardized.active_500hz,
    list(CANONICAL_LEADS_FROM_CODE15),
    D_BETA_TARGET_FS,
  )
  return {name: float(getattr(measurements, name)) for name in CIIS_FEATURE_NAMES}


def sample_amplitude_scale_check(
  traces: npt.NDArray[np.floating],
  atol: float = 1e-6,
) -> dict[str, Any]:
  """Sample-scale check on amplitude units before bulk processing.

  Operates on a small sample of padded ``(4096, 12)`` traces (outcome-free
  selection). Verifies that per-trace peak amplitudes are consistent with
  the documented millivolt unit; implausible scales fail the check.
  """
  peaks: list[float] = []
  for trace in traces:
    start, end, _ = detect_active_region(np.asarray(trace, dtype=np.float32), atol=atol)
    if end > start:
      peaks.append(float(np.max(np.abs(trace[start:end]))))
  if not peaks:
    raise ValueError("No active regions detected in the amplitude sample.")
  p95 = float(np.percentile(peaks, 95))
  lo, hi = AMPLITUDE_SCALE_PLAUSIBLE_MV
  if lo <= p95 <= hi:
    verdict = "plausible_mV"
  elif p95 > hi:
    verdict = "implausibly_large_check_uv_or_gain_error"
  else:
    verdict = "implausibly_small_check_unit_or_scaling"
  return {
    "n_sample": len(peaks),
    "p95_peak_amplitude": p95,
    "max_peak_amplitude": float(np.max(peaks)),
    "min_peak_amplitude": float(np.min(peaks)),
    "verdict": verdict,
    "plausible_band_mV": [lo, hi],
  }


def qc_gate_trace(standardized: StandardizedCode15Trace) -> dict[str, bool]:
  """Deterministic development-only QC gate flags for one trace.

  Flags (True = problem detected):

  - ``flatline``: any lead with near-zero variance in the active region.
  - ``missing_lead``: any all-zero lead in the active region.
  - ``saturation``: repeated clipping at an identical extreme value.
  - ``low_amplitude``: global peak below the plausible clinical floor.
  - ``active_length_uncertain``: active region off the released 7 s/10 s grid.
  """
  active = standardized.active_500hz
  per_lead_std = active.std(axis=0)
  per_lead_max = np.max(np.abs(active), axis=0)
  flatline = bool(np.any(per_lead_std < 1e-4))
  missing_lead = bool(np.any(per_lead_max < 1e-6))
  saturation = False
  for lead in range(active.shape[1]):
    column = active[:, lead]
    peak = float(per_lead_max[lead])
    if peak > 0:
      repeated = float(np.mean(np.abs(column) >= peak - 1e-9))
      if repeated > 0.01:
        saturation = True
  low_amplitude = bool(np.max(per_lead_max) < 0.02)
  return {
    "flatline": flatline,
    "missing_lead": missing_lead,
    "saturation": saturation,
    "low_amplitude": low_amplitude,
    "active_length_uncertain": standardized.active_length_uncertain,
  }


def qc_gate_accept(flags: dict[str, bool]) -> bool:
  """Deterministic acceptance rule.

  ``active_length_uncertain`` is a **reporting** flag only (real CODE-15%
  traces sit off the release's approximate 7 s/10 s description); it does
  not reject a trace. Physiologic QC flags reject.
  """
  physiologic = {k: v for k, v in flags.items() if k != "active_length_uncertain"}
  return not any(physiologic.values())


def features_sha256(features: npt.NDArray[np.floating]) -> str:
  """Stable content hash for a feature matrix (extracted-representation manifest)."""
  return hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()


def save_partition_checkpoint(
  path: Path,
  exam_ids: list[int],
  features_a: npt.NDArray[np.floating],
  features_b: npt.NDArray[np.floating],
) -> None:
  """Write a validated per-partition extraction checkpoint (npz)."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    path,
    exam_ids=np.asarray(exam_ids, dtype=np.int64),
    features_a=features_a,
    features_b=features_b,
  )


def load_partition_checkpoint(
  path: Path,
  n_a: int,
  n_b: int,
) -> tuple[list[int], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
  """Load and validate a per-partition checkpoint; dimension mismatches fail fast."""
  path = Path(path)
  if not path.exists():
    return [], np.empty((0, n_a)), np.empty((0, n_b))
  with np.load(path) as data:
    exam_ids = data["exam_ids"].tolist()
    features_a = data["features_a"]
    features_b = data["features_b"]
    if features_a.shape[1] != n_a or features_b.shape[1] != n_b:
      raise ValueError(
        f"Checkpoint at '{path}' has feature dimensions "
        f"A={features_a.shape[1]}, B={features_b.shape[1]}; expected {n_a}, {n_b}."
      )
    if len(exam_ids) != len(features_a) or len(exam_ids) != len(features_b):
      raise ValueError(f"Checkpoint at '{path}' has ragged arrays.")
  return [int(e) for e in exam_ids], features_a, features_b


def write_extraction_manifest(
  path: Path,
  manifest: dict[str, Any],
) -> Path:
  """Persist the extraction manifest (per-partition failures, hashes, config)."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return path


@dataclass
class ExtractionRecord:
  """Per-exam extraction outcome (features + QC audit)."""

  exam_id: int
  features_a: npt.NDArray[np.float64] | None  # 15 CIIS features
  features_b: npt.NDArray[np.float64] | None  # 768 D-BETA features
  qc_flags: dict[str, bool]
  error: str | None = None


def extract_exams_representations(
  partition_path: Path,
  exam_ids: list[int],
  dbeta_adapter: Any | None = None,
  apply_qc_gate: bool = True,
) -> list[ExtractionRecord]:
  """Extract frozen A (CIIS-15) and B (D-BETA-768) features for exam IDs.

  Exam IDs are joined to HDF5 rows via the ``exam_id`` dataset (never
  positionally). The frozen extractors are replayed without fitting; the
  deterministic QC gate marks (but does not silently drop) invalid exams so
  every denominator is auditable.
  """
  row_index = build_exam_id_row_index(partition_path)
  records: list[ExtractionRecord] = []

  batch: list[tuple[npt.NDArray[np.float64], list[str], int]] = []
  batch_meta: list[int] = []

  def flush() -> None:
    if not batch_meta:
      return
    if dbeta_adapter is not None:
      result = dbeta_adapter.embed_batch(batch)
      embeddings = result.embeddings
      valid = result.valid_mask
    else:
      embeddings = np.empty((len(batch_meta), 0))
      valid = [False] * len(batch_meta)
    for pos, exam_id in enumerate(batch_meta):
      standardized = batch_meta_standardized[pos]
      flags = qc_gate_trace(standardized) if apply_qc_gate else {}
      error = None
      features_a: npt.NDArray[np.float64] | None = None
      features_b: npt.NDArray[np.float64] | None = None
      if apply_qc_gate and not qc_gate_accept(flags):
        error = "qc_gate_failed:" + ",".join(sorted(k for k, v in flags.items() if v))
      else:
        try:
          ciis = extract_ciis_from_standardized(standardized)
          features_a = np.asarray([ciis[name] for name in CIIS_FEATURE_NAMES])
        except Exception as exc:  # noqa: BLE001 - per-exam errors are recorded
          error = f"ciis_error:{exc}"
        if error is None and dbeta_adapter is not None and valid[pos]:
          features_b = np.asarray(embeddings[pos], dtype=np.float64)
          if not np.all(np.isfinite(features_b)):
            error = "dbeta_nonfinite"
            features_b = None
        elif error is None and dbeta_adapter is not None:
          error = "dbeta_invalid"
      records.append(
        ExtractionRecord(
          exam_id=exam_id,
          features_a=features_a,
          features_b=features_b,
          qc_flags=flags,
          error=error,
        ),
      )
    batch.clear()
    batch_meta.clear()
    batch_meta_standardized.clear()

  batch_meta_standardized: list[StandardizedCode15Trace] = []
  for exam_id in exam_ids:
    if exam_id not in row_index:
      records.append(
        ExtractionRecord(
          exam_id=exam_id,
          features_a=None,
          features_b=None,
          qc_flags={},
          error="exam_id_not_in_partition",
        ),
      )
      continue
    try:
      trace = read_exam_trace(partition_path, row_index[exam_id])
      standardized = standardize_code15_trace(trace)
    except Exception as exc:  # noqa: BLE001 - per-exam errors are recorded, never fatal
      records.append(
        ExtractionRecord(
          exam_id=exam_id,
          features_a=None,
          features_b=None,
          qc_flags={},
          error=f"standardize_error:{exc}",
        ),
      )
      continue
    batch.append((standardized.active_500hz, list(CANONICAL_LEADS_FROM_CODE15), D_BETA_TARGET_FS))
    batch_meta.append(exam_id)
    batch_meta_standardized.append(standardized)
    if len(batch_meta) >= 32:
      flush()
  flush()
  return records
