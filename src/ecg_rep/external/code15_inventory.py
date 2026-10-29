"""E0: Provenance, manifest, and noninvasive baseline audit for CODE-15%.

This module inventories the external CODE-15% source at ``~/data/code15pct/``
and the canonical frozen MIMIC probe artifacts **without reading waveform
payloads**. It never writes into the source directory and never writes into
the canonical MIMIC ``artifacts/probe`` tree.

Governance notes (see ``docs/plan_20261001b/`` plan, Agent E0):

- ``normal_ecg`` is a machine annotation from a different study; it is never
  waveform quality control.
- ``nn_predicted_age`` is an external learned-model feature from another
  study; it is never a baseline predictor in this study.
- ``death``/``timey`` are annotated on the first examination per patient only.
  This module documents their parsed dtypes/nullness but performs no endpoint
  modeling (that is E1's job, behind gate G-E1).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from ecg_rep.representations import CIIS_FEATURE_NAMES, CIIS_FEATURE_SCHEMA_HASH
from ecg_rep.scoring.dbeta import DEFAULT_DBETA_MODEL_NAME, DEFAULT_DBETA_REVISION

logger = logging.getLogger(__name__)

CODE15_SOURCE_VERSION = "CODE-15% v1.0.0 (Zenodo record 4916206)"
CODE15_EXPECTED_PARTITION_COUNT = 18
CODE15_METADATA_CSV_NAME = "exams.csv"
CODE15_PARTITION_GLOB = "exams_part*.hdf5"
CODE15_ZIPPED_PARTITION_GLOB = "exams_part*.hdf5.zip"

#: exams.csv columns documented by the official CODE-15% release description.
CODE15_EXAMS_COLUMNS = (
  "exam_id",
  "age",
  "is_male",
  "nn_predicted_age",
  "1dAVb",
  "RBBB",
  "LBBB",
  "SB",
  "ST",
  "AF",
  "patient_id",
  "death",
  "timey",
  "normal_ecg",
  "trace_file",
)

#: Aggregated-only column semantics. No patient-level values may be committed.
CODE15_COLUMN_SEMANTICS: dict[str, str] = {
  "exam_id": "Unique examination identifier; joins to HDF5 exam_id dataset.",
  "age": "Patient age in years at the moment of the exam.",
  "is_male": "Recorded sex indicator (True if male).",
  "nn_predicted_age": (
    "Age predicted by an external neural network from a different study. "
    "FORBIDDEN as a baseline predictor in this study."
  ),
  "1dAVb": "First-degree AV block annotation from the original study.",
  "RBBB": "Right bundle branch block annotation.",
  "LBBB": "Left bundle branch block annotation.",
  "SB": "Sinus bradycardia annotation.",
  "ST": "Sinus tachycardia annotation.",
  "AF": "Atrial fibrillation annotation.",
  "patient_id": "Patient identifier; defines the patient-level split boundary.",
  "death": "Death in follow-up; available on the first exam per patient only.",
  "timey": (
    "Time to death if died, otherwise follow-up time; first exam per patient "
    "only. Unit and precision MUST be verified in E1 before any endpoint use."
  ),
  "normal_ecg": (
    "Automatic annotation system verdict from the original study. FORBIDDEN "
    "as waveform quality control in this study."
  ),
  "trace_file": "HDF5 partition file that holds this exam's tracing.",
}

#: Waveform release description from the official CODE-15% documentation.
CODE15_WAVEFORM_SPEC = {
  "n_leads": 12,
  "lead_order": ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"],
  "sampling_rate_hz": 400,
  "n_samples_padded": 4096,
  "native_durations_seconds": [7, 10],
  "padding": "Symmetric zero padding on both sides (e.g. 648+648 for 7 s traces).",
}

#: Canonical MIMIC fitted-probe files that must remain byte-identical.
MIMIC_CANONICAL_PROBE_FILES = (
  "models/M_A_ECG.json",
  "models/M_A_ECG.joblib",
  "models/M_B_ECG.json",
  "models/M_B_ECG.joblib",
  "models/M_AB.json",
  "models/M_AB.joblib",
  "models/M_B_PCA15.json",
  "models/M_B_PCA15.joblib",
  "models/M_C.json",
  "models/M_C.joblib",
  "probe_results.json",
  "probe_results_context.json",
  "test_predictions_ecg_only.parquet",
  "test_predictions_context.parquet",
)

_REPO_MANIFEST_FIELDS = (
  "source_dataset_version",
  "source_repository_sha",
  "code_revision",
  "schema_hash",
  "dbeta_revision",
)


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
  """Return the SHA256 hex digest of a file read in bounded chunks."""
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    while True:
      chunk = handle.read(chunk_bytes)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


@dataclass(frozen=True)
class FileRecord:
  """One inventoried file on disk with its size and optional checksum."""

  name: str
  size_bytes: int
  sha256: str | None = None


def discover_source_files(data_dir: Path) -> dict[str, Any]:
  """Recursively inventory ``data_dir`` without reading waveform contents.

  Returns file names/sizes, duplicate-name detection, and which expected
  partition files are present or missing (zipped or extracted accepted).
  """
  data_dir = Path(data_dir).expanduser().resolve()
  if not data_dir.is_dir():
    raise FileNotFoundError(f"CODE-15% data directory not found: {data_dir}")

  all_files = sorted(p for p in data_dir.rglob("*") if p.is_file())
  records: list[FileRecord] = []
  for path in all_files:
    rel = path.relative_to(data_dir).as_posix()
    records.append(FileRecord(name=rel, size_bytes=path.stat().st_size))

  names = [rec.name for rec in records]
  duplicates = sorted({n for n in names if names.count(n) > 1})

  partition_names = {
    rec.name.split("/")[-1].removesuffix(".zip") for rec in records
  }
  missing_partitions = [
    f"exams_part{i}.hdf5"
    for i in range(CODE15_EXPECTED_PARTITION_COUNT)
    if f"exams_part{i}.hdf5" not in partition_names
  ]
  unexpected_hdf5 = sorted(
    rec.name
    for rec in records
    if rec.name.endswith((".hdf5", ".hdf5.zip", ".h5"))
    and rec.name.split("/")[-1].removesuffix(".zip") not in {
      f"exams_part{i}.hdf5" for i in range(CODE15_EXPECTED_PARTITION_COUNT)
    }
  )

  return {
    "data_dir": str(data_dir),
    "n_files": len(records),
    "files": [
      {"name": rec.name, "size_bytes": rec.size_bytes, "sha256": rec.sha256}
      for rec in records
    ],
    "duplicate_names": duplicates,
    "missing_partitions": missing_partitions,
    "unexpected_hdf5_files": unexpected_hdf5,
    "metadata_csv_present": any(
      rec.name.split("/")[-1] == CODE15_METADATA_CSV_NAME for rec in records
    ),
  }


def summarize_exams_schema(csv_path: Path) -> dict[str, Any]:
  """Parse ``exams.csv`` and document actual dtypes, nulls, and row counts.

  Only aggregate schema information is returned; no patient-level values.
  """
  csv_path = Path(csv_path)
  if not csv_path.exists():
    raise FileNotFoundError(f"CODE-15% metadata CSV not found: {csv_path}")
  df = pl.read_csv(csv_path, infer_schema_length=10_000, try_parse_dates=False)

  columns: dict[str, Any] = {}
  for col in df.columns:
    series = df[col]
    columns[col] = {
      "dtype": str(series.dtype),
      "n_null": int(series.null_count()),
      "n_unique": int(series.n_unique()),
    }

  missing_expected = [c for c in CODE15_EXAMS_COLUMNS if c not in df.columns]
  extra_columns = [c for c in df.columns if c not in CODE15_EXAMS_COLUMNS]
  return {
    "csv_path": str(csv_path),
    "n_rows": int(df.height),
    "n_columns": int(df.width),
    "columns": columns,
    "missing_expected_columns": missing_expected,
    "extra_columns": extra_columns,
  }


def inspect_hdf5_partition(path: Path) -> dict[str, Any]:
  """Inspect HDF5 dataset keys/shapes/dtypes without reading tracing data.

  The Zenodo prose and its illustrative snippet disagree on dataset names
  (``tracings``/``exam_id`` vs ``signal``/``id_exam``); the actual keys are
  discovered here and never hard-coded.
  """
  import h5py  # Local import keeps h5py optional for MIMIC-only workflows.

  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"HDF5 partition not found: {path}")

  datasets: dict[str, Any] = {}
  with h5py.File(path, "r") as handle:
    def _visit(name: str, obj: Any) -> None:
      if isinstance(obj, h5py.Dataset):
        datasets[name] = {
          "shape": list(obj.shape),
          "dtype": str(obj.dtype),
          "n_chunks": int(obj.id.get_num_chunks()) if obj.chunks else 0,
        }

    handle.visititems(_visit)

  if not datasets:
    raise ValueError(f"No HDF5 datasets found in {path}")
  return {
    "path": str(path),
    "datasets": datasets,
  }


def build_partition_key_map(data_dir: Path) -> dict[str, Any]:
  """Map every expected partition to its HDF5 dataset keys and shapes."""
  data_dir = Path(data_dir).expanduser().resolve()
  partitions: dict[str, Any] = {}
  errors: dict[str, str] = {}
  for i in range(CODE15_EXPECTED_PARTITION_COUNT):
    name = f"exams_part{i}.hdf5"
    path = data_dir / name
    if not path.exists():
      path = data_dir / f"{name}.zip"
    if not path.exists():
      errors[name] = "missing"
      continue
    try:
      partitions[name] = inspect_hdf5_partition(path)
    except Exception as exc:  # noqa: BLE001 - reported, never silently skipped
      errors[name] = f"{type(exc).__name__}: {exc}"
  return {"partitions": partitions, "errors": errors}


def inventory_mimic_probe_artifacts(
  probe_dir: Path,
  compute_checksums: bool = True,
) -> dict[str, Any]:
  """Enumerate canonical frozen MIMIC fitted-probe files with checksums.

  These hashes form the E0 baseline proving that no canonical MIMIC artifact
  changes during external-validation work.
  """
  probe_dir = Path(probe_dir)
  files: dict[str, Any] = {}
  missing: list[str] = []
  for rel in MIMIC_CANONICAL_PROBE_FILES:
    path = probe_dir / rel
    if not path.exists():
      missing.append(rel)
      continue
    files[rel] = {
      "size_bytes": path.stat().st_size,
      "sha256": sha256_file(path) if compute_checksums else None,
    }

  preflight_path = Path("artifacts/reps/ciis_preflight_acceptance_report.json")
  preflight_status = None
  if preflight_path.exists():
    preflight_status = json.loads(preflight_path.read_text(encoding="utf-8")).get("status")

  return {
    "probe_dir": str(probe_dir),
    "files": files,
    "missing_files": missing,
    "ciis_feature_count": len(CIIS_FEATURE_NAMES),
    "ciis_feature_schema_hash": CIIS_FEATURE_SCHEMA_HASH,
    "ciis_preflight_status": preflight_status,
    "dbeta_model_name": DEFAULT_DBETA_MODEL_NAME,
    "dbeta_revision": DEFAULT_DBETA_REVISION,
  }


def build_code15_source_manifest(
  data_dir: Path,
  compute_checksums: bool = True,
  inspect_keys: bool = True,
  progress_every: int | None = None,
) -> dict[str, Any]:
  """Build the immutable machine-readable CODE-15% source manifest.

  The manifest records source dataset version/checksums, repository state,
  exams.csv schema aggregates, and the HDF5 key/index map. It contains no
  patient-level values and is safe to commit.
  """
  import subprocess

  code_revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True,
  ).strip()

  inventory = discover_source_files(data_dir)
  if compute_checksums:
    total = len(inventory["files"])
    for idx, entry in enumerate(inventory["files"], start=1):
      path = Path(inventory["data_dir"]) / entry["name"]
      entry["sha256"] = sha256_file(path)
      if progress_every and (idx % progress_every == 0 or idx == total):
        logger.info("Checksummed %d/%d source files", idx, total)

  csv_path = Path(inventory["data_dir"]) / CODE15_METADATA_CSV_NAME
  schema = summarize_exams_schema(csv_path) if inventory["metadata_csv_present"] else None

  key_map = build_partition_key_map(data_dir) if inspect_keys else None

  return {
    "manifest_schema_version": "code15_source_manifest_v1",
    "source_dataset_version": CODE15_SOURCE_VERSION,
    "source_repository_sha": code_revision,
    "code_revision": code_revision,
    "schema_hash": CIIS_FEATURE_SCHEMA_HASH,
    "dbeta_model_name": DEFAULT_DBETA_MODEL_NAME,
    "dbeta_revision": DEFAULT_DBETA_REVISION,
    "waveform_spec": CODE15_WAVEFORM_SPEC,
    "column_semantics": CODE15_COLUMN_SEMANTICS,
    "inventory": inventory,
    "exams_schema": schema,
    "hdf5_key_map": key_map,
    "governance_forbidden_features": ["nn_predicted_age", "normal_ecg"],
  }


def write_manifest(manifest: dict[str, Any], path: Path, force: bool = False) -> Path:
  """Write a manifest JSON atomically, refusing to overwrite unless forced.

  Source manifests are treated as immutable provenance records: a rerun must
  either match the existing file or be written to a new location.
  """
  path = Path(path)
  if path.exists() and not force:
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != manifest:
      raise RuntimeError(
        f"Refusing to overwrite existing manifest at '{path}' with different "
        "content. Source manifests are immutable; pass force=True only after "
        "documenting a deviation."
      )
    return path
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  tmp.replace(path)
  return path


def load_manifest(path: Path) -> dict[str, Any]:
  """Load and minimally validate a previously written source manifest."""
  path = Path(path)
  if not path.exists():
    raise FileNotFoundError(f"Manifest not found: {path}")
  manifest = json.loads(path.read_text(encoding="utf-8"))
  for field in _REPO_MANIFEST_FIELDS:
    if field not in manifest:
      raise ValueError(f"Manifest at '{path}' is missing required field '{field}'")
  return manifest


def verify_source_against_manifest(manifest: dict[str, Any], data_dir: Path) -> dict[str, Any]:
  """Verify that the source directory still matches an immutable manifest.

  Compares file names and sizes (cheap) and optionally checksums. Never
  writes anything and never reads waveform payloads.
  """
  data_dir = Path(data_dir).expanduser().resolve()
  current = discover_source_files(data_dir)
  mismatches: list[str] = []

  recorded = {
    entry["name"]: entry for entry in manifest["inventory"]["files"] if entry.get("sha256")
  }
  for entry in current["files"]:
    prior = recorded.get(entry["name"])
    if prior is None:
      mismatches.append(f"new file: {entry['name']}")
    elif prior["size_bytes"] != entry["size_bytes"]:
      mismatches.append(f"size changed: {entry['name']}")
  for name in recorded:
    if not (data_dir / name).exists():
      mismatches.append(f"missing file: {name}")

  return {
    "match": not mismatches,
    "mismatches": mismatches,
    "n_files_current": current["n_files"],
    "n_files_manifest": len(recorded),
  }
