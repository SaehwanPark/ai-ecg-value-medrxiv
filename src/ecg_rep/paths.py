"""Resolve local MIMIC-IV and MIMIC-IV-ECG filesystem roots."""

from __future__ import annotations

import os
from pathlib import Path
import polars as pl

from ecg_rep.data import DataPaths, load_admissions, load_patients, load_record_list
from ecg_rep.elixhauser import load_diagnoses

DEFAULT_MIMIC_ROOT = "~/data/mimiciv/3.1"
DEFAULT_ECG_ROOT = "~/data/mimic-iv-ecg/1.0"
ENV_MIMIC_ROOT = "MIMIC_ROOT"
ENV_ECG_ROOT = "MIMIC_ECG_ROOT"
ENV_ECG_ROOT_ALT = "ECG_ROOT"


def resolve_data_paths(
  mimic_root: str | Path | None = None,
  ecg_root: str | Path | None = None,
) -> DataPaths:
  """Resolve MIMIC and ECG roots with explicit, env, then default precedence.

  Precedence:
    1. Explicit function arguments / CLI flags.
    2. Environment variables (`MIMIC_ROOT`, `MIMIC_ECG_ROOT` / `ECG_ROOT`).
    3. Local defaults under `~/data/`.
  """
  resolved_mimic = (
    mimic_root
    if mimic_root is not None
    else os.environ.get(ENV_MIMIC_ROOT, DEFAULT_MIMIC_ROOT)
  )
  resolved_ecg = (
    ecg_root
    if ecg_root is not None
    else os.environ.get(ENV_ECG_ROOT, os.environ.get(ENV_ECG_ROOT_ALT, DEFAULT_ECG_ROOT))
  )

  return DataPaths(
    mimic_iv_dir=Path(str(resolved_mimic)).expanduser().resolve(),
    mimic_iv_ecg_dir=Path(str(resolved_ecg)).expanduser().resolve(),
  )


def _find_data_file(base_dir: Path, relative_candidates: list[str]) -> Path:
  """Return the first existing candidate path under ``base_dir``."""
  for rel in relative_candidates:
    candidate = base_dir / rel
    if candidate.exists():
      return candidate
  return base_dir / relative_candidates[0]


def load_dataset_tables(paths: DataPaths) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
  """Load record list, patients, and admissions for cohort construction."""
  record_path = _find_data_file(
    paths.mimic_iv_ecg_dir,
    ["record_list.csv", "files/record_list.csv"],
  )
  patients_path = _find_data_file(
    paths.mimic_iv_dir,
    ["hosp/patients.csv.gz", "hosp/patients.csv", "patients.csv.gz", "patients.csv"],
  )
  admissions_path = _find_data_file(
    paths.mimic_iv_dir,
    [
      "hosp/admissions.csv.gz",
      "hosp/admissions.csv",
      "admissions.csv.gz",
      "admissions.csv",
    ],
  )

  if not record_path.exists():
    raise FileNotFoundError(f"Record list not found at: {record_path}")
  if not patients_path.exists():
    raise FileNotFoundError(f"Patients table not found at: {patients_path}")
  if not admissions_path.exists():
    raise FileNotFoundError(f"Admissions table not found at: {admissions_path}")

  return (
    load_record_list(record_path),
    load_patients(patients_path),
    load_admissions(admissions_path),
  )


def load_diagnoses_table(paths: DataPaths) -> pl.DataFrame:
  """Load MIMIC-IV hospital diagnoses for pre-index comorbidity construction."""
  diagnoses_path = _find_data_file(
    paths.mimic_iv_dir,
    [
      "hosp/diagnoses_icd.csv.gz",
      "hosp/diagnoses_icd.csv",
      "diagnoses_icd.csv.gz",
      "diagnoses_icd.csv",
    ],
  )
  if not diagnoses_path.exists():
    raise FileNotFoundError(f"Diagnoses table not found at: {diagnoses_path}")
  return load_diagnoses(diagnoses_path)
