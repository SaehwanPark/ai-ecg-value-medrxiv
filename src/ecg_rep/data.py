"""Data loading and waveform inspection for MIMIC-IV-ECG and MIMIC-IV datasets."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import numpy as np
import numpy.typing as npt
import polars as pl
import wfdb

CANONICAL_12_LEADS: tuple[str, ...] = (
  "I",
  "II",
  "III",
  "aVR",
  "aVF",
  "aVL",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
)

DEFAULT_SAMPLING_RATE_HZ: int = 500
DEFAULT_SIGNAL_LENGTH_SAMPLES: int = 5000
DEFAULT_DURATION_SECONDS: float = 10.0


@dataclass(frozen=True)
class DataPaths:
  """Immutable configuration for dataset filesystem locations."""

  mimic_iv_ecg_dir: Path = Path.home() / "data" / "mimic-iv-ecg" / "1.0"
  mimic_iv_dir: Path = Path.home() / "data" / "mimiciv" / "3.1"

  @property
  def record_list_path(self) -> Path:
    return self.mimic_iv_ecg_dir / "record_list.csv"

  @property
  def machine_measurements_path(self) -> Path:
    return self.mimic_iv_ecg_dir / "machine_measurements.csv"

  @property
  def patients_path(self) -> Path:
    return self.mimic_iv_dir / "hosp" / "patients.csv.gz"

  @property
  def admissions_path(self) -> Path:
    return self.mimic_iv_dir / "hosp" / "admissions.csv.gz"

  @property
  def diagnoses_path(self) -> Path:
    return self.mimic_iv_dir / "hosp" / "diagnoses_icd.csv.gz"


@dataclass(frozen=True)
class EcgHeaderInfo:
  """Metadata extracted from a WFDB header."""

  record_name: str
  n_sig: int
  fs: int
  sig_len: int
  duration_seconds: float
  sig_name: tuple[str, ...]
  units: tuple[str, ...]
  adc_gain: tuple[float, ...]
  baseline: tuple[int, ...]
  comments: tuple[str, ...]


@dataclass(frozen=True)
class WaveformEligibilityCriteria:
  """Criteria defining technical validity for an ECG waveform."""

  required_leads: tuple[str, ...] = CANONICAL_12_LEADS
  expected_fs: int = DEFAULT_SAMPLING_RATE_HZ
  expected_sig_len: int = DEFAULT_SIGNAL_LENGTH_SAMPLES
  expected_units: str = "mV"


def _clean_record_path(record_path: Path | str) -> str:
  """Strip .hea or .dat suffix from record path for WFDB compatibility."""
  p = str(record_path)
  if p.endswith(".hea"):
    return p[:-4]
  if p.endswith(".dat"):
    return p[:-4]
  return p


def read_ecg_header(record_path: Path | str) -> EcgHeaderInfo:
  """Read header metadata for a single WFDB ECG record.

  Args:
    record_path: Path to the WFDB record (with or without .hea suffix).

  Returns:
    EcgHeaderInfo dataclass containing parsed signal metadata.
  """
  clean_path = _clean_record_path(record_path)
  hea_file = Path(f"{clean_path}.hea")
  if hea_file.exists():
    try:
      with open(hea_file, "r", encoding="ascii", errors="ignore") as f:
        lines = f.readlines()
      if lines:
        parts = lines[0].split()
        if len(parts) >= 4:
          rec_name = parts[0]
          n_sig = int(parts[1])
          fs = int(parts[2])
          sig_len = int(parts[3])
          duration = sig_len / fs if fs > 0 else 0.0
          sig_name_list: list[str] = []
          units_list: list[str] = []
          adc_gain_list: list[float] = []
          baseline_list: list[int] = []
          for l in lines[1 : 1 + n_sig]:
            sp = l.split()
            if len(sp) >= 9:
              gain_units = sp[2]
              u = gain_units.split("/")[-1] if "/" in gain_units else ""
              units_list.append(u)
              sig_name_list.append(sp[8])
              try:
                g_str = gain_units.split("(")[0].split("/")[0]
                adc_gain_list.append(float(g_str))
              except Exception:
                adc_gain_list.append(200.0)
              try:
                baseline_list.append(int(sp[4]))
              except Exception:
                baseline_list.append(0)
          comments_list: list[str] = [
            l.strip()[1:].strip() for l in lines[1 + n_sig :] if l.strip().startswith("#")
          ]
          if len(sig_name_list) == n_sig:
            return EcgHeaderInfo(
              record_name=rec_name,
              n_sig=n_sig,
              fs=fs,
              sig_len=sig_len,
              duration_seconds=duration,
              sig_name=tuple(sig_name_list),
              units=tuple(units_list),
              adc_gain=tuple(adc_gain_list),
              baseline=tuple(baseline_list),
              comments=tuple(comments_list),
            )
    except Exception:
      pass

  header = wfdb.rdheader(clean_path)
  fs = int(header.fs) if header.fs is not None else 0
  sig_len = int(header.sig_len) if header.sig_len is not None else 0
  duration = sig_len / fs if fs > 0 else 0.0

  sig_name_list = [str(n) for n in (header.sig_name or [])]
  units_list = [str(u) for u in (getattr(header, "units", None) or [])]
  adc_gain_list = [float(g) for g in (getattr(header, "adc_gain", None) or [])]
  baseline_list = [int(b) for b in (getattr(header, "baseline", None) or [])]
  comments_list = [str(c) for c in (header.comments or [])]

  return EcgHeaderInfo(
    record_name=str(header.record_name),
    n_sig=int(header.n_sig) if header.n_sig is not None else 0,
    fs=fs,
    sig_len=sig_len,
    duration_seconds=duration,
    sig_name=tuple(sig_name_list),
    units=tuple(units_list),
    adc_gain=tuple(adc_gain_list),
    baseline=tuple(baseline_list),
    comments=tuple(comments_list),
  )


def read_ecg_waveform(
  record_path: Path | str,
) -> tuple[npt.NDArray[np.float64], list[str], int]:
  """Read waveform signals and channel names for a single WFDB record.

  Args:
    record_path: Path to the WFDB record.

  Returns:
    Tuple of (signal_array [sig_len, n_sig], lead_names, sampling_rate_hz).
  """
  clean_path = _clean_record_path(record_path)
  signal_raw, fields = wfdb.rdsamp(clean_path)
  if signal_raw is None:
    raise FileNotFoundError(f"Could not load signal for record {record_path}")
  signal_array = np.asarray(signal_raw, dtype=np.float64)
  fields_dict: dict[str, Any] = cast(dict[str, Any], fields)
  lead_names: list[str] = [str(name) for name in fields_dict.get("sig_name", [])]
  fs: int = int(fields_dict.get("fs", DEFAULT_SAMPLING_RATE_HZ))
  return signal_array, lead_names, fs


def validate_waveform_eligibility(
  header: EcgHeaderInfo,
  criteria: WaveformEligibilityCriteria | None = None,
) -> tuple[bool, str | None]:
  """Check whether an ECG record meets technical waveform eligibility criteria.

  Args:
    header: Parsed WFDB header info.
    criteria: Custom criteria; defaults to standard 12-lead 500 Hz 10s.

  Returns:
    Tuple of (is_eligible, failure_reason). failure_reason is None if eligible.
  """
  crit = criteria or WaveformEligibilityCriteria()

  if header.fs != crit.expected_fs:
    return False, f"Unexpected sampling rate: {header.fs} Hz (expected {crit.expected_fs} Hz)"

  if header.sig_len != crit.expected_sig_len:
    return (
      False,
      f"Unexpected signal length: {header.sig_len} samples (expected {crit.expected_sig_len} samples)",
    )

  missing_leads = [lead for lead in crit.required_leads if lead not in header.sig_name]
  if missing_leads:
    return False, f"Missing required leads: {missing_leads}"

  if crit.expected_units:
    invalid_units = [u for u in header.units if u != crit.expected_units]
    if invalid_units:
      return False, f"Non-standard signal units found: {set(invalid_units)}"

  return True, None


def load_record_list(path: Path | str) -> pl.DataFrame:
  """Load and type-cast MIMIC-IV-ECG record_list.csv.

  Args:
    path: Path to record_list.csv.

  Returns:
    DataFrame with columns: subject_id (i64), study_id (i64), file_name (i64),
    ecg_time (str), path (str), ecg_datetime (Datetime), ecg_year (i64).
  """
  df = pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "study_id": pl.Int64,
      "file_name": pl.Int64,
      "ecg_time": pl.String,
      "path": pl.String,
    },
  )
  return df.with_columns(
    pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ecg_datetime"),
    pl.col("ecg_time").str.slice(0, 4).cast(pl.Int64).alias("ecg_year"),
  )


def load_machine_measurements(path: Path | str) -> pl.DataFrame:
  """Load and parse MIMIC-IV-ECG machine_measurements.csv.

  Args:
    path: Path to machine_measurements.csv.

  Returns:
    DataFrame with machine measurements and parsed ecg_datetime.
  """
  df = pl.read_csv(
    path,
    infer_schema_length=10000,
    schema_overrides={
      "subject_id": pl.Int64,
      "study_id": pl.Int64,
      "cart_id": pl.Int64,
      "ecg_time": pl.String,
      "rr_interval": pl.Int64,
      "p_onset": pl.Int64,
      "p_end": pl.Int64,
      "qrs_onset": pl.Int64,
      "qrs_end": pl.Int64,
      "t_end": pl.Int64,
      "p_axis": pl.Int64,
      "qrs_axis": pl.Int64,
      "t_axis": pl.Int64,
    },
  )
  return df.with_columns(
    pl.col("ecg_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("ecg_datetime")
  )


def load_patients(path: Path | str) -> pl.DataFrame:
  """Load and parse MIMIC-IV hosp/patients.csv.gz.

  Args:
    path: Path to patients.csv.gz or uncompressed patients.csv.

  Returns:
    DataFrame with columns: subject_id (i64), gender (str), anchor_age (i64),
    anchor_year (i64), anchor_year_group (str), dod (Date).
  """
  df = pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "gender": pl.String,
      "anchor_age": pl.Int64,
      "anchor_year": pl.Int64,
      "anchor_year_group": pl.String,
      "dod": pl.String,
    },
  )
  return df.with_columns(
    pl.col("dod").str.to_date("%Y-%m-%d", strict=False).alias("dod_date")
  )


def load_admissions(path: Path | str) -> pl.DataFrame:
  """Load and parse MIMIC-IV hosp/admissions.csv.gz.

  Args:
    path: Path to admissions.csv.gz or uncompressed admissions.csv.

  Returns:
    DataFrame with typed identifiers, timestamps, and encounter metadata.
  """
  df = pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "hadm_id": pl.Int64,
      "admittime": pl.String,
      "dischtime": pl.String,
      "deathtime": pl.String,
      "admission_type": pl.String,
      "admission_location": pl.String,
      "discharge_location": pl.String,
      "insurance": pl.String,
      "language": pl.String,
      "marital_status": pl.String,
      "race": pl.String,
      "edregtime": pl.String,
      "edouttime": pl.String,
      "hospital_expire_flag": pl.Int64,
    },
  )
  return df.with_columns(
    pl.col("admittime").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("admit_dt"),
    pl.col("dischtime").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("disch_dt"),
    pl.col("deathtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("death_dt"),
    pl.col("edregtime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("edreg_dt"),
    pl.col("edouttime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("edout_dt"),
  )


def compute_criteria_hash(crit: WaveformEligibilityCriteria) -> str:
  """Deterministic SHA-256 fingerprint for waveform eligibility criteria."""
  payload = json.dumps(
    {
      "expected_fs": crit.expected_fs,
      "expected_sig_len": crit.expected_sig_len,
      "expected_units": crit.expected_units,
      "required_leads": sorted(list(crit.required_leads)),
    },
    sort_keys=True,
  ).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def compute_record_list_hash(records_df: pl.DataFrame) -> str:
  """Deterministic SHA-256 fingerprint of the source record list."""
  cols = [c for c in ("study_id", "path") if c in records_df.columns]
  if not cols or "study_id" not in cols:
    return ""
  sorted_df = records_df.select(cols).sort("study_id")
  payload = b"".join(
    f"{r['study_id']}:{r.get('path', '')}\n".encode("utf-8")
    for r in sorted_df.iter_rows(named=True)
  )
  return hashlib.sha256(payload).hexdigest()


def build_waveform_eligibility_manifest(
  records_df: pl.DataFrame,
  ecg_data_dir: Path | str | None = None,
  criteria: WaveformEligibilityCriteria | None = None,
  relative_path_col: str = "path",
  progress_every: int = 5000,
  is_complete: bool = True,
  n_source_records: int | None = None,
  provenance_output_path: Path | str | None = None,
) -> pl.DataFrame:
  """Build a deterministic, representation-independent technical waveform eligibility manifest.

  Inspects WFDB headers to verify technical signal eligibility (12 leads, 500 Hz, 5000 samples, mV units)
  prior to index ECG selection.
  """
  crit = criteria or WaveformEligibilityCriteria()
  data_dir = Path(ecg_data_dir) if ecg_data_dir is not None else None

  study_ids: list[int] = []
  subject_ids: list[int | None] = []
  eligible_flags: list[bool] = []
  failure_reasons: list[str | None] = []
  fs_list: list[int | None] = []
  sig_len_list: list[int | None] = []
  n_leads_list: list[int | None] = []
  units_valid_list: list[bool | None] = []

  n_records = len(records_df)
  rows = list(records_df.iter_rows(named=True))

  for idx, row in enumerate(rows):
    sid = int(row["study_id"])
    sub_id = int(row["subject_id"]) if "subject_id" in row and row["subject_id"] is not None else None
    rel_path = str(row.get(relative_path_col, ""))

    rec_path = data_dir / rel_path if data_dir is not None else Path(rel_path)

    study_ids.append(sid)
    subject_ids.append(sub_id)

    # Fast-path if records_df already contains pre-annotated waveform eligibility
    if "waveform_eligible" in row and row["waveform_eligible"] is not None:
      is_el = bool(row["waveform_eligible"])
      eligible_flags.append(is_el)
      failure_reasons.append(str(row.get("failure_reason")) if not is_el and "failure_reason" in row else None)
      fs_list.append(int(row.get("fs", crit.expected_fs)) if is_el else None)
      sig_len_list.append(int(row.get("sig_len", crit.expected_sig_len)) if is_el else None)
      n_leads_list.append(len(crit.required_leads) if is_el else None)
      units_valid_list.append(True if is_el else None)
      continue

    try:
      header = read_ecg_header(rec_path)
      is_el, reason = validate_waveform_eligibility(header, crit)
      eligible_flags.append(is_el)
      failure_reasons.append(reason)
      fs_list.append(header.fs)
      sig_len_list.append(header.sig_len)
      n_leads_list.append(header.n_sig)
      units_valid_list.append(
        all(u == crit.expected_units for u in header.units) if crit.expected_units and header.units else None
      )
    except Exception as exc:  # noqa: BLE001
      eligible_flags.append(False)
      failure_reasons.append(f"Header read error: {exc}")
      fs_list.append(None)
      sig_len_list.append(None)
      n_leads_list.append(None)
      units_valid_list.append(None)

    done = idx + 1
    if progress_every > 0 and (done % progress_every == 0 or done == n_records):
      print(f"[Manifest] Evaluated {done}/{n_records} ({100.0 * done / n_records:.1f}%) headers", flush=True)

  manifest_df = pl.DataFrame({
    "study_id": study_ids,
    "subject_id": subject_ids,
    "waveform_eligible": eligible_flags,
    "failure_reason": failure_reasons,
    "fs": fs_list,
    "sig_len": sig_len_list,
    "n_leads": n_leads_list,
    "units_valid": units_valid_list,
  })

  if provenance_output_path is not None:
    prov_path = Path(provenance_output_path)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_dict = {
      "n_source_records": n_source_records if n_source_records is not None else n_records,
      "n_manifest_records": n_records,
      "n_eligible_records": int(sum(eligible_flags)),
      "complete": is_complete,
      "dataset_version": "1.0",
      "code_version": "0.1.0",
      "criteria": {
        "required_leads": list(crit.required_leads),
        "expected_fs": crit.expected_fs,
        "expected_sig_len": crit.expected_sig_len,
        "expected_units": crit.expected_units,
      },
      "criteria_hash": compute_criteria_hash(crit),
      "record_list_hash": compute_record_list_hash(records_df),
    }
    prov_path.write_text(json.dumps(prov_dict, indent=2) + "\n", encoding="utf-8")

  return manifest_df


def load_waveform_eligibility_manifest(
  manifest_path: Path | str,
  require_complete: bool = False,
  expected_study_ids: set[int] | None = None,
  expected_criteria: WaveformEligibilityCriteria | None = WaveformEligibilityCriteria(),
  expected_record_list_hash: str | None = None,
  enforce_criteria_hash: bool = True,
) -> tuple[set[int], pl.DataFrame]:
  """Load a waveform eligibility manifest and return eligible study_ids and full DataFrame.

  Args:
    manifest_path: Path to .parquet or .csv manifest.
    require_complete: If True, checks that the manifest is complete (not a partial sample).
    expected_study_ids: If supplied, verifies that the manifest contains all expected study_ids.
    expected_criteria: Expected WaveformEligibilityCriteria to check against companion provenance.
    expected_record_list_hash: Optional SHA-256 hash of expected record list.
    enforce_criteria_hash: If True, enforces criteria_hash match when companion provenance exists.

  Returns:
    Tuple of (eligible_ids set, full manifest DataFrame).
  """
  p = Path(manifest_path)
  if not p.exists():
    raise FileNotFoundError(f"Waveform eligibility manifest not found at '{manifest_path}'.")
  df = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
  if "study_id" not in df.columns:
    raise ValueError(f"Manifest at {manifest_path} must contain 'study_id' column.")
  eligible_col = None
  for col in ("waveform_eligible", "is_waveform_valid", "is_eligible"):
    if col in df.columns:
      eligible_col = col
      break
  if eligible_col is None:
    raise ValueError(f"Manifest at {manifest_path} must contain 'waveform_eligible' column.")

  prov_path = p.with_suffix(".json")
  if require_complete and not prov_path.exists():
    raise ValueError(
      f"Waveform eligibility companion provenance JSON missing at '{prov_path}'. "
      "Canonical cohort derivation requires companion provenance JSON."
    )
  if prov_path.exists():
    try:
      prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
      raise ValueError(f"Corrupted companion provenance JSON at '{prov_path}': {err}") from err

    if require_complete and not prov_data.get("complete", True):
      raise ValueError(
        f"Waveform eligibility manifest at '{manifest_path}' is marked incomplete (e.g. generated with --limit). "
        "Canonical cohort derivation requires a complete manifest."
      )
    if enforce_criteria_hash and expected_criteria is not None:
      crit_hash = prov_data.get("criteria_hash")
      if require_complete and not crit_hash:
        raise ValueError(
          f"Waveform eligibility manifest at '{manifest_path}' missing criteria_hash in provenance."
        )
      if crit_hash:
        expected_hash = compute_criteria_hash(expected_criteria)
        if crit_hash != expected_hash:
          raise ValueError(
            f"Waveform eligibility manifest criteria hash mismatch at '{manifest_path}'!\n"
            f"Found: {crit_hash}\n"
            f"Expected: {expected_hash}\n"
            "The manifest was generated under different waveform eligibility criteria."
          )
    if expected_record_list_hash is not None:
      rec_hash = prov_data.get("record_list_hash")
      if require_complete and not rec_hash:
        raise ValueError(
          f"Waveform eligibility manifest at '{manifest_path}' missing record_list_hash in provenance."
        )
      if rec_hash and rec_hash != expected_record_list_hash:
        raise ValueError(
          f"Waveform eligibility manifest record list content hash mismatch at '{manifest_path}'!\n"
          f"Found: {rec_hash}\n"
          f"Expected: {expected_record_list_hash}"
        )

  manifest_sids = set(int(v) for v in df["study_id"].to_list())
  if expected_study_ids is not None:
    missing = expected_study_ids - manifest_sids
    if missing:
      raise ValueError(
        f"Waveform eligibility manifest at '{manifest_path}' is incomplete for the source records: "
        f"missing {len(missing):,} study IDs ({len(manifest_sids):,} in manifest vs "
        f"{len(expected_study_ids):,} in source records)."
      )

  eligible_df = df.filter(pl.col(eligible_col) == True)  # noqa: E712
  eligible_ids = set(int(v) for v in eligible_df["study_id"].to_list())
  return eligible_ids, df

