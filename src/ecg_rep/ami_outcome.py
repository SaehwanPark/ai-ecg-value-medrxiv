"""Development-only feasibility for an index-encounter AMI label.

No phenotype in this module is an approved outcome. Missing troponin is not a
negative label. Machine ECG text is not an AMI label. Decision D14 closed the
adjudicated AMI pathway as not pursued, so A versus B evaluation on an
adjudicated label is refused; the coded-label endpoint is ``ami_administrative``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

import polars as pl

AMI_STATUS = "feasibility-only"
ADJUDICATED_EVALUATION_STATUS = "not_pursued_D14"
LINKED_SETTINGS = ("ED", "ICU", "inpatient_non_ICU")


def refuse_ami_model_evaluation() -> None:
  """Stop adjudicated AMI probe fitting; no adjudicated label exists."""
  raise RuntimeError(
    "Adjudicated AMI evaluation is not pursued (decision D14, 2026-10-05). "
    "Use `ami-administrative` for the principal-discharge coded-label endpoint."
  )


def normalize_icd(code: str) -> str:
  """Drop punctuation and uppercase an ICD code."""
  return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()


def classify_ami_code(code: str, icd_version: int) -> str | None:
  """Return a candidate AMI family, or None when the code is not an AMI stem.

  I22 is subsequent myocardial infarction, kept separate from I21. Chronic
  ischemic codes and unstable angina are not AMI.
  """
  normalized = normalize_icd(code)
  if int(icd_version) == 10:
    if normalized.startswith("I21"):
      return "icd10_ami"
    if normalized.startswith("I22"):
      return "icd10_subsequent_mi"
    return None
  if int(icd_version) == 9 and normalized.startswith("410"):
    return "icd9_ami"
  return None


def encounter_code_flags(diagnoses: pl.DataFrame) -> pl.DataFrame:
  """Collapse diagnosis rows to one hadm-level candidate-code summary.

  Absence of an AMI code is not a negative AMI label. The flags only say
  whether a candidate code was present.
  """
  required = {"hadm_id", "icd_code", "icd_version"}
  if required - set(diagnoses.columns):
    raise ValueError(f"Diagnoses missing {sorted(required - set(diagnoses.columns))}")
  seq = pl.col("seq_num") if "seq_num" in diagnoses.columns else pl.lit(None)
  normalized = (
    pl.col("icd_code").cast(pl.Utf8).str.replace_all(r"[^A-Za-z0-9]", "").str.to_uppercase()
  )
  family = (
    pl.when((pl.col("icd_version") == 10) & normalized.str.starts_with("I21"))
    .then(pl.lit("icd10_ami"))
    .when((pl.col("icd_version") == 10) & normalized.str.starts_with("I22"))
    .then(pl.lit("icd10_subsequent_mi"))
    .when((pl.col("icd_version") == 9) & normalized.str.starts_with("410"))
    .then(pl.lit("icd9_ami"))
    .otherwise(pl.lit(None, dtype=pl.String))
  )
  labeled = diagnoses.with_columns(seq.alias("seq_num"), family.alias("ami_family"))
  coded = labeled.filter(pl.col("ami_family").is_not_null())
  if coded.height == 0:
    return pl.DataFrame({
      "hadm_id": pl.Series([], dtype=pl.Int64),
      "any_candidate_ami_code": pl.Series([], dtype=pl.Boolean),
      "principal_candidate_ami_code": pl.Series([], dtype=pl.Boolean),
      "ami_families": pl.Series([], dtype=pl.String),
    })
  return coded.group_by("hadm_id").agg(
    pl.lit(True).alias("any_candidate_ami_code"),
    ((pl.col("seq_num") == 1) & pl.col("ami_family").is_in(["icd10_ami", "icd9_ami"])).any()
    .alias("principal_candidate_ami_code"),
    pl.col("ami_family").unique().sort().str.join("|").alias("ami_families"),
  )


def feasibility_counts(
  cohort: pl.DataFrame,
  setting: pl.DataFrame,
  code_flags: pl.DataFrame,
  troponin_hadm: pl.DataFrame | None = None,
) -> dict[str, Any]:
  """Count development linkage and candidate evidence. Do not assign negatives."""
  dev = cohort.filter(pl.col("split") == "dev").select(["study_id", "subject_id"])
  linked = setting.select(["study_id", "setting", "hadm_id"]).join(dev, on="study_id", how="inner")
  acute = linked.filter(pl.col("setting").is_in(list(LINKED_SETTINGS)) & pl.col("hadm_id").is_not_null())
  flagged = acute.join(code_flags, on="hadm_id", how="left").with_columns(
    pl.col("any_candidate_ami_code").fill_null(False),
    pl.col("principal_candidate_ami_code").fill_null(False),
  )
  if troponin_hadm is not None:
    flagged = flagged.join(troponin_hadm.select(["hadm_id", "troponin_during_encounter"]), on="hadm_id", how="left")
  else:
    flagged = flagged.with_columns(pl.lit(None).cast(pl.Boolean).alias("troponin_during_encounter"))
  both = flagged.filter(pl.col("any_candidate_ami_code") & pl.col("troponin_during_encounter").fill_null(False))
  return {
    "status": AMI_STATUS,
    "model_evaluation": ADJUDICATED_EVALUATION_STATUS,
    "n_development": dev.height,
    "n_linked_ed_icu_or_inpatient": acute.height,
    "n_candidate_ami_code": int(flagged.filter(pl.col("any_candidate_ami_code")).height),
    "n_principal_ami_code": int(flagged.filter(pl.col("principal_candidate_ami_code")).height),
    "n_troponin_during_encounter": (
      None if troponin_hadm is None else int(flagged.filter(pl.col("troponin_during_encounter").fill_null(False)).height)
    ),
    "n_code_and_troponin": None if troponin_hadm is None else both.height,
    "missing_troponin_is_negative": False,
    "machine_text_is_ami_label": False,
    "test_labels_measured": False,
  }


def render_ami_feasibility(summary: dict[str, Any]) -> str:
  """Feasibility memo. It does not report A versus B discrimination."""
  return "\n".join([
    "# Index-encounter AMI feasibility",
    "",
    f"Status: `{summary['status']}`. Model evaluation: `{summary['model_evaluation']}`.",
    f"Source SHA: `{summary.get('source_sha', 'unknown')}`.",
    "",
    "This audit asks whether a development-split index encounter has candidate "
    "administrative AMI evidence and a troponin measured during that encounter. "
    "It does not define cases or controls. A missing troponin is not a negative "
    "label. An ICD code without a reviewed troponin pattern is not a definite AMI. "
    "ECG machine text was not used as a label. The locked test was not measured.",
    "",
    f"Development index ECGs: {summary['n_development']:,}.",
    f"ED, ICU, or inpatient linkage with a `hadm_id`: {summary['n_linked_ed_icu_or_inpatient']:,}.",
    f"Linked encounters with any candidate AMI code (I21, I22, or 410): {summary['n_candidate_ami_code']:,}.",
    f"Of those links, principal I21 or 410 codes: {summary['n_principal_ami_code']:,}.",
    f"Linked encounters with any troponin I or T on the same `hadm_id`: {summary['n_troponin_during_encounter']}.",
    f"Linked encounters with both a candidate code and a troponin on that encounter: {summary['n_code_and_troponin']}.",
    "",
    "I22 is counted inside the any-code total and is not treated as a principal acute AMI code. "
    "Troponin presence ignores assay, units, sex-specific 99th percentiles, and rise-and-fall. "
    "Those gaps are large enough that F-AMI is not passed. No AMI probe was fit.",
    "",
    "Decision D14 (2026-10-05) closed the adjudicated pathway as not pursued, so the pragmatic "
    "screen of about 100 confirmed locked-test positives is not evaluated. This development count "
    "is not a confirmed case count. The coded-label endpoint is the decision-D9 administrative analysis.",
    "",
  ])


def troponin_hadm_flags(labevents_path: Path, itemids: list[int]) -> pl.DataFrame:
  """Encounters with any troponin row. Timing relative to the ECG is not applied."""
  frame = (
    pl.scan_csv(
      labevents_path,
      schema_overrides={"hadm_id": pl.Int64, "itemid": pl.Int64},
    )
    .filter(pl.col("itemid").is_in(itemids) & pl.col("hadm_id").is_not_null())
    .select("hadm_id")
    .unique()
    .collect()
  )
  return frame.with_columns(pl.lit(True).alias("troponin_during_encounter"))


def run_ami_feasibility(
  cohort_path: Path,
  setting_path: Path,
  diagnoses_path: Path,
  labevents_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
) -> dict[str, Any]:
  """Development feasibility only. Refuses to write model metrics."""
  cohort = pl.read_parquet(cohort_path)
  if cohort.filter(pl.col("split") == "test").height == 0:
    raise ValueError("Cohort has no test split. Refusing an unlabeled extract.")
  setting = pl.read_parquet(setting_path)
  diagnoses = pl.scan_csv(
    diagnoses_path,
    schema_overrides={
      "subject_id": pl.Int64,
      "hadm_id": pl.Int64,
      "seq_num": pl.Int64,
      "icd_code": pl.String,
      "icd_version": pl.Int64,
    },
  ).select(["hadm_id", "seq_num", "icd_code", "icd_version"]).collect()
  flags = encounter_code_flags(diagnoses)
  troponin = troponin_hadm_flags(labevents_path, [51002, 52642, 51003])
  summary = feasibility_counts(cohort, setting, flags, troponin)
  summary["source_sha"] = source_sha
  summary["decision"] = "no-go for model evaluation"
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "ami_feasibility_audit.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )
  report = render_ami_feasibility(summary)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  return summary
