"""Exploratory probes for principal-discharge AMI coding.

Decision D9. A positive label is an administrative code, not a confirmed
infarction and not an incident event. A negative label is an encounter without
an AMI-family code. Ambiguous codes are excluded. Troponin does not decide
eligibility. The adjudicated pathway is closed as not pursued (decision D14).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import polars as pl

from ecg_rep.ami_outcome import LINKED_SETTINGS
from ecg_rep.clinical_setting import PUBLIC_MIN_CELL
from ecg_rep.analysis import representation_gain
from ecg_rep.evaluate import C0_FEATURE_COLS, _join_representation_matrix, _split_xy, fit_and_eval_model
from ecg_rep.mortality_sensitivity import public_suppression
from ecg_rep.probe import ProbeConfig, bootstrap_paired_difference

PRIMARY_WINDOW_HOURS = 24
LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_AMBIGUOUS = "ambiguous"
LABEL_I22_ONLY = "i22_only"


def administrative_code_flags(diagnoses: pl.DataFrame) -> pl.DataFrame:
  """One row per hadm with the primary and any-position administrative classes.

  Principal means ``seq_num == 1``. Type 2 is I21.A1. ICD-9 episode digit 2 is
  a subsequent episode. A missing episode digit stays with the initial group
  because the record does not mark it as subsequent.
  """
  required = {"hadm_id", "seq_num", "icd_code", "icd_version"}
  if required - set(diagnoses.columns):
    raise ValueError(f"Diagnoses missing {sorted(required - set(diagnoses.columns))}.")
  normalized = (
    pl.col("icd_code").cast(pl.Utf8).str.replace_all(r"[^A-Za-z0-9]", "").str.to_uppercase()
  )
  labeled = diagnoses.with_columns(normalized.alias("code"))
  is_i21 = (pl.col("icd_version") == 10) & pl.col("code").str.starts_with("I21")
  is_type2 = (pl.col("icd_version") == 10) & pl.col("code").str.starts_with("I21A1")
  is_i22 = (pl.col("icd_version") == 10) & pl.col("code").str.starts_with("I22")
  is_410 = (pl.col("icd_version") == 9) & pl.col("code").str.starts_with("410")
  episode = pl.col("code").str.slice(-1)
  is_410_subsequent = is_410 & (pl.col("code").str.len_chars() >= 5) & (episode == "2")
  is_410_initial = is_410 & ~is_410_subsequent
  is_principal = pl.col("seq_num") == 1
  acute_principal = is_principal & ((is_i21 & ~is_type2) | is_410_initial)
  family = is_i21 | is_i22 | is_410
  grouped = labeled.group_by("hadm_id").agg(
    acute_principal.any().alias("principal_acute"),
    family.any().alias("any_family"),
    (is_i21 | is_410).any().alias("any_i21_or_410"),
    is_i22.any().alias("any_i22"),
    is_type2.any().alias("any_type2"),
    (is_principal & is_type2).any().alias("principal_type2"),
  )
  return grouped.with_columns(
    pl.when(pl.col("principal_acute"))
    .then(pl.lit(LABEL_POSITIVE))
    .when(pl.col("any_family"))
    .then(pl.lit(LABEL_AMBIGUOUS))
    .otherwise(pl.lit(LABEL_NEGATIVE))
    .alias("primary_label"),
    pl.when(pl.col("any_i21_or_410"))
    .then(pl.lit(LABEL_POSITIVE))
    .when(pl.col("any_i22"))
    .then(pl.lit(LABEL_I22_ONLY))
    .otherwise(pl.lit(LABEL_NEGATIVE))
    .alias("any_position_label"),
  )


def primary_timing_eligible() -> pl.Expr:
  """ED at the ECG timestamp, or a hospital ECG in ``[admittime, admittime + 24h)``.

  The 24-hour limit is a study-design window. An ECG at exactly 24 hours is
  outside it.
  """
  linked = pl.col("setting").is_in(list(LINKED_SETTINGS)) & pl.col("hadm_id").is_not_null()
  within_day = (
    pl.col("admit_dt").is_not_null()
    & (pl.col("ecg_datetime") >= pl.col("admit_dt"))
    & (pl.col("ecg_datetime") < pl.col("admit_dt") + pl.duration(hours=PRIMARY_WINDOW_HOURS))
  )
  return linked & ((pl.col("setting") == "ED") | within_day)


def _population_mask(frame: pl.DataFrame, population: str) -> pl.Expr:
  pair = pl.col("representation_pair_valid").fill_null(False)
  if population == "primary_principal_timed":
    return pair & pl.col("primary_timing_eligible") & pl.col("primary_label").is_in([LABEL_POSITIVE, LABEL_NEGATIVE])
  if population == "sensitivity_any_position_linked":
    linked = pl.col("setting").is_in(list(LINKED_SETTINGS)) & pl.col("hadm_id").is_not_null()
    return pair & linked & pl.col("any_position_label").is_in([LABEL_POSITIVE, LABEL_NEGATIVE])
  if population == "sensitivity_troponin_tested":
    return (
      pair
      & pl.col("primary_timing_eligible")
      & pl.col("primary_label").is_in([LABEL_POSITIVE, LABEL_NEGATIVE])
      & pl.col("troponin_during_encounter").fill_null(False)
    )
  raise ValueError(f"Unknown AMI population {population}.")


def _label_column(population: str) -> str:
  if population == "sensitivity_any_position_linked":
    return "any_position_label"
  return "primary_label"


def attrition_table(frame: pl.DataFrame) -> pl.DataFrame:
  """Row counts by split before any probe is fit."""
  rows: list[dict[str, Any]] = []
  linked = pl.col("setting").is_in(list(LINKED_SETTINGS)) & pl.col("hadm_id").is_not_null()
  for split in ("dev", "val", "test"):
    sub = frame.filter(pl.col("split") == split)
    linked_sub = sub.filter(linked)
    timed = sub.filter(pl.col("primary_timing_eligible"))
    rows.append({
      "split": split,
      "n_index": sub.height,
      "n_linked": linked_sub.height,
      "n_timing_eligible": timed.height,
      "n_primary_positive": int(timed.filter(pl.col("primary_label") == LABEL_POSITIVE).height),
      "n_primary_negative": int(timed.filter(pl.col("primary_label") == LABEL_NEGATIVE).height),
      "n_primary_ambiguous": int(timed.filter(pl.col("primary_label") == LABEL_AMBIGUOUS).height),
      "n_type2_among_timed": int(timed.filter(pl.col("any_type2").fill_null(False)).height),
      "n_i22_only_among_linked": int(linked_sub.filter(pl.col("any_position_label") == LABEL_I22_ONLY).height),
      "n_primary_pair_valid": int(timed.filter(
        pl.col("primary_label").is_in([LABEL_POSITIVE, LABEL_NEGATIVE])
        & pl.col("representation_pair_valid").fill_null(False)
      ).height),
    })
  return pl.DataFrame(rows)


def _metric_shell(population: str, status: str) -> dict[str, Any]:
  """One metrics row. Discrimination stays empty until both classes exist."""
  return {
    "population": population,
    "status": status,
    "n": None,
    "n_events": None,
    "event_prevalence": None,
    "best_c_a": None,
    "best_c_b": None,
    "auroc_a": None,
    "auroc_b": None,
    "auprc_a": None,
    "auprc_b": None,
    "delta_auroc_b_minus_a": None,
    "delta_auroc_ci_low": None,
    "delta_auroc_ci_high": None,
    "delta_auprc_b_minus_a": None,
    "delta_auprc_ci_low": None,
    "delta_auprc_ci_high": None,
    "brier_a": None,
    "brier_b": None,
    "calibration_intercept_b": None,
    "calibration_slope_b": None,
    "n_bootstraps": 0,
  }


def publish_attrition(frame: pl.DataFrame) -> pl.DataFrame:
  """Blank a label count below 11 and the timing total that would recover it."""
  label_cols = ("n_primary_positive", "n_primary_negative", "n_primary_ambiguous")
  small_label = pl.any_horizontal([pl.col(col) < PUBLIC_MIN_CELL for col in label_cols])
  expressions: list[pl.Expr] = [
    pl.when(pl.col(col) < PUBLIC_MIN_CELL).then(None).otherwise(pl.col(col)).alias(col)
    for col in frame.columns
    if col != "split" and col not in label_cols and col != "n_timing_eligible"
  ]
  for col in label_cols:
    expressions.append(
      pl.when(pl.col(col) < PUBLIC_MIN_CELL).then(None).otherwise(pl.col(col)).alias(col)
    )
  expressions.append(
    pl.when(small_label).then(None).otherwise(pl.col("n_timing_eligible")).alias("n_timing_eligible")
  )
  return frame.with_columns(expressions)


def publish_metrics(frame: pl.DataFrame) -> pl.DataFrame:
  """Blank discrimination when a population's locked-test cell is below 11."""
  sparse = pl.col("n").is_not_null() & (
    (pl.col("n") < PUBLIC_MIN_CELL) | (pl.col("n_events").fill_null(0) < PUBLIC_MIN_CELL)
  )
  metric_cols = [
    "n", "n_events", "event_prevalence", "auroc_a", "auroc_b", "auprc_a", "auprc_b",
    "delta_auroc_b_minus_a", "delta_auroc_ci_low", "delta_auroc_ci_high",
    "delta_auprc_b_minus_a", "delta_auprc_ci_low", "delta_auprc_ci_high",
    "brier_a", "brier_b", "calibration_intercept_b", "calibration_slope_b",
  ]
  return frame.with_columns([
    pl.when(sparse).then(None).otherwise(pl.col(col)).alias(col)
    for col in metric_cols
    if col in frame.columns
  ])


def _can_fit(frame: pl.DataFrame) -> bool:
  for split in ("dev", "val"):
    y = frame.filter(pl.col("split") == split)["ami_positive"]
    if y.len() < 2 or y.n_unique() < 2:
      return False
  return True


def _fit_pair(
  frame: pl.DataFrame,
  feature_a: list[str],
  feature_b: list[str],
  population: str,
  n_bootstraps: int,
  seed: int,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None, pl.DataFrame | None]:
  """Fit A and B on the same rows. The fitter's label column is reused by rename."""
  labeled = frame.with_columns((pl.col(_label_column(population)) == LABEL_POSITIVE).cast(pl.Int64).alias("ami_positive"))
  if not _can_fit(labeled):
    return _metric_shell(population, "blocked_missing_class"), None, None, None
  # The shared probe fitter reads a column named mortality_30d. On this frame it is the administrative AMI code, not 30-day death.
  trainable = labeled.with_columns(pl.col("ami_positive").alias("mortality_30d"))
  result_a, probe_a = fit_and_eval_model(trainable, feature_a, f"{population}_a", config=ProbeConfig())
  result_b, probe_b = fit_and_eval_model(trainable, feature_b, f"{population}_b", config=ProbeConfig())
  test_a = _split_xy(trainable, feature_a, "test")
  test_b = _split_xy(trainable, feature_b, "test")
  if not np.array_equal(test_a.study_ids, test_b.study_ids):
    raise ValueError(f"{population} A and B test rows differ.")
  scores_a = probe_a.predict_proba(test_a.x)
  scores_b = probe_b.predict_proba(test_b.x)
  y = test_a.y
  both = len(y) >= 2 and len(np.unique(y)) >= 2
  row = _metric_shell(population, "complete" if both else "test_one_class")
  row.update({
    "n": int(len(y)),
    "n_events": int(y.sum()) if len(y) else 0,
    "event_prevalence": float(y.mean()) if len(y) else None,
    "best_c_a": result_a["best_c"],
    "best_c_b": result_b["best_c"],
  })
  if both:
    gain_auroc = representation_gain(y, scores_a, scores_b, metric="auroc")
    gain_auprc = representation_gain(y, scores_a, scores_b, metric="auprc")
    boot_auroc = bootstrap_paired_difference(y, scores_a, scores_b, "auroc", n_bootstraps, seed)
    boot_auprc = bootstrap_paired_difference(y, scores_a, scores_b, "auprc", n_bootstraps, seed + 1)
    metrics_b = result_b["test_metrics"]
    row.update({
      "auroc_a": gain_auroc.score_a,
      "auroc_b": gain_auroc.score_b,
      "auprc_a": gain_auprc.score_a,
      "auprc_b": gain_auprc.score_b,
      "delta_auroc_b_minus_a": boot_auroc.point_estimate,
      "delta_auroc_ci_low": boot_auroc.ci_lower,
      "delta_auroc_ci_high": boot_auroc.ci_upper,
      "delta_auprc_b_minus_a": boot_auprc.point_estimate,
      "delta_auprc_ci_low": boot_auprc.ci_lower,
      "delta_auprc_ci_high": boot_auprc.ci_upper,
      "brier_a": result_a["test_metrics"]["brier"],
      "brier_b": metrics_b["brier"],
      "calibration_intercept_b": metrics_b["calibration_intercept"],
      "calibration_slope_b": metrics_b["calibration_slope"],
      "n_bootstraps": n_bootstraps,
    })
  test_rows = frame.filter(pl.col("split") == "test")
  return row, scores_a, scores_b, test_rows


def setting_rows(
  frame: pl.DataFrame,
  scores_a: np.ndarray,
  scores_b: np.ndarray,
  n_bootstraps: int,
  seed: int,
) -> pl.DataFrame:
  """Primary-test discrimination inside care settings. Models are not refit."""
  labeled = frame.with_columns(
    pl.Series("score_a", scores_a),
    pl.Series("score_b", scores_b),
  )
  rows: list[dict[str, Any]] = []
  for setting in LINKED_SETTINGS:
    sub = labeled.filter(pl.col("setting") == setting)
    y = np.asarray(sub["ami_positive"].to_numpy(), dtype=np.int64)
    both = len(y) >= 2 and len(np.unique(y)) >= 2
    row: dict[str, Any] = {
      "setting": setting,
      "n": int(len(y)),
      "n_events": int(y.sum()) if len(y) else 0,
      "event_prevalence": float(y.mean()) if len(y) else None,
      "auroc_a": None,
      "auroc_b": None,
      "delta_auroc_b_minus_a": None,
      "delta_auroc_ci_low": None,
      "delta_auroc_ci_high": None,
    }
    if both:
      gain = representation_gain(y, sub["score_a"].to_numpy(), sub["score_b"].to_numpy(), metric="auroc")
      boot = bootstrap_paired_difference(
        y,
        np.asarray(sub["score_a"].to_numpy(), dtype=np.float64),
        np.asarray(sub["score_b"].to_numpy(), dtype=np.float64),
        "auroc",
        n_bootstraps,
        seed,
      )
      row.update({
        "auroc_a": gain.score_a,
        "auroc_b": gain.score_b,
        "delta_auroc_b_minus_a": boot.point_estimate,
        "delta_auroc_ci_low": boot.ci_lower,
        "delta_auroc_ci_high": boot.ci_upper,
      })
    rows.append(row)
  return pl.DataFrame(rows)


def _num(row: dict[str, Any], key: str) -> str:
  value = row.get(key)
  if not isinstance(value, (int, float)):
    return "suppressed"
  return f"{float(value):.3f}"


def _setting_sentence(by_setting: pl.DataFrame) -> str:
  """One sentence for the primary-test setting rows. Models are not refit."""
  if by_setting.height == 0 or "setting" not in by_setting.columns:
    return ""
  pieces = []
  for row in by_setting.iter_rows(named=True):
    pieces.append(
      f"{row['setting']} {_num(row, 'delta_auroc_b_minus_a')} "
      f"({_num(row, 'delta_auroc_ci_low')} to {_num(row, 'delta_auroc_ci_high')})"
    )
  return (
    "Within the primary locked test, B−A AUROC is "
    + "; ".join(pieces)
    + ". Those intervals use the primary probes and do not refit them inside each setting."
  )


def _interpretation(metrics: pl.DataFrame, by_setting: pl.DataFrame) -> str:
  """State the coding result without turning it into adjudicated AMI."""
  rows = {str(row["population"]): row for row in metrics.iter_rows(named=True)}
  primary = rows.get("primary_principal_timed", {})
  context = rows.get("primary_c0", {})
  broad = rows.get("sensitivity_any_position_linked", {})
  tested = rows.get("sensitivity_troponin_tested", {})
  boundary = ""
  best_c = primary.get("best_c_a")
  if isinstance(best_c, (int, float)) and float(best_c) >= 1000:
    boundary = (
      " The CIIS-only probe selected C=1000, the top of the prespecified grid "
      "from 0.0001 to 1000. That grid was not extended after the locked test was scored."
    )
  parts = [
    "On the primary locked-test population, the CIIS-derived probe AUROC is "
    f"{_num(primary, 'auroc_a')} and the frozen D-BETA probe AUROC is {_num(primary, 'auroc_b')}.",
    f"B−A AUROC is {_num(primary, 'delta_auroc_b_minus_a')} "
    f"({_num(primary, 'delta_auroc_ci_low')} to {_num(primary, 'delta_auroc_ci_high')}).",
    f"B−A AUPRC is {_num(primary, 'delta_auprc_b_minus_a')} "
    f"({_num(primary, 'delta_auprc_ci_low')} to {_num(primary, 'delta_auprc_ci_high')}).",
    f"D-BETA calibration intercept is {_num(primary, 'calibration_intercept_b')} "
    f"and slope is {_num(primary, 'calibration_slope_b')}.",
    boundary,
    "Adding the original C0 block on the same rows gives B−A AUROC "
    f"{_num(context, 'delta_auroc_b_minus_a')} "
    f"({_num(context, 'delta_auroc_ci_low')} to {_num(context, 'delta_auroc_ci_high')}).",
    "The any-position analysis is a broader code set on the broader linked index-ECG cohort. "
    f"Its B−A AUROC is {_num(broad, 'delta_auroc_b_minus_a')} "
    f"({_num(broad, 'delta_auroc_ci_low')} to {_num(broad, 'delta_auroc_ci_high')}).",
    "Encounters whose only AMI-family code is I22 are excluded from that binary fit. "
    "Their counts are omitted where the public cell minimum is not met.",
    "The troponin-tested subset is a different ascertainment population, not a gold standard. "
    f"Its B−A AUROC is {_num(tested, 'delta_auroc_b_minus_a')} "
    f"({_num(tested, 'delta_auroc_ci_low')} to {_num(tested, 'delta_auroc_ci_high')}).",
    "Troponin was not a model feature.",
    "The earlier feasibility count of 1,644 principal I21 or 410 codes included type 2 and "
    "subsequent-episode 410, which this protocol does not call positive, and it did not apply "
    "the timing window. The attrition table is the count after both rules.",
    "A code-negative encounter is an encounter without an AMI-family discharge code. "
    "A discharge code does not establish that infarction was present when the ECG was acquired.",
    _setting_sentence(by_setting),
  ]
  return " ".join(part.strip() for part in parts if part.strip())


def render_administrative_results(
  attrition: pl.DataFrame,
  metrics: pl.DataFrame,
  by_setting: pl.DataFrame,
  audit: dict[str, Any],
) -> str:
  """Aggregate note. The endpoint is discharge coding."""
  def table(frame: pl.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in frame.iter_rows(named=True):
      cells = []
      for col in columns:
        value = row[col]
        if value is None:
          cells.append("suppressed")
        elif isinstance(value, float) and (col == "n" or col.startswith("n_")):
          cells.append(str(int(value)))
        elif isinstance(value, float):
          cells.append(f"{value:.3f}")
        else:
          cells.append(str(value))
      body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])

  return "\n".join([
    "# Administrative index-encounter AMI coding",
    "",
    "Analysis status: `exploratory_administrative`. Decision D9.",
    f"Protocol hash: `{audit['protocol_hash']}`.",
    f"Source SHA: `{audit['source_sha']}`.",
    "",
    "The question is how much additional information a frozen D-BETA embedding "
    "provides, relative to the 15 CIIS-derived features, for identifying an "
    "encounter whose principal discharge diagnosis is coded as acute myocardial "
    "infarction. This is not clinically confirmed AMI and it is not prediction "
    "of a future infarction. A code-negative encounter has no I21, I22, or 410 "
    "on the linked admission. That is not a documented rule-out.",
    "",
    "The preserved earliest index ECG is kept when it falls in the ED or in the "
    "half-open 24 hours after hospital admission. A different ECG from the same "
    "encounter is not substituted. Ambiguous encounters, including secondary AMI "
    "codes, I22, type 2 I21.A1, and ICD-9 410 episode digit 2, are excluded from "
    "the primary comparison. Troponin is not required. The adjudicated pathway "
    "is closed as not pursued (decision D14). MIMIC-IV-Note is not installed here, so no chart-note "
    "plausibility sample was reviewed. `waveform_note_links` is not note text.",
    "",
    "## Attrition",
    "",
    table(attrition),
    "",
    "## Locked-test discrimination",
    "",
    "A and B probes are fit on the named population only. Regularization uses "
    "validation log-loss. Bootstrap intervals resample the locked test and do "
    "not refit. `primary_c0` adds the original C0 block on the same primary rows.",
    "",
    table(metrics),
    "",
    "## Primary result by care setting",
    "",
    table(by_setting),
    "",
    "MIMIC evaluation remains in-domain because D-BETA was pretrained on MIMIC-IV-ECG.",
    "",
    "## Interpretation",
    "",
    _interpretation(metrics, by_setting),
    "",
  ])


def build_analysis_frame(
  cohort: pl.DataFrame,
  setting: pl.DataFrame,
  admissions: pl.DataFrame,
  diagnoses: pl.DataFrame,
  troponin: pl.DataFrame,
  pair_validity: pl.DataFrame,
) -> pl.DataFrame:
  """Attach codes and timing. This step does not read model scores."""
  flags = administrative_code_flags(diagnoses)
  admit = admissions.select(["hadm_id", "admit_dt"]).unique(subset=["hadm_id"])
  frame = (
    cohort.select(["subject_id", "study_id", "split", "ecg_datetime"])
    .join(setting.select(["study_id", "setting", "hadm_id"]), on="study_id", how="left")
    .join(admit, on="hadm_id", how="left")
    .join(flags, on="hadm_id", how="left")
    .join(troponin, on="hadm_id", how="left")
    .join(pair_validity.select(["study_id", "representation_pair_valid"]), on="study_id", how="left")
  )
  frame = frame.with_columns(
    pl.col("primary_label").fill_null(LABEL_NEGATIVE),
    pl.col("any_position_label").fill_null(LABEL_NEGATIVE),
    pl.col("any_type2").fill_null(False),
    pl.col("troponin_during_encounter").fill_null(False),
    primary_timing_eligible().alias("primary_timing_eligible"),
  )
  # fill_null on the label before the timing expression is applied would mark
  # unlinked rows negative. Recompute labels only where a hadm was joined.
  return frame.with_columns(
    pl.when(pl.col("hadm_id").is_null())
    .then(pl.lit(None))
    .otherwise(pl.col("primary_label"))
    .alias("primary_label"),
    pl.when(pl.col("hadm_id").is_null())
    .then(pl.lit(None))
    .otherwise(pl.col("any_position_label"))
    .alias("any_position_label"),
  )


def run_administrative_ami(
  cohort_path: Path,
  setting_path: Path,
  admissions: pl.DataFrame,
  diagnoses_path: Path,
  labevents_path: Path,
  pair_validity_path: Path,
  c0_path: Path,
  meta_a_path: Path,
  features_a_path: Path,
  meta_b_path: Path,
  embeddings_b_path: Path,
  protocol_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = 200,
  seed: int = 42,
) -> dict[str, Any]:
  """Fit outcome-specific A and B probes. Mortality probes are not reused."""
  from ecg_rep.ami_outcome import troponin_hadm_flags

  protocol_text = protocol_path.read_text(encoding="utf-8")
  cohort = pl.read_parquet(cohort_path)
  setting = pl.read_parquet(setting_path)
  diagnoses = pl.scan_csv(
    diagnoses_path,
    schema_overrides={
      "hadm_id": pl.Int64,
      "seq_num": pl.Int64,
      "icd_code": pl.String,
      "icd_version": pl.Int64,
    },
  ).select(["hadm_id", "seq_num", "icd_code", "icd_version"]).collect()
  troponin = troponin_hadm_flags(labevents_path, [51002, 52642, 51003])
  pair = pl.read_parquet(pair_validity_path)
  labeled = build_analysis_frame(cohort, setting, admissions, diagnoses, troponin, pair)
  attrition = attrition_table(labeled)
  c0 = pl.read_parquet(c0_path)
  c0_cols = [col for col in C0_FEATURE_COLS if col in c0.columns]
  prepared = labeled.join(c0.select(["study_id", *c0_cols]), on="study_id", how="left")
  prepared = _join_representation_matrix(
    prepared, pl.read_parquet(meta_a_path), np.load(features_a_path), "za", "representation_a_valid",
  )
  prepared = _join_representation_matrix(
    prepared, pl.read_parquet(meta_b_path), np.load(embeddings_b_path), "zb", "representation_b_valid",
  )
  za_cols = [col for col in prepared.columns if col.startswith("za_")]
  zb_cols = [col for col in prepared.columns if col.startswith("zb_")]
  populations = (
    ("primary_principal_timed", za_cols, zb_cols),
    ("primary_c0", c0_cols + za_cols, c0_cols + zb_cols),
    ("sensitivity_any_position_linked", za_cols, zb_cols),
    ("sensitivity_troponin_tested", za_cols, zb_cols),
  )
  metric_rows: list[dict[str, Any]] = []
  primary_setting = pl.DataFrame()
  for offset, (population, cols_a, cols_b) in enumerate(populations):
    mask_name = "primary_principal_timed" if population == "primary_c0" else population
    label_name = "primary_principal_timed" if population == "primary_c0" else population
    subset = prepared.filter(_population_mask(prepared, mask_name))
    if population == "primary_c0":
      subset = subset.drop_nulls(c0_cols)
    row, scores_a, scores_b, test_rows = _fit_pair(
      subset, cols_a, cols_b, label_name, n_bootstraps, seed + offset,
    )
    row["population"] = population
    metric_rows.append(row)
    if (
      population == "primary_principal_timed"
      and scores_a is not None
      and scores_b is not None
      and test_rows is not None
    ):
      scored = test_rows.with_columns((pl.col("primary_label") == LABEL_POSITIVE).cast(pl.Int64).alias("ami_positive"))
      primary_setting = setting_rows(scored, scores_a, scores_b, n_bootstraps, seed + 20)
  if primary_setting.height == 0:
    primary_setting = pl.DataFrame({
      "setting": [],
      "n": [],
      "n_events": [],
      "event_prevalence": [],
      "auroc_a": [],
      "auroc_b": [],
      "delta_auroc_b_minus_a": [],
      "delta_auroc_ci_low": [],
      "delta_auroc_ci_high": [],
    })
  public_setting = public_suppression(primary_setting, ("setting",)) if primary_setting.height else primary_setting
  metrics = publish_metrics(pl.DataFrame(metric_rows))
  public_attrition = publish_attrition(attrition)
  audit = {
    "decision_ids": "D9",
    "source_sha": source_sha,
    "protocol_hash": hashlib.sha256(protocol_text.encode("utf-8")).hexdigest(),
    "window_hours": PRIMARY_WINDOW_HOURS,
    "troponin_required": False,
    "adjudicated_evaluation": "not_pursued_D14",
    "claim": "prediction of principal-discharge AMI coding",
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  labeled.select([
    "study_id", "subject_id", "split", "setting", "hadm_id", "primary_label",
    "any_position_label", "primary_timing_eligible", "troponin_during_encounter",
  ]).write_parquet(output_dir / "administrative_ami_labels.parquet")
  (output_dir / "administrative_ami_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
  attrition.write_csv(output_dir / "administrative_ami_attrition.csv")
  pl.DataFrame(metric_rows).write_csv(output_dir / "administrative_ami_metrics.csv")
  public_rows = public_setting.drop("suppress_status", strict=False)
  report = render_administrative_results(public_attrition, metrics, public_rows, audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  tables = report_path.parent / "tables"
  tables.mkdir(parents=True, exist_ok=True)
  public_attrition.write_csv(tables / "ami_administrative_attrition.csv")
  metrics.write_csv(tables / "ami_administrative_metrics.csv")
  public_setting.drop("suppress_status", strict=False).write_csv(tables / "ami_administrative_by_setting.csv")
  return audit
