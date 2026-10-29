"""Leakage-safe Elixhauser comorbidity indicators from prior admissions only."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any
import polars as pl

from ecg_rep.context import prior_completed_admissions
from ecg_rep.elixhauser_maps import (
  ELIXHAUSER_CATEGORIES,
  ELIXHAUSER_ICD10,
  ELIXHAUSER_ICD9,
)


def _normalize_icd_code(code: str) -> str:
  return re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()


def _feature_col(category: str) -> str:
  slug = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
  return f"elix_{slug}"


ELIXHAUSER_FEATURE_NAMES: tuple[str, ...] = tuple(_feature_col(c) for c in ELIXHAUSER_CATEGORIES)


@lru_cache(maxsize=1)
def _indexes() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
  """Exact normalized code → tuple of categories indexes for ICD-9 and ICD-10 maps."""
  icd9: dict[str, list[str]] = {}
  for category, codes in ELIXHAUSER_ICD9.items():
    for code in codes:
      norm = _normalize_icd_code(code)
      if category not in icd9.setdefault(norm, []):
        icd9[norm].append(category)

  icd10: dict[str, list[str]] = {}
  for category, codes in ELIXHAUSER_ICD10.items():
    for code in codes:
      norm = _normalize_icd_code(code)
      if category not in icd10.setdefault(norm, []):
        icd10[norm].append(category)

  return (
    {k: tuple(sorted(v)) for k, v in icd9.items()},
    {k: tuple(sorted(v)) for k, v in icd10.items()},
  )


def map_icd_to_elixhauser_categories(icd_code: str, icd_version: int) -> tuple[str, ...]:
  """Return all matching Elixhauser categories for one ICD code across all prefix lengths.

  Supports cross-prefix overlaps (e.g., I278 matching both Chronic Pulmonary Disease via
  prefix I278 and Pulmonary Circulation Disorders via prefix I27).
  """
  normalized = _normalize_icd_code(icd_code)
  if len(normalized) < 3:
    return ()
  icd9, icd10 = _indexes()
  index = icd9 if int(icd_version) == 9 else icd10
  matches: set[str] = set()
  for length in range(len(normalized), 2, -1):
    hit = index.get(normalized[:length])
    if hit is not None:
      matches.update(hit)
  return tuple(sorted(matches))


def map_icd_to_elixhauser(icd_code: str, icd_version: int) -> str | None:
  """Return the first matching Elixhauser category (backwards compatibility)."""
  cats = map_icd_to_elixhauser_categories(icd_code, icd_version)
  return cats[0] if cats else None


def load_diagnoses(path: Path | str) -> pl.DataFrame:
  """Load MIMIC-IV ``diagnoses_icd``."""
  return pl.read_csv(
    path,
    schema_overrides={
      "subject_id": pl.Int64,
      "hadm_id": pl.Int64,
      "seq_num": pl.Int64,
      "icd_code": pl.String,
      "icd_version": pl.Int64,
    },
  )


def assign_elixhauser_categories(diagnoses_df: pl.DataFrame) -> pl.DataFrame:
  """Annotate diagnosis rows with Elixhauser categories list and primary category."""
  categories_list = [
    list(map_icd_to_elixhauser_categories(str(code), int(version)))
    for code, version in diagnoses_df.select(["icd_code", "icd_version"]).iter_rows()
  ]
  primary_categories = [c[0] if c else None for c in categories_list]
  return diagnoses_df.with_columns(
    pl.Series("elixhauser_category", primary_categories),
    pl.Series("elixhauser_categories", categories_list),
  )


def apply_quan_exclusions(indicators_df: pl.DataFrame) -> pl.DataFrame:
  """Apply standard Quan et al. (2005) hierarchy exclusions to Elixhauser indicator columns:
  1. Complicated hypertension zeroes out uncomplicated hypertension.
  2. Complicated diabetes zeroes out uncomplicated diabetes.
  3. Metastatic cancer zeroes out solid tumor without metastasis.
  """
  out = indicators_df
  if (
    "elix_hypertension_complicated" in out.columns
    and "elix_hypertension_uncomplicated" in out.columns
  ):
    out = out.with_columns(
      pl.when(pl.col("elix_hypertension_complicated") == 1)
      .then(pl.lit(0).cast(pl.Int8))
      .otherwise(pl.col("elix_hypertension_uncomplicated"))
      .alias("elix_hypertension_uncomplicated")
    )
  if (
    "elix_diabetes_complicated" in out.columns
    and "elix_diabetes_uncomplicated" in out.columns
  ):
    out = out.with_columns(
      pl.when(pl.col("elix_diabetes_complicated") == 1)
      .then(pl.lit(0).cast(pl.Int8))
      .otherwise(pl.col("elix_diabetes_uncomplicated"))
      .alias("elix_diabetes_uncomplicated")
    )
  if (
    "elix_metastatic_cancer" in out.columns
    and "elix_solid_tumor_without_metastasis" in out.columns
  ):
    out = out.with_columns(
      pl.when(pl.col("elix_metastatic_cancer") == 1)
      .then(pl.lit(0).cast(pl.Int8))
      .otherwise(pl.col("elix_solid_tumor_without_metastasis"))
      .alias("elix_solid_tumor_without_metastasis")
    )
  return out


def elixhauser_indicators_from_prior_diagnoses(
  prior_diag_df: pl.DataFrame,
  subject_ids: pl.Series | None = None,
  apply_exclusions: bool = True,
) -> pl.DataFrame:
  """Collapse prior diagnoses to subject-level binary Elixhauser indicators.

  Supports one-to-many category mapping (e.g. I426 mapping to both Alcohol Abuse
  and Congestive Heart Failure) and applies standard Quan (2005) hierarchy exclusions.
  """
  if subject_ids is None:
    subjects = prior_diag_df.select("subject_id").unique()
  else:
    subjects = pl.DataFrame({"subject_id": subject_ids}).unique()

  if len(prior_diag_df) == 0:
    out = subjects
    for col in ELIXHAUSER_FEATURE_NAMES:
      out = out.with_columns(pl.lit(0).cast(pl.Int8).alias(col))
    return out

  # Build subject -> category pairs with one-to-many support
  subject_cat_pairs: list[dict[str, Any]] = []
  for row in prior_diag_df.select(["subject_id", "icd_code", "icd_version"]).iter_rows(named=True):
    cats = map_icd_to_elixhauser_categories(str(row["icd_code"]), int(row["icd_version"]))
    for cat in cats:
      subject_cat_pairs.append({
        "subject_id": row["subject_id"],
        "elixhauser_category": cat,
        "flag": 1,
      })

  if not subject_cat_pairs:
    out = subjects
    for col in ELIXHAUSER_FEATURE_NAMES:
      out = out.with_columns(pl.lit(0).cast(pl.Int8).alias(col))
    return out

  present = pl.DataFrame(subject_cat_pairs).unique()
  wide = present.pivot(
    values="flag",
    index="subject_id",
    on="elixhauser_category",
    aggregate_function="max",
  )
  out = subjects.join(wide, on="subject_id", how="left")
  for category, col in zip(ELIXHAUSER_CATEGORIES, ELIXHAUSER_FEATURE_NAMES, strict=True):
    if category in out.columns:
      out = out.with_columns(pl.col(category).fill_null(0).cast(pl.Int8).alias(col))
      if category != col:
        out = out.drop(category)
    elif col in out.columns:
      out = out.with_columns(pl.col(col).fill_null(0).cast(pl.Int8))
    else:
      out = out.with_columns(pl.lit(0).cast(pl.Int8).alias(col))

  indicators = out.select(["subject_id", *ELIXHAUSER_FEATURE_NAMES])
  if apply_exclusions:
    indicators = apply_quan_exclusions(indicators)
  return indicators


def build_c0_features(
  cohort_df: pl.DataFrame,
  admissions_df: pl.DataFrame,
  diagnoses_df: pl.DataFrame,
) -> pl.DataFrame:
  """Build primary structured context ``C0`` = age + sex + prior Elixhauser.

  Elixhauser indicators use only diagnoses from admissions completed before the
  index ECG (`disch_dt < ecg_datetime`). Index-admission discharge diagnoses are
  excluded by that temporal gate.
  """
  required = {"subject_id", "study_id", "ecg_datetime", "age_at_ecg", "gender"}
  missing = required - set(cohort_df.columns)
  if missing:
    raise ValueError(f"Cohort frame missing required C0 columns: {sorted(missing)}")

  prior_adm = prior_completed_admissions(admissions_df, cohort_df)
  if len(prior_adm) == 0:
    prior_diag = diagnoses_df.clear()
  else:
    prior_diag = diagnoses_df.join(
      prior_adm.select(["subject_id", "hadm_id"]).unique(),
      on=["subject_id", "hadm_id"],
      how="inner",
    )

  elix = elixhauser_indicators_from_prior_diagnoses(
    prior_diag,
    subject_ids=cohort_df["subject_id"],
  )
  base = cohort_df.select(["subject_id", "study_id", "age_at_ecg", "gender"]).with_columns(
    (pl.col("gender").str.to_uppercase() == "F").cast(pl.Int8).alias("sex_female")
  )
  return base.join(elix, on="subject_id", how="left").select(
    [
      "subject_id",
      "study_id",
      "age_at_ecg",
      "sex_female",
      *ELIXHAUSER_FEATURE_NAMES,
    ]
  )
