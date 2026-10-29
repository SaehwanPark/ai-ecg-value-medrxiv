"""Machine-text ECG phenotypes.

Rules are in ``configs/ecg_phenotype_dictionary.yaml``. The study team approved
them under G2 at the current rule hash; they are an operational lexicon, not an
adjudicated diagnosis map. A hedged phrase such as ``consider acute ST elevation``
is unknown, not a definite overt finding and not a negative ECG. Missing text
is unknown. This module does not join mortality or model scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import re

import polars as pl
import yaml

SUMMARY_OVERT = "overt_major"
SUMMARY_NEGATIVE = "no_overt_major"
SUMMARY_UNKNOWN = "unknown"
SUMMARY_LABELS = (SUMMARY_OVERT, SUMMARY_NEGATIVE, SUMMARY_UNKNOWN)
DICTIONARY_STATUS = "clinical_review_pending"


@dataclass(frozen=True)
class PhraseRule:
  """One non-exclusive phrase rule."""

  rule_id: str
  patterns: tuple[re.Pattern[str], ...]
  downgrade_if: tuple[re.Pattern[str], ...] = ()
  downgrade_to: str | None = None
  require_any: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class PhenotypeDictionary:
  """Versioned draft lexicon."""

  version: str
  status: str
  uncertainty: tuple[re.Pattern[str], ...]
  negation: tuple[re.Pattern[str], ...]
  temporality: tuple[re.Pattern[str], ...]
  uninterpretable: tuple[re.Pattern[str], ...]
  overt_major: tuple[PhraseRule, ...]
  other_major: tuple[PhraseRule, ...]
  nonspecific: tuple[re.Pattern[str], ...]
  rule_hash: str


@dataclass(frozen=True)
class PhenotypeCall:
  """Study-level draft phenotype. Component flags are non-exclusive."""

  summary: str
  reason: str
  definite_overt: tuple[str, ...]
  uncertain_overt: tuple[str, ...]
  negated_overt: tuple[str, ...]
  other_major: tuple[str, ...]
  nonspecific: bool
  quality_warning: bool


def _compile_many(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
  return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


def _compile_rule(raw: dict[str, Any]) -> PhraseRule:
  return PhraseRule(
    rule_id=str(raw["id"]),
    patterns=_compile_many(list(raw["patterns"])),
    downgrade_if=_compile_many(list(raw.get("downgrade_if_patterns", []))),
    downgrade_to=raw.get("downgrade_to"),
    require_any=_compile_many(list(raw.get("require_any_patterns", []))),
  )


def load_phenotype_dictionary(path: Path) -> PhenotypeDictionary:
  """Load and fingerprint the draft YAML lexicon."""
  text = path.read_text(encoding="utf-8")
  raw = yaml.safe_load(text)
  if not isinstance(raw, dict):
    raise ValueError(f"Phenotype dictionary at {path} is not a mapping.")
  return PhenotypeDictionary(
    version=str(raw["version"]),
    status=str(raw["status"]),
    uncertainty=_compile_many(list(raw["uncertainty_patterns"])),
    negation=_compile_many(list(raw["negation_patterns"])),
    temporality=_compile_many(list(raw["temporality_patterns"])),
    uninterpretable=_compile_many(list(raw["uninterpretable_patterns"])),
    overt_major=tuple(_compile_rule(item) for item in raw["overt_major"]),
    other_major=tuple(_compile_rule(item) for item in raw["other_major"]),
    nonspecific=_compile_many(list(raw.get("nonspecific_patterns", []))),
    rule_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
  )


def _any_match(patterns: tuple[re.Pattern[str], ...], line: str) -> bool:
  return any(pattern.search(line) for pattern in patterns)


def _normalize_lines(lines: Sequence[str | None]) -> list[str]:
  cleaned: list[str] = []
  for line in lines:
    if line is None:
      continue
    text = " ".join(str(line).upper().split())
    if text:
      cleaned.append(text)
  return cleaned


def _boilerplate(line: str) -> bool:
  return "AGE NOT ENTERED" in line or "WITHOUT KNOWING PATIENT" in line


def classify_report(lines: Sequence[str | None], dictionary: PhenotypeDictionary) -> PhenotypeCall:
  """Classify one machine report. Hedged overt language stays unknown."""
  cleaned = _normalize_lines(lines)
  if not cleaned:
    return PhenotypeCall(SUMMARY_UNKNOWN, "missing_report", (), (), (), (), False, False)

  definite_overt: list[str] = []
  uncertain_overt: list[str] = []
  negated_overt: list[str] = []
  other_flags: list[str] = []
  nonspecific = False
  quality_warning = False
  informative = False

  for line in cleaned:
    if _any_match(dictionary.uninterpretable, line):
      quality_warning = True
    if _boilerplate(line):
      continue
    informative = True
    if _any_match(dictionary.nonspecific, line):
      nonspecific = True
    for rule in dictionary.overt_major:
      if not _any_match(rule.patterns, line):
        continue
      if _any_match(dictionary.negation, line):
        negated_overt.append(rule.rule_id)
        continue
      if rule.downgrade_to and _any_match(rule.downgrade_if, line):
        other_flags.append(rule.downgrade_to)
        continue
      if _any_match(dictionary.uncertainty, line) or _any_match(dictionary.temporality, line):
        # Historical or hedged acute language is not a definite emergency pattern.
        uncertain_overt.append(rule.rule_id)
        continue
      definite_overt.append(rule.rule_id)
    for rule in dictionary.other_major:
      if not _any_match(rule.patterns, line):
        continue
      if rule.require_any and not _any_match(rule.require_any, line):
        continue
      if _any_match(dictionary.negation, line) or _any_match(dictionary.uncertainty, line):
        continue
      other_flags.append(rule.rule_id)

  definite = tuple(sorted(set(definite_overt)))
  uncertain = tuple(sorted(set(uncertain_overt)))
  negated = tuple(sorted(set(negated_overt)))
  other = tuple(sorted(set(other_flags)))
  only_warnings = all(
    _any_match(dictionary.uninterpretable, line) or _boilerplate(line) for line in cleaned
  )
  if definite:
    summary, reason = SUMMARY_OVERT, "definite_overt_phrase"
  elif only_warnings or not informative:
    summary, reason = SUMMARY_UNKNOWN, "uninterpretable_or_boilerplate_only"
  elif uncertain:
    # A hedged or historical emergency phrase is not treated as a negative ECG.
    summary, reason = SUMMARY_UNKNOWN, "uncertain_or_historical_overt_language"
  else:
    summary, reason = SUMMARY_NEGATIVE, "no_definite_overt_phrase"
  return PhenotypeCall(summary, reason, definite, uncertain, negated, other, nonspecific, quality_warning)


def hedged_trigger_kinds(
  lines: Sequence[str | None],
  dictionary: PhenotypeDictionary,
) -> tuple[bool, bool]:
  """Why a report has hedged overt language: ``(uncertainty_wording, temporality_wording)``.

  Mirrors the branch in ``classify_report`` that files an overt rule match under
  ``uncertain_overt``, so a tier audit can say how much of a hedged tier is
  historical wording rather than "consider" wording. It does not change any call.
  """
  uncertainty = False
  temporality = False
  for line in _normalize_lines(lines):
    if _boilerplate(line) or _any_match(dictionary.negation, line):
      continue
    for rule in dictionary.overt_major:
      if not _any_match(rule.patterns, line):
        continue
      if rule.downgrade_to and _any_match(rule.downgrade_if, line):
        continue
      uncertainty = uncertainty or _any_match(dictionary.uncertainty, line)
      temporality = temporality or _any_match(dictionary.temporality, line)
  return uncertainty, temporality


def phenotype_frame(
  reports: pl.DataFrame,
  dictionary: PhenotypeDictionary,
  report_columns: list[str] | None = None,
) -> pl.DataFrame:
  """Classify each ``study_id`` without attaching outcomes or scores."""
  columns = report_columns or [f"report_{i}" for i in range(18)]
  missing = [col for col in ["study_id", *columns] if col not in reports.columns]
  if missing:
    raise ValueError(f"Report frame missing columns: {missing}")
  rows: list[dict[str, Any]] = []
  for record in reports.select(["study_id", *columns]).iter_rows(named=True):
    call = classify_report([record[col] for col in columns], dictionary)
    row: dict[str, Any] = {
      "study_id": record["study_id"],
      "phenotype_summary": call.summary,
      "phenotype_reason": call.reason,
      "definite_overt": "|".join(call.definite_overt),
      "uncertain_overt": "|".join(call.uncertain_overt),
      "other_major": "|".join(call.other_major),
      "nonspecific": call.nonspecific,
      "quality_warning": call.quality_warning,
      "dictionary_version": dictionary.version,
      "dictionary_hash": dictionary.rule_hash,
    }
    rows.append(row)
  return pl.DataFrame(rows)


def coverage_summary(phenotypes: pl.DataFrame) -> dict[str, Any]:
  """Aggregate counts. This summary has no mortality or risk-group fields."""
  n = phenotypes.height
  summary_counts = {
    str(row["phenotype_summary"]): int(row["len"])
    for row in phenotypes.group_by("phenotype_summary").len().iter_rows(named=True)
  }
  reason_counts = {
    str(row["phenotype_reason"]): int(row["len"])
    for row in phenotypes.group_by("phenotype_reason").len().iter_rows(named=True)
  }

  def _flag_counts(column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in phenotypes[column].to_list():
      if not value:
        continue
      for flag in str(value).split("|"):
        counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))

  return {
    "n_studies": n,
    "summary_counts": summary_counts,
    "reason_counts": reason_counts,
    "definite_overt_counts": _flag_counts("definite_overt"),
    "uncertain_overt_counts": _flag_counts("uncertain_overt"),
    "other_major_counts": _flag_counts("other_major"),
    "n_nonspecific": int(phenotypes.filter(pl.col("nonspecific")).height),
    "n_quality_warning": int(phenotypes.filter(pl.col("quality_warning")).height),
    "joined_to_mortality": False,
    "joined_to_model_scores": False,
    "split_policy": "development_only",
  }


def run_development_phenotype_coverage(
  cohort_path: Path,
  measurements_path: Path,
  dictionary_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
) -> dict[str, Any]:
  """Classify development index ECGs only. Refuses to attach outcomes."""
  dictionary = load_phenotype_dictionary(dictionary_path)
  if dictionary.status != DICTIONARY_STATUS:
    raise ValueError(
      f"Dictionary status is {dictionary.status}; coverage expects {DICTIONARY_STATUS}."
    )
  cohort = pl.read_parquet(cohort_path)
  if "split" not in cohort.columns:
    raise ValueError("Cohort is missing split. Refusing to classify an unlabeled population.")
  dev_ids = cohort.filter(pl.col("split") == "dev").select(["study_id"])
  report_columns = [f"report_{i}" for i in range(18)]
  reports = (
    pl.scan_csv(measurements_path, infer_schema_length=500)
    .select(["study_id", *report_columns])
    .join(dev_ids.lazy(), on="study_id", how="inner")
    .collect()
  )
  if reports.height != dev_ids.height:
    missing = dev_ids.height - reports.height
    raise ValueError(f"Development coverage is missing {missing} machine-report rows.")
  phenotypes = phenotype_frame(reports, dictionary)
  summary = coverage_summary(phenotypes)
  summary["dictionary_version"] = dictionary.version
  summary["dictionary_hash"] = dictionary.rule_hash
  summary["source_sha"] = source_sha
  summary["n_development_ids"] = dev_ids.height
  output_dir.mkdir(parents=True, exist_ok=True)
  phenotypes.write_parquet(output_dir / "ecg_phenotype_v1.parquet")
  (output_dir / "ecg_phenotype_mapping_audit.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  report = render_phenotype_results(summary, dictionary, source_sha)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  (output_dir / "phenotype_results.md").write_text(report, encoding="utf-8")
  return summary


def render_phenotype_results(summary: dict[str, Any], dictionary: PhenotypeDictionary, source_sha: str) -> str:
  """Review memo for the draft lexicon. It is not a mortality analysis."""
  counts = summary["summary_counts"]
  n = int(summary["n_studies"])

  def _count(label: str) -> int:
    return int(counts.get(label, 0))

  lines = [
    "# Overt ECG phenotype coverage",
    "",
    f"Dictionary version: `{dictionary.version}`. Status: `{dictionary.status}`.",
    f"Rule hash: `{dictionary.rule_hash}`. Source SHA: `{source_sha}`.",
    "",
    "This is a development-split coverage audit of machine-generated report lines. "
    "It was not joined to 30-day mortality, A/B scores, or the locked test. "
    "The study team approved this lexicon under G2 on 2026-10-05, bound to the rule hash above; "
    "Jennifer Lewey's clinical review is post hoc. `unknown` is not a normal ECG, "
    "and `no_overt_major` means the prespecified overt phrases were absent, not that the tracing is normal.",
    "",
    "Hedged acute language (`consider`, `possible`, `cannot rule out`, age-undetermined infarct wording) "
    "does not count as a definite overt finding. The common machine line "
    "`*** CONSIDER ACUTE ST ELEVATION MI ***` therefore stays in `unknown`.",
    "",
    f"Development studies classified: {n:,}.",
    f"`overt_major` {_count(SUMMARY_OVERT):,}; "
    f"`no_overt_major` {_count(SUMMARY_NEGATIVE):,}; "
    f"`unknown` {_count(SUMMARY_UNKNOWN):,}.",
    "",
    "## Reasons",
    "",
    "```",
    json.dumps(summary["reason_counts"], indent=2, sort_keys=True),
    "```",
    "",
    "## Definite overt flags",
    "",
    "```",
    json.dumps(summary["definite_overt_counts"], indent=2, sort_keys=True),
    "```",
    "",
    "## Uncertain overt flags",
    "",
    "```",
    json.dumps(summary["uncertain_overt_counts"], indent=2, sort_keys=True),
    "```",
    "",
    "## Other major flags",
    "",
    "These flags are non-exclusive and are not collapsed into `overt_major`.",
    "",
    "```",
    json.dumps(summary["other_major_counts"], indent=2, sort_keys=True),
    "```",
    "",
    f"Nonspecific wording was present in {summary['n_nonspecific']:,} development studies. "
    f"A data-quality warning was present in {summary['n_quality_warning']:,}.",
    "",
    "Locked-test analyses may use this lexicon only at the approved rule hash. "
    "The blinded post-hoc review sample is drawn without showing A/B scores, mortality, or discordance group.",
    "",
  ]
  return "\n".join(lines)
