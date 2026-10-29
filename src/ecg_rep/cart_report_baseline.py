"""LM1 stronger automated ECG comparator from machine-generated cart report text.

Representation A'' concatenates the existing A' cart global-measurement features with
indicators for exact, normalized machine-statement lines read from
``machine_measurements.csv`` ``report_0``..``report_17``. The vocabulary is fitted on
development patients only, using availability thresholds rather than outcome
association. Cardiologist-authored interpretations, clinical notes, diagnosis codes,
post-ECG data, and outcome labels are never read by the feature builder.

A'' is a rich automated ECG-cart output comparator, not a pure waveform-only
representation: report statements can also encode demographic-entry and acquisition
workflow information. Frozen vocabulary/process phrases are not redesigned using
performance. "ECG-only" model keys denote absence of the additional C0 block, not
absence of demographic or workflow information in the cart output itself.

Analysis status: ``post hoc extension``. This module never modifies the primary
A/A'/B artifacts or the frozen locked-test predictions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import hashlib
import json
import re

import numpy as np
import numpy.typing as npt
import polars as pl
import yaml

from ecg_rep.machine_baseline import (
  MACHINE_RAW_COLUMNS,
  derive_machine_features,
  impute_with_dev_medians,
  machine_feature_columns,
)

REPORT_COLUMN_COUNT = 18
REPORT_COLUMNS: tuple[str, ...] = tuple(f"report_{i}" for i in range(REPORT_COLUMN_COUNT))

# Frozen boilerplate strings already recognized by the repository
# (``ecg_phenotypes._boilerplate``) plus the identical per-ECG duplicates guard.
# Additions must be versioned in configs/cart_report_comparator_v1.yaml.
DEFAULT_BOILERPLATE: tuple[str, ...] = (
  "AGE NOT ENTERED",
  "WITHOUT KNOWING PATIENT",
)

FALLBACK_HASH_DIM = 4096
FALLBACK_NGRAM_RANGE: tuple[int, int] = (1, 2)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CartReportConfig:
  """Versioned frozen configuration for the A'' cart-report representation.

  Field defaults are the recommended LM1 protocol values; the authoritative copy
  lives in ``configs/cart_report_comparator_v1.yaml`` and is committed with a
  SHA-256 hash in the feasibility report.
  """

  version: str = "cart_report_comparator_v1"
  min_statement_dev_patients: int = 100
  max_statement_dev_frequency: float = 0.995
  coverage_gate_min_fraction: float = 0.70
  max_vocabulary_features: int = 5000
  boilerplate: tuple[str, ...] = DEFAULT_BOILERPLATE
  fallback_hash_dim: int = FALLBACK_HASH_DIM
  fallback_ngram_min: int = FALLBACK_NGRAM_RANGE[0]
  fallback_ngram_max: int = FALLBACK_NGRAM_RANGE[1]
  representation_type: str = "exact_statements"  # or "hash_fallback"
  missing_data_policy: str = "dev_median_impute_a_prime_zero_report_block"
  # Populated after the frozen vocabulary is chosen (exact-statement mode).
  vocabulary: tuple[str, ...] = field(default_factory=tuple)

  def to_dict(self) -> dict[str, Any]:
    return {
      "version": self.version,
      "min_statement_dev_patients": self.min_statement_dev_patients,
      "max_statement_dev_frequency": self.max_statement_dev_frequency,
      "coverage_gate_min_fraction": self.coverage_gate_min_fraction,
      "max_vocabulary_features": self.max_vocabulary_features,
      "boilerplate": list(self.boilerplate),
      "fallback_hash_dim": self.fallback_hash_dim,
      "fallback_ngram_min": self.fallback_ngram_min,
      "fallback_ngram_max": self.fallback_ngram_max,
      "representation_type": self.representation_type,
      "missing_data_policy": self.missing_data_policy,
      "vocabulary_size": len(self.vocabulary),
      "vocabulary_sha256": vocabulary_hash(self.vocabulary) if self.vocabulary else None,
    }


def load_cart_report_config(path: Path) -> CartReportConfig:
  """Load the versioned config; unknown keys are rejected to catch typos.

  An embedded ``vocabulary`` block (the frozen dev-derived statements) and a
  ``frozen_vocabulary_sha256`` are cross-checked against each other when present.
  """
  raw = yaml.safe_load(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError(f"Cart report config at {path} is not a mapping.")
  known = set(CartReportConfig.__dataclass_fields__)
  known.add("frozen_vocabulary_sha256")
  unknown = set(raw) - known
  if unknown:
    raise ValueError(f"Unknown cart report config keys: {sorted(unknown)}")
  vocabulary = tuple(str(s) for s in raw.get("vocabulary", []))
  frozen_hash = raw.get("frozen_vocabulary_sha256")
  if frozen_hash is not None and vocabulary:
    if vocabulary_hash(vocabulary) != str(frozen_hash):
      raise ValueError("Embedded vocabulary does not match frozen_vocabulary_sha256.")
  elif frozen_hash is not None and not vocabulary and str(frozen_hash):
    raise ValueError("Config records frozen_vocabulary_sha256 without an embedded vocabulary.")
  return CartReportConfig(
    version=str(raw["version"]),
    min_statement_dev_patients=int(raw["min_statement_dev_patients"]),
    max_statement_dev_frequency=float(raw["max_statement_dev_frequency"]),
    coverage_gate_min_fraction=float(raw["coverage_gate_min_fraction"]),
    max_vocabulary_features=int(raw["max_vocabulary_features"]),
    boilerplate=tuple(str(s) for s in raw["boilerplate"]),
    fallback_hash_dim=int(raw["fallback_hash_dim"]),
    fallback_ngram_min=int(raw["fallback_ngram_min"]),
    fallback_ngram_max=int(raw["fallback_ngram_max"]),
    representation_type=str(raw.get("representation_type", "exact_statements")),
    missing_data_policy=str(raw.get("missing_data_policy", "dev_median_impute_a_prime_zero_report_block")),
    vocabulary=vocabulary,
  )


def config_sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def vocabulary_hash(vocabulary: Sequence[str]) -> str:
  """SHA-256 over the newline-joined lexical vocabulary; order-sensitive."""
  return hashlib.sha256("\n".join(vocabulary).encode("utf-8")).hexdigest()


def normalize_report_lines(lines: Sequence[str | None], boilerplate: Sequence[str] = DEFAULT_BOILERPLATE) -> list[str]:
  """Uppercase, collapse whitespace, drop empties and boilerplate, deduplicate in order.

  Deterministic pure function of the input text and the frozen boilerplate list.
  """
  blocked = {str(b).upper() for b in boilerplate}
  out: list[str] = []
  seen: set[str] = set()
  for line in lines:
    if line is None:
      continue
    text = _WHITESPACE_RE.sub(" ", str(line)).strip().upper()
    if not text or text in blocked or text in seen:
      continue
    seen.add(text)
    out.append(text)
  return out


def raw_non_empty_count(lines: Sequence[str | None], boilerplate: Sequence[str] = DEFAULT_BOILERPLATE) -> int:
  """Count non-empty normalized lines before deduplication and before vocabulary filtering.

  Boilerplate lines are removed first, consistent with the frozen text pipeline.
  """
  blocked = {str(b).upper() for b in boilerplate}
  total = 0
  for line in lines:
    if line is None:
      continue
    text = _WHITESPACE_RE.sub(" ", str(line)).strip().upper()
    if text and text not in blocked:
      total += 1
  return total


def build_report_vocabulary(
  reports: pl.DataFrame,
  config: CartReportConfig,
) -> tuple[list[str], dict[str, Any]]:
  """Fit the exact-statement vocabulary on development rows only.

  ``reports`` must contain ``study_id``, ``subject_id`` and the 18 report columns.
  A statement is retained when it appears for at least ``min_statement_dev_patients``
  distinct development patients and for at most
  ``max_statement_dev_frequency`` of development patients. The vocabulary is
  ordered lexically for determinism. Outcome columns must not be present.
  """
  _require_report_frame(reports)
  patient_statement = dev_patient_statement_sets(reports, config.boilerplate)
  n_dev_patients = len(patient_statement)
  counts: Counter[str] = Counter()
  for statements in patient_statement.values():
    counts.update(statements)
  threshold = config.max_statement_dev_frequency * n_dev_patients
  vocab = sorted(
    statement
    for statement, count in counts.items()
    if count >= config.min_statement_dev_patients and count <= threshold
  )
  candidate_thresholds = (10, 25, 50, 100, 200, 500, 1000)
  coverage_by_threshold = {
    str(t): int(sum(1 for c in counts.values() if c >= t)) for t in candidate_thresholds
  }
  diagnostics = {
    "n_dev_patients": n_dev_patients,
    "n_unique_normalized_statements": len(counts),
    "n_retained_statements": len(vocab),
    "min_statement_dev_patients": config.min_statement_dev_patients,
    "max_statement_dev_frequency": config.max_statement_dev_frequency,
    "max_patient_count_inclusive": int(threshold),
    "n_statements_above_candidate_thresholds": coverage_by_threshold,
    "top_statements_by_dev_patients": dict(
      sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:25]
    ),
  }
  return vocab, diagnostics


def dev_patient_statement_sets(
  reports: pl.DataFrame,
  boilerplate: Sequence[str] = DEFAULT_BOILERPLATE,
) -> dict[int, set[str]]:
  """Map subject_id -> set of normalized statements across that patient's development ECGs."""
  patient_map: dict[int, set[str]] = {}
  for row in reports.iter_rows(named=True):
    subject = int(row["subject_id"])
    statements = normalize_report_lines([row[c] for c in REPORT_COLUMNS], boilerplate)
    patient_map.setdefault(subject, set()).update(statements)
  return patient_map


class ReportFeatureBuilder:
  """Frozen transformer from report columns to the A'' report feature block.

  The builder is fitted on development patients only. At transform time, unseen or
  rare statements map into the ``rare_unseen_count`` scalar and never alter the
  vocabulary or the feature order.
  """

  META_FEATURES: tuple[str, ...] = (
    "report_n_nonempty_lines",
    "report_n_retained_statements",
    "report_any_present",
    "report_rare_unseen_count",
  )

  def __init__(
    self,
    config: CartReportConfig,
    vocabulary: Sequence[str] | None = None,
  ) -> None:
    self.config = config
    if config.representation_type == "exact_statements":
      self.vocabulary: tuple[str, ...] | None = (
        tuple(sorted(vocabulary)) if vocabulary is not None else tuple(sorted(config.vocabulary))
      )
      if not self.vocabulary:
        raise ValueError("Exact-statement mode requires a frozen vocabulary.")
      self._vocab_index = {statement: i for i, statement in enumerate(self.vocabulary)}
    elif config.representation_type == "hash_fallback":
      self.vocabulary = None
      if vocabulary is not None:
        raise ValueError("Hash-fallback mode does not take a vocabulary.")
    else:
      raise ValueError(f"Unknown representation_type: {config.representation_type}")

  @property
  def feature_columns(self) -> list[str]:
    """Stable ordered feature names: lexical statements first, then meta features."""
    if self.config.representation_type == "hash_fallback":
      return [f"report_hash_{i}" for i in range(self.config.fallback_hash_dim)] + list(self.META_FEATURES)
    assert self.vocabulary is not None
    return [f"stmt::{s}" for s in self.vocabulary] + list(self.META_FEATURES)

  def transform(self, reports: pl.DataFrame) -> pl.DataFrame:
    """Return one row per ``study_id`` with binary statement indicators plus meta counts."""
    _require_report_frame(reports)
    blocked = tuple(self.config.boilerplate)
    features: list[dict[str, float]] = []
    for row in reports.iter_rows(named=True):
      lines = [row[c] for c in REPORT_COLUMNS]
      n_nonempty = raw_non_empty_count(lines, blocked)
      normalized = normalize_report_lines(lines, blocked)
      row_features = dict.fromkeys(self.feature_columns, 0.0)
      rare = 0
      if self.config.representation_type == "exact_statements":
        assert self.vocabulary is not None
        for statement in normalized:
          idx = self._vocab_index.get(statement)
          if idx is None:
            rare += 1
          else:
            row_features[f"stmt::{statement}"] = 1.0
        n_retained = int(sum(row_features[f"stmt::{s}"] for s in self.vocabulary))
      else:
        for value in _hash_word_ngrams(normalized, self.config):
          row_features[f"report_hash_{value[0]}"] += float(value[1])
        n_retained = int(sum(1 for _ in normalized))
      row_features["report_n_nonempty_lines"] = float(n_nonempty)
      row_features["report_n_retained_statements"] = float(n_retained)
      row_features["report_any_present"] = 1.0 if normalized else 0.0
      row_features["report_rare_unseen_count"] = float(rare)
      features.append(row_features)
    key_cols = [c for c in ("subject_id", "study_id") if c in reports.columns]
    frame = pl.DataFrame(features, schema=self.feature_columns).with_columns(
      reports.select(key_cols)
    )
    return frame.select([*key_cols, *self.feature_columns])


def _hash_word_ngrams(lines: Sequence[str], config: CartReportConfig) -> list[tuple[int, int]]:
  """Deterministic hashed word 1-2 gram counts; no IDF, no label-based selection."""
  counts: Counter[int] = Counter()
  for line in lines:
    words = line.split()
    for n in range(config.fallback_ngram_min, config.fallback_ngram_max + 1):
      for start in range(len(words) - n + 1):
        gram = " ".join(words[start : start + n])
        idx = int.from_bytes(hashlib.sha256(gram.encode("utf-8")).digest()[:8], "big") % config.fallback_hash_dim
        counts[idx] += 1
  return sorted(counts.items())


def _require_report_frame(reports: pl.DataFrame) -> None:
  """Reject frames with outcome-like or score-like columns before any feature work."""
  required = {"study_id", *REPORT_COLUMNS}
  missing = required - set(reports.columns)
  if missing:
    raise ValueError(f"Report frame missing required columns: {sorted(missing)}")
  forbidden = {
    col
    for col in reports.columns
    if col not in required and col != "subject_id"
  }
  if forbidden:
    raise ValueError(
      "Feature builder accepts only study_id, subject_id, and report columns; "
      f"refusing extra columns: {sorted(forbidden)}"
    )


def run_cart_report_feasibility(
  cohort_path: Path,
  measurements_path: Path,
  config_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
) -> dict[str, Any]:
  """Development-only inventory of machine reports; never joins mortality or test rows."""
  config = load_cart_report_config(config_path)
  cohort = pl.read_parquet(cohort_path)
  if "split" not in cohort.columns:
    raise ValueError("Cohort is missing split.")
  dev_ids = cohort.filter(pl.col("split") == "dev").select(["subject_id", "study_id"])
  machine = pl.read_csv(
    measurements_path,
    columns=["subject_id", "study_id", *REPORT_COLUMNS],
    infer_schema_length=100_000,
  )
  reports = machine.join(dev_ids, on=["subject_id", "study_id"], how="inner")
  if reports.height != dev_ids.height:
    missing = dev_ids.height - reports.height
    raise ValueError(f"Development machine-report rows missing for {missing} cohort ECGs.")

  n_dev = reports.height
  n_missing_any = int(
    reports.select(
      pl.all_horizontal(pl.col(c).is_null() for c in REPORT_COLUMNS).sum()
    ).item()
    or 0
  )
  line_counts: list[int] = []
  boilerplate_hits = 0
  blocked = {str(b).upper() for b in config.boilerplate}
  # Single pass: per-ECG line counts plus patient-level statement sets.
  patient_statement: dict[int, set[str]] = {}
  for row in reports.iter_rows(named=True):
    lines = [row[c] for c in REPORT_COLUMNS]
    line_counts.append(raw_non_empty_count(lines, config.boilerplate))
    boilerplate_hits += sum(
      1
      for line in lines
      if line is not None and _WHITESPACE_RE.sub(" ", str(line)).strip().upper() in blocked
    )
    statements = normalize_report_lines(lines, config.boilerplate)
    patient_statement.setdefault(int(row["subject_id"]), set()).update(statements)
  vocab, vocab_diagnostics = build_report_vocabulary(reports, config)

  # Coverage gate: fraction of development patients with at least one retained
  # (in-vocabulary) exact statement, per the LM1 fallback rule.
  vocab_set = set(vocab)
  n_covered = sum(1 for statements in patient_statement.values() if any(s in vocab_set for s in statements))
  coverage = n_covered / len(patient_statement) if patient_statement else 0.0
  gates = {
    "coverage_gate_fraction": coverage,
    "coverage_gate_threshold": config.coverage_gate_min_fraction,
    "coverage_gate_pass": coverage >= config.coverage_gate_min_fraction,
    "vocabulary_size": len(vocab),
    "vocabulary_gate_max": config.max_vocabulary_features,
    "vocabulary_gate_pass": len(vocab) <= config.max_vocabulary_features,
  }
  chosen = "exact_statements" if gates["coverage_gate_pass"] and gates["vocabulary_gate_pass"] else "hash_fallback"

  results: dict[str, Any] = {
    "status": "complete",
    "development_only": True,
    "outcomes_joined": False,
    "config_version": config.version,
    "config_sha256": config_sha256(config_path),
    "source_sha": source_sha,
    "n_development_ecgs": n_dev,
    "n_missing_all_reports": n_missing_any,
    "mean_nonempty_lines": float(np.mean(line_counts)) if line_counts else 0.0,
    "median_nonempty_lines": float(np.median(line_counts)) if line_counts else 0.0,
    "max_nonempty_lines": int(np.max(line_counts)) if line_counts else 0,
    "boilerplate_line_hits": int(boilerplate_hits),
    "vocabulary_diagnostics": vocab_diagnostics,
    "gates": gates,
    "chosen_representation": chosen,
    "vocabulary_sha256": vocabulary_hash(vocab) if vocab else None,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "cart_report_feasibility.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  report = _render_feasibility_report(results)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  return results


def _render_feasibility_report(results: dict[str, Any]) -> str:
  gates = results["gates"]
  lines = [
    "# LM1 cart-report comparator feasibility (development only)",
    "",
    "```yaml",
    f"status: complete",
    f"producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {results['source_sha']}",
    "analysis_status: post_hoc_extension",
    f"inputs: [{results['config_version']}]",
    f"decisions: [representation={results['chosen_representation']}]",
    f"evidence: [cart_report_feasibility.json]",
    "unresolved: []",
    "verification: [development-only inventory; outcomes never joined]",
    "next_action: freeze config and evaluate A'' on the locked test (PR LM1b)",
    "```",
    "",
    "This audit read development-split ECGs only. No outcome column, no validation or",
    "locked-test row, and no model score entered any number below.",
    "",
    f"- Development ECGs inventoried: {results['n_development_ecgs']:,}",
    f"- ECGs with every report field missing: {results['n_missing_all_reports']:,}",
    f"- Non-empty report lines per ECG (mean/median/max): {results['mean_nonempty_lines']:.2f} / "
    f"{results['median_nonempty_lines']:.0f} / {results['max_nonempty_lines']}",
    f"- Boilerplate line hits: {results['boilerplate_line_hits']:,}",
    "",
    "## Objective gates",
    "",
    f"- Coverage gate: {gates['coverage_gate_fraction']:.4f} of development ECGs have at least one "
    f"retained statement (threshold {gates['coverage_gate_threshold']:.2f}): "
    f"{'PASS' if gates['coverage_gate_pass'] else 'FAIL'}",
    f"- Vocabulary gate: {gates['vocabulary_size']:,} retained statements "
    f"(maximum {gates['vocabulary_gate_max']:,}): {'PASS' if gates['vocabulary_gate_pass'] else 'FAIL'}",
    "",
    f"Chosen representation: `{results['chosen_representation']}`.",
    f"Vocabulary SHA-256: `{results['vocabulary_sha256']}`.",
    f"Config SHA-256: `{results['config_sha256']}`.",
    "",
    "The choice between exact statements and hashing used availability gates only,",
    "never mortality association.",
    "",
  ]
  return "\n".join(lines)


# ---------------------------------------------------------------------------
# LM1b: frozen locked-test evaluation under the frozen config/vocabulary.
# ---------------------------------------------------------------------------


def load_freeze(config_path: Path, feasibility_path: Path) -> CartReportConfig:
  """Verify the frozen config matches the committed feasibility choice.

  The vocabulary itself is verified later, after the deterministic dev-only
  derivation is recomputed and hashed against the feasibility artifact.
  """
  config = load_cart_report_config(config_path)
  feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
  if feasibility.get("chosen_representation") != config.representation_type:
    raise ValueError(
      "Config representation_type does not match the frozen feasibility choice: "
      f"{config.representation_type} != {feasibility.get('chosen_representation')}"
    )
  if not feasibility.get("vocabulary_sha256") and config.representation_type == "exact_statements":
    raise ValueError("Feasibility artifact is missing the frozen vocabulary hash.")
  return config


def freeze_vocabulary(
  config: CartReportConfig,
  dev_reports: pl.DataFrame,
  frozen_vocab_hash: str,
) -> CartReportConfig:
  """Recompute the dev-only vocabulary and verify it against the frozen hash.

  The derivation reads development rows and availability counts only; the hash
  check guarantees the evaluation vocabulary is byte-identical to the one frozen
  in the feasibility artifact before any outcome evaluation.
  """
  if config.representation_type != "exact_statements":
    return config
  if config.vocabulary:
    if vocabulary_hash(config.vocabulary) != frozen_vocab_hash:
      raise ValueError("Config-embedded vocabulary hash does not match the frozen feasibility vocabulary.")
    return config
  vocab, _diag = build_report_vocabulary(dev_reports, config)
  if vocabulary_hash(vocab) != frozen_vocab_hash:
    raise ValueError(
      "Recomputed development vocabulary does not match the frozen feasibility hash; "
      "refusing to evaluate against a silently different vocabulary."
    )
  import dataclasses

  return dataclasses.replace(config, vocabulary=tuple(vocab))


def build_cart_report_frame(
  cohort_path: Path,
  machine: pl.DataFrame,
  c0_path: Path,
  reps_dir: Path,
  config: CartReportConfig,
) -> tuple[pl.DataFrame, ReportFeatureBuilder]:
  """Assemble the full analysis frame: cohort + paired-valid + machine reports + C0 + A + B.

  Report features are built for every machine_measurements row from report text
  only; the ``cart_report_present`` flag marks rows with a machine_measurements
  row (the A''-valid population is defined from it before any outcome evaluation).
  ``config.vocabulary`` must already be frozen/verified (see ``freeze_vocabulary``).
  """
  from ecg_rep.evaluate import C0_FEATURE_COLS, _join_representation_matrix  # pyright: ignore[reportPrivateUsage]

  cohort = pl.read_parquet(cohort_path)
  paired = pl.read_parquet(reps_dir / "representation_pair_validity.parquet").filter(
    pl.col("representation_pair_valid")
  )
  c0 = pl.read_parquet(c0_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]

  # Report features for every machine_measurements row (no outcomes present here).
  builder = ReportFeatureBuilder(config)
  report_features = builder.transform(machine.select(["subject_id", "study_id", *REPORT_COLUMNS]))

  frame = (
    cohort.join(paired.select(["subject_id", "study_id"]), on=["subject_id", "study_id"])
    .join(
      machine.select(["subject_id", "study_id", *MACHINE_RAW_COLUMNS]),
      on=["subject_id", "study_id"],
      how="left",
    )
    .join(c0.select(["subject_id", "study_id", *c0_cols]), on=["subject_id", "study_id"])
  )
  frame, medians = impute_with_dev_medians(derive_machine_features(frame))
  frame = _join_representation_matrix(
    frame,
    pl.read_parquet(reps_dir / "ciis_features_meta.parquet"),
    np.load(reps_dir / "ciis_features.npy"),
    "za",
    "representation_a_valid",
  )
  frame = _join_representation_matrix(
    frame,
    pl.read_parquet(reps_dir / "dbeta_embedding_meta.parquet"),
    np.load(reps_dir / "dbeta_embeddings.npy"),
    "zb",
    "representation_b_valid",
  ).filter(pl.col("representation_a_valid") & pl.col("representation_b_valid"))
  frame = frame.join(report_features, on=["subject_id", "study_id"], how="left").with_columns(
    pl.col(report_features.columns[1]).is_not_null().alias("cart_report_present")
  )
  frame = frame.with_columns(
    *[pl.col(c).fill_null(0.0) for c in builder.feature_columns]
  )
  return frame, builder


def population_hash(
  subject_ids: npt.NDArray[np.int64], study_ids: npt.NDArray[np.int64]
) -> str:
  """Stable, order-independent SHA-256 over sorted ``(subject_id, study_id)`` pairs.

  Format: one ``"<subject_id>,<study_id>"`` line per pair, lexically sorted, joined
  by newlines, UTF-8 encoded. The hash identifies the patient set without exposing
  patient-level data and lets independent runs verify population identity.
  """
  if len(subject_ids) != len(study_ids):
    raise ValueError("Population hash requires equal-length id arrays.")
  pairs = sorted(
    (int(s), int(d)) for s, d in zip(np.asarray(subject_ids), np.asarray(study_ids))
  )
  payload = "\n".join(f"{s},{d}" for s, d in pairs)
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CAPACITY_GRID: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10, 0.20)


def run_cart_report_evaluation(
  cohort_path: Path,
  measurements_path: Path,
  c0_path: Path,
  reps_dir: Path,
  config_path: Path,
  feasibility_path: Path,
  reference_ecg_only_path: Path,
  reference_context_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = 1000,
) -> dict[str, Any]:
  """LM1b frozen evaluation: fit A'' probes under the dev/val protocol on the
  A''-valid paired-valid surface and report the six required contrasts plus
  capacity rows. Uses the frozen vocabulary verified against the feasibility
  artifact; never overwrites primary artifacts.
  """
  from ecg_rep.analysis import capacity_evaluation
  from ecg_rep.evaluate import C0_FEATURE_COLS, _fit_probe  # pyright: ignore[reportPrivateUsage]
  from ecg_rep.probe import bootstrap_paired_difference, evaluate_predictions, metric_bundle_to_dict

  config = load_freeze(config_path, feasibility_path)
  feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
  machine = pl.read_csv(
    measurements_path,
    columns=["subject_id", "study_id", *MACHINE_RAW_COLUMNS, *REPORT_COLUMNS],
    infer_schema_length=100_000,
  )
  # Freeze-verify the vocabulary against the committed feasibility hash before
  # any outcome row is touched (derivation is development-only and availability-based).
  cohort_check = pl.read_parquet(cohort_path)
  dev_ids = cohort_check.filter(pl.col("split") == "dev").select(["subject_id", "study_id"])
  dev_reports = machine.join(dev_ids, on=["subject_id", "study_id"], how="inner")
  config = freeze_vocabulary(config, dev_reports, str(feasibility["vocabulary_sha256"]))
  del cohort_check, dev_reports

  frame, builder = build_cart_report_frame(cohort_path, machine, c0_path, reps_dir, config)
  c0_cols = [
    c
    for c in C0_FEATURE_COLS
    if c in frame.columns
  ]
  za = [c for c in frame.columns if c.startswith("za_")]
  zb = [c for c in frame.columns if c.startswith("zb_")]
  ap = machine_feature_columns()
  report_cols = builder.feature_columns
  a2 = [*ap, *report_cols]

  # A''-valid surface: paired-valid rows with a machine_measurements row present.
  missing_by_split = {
    split: int(
      frame.filter((pl.col("split") == split) & ~pl.col("cart_report_present")).height
    )
    for split in ("dev", "val", "test")
  }
  surface = frame.filter(pl.col("cart_report_present"))
  n_surface = {
    split: int(surface.filter(pl.col("split") == split).height) for split in ("dev", "val", "test")
  }
  n_surface_events = {
    split: int(surface.filter(pl.col("split") == split)["mortality_30d"].sum())
    for split in ("dev", "val", "test")
  }

  # Same-population models: every model (re)fit on the identical A''-valid surface.
  specs: dict[str, list[str]] = {
    "LM1_A2_ECG": a2,
    "LM1_B_ECG": zb,
    "LM1_A2_B_ECG": [*a2, *zb],
    "LM1_AP_ECG": ap,
    "LM1_A_ECG": za,
    "LM1_A_AP_ECG": [*za, *ap],
    "LM1_C_A2": [*c0_cols, *a2],
    "LM1_C_B": [*c0_cols, *zb],
    "LM1_C_A2_B": [*c0_cols, *a2, *zb],
    "LM1_C_AP": [*c0_cols, *ap],
  }
  preds: dict[str, npt.NDArray[np.float64]] = {}
  models: dict[str, Any] = {}
  test_ids: pl.DataFrame | None = None
  for name, cols in specs.items():
    probe, _dev, _val, test = _fit_probe(surface, cols, model_name=name)
    p = probe.predict_proba(test.x)
    if test_ids is None:
      test_ids = pl.DataFrame({
        "subject_id": test.subject_ids,
        "study_id": test.study_ids,
        "y": test.y,
      })
    preds[name] = p
    models[name] = {
      "n_features": len(cols),
      "best_c": probe.best_c,
      "test_metrics": metric_bundle_to_dict(evaluate_predictions(test.y, p)),
    }
  assert test_ids is not None
  test_population_sha256 = population_hash(
    test_ids["subject_id"].to_numpy(), test_ids["study_id"].to_numpy()
  )

  # Frozen reference predictions for the same locked-test rows (never refit).
  joined = (
    test_ids.with_columns(*(pl.Series(k, v) for k, v in preds.items()))
    .join(
      pl.read_parquet(reference_ecg_only_path).select(["subject_id", "study_id", "p_a_ecg", "p_b_ecg"]),
      on=["subject_id", "study_id"],
    )
    .join(
      pl.read_parquet(reference_context_path).select(["subject_id", "study_id", "p_c", "p_a", "p_b"]),
      on=["subject_id", "study_id"],
    )
  )
  if joined.height != test_ids.height:
    raise ValueError(f"Frozen predictions cover {joined.height} of {test_ids.height} locked-test rows.")

  def col(name: str) -> npt.NDArray[np.float64]:
    return joined[name].to_numpy().astype(np.float64)

  y = joined["y"].to_numpy().astype(np.int64)
  n_test = int(len(y))
  n_events = int(y.sum())
  population_identical = n_test == 25430 and n_events == 924

  def contrast(name: str, ref: str, new: str, n_boot: int) -> dict[str, Any]:
    c = bootstrap_paired_difference(
      y, col(ref), col(new), metric="auroc", n_bootstraps=n_boot, seed=42
    )
    c_prc = bootstrap_paired_difference(
      y, col(ref), col(new), metric="auprc", n_bootstraps=n_boot, seed=42
    )
    return {
      "reference": ref,
      "model": new,
      "auroc": {
        "point_estimate": c.point_estimate,
        "ci_lower": c.ci_lower,
        "ci_upper": c.ci_upper,
      },
      "auprc": {
        "point_estimate": c_prc.point_estimate,
        "ci_lower": c_prc.ci_lower,
        "ci_upper": c_prc.ci_upper,
      },
    }

  contrasts = {
    "a2_minus_ap_ecg": contrast("a2_minus_ap_ecg", "LM1_AP_ECG", "LM1_A2_ECG", n_bootstraps),
    "b_minus_a2_ecg": contrast("b_minus_a2_ecg", "LM1_A2_ECG", "LM1_B_ECG", n_bootstraps),
    "a2_b_minus_b_ecg": contrast("a2_b_minus_b_ecg", "LM1_B_ECG", "LM1_A2_B_ECG", n_bootstraps),
    "c_a2_minus_c_ap": contrast("c_a2_minus_c_ap", "LM1_C_AP", "LM1_C_A2", n_bootstraps),
    "c_b_minus_c_a2": contrast("c_b_minus_c_a2", "LM1_C_A2", "LM1_C_B", n_bootstraps),
    "c_a2_b_minus_c_b": contrast("c_a2_b_minus_c_b", "LM1_C_B", "LM1_C_A2_B", n_bootstraps),
  }
  # Sensitivity: frozen-B (primary-surface) minus A''-surface refit B on identical rows.
  contrasts["frozen_c_b_minus_c_a2"] = contrast(
    "frozen_c_b_minus_c_a2", "LM1_C_A2", "p_b", n_bootstraps
  )
  contrasts["frozen_ecg_b_minus_a2_ecg"] = contrast(
    "frozen_ecg_b_minus_a2_ecg", "LM1_A2_ECG", "p_b_ecg", n_bootstraps
  )

  # Fixed-capacity rows for the context-adjusted primary pair on this surface.
  capacity_rows: list[dict[str, Any]] = []
  for frac in CAPACITY_GRID:
    cap_b = capacity_evaluation(y, col("LM1_C_B"), frac)
    cap_a2 = capacity_evaluation(y, col("LM1_C_A2"), frac)
    cap_c0 = capacity_evaluation(y, col("p_c"), frac)
    capacity_rows.append(
      {
        "capacity_fraction": frac,
        "b_n_flagged": cap_b.n_flagged,
        "b_events_captured": cap_b.n_events_captured,
        "b_ppv": cap_b.ppv,
        "a2_events_captured": cap_a2.n_events_captured,
        "a2_ppv": cap_a2.ppv,
        "c0_events_captured": cap_c0.n_events_captured,
        "c0_ppv": cap_c0.ppv,
      }
    )

  results: dict[str, Any] = {
    "status": "complete",
    "post_hoc": True,
    "representation": "A_double_cart_report_comparator",
    "config_version": config.version,
    "config_sha256": config_sha256(config_path),
    "vocabulary_sha256": vocabulary_hash(builder.vocabulary or ()),
    "representation_type": config.representation_type,
    "n_features_a2": len(a2),
    "n_features_report_block": len(report_cols),
    "missing_machine_report_by_split": missing_by_split,
    "n_surface_by_split": n_surface,
    "n_surface_events_by_split": n_surface_events,
    "n_test": n_test,
    "n_test_events": n_events,
    "test_population_sha256": test_population_sha256,
    "primary_paired_valid_population_identical": population_identical,
    "models": models,
    "contrasts": contrasts,
    "capacity": capacity_rows,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "cart_report_models.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  # Feature manifest: frozen vocabulary and feature order.
  (output_dir / "cart_report_feature_manifest.json").write_text(
    json.dumps(
      {
        "config_version": config.version,
        "config_sha256": config_sha256(config_path),
        "representation_type": config.representation_type,
        "vocabulary_size": len(builder.vocabulary or ()),
        "vocabulary_sha256": vocabulary_hash(builder.vocabulary or ()),
        "vocabulary": list(builder.vocabulary or ()),
        "a_prime_features": list(ap),
        "meta_features": list(ReportFeatureBuilder.META_FEATURES),
        "a2_feature_order": a2,
      },
      indent=2,
      sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
  )
  # Contrast table (committed under docs/manuscript/tables/).
  report_path.parent.mkdir(parents=True, exist_ok=True)
  tables_dir = report_path.parent.parent / "manuscript" / "tables"
  tables_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for name, v in contrasts.items():
    for metric in ("auroc", "auprc"):
      rows.append({
        "contrast": name,
        "metric": metric,
        "reference": v["reference"],
        "model": v["model"],
        "point_estimate": v[metric]["point_estimate"],
        "ci_lower": v[metric]["ci_lower"],
        "ci_upper": v[metric]["ci_upper"],
        "n_test": n_test,
        "n_events": n_events,
      })
  pl.DataFrame(rows).write_csv(tables_dir / "supp_cart_report_comparator.csv")
  _render_lm1_results_report(report_path, results)
  return results


def _render_lm1_results_report(report_path: Path, results: dict[str, Any]) -> None:
  lines = [
    "# LM1 cart-report comparator results (A'')",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {results['config_sha256']}",
    "analysis_status: post_hoc_extension",
    "inputs: [configs/cart_report_comparator_v1.yaml, artifacts/limitation_mitigation/lm1]",
    "decisions: [frozen vocabulary before locked-test evaluation]",
    "evidence: [cart_report_models.json, cart_report_feature_manifest.json]",
    "unresolved: []",
    "verification: [dev/val refits on identical A''-valid surface, frozen B sensitivity]",
    "next_action: update claims register C5 boundary (LM9 reconciliation)",
    "```",
    "",
    f"A'' = 21 cart global-measurement features + {results['n_features_report_block']} report features "
    f"({results['representation_type']}); {results['n_features_a2']} columns total.",
    "",
    "A'' is a **rich automated ECG-cart output comparator**, not a pure waveform-only",
    "representation. Machine statements may encode demographic-entry and acquisition workflow",
    "as well as waveform interpretation. The frozen vocabulary includes age/sex-entry and",
    "process phrases. ECG-only model keys mean no added C0 block, not absence of cart metadata.",
    "A phrase-exclusion sensitivity and cart/era audit are deferred; no feature is redesigned",
    "based on locked-test performance.",
    "",
    "## Population",
    "",
    f"- Locked-test rows on the A''-valid paired-valid surface: {results['n_test']:,} "
    f"({results['n_test_events']} deaths)",
    f"- Machine-report rows missing by split: {results['missing_machine_report_by_split']}",
    f"- Surface n by split: {results['n_surface_by_split']}",
    f"- Surface events by split: {results['n_surface_events_by_split']}",
    f"- Primary paired-valid population identical: {results['primary_paired_valid_population_identical']}",
    "",
    "## Required contrasts (AUROC, outcome-stratified paired bootstrap, 1,000 resamples, seed 42)",
    "",
    "| Contrast | Estimate | 95% CI |",
    "|---|---|---|",
  ]
  for name, v in results["contrasts"].items():
    a = v["auroc"]
    lines.append(
      f"| {name} | {a['point_estimate']:+.4f} | {a['ci_lower']:+.4f} to {a['ci_upper']:+.4f} |"
    )
  lines += [
    "",
    "## Capacity (context-adjusted C0+B vs C0+A'')",
    "",
    "| Capacity | B flagged | B captured | B PPV | A'' captured | A'' PPV | C0 captured |",
    "|---|---|---|---|---|---|---|",
  ]
  for row in results["capacity"]:
    lines.append(
      f"| {row['capacity_fraction']:.3f} | {row['b_n_flagged']} | {row['b_events_captured']} | "
      f"{row['b_ppv']:.3f} | {row['a2_events_captured']} | {row['a2_ppv']:.3f} | "
      f"{row['c0_events_captured']} |"
    )
  lines += [
    "",
    "All A'' rows are post hoc. The frozen primary B−A result is unchanged.",
    "",
  ]
  report_path.write_text("\n".join(lines), encoding="utf-8")
