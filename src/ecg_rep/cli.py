"""CLI for cohort construction and representation extraction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import polars as pl

from ecg_rep.analyze import run_scientific_analysis
from ecg_rep.context import availability_ledger_markdown
from ecg_rep.elixhauser import ELIXHAUSER_FEATURE_NAMES, build_c0_features
from ecg_rep.evaluate import (
  run_context_adjusted_evaluation,
  run_ecg_only_evaluation,
  run_random_dbeta_control,
)
from ecg_rep.data import (
  build_waveform_eligibility_manifest,
  compute_record_list_hash,
  load_machine_measurements,
  load_record_list,
  load_waveform_eligibility_manifest,
  read_ecg_waveform,
)
from ecg_rep.paths import resolve_data_paths, load_dataset_tables, load_diagnoses_table
from ecg_rep.representations import (
  CIIS_EXTRACTOR_VERSION,
  CIIS_FEATURE_SCHEMA_HASH,
  extract_ciis_feature_cohort,
  extract_dbeta_embedding_cohort,
  join_representation_validity,
  save_representation_artifacts,
  verify_representation_artifact_integrity,
)
from ecg_rep.scoring.dbeta import DbetaAdapter, DbetaConfig, load_dbeta_model
from ecg_rep.scoring.traditional import (
  CIIS_FEATURE_NAMES,
  extract_ciis_measurements_from_waveform,
  extract_independent_benchmark_measurements,
  run_ciis_measurement_validation_study,
  validate_ciis_feature_distributions,
)
from ecg_rep.split import (
  DEFAULT_SPLIT_SEED,
  SplitRatios,
  build_primary_cohort_and_split,
  compute_split_summary_statistics,
  generate_cohort_flow_markdown,
  save_split_assignments,
)


def resolve_torch_device(requested: str) -> str:
  """Resolve ``auto`` to the fastest available local torch device."""
  if requested != "auto":
    return requested
  try:
    import torch
  except ImportError:
    return "cpu"
  if torch.cuda.is_available():
    return "cuda"
  if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def build_parser() -> argparse.ArgumentParser:
  """Build the top-level argument parser."""
  parser = argparse.ArgumentParser(
    prog="ecg-rep",
    description="ECG representation study utilities (CIIS vs frozen D-BETA).",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  cohort = sub.add_parser(
    "cohort",
    help="Build the primary adult index-ECG cohort and patient-disjoint splits.",
  )
  cohort.add_argument("--mimic-root", type=str, default=None)
  cohort.add_argument("--ecg-root", type=str, default=None)
  cohort.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/cohort"),
    help="Directory for cohort parquet, split assignments, and reports.",
  )
  cohort.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
  cohort.add_argument("--dev-ratio", type=float, default=0.6)
  cohort.add_argument("--val-ratio", type=float, default=0.2)
  cohort.add_argument("--test-ratio", type=float, default=0.2)
  cohort.add_argument(
    "--preserve-split-from",
    type=Path,
    default=None,
    help="Path to existing split assignments CSV/parquet to lock patient partition assignments.",
  )
  cohort.add_argument(
    "--locked-reanalysis",
    action="store_true",
    help="Enforce audit-corrected locked-set reanalysis mode, requiring frozen split preservation.",
  )
  cohort.add_argument(
    "--allow-new-test",
    action="store_true",
    help="Explicitly permit new unassigned subjects in the test split during locked reanalysis.",
  )
  cohort.add_argument(
    "--new-split",
    action="store_true",
    help="Explicitly permit generating a new randomized split, bypassing locked partition protection.",
  )
  cohort.add_argument(
    "--waveform-eligibility-manifest",
    type=Path,
    default=None,
    help="Path to precomputed waveform eligibility manifest (parquet/csv) to filter eligible study_ids.",
  )
  cohort.add_argument(
    "--compute-waveform-manifest",
    action="store_true",
    help="Scan WFDB headers to generate waveform eligibility manifest before cohort construction.",
  )
  cohort.add_argument(
    "--allow-incomplete-manifest",
    action="store_true",
    help="Allow partial/smoke-test waveform eligibility manifest for testing.",
  )

  manifest = sub.add_parser(
    "manifest",
    help="Generate technical waveform eligibility manifest by scanning WFDB headers.",
  )
  manifest.add_argument("--mimic-root", type=str, default=None)
  manifest.add_argument("--ecg-root", type=str, default=None)
  manifest.add_argument(
    "--output-path",
    type=Path,
    default=None,
    help="Path to save generated waveform eligibility manifest (defaults to canonical manifest or _sample.parquet if --limit is passed).",
  )
  manifest.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke test.")
  manifest.add_argument("--progress-every", type=int, default=5000)

  val_ciis = sub.add_parser(
    "validate-ciis",
    help="Run independent benchmark morphology validation study on a reproducible sample.",
  )
  val_ciis.add_argument("--mimic-root", type=str, default=None)
  val_ciis.add_argument("--ecg-root", type=str, default=None)
  val_ciis.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/cohort/primary_cohort.parquet"),
    help="Frozen cohort parquet from `ecg-rep cohort`.",
  )
  val_ciis.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/reps"),
    help="Directory to save ciis_measurement_validation.json.",
  )
  val_ciis.add_argument(
    "--sample-size",
    type=int,
    default=100,
    help="Number of ECG records to sample for independent validation.",
  )
  val_ciis.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for validation record sampling.",
  )

  preflight = sub.add_parser(
    "preflight",
    help="Run CIIS morphology preflight validation on standard and enriched difficult cases.",
  )
  preflight.add_argument("--mimic-root", type=str, default=None)
  preflight.add_argument("--ecg-root", type=str, default=None)
  preflight.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/cohort/primary_cohort.parquet"),
    help="Optional path to cohort parquet.",
  )
  preflight.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/reps"),
    help="Directory to save preflight acceptance report.",
  )
  preflight.add_argument("--n-standard", type=int, default=200, help="Number of standard tracings.")
  preflight.add_argument("--n-difficult", type=int, default=100, help="Number of enriched difficult cases.")
  preflight.add_argument("--seed", type=int, default=42, help="Random seed.")
  preflight.add_argument(
    "--no-enforce",
    action="store_true",
    help="Do not raise exception if preflight acceptance criteria fail.",
  )

  reps = sub.add_parser(
    "representations",
    help="Extract CIIS features and frozen D-BETA embeddings for index ECGs.",
  )
  reps.add_argument("--mimic-root", type=str, default=None)
  reps.add_argument("--ecg-root", type=str, default=None)
  reps.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/cohort/primary_cohort.parquet"),
    help="Frozen cohort parquet from `ecg-rep cohort`.",
  )
  reps.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/reps"),
    help="Directory for representation arrays, metadata, and provenance.",
  )
  reps.add_argument("--limit", type=int, default=None, help="Optional record cap for smoke runs.")
  reps.add_argument(
    "--validate-ciis-sample",
    type=int,
    default=100,
    help="Sample size for independent morphology benchmark validation (0 disables).",
  )
  reps.add_argument("--batch-size", type=int, default=32)
  reps.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Torch device for D-BETA (auto|cpu|mps|cuda).",
  )
  reps.add_argument(
    "--skip-ciis",
    action="store_true",
    help="Skip CIIS extraction and reuse existing CIIS artifacts in --output-dir.",
  )
  reps.add_argument(
    "--skip-dbeta",
    action="store_true",
    help="Extract only CIIS features (useful when D-BETA weights are unavailable).",
  )
  reps.add_argument(
    "--dbeta-random-weights",
    action="store_true",
    help="Extract representations using an architecture-matched randomly initialized D-BETA encoder.",
  )
  reps.add_argument(
    "--dbeta-random-seed",
    type=int,
    default=42,
    help="Random seed for random-weights D-BETA control.",
  )
  reps.add_argument(
    "--progress-every",
    type=int,
    default=1000,
    help="Print CIIS/D-BETA progress every N records (0 disables).",
  )
  reps.add_argument(
    "--checkpoint-interval",
    type=int,
    default=5000,
    help="Write CIIS/D-BETA checkpoints every N records (0 disables).",
  )
  reps.add_argument(
    "--override-preflight",
    action="store_true",
    help="Allow representation extraction without a matching PASSED preflight acceptance report (development escape hatch).",
  )
  reps.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="Number of worker processes for CIIS feature extraction (default 1).",
  )

  context = sub.add_parser(
    "context",
    help="Build leakage-safe C0 features (age, sex, prior-admission Elixhauser).",
  )
  context.add_argument("--mimic-root", type=str, default=None)
  context.add_argument("--ecg-root", type=str, default=None)
  context.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/cohort/primary_cohort.parquet"),
  )
  context.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/context"),
  )

  evaluate = sub.add_parser(
    "evaluate",
    help="Fit locked ridge-logistic probes and write paired metrics.",
  )
  evaluate.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/cohort/primary_cohort.parquet"),
  )
  evaluate.add_argument(
    "--c0-path",
    type=Path,
    default=Path("artifacts/context/c0_features.parquet"),
  )
  evaluate.add_argument(
    "--reps-dir",
    type=Path,
    default=Path("artifacts/reps"),
  )
  evaluate.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/probe"),
  )
  evaluate.add_argument("--n-bootstraps", type=int, default=1000)
  evaluate.add_argument(
    "--ecg-only",
    action="store_true",
    help="Skip context-adjusted models even if C0 exists.",
  )
  evaluate.add_argument(
    "--run-random-control",
    action="store_true",
    help="Run negative control comparing pretrained D-BETA vs architecture-matched random D-BETA.",
  )
  evaluate.add_argument(
    "--reps-random-dir",
    type=Path,
    default=None,
    help="Directory containing random D-BETA artifacts (defaults to --reps-dir).",
  )
  evaluate.add_argument(
    "--random-seed",
    type=int,
    default=42,
    help="Seed used for random D-BETA artifacts.",
  )

  analyze = sub.add_parser(
    "analyze",
    help="Run RQ scientific estimands from locked-test prediction tables.",
  )
  analyze.add_argument(
    "--probe-dir",
    type=Path,
    default=Path("artifacts/probe"),
  )
  analyze.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/cohort/primary_cohort.parquet"),
  )
  analyze.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/analysis"),
  )
  analyze.add_argument(
    "--reps-dir",
    type=Path,
    default=Path("artifacts/reps"),
    help="Representation provenance/arrays for RQ5 extraction-cost accounting.",
  )
  analyze.add_argument(
    "--n-bootstraps",
    type=int,
    default=1000,
    help="Number of bootstrap resamples for uncertainty estimation.",
  )
  analyze.add_argument(
    "--calibration-scores",
    type=Path,
    default=Path("artifacts/probe/dev_val_scores_context.parquet"),
    help="Frozen-probe dev/val score table for rank-calibrated cascade "
    "policies (create with `ecg-rep dump-probe-scores`). Skipped if absent.",
  )

  dump_scores = sub.add_parser(
    "dump-probe-scores",
    help="Score development+validation rows with the frozen context-adjusted "
    "probes (read-only) to build the cascade rank-calibration table.",
  )
  dump_scores.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/cohort/primary_cohort.parquet"),
  )
  dump_scores.add_argument(
    "--c0-path",
    type=Path,
    default=Path("artifacts/context/c0_features.parquet"),
  )
  dump_scores.add_argument(
    "--meta-a-path",
    type=Path,
    default=Path("artifacts/reps/ciis_features_meta.parquet"),
  )
  dump_scores.add_argument(
    "--features-a-path",
    type=Path,
    default=Path("artifacts/reps/ciis_features.npy"),
  )
  dump_scores.add_argument(
    "--meta-b-path",
    type=Path,
    default=Path("artifacts/reps/dbeta_embedding_meta.parquet"),
  )
  dump_scores.add_argument(
    "--embeddings-b-path",
    type=Path,
    default=Path("artifacts/reps/dbeta_embeddings.npy"),
  )
  dump_scores.add_argument(
    "--models-dir",
    type=Path,
    default=Path("artifacts/probe/models"),
    help="Frozen probe directory (M_A/M_B joblib artifacts loaded read-only).",
  )
  dump_scores.add_argument(
    "--output-path",
    type=Path,
    default=Path("artifacts/probe/dev_val_scores_context.parquet"),
  )
  dump_scores.add_argument(
    "--verify-provenance",
    action="store_true",
    help="Force representation-artifact integrity verification.",
  )

  cascade_fit = sub.add_parser(
    "probability-cascade-fit-routing",
    help="Fit frozen Stage A routing cutoffs on one authorized dev/val split.",
  )
  cascade_fit.add_argument("--config", type=Path, required=True)
  cascade_fit.add_argument("--scores-path", type=Path, required=True)
  cascade_fit.add_argument("--split", choices=("dev", "val"), required=True)
  cascade_fit.add_argument("--output-dir", type=Path, default=Path("artifacts/probability_cascade/v2/manual-fit"))

  cascade_replay = sub.add_parser(
    "probability-cascade-replay",
    help="Replay a frozen patient-wise policy; private row outputs stay under artifacts/probability_cascade/v2/.",
  )
  cascade_replay.add_argument("--config", type=Path, required=True)
  cascade_replay.add_argument("--policy", type=Path, required=True)
  cascade_replay.add_argument("--scores-path", type=Path, required=True)
  cascade_replay.add_argument("--split", choices=("dev", "val", "test"), required=True)
  cascade_replay.add_argument("--precomputed-b-reference", action="store_true")
  cascade_replay.add_argument("--allow-locked-test", action="store_true", help=argparse.SUPPRESS)
  cascade_replay.add_argument("--output-dir", type=Path, default=Path("artifacts/probability_cascade/v2/manual-replay"))

  cascade_run = sub.add_parser(
    "probability-cascade-run",
    help="Generate frozen ECG-only scores and run v2 routing, frontier, calibration, and exploratory DCA.",
  )
  cascade_run.add_argument(
    "--config",
    type=Path,
    default=Path("configs/probability_cascade_v2.yaml"),
    help="Versioned exploratory run configuration.",
  )

  setting = sub.add_parser(
    "clinical-setting",
    help="Assign a timestamped care setting to each preserved index ECG.",
  )
  setting.add_argument("--mimic-root", type=str, default=None)
  setting.add_argument("--ecg-root", type=str, default=None)
  setting.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  setting.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  setting.add_argument(
    "--predictions-context-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_context.parquet"),
  )
  setting.add_argument(
    "--predictions-ecg-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_ecg_only.parquet"),
  )
  setting.add_argument(
    "--pair-validity-path",
    type=Path,
    default=Path("artifacts/reps/representation_pair_validity.parquet"),
  )
  setting.add_argument("--output-dir", type=Path, default=Path("artifacts/clinical/setting"))
  setting.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/setting_results.md"),
  )
  setting.add_argument("--n-bootstraps", type=int, default=1000)
  setting.add_argument("--seed", type=int, default=42)

  phenotypes = sub.add_parser(
    "clinical-phenotypes",
    help="Draft overt ECG phenotypes on the development split only.",
  )
  phenotypes.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  phenotypes.add_argument("--ecg-root", type=str, default=None)
  phenotypes.add_argument(
    "--dictionary",
    type=Path,
    default=Path("configs/ecg_phenotype_dictionary.yaml"),
  )
  phenotypes.add_argument("--output-dir", type=Path, default=Path("artifacts/clinical/phenotypes"))
  phenotypes.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/phenotype_results.md"),
  )
  phenotypes.add_argument(
    "--split",
    type=str,
    default="dev",
    help="Only dev is allowed until the phenotype dictionary is approved.",
  )

  acute = sub.add_parser(
    "clinical-acute-context",
    help="Inventory leak-safe pre-ECG labs and vitals. Does not refit models.",
  )
  acute.add_argument("--mimic-root", type=str, default=None)
  acute.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  acute.add_argument("--setting-path", type=Path, default=Path("artifacts/clinical/setting/index_ecg_setting.parquet"))
  acute.add_argument("--output-dir", type=Path, default=Path("artifacts/clinical/context"))
  acute.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/acute_context_results.md"),
  )

  ami = sub.add_parser(
    "ami-feasibility",
    help="Development-only index-encounter AMI evidence audit. Does not fit models.",
  )
  ami.add_argument("--mimic-root", type=str, default=None)
  ami.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  ami.add_argument(
    "--setting-path",
    type=Path,
    default=Path("artifacts/clinical/setting/index_ecg_setting.parquet"),
  )
  ami.add_argument("--output-dir", type=Path, default=Path("artifacts/ami/feasibility"))
  ami.add_argument("--report-path", type=Path, default=Path("docs/clinical_extensions/ami_feasibility.md"))

  sub.add_parser(
    "ami-evaluate",
    help="Closed (decision D14): adjudicated AMI is not pursued. Use ami-administrative.",
  )

  administrative = sub.add_parser(
    "ami-administrative",
    help="Fit probes for principal-discharge AMI coding. Not adjudicated AMI.",
  )
  administrative.add_argument("--mimic-root", type=str, default=None)
  administrative.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  administrative.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  administrative.add_argument(
    "--setting-path",
    type=Path,
    default=Path("artifacts/clinical/setting/index_ecg_setting.parquet"),
  )
  administrative.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  administrative.add_argument(
    "--pair-validity-path",
    type=Path,
    default=Path("artifacts/reps/representation_pair_validity.parquet"),
  )
  administrative.add_argument(
    "--protocol",
    type=Path,
    default=Path("configs/ami_administrative_protocol_v1.yaml"),
  )
  administrative.add_argument("--output-dir", type=Path, default=Path("artifacts/ami/administrative"))
  administrative.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/ami_administrative_results.md"),
  )
  administrative.add_argument("--n-bootstraps", type=int, default=200)
  administrative.add_argument("--seed", type=int, default=42)

  sensitivity = sub.add_parser(
    "clinical-mortality-sensitivity",
    help="Join the frozen lexicon to frozen locked-test mortality scores.",
  )
  sensitivity.add_argument("--ecg-root", type=str, default=None)
  sensitivity.add_argument(
    "--predictions-context-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_context.parquet"),
  )
  sensitivity.add_argument(
    "--predictions-ecg-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_ecg_only.parquet"),
  )
  sensitivity.add_argument(
    "--dictionary",
    type=Path,
    default=Path("configs/ecg_phenotype_dictionary.yaml"),
  )
  sensitivity.add_argument("--output-dir", type=Path, default=Path("artifacts/clinical/sensitivity"))
  sensitivity.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/mortality_sensitivity.md"),
  )
  sensitivity.add_argument("--n-bootstraps", type=int, default=1000)
  sensitivity.add_argument("--seed", type=int, default=42)

  next_feasibility = sub.add_parser(
    "clinical-next-step-feasibility",
    help="A6 development-only feasibility of post-ECG next-step outcomes. Joins no scores or mortality.",
  )
  next_feasibility.add_argument("--mimic-root", type=str, default=None)
  next_feasibility.add_argument("--ecg-root", type=str, default=None)
  next_feasibility.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  next_feasibility.add_argument(
    "--setting-path",
    type=Path,
    default=Path("artifacts/clinical/setting/index_ecg_setting.parquet"),
  )
  next_feasibility.add_argument(
    "--pair-validity-path",
    type=Path,
    default=Path("artifacts/reps/representation_pair_validity.parquet"),
  )
  next_feasibility.add_argument("--config", type=Path, default=Path("configs/next_step_outcomes_v1.yaml"))
  next_feasibility.add_argument(
    "--cache-dir",
    type=Path,
    default=Path("artifacts/clinical_extensions/next_step/cache"),
  )
  next_feasibility.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/clinical_extensions/next_step"),
  )
  next_feasibility.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/next_step_feasibility.md"),
  )
  next_feasibility.add_argument(
    "--split",
    type=str,
    default="dev",
    help="Only dev is allowed; the outcome set cannot be tuned on validation or test.",
  )
  next_feasibility.add_argument("--rebuild-cache", action="store_true")

  next_run = sub.add_parser(
    "clinical-next-step",
    help="A6 locked-test run: post-ECG next-step evaluation by A/B risk group (frozen config).",
  )
  next_run.add_argument("--mimic-root", type=str, default=None)
  next_run.add_argument("--ecg-root", type=str, default=None)
  next_run.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  next_run.add_argument(
    "--setting-path",
    type=Path,
    default=Path("artifacts/clinical/setting/index_ecg_setting.parquet"),
  )
  next_run.add_argument(
    "--pair-validity-path",
    type=Path,
    default=Path("artifacts/reps/representation_pair_validity.parquet"),
  )
  next_run.add_argument(
    "--predictions-context-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_context.parquet"),
  )
  next_run.add_argument(
    "--predictions-ecg-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_ecg_only.parquet"),
  )
  next_run.add_argument("--config", type=Path, default=Path("configs/next_step_outcomes_v1.yaml"))
  next_run.add_argument(
    "--cache-dir",
    type=Path,
    default=Path("artifacts/clinical_extensions/next_step/cache"),
  )
  next_run.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/clinical_extensions/next_step"),
  )
  next_run.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/next_step_results.md"),
  )
  next_run.add_argument("--n-bootstraps", type=int, default=200)
  next_run.add_argument("--seed", type=int, default=42)

  doc_audit = sub.add_parser(
    "clinical-documented-policy-audit",
    help="A7 development-only tier audit and blinded review sample. Joins no outcomes or scores.",
  )
  doc_audit.add_argument("--ecg-root", type=str, default=None)
  doc_audit.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  doc_audit.add_argument(
    "--dictionary",
    type=Path,
    default=Path("configs/ecg_phenotype_dictionary.yaml"),
  )
  doc_audit.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/clinical_extensions/documented_policy"),
  )

  doc_policy = sub.add_parser(
    "clinical-documented-policy",
    help="A7 locked-test run: documented-abnormality tiers versus A and B at equal capacity.",
  )
  doc_policy.add_argument("--ecg-root", type=str, default=None)
  doc_policy.add_argument(
    "--predictions-context-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_context.parquet"),
  )
  doc_policy.add_argument(
    "--predictions-ecg-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_ecg_only.parquet"),
  )
  doc_policy.add_argument(
    "--dictionary",
    type=Path,
    default=Path("configs/ecg_phenotype_dictionary.yaml"),
  )
  doc_policy.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/clinical_extensions/documented_policy"),
  )
  doc_policy.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/documented_abnormality_policy.md"),
  )
  doc_policy.add_argument("--n-bootstraps", type=int, default=200)
  doc_policy.add_argument("--seed", type=int, default=42)

  cacute = sub.add_parser(
    "clinical-cacute",
    help="Refit C0 and Cacute probes on one non-ambiguous paired-valid subset.",
  )
  cacute.add_argument("--mimic-root", type=str, default=None)
  cacute.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  cacute.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  cacute.add_argument("--setting-path", type=Path, default=Path("artifacts/clinical/setting/index_ecg_setting.parquet"))
  cacute.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  cacute.add_argument("--output-dir", type=Path, default=Path("artifacts/clinical/cacute"))
  cacute.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/cacute_results.md"),
  )
  cacute.add_argument("--n-bootstraps", type=int, default=200)
  cacute.add_argument("--seed", type=int, default=42)

  subgroups = sub.add_parser(
    "clinical-subgroups",
    help="Describe frozen A/B differences by sex, age, and race with setting mix.",
  )
  subgroups.add_argument("--mimic-root", type=str, default=None)
  subgroups.add_argument(
    "--predictions-context-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_context.parquet"),
  )
  subgroups.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  subgroups.add_argument("--setting-path", type=Path, default=Path("artifacts/clinical/setting/index_ecg_setting.parquet"))
  subgroups.add_argument("--output-dir", type=Path, default=Path("artifacts/clinical/subgroups"))
  subgroups.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/subgroup_results.md"),
  )
  subgroups.add_argument("--n-bootstraps", type=int, default=200)
  subgroups.add_argument("--seed", type=int, default=42)

  dca = sub.add_parser(
    "clinical-dca",
    help="Whole-cohort decision curve analysis of frozen locked-test predictions (secondary layer).",
  )
  dca.add_argument(
    "--predictions-context-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_context.parquet"),
  )
  dca.add_argument("--output-dir", type=Path, default=Path("artifacts/clinical/dca"))
  dca.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/clinical_extensions/dca_results.md"),
  )

  rand_ctl = sub.add_parser(
    "random-control",
    help="Random-encoder negative control only (plan §5.4); never rewrites the canonical probe outputs.",
  )
  rand_ctl.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  rand_ctl.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  rand_ctl.add_argument("--random-seeds", type=int, nargs="+", default=[42, 7])
  rand_ctl.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  rand_ctl.add_argument(
    "--reference-predictions-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_ecg_only.parquet"),
  )
  rand_ctl.add_argument(
    "--reference-context-predictions-path",
    type=Path,
    default=Path("artifacts/probe/test_predictions_context.parquet"),
  )
  rand_ctl.add_argument("--output-dir", type=Path, default=Path("artifacts/probe/random_control"))
  rand_ctl.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))

  pv_audit = sub.add_parser(
    "paired-valid-audit",
    help="Characterize locked-test patients excluded from the paired-valid analysis (round-1 audit item 5).",
  )
  pv_audit.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  pv_audit.add_argument(
    "--pair-validity-path", type=Path, default=Path("artifacts/reps/representation_pair_validity.parquet")
  )
  pv_audit.add_argument(
    "--setting-path", type=Path, default=Path("artifacts/clinical/setting/index_ecg_setting.parquet")
  )
  pv_audit.add_argument("--output-dir", type=Path, default=Path("artifacts/manuscript"))
  pv_audit.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))
  rand_ctl.add_argument("--n-bootstraps", type=int, default=1000)

  lm2a = sub.add_parser(
    "pretraining-label-efficiency",
    help=(
      "LM2A: refit every representation at restricted dev/val label budgets and evaluate on the "
      "unchanged paired-valid locked test (post hoc extension of a prespecified control)."
    ),
  )
  lm2a.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  lm2a.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  lm2a.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  lm2a.add_argument("--config", type=Path, default=Path("configs/label_efficiency_v1.yaml"))
  lm2a.add_argument("--output-dir", type=Path, default=Path("artifacts/probe/label_efficiency"))
  lm2a.add_argument("--report-path", type=Path, default=Path("docs/limitation_mitigation/lm2a_label_efficiency.md"))

  lm2b = sub.add_parser(
    "pretraining-ami-control",
    help=(
      "LM2B: per-seed random-vs-pretrained D-BETA probes on the administrative-AMI endpoint "
      "(post hoc extension of a prespecified control)."
    ),
  )
  lm2b.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  lm2b.add_argument(
    "--setting-path", type=Path, default=Path("artifacts/clinical/setting/index_ecg_setting.parquet")
  )
  lm2b.add_argument("--mimic-root", type=str, default=None)
  lm2b.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  lm2b.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  lm2b.add_argument(
    "--pair-validity-path", type=Path, default=Path("artifacts/reps/representation_pair_validity.parquet")
  )
  lm2b.add_argument("--random-seeds", type=int, nargs="+", default=[42, 7])
  lm2b.add_argument("--output-dir", type=Path, default=Path("artifacts/ami/pretraining_control"))
  lm2b.add_argument("--report-path", type=Path, default=Path("docs/limitation_mitigation/lm2b_ami_control.md"))
  cmax_feas = sub.add_parser(
    "cmax-feasibility",
    help=(
      "LM3a: development-only pre-ECG lab-panel and ICU-vital availability audit. "
      "No model fits; no locked-test outcomes."
    ),
  )
  cmax_feas.add_argument("--mimic-root", type=str, default=None)
  cmax_feas.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  cmax_feas.add_argument(
    "--setting-path", type=Path, default=Path("artifacts/clinical/setting/index_ecg_setting.parquet")
  )
  cmax_feas.add_argument("--config", type=Path, default=Path("configs/cmax_context_v1.yaml"))
  cmax_feas.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm3"))
  cmax_feas.add_argument(
    "--report-path", type=Path, default=Path("docs/limitation_mitigation/lm3_cmax_feasibility.md")
  )

  cmax_eval = sub.add_parser(
    "cmax-evaluate",
    help=(
      "LM3b: frozen same-subset refits of C0/Cacute/Cmax context models with A, B, A+B, and "
      "the A'' strong comparator; capacity, temporal-buffer, ICU, and calibration outputs."
    ),
  )
  cmax_eval.add_argument("--mimic-root", type=str, default=None)
  cmax_eval.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  cmax_eval.add_argument(
    "--setting-path", type=Path, default=Path("artifacts/clinical/setting/index_ecg_setting.parquet")
  )
  cmax_eval.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  cmax_eval.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  cmax_eval.add_argument(
    "--measurements-path",
    type=Path,
    default=Path("~/data/mimic-iv-ecg/1.0/machine_measurements.csv").expanduser(),
  )
  cmax_eval.add_argument("--config", type=Path, default=Path("configs/cmax_context_v1.yaml"))
  cmax_eval.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm3"))
  cmax_eval.add_argument(
    "--report-path", type=Path, default=Path("docs/limitation_mitigation/lm3_cmax_results.md")
  )
  cmax_eval.add_argument("--n-bootstraps", type=int, default=200)

  refit_unc = sub.add_parser(
    "refit-uncertainty",
    help=(
      "LM5: three-surface paired patient bootstrap with full probe refit and "
      "validation retune inside every replicate (headline models A, B, C0+A, C0+B); "
      "master seed 52026; optional frozen training-only/test-only decomposition."
    ),
  )
  refit_unc.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  refit_unc.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  refit_unc.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  refit_unc.add_argument("--n-replicates", type=int, default=200)
  refit_unc.add_argument("--master-seed", type=int, default=52026)
  refit_unc.add_argument("--n-jobs", type=int, default=4)
  refit_unc.add_argument("--skip-decomposition", action="store_true")
  refit_unc.add_argument(
    "--reuse-main-bootstrap",
    action="store_true",
    help="Keep an already written main bootstrap table and run decomposition only.",
  )
  refit_unc.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm5"))
  refit_unc.add_argument(
    "--report-path", type=Path, default=Path("docs/limitation_mitigation/lm5_refit_uncertainty.md")
  )
  refit_unc.add_argument(
    "--supp-table", type=Path, default=Path("docs/manuscript/tables/supp_refit_uncertainty.csv")
  )
  refit_unc.add_argument(
    "--primary-ecg-results", type=Path, default=Path("artifacts/probe/probe_results.json")
  )
  refit_unc.add_argument(
    "--primary-context-results", type=Path, default=Path("artifacts/probe/probe_results_context.json")
  )

  subgroup_mult = sub.add_parser(
    "subgroup-multiplicity",
    help=(
      "LM7: simultaneous 95%% bands and max-T pairwise intervals for the prespecified "
      "sex, age, and race/ethnicity B-A contrasts. Does not add subgroup levels."
    ),
  )
  subgroup_mult.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  subgroup_mult.add_argument(
    "--predictions-context", type=Path, default=Path("artifacts/probe/test_predictions_context.parquet")
  )
  subgroup_mult.add_argument("--mimic-root", type=str, default=None)
  subgroup_mult.add_argument("--config", type=Path, default=Path("configs/lm7_subgroup_multiplicity_v1.yaml"))
  subgroup_mult.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm7"))
  subgroup_mult.add_argument(
    "--report-path", type=Path, default=Path("docs/limitation_mitigation/lm7_subgroup_multiplicity.md")
  )

  ascertain = sub.add_parser(
    "ascertainment-audit",
    help=(
      "LM6A: compare outcome-ascertainable and unascertainable index ECGs, and recover "
      "known 30-day survivors with a timestamp strictly after day 30. Does not replace the primary cohort."
    ),
  )
  ascertain.add_argument("--mimic-root", type=str, default=None)
  ascertain.add_argument("--ecg-root", type=str, default=None)
  ascertain.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  ascertain.add_argument(
    "--waveform-manifest",
    type=Path,
    default=Path("artifacts/cohort/waveform_eligibility_manifest.parquet"),
  )
  ascertain.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  ascertain.add_argument("--config", type=Path, default=Path("configs/lm6_ascertainment_history_v1.yaml"))
  ascertain.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm6"))
  ascertain.add_argument(
    "--report-path", type=Path, default=Path("docs/limitation_mitigation/lm6a_ascertainment.md")
  )
  ascertain.add_argument("--horizon-days", type=float, default=None)

  history = sub.add_parser(
    "history-sensitivity",
    help=(
      "LM6B: B-A within pre-index history strata, and (C0+H+B)-(C0+H+A) on the paired-valid surface. "
      "History features are strictly pre-index."
    ),
  )
  history.add_argument("--mimic-root", type=str, default=None)
  history.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  history.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  history.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  history.add_argument(
    "--predictions-context", type=Path, default=Path("artifacts/probe/test_predictions_context.parquet")
  )
  history.add_argument("--config", type=Path, default=Path("configs/lm6_ascertainment_history_v1.yaml"))
  history.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm6"))
  history.add_argument(
    "--report-path", type=Path, default=Path("docs/limitation_mitigation/lm6b_history.md")
  )
  history.add_argument("--n-bootstraps", type=int, default=None)

  scaling = sub.add_parser(
    "compute-scaling",
    help=(
      "LM8: equal-resource throughput scaling. CIIS uses process workers; "
      "D-BETA CPU uses explicit intra-op threads. No single worker-vs-thread ratio."
    ),
  )
  scaling.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  scaling.add_argument("--ecg-root", type=Path, default=Path("~/data/mimic-iv-ecg/1.0").expanduser())
  scaling.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm8"))
  scaling.add_argument(
    "--report-path", type=Path, default=Path("docs/limitation_mitigation/lm8_compute_scaling.md")
  )
  scaling.add_argument(
    "--smoke-only",
    action="store_true",
    help="Freeze batch sizes after the smoke run and do not start timed evaluation.",
  )

  boot_sens = sub.add_parser(
    "bootstrap-estimand-sensitivity",
    help=(
      "Outcome-stratified vs ordinary paired patient-bootstrap intervals for the primary "
      "headline contrasts, from frozen locked-test predictions (round-1 audit item 8)."
    ),
  )
  boot_sens.add_argument(
    "--predictions-ecg-only-path", type=Path, default=Path("artifacts/probe/test_predictions_ecg_only.parquet")
  )
  boot_sens.add_argument(
    "--predictions-context-path", type=Path, default=Path("artifacts/probe/test_predictions_context.parquet")
  )
  boot_sens.add_argument(
    "--analysis-results-path", type=Path, default=Path("artifacts/analysis/scientific_analysis_results.json")
  )
  boot_sens.add_argument("--output-dir", type=Path, default=Path("artifacts/manuscript"))
  boot_sens.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))
  boot_sens.add_argument("--n-bootstraps", type=int, default=1000)
  boot_sens.add_argument("--seed", type=int, default=42)

  reporting = sub.add_parser(
    "manuscript-reporting",
    help="Table 1, calibration intervals/curves, canonical CIIS reference, ascertainment sensitivity (frozen predictions).",
  )
  reporting.add_argument("--mimic-root", default=None)
  reporting.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  reporting.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  reporting.add_argument(
    "--setting-path", type=Path, default=Path("artifacts/clinical/setting/index_ecg_setting.parquet")
  )
  reporting.add_argument(
    "--pair-validity-path", type=Path, default=Path("artifacts/reps/representation_pair_validity.parquet")
  )
  reporting.add_argument(
    "--predictions-context-path", type=Path, default=Path("artifacts/probe/test_predictions_context.parquet")
  )
  reporting.add_argument(
    "--predictions-ecg-only-path", type=Path, default=Path("artifacts/probe/test_predictions_ecg_only.parquet")
  )
  reporting.add_argument("--output-dir", type=Path, default=Path("artifacts/manuscript"))
  reporting.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))
  reporting.add_argument("--figures-dir", type=Path, default=Path("docs/manuscript/figures"))
  reporting.add_argument("--n-bootstraps", type=int, default=1000)
  reporting.add_argument("--seed", type=int, default=42)

  ptbxl = sub.add_parser(
    "ptbxl-measurement-validity",
    help="Compare the frozen CIIS extractor with PTB-XL+ GE 12SL per-lead measurements (no MIMIC data).",
  )
  ptbxl.add_argument("--ptbxl-root", type=Path, default=Path("~/data/ptbxl").expanduser())
  ptbxl.add_argument(
    "--features-12sl-path", type=Path, default=Path("~/data/ptbxl_plus/12sl_features.csv").expanduser()
  )
  ptbxl.add_argument(
    "--sample-ids-path", type=Path, default=Path("~/data/ptbxl/sample_ids_seed2026.txt").expanduser()
  )
  ptbxl.add_argument("--output-dir", type=Path, default=Path("artifacts/external/ptbxl"))
  ptbxl.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))
  ptbxl.add_argument("--n-bootstraps", type=int, default=1000)

  machine = sub.add_parser(
    "machine-baseline",
    help="Post hoc cart global-measurement comparator A' with the frozen probe protocol.",
  )
  machine.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  machine.add_argument(
    "--machine-measurements-path",
    type=Path,
    default=Path("~/data/mimic-iv-ecg/1.0/machine_measurements.csv").expanduser(),
  )
  machine.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  machine.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  machine.add_argument(
    "--reference-ecg-only-path", type=Path, default=Path("artifacts/probe/test_predictions_ecg_only.parquet")
  )
  machine.add_argument(
    "--reference-context-path", type=Path, default=Path("artifacts/probe/test_predictions_context.parquet")
  )
  machine.add_argument("--output-dir", type=Path, default=Path("artifacts/machine_baseline"))
  machine.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))
  machine.add_argument("--n-bootstraps", type=int, default=1000)

  cart_report = sub.add_parser(
    "cart-report-feasibility",
    help="LM1a development-only inventory of machine-report text for the A'' comparator.",
  )
  cart_report.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  cart_report.add_argument(
    "--machine-measurements-path",
    type=Path,
    default=Path("~/data/mimic-iv-ecg/1.0/machine_measurements.csv").expanduser(),
  )
  cart_report.add_argument("--config", type=Path, default=Path("configs/cart_report_comparator_v1.yaml"))
  cart_report.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm1"))
  cart_report.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/limitation_mitigation/lm1_cart_report_feasibility.md"),
  )

  cart_eval = sub.add_parser(
    "cart-report-baseline",
    help="LM1b frozen A'' locked-test evaluation under the frozen config/vocabulary.",
  )
  cart_eval.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  cart_eval.add_argument(
    "--machine-measurements-path",
    type=Path,
    default=Path("~/data/mimic-iv-ecg/1.0/machine_measurements.csv").expanduser(),
  )
  cart_eval.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  cart_eval.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  cart_eval.add_argument("--config", type=Path, default=Path("configs/cart_report_comparator_v1.yaml"))
  cart_eval.add_argument(
    "--feasibility-path",
    type=Path,
    default=Path("artifacts/limitation_mitigation/lm1/cart_report_feasibility.json"),
  )
  cart_eval.add_argument(
    "--reference-ecg-only-path", type=Path, default=Path("artifacts/probe/test_predictions_ecg_only.parquet")
  )
  cart_eval.add_argument(
    "--reference-context-path", type=Path, default=Path("artifacts/probe/test_predictions_context.parquet")
  )
  cart_eval.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm1"))
  cart_eval.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/limitation_mitigation/lm1_cart_report_results.md"),
  )
  cart_eval.add_argument("--n-bootstraps", type=int, default=1000)

  addon = sub.add_parser(
    "cart-addon-contrast",
    help="LM1 addendum (round 09): paired add-on contrast (C0+A''+B) − (C0+A'') with ledger determinism gates.",
  )
  addon.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  addon.add_argument(
    "--machine-measurements-path",
    type=Path,
    default=Path("~/data/mimic-iv-ecg/1.0/machine_measurements.csv").expanduser(),
  )
  addon.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  addon.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  addon.add_argument("--config", type=Path, default=Path("configs/cart_report_comparator_v1.yaml"))
  addon.add_argument(
    "--feasibility-path",
    type=Path,
    default=Path("artifacts/limitation_mitigation/lm1/cart_report_feasibility.json"),
  )
  addon.add_argument(
    "--ledger-path",
    type=Path,
    default=Path("artifacts/limitation_mitigation/lm1/cart_report_models.json"),
  )
  addon.add_argument(
    "--reference-aggregates-path",
    type=Path,
    default=Path("docs/limitation_mitigation/lm1_reference_aggregates.json"),
    help="Committed nonidentifying aggregate manifest used for the population-identity gate.",
  )
  addon.add_argument("--output-dir", type=Path, default=Path("artifacts/limitation_mitigation/lm1"))
  addon.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/limitation_mitigation/lm1_cart_addon_contrast.md"),
  )
  addon.add_argument(
    "--n-bootstraps",
    type=int,
    default=1000,
    help="Resamples for the NEW add-on contrast only; ledger verification always uses the historical 1,000.",
  )

  recal = sub.add_parser(
    "code15-recalibrate",
    help="LM4: fit R1/R2 recalibration of transported frozen A/B scores on CODE validation, evaluate once on the CODE locked test.",
  )
  recal.add_argument(
    "--features-dir", type=Path, default=Path("artifacts/external/code15pct/features")
  )
  recal.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_index_cohort.parquet"),
  )
  recal.add_argument("--model-dir", type=Path, default=Path("artifacts/probe"))
  recal.add_argument(
    "--split-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_split_assignments.csv"),
  )
  recal.add_argument("--config", type=Path, default=Path("configs/code15_recalibration_v1.yaml"))
  recal.add_argument("--output-dir", type=Path, default=Path("artifacts/external/code15"))
  recal.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/external_validation/08_code15_recalibration.md"),
  )
  recal.add_argument("--n-bootstraps", type=int, default=1000)
  recal.add_argument("--skip-val-refit-sensitivity", action="store_true")

  compute = sub.add_parser(
    "compute-benchmark",
    help="Protocol-matched CIIS vs D-BETA timing on one in-memory ECG sample (outcome-blind).",
  )
  compute.add_argument("--cohort-path", type=Path, default=Path("artifacts/cohort/primary_cohort.parquet"))
  compute.add_argument("--ecg-root", type=Path, default=Path("~/data/mimic-iv-ecg/1.0").expanduser())
  compute.add_argument("--output-dir", type=Path, default=Path("artifacts/compute"))
  compute.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))
  compute.add_argument("--n-ecgs", type=int, default=1024)
  compute.add_argument("--repeats", type=int, default=5)
  compute.add_argument("--devices", nargs="+", default=["cpu", "mps"])
  compute.add_argument("--c0-path", type=Path, default=Path("artifacts/context/c0_features.parquet"))
  compute.add_argument("--reps-dir", type=Path, default=Path("artifacts/reps"))
  compute.add_argument("--skip-extraction", action="store_true", help="Run only the downstream probe benchmark.")
  compute.add_argument("--skip-downstream", action="store_true", help="Run only the extraction benchmark.")

  figures = sub.add_parser(
    "manuscript-figures",
    help="Render manuscript Figures 1-6 and Tables 2-5 from frozen artifacts (no fitting).",
  )
  figures.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
  figures.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))
  figures.add_argument("--figures-dir", type=Path, default=Path("docs/manuscript/figures"))
  figures.add_argument(
    "--clinical-tables-dir", type=Path, default=Path("docs/clinical_extensions/tables")
  )
  figures.add_argument("--n-bootstraps", type=int, default=1000)

  submission_figures = sub.add_parser(
    "submission-figures",
    help="Render the five submission-facing v2.16 figures from committed outputs (no fitting; audit 20261021 round05).",
  )
  submission_figures.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
  submission_figures.add_argument("--tables-dir", type=Path, default=Path("docs/manuscript/tables"))
  submission_figures.add_argument("--figures-dir", type=Path, default=Path("docs/manuscript/figures/submission"))
  submission_figures.add_argument(
    "--clinical-tables-dir", type=Path, default=Path("docs/clinical_extensions/tables")
  )

  ledger = sub.add_parser(
    "manuscript-ledger",
    help="Resolve every manuscript number from its artifact key into results_ledger.{md,csv}.",
  )
  ledger.add_argument("--spec", type=Path, default=Path("docs/manuscript/ledger_spec.csv"))
  ledger.add_argument("--root", type=Path, default=Path("."))
  ledger.add_argument("--output-md", type=Path, default=Path("docs/manuscript/results_ledger.md"))
  ledger.add_argument("--output-csv", type=Path, default=Path("docs/manuscript/results_ledger.csv"))
  ledger.add_argument("--check", action="store_true", help="Fail if resolved values differ from the committed ledger.")

  audit = sub.add_parser(
    "external-code15",
    help="External CODE-15%% validation commands (inventory, feasibility, ...).",
  )
  external_sub = audit.add_subparsers(dest="external_command", required=True)

  audit_inv = external_sub.add_parser(
    "audit",
    help="E0: Noninvasive source/provenance inventory. Reads no waveform payloads.",
  )
  audit_inv.add_argument(
    "--data-dir",
    type=Path,
    default=Path("~/data/code15pct/").expanduser(),
    help="Read-only CODE-15%% source directory.",
  )
  audit_inv.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/inventory"),
    help="Manifest output directory (never the canonical artifacts/probe tree).",
  )
  audit_inv.add_argument(
    "--skip-checksums",
    action="store_true",
    help="Skip full-file SHA256 checksums (faster, weaker provenance).",
  )
  audit_inv.add_argument(
    "--skip-hdf5-keys",
    action="store_true",
    help="Skip per-partition HDF5 dataset key inspection.",
  )
  audit_inv.add_argument(
    "--probe-dir",
    type=Path,
    default=Path("artifacts/probe"),
    help="Canonical MIMIC probe artifact directory to baseline-checksum.",
  )

  audit_prep = external_sub.add_parser(
    "prepare",
    help="E1: Build the first-exam index cohort, endpoint labels, feasibility census, and sealed split.",
  )
  audit_prep.add_argument(
    "--data-dir",
    type=Path,
    default=Path("~/data/code15pct/").expanduser(),
    help="Read-only CODE-15%% source directory.",
  )
  audit_prep.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort"),
    help="Cohort/census output directory (never the canonical artifacts/probe tree).",
  )
  audit_prep.add_argument(
    "--report-path",
    type=Path,
    default=Path("docs/external_validation/01_code15_feasibility.md"),
    help="Endpoint memo to update with the aggregate feasibility census.",
  )
  audit_prep.add_argument(
    "--timey-unit",
    type=str,
    default="years",
    help="Verified timey unit; must match the endpoint memo (fail fast otherwise).",
  )
  audit_prep.add_argument("--seed", type=int, default=1912)

  extract = external_sub.add_parser(
    "extract",
    help="E2: Extract frozen CIIS/D-BETA representations for CODE-15%% index ECGs.",
  )
  extract.add_argument(
    "--data-dir",
    type=Path,
    default=Path("~/data/code15pct/").expanduser(),
    help="Read-only CODE-15%% source directory.",
  )
  extract.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_index_cohort.parquet"),
    help="Index cohort parquet from `external-code15 prepare`.",
  )
  extract.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Feature output directory (defaults to features/, or features_preflight/ with --limit).",
  )
  extract.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Development-only sample cap (preflight mode; outcome-free sampling).",
  )
  extract.add_argument("--device", type=str, default="auto", help="Torch device for D-BETA.")
  extract.add_argument("--batch-size", type=int, default=32)
  extract.add_argument("--checkpoint-interval", type=int, default=2000)
  extract.add_argument("--partitions", type=int, nargs="*", default=None,
    help="Subset of partition indices to process (default: all found).")

  transport = external_sub.add_parser(
    "transport",
    help="E3: Evaluate the frozen MIMIC probes on extracted CODE-15%% features.",
  )
  transport.add_argument(
    "--features-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/features"),
    help="Feature directory from `external-code15 extract`.",
  )
  transport.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_index_cohort.parquet"),
    help="Index cohort parquet from `external-code15 prepare`.",
  )
  transport.add_argument(
    "--split-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_split_assignments.csv"),
    help="Split assignment CSV from `external-code15 prepare`.",
  )
  transport.add_argument(
    "--model-dir",
    type=Path,
    default=Path("artifacts/probe"),
    help="Canonical frozen MIMIC probe directory (hash-verified, never modified).",
  )
  transport.add_argument(
    "--baseline-manifest",
    type=Path,
    default=Path("artifacts/external/code15pct/inventory/mimic_probe_baseline.json"),
    help="E0 baseline manifest used for frozen-artifact hash verification.",
  )
  transport.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/transport"),
    help="Output directory for the aggregate transport results JSON.",
  )
  transport.add_argument("--n-bootstraps", type=int, default=1000)
  transport.add_argument("--seed", type=int, default=1912)

  verify_alignment = external_sub.add_parser(
    "verify-alignment",
    help="Independently compare CODE feature checkpoints to saved rows by exam_id.",
  )
  verify_alignment.add_argument(
    "--features-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/features"),
    help="Feature directory containing saved arrays, metadata, and partition checkpoints.",
  )
  verify_alignment.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_index_cohort.parquet"),
    help="Index cohort parquet used to verify exam_id-to-patient_id correspondence.",
  )
  verify_alignment.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/alignment"),
    help="Directory for the aggregate-only alignment verification report.",
  )

  waveform_audit = external_sub.add_parser(
    "audit-waveforms",
    help="Audit CODE-15%% duration, padding, extraction failures, and locked-test performance by waveform.",
  )
  waveform_audit.add_argument(
    "--data-dir",
    type=Path,
    default=Path("~/data/code15pct/").expanduser(),
    help="Read-only CODE-15%% HDF5 source directory.",
  )
  waveform_audit.add_argument(
    "--features-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/features"),
    help="Repaired extracted features, metadata, and extraction manifest.",
  )
  waveform_audit.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_index_cohort.parquet"),
    help="Index cohort parquet with HDF5 source filenames and one-year labels.",
  )
  waveform_audit.add_argument(
    "--split-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_split_assignments.csv"),
    help="Patient-level development/validation/test assignment CSV.",
  )
  waveform_audit.add_argument(
    "--model-dir",
    type=Path,
    default=Path("artifacts/probe"),
    help="Canonical frozen MIMIC E3 probe directory (hash-verified, never modified).",
  )
  waveform_audit.add_argument(
    "--baseline-manifest",
    type=Path,
    default=Path("artifacts/external/code15pct/inventory/mimic_probe_baseline.json"),
    help="E0 baseline manifest used for frozen-probe hash verification.",
  )
  waveform_audit.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/waveform_audit"),
    help="Directory for the aggregate-only waveform audit report.",
  )
  waveform_audit.add_argument("--n-bootstraps", type=int, default=1000)
  waveform_audit.add_argument("--seed", type=int, default=20261002)
  waveform_audit.add_argument("--batch-size", type=int, default=256)

  probes_cmd = external_sub.add_parser(
    "probes",
    help="E4: Fit CODE-15%% linear probes (A/B/A+B, demographics-adjusted, PCA-15 control).",
  )
  probes_cmd.add_argument(
    "--features-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/features"),
    help="Feature directory from `external-code15 extract`.",
  )
  probes_cmd.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_index_cohort.parquet"),
    help="Index cohort parquet from `external-code15 prepare`.",
  )
  probes_cmd.add_argument(
    "--split-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_split_assignments.csv"),
    help="Split assignment CSV from `external-code15 prepare`.",
  )
  probes_cmd.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/probes"),
    help="Output directory for probe artifacts and the aggregate report.",
  )
  probes_cmd.add_argument("--horizon", type=str, default="365d", choices=["365d", "90d", "30d"])
  probes_cmd.add_argument("--seed", type=int, default=42)
  probes_cmd.add_argument("--n-bootstraps", type=int, default=1000)

  hetero = external_sub.add_parser(
    "heterogeneity",
    help="E5: residual-risk gradients, discordance cells, age/sex subgroups (dev/val cutpoints).",
  )
  hetero.add_argument(
    "--features-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/features"),
    help="Feature directory from `external-code15 extract`.",
  )
  hetero.add_argument(
    "--model-dir",
    type=Path,
    default=Path("artifacts/probe"),
    help="Canonical frozen MIMIC probe directory (hash-verified, never modified).",
  )
  hetero.add_argument(
    "--baseline-manifest",
    type=Path,
    default=Path("artifacts/external/code15pct/inventory/mimic_probe_baseline.json"),
    help="E0 baseline manifest used for frozen-artifact hash verification.",
  )
  hetero.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/heterogeneity"),
    help="Output directory for the aggregate heterogeneity report.",
  )
  hetero.add_argument(
    "--split-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_split_assignments.csv"),
    help="Split assignment CSV from `external-code15 prepare`.",
  )
  hetero.add_argument("--horizon", type=str, default="365d", choices=["365d", "90d", "30d"])
  hetero.add_argument("--seed", type=int, default=1912)
  hetero.add_argument("--n-bootstraps", type=int, default=1000)

  deploy = external_sub.add_parser(
    "deployment",
    help="E6: cascade policy operating curves, capture metrics and compute benchmark.",
  )
  deploy.add_argument(
    "--features-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/features"),
    help="Feature directory from `external-code15 extract`.",
  )
  deploy.add_argument(
    "--model-dir",
    type=Path,
    default=Path("artifacts/probe"),
    help="Canonical frozen MIMIC probe directory (hash-verified, never modified).",
  )
  deploy.add_argument(
    "--baseline-manifest",
    type=Path,
    default=Path("artifacts/external/code15pct/inventory/mimic_probe_baseline.json"),
    help="E0 baseline manifest used for frozen-artifact hash verification.",
  )
  deploy.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/deployment"),
    help="Output directory for the aggregate deployment report.",
  )
  deploy.add_argument(
    "--split-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_split_assignments.csv"),
    help="Split assignment CSV from `external-code15 prepare`.",
  )
  deploy.add_argument("--horizon", type=str, default="365d", choices=["365d", "90d", "30d"])
  deploy.add_argument("--seed", type=int, default=1912)
  deploy.add_argument("--n-bootstraps", type=int, default=1000)
  deploy.add_argument(
    "--random-routing-seeds",
    type=int,
    default=200,
    help="Number of deterministic seeds for the random-routing stability summary.",
  )
  failure_audit = external_sub.add_parser(
    "failure-audit",
    help="Audit02: characterize extraction failures by age, sex, and 1-year mortality (aggregate-only).",
  )
  failure_audit.add_argument(
    "--manifest-path",
    type=Path,
    default=Path("artifacts/external/code15pct/features/extraction_manifest.json"),
    help="Extraction manifest with recorded failure records.",
  )
  failure_audit.add_argument(
    "--cohort-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_index_cohort.parquet"),
    help="Index cohort parquet from `external-code15 prepare`.",
  )
  failure_audit.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/inventory"),
    help="Output directory for the aggregate failure-audit report.",
  )
  probe_ipcw = external_sub.add_parser(
    "probe-ipcw-fit",
    help="Audit02 P1-3: IPCW-weighted fixed-horizon probe fitting sensitivity.",
  )
  probe_ipcw.add_argument(
    "--features-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/features"),
    help="Feature directory from `external-code15 extract`.",
  )
  probe_ipcw.add_argument(
    "--split-path",
    type=Path,
    default=Path("artifacts/external/code15pct/cohort/code15_split_assignments.csv"),
    help="Split assignment CSV from `external-code15 prepare`.",
  )
  probe_ipcw.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/external/code15pct/probes"),
    help="Output directory for the aggregate IPCW-fit sensitivity report.",
  )
  probe_ipcw.add_argument("--horizon", type=str, default="365d", choices=["365d", "90d", "30d"])
  probe_ipcw.add_argument("--seed", type=int, default=42)
  probe_ipcw.add_argument("--n-bootstraps", type=int, default=1000)
  return parser



def run_cohort(args: argparse.Namespace) -> int:
  """Execute cohort derivation and write inspectable artifacts."""
  paths = resolve_data_paths(args.mimic_root, args.ecg_root)
  records_df, patients_df, admissions_df = load_dataset_tables(paths)
  ratios = SplitRatios(dev=args.dev_ratio, val=args.val_ratio, test=args.test_ratio)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  default_manifest_path = output_dir / "waveform_eligibility_manifest.parquet"

  eligible_study_ids: set[int] | None = None
  manifest_df: pl.DataFrame | None = None
  source_study_ids = set(records_df["study_id"].to_list())

  if args.compute_waveform_manifest:
    # Explicit user flag takes precedence over any existing default manifest
    print(f"Force-computing waveform eligibility manifest across {len(records_df):,} records...")
    manifest_df = build_waveform_eligibility_manifest(
      records_df=records_df,
      ecg_data_dir=paths.mimic_iv_ecg_dir,
      is_complete=True,
      n_source_records=len(records_df),
      provenance_output_path=default_manifest_path.with_suffix(".json"),
    )
    manifest_df.write_parquet(default_manifest_path)
    eligible_study_ids = set(
      int(v) for v in manifest_df.filter(pl.col("waveform_eligible") == True)["study_id"].to_list()  # noqa: E712
    )
    print(f"Wrote complete waveform eligibility manifest to {default_manifest_path} ({len(eligible_study_ids):,} eligible studies)")
  elif args.waveform_eligibility_manifest is not None:
    manifest_p = Path(args.waveform_eligibility_manifest)
    if not manifest_p.exists():
      raise FileNotFoundError(
        f"Waveform eligibility manifest not found at '{manifest_p}'."
      )
    expected_rec_hash = compute_record_list_hash(records_df) if not args.allow_incomplete_manifest else None
    eligible_study_ids, manifest_df = load_waveform_eligibility_manifest(
      manifest_p,
      require_complete=not args.allow_incomplete_manifest,
      expected_study_ids=None if args.allow_incomplete_manifest else source_study_ids,
      expected_record_list_hash=expected_rec_hash,
    )
    print(f"Loaded waveform eligibility manifest from {manifest_p} ({len(eligible_study_ids):,} eligible studies)")
  elif default_manifest_path.exists():
    try:
      expected_rec_hash = compute_record_list_hash(records_df) if not args.allow_incomplete_manifest else None
      eligible_study_ids, manifest_df = load_waveform_eligibility_manifest(
        default_manifest_path,
        require_complete=not args.allow_incomplete_manifest,
        expected_study_ids=None if args.allow_incomplete_manifest else source_study_ids,
        expected_record_list_hash=expected_rec_hash,
      )
      print(f"Loaded existing waveform eligibility manifest from {default_manifest_path} ({len(eligible_study_ids):,} eligible studies)")
    except ValueError as exc:
      print(f"Existing default manifest at {default_manifest_path} cannot be used: {exc}")
      print(f"Auto-computing complete manifest across {len(records_df):,} records...")
      manifest_df = build_waveform_eligibility_manifest(
        records_df=records_df,
        ecg_data_dir=paths.mimic_iv_ecg_dir,
        is_complete=True,
        n_source_records=len(records_df),
        provenance_output_path=default_manifest_path.with_suffix(".json"),
      )
      manifest_df.write_parquet(default_manifest_path)
      eligible_study_ids = set(
        int(v) for v in manifest_df.filter(pl.col("waveform_eligible") == True)["study_id"].to_list()  # noqa: E712
      )
      print(f"Wrote auto-computed waveform eligibility manifest to {default_manifest_path} ({len(eligible_study_ids):,} eligible studies)")
  else:
    print(f"No waveform eligibility manifest found. Automatically computing complete manifest across {len(records_df):,} records...")
    manifest_df = build_waveform_eligibility_manifest(
      records_df=records_df,
      ecg_data_dir=paths.mimic_iv_ecg_dir,
      is_complete=True,
      n_source_records=len(records_df),
      provenance_output_path=default_manifest_path.with_suffix(".json"),
    )
    manifest_df.write_parquet(default_manifest_path)
    eligible_study_ids = set(
      int(v) for v in manifest_df.filter(pl.col("waveform_eligible") == True)["study_id"].to_list()  # noqa: E712
    )
    print(f"Wrote auto-computed waveform eligibility manifest to {default_manifest_path} ({len(eligible_study_ids):,} eligible studies)")

  # Locked-reanalysis UX guardrail
  if getattr(args, "locked_reanalysis", False) and getattr(args, "new_split", False):
    raise ValueError("Cannot specify both --locked-reanalysis and --new-split.")

  split_path = output_dir / "split_assignments.csv"
  preserve_split = args.preserve_split_from

  if getattr(args, "locked_reanalysis", False):
    if preserve_split is None:
      if split_path.exists():
        preserve_split = split_path
        print(f"[Locked Reanalysis] Preserving partition assignments from existing: {preserve_split}")
      else:
        raise ValueError(
          "Locked reanalysis mode (--locked-reanalysis) requires a frozen split assignments file. "
          f"Please specify --preserve-split-from <path> (target not found at default {split_path})."
        )
  elif not getattr(args, "new_split", False) and preserve_split is None:
    if split_path.exists():
      raise RuntimeError(
        f"Existing split assignments detected at '{split_path}'. "
        "To preserve these locked partitions for audit-corrected locked-set reanalysis, run:\n"
        f"  ecg-rep cohort --locked-reanalysis --preserve-split-from {split_path}\n"
        "To intentionally discard and generate a new randomized split, pass '--new-split'."
      )

  is_locked = bool(getattr(args, "locked_reanalysis", False))
  allow_new_test = bool(getattr(args, "allow_new_test", False)) if is_locked else True
  result = build_primary_cohort_and_split(
    records_df=records_df,
    patients_df=patients_df,
    admissions_df=admissions_df,
    ratios=ratios,
    seed=args.seed,
    existing_split_path=preserve_split,
    eligible_study_ids=eligible_study_ids,
    allow_new_test=allow_new_test,
  )
  summary = compute_split_summary_statistics(result.cohort_df)

  cohort_path = output_dir / "primary_cohort.parquet"
  split_path = output_dir / "split_assignments.csv"
  report_path = output_dir / "cohort_flow.md"
  ledger_path = output_dir / "predictor_availability_ledger.md"
  summary_path = output_dir / "split_summary.json"
  pres_report_path = output_dir / "preservation_report.json"

  result.cohort_df.write_parquet(cohort_path)
  save_split_assignments(result.cohort_df, split_path)
  report_path.write_text(
    generate_cohort_flow_markdown(result, summary_stats=summary),
    encoding="utf-8",
  )
  ledger_path.write_text(availability_ledger_markdown(), encoding="utf-8")
  summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if result.preservation_report is not None:
    pres_dict = {
      "n_preserved": result.preservation_report.n_preserved,
      "n_dropped": result.preservation_report.n_dropped,
      "n_new_dev": result.preservation_report.n_new_dev,
      "n_new_val": result.preservation_report.n_new_val,
      "n_new_test": result.preservation_report.n_new_test,
    }
    pres_report_path.write_text(json.dumps(pres_dict, indent=2) + "\n", encoding="utf-8")

  print(f"Wrote cohort ({len(result.cohort_df):,} patients) to {cohort_path}")
  print(f"Wrote split assignments to {split_path}")
  print(f"Wrote attrition report to {report_path}")
  print(
    "Split sizes:",
    {k: result.split_subject_counts[k] for k in ("dev", "val", "test")},
  )
  print(
    "30d events:",
    {
      "dev": summary.get("dev_mortality_30d_events"),
      "val": summary.get("val_mortality_30d_events"),
      "test": summary.get("test_mortality_30d_events"),
    },
  )
  return 0


def run_manifest(args: argparse.Namespace) -> int:
  """Build waveform eligibility manifest independently from WFDB headers."""
  paths = resolve_data_paths(args.mimic_root, args.ecg_root)
  records_df = load_record_list(paths.record_list_path)
  n_source_records = len(records_df)
  is_complete = args.limit is None

  if args.output_path is not None:
    output_path = Path(args.output_path)
  elif args.limit is not None:
    output_path = Path("artifacts/cohort/waveform_eligibility_manifest_sample.parquet")
  else:
    output_path = Path("artifacts/cohort/waveform_eligibility_manifest.parquet")

  if args.limit is not None:
    records_df = records_df.head(args.limit)

  print(f"Building waveform eligibility manifest for {len(records_df):,} records (complete={is_complete})...")
  manifest_df = build_waveform_eligibility_manifest(
    records_df=records_df,
    ecg_data_dir=paths.mimic_iv_ecg_dir,
    progress_every=args.progress_every,
    is_complete=is_complete,
    n_source_records=n_source_records,
    provenance_output_path=output_path.with_suffix(".json"),
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  manifest_df.write_parquet(output_path)
  n_eligible = int(manifest_df["waveform_eligible"].sum())
  print(f"Wrote waveform eligibility manifest ({n_eligible:,}/{len(manifest_df):,} eligible) to {output_path}")
  return 0


def run_representations(args: argparse.Namespace) -> int:
  """Extract shared-index ECG representations A and B."""
  paths = resolve_data_paths(args.mimic_root, args.ecg_root)
  cohort_path = Path(args.cohort_path)
  if not cohort_path.exists():
    raise FileNotFoundError(
      f"Cohort parquet not found at {cohort_path}. Run `ecg-rep cohort` first."
    )
  cohort_df = pl.read_parquet(cohort_path)
  if args.limit is not None:
    cohort_df = cohort_df.head(args.limit)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  if args.skip_ciis:
    meta_path = output_dir / "ciis_features_meta.parquet"
    if not meta_path.exists():
      raise FileNotFoundError(
        f"--skip-ciis requires existing CIIS artifacts under {output_dir}"
      )
    verify_representation_artifact_integrity(output_dir, cohort_df, check_b=False)
    meta_a = pl.read_parquet(meta_path)
    print(f"Reused and verified CIIS artifacts from {output_dir} ({len(meta_a):,} rows)")
  else:
    # Mandatory Preflight Gate for canonical non-smoke extraction
    if args.limit is None and not getattr(args, "override_preflight", False):
      preflight_report_path = output_dir / "ciis_preflight_acceptance_report.json"
      if not preflight_report_path.exists():
        raise RuntimeError(
          f"Preflight acceptance report missing at '{preflight_report_path}'. "
          "The canonical study pipeline requires running `ecg-rep preflight` prior to full representation extraction. "
          "Run `ecg-rep preflight --output-dir artifacts/reps` or pass `--override-preflight` to bypass during development."
        )
      preflight_data = json.loads(preflight_report_path.read_text(encoding="utf-8"))
      if preflight_data.get("status") != "PASSED":
        raise RuntimeError(
          f"Preflight acceptance report at '{preflight_report_path}' status is '{preflight_data.get('status')}', expected 'PASSED'. "
          "All morphology acceptance criteria must pass before canonical extraction. Pass `--override-preflight` to bypass."
        )
      rep_ext_ver = preflight_data.get("extractor_version")
      if rep_ext_ver != CIIS_EXTRACTOR_VERSION:
        raise RuntimeError(
          f"Preflight acceptance report extractor version mismatch: '{rep_ext_ver}' != current '{CIIS_EXTRACTOR_VERSION}'. "
          "Rerun `ecg-rep preflight` with current extractor."
        )
      rep_schema_hash = preflight_data.get("feature_schema_hash")
      if rep_schema_hash != CIIS_FEATURE_SCHEMA_HASH:
        raise RuntimeError(
          f"Preflight acceptance report schema hash mismatch: '{rep_schema_hash}' != current '{CIIS_FEATURE_SCHEMA_HASH}'. "
          "Rerun `ecg-rep preflight` with current feature schema."
        )
      print(f"[CIIS Representations] Verified PASSED preflight report ({rep_ext_ver}, schema: {rep_schema_hash[:8]}...)")

    meta_a, features_a, prov_a = extract_ciis_feature_cohort(
      cohort_df,
      ecg_data_dir=paths.mimic_iv_ecg_dir,
      progress_every=args.progress_every,
      checkpoint_path=output_dir / "ciis_checkpoint.npz",
      checkpoint_interval=args.checkpoint_interval,
      num_workers=getattr(args, "num_workers", 1),
    )
    save_representation_artifacts(output_dir, meta_a, features_a, prov_a, "ciis_features")
    print(
      f"CIIS features: {prov_a.n_valid}/{prov_a.n_records} valid "
      f"in {prov_a.elapsed_seconds:.1f}s -> {output_dir}"
    )

    # Perform distribution QC and multi-tier measurement validation for Representation A (Z_A)
    ciis_full_df = meta_a.select(["subject_id", "study_id"]).hstack(
      pl.DataFrame(features_a, schema=list(CIIS_FEATURE_NAMES))
    )
    dist_qc = validate_ciis_feature_distributions(ciis_full_df)
    (output_dir / "ciis_distribution_qc.json").write_text(
      json.dumps(dist_qc, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    print(
      f"[CIIS QC] Physiologic bounds audit: "
      f"{'PASS' if dist_qc['all_physiologically_valid'] else 'WARN (out-of-bounds detected)'}"
    )

    reference_df = None
    if getattr(args, "validate_ciis_sample", 0) > 0 and len(cohort_df) > 0:
      n_val_sample = min(args.validate_ciis_sample, len(cohort_df))
      val_sample_df = cohort_df.head(n_val_sample)
      ref_rows = []
      for r in val_sample_df.iter_rows(named=True):
        sid = int(r["study_id"])
        rp = paths.mimic_iv_ecg_dir / str(r["path"])
        try:
          sig, lnames, fs = read_ecg_waveform(rp)
          b = extract_independent_benchmark_measurements(sig, lnames, fs)
          row_dict: dict[str, Any] = {"study_id": sid}
          for fn in CIIS_FEATURE_NAMES:
            row_dict[fn] = b.get(fn, 0.0)
          ref_rows.append(row_dict)
        except Exception:
          pass
      if ref_rows:
        reference_df = pl.DataFrame(ref_rows)
        print(f"[CIIS Validation] Generated independent benchmark measurements for {len(reference_df)} records")

    machine_measurements_df = None
    if paths.machine_measurements_path.exists():
      try:
        machine_measurements_df = load_machine_measurements(paths.machine_measurements_path)
      except Exception as exc:  # noqa: BLE001
        print(f"[CIIS Validation] Could not load machine_measurements.csv: {exc}")

    val_study = run_ciis_measurement_validation_study(
      extracted_df=ciis_full_df,
      reference_df=reference_df,
      machine_measurements_df=machine_measurements_df,
    )
    val_json = json.dumps(val_study, indent=2, sort_keys=True) + "\n"
    (output_dir / "ciis_measurement_validation.json").write_text(val_json, encoding="utf-8")
    (output_dir / "ciis_algorithmic_agreement_validation.json").write_text(val_json, encoding="utf-8")

  if args.skip_dbeta:
    print("Skipped D-BETA extraction (--skip-dbeta).")
    return 0

  device = resolve_torch_device(args.device)
  print(f"D-BETA device: {device}")

  if args.dbeta_random_weights:
    seed = args.dbeta_random_seed
    print(f"Extracting random D-BETA negative control (seed {seed}) on {device}...")
    meta_rand, emb_rand, prov_rand = extract_dbeta_embedding_cohort(
      cohort_df,
      ecg_data_dir=paths.mimic_iv_ecg_dir,
      batch_size=args.batch_size,
      device=device,
      progress_every=args.progress_every,
      checkpoint_path=output_dir / f"dbeta_random_seed{seed}_checkpoint.npz",
      checkpoint_interval=args.checkpoint_interval,
      random_weights=True,
      random_seed=seed,
    )
    save_representation_artifacts(
      output_dir,
      meta_rand,
      emb_rand,
      prov_rand,
      f"dbeta_random_seed{seed}_embeddings",
    )
    print(
      f"Random D-BETA embeddings (seed {seed}): {prov_rand.n_valid}/{prov_rand.n_records} valid "
      f"in {prov_rand.elapsed_seconds:.1f}s -> {output_dir}"
    )
    return 0
  config = DbetaConfig(device=device, batch_size=args.batch_size)
  model = load_dbeta_model(
    model_name=config.model_name,
    revision=config.model_version or "20ff3ccce1759d7d629171e15befafa9a424d2ca",
    device=config.device,
    trust_remote_code=config.trust_remote_code,
    local_checkpoint_path=config.local_checkpoint_path,
  )
  adapter = DbetaAdapter(config=config, model=model)
  meta_b, embeddings_b, prov_b = extract_dbeta_embedding_cohort(
    cohort_df,
    adapter=adapter,
    ecg_data_dir=paths.mimic_iv_ecg_dir,
    batch_size=args.batch_size,
    device=device,
    progress_every=args.progress_every,
    checkpoint_path=output_dir / "dbeta_checkpoint.npz",
    checkpoint_interval=args.checkpoint_interval,
  )
  save_representation_artifacts(output_dir, meta_b, embeddings_b, prov_b, "dbeta_embeddings")
  paired = join_representation_validity(meta_a, meta_b)
  paired.write_parquet(output_dir / "representation_pair_validity.parquet")
  pair_n = int(paired.filter(pl.col("representation_pair_valid")).height)
  print(
    f"D-BETA embeddings: {prov_b.n_valid}/{prov_b.n_records} valid "
    f"in {prov_b.elapsed_seconds:.1f}s"
  )
  print(f"Paired valid A∩B: {pair_n}/{len(paired)}")
  return 0


def run_context(args: argparse.Namespace) -> int:
  """Build and persist primary structured context C0."""
  paths = resolve_data_paths(args.mimic_root, args.ecg_root)
  cohort_path = Path(args.cohort_path)
  if not cohort_path.exists():
    raise FileNotFoundError(
      f"Cohort parquet not found at {cohort_path}. Run `ecg-rep cohort` first."
    )
  cohort_df = pl.read_parquet(cohort_path)
  _, _, admissions_df = load_dataset_tables(paths)
  diagnoses_df = load_diagnoses_table(paths)
  c0 = build_c0_features(cohort_df, admissions_df, diagnoses_df)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "c0_features.parquet"
  c0.write_parquet(out_path)
  prevalence: dict[str, float] = {}
  for col in ELIXHAUSER_FEATURE_NAMES:
    mean_val = c0.get_column(col).mean()
    prevalence[col] = float(mean_val) if isinstance(mean_val, (int, float)) else 0.0
  (output_dir / "c0_elixhauser_prevalence.json").write_text(
    json.dumps(prevalence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(f"Wrote C0 features for {len(c0):,} patients to {out_path}")
  print(f"Elixhauser feature count: {len(ELIXHAUSER_FEATURE_NAMES)}")
  top = sorted(prevalence.items(), key=lambda kv: kv[1], reverse=True)[:5]
  print("Top Elixhauser prevalences:", {k: round(v, 4) for k, v in top})
  return 0


def run_evaluate(args: argparse.Namespace) -> int:
  """Run ECG-only and optional context-adjusted locked probe evaluation."""
  reps_dir = Path(args.reps_dir)
  meta_a = reps_dir / "ciis_features_meta.parquet"
  feats_a = reps_dir / "ciis_features.npy"
  meta_b = reps_dir / "dbeta_embedding_meta.parquet"
  emb_b = reps_dir / "dbeta_embeddings.npy"
  if not meta_a.exists() or not feats_a.exists():
    raise FileNotFoundError(
      f"CIIS representation artifacts not found under {reps_dir}. "
      "Run `ecg-rep representations` first."
    )

  cohort_df = pl.read_parquet(Path(args.cohort_path))
  verify_representation_artifact_integrity(
    reps_dir=reps_dir,
    cohort=cohort_df,
    check_b=(meta_b.exists() and emb_b.exists()),
  )

  out = Path(args.output_dir)
  ecg_results = run_ecg_only_evaluation(
    cohort_path=Path(args.cohort_path),
    meta_a_path=meta_a,
    features_a_path=feats_a,
    meta_b_path=meta_b if meta_b.exists() else None,
    embeddings_b_path=emb_b if emb_b.exists() else None,
    output_dir=out,
    n_bootstraps=args.n_bootstraps,
    verify_provenance=True,
  )
  print("ECG-only status:", ecg_results.get("status"))
  if "rq1_ecg_only" in ecg_results:
    rq1 = ecg_results["rq1_ecg_only"]
    print(
      f"RQ1 ECG-only ΔAUROC={rq1['gain']:+.4f} "
      f"(A={rq1['score_a']:.4f}, B={rq1['score_b']:.4f})"
    )

  c0_path = Path(args.c0_path)
  if (
    not args.ecg_only
    and c0_path.exists()
    and meta_b.exists()
    and emb_b.exists()
  ):
    ctx_results = run_context_adjusted_evaluation(
      cohort_path=Path(args.cohort_path),
      c0_path=c0_path,
      meta_a_path=meta_a,
      features_a_path=feats_a,
      meta_b_path=meta_b,
      embeddings_b_path=emb_b,
      output_dir=out,
      n_bootstraps=args.n_bootstraps,
    )
    rq1c = ctx_results["rq1_context_adjusted"]
    print(
      f"RQ1 context-adjusted ΔAUROC={rq1c['gain']:+.4f} "
      f"(A={rq1c['score_a']:.4f}, B={rq1c['score_b']:.4f})"
    )
  else:
    print("Context-adjusted evaluation skipped (missing C0/D-BETA or --ecg-only).")

  if args.run_random_control:
    rand_dir = Path(args.reps_random_dir) if args.reps_random_dir is not None else reps_dir
    seed = args.random_seed
    meta_rand = rand_dir / f"dbeta_random_seed{seed}_meta.parquet"
    emb_rand = rand_dir / f"dbeta_random_seed{seed}_embeddings.npy"
    if not meta_rand.exists() or not emb_rand.exists():
      raise FileNotFoundError(
        f"Random D-BETA artifacts not found under {rand_dir} for seed {seed}. "
        "Run `ecg-rep representations --dbeta-random-weights` first."
      )
    if not meta_b.exists() or not emb_b.exists():
      raise FileNotFoundError(
        f"Pretrained D-BETA artifacts not found under {reps_dir}. "
        "Run `ecg-rep representations` first."
      )
    rand_results = run_random_dbeta_control(
      cohort_path=Path(args.cohort_path),
      meta_random_path=meta_rand,
      embeddings_random_path=emb_rand,
      meta_pretrained_path=meta_b,
      embeddings_pretrained_path=emb_b,
      output_dir=out,
      n_bootstraps=args.n_bootstraps,
    )
    print(
      f"Random D-BETA Control ΔAUROC={rand_results['gain']:+.4f} "
      f"(Random={rand_results['score_random_encoder']:.4f}, "
      f"Pretrained={rand_results['score_pretrained_dbeta']:.4f})"
    )
  return 0


def run_ami_administrative_command(args: argparse.Namespace) -> int:
  """Outcome-specific probes for principal-discharge AMI coding."""
  from ecg_rep.ami_administrative import run_administrative_ami
  from ecg_rep.paths import _find_data_file, load_dataset_tables, resolve_data_paths

  paths = resolve_data_paths(args.mimic_root, None)
  _records, _patients, admissions = load_dataset_tables(paths)
  diagnoses = _find_data_file(
    paths.mimic_iv_dir, ["hosp/diagnoses_icd.csv.gz", "hosp/diagnoses_icd.csv"],
  )
  labs = _find_data_file(paths.mimic_iv_dir, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])
  reps = Path(args.reps_dir)
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  audit = run_administrative_ami(
    cohort_path=Path(args.cohort_path),
    setting_path=Path(args.setting_path),
    admissions=admissions,
    diagnoses_path=diagnoses,
    labevents_path=labs,
    pair_validity_path=Path(args.pair_validity_path),
    c0_path=Path(args.c0_path),
    meta_a_path=reps / "ciis_features_meta.parquet",
    features_a_path=reps / "ciis_features.npy",
    meta_b_path=reps / "dbeta_embedding_meta.parquet",
    embeddings_b_path=reps / "dbeta_embeddings.npy",
    protocol_path=Path(args.protocol),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  print("Administrative AMI protocol hash:", audit["protocol_hash"])
  print("Adjudicated evaluation:", audit["adjudicated_evaluation"])
  return 0


def run_ami_feasibility_command(args: argparse.Namespace) -> int:
  """Development AMI evidence audit. Model evaluation is refused."""
  from ecg_rep.ami_outcome import refuse_ami_model_evaluation, run_ami_feasibility
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  if getattr(args, "evaluate", False):
    refuse_ami_model_evaluation()
  paths = resolve_data_paths(args.mimic_root, None)
  diagnoses = _find_data_file(
    paths.mimic_iv_dir, ["hosp/diagnoses_icd.csv.gz", "hosp/diagnoses_icd.csv"],
  )
  labs = _find_data_file(paths.mimic_iv_dir, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  summary = run_ami_feasibility(
    cohort_path=Path(args.cohort_path),
    setting_path=Path(args.setting_path),
    diagnoses_path=diagnoses,
    labevents_path=labs,
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
  )
  print("AMI feasibility decision:", summary["decision"])
  print("Development candidate codes:", summary["n_candidate_ami_code"])
  return 0


def run_clinical_acute_context(args: argparse.Namespace) -> int:
  """Development availability inventory. Cacute probes are fit by ``clinical-cacute``."""
  from ecg_rep.acute_context import inventory_development_availability
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  paths = resolve_data_paths(args.mimic_root, None)
  labs = _find_data_file(paths.mimic_iv_dir, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])
  charts = _find_data_file(paths.mimic_iv_dir, ["icu/chartevents.csv.gz", "icu/chartevents.csv"])
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  lab_map = {
    "creatinine_blood": [50912],
    "potassium_blood": [50971],
    "hemoglobin_blood": [51222],
    "wbc_blood": [51301],
    "lactate_blood_gas": [50813],
    "troponin_i": [51002, 52642],
    "troponin_t": [51003],
  }
  vital_map = {
    "heart_rate": [220045],
    "respiratory_rate": [220210],
    "spo2": [220277],
    "sbp_noninvasive": [220179],
    "dbp_noninvasive": [220180],
    "sbp_arterial": [220050],
    "dbp_arterial": [220051],
    "temperature_celsius": [223762],
    "temperature_fahrenheit": [223761],
  }
  audit = inventory_development_availability(
    cohort_path=Path(args.cohort_path),
    labevents_path=labs,
    chartevents_path=charts,
    lab_map=lab_map,
    vital_map=vital_map,
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    setting_path=Path(args.setting_path),
  )
  print("Development N:", audit["n_development"])
  print("Model refit:", audit["model_refit"])
  return 0


def run_clinical_subgroups(args: argparse.Namespace) -> int:
  """Frozen-score subgroup tables. Standardization is descriptive reweighting."""
  from ecg_rep.analysis import subject_race_from_admissions
  from ecg_rep.paths import load_dataset_tables, resolve_data_paths
  from ecg_rep.subgroup_casemix import run_subgroup_casemix

  paths = resolve_data_paths(args.mimic_root, None)
  _records, _patients, admissions = load_dataset_tables(paths)
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  audit = run_subgroup_casemix(
    predictions_path=Path(args.predictions_context_path),
    cohort_path=Path(args.cohort_path),
    setting_path=Path(args.setting_path),
    race_map=subject_race_from_admissions(admissions),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  print("Locked-test rows:", audit["n_locked_test"])
  return 0


def run_clinical_dca(args: argparse.Namespace) -> int:
  """Whole-cohort decision curve analysis of the frozen locked-test predictions."""
  import numpy as np

  from ecg_rep.dca import run_whole_cohort_dca

  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  audit = run_whole_cohort_dca(
    predictions_context_path=Path(args.predictions_context_path),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
  )
  print("Locked-test rows:", audit["n_rows"])
  print("Thresholds:", ", ".join(f"{100 * t:g}%" for t in audit["thresholds"]))
  return 0


def run_ptbxl_measurement_validity_command(args: argparse.Namespace) -> int:
  """S7: external measurement validity of representation A against GE 12SL on PTB-XL."""
  from ecg_rep.external.ptbxl_measurement_validity import run_ptbxl_measurement_validity

  root = Path(args.ptbxl_root)
  results = run_ptbxl_measurement_validity(
    records_root=root,
    database_path=root / "ptbxl_database.csv",
    statements_path=root / "scp_statements.csv",
    features_12sl_path=Path(args.features_12sl_path),
    sample_ids_path=Path(args.sample_ids_path),
    output_dir=Path(args.output_dir),
    tables_dir=Path(args.tables_dir),
    n_bootstraps=args.n_bootstraps,
  )
  mi = results["mi_discrimination"]
  print(
    f"PTB-XL: {results['n_extractor_valid']}/{results['n_sampled']} scored; "
    f"category kappa {results['categories']['category_quadratic_kappa']:.2f}; "
    f"MI AUROC extractor {mi['auroc_extractor']:.3f} vs 12SL {mi['auroc_12sl']:.3f}"
  )
  return 0


def run_machine_baseline_command(args: argparse.Namespace) -> int:
  """S8: post hoc cart global-measurement comparator A'."""
  from ecg_rep.machine_baseline import run_machine_baseline

  results = run_machine_baseline(
    cohort_path=Path(args.cohort_path),
    machine_measurements_path=Path(args.machine_measurements_path),
    c0_path=Path(args.c0_path),
    reps_dir=Path(args.reps_dir),
    reference_ecg_only_path=Path(args.reference_ecg_only_path),
    reference_context_path=Path(args.reference_context_path),
    output_dir=Path(args.output_dir),
    tables_dir=Path(args.tables_dir),
    n_bootstraps=args.n_bootstraps,
  )
  b_minus_ap = results["contrasts"]["context_b_minus_ap"]["auroc"]
  print(
    f"A' context-adjusted: (C0+B) - (C0+A') AUROC {b_minus_ap['point_estimate']:+.4f} "
    f"({b_minus_ap['ci_lower']:+.4f} to {b_minus_ap['ci_upper']:+.4f})"
  )
  return 0


def run_cart_report_feasibility_command(args: argparse.Namespace) -> int:
  """LM1a: development-only machine-report inventory for the A'' comparator."""
  from ecg_rep.cart_report_baseline import run_cart_report_feasibility

  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  results = run_cart_report_feasibility(
    cohort_path=Path(args.cohort_path),
    measurements_path=Path(args.machine_measurements_path),
    config_path=Path(args.config),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
  )
  gates = results["gates"]
  print(
    f"LM1 feasibility: {results['n_development_ecgs']:,} development ECGs; "
    f"coverage {gates['coverage_gate_fraction']:.4f} "
    f"({'PASS' if gates['coverage_gate_pass'] else 'FAIL'}); "
    f"vocabulary {gates['vocabulary_size']:,} "
    f"({'PASS' if gates['vocabulary_gate_pass'] else 'FAIL'}); "
    f"chosen representation: {results['chosen_representation']}"
  )
  return 0


def run_cart_report_evaluation_command(args: argparse.Namespace) -> int:
  """LM1b: frozen A'' locked-test evaluation."""
  from ecg_rep.cart_report_baseline import run_cart_report_evaluation

  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  results = run_cart_report_evaluation(
    cohort_path=Path(args.cohort_path),
    measurements_path=Path(args.machine_measurements_path),
    c0_path=Path(args.c0_path),
    reps_dir=Path(args.reps_dir),
    config_path=Path(args.config),
    feasibility_path=Path(args.feasibility_path),
    reference_ecg_only_path=Path(args.reference_ecg_only_path),
    reference_context_path=Path(args.reference_context_path),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
  )
  b_minus_a2 = results["contrasts"]["c_b_minus_c_a2"]["auroc"]
  print(
    f"LM1 (C0+B) - (C0+A'') AUROC {b_minus_a2['point_estimate']:+.4f} "
    f"({b_minus_a2['ci_lower']:+.4f} to {b_minus_a2['ci_upper']:+.4f}); "
    f"n_test={results['n_test']:,} events={results['n_test_events']}"
  )
  return 0


def run_cart_addon_contrast_command(args: argparse.Namespace) -> int:
  """LM1 addendum: paired add-on contrast (C0+A''+B) − (C0+A'') with determinism gates."""
  from ecg_rep.cart_addon_contrast import run_cart_addon_contrast

  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  results = run_cart_addon_contrast(
    cohort_path=Path(args.cohort_path),
    measurements_path=Path(args.machine_measurements_path),
    c0_path=Path(args.c0_path),
    reps_dir=Path(args.reps_dir),
    config_path=Path(args.config),
    feasibility_path=Path(args.feasibility_path),
    ledger_path=Path(args.ledger_path),
    reference_aggregates_path=Path(args.reference_aggregates_path),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
  )
  addon = results["contrasts"]["c_a2_b_minus_c_a2"]["auroc"]
  print(
    f"LM1 addon (C0+A''+B) - (C0+A'') AUROC {addon['point_estimate']:+.4f} "
    f"({addon['ci_lower']:+.4f} to {addon['ci_upper']:+.4f}); "
    f"n_test={results['n_test']:,} events={results['n_test_events']}; "
    f"resamples={results['n_bootstraps_addon']} (verification: {results['n_bootstraps_verification']})"
  )
  return 0


def run_code15_recalibration_command(args: argparse.Namespace) -> int:
  """LM4: CODE-15% local recalibration of transported frozen MIMIC scores."""
  from ecg_rep.external.code15_recalibration import (
    load_recalibration_config,
    run_code15_recalibration,
  )

  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  config = load_recalibration_config(Path(args.config))
  results = run_code15_recalibration(
    features_dir=Path(args.features_dir),
    cohort_path=Path(args.cohort_path),
    model_dir=Path(args.model_dir),
    split_path=Path(args.split_path),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    config=config,
    n_bootstraps=args.n_bootstraps,
    with_val_refit_sensitivity=not args.skip_val_refit_sensitivity,
  )
  for score_name, block in results["scores"].items():
    r2 = block["locked_test"]["R2_intercept_slope"]
    frozen = block["locked_test"]["R0_frozen"]
    print(
      f"LM4 score {score_name}: Brier {frozen['brier']:.4f} -> {r2['brier']:.4f} (R2); "
      f"calibration slope {frozen['slope']:.3f} -> {r2['slope']:.3f}; "
      f"O/E {frozen['oe_ratio']:.3f} -> {r2['oe_ratio']:.3f}"
    )
  return 0


def run_compute_benchmark_command(args: argparse.Namespace) -> int:
  """S4: protocol-matched compute benchmark."""
  from ecg_rep.compute_benchmark import run_compute_benchmark, run_downstream_benchmark

  if not args.skip_extraction:
    results = run_compute_benchmark(
      cohort_path=Path(args.cohort_path),
      ecg_root=Path(args.ecg_root),
      output_dir=Path(args.output_dir),
      tables_dir=Path(args.tables_dir),
      n_ecgs=args.n_ecgs,
      repeats=args.repeats,
      devices=tuple(args.devices),
    )
    for p in results["pathways"]:
      print(f"{p['pathway']}: {p['median_ms_per_ecg']:.2f} ms/ECG")
  if not args.skip_downstream:
    downstream = run_downstream_benchmark(
      cohort_path=Path(args.cohort_path),
      c0_path=Path(args.c0_path),
      reps_dir=Path(args.reps_dir),
      output_dir=Path(args.output_dir),
      tables_dir=Path(args.tables_dir),
    )
    for p in downstream["pathways"]:
      print(f"{p['pathway']}: fit {p['fit_seconds_median']:.1f} s, predict {p['predict_us_per_ecg_median']:.2f} us/ECG")
  return 0


def run_manuscript_figures_command(args: argparse.Namespace) -> int:
  """S5: manuscript figures and tables from frozen artifacts."""
  from ecg_rep.manuscript_figures import run_manuscript_figures

  result = run_manuscript_figures(
    artifacts_root=Path(args.artifacts_root),
    tables_dir=Path(args.tables_dir),
    figures_dir=Path(args.figures_dir),
    clinical_tables_dir=Path(getattr(args, "clinical_tables_dir", Path("docs/clinical_extensions/tables"))),
    n_bootstraps=args.n_bootstraps,
  )
  print(f"Rendered {len(result['figures'])} figures and {len(result['tables'])} tables.")
  return 0


def run_submission_figures_command(args: argparse.Namespace) -> int:
  """Round-5 audit: five submission-facing figures from committed outputs (display only)."""
  from ecg_rep.submission_figures import run_submission_figures

  result = run_submission_figures(
    figures_dir=Path(args.figures_dir),
    tables_dir=Path(args.tables_dir),
    artifacts_root=Path(args.artifacts_root),
    clinical_tables_dir=Path(getattr(args, "clinical_tables_dir", Path("docs/clinical_extensions/tables"))),
  )
  print(f"Rendered {len(result['figures'])} submission figures (v3.0 medRxiv editorial lineup; scientific figure artwork unchanged from v2.32) with counts {result['counts']}.")
  return 0



def run_manuscript_ledger_command(args: argparse.Namespace) -> int:
  """S6: results ledger resolved from artifact keys."""
  from ecg_rep.manuscript_ledger import build_ledger, ledger_drift, read_spec, write_ledger

  rows = build_ledger(read_spec(Path(args.spec)), Path(args.root))
  if args.check:
    drift = ledger_drift(rows, Path(args.output_csv))
    if drift:
      print(f"Ledger drift in {len(drift)} entries: {', '.join(drift)}")
      return 1
    print(f"Ledger matches artifacts ({len(rows)} entries).")
    return 0
  write_ledger(rows, Path(args.output_md), Path(args.output_csv))
  print(f"Wrote {len(rows)} ledger entries.")
  return 0


def random_control_table_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
  """Flatten the per-seed random-control summary into one row per seed and estimand."""
  rows: list[dict[str, Any]] = []
  for seed, res in summary["seeds"].items():
    ecg, ctx = res["ecg_only"], res["context_adjusted"]
    for estimand, metrics, contrasts in (
      (
        "ecg_only",
        ecg["test_metrics_random"],
        {
          "pretrained_minus_random_auroc": ecg["bootstrap"],
          "pretrained_minus_random_auprc": ecg["bootstrap_auprc"],
          "random_minus_a_auroc": ecg["random_minus_frozen_a_ecg_auroc"],
          "random_minus_a_auprc": ecg["random_minus_frozen_a_ecg_auprc"],
        },
      ),
      (
        "context_adjusted",
        ctx["test_metrics_random"],
        {
          "pretrained_minus_random_auroc": ctx["pretrained_minus_random_auroc"],
          "pretrained_minus_random_auprc": ctx["pretrained_minus_random_auprc"],
          "random_minus_a_auroc": ctx["random_minus_frozen_a_auroc"],
          "random_minus_a_auprc": ctx["random_minus_frozen_a_auprc"],
        },
      ),
    ):
      row: dict[str, Any] = {
        "seed": int(seed),
        "estimand": estimand,
        "n_test": metrics["n"],
        "n_events": metrics["n_events"],
        "random_auroc": metrics["auroc"],
        "random_auprc": metrics["auprc"],
        "random_calibration_slope": metrics["calibration_slope"],
      }
      for name, interval in contrasts.items():
        row[name] = interval["point_estimate"]
        row[f"{name}_ci_lower"] = interval["ci_lower"]
        row[f"{name}_ci_upper"] = interval["ci_upper"]
      rows.append(row)
  return rows


def run_random_control_command(args: argparse.Namespace) -> int:
  """Fit the ECG-only probe on random-weight D-BETA embeddings and contrast it with pretrained D-BETA."""
  from ecg_rep.evaluate import run_random_control_context

  reps_dir = Path(args.reps_dir)
  out_root = Path(args.output_dir)
  if out_root.resolve() == Path("artifacts/probe").resolve():
    raise ValueError("random-control must not write into the canonical artifacts/probe directory.")
  summary: dict[str, Any] = {"seeds": {}}
  for seed in args.random_seeds:
    meta_rand = reps_dir / f"dbeta_random_seed{seed}_meta.parquet"
    emb_rand = reps_dir / f"dbeta_random_seed{seed}_embeddings.npy"
    if not meta_rand.exists() or not emb_rand.exists():
      raise FileNotFoundError(
        f"Random D-BETA artifacts for seed {seed} not found under {reps_dir}. "
        "Run `ecg-rep representations --skip-ciis --dbeta-random-weights` first."
      )
    out = out_root / f"seed{seed}"
    ecg = run_random_dbeta_control(
      cohort_path=Path(args.cohort_path),
      meta_random_path=meta_rand,
      embeddings_random_path=emb_rand,
      meta_pretrained_path=reps_dir / "dbeta_embedding_meta.parquet",
      embeddings_pretrained_path=reps_dir / "dbeta_embeddings.npy",
      output_dir=out,
      n_bootstraps=args.n_bootstraps,
      pair_validity_path=reps_dir / "representation_pair_validity.parquet",
      reference_predictions_path=Path(args.reference_predictions_path),
    )
    ctx = run_random_control_context(
      cohort_path=Path(args.cohort_path),
      c0_path=Path(args.c0_path),
      meta_random_path=meta_rand,
      embeddings_random_path=emb_rand,
      pair_validity_path=reps_dir / "representation_pair_validity.parquet",
      reference_context_predictions_path=Path(args.reference_context_predictions_path),
      output_dir=out,
      n_bootstraps=args.n_bootstraps,
    )
    summary["seeds"][str(seed)] = {"ecg_only": ecg, "context_adjusted": ctx}
    print(
      f"seed {seed}: ECG-only random AUROC {ecg['score_random_encoder']:.4f} vs pretrained "
      f"{ecg['score_pretrained_dbeta']:.4f}; C0+random AUROC "
      f"{ctx['test_metrics_random']['auroc']:.4f}"
    )
  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "random_control_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  rows = random_control_table_rows(summary)
  tables_dir = Path(args.tables_dir)
  tables_dir.mkdir(parents=True, exist_ok=True)
  pl.DataFrame(rows).write_csv(tables_dir / "supp_random_encoder_control.csv")
  return 0


def run_pretraining_label_efficiency_command(args: argparse.Namespace) -> int:
  """LM2A: label-efficiency stress test of the random-encoder finding."""
  from ecg_rep.pretraining_stress import load_label_efficiency_config, run_label_efficiency

  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  cfg = load_label_efficiency_config(Path(args.config))
  results = run_label_efficiency(
    cohort_path=Path(args.cohort_path),
    reps_dir=Path(args.reps_dir),
    c0_path=Path(args.c0_path),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    config=cfg,
  )
  print("LM2A label-efficiency complete:", results["status"])
  print("Locked test:", results["n_test"], "rows,", results["n_test_events"], "deaths (unchanged).")
  return 0


def run_pretraining_ami_control_command(args: argparse.Namespace) -> int:
  """LM2B: architecture-matched random-vs-pretrained control for administrative AMI."""
  from ecg_rep.paths import _find_data_file, resolve_data_paths
  from ecg_rep.pretraining_stress import run_ami_random_control

  paths = resolve_data_paths(args.mimic_root, None)
  admissions_path = _find_data_file(
    paths.mimic_iv_dir, ["hosp/admissions.csv.gz", "hosp/admissions.csv"]
  )
  diagnoses_path = _find_data_file(
    paths.mimic_iv_dir, ["hosp/diagnoses_icd.csv.gz", "hosp/diagnoses_icd.csv"]
  )
  labs_path = _find_data_file(paths.mimic_iv_dir, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  results = run_ami_random_control(
    cohort_path=Path(args.cohort_path),
    setting_path=Path(args.setting_path),
    admissions_path=Path(admissions_path),
    diagnoses_path=Path(diagnoses_path),
    labevents_path=Path(labs_path),
    pair_validity_path=Path(args.pair_validity_path),
    c0_path=Path(args.c0_path),
    reps_dir=Path(args.reps_dir),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    random_seeds=tuple(args.random_seeds),
  )
  print("LM2B AMI pretraining control complete:", results["status"])
  return 0


def run_cmax_feasibility_command(args: argparse.Namespace) -> int:
  """LM3a: development-only Cmax availability audit (no outcomes, no model fits)."""
  from ecg_rep.cmax_context import run_cmax_feasibility
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  paths = resolve_data_paths(args.mimic_root, None)
  labs_path = _find_data_file(paths.mimic_iv_dir, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])
  charts_path = _find_data_file(
    paths.mimic_iv_dir, ["icu/chartevents.csv.gz", "icu/chartevents.csv"]
  )
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  audit = run_cmax_feasibility(
    cohort_path=Path(args.cohort_path),
    setting_path=Path(args.setting_path),
    labevents_path=Path(labs_path),
    chartevents_path=Path(charts_path),
    config_path=Path(args.config),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
  )
  print("Cmax feasibility complete:", audit["config_version"])
  print("Retained concepts:", len(audit["retained_concepts"]))
  print("ICU vitals decision:", audit["icu_vitals"]["decision"])
  return 0


def run_cmax_evaluate_command(args: argparse.Namespace) -> int:
  """LM3b: frozen same-subset Cmax evaluation (locked test touched once per model)."""
  from ecg_rep.cmax_context import run_cmax_evaluation
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  paths = resolve_data_paths(args.mimic_root, None)
  labs_path = _find_data_file(paths.mimic_iv_dir, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])
  charts_path = _find_data_file(
    paths.mimic_iv_dir, ["icu/chartevents.csv.gz", "icu/chartevents.csv"]
  )
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  results = run_cmax_evaluation(
    cohort_path=Path(args.cohort_path),
    setting_path=Path(args.setting_path),
    labevents_path=Path(labs_path),
    chartevents_path=Path(charts_path),
    c0_path=Path(args.c0_path),
    reps_dir=Path(args.reps_dir),
    config_path=Path(args.config),
    measurements_path=Path(args.measurements_path) if args.measurements_path else None,
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
  )
  print("Cmax evaluation complete:", results["status"])
  print("Surface:", results["n_surface"], "Locked test:", results["n_test"])
  return 0


def run_refit_uncertainty_command(args: argparse.Namespace) -> int:
  """LM5: refit-aware three-surface bootstrap for the headline B-A contrasts."""
  from ecg_rep.refit_uncertainty import MASTER_SEED, run_refit_uncertainty

  reps = Path(args.reps_dir)
  summary = run_refit_uncertainty(
    cohort_path=Path(args.cohort_path),
    c0_path=Path(args.c0_path),
    meta_a_path=reps / "ciis_features_meta.parquet",
    features_a_path=reps / "ciis_features.npy",
    meta_b_path=reps / "dbeta_embedding_meta.parquet",
    embeddings_b_path=reps / "dbeta_embeddings.npy",
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    supp_table_path=Path(args.supp_table),
    n_replicates=int(args.n_replicates),
    master_seed=int(args.master_seed),
    n_jobs=int(args.n_jobs),
    with_decomposition=not args.skip_decomposition,
    reuse_main=bool(args.reuse_main_bootstrap),
    primary_ecg_results_path=Path(args.primary_ecg_results),
    primary_context_results_path=Path(args.primary_context_results),
  )
  print("Refit-uncertainty complete:", summary["analysis_status"])
  for contrast, s in summary["contrasts"].items():
    print(
      f"  {contrast}: median {s['median']:+.4f} [{s['pct_2_5']:+.4f},{s['pct_97_5']:+.4f}] "
      f"P(>0)={s['fraction_positive']:.3f}"
    )
  print("Master seed:", MASTER_SEED)
  return 0


def run_subgroup_multiplicity_command(args: argparse.Namespace) -> int:
  """LM7: simultaneous bands for the existing context-adjusted subgroup contrasts."""
  from ecg_rep.paths import _find_data_file, resolve_data_paths
  from ecg_rep.subgroup_multiplicity import run_subgroup_multiplicity

  paths = resolve_data_paths(args.mimic_root, None)
  admissions = _find_data_file(paths.mimic_iv_dir, ["hosp/admissions.csv.gz", "hosp/admissions.csv"])
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  summary = run_subgroup_multiplicity(
    cohort_path=Path(args.cohort_path),
    predictions_context_path=Path(args.predictions_context),
    admissions_path=admissions,
    config_path=Path(args.config),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_commit=sha,
  )
  print("Subgroup multiplicity n:", summary["n"], "events:", summary["n_events"])
  return 0


def run_ascertainment_audit_command(args: argparse.Namespace) -> int:
  """LM6A: selection SMDs and direct evidence-of-survival recovery."""
  from ecg_rep.ascertainment_history import load_lm6_config, run_ascertainment_audit

  config = load_lm6_config(Path(args.config))
  if args.horizon_days is not None and float(args.horizon_days) != config.horizon_days:
    raise ValueError(
      f"--horizon-days {args.horizon_days} does not match frozen config horizon {config.horizon_days}."
    )
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  summary = run_ascertainment_audit(
    config_path=Path(args.config),
    cohort_path=Path(args.cohort_path),
    waveform_manifest=Path(args.waveform_manifest),
    c0_path=Path(args.c0_path),
    mimic_root=args.mimic_root,
    ecg_root=args.ecg_root,
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_commit=sha,
  )
  print("Ascertainment audit:", summary["recovery"]["by_class"])
  print("Scoring:", summary["scoring_status"])
  return 0


def run_history_sensitivity_command(args: argparse.Namespace) -> int:
  """LM6B: history-stratum B-A and the C0+H contrast on frozen primary splits."""
  from ecg_rep.ascertainment_history import load_lm6_config, run_history_sensitivity

  config = load_lm6_config(Path(args.config))
  n_boot = config.n_replicates if args.n_bootstraps is None else int(args.n_bootstraps)
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  summary = run_history_sensitivity(
    config_path=Path(args.config),
    cohort_path=Path(args.cohort_path),
    c0_path=Path(args.c0_path),
    reps_dir=Path(args.reps_dir),
    predictions_context_path=Path(args.predictions_context),
    mimic_root=args.mimic_root,
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_commit=sha,
    n_replicates=n_boot,
  )
  print("History sensitivity:", summary["contrasts"].keys())
  return 0


def run_compute_scaling_command(args: argparse.Namespace) -> int:
  """LM8: smoke-freeze batch sizes, then time the frozen grid once."""
  from ecg_rep.compute_scaling import run_compute_scaling

  summary = run_compute_scaling(
    cohort_path=Path(args.cohort_path),
    ecg_root=Path(args.ecg_root),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    skip_timing=bool(args.smoke_only),
  )
  print("LM8 timed_evaluation:", summary["timed_evaluation"])
  return 0


def run_manuscript_reporting_command(args: argparse.Namespace) -> int:
  """Evaluation-only manuscript tables and calibration on frozen locked-test predictions."""
  from ecg_rep.analysis import subject_race_from_admissions
  from ecg_rep.manuscript_reporting import run_manuscript_reporting
  from ecg_rep.paths import load_dataset_tables, resolve_data_paths

  paths = resolve_data_paths(args.mimic_root, None)
  _records, _patients, admissions = load_dataset_tables(paths)
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  report = run_manuscript_reporting(
    cohort_path=Path(args.cohort_path),
    c0_path=Path(args.c0_path),
    setting_path=Path(args.setting_path),
    pair_validity_path=Path(args.pair_validity_path),
    predictions_context_path=Path(args.predictions_context_path),
    predictions_ecg_only_path=Path(args.predictions_ecg_only_path),
    race_map=subject_race_from_admissions(admissions),
    output_dir=Path(args.output_dir),
    tables_dir=Path(args.tables_dir),
    figures_dir=Path(args.figures_dir),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  print("Locked-test rows:", report["n_test_analytic"], "events:", report["n_test_events"])
  for key, path in report["outputs"].items():
    print(f"{key}: {path}")
  return 0


def run_paired_valid_audit_command(args: argparse.Namespace) -> int:
  """Aggregate audit of locked-test patients excluded from the paired-valid analysis."""
  from ecg_rep.manuscript_reporting import run_paired_valid_exclusion_audit

  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  report = run_paired_valid_exclusion_audit(
    cohort_path=Path(args.cohort_path),
    pair_validity_path=Path(args.pair_validity_path),
    setting_path=Path(args.setting_path),
    output_dir=Path(args.output_dir),
    tables_dir=Path(args.tables_dir),
    source_sha=sha,
  )
  print(
    f"Locked test: {report['n_locked_test']} rows; paired-valid {report['n_paired_valid']}; "
    f"excluded {report['n_excluded']} (a_only {report['groups'][1]['n']}, "
    f"b_only {report['groups'][2]['n']}, neither {report['groups'][3]['n']})"
  )
  print(f"table: {report['outputs']['table']}")
  return 0


def run_bootstrap_estimand_sensitivity_command(args: argparse.Namespace) -> int:
  """Outcome-stratified vs ordinary paired patient-bootstrap intervals for headline contrasts."""
  from ecg_rep.manuscript_reporting import run_bootstrap_estimand_sensitivity

  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  report = run_bootstrap_estimand_sensitivity(
    predictions_ecg_path=Path(args.predictions_ecg_only_path),
    predictions_context_path=Path(args.predictions_context_path),
    analysis_results_path=Path(args.analysis_results_path),
    output_dir=Path(args.output_dir),
    tables_dir=Path(args.tables_dir),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  gap = report["max_abs_gap_recomputed_vs_frozen_stratified"]
  print(f"Recomputed stratified intervals match frozen headline values within {gap:.2e}.")
  print(f"table: {report['outputs']['table']}")
  return 0


def run_clinical_cacute(args: argparse.Namespace) -> int:
  """Refit the secondary context block. This does not replace the original C0 probes."""
  from ecg_rep.cacute import run_cacute_comparison
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  paths = resolve_data_paths(args.mimic_root, None)
  labs = _find_data_file(paths.mimic_iv_dir, ["hosp/labevents.csv.gz", "hosp/labevents.csv"])
  reps = Path(args.reps_dir)
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  audit = run_cacute_comparison(
    cohort_path=Path(args.cohort_path),
    c0_path=Path(args.c0_path),
    setting_path=Path(args.setting_path),
    labevents_path=labs,
    meta_a_path=reps / "ciis_features_meta.parquet",
    features_a_path=reps / "ciis_features.npy",
    meta_b_path=reps / "dbeta_embedding_meta.parquet",
    embeddings_b_path=reps / "dbeta_embeddings.npy",
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  print("Fixed test rows:", audit["n_fixed_test"])
  return 0


def run_clinical_mortality_sensitivity(args: argparse.Namespace) -> int:
  """Apply the frozen lexicon to frozen scores. The lexicon file is not edited."""
  from ecg_rep.mortality_sensitivity import run_mortality_sensitivity
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  paths = resolve_data_paths(None, args.ecg_root)
  measurements = _find_data_file(
    paths.mimic_iv_ecg_dir,
    ["machine_measurements.csv", "machine_measurements.csv.gz"],
  )
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  audit = run_mortality_sensitivity(
    predictions_context_path=Path(args.predictions_context_path),
    predictions_ecg_path=Path(args.predictions_ecg_path),
    measurements_path=measurements,
    dictionary_path=Path(args.dictionary),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  print("Locked-test rows:", audit["n_locked_test"])
  print("Lexicon hash:", audit["lexicon_hash"])
  return 0


def run_clinical_next_step_feasibility(args: argparse.Namespace) -> int:
  """A6 development-only feasibility. Locked-test splits are refused."""
  from ecg_rep.data import load_admissions
  from ecg_rep.next_step import run_next_step_feasibility
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  if args.split != "dev":
    raise SystemExit(
      "Next-step feasibility uses the development split only, so the outcome set cannot be "
      "tuned on validation or test."
    )
  paths = resolve_data_paths(args.mimic_root, args.ecg_root)
  hosp = paths.mimic_iv_dir
  audit = run_next_step_feasibility(
    cohort_path=Path(args.cohort_path),
    setting_path=Path(args.setting_path),
    pair_validity_path=Path(args.pair_validity_path),
    config_path=Path(args.config),
    labevents_path=_find_data_file(hosp, ["hosp/labevents.csv.gz", "hosp/labevents.csv"]),
    poe_path=_find_data_file(hosp, ["hosp/poe.csv.gz", "hosp/poe.csv"]),
    record_list_path=_find_data_file(paths.mimic_iv_ecg_dir, ["record_list.csv", "record_list.csv.gz"]),
    transfers_path=_find_data_file(hosp, ["hosp/transfers.csv.gz", "hosp/transfers.csv"]),
    admissions_df=load_admissions(_find_data_file(hosp, ["hosp/admissions.csv.gz", "hosp/admissions.csv"])),
    icustays_path=_find_data_file(hosp, ["icu/icustays.csv.gz", "icu/icustays.csv"]),
    cache_dir=Path(args.cache_dir),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    split=args.split,
    rebuild_cache=args.rebuild_cache,
  )
  print("Config hash:", audit["config_hash"])
  print("In-scope development ECGs:", audit["n_in_scope"], "ICU excluded:", audit["n_icu_excluded"])
  return 0


def run_clinical_next_step(args: argparse.Namespace) -> int:
  """A6 locked-test run. Refuses a config that is not the frozen one."""
  from ecg_rep.data import load_admissions
  from ecg_rep.next_step_analysis import run_next_step
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  paths = resolve_data_paths(args.mimic_root, args.ecg_root)
  hosp = paths.mimic_iv_dir
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  audit = run_next_step(
    cohort_path=Path(args.cohort_path),
    setting_path=Path(args.setting_path),
    pair_validity_path=Path(args.pair_validity_path),
    predictions_context_path=Path(args.predictions_context_path),
    predictions_ecg_path=Path(args.predictions_ecg_path),
    config_path=Path(args.config),
    labevents_path=_find_data_file(hosp, ["hosp/labevents.csv.gz", "hosp/labevents.csv"]),
    poe_path=_find_data_file(hosp, ["hosp/poe.csv.gz", "hosp/poe.csv"]),
    record_list_path=_find_data_file(paths.mimic_iv_ecg_dir, ["record_list.csv", "record_list.csv.gz"]),
    transfers_path=_find_data_file(hosp, ["hosp/transfers.csv.gz", "hosp/transfers.csv"]),
    admissions_df=load_admissions(_find_data_file(hosp, ["hosp/admissions.csv.gz", "hosp/admissions.csv"])),
    icustays_path=_find_data_file(hosp, ["icu/icustays.csv.gz", "icu/icustays.csv"]),
    cache_dir=Path(args.cache_dir),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  print("Fixed locked-test rows:", audit["n_fixed_test"], "deaths:", audit["n_fixed_deaths"])
  print("Config hash:", audit["config_hash"])
  return 0


def run_clinical_documented_policy_audit(args: argparse.Namespace) -> int:
  """Development-only tier audit and blinded review sample for A7."""
  from ecg_rep.documented_abnormality import run_development_audit
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  paths = resolve_data_paths(None, args.ecg_root)
  measurements = _find_data_file(
    paths.mimic_iv_ecg_dir,
    ["machine_measurements.csv", "machine_measurements.csv.gz"],
  )
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  audit = run_development_audit(
    cohort_path=Path(args.cohort_path),
    measurements_path=measurements,
    dictionary_path=Path(args.dictionary),
    output_dir=Path(args.output_dir),
    source_sha=sha,
  )
  print("Development reports:", audit["n_development"])
  print("Tier counts:", audit["tier_counts"])
  print("Policy spec hash:", audit["policy_spec_hash"])
  print("Review sample rows:", audit["n_review_sample"])
  return 0


def run_clinical_documented_policy(args: argparse.Namespace) -> int:
  """A7 locked-test run. Refuses a changed lexicon or policy specification."""
  from ecg_rep.documented_abnormality import run_documented_policy
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  paths = resolve_data_paths(None, args.ecg_root)
  measurements = _find_data_file(
    paths.mimic_iv_ecg_dir,
    ["machine_measurements.csv", "machine_measurements.csv.gz"],
  )
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  output_dir = Path(args.output_dir)
  audit = run_documented_policy(
    predictions_context_path=Path(args.predictions_context_path),
    predictions_ecg_path=Path(args.predictions_ecg_path),
    measurements_path=measurements,
    dictionary_path=Path(args.dictionary),
    dev_audit_path=output_dir / "development_tier_audit.json",
    output_dir=output_dir,
    report_path=Path(args.report_path),
    source_sha=sha,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  print("Locked-test rows:", audit["n_locked_test"])
  print("Tier counts:", audit["tier_counts"])
  print("Policy spec hash:", audit["policy_spec_hash"])
  return 0


def run_clinical_phenotypes(args: argparse.Namespace) -> int:
  """Development-only phenotype coverage. Locked-test outcome joins are refused."""
  from ecg_rep.ecg_phenotypes import run_development_phenotype_coverage
  from ecg_rep.paths import _find_data_file, resolve_data_paths

  if args.split != "dev":
    raise SystemExit(
      "Phenotype coverage audits use the development split only, so the lexicon cannot be "
      "tuned on validation or test text. Locked-test use goes through hash-checked analysis commands."
    )
  paths = resolve_data_paths(None, args.ecg_root)
  measurements = _find_data_file(
    paths.mimic_iv_ecg_dir,
    ["machine_measurements.csv", "machine_measurements.csv.gz"],
  )
  if not measurements.exists():
    raise FileNotFoundError(measurements)
  sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  summary = run_development_phenotype_coverage(
    cohort_path=Path(args.cohort_path),
    measurements_path=measurements,
    dictionary_path=Path(args.dictionary),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    source_sha=sha,
  )
  print("Development phenotype summary:", summary["summary_counts"])
  print("Dictionary hash:", summary["dictionary_hash"])
  return 0


def run_clinical_setting(args: argparse.Namespace) -> int:
  """Classify index ECG care setting without refitting mortality probes."""
  from ecg_rep.clinical_setting import run_setting_characterization

  audit = run_setting_characterization(
    cohort_path=Path(args.cohort_path),
    output_dir=Path(args.output_dir),
    report_path=Path(args.report_path),
    predictions_context_path=Path(args.predictions_context_path),
    predictions_ecg_path=Path(args.predictions_ecg_path),
    c0_path=Path(args.c0_path),
    pair_validity_path=Path(args.pair_validity_path),
    mimic_root=args.mimic_root,
    ecg_root=args.ecg_root,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  print("Setting counts:", audit["setting_counts"])
  print("Rule hash:", audit["setting_rule_hash"])
  print("Wrote", args.output_dir)
  return 0


def run_analyze(args: argparse.Namespace) -> int:
  """Run scientific estimands from probe prediction artifacts."""
  probe_dir = Path(args.probe_dir)
  report = run_scientific_analysis(
    predictions_ecg_path=probe_dir / "test_predictions_ecg_only.parquet",
    predictions_context_path=probe_dir / "test_predictions_context.parquet",
    cohort_path=Path(args.cohort_path),
    output_dir=Path(args.output_dir),
    reps_dir=Path(args.reps_dir),
    n_bootstraps=args.n_bootstraps,
    calibration_scores_path=Path(args.calibration_scores),
  )
  print("Scientific analysis status:", report.get("status"))
  print("Analyses:", sorted(report.get("analyses", {}).keys()))
  return 0


def run_dump_probe_scores(args: argparse.Namespace) -> int:
  """Score dev+val rows with the frozen context-adjusted probes (read-only)."""
  from ecg_rep.evaluate import dump_context_calibration_scores

  models_dir = Path(args.models_dir)
  for required in ("M_A.joblib", "M_B.joblib"):
    if not (models_dir / required).exists():
      raise FileNotFoundError(f"Frozen probe artifact missing: {models_dir / required}")
  summary = dump_context_calibration_scores(
    cohort_path=Path(args.cohort_path),
    c0_path=Path(args.c0_path),
    meta_a_path=Path(args.meta_a_path),
    features_a_path=Path(args.features_a_path),
    meta_b_path=Path(args.meta_b_path),
    embeddings_b_path=Path(args.embeddings_b_path),
    models_dir=models_dir,
    output_path=Path(args.output_path),
    verify_provenance=args.verify_provenance,
  )
  print("Calibration score table:", json.dumps(summary, indent=2))
  return 0


def _load_probability_cascade_config(path: Path) -> dict[str, Any]:
  import yaml

  raw = yaml.safe_load(path.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError("Probability-cascade config must be a YAML mapping.")
  return raw


def _authorize_probability_cascade_run(
  config: dict[str, Any],
  split: str,
  operation: str,
  *,
  allow_locked_test: bool = False,
) -> dict[str, Any]:
  if split not in ("dev", "val", "test"):
    raise ValueError("split must be dev, val, or test.")
  if operation == "fit-routing" and split == "test":
    raise RuntimeError("Routing cutoffs may never be fitted on locked-test rows.")
  if config.get("research_mode") != "exploratory":
    raise RuntimeError("Declare research_mode: exploratory to run the probability cascade.")
  if config.get("analysis_status") not in (
    "post_hoc_exploratory", "internal_resplit", "new_temporal_evaluation", "external_validation"
  ):
    raise RuntimeError("Config must declare an explicit exploratory analysis_status.")
  if split == "test":
    if config.get("analysis_status") != "post_hoc_exploratory":
      raise RuntimeError("Previously inspected locked-test replay must be labeled post_hoc_exploratory.")
  primary = config.get("primary")
  if not isinstance(primary, dict):
    raise ValueError("Probability-cascade config is missing the primary contract.")
  expected_contract = {
    "target_dataset": "mimic",
    "outcome": "mortality_30d",
    "outcome_horizon_days": 30,
    "split_unit": "subject_id",
    "model_scope": "ecg_only",
    "stage_a": "M_A_ECG",
    "stage_b": "M_B_ECG",
    "stage_a_score_column": "p_a_ecg",
    "stage_b_score_column": "p_b_ecg",
  }
  if any(primary.get(key) != value for key, value in expected_contract.items()):
    raise RuntimeError("Empirical routing is restricted to the frozen ECG-only MIMIC 30-day contract.")
  if operation == "fit-routing":
    reference_split = primary.get("routing_reference_split")
    if reference_split not in ("dev", "val"):
      raise RuntimeError("Config must designate exactly one dev/val routing_reference_split.")
    if split != reference_split:
      raise RuntimeError(f"Only designated reference split {reference_split!r} may fit routing cutoffs.")
  return primary


def _probability_cascade_output_dir(path: Path) -> Path:
  private_root = Path("artifacts/probability_cascade/v2").resolve()
  output_dir = path.resolve()
  if not output_dir.is_relative_to(private_root):
    raise ValueError("Patient-level cascade outputs must remain under artifacts/probability_cascade/v2/.")
  output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
  output_dir.chmod(0o700)
  return output_dir


def _cascade_score_frame(path: Path, split_column: str, split: str, columns: list[str]) -> pl.DataFrame:
  scan = pl.scan_parquet(path)
  available = set(scan.collect_schema().names())
  required = {split_column, *columns}
  missing = required - available
  if missing:
    raise ValueError(f"Score table missing required columns: {sorted(missing)}")
  outcome_columns = {"mortality_30d", "mortality_365d", "label", "labels", "outcome", "y"}
  if {column.casefold() for column in available} & outcome_columns:
    raise ValueError("Cascade score tables must not contain outcome/label columns.")
  observed_splits = set(scan.select(split_column).collect().get_column(split_column).to_list())
  if None in observed_splits:
    raise ValueError("Score table contains missing split assignments.")
  if split == "test":
    if observed_splits != {"test"}:
      raise ValueError("Locked-test replay requires a test-only score source.")
  elif not observed_splits <= {"dev", "val"}:
    raise ValueError("Non-test routing inputs may contain only dev/val rows, never test or unknown splits.")
  if split not in observed_splits:
    raise ValueError(f"Score table contains no rows for split={split!r}.")
  frame = scan.filter(pl.col(split_column) == split).select(columns).collect()
  if frame.is_empty():
    raise ValueError(f"Score table contains no rows for split={split!r}.")
  return frame


def _verify_cascade_score_source(primary: dict[str, Any], path: Path, split: str) -> str:
  hashes = primary.get("score_table_sha256_by_split")
  actual = _sha256_file(path)
  if hashes is not None:
    if not isinstance(hashes, dict):
      raise ValueError("score_table_sha256_by_split must be a mapping when supplied.")
    expected = hashes.get(split)
    if expected is not None:
      if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"Configured score-table SHA-256 is invalid for split={split!r}.")
      if actual != expected:
        raise RuntimeError("Score-table SHA-256 does not match the configured source manifest.")
  return actual


def _validate_unique_ecg_keys(frame: pl.DataFrame) -> None:
  """Require one keyed index ECG per patient before fitting or replaying."""
  identifiers: dict[str, list[Any]] = {
    name: frame.get_column(name).to_list() for name in ("subject_id", "study_id")
  }
  for name, values in identifiers.items():
    if any(
      value is None or isinstance(value, bool) or not isinstance(value, int) or value <= 0
      for value in values
    ):
      raise ValueError(f"Score table {name} keys must be non-null positive integers.")
    if len(set(values)) != len(values):
      raise ValueError(f"Score table must have unique {name} keys (one index ECG per patient).")


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
  with path.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  path.chmod(0o600)


def run_probability_cascade_fit_routing(args: argparse.Namespace) -> int:
  """Fit reference cutoffs only after the protocol and source gates pass."""
  from ecg_rep.probability_cascade import (
    RoutingPolicy,
    ThresholdFit,
    fit_reference_cutoff,
    save_policy,
  )

  config = _load_probability_cascade_config(Path(args.config))
  primary = _authorize_probability_cascade_run(config, args.split, "fit-routing")
  score_column = primary["stage_a_score_column"]
  schema_a = primary.get("stage_a_feature_schema_sha256")
  schema_b = primary.get("stage_b_feature_schema_sha256")
  if not isinstance(schema_a, str) or not isinstance(schema_b, str):
    raise RuntimeError("Verified Stage A/B feature schema hashes are required before fitting cutoffs.")
  score_source_sha256 = _verify_cascade_score_source(primary, Path(args.scores_path), args.split)
  frame = _cascade_score_frame(
    Path(args.scores_path), "split", args.split,
    ["subject_id", "study_id", score_column],
  )
  _validate_unique_ecg_keys(frame)
  raw_scores = frame.get_column(score_column).to_list()
  valid_scores: list[float] = []
  n_invalid = 0
  for score in raw_scores:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
      n_invalid += 1
      continue
    value = float(score)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
      n_invalid += 1
      continue
    valid_scores.append(value)
  targets = primary.get("reference_routing_fractions")
  if not isinstance(targets, list) or targets != [0.05, 0.10, 0.20, 0.40]:
    raise ValueError("The approved protocol must preserve routing targets 5%, 10%, 20%, and 40%.")

  candidates: list[tuple[ThresholdFit, RoutingPolicy]] = []
  for target_value in targets:
    fit = fit_reference_cutoff(
      valid_scores,
      float(target_value),
      args.split,
      reference_source_sha256=score_source_sha256,
    )
    policy = RoutingPolicy(
      version="probability_cascade_v1",
      model_scope=primary["model_scope"],
      reference_dataset=primary["target_dataset"],
      outcome_horizon_days=primary["outcome_horizon_days"],
      route_mode="threshold",
      tau_r=fit.tau_r,
      inclusive=True,
      model_hash_a=primary["stage_a_model_sha256"],
      model_hash_b=primary["stage_b_model_sha256"],
      feature_schema_hash_a=schema_a,
      feature_schema_hash_b=schema_b,
      reference_split=fit.reference_split,
      reference_source_sha256=fit.reference_source_sha256,
      target_utilization=fit.target_utilization,
    )
    candidates.append((fit, policy))

  output_dir = _probability_cascade_output_dir(Path(args.output_dir))
  policies: list[dict[str, Any]] = []
  for fit, policy in candidates:
    policy_path = output_dir / f"routing-{int(fit.target_utilization * 100):02d}pct-{policy.policy_id[-16:]}.yaml"
    save_policy(policy, policy_path)
    policies.append({
      **asdict(fit),
      "policy_id": policy.policy_id,
      "policy_path": str(policy_path),
      "reference_dataset": policy.reference_dataset,
      "model_scope": policy.model_scope,
      "outcome_horizon_days": policy.outcome_horizon_days,
      "reference_source_sha256": policy.reference_source_sha256,
      "model_hash_a": policy.model_hash_a,
      "model_hash_b": policy.model_hash_b,
      "feature_schema_hash_a": policy.feature_schema_hash_a,
      "feature_schema_hash_b": policy.feature_schema_hash_b,
    })
  summary_path = output_dir / "routing_cutoffs.json"
  _write_new_json(summary_path, {
    "status": "reference_cutoffs_fitted_not_calibration_or_test_evidence",
    "split": args.split,
    "score_column": score_column,
    "n_reference_rows": len(raw_scores),
    "n_invalid_or_missing_stage_a": n_invalid,
    "n_valid_stage_a": len(valid_scores),
    "config_sha256": _sha256_file(Path(args.config)),
    "score_table_sha256": score_source_sha256,
    "policies": policies,
  })
  print(f"Reference routing cutoffs: {summary_path}")
  return 0


def run_probability_cascade_replay(args: argparse.Namespace) -> int:
  """Replay a frozen policy on one authorized split without loading outcomes."""
  from ecg_rep.probability_cascade import (
    RoutingPolicy,
    StageAScore,
    StageBScore,
    load_policy,
    replay_scores,
  )

  config = _load_probability_cascade_config(Path(args.config))
  primary = _authorize_probability_cascade_run(
    config,
    args.split,
    "replay",
    allow_locked_test=args.allow_locked_test,
  )
  policy = load_policy(Path(args.policy))
  if (
    policy.model_scope != primary.get("model_scope")
    or policy.reference_dataset != primary.get("target_dataset")
    or policy.outcome_horizon_days != primary.get("outcome_horizon_days")
    or policy.model_hash_a != primary.get("stage_a_model_sha256")
    or policy.model_hash_b != primary.get("stage_b_model_sha256")
    or policy.feature_schema_hash_a != primary.get("stage_a_feature_schema_sha256")
    or policy.feature_schema_hash_b != primary.get("stage_b_feature_schema_sha256")
  ):
    raise ValueError("Policy provenance does not match the authorized config.")
  if policy.route_mode == "threshold":
    if policy.reference_split != primary.get("routing_reference_split"):
      raise ValueError("Policy reference split does not match the configured reference split.")
    reference_hashes = primary.get("score_table_sha256_by_split")
    if reference_hashes is not None:
      if not isinstance(reference_hashes, dict):
        raise ValueError("score_table_sha256_by_split must be a mapping when supplied.")
      expected_reference_hash = reference_hashes.get(policy.reference_split)
      if expected_reference_hash is not None and policy.reference_source_sha256 != expected_reference_hash:
        raise ValueError("Policy reference score-source hash does not match the configured source manifest.")
  score_source_sha256 = _verify_cascade_score_source(primary, Path(args.scores_path), args.split)
  score_a_column = primary["stage_a_score_column"]
  score_b_column = primary["stage_b_score_column"]
  input_hash_column = "input_sha256"
  columns = ["subject_id", "study_id", input_hash_column, score_a_column]
  if args.precomputed_b_reference:
    columns.append(score_b_column)
  frame = _cascade_score_frame(
    Path(args.scores_path), "split", args.split, columns,
  )
  _validate_unique_ecg_keys(frame)
  score_rows = frame.iter_rows(named=True)
  scores_a: list[StageAScore] = []
  scores_b: list[StageBScore | None] = []
  for row in score_rows:
    scores_a.append(StageAScore(
      probability=row[score_a_column],
      model_scope=policy.model_scope,
      dataset_id=policy.reference_dataset,
      outcome_horizon_days=policy.outcome_horizon_days,
      model_hash=policy.model_hash_a,
      feature_schema_hash=policy.feature_schema_hash_a,
      input_sha256=str(row[input_hash_column]),
      subject_id=int(row["subject_id"]),
      study_id=int(row["study_id"]),
    ))
    if args.precomputed_b_reference:
      probability_b = row[score_b_column]
      scores_b.append(None if probability_b is None else StageBScore(
        probability=probability_b,
        model_scope=policy.model_scope,
        dataset_id=policy.reference_dataset,
        outcome_horizon_days=policy.outcome_horizon_days,
        model_hash=policy.model_hash_b,
        feature_schema_hash=policy.feature_schema_hash_b,
        input_sha256=str(row[input_hash_column]),
        subject_id=int(row["subject_id"]),
        study_id=int(row["study_id"]),
      ))
  result = replay_scores(
    scores_a,
    policy,
    scores_b if args.precomputed_b_reference else None,
    allow_precomputed_b_reference=args.precomputed_b_reference,
  )
  output_dir = _probability_cascade_output_dir(Path(args.output_dir))
  assignments_path = output_dir / "assignments.parquet"
  pl.DataFrame([asdict(row) for row in result.rows]).write_parquet(assignments_path)
  assignments_path.chmod(0o600)
  audit_path: Path | None = None
  if result.counterfactual_b_reference is not None:
    audit_path = output_dir / "counterfactual_b_reference.parquet"
    audit_rows = [asdict(row) for row in result.counterfactual_b_reference]
    audit_frame = pl.DataFrame(audit_rows) if audit_rows else pl.DataFrame(schema={
      "policy_id": pl.String,
      "input_sha256": pl.String,
      "stage_b_probability": pl.Float64,
      "model_hash_b": pl.String,
      "feature_schema_hash_b": pl.String,
      "dataset_id": pl.String,
      "subject_id": pl.Int64,
      "study_id": pl.Int64,
    })
    audit_frame.write_parquet(audit_path)
    audit_path.chmod(0o600)
  report_path = output_dir / "replay_manifest.json"
  _write_new_json(report_path, {
    "split": args.split,
    "policy_id": policy.policy_id,
    "reference_dataset": policy.reference_dataset,
    "model_scope": policy.model_scope,
    "outcome_horizon_days": policy.outcome_horizon_days,
    "config_sha256": _sha256_file(Path(args.config)),
    "score_table_sha256": score_source_sha256,
    "n_rows": len(result.rows),
    "n_routed_to_b": sum(row.route_to_b is True for row in result.rows),
    "routed_to_b_fraction": sum(row.route_to_b is True for row in result.rows) / len(result.rows),
    "n_unable_to_route": sum(row.route_status == "unable_to_route" for row in result.rows),
    "precomputed_b_reference_loaded": args.precomputed_b_reference,
    "assignments_path": str(assignments_path),
    "counterfactual_b_reference_path": str(audit_path) if audit_path else None,
    "claims": "raw selective mixture is diagnostic only; not calibrated final risk",
    "resource_interpretation": "routed_to_b_fraction is a routing decision fraction; runtime invocations and compute costs are not measured",
  })
  print(f"Probability-cascade replay: {report_path}")
  return 0


def run_probability_cascade_v2(args: argparse.Namespace) -> int:
  """Run the v2 score-to-results workflow from one versioned config."""
  from ecg_rep.probability_cascade_run import run_probability_cascade

  manifest = run_probability_cascade(Path(args.config))
  print(json.dumps({
    "status": manifest["status"],
    "run_id": manifest["run_id"],
    "assessment": manifest.get("assessment"),
    "outputs": manifest.get("outputs"),
  }, indent=2, sort_keys=True))
  return 0 if manifest["status"] == "complete" else 1


def run_validate_ciis(args: argparse.Namespace) -> int:
  """Run independent benchmark morphology measurement validation study on a reproducible ECG sample."""
  paths = resolve_data_paths(args.mimic_root, args.ecg_root)
  cohort_path = Path(args.cohort_path)
  if not cohort_path.exists():
    raise FileNotFoundError(f"Cohort file not found at {cohort_path}. Run `ecg-rep cohort` first.")
  cohort_df = pl.read_parquet(cohort_path)

  n_sample = min(args.sample_size, len(cohort_df))
  rng = np.random.default_rng(args.seed)
  sample_indices = rng.choice(len(cohort_df), size=n_sample, replace=False)
  sample_indices.sort()
  sample_df = cohort_df[sample_indices]
  print(f"[Validate CIIS] Selected {n_sample} index ECGs for independent morphology measurement validation (seed={args.seed})")

  extracted_rows: list[dict[str, Any]] = []
  reference_rows: list[dict[str, Any]] = []

  for idx, row in enumerate(sample_df.iter_rows(named=True)):
    sid = int(row["study_id"])
    rec_path = paths.mimic_iv_ecg_dir / str(row["path"])
    try:
      signal_array, lead_names, fs = read_ecg_waveform(rec_path)
      # Primary extraction
      meas = extract_ciis_measurements_from_waveform(signal_array, lead_names, fs)
      ext_dict: dict[str, Any] = {"study_id": sid}
      for k in CIIS_FEATURE_NAMES:
        ext_dict[k] = getattr(meas, k)
      extracted_rows.append(ext_dict)

      # Independent benchmark extraction
      bench = extract_independent_benchmark_measurements(signal_array, lead_names, fs)
      ref_dict: dict[str, Any] = {"study_id": sid}
      for k in CIIS_FEATURE_NAMES:
        ref_dict[k] = bench.get(k, 0.0)
      reference_rows.append(ref_dict)
    except Exception as exc:  # noqa: BLE001
      print(f"[Validate CIIS] Warning: record {sid} extraction error: {exc}")

    done = idx + 1
    if done % 25 == 0 or done == n_sample:
      print(f"[Validate CIIS] Processed {done}/{n_sample} validation records", flush=True)

  extracted_df = pl.DataFrame(extracted_rows)
  reference_df = pl.DataFrame(reference_rows)

  report = run_ciis_measurement_validation_study(
    extracted_df=extracted_df,
    reference_df=reference_df,
  )
  report["sample_size"] = n_sample
  report["seed"] = args.seed
  report["n_extracted"] = len(extracted_df)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  val_path = output_dir / "ciis_measurement_validation.json"
  agree_path = output_dir / "ciis_algorithmic_agreement_validation.json"
  rep_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
  val_path.write_text(rep_json, encoding="utf-8")
  agree_path.write_text(rep_json, encoding="utf-8")
  print(f"[Validate CIIS] Wrote independent measurement validation report to {val_path}")

  if "independent_reference_validation" in report:
    feats = report["independent_reference_validation"].get("features", {})
    print("\n--- CIIS Morphology Independent Validation Summary ---")
    for feat_name, stat in feats.items():
      print(
        f"  {feat_name:25s} | MAE={stat['mae']:.3f} | Bias={stat['bias']:+.3f} | "
        f"r={stat['pearson_r']:.3f} | rho={stat['spearman_rho']:.3f} | Outliers={stat['outlier_rate']:.1%}"
      )

  return 0


def run_preflight(args: argparse.Namespace) -> int:
  """Run CIIS morphology preflight validation on standard and enriched difficult cases."""
  from ecg_rep.preflight import run_ciis_preflight_validation

  paths = resolve_data_paths(args.mimic_root, args.ecg_root)
  cohort_path = Path(args.cohort_path)
  cohort_df = pl.read_parquet(cohort_path) if cohort_path.exists() else None

  report = run_ciis_preflight_validation(
    output_dir=args.output_dir,
    cohort_df=cohort_df,
    ecg_data_dir=paths.mimic_iv_ecg_dir if paths.mimic_iv_ecg_dir.exists() else None,
    n_standard=args.n_standard,
    n_difficult=args.n_difficult,
    seed=args.seed,
    enforce_acceptance=not args.no_enforce,
  )
  print(f"[Preflight] Finished CIIS preflight validation. Verdict: {report['status']}")
  return 0 if report["status"] == "PASSED" else 1


def run_external_code15_audit(args: argparse.Namespace) -> int:
  """E0: inventory CODE-15% sources and baseline MIMIC probe artifacts."""
  from ecg_rep.external.code15_inventory import (
    build_code15_source_manifest,
    inventory_mimic_probe_artifacts,
    write_manifest,
  )

  output_dir = Path(args.output_dir)
  if output_dir.resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")

  output_dir.mkdir(parents=True, exist_ok=True)
  probe_baseline = inventory_mimic_probe_artifacts(
    Path(args.probe_dir),
    compute_checksums=not args.skip_checksums,
  )
  probe_path = output_dir / "mimic_probe_baseline.json"
  write_manifest(probe_baseline, probe_path)
  print(
    f"MIMIC probe baseline: {len(probe_baseline['files'])} files hashed, "
    f"{len(probe_baseline['missing_files'])} missing -> {probe_path}"
  )

  source_manifest = build_code15_source_manifest(
    Path(args.data_dir),
    compute_checksums=not args.skip_checksums,
    inspect_keys=not args.skip_hdf5_keys,
    progress_every=1 if not args.skip_checksums else None,
  )
  manifest_path = output_dir / "code15_source_manifest.json"
  write_manifest(source_manifest, manifest_path)

  inventory = source_manifest["inventory"]
  print(f"Source files: {inventory['n_files']} (duplicates: {inventory['duplicate_names']})")
  print(f"Missing partitions: {inventory['missing_partitions']}")
  if source_manifest["exams_schema"] is not None:
    schema = source_manifest["exams_schema"]
    print(f"exams.csv: {schema['n_rows']:,} rows, {schema['n_columns']} columns")
  key_map = source_manifest["hdf5_key_map"]
  if key_map is not None:
    print(
      "HDF5 key map: "
      f"{len(key_map['partitions'])} partitions inspected, "
      f"{len(key_map['errors'])} errors"
    )
    for part_name, part in sorted(key_map["partitions"].items())[:1]:
      print(f"  e.g. {part_name}: {sorted(part['datasets'])}")
  print(f"Wrote source manifest to {manifest_path}")
  return 0


def run_external_code15_prepare(args: argparse.Namespace) -> int:
  """E1: index cohort, endpoint labels, feasibility census, sealed split."""
  from ecg_rep.external.code15_labels import (
    HORIZON_DAYS,
    attach_horizon_labels,
    build_index_cohort,
    build_patient_split,
    feasibility_census,
    go_no_go_decision,
    resolve_timey_unit,
    timey_unit_evidence,
  )

  output_dir = Path(args.output_dir)
  if output_dir.resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")
  resolve_timey_unit(args.timey_unit)  # fail fast on unapproved units

  exams_path = Path(args.data_dir).expanduser() / "exams.csv"
  if not exams_path.exists():
    raise FileNotFoundError(
      f"exams.csv not found under '{args.data_dir}'. Run `ecg-rep external-code15 audit` first."
    )
  exams = pl.read_csv(exams_path, infer_schema_length=10_000, try_parse_dates=False)

  result = build_index_cohort(exams)
  labeled = attach_horizon_labels(result.cohort)

  output_dir.mkdir(parents=True, exist_ok=True)
  cohort_path = output_dir / "code15_index_cohort.parquet"
  labeled.write_parquet(cohort_path)

  split = build_patient_split(labeled, seed=args.seed)
  split_path = output_dir / "code15_split_assignments.csv"
  split.write_csv(split_path)

  census = feasibility_census(labeled)
  evidence = timey_unit_evidence(result.cohort)
  decision = go_no_go_decision(labeled, split)
  summary = {
    "outcome_status": (
      "resolved_no_go_30d_event_screen"
      if decision["decision"] == "NO-GO-30d"
      else "resolved_go_30d"
    ),
    "attrition": result.attrition,
    "issues": result.issues,
    "timey_unit_evidence": evidence,
    "feasibility_census": census,
    "go_no_go": decision,
    "split_seed": args.seed,
    "horizons_days": HORIZON_DAYS,
  }
  census_path = output_dir / "code15_feasibility_census.json"
  census_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  print(f"Index cohort: {result.attrition['n_index_cohort_patients']:,} patients -> {cohort_path}")
  for issue in result.issues:
    print(f"[issue] {issue}")
  print(
    "Split sizes:",
    split.group_by("split").len().sort("split").to_dicts(),
  )
  for name, hz in census["horizons"].items():
    print(
      f"{name}: events={hz['n_events']:,} nonevents={hz['n_nonevents_complete_followup']:,} "
      f"unknown={hz['n_unknown']:,}"
    )
  print(f"Go/No-Go ({decision['rule']}): {decision['decision']}")
  print(f"Per-split 30d events: {decision['per_split_events']}")
  print(f"Wrote feasibility census to {census_path} (cohort artifacts stay gitignored)")
  return 0


def run_external_code15_extract(args: argparse.Namespace) -> int:
  """E2: extract frozen A/B representations with per-partition checkpoints."""
  import time

  from ecg_rep.external.code15_waveforms import (
    build_exam_id_row_index,
    extract_exams_representations,
    features_sha256,
    load_partition_checkpoint,
    save_partition_checkpoint,
  )
  from ecg_rep.scoring.dbeta import DbetaAdapter, DbetaConfig, load_dbeta_model

  output_dir = Path(args.output_dir) if args.output_dir is not None else (
    Path("artifacts/external/code15pct/features_preflight")
    if args.limit is not None
    else Path("artifacts/external/code15pct/features")
  )
  if output_dir.resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")
  if not Path(args.cohort_path).exists():
    raise FileNotFoundError(
      f"Index cohort not found at '{args.cohort_path}'. Run `ecg-rep external-code15 prepare` first."
    )
  cohort = pl.read_parquet(args.cohort_path)
  if args.limit is not None:
    cohort = cohort.head(args.limit)

  device = resolve_torch_device(args.device)
  adapter: DbetaAdapter | None = None
  if device != "skip":
    model = load_dbeta_model(
      revision="20ff3ccce1759d7d629171e15befafa9a424d2ca",
      device=device,
    )
    adapter = DbetaAdapter(config=DbetaConfig(device=device, batch_size=args.batch_size), model=model)

  data_dir = Path(args.data_dir).expanduser()
  partition_indices = (
    args.partitions
    if args.partitions is not None
    else list(range(18))
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  all_exam_ids: list[int] = []
  all_features_a: list[Any] = []
  all_features_b: list[Any] = []
  failures: dict[str, Any] = {}
  started = time.time()

  for part_idx in partition_indices:
    partition_path = data_dir / f"exams_part{part_idx}.hdf5"
    if not partition_path.exists():
      failures[f"part{part_idx}"] = "missing_partition"
      continue
    exam_ids_in_part = cohort.filter(
      pl.col("trace_file") == f"exams_part{part_idx}.hdf5",
    )["exam_id"].to_list()
    if not exam_ids_in_part:
      continue
    checkpoint_path = output_dir / f"checkpoint_part{part_idx}.npz"
    done_ids, done_a, done_b = load_partition_checkpoint(
      checkpoint_path, n_a=15, n_b=768,
    )
    done_set = set(done_ids)
    todo = [e for e in exam_ids_in_part if e not in done_set]
    print(f"[part{part_idx}] {len(done_ids)} checkpointed, {len(todo)} to extract", flush=True)

    records = extract_exams_representations(
      partition_path,
      todo,
      dbeta_adapter=adapter,
    )
    new_ids = list(done_ids)
    new_a = [done_a] if len(done_a) else []
    new_b = [done_b] if len(done_b) else []
    part_failures = 0
    for rec in records:
      if rec.error is not None or rec.features_a is None or rec.features_b is None:
        part_failures += 1
        failures[f"part{part_idx}/exam{rec.exam_id}"] = rec.error
        continue
      new_ids.append(rec.exam_id)
      new_a.append(rec.features_a)
      new_b.append(rec.features_b)
      all_exam_ids.append(rec.exam_id)
      all_features_a.append(rec.features_a)
      all_features_b.append(rec.features_b)

    if args.checkpoint_interval > 0:
      save_partition_checkpoint(
        checkpoint_path,
        new_ids,
        np.asarray(new_a, dtype=np.float64),
        np.asarray(new_b, dtype=np.float64),
      )
    if part_failures:
      print(f"[part{part_idx}] {part_failures} exams failed QC/extraction", flush=True)

  # Row alignment invariant: features are stored in ascending exam_id order,
  # identical to extracted_meta.parquet. Extraction order (partition by
  # partition, cohort row order) is NOT exam_id-monotonic, so both the
  # feature arrays and the meta frame must be explicitly re-sorted by
  # exam_id before saving.
  if len(all_exam_ids) == 0:
    raise RuntimeError("No exams were extracted; refusing to write empty artifacts.")
  ids_arr = np.asarray(all_exam_ids, dtype=np.int64)
  order = np.argsort(ids_arr, kind="stable")
  features_a = np.asarray(all_features_a, dtype=np.float64)[order]
  features_b = np.asarray(all_features_b, dtype=np.float64)[order]
  np.save(output_dir / "ciis_features.npy", features_a)
  np.save(output_dir / "dbeta_embeddings.npy", features_b)
  meta = cohort.filter(pl.col("exam_id").is_in(all_exam_ids)).sort("exam_id")
  if meta["exam_id"].to_numpy().tobytes() != ids_arr[order].tobytes():
    raise RuntimeError("Feature/meta exam_id alignment failed; refusing to save.")
  meta.write_parquet(output_dir / "extracted_meta.parquet")

  manifest = {
    "extraction_schema_version": "code15_extract_v1",
    "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "dbeta_revision": "20ff3ccce1759d7d629171e15befafa9a424d2ca",
    "preprocessing_mode": "trim_pad_then_400_to_500hz_locked_pipeline",
    "seed": None,
    "training_cohort": "none (frozen replay)",
    "scoring_cohort": "code15_index_cohort",
    "n_extracted": int(len(all_exam_ids)),
    "n_failed": len(failures),
    "failures": failures,
    "features_a_sha256": features_sha256(features_a) if len(features_a) else None,
    "features_b_sha256": features_sha256(features_b) if len(features_b) else None,
    "elapsed_seconds": round(time.time() - started, 1),
    "preflight_mode": args.limit is not None,
  }
  (output_dir / "extraction_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
  )
  print(
    f"Extracted A/B for {len(all_exam_ids):,} exams "
    f"({len(failures)} failures) in {manifest['elapsed_seconds']}s -> {output_dir}"
  )
  return 0


def run_external_code15_transport(args: argparse.Namespace) -> int:
  """E3: evaluate the unchanged frozen MIMIC probes on CODE-15% features."""
  import json as _json

  from ecg_rep.external.code15_transport import evaluate_transport

  if Path(args.output_dir).resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")

  result = evaluate_transport(
    Path(args.features_dir),
    Path(args.cohort_path),
    Path(args.model_dir),
    baseline_manifest_path=Path(args.baseline_manifest)
    if Path(args.baseline_manifest).exists()
    else None,
    split_path=Path(args.split_path) if Path(args.split_path).exists() else None,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "transport_results.json"
  out_path.write_text(_json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  for track in result["tracks"]:
    a = track["metrics_a"]
    b = track["metrics_b"]
    print(
      f"[{track['track_name']}] n={track['n']:,} events={track['n_events']:,} "
      f"AUROC A={_fmt(a['auroc'])} B={_fmt(b['auroc'])} "
      f"Brier A={_fmt(a['brier'], 5)} B={_fmt(b['brier'], 5)} "
      f"slope A={_fmt(a['slope'])} B={_fmt(b['slope'])}"
    )
  for name, att in result["attrition"].items():
    print(f"[{name}] {att}")
  ipcw = result.get("ipcw_sensitivity")
  if ipcw is not None:
    delta = ipcw["contrasts"]["B_minus_A"]["ipcw_delta_auroc"]
    print(
      f"[locked_test_365d IPCW] A={_fmt(ipcw['models']['A']['ipcw_auroc'])} "
      f"B={_fmt(ipcw['models']['B']['ipcw_auroc'])} "
      f"B-A={delta['point_estimate']:.4f} "
      f"CI=({_fmt(delta['ci_lower'])}, {_fmt(delta['ci_upper'])})"
    )
    print(f"[locked_test_365d IPCW population] {ipcw['test_population']}")
  print(f"Aggregate results -> {out_path}")
  return 0


def run_external_code15_alignment_audit(args: argparse.Namespace) -> int:
  """Independently compare saved CODE features and labels by exam ID."""
  import json as _json

  from ecg_rep.external.code15_alignment_audit import audit_code15_feature_alignment

  result = audit_code15_feature_alignment(args.features_dir, args.cohort_path)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "alignment_audit.json"
  out_path.write_text(
    _json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  print(
    f"[alignment] status={result['status']} rows={result['n_checkpoint_rows']:,} "
    f"A_exact={result['n_ciis_rows_exactly_matched']:,} "
    f"B_exact={result['n_dbeta_rows_exactly_matched']:,}"
  )
  print(f"Aggregate verification -> {out_path}")
  return 0 if result["status"] == "passed" else 1


def run_external_code15_waveform_audit(args: argparse.Namespace) -> int:
  """Scan source waveforms and compare frozen E3 performance by waveform."""
  import json as _json

  from ecg_rep.external.code15_waveform_audit import audit_code15_waveforms

  result = audit_code15_waveforms(
    data_dir=args.data_dir,
    features_dir=args.features_dir,
    cohort_path=args.cohort_path,
    split_path=args.split_path,
    model_dir=args.model_dir,
    baseline_manifest_path=args.baseline_manifest,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
    batch_size=args.batch_size,
    progress=lambda message: print(f"[waveform audit] {message}", flush=True),
  )
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "waveform_audit.json"
  out_path.write_text(_json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  scan = result["scan_counts"]
  overall = result["locked_test_complete_case_overall"]
  interaction = result["locked_test_duration_padding_interaction"]
  estimate = interaction["interaction_estimate"]
  ci_lower = interaction["ci_lower"]
  ci_upper = interaction["ci_upper"]
  interaction_summary = "suppressed" if estimate is None else (
    f"{estimate:+.4f} (95% CI {ci_lower:+.4f}, {ci_upper:+.4f})"
  )
  print(
    f"[waveform audit] status={result['status']} "
    f"cohort={scan['n_index_cohort']:,} extracted={scan['n_extracted_feature_rows']:,} "
    f"failures={scan['n_manifest_extraction_failures']:,}"
  )
  print(
    f"[locked-test 365d complete case] n={overall['n_locked_test_complete_case']:,} "
    f"events={overall['n_events']:,} A={overall['auroc_a']:.4f} "
    f"B={overall['auroc_b']:.4f} B-A={overall['delta_b_minus_a']:+.4f}"
  )
  print(f"[waveform interaction: 2934+581x2 vs 4096 unpadded] {interaction_summary}")
  print(f"Aggregate waveform audit -> {out_path}")
  return 0 if result["status"] == "passed" else 1


def _fmt(value: float | None, digits: int = 4) -> str:
  """Format a possibly-suppressed metric for console output."""
  return "n/a" if value is None else f"{value:.{digits}f}"


def run_external_code15_probes(args: argparse.Namespace) -> int:
  """E4: fit and evaluate CODE-15% linear probes (representation replication)."""
  import json as _json

  import numpy as np
  import polars as pl

  from ecg_rep.external.code15_probes import run_code15_probe_analysis

  if Path(args.output_dir).resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")

  features_dir = Path(args.features_dir)
  features_a = np.load(features_dir / "ciis_features.npy")
  features_b = np.load(features_dir / "dbeta_embeddings.npy")
  meta = pl.read_parquet(features_dir / "extracted_meta.parquet").sort("exam_id")
  if len(meta) != len(features_a) or len(meta) != len(features_b):
    raise ValueError("Feature/meta row mismatch; rerun extraction.")

  split = pl.read_csv(Path(args.split_path))
  split_map = dict(zip(split["patient_id"].to_list(), split["split"].to_list(), strict=True))
  meta = meta.with_columns(
    pl.col("patient_id").replace_strict(split_map, default=None).alias("split"),
  )
  n_unsplit = int(meta["split"].null_count())
  if n_unsplit:
    raise ValueError(f"{n_unsplit} extracted exams have no split assignment; refusing to proceed.")

  report = run_code15_probe_analysis(
    features_a, features_b, meta, horizon=args.horizon,
    seed=args.seed, n_bootstraps=args.n_bootstraps,
  )

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / f"code15_probe_report_{args.horizon}.json"
  payload = {
    "schema_version": "code15_probes_v2",
    "horizon": report.horizon,
    "n_dev": report.n_dev,
    "n_val": report.n_val,
    "n_test": report.n_test,
    "n_events_test": report.n_events_test,
    "attrition": report.attrition,
    "tuning": report.tuning,
    "pca15_info": report.pca15_info,
    "test_bundles": report.test_bundles,
    "contrasts": report.contrasts,
    "fixed_capacity": report.fixed_capacity,
    "ipcw_sensitivity": report.ipcw_sensitivity,
  }
  out_path.write_text(_json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  pl.DataFrame(report.fixed_capacity["metric_rows"]).write_csv(
    output_dir / f"code15_fixed_capacity_metrics_{args.horizon}.csv",
  )
  pl.DataFrame(report.fixed_capacity["contrast_rows"]).write_csv(
    output_dir / f"code15_fixed_capacity_contrasts_{args.horizon}.csv",
  )

  for name, bundle in report.test_bundles.items():
    print(
      f"[{name}] auroc={_fmt(bundle['auroc'])} auprc={_fmt(bundle['auprc'])} "
      f"brier={_fmt(bundle['brier'], 5)} slope={_fmt(bundle['calibration_slope'])} "
      f"best_c={report.tuning[name]['best_c']} "
      f"boundary={report.tuning[name]['boundary_selection_review']['selected_boundary']}"
    )
  for cname, cval in report.contrasts.items():
    print(
      f"[{cname}] point={cval['point_estimate']:.4f} "
      f"CI=({cval['ci_lower']:.4f}, {cval['ci_upper']:.4f})"
    )
  for name, metrics in report.ipcw_sensitivity["models"].items():
    print(
      f"[IPCW {name}] auroc={_fmt(metrics['ipcw_auroc'])} "
      f"complete_case={_fmt(metrics['complete_case_auroc'])} "
      f"delta={_fmt(metrics['difference_ipcw_minus_complete_case'])}"
    )
  for name, contrast in report.ipcw_sensitivity["contrasts"].items():
    delta = contrast["ipcw_delta_auroc"]
    print(
      f"[IPCW {name}] delta={delta['point_estimate']:.4f} "
      f"CI=({_fmt(delta['ci_lower'])}, {_fmt(delta['ci_upper'])})"
    )
  print(f"IPCW test population: {report.ipcw_sensitivity['test_population']}")
  for row in report.fixed_capacity["metric_rows"]:
    if row["capacity_fraction"] == 0.05:
      print(
        f"[fixed-capacity {row['model']}@5%] captured={row['n_events_captured']} "
        f"sensitivity={_fmt(row['sensitivity'])} ppv={_fmt(row['ppv'])} "
        f"nne={_fmt(row['number_needed_to_evaluate'])}"
      )
  for row in report.fixed_capacity["contrast_rows"]:
    if row["capacity_fraction"] == 0.05 and row["contrast"] == "B_minus_A_under_C_demo":
      print(
        f"[fixed-capacity B-A@5%] delta={row['delta_events_captured']} "
        f"CI=({row['delta_events_ci_low']}, {row['delta_events_ci_high']})"
      )
  print(f"Report -> {out_path}")
  return 0


def run_external_code15_heterogeneity(args: argparse.Namespace) -> int:
  """E5: heterogeneity/discordance/subgroups with dev+val-prespecified cutpoints."""
  import json as _json

  import numpy as np
  import polars as pl

  from ecg_rep.external.code15_heterogeneity import run_heterogeneity_analysis

  if Path(args.output_dir).resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")

  features_dir = Path(args.features_dir)
  features_a = np.load(features_dir / "ciis_features.npy")
  features_b = np.load(features_dir / "dbeta_embeddings.npy")
  meta = pl.read_parquet(features_dir / "extracted_meta.parquet").sort("exam_id")
  if len(meta) != len(features_a) or len(meta) != len(features_b):
    raise ValueError("Feature/meta row mismatch; rerun extraction.")
  split = pl.read_csv(Path(args.split_path))
  split_map = dict(zip(split["patient_id"].to_list(), split["split"].to_list(), strict=True))
  meta = meta.with_columns(
    pl.col("patient_id").replace_strict(split_map, default=None).alias("split"),
  )
  n_unsplit = int(meta["split"].null_count())
  if n_unsplit:
    raise ValueError(f"{n_unsplit} extracted exams have no split assignment; refusing to proceed.")

  report = run_heterogeneity_analysis(
    features_a, features_b, meta,
    model_dir=Path(args.model_dir),
    baseline_manifest_path=Path(args.baseline_manifest)
    if Path(args.baseline_manifest).exists()
    else None,
    horizon=args.horizon,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / f"heterogeneity_report_{args.horizon}.json"
  from dataclasses import asdict

  payload = {"schema_version": "code15_heterogeneity_v2", **asdict(report)}
  out_path.write_text(
    _json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
  )

  print(f"Attrition: {report.attrition}")
  print(f"Cutpoint A: {report.cutpoints_a['high_cutpoint']:.4f}; B: {report.cutpoints_b['high_cutpoint']:.4f}")
  for cell in report.discordance_cells:
    print(f"[{cell['cell']}] n={cell['n']:,} events={cell['n_events']}")
  for row in report.age_subgroups + report.sex_subgroups:
    mb = row["metrics_b"]
    print(f"[{row['group']}] n={row['n']:,} events={row['n_events']} auroc_B={_fmt(mb and mb['auroc'])}")
  for row in report.interaction_contrasts:
    print(
      f"[interaction {row['contrast']}] estimate={_fmt(row['interaction_estimate'])} "
      f"CI=({_fmt(row['ci_lower'])}, {_fmt(row['ci_upper'])})"
    )
  print(f"Report -> {out_path}")
  return 0


def run_external_code15_deployment(args: argparse.Namespace) -> int:
  """E6: cascade policy operating curves, capture metrics and compute benchmark."""
  import json as _json

  import numpy as np
  import polars as pl

  from ecg_rep.external.code15_deployment import (
    ECDFRankCalibrator,
    bootstrap_primary_operating_point,
    benchmark_extraction_throughput,
    cascade_policy_comparison,
  )
  from ecg_rep.external.code15_transport import load_frozen_probes, transport_scores

  if Path(args.output_dir).resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")

  features_dir = Path(args.features_dir)
  features_a = np.load(features_dir / "ciis_features.npy")
  features_b = np.load(features_dir / "dbeta_embeddings.npy")
  meta = pl.read_parquet(features_dir / "extracted_meta.parquet").sort("exam_id")
  if len(meta) != len(features_a) or len(meta) != len(features_b):
    raise ValueError("Feature/meta row mismatch; rerun extraction.")
  split = pl.read_csv(Path(args.split_path))
  split_map = dict(zip(split["patient_id"].to_list(), split["split"].to_list(), strict=True))
  meta = meta.with_columns(
    pl.col("patient_id").replace_strict(split_map, default=None).alias("split"),
  )
  n_unsplit = int(meta["split"].null_count())
  if n_unsplit:
    raise ValueError(f"{n_unsplit} extracted exams have no split assignment; refusing to proceed.")

  probes = load_frozen_probes(
    Path(args.model_dir),
    Path(args.baseline_manifest) if Path(args.baseline_manifest).exists() else None,
  )
  from ecg_rep.external.code15_labels import horizon_label_to_int

  meta = horizon_label_to_int(meta, args.horizon)
  p_a, p_b = transport_scores(probes, features_a, features_b)

  label_col = f"y_{args.horizon}"
  labels = meta[label_col].to_list()
  splits = meta["split"].to_list()
  keep = np.asarray([v is not None for v in labels], dtype=bool)
  y = np.asarray([v for v in labels if v is not None], dtype=np.int64)
  pa, pb = p_a[keep], p_b[keep]
  split_kept = [s for s, k in zip(splits, keep, strict=True) if k]
  dev_val_mask = np.asarray([s in ("dev", "val") for s in split_kept], dtype=bool)
  test_mask = np.asarray([s == "test" for s in split_kept], dtype=bool)
  rank_calibrator_a = ECDFRankCalibrator().fit(pa[dev_val_mask])
  rank_calibrator_b = ECDFRankCalibrator().fit(pb[dev_val_mask])

  rows = cascade_policy_comparison(
    y[test_mask], pa[test_mask], pb[test_mask],
    rank_calibrator_a=rank_calibrator_a,
    rank_calibrator_b=rank_calibrator_b,
  )
  from ecg_rep.external.code15_deployment import random_routing_seed_stability

  random_seed_stability = random_routing_seed_stability(
    y[test_mask],
    rank_calibrator_a.transform(pa[test_mask]),
    rank_calibrator_b.transform(pb[test_mask]),
    n_seeds=args.random_routing_seeds,
    base_seed=args.seed,
  )
  operating_point = bootstrap_primary_operating_point(
    y[test_mask], pa[test_mask], pb[test_mask],
    rank_calibrator_a=rank_calibrator_a,
    rank_calibrator_b=rank_calibrator_b,
    n_bootstraps=args.n_bootstraps,
    seed=args.seed,
  )
  benchmark = benchmark_extraction_throughput()

  extraction_log = features_dir.parent / "full_extraction.log"
  report = {
    "schema_version": "code15_deployment_v3",
    "horizon": args.horizon,
    "n_test": int(test_mask.sum()),
    "n_events_test": int(y[test_mask].sum()),
    "rank_calibration": {
      "fit_split": "dev+val",
      "n_reference_scores": int(dev_val_mask.sum()),
      "bootstrap_policy": "frozen calibration reference across test resamples",
    },
    "operating_rows": rows,
    "random_routing_seed_stability": random_seed_stability,
    "bootstrap_primary_operating_point": operating_point,
    "compute_benchmark": benchmark,
    "extraction_log_available": extraction_log.exists(),
    "enrichment_note": "Captured deaths are event enrichment, not proven "
    "prevention or treatment benefit; no clinical-safety or cost-effectiveness claims.",
  }
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / f"deployment_report_{args.horizon}.json"
  out_path.write_text(_json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  primary = [r for r in rows if r["capacity_fraction"] == 0.05]
  for r in primary:
    identity_recovery = r["recovery_fraction"]
    print(
      f"[{r['policy']}@u={r['utilization']}] captured={r['events_captured']} "
      f"shared_fullB_events={r['events_also_captured_by_full_b']} "
      f"identity_recovery={identity_recovery} "
      f"relative_capture={r['relative_capture_ratio']} "
      f"fullB_events_missed={r['full_b_events_missed']} B_evals={r['n_b_evaluations']}"
    )
  print(f"Report -> {out_path}")
  return 0


def run_external_code15_probe_ipcw_fit(args: argparse.Namespace) -> int:
  """Audit02 P1-3: IPCW-weighted probe fitting sensitivity."""
  import json as _json

  import numpy as np
  import polars as pl

  from ecg_rep.external.code15_ipcw_probes import run_code15_ipcw_fit_sensitivity

  if Path(args.output_dir).resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")

  features_dir = Path(args.features_dir)
  features_a = np.load(features_dir / "ciis_features.npy")
  features_b = np.load(features_dir / "dbeta_embeddings.npy")
  meta = pl.read_parquet(features_dir / "extracted_meta.parquet").sort("exam_id")
  if len(meta) != len(features_a) or len(meta) != len(features_b):
    raise ValueError("Feature/meta row mismatch; rerun extraction.")

  split = pl.read_csv(Path(args.split_path))
  split_map = dict(zip(split["patient_id"].to_list(), split["split"].to_list(), strict=True))
  meta = meta.with_columns(
    pl.col("patient_id").replace_strict(split_map, default=None).alias("split"),
  )
  n_unsplit = int(meta["split"].null_count())
  if n_unsplit:
    raise ValueError(f"{n_unsplit} extracted exams have no split assignment; refusing to proceed.")

  report = run_code15_ipcw_fit_sensitivity(
    features_a, features_b, meta, horizon=args.horizon,
    seed=args.seed, n_bootstraps=args.n_bootstraps,
  )
  report["schema_version"] = "code15_ipcw_fit_sensitivity_v1"

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / f"code15_ipcw_fit_sensitivity_{args.horizon}.json"
  out_path.write_text(_json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  print(f"IPCW-fit sensitivity -> {out_path}")
  sens = report["ipcw_sensitivity"]
  for name, metrics in sens["models"].items():
    print(
      f"  {name}: ipcw={metrics['ipcw_auroc']:.4f} complete_case={metrics['complete_case_auroc']:.4f}",
    )
  for cname, contrast in sens["contrasts"].items():
    delta = contrast["ipcw_delta_auroc"]
    print(
      f"  {cname}: ipcw delta {delta['point_estimate']:+.4f} "
      f"({delta['ci_lower']:+.4f}, {delta['ci_upper']:+.4f})",
    )
  return 0


def run_external_code15_failure_audit(args: argparse.Namespace) -> int:
  """Audit02: characterize extraction failures by age, sex, and mortality."""
  import json as _json

  import polars as pl

  from ecg_rep.external.code15_failure_audit import summarize_failure_demographics

  if Path(args.output_dir).resolve().is_relative_to(Path("artifacts/probe").resolve()):
    raise ValueError("External outputs must never be written into artifacts/probe.")

  manifest = _json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
  failures = manifest.get("failures")
  if not isinstance(failures, dict) or not failures:
    raise ValueError("Extraction manifest contains no failure records to audit.")

  cohort = pl.read_parquet(Path(args.cohort_path))
  report = summarize_failure_demographics(failures, cohort)
  report["manifest_path"] = str(args.manifest_path)
  report["cohort_path"] = str(args.cohort_path)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / "failure_audit.json"
  out_path.write_text(_json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  print(f"Failure audit -> {out_path}")
  print(f"  failures: {report['n_failure_records']} (rate {report['failure_rate']:.4%})")
  for category, summary in report["by_category"].items():
    print(
      f"  {category}: n={summary['n']} age_mean={summary['age_mean']} "
      f"pct_female={summary['pct_female']} deaths_365d={summary['n_deaths_365d']}",
    )
  print(
    f"  all_failures: deaths_365d={report['all_failures']['n_deaths_365d']} "
    f"rate={report['all_failures']['mortality_rate_365d']}; "
    f"cohort_reference rate={report['cohort_reference']['mortality_rate_365d']}",
  )
  return 0


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entrypoint."""
  parser = build_parser()
  args = parser.parse_args(list(argv) if argv is not None else None)
  if args.command == "cohort":
    return run_cohort(args)
  if args.command == "manifest":
    return run_manifest(args)
  if args.command == "validate-ciis":
    return run_validate_ciis(args)
  if args.command == "preflight":
    return run_preflight(args)
  if args.command == "representations":
    return run_representations(args)
  if args.command == "context":
    return run_context(args)
  if args.command == "evaluate":
    return run_evaluate(args)
  if args.command == "analyze":
    return run_analyze(args)
  if args.command == "dump-probe-scores":
    return run_dump_probe_scores(args)
  if args.command == "probability-cascade-fit-routing":
    return run_probability_cascade_fit_routing(args)
  if args.command == "probability-cascade-replay":
    return run_probability_cascade_replay(args)
  if args.command == "probability-cascade-run":
    return run_probability_cascade_v2(args)
  if args.command == "clinical-setting":
    return run_clinical_setting(args)
  if args.command == "clinical-phenotypes":
    return run_clinical_phenotypes(args)
  if args.command == "clinical-acute-context":
    return run_clinical_acute_context(args)
  if args.command == "ami-feasibility":
    return run_ami_feasibility_command(args)
  if args.command == "ami-evaluate":
    from ecg_rep.ami_outcome import refuse_ami_model_evaluation
    refuse_ami_model_evaluation()
  if args.command == "ami-administrative":
    return run_ami_administrative_command(args)
  if args.command == "clinical-mortality-sensitivity":
    return run_clinical_mortality_sensitivity(args)
  if args.command == "clinical-next-step":
    return run_clinical_next_step(args)
  if args.command == "clinical-next-step-feasibility":
    return run_clinical_next_step_feasibility(args)
  if args.command == "clinical-documented-policy-audit":
    return run_clinical_documented_policy_audit(args)
  if args.command == "clinical-documented-policy":
    return run_clinical_documented_policy(args)
  if args.command == "clinical-cacute":
    return run_clinical_cacute(args)
  if args.command == "clinical-subgroups":
    return run_clinical_subgroups(args)
  if args.command == "clinical-dca":
    return run_clinical_dca(args)
  if args.command == "manuscript-reporting":
    return run_manuscript_reporting_command(args)
  if args.command == "paired-valid-audit":
    return run_paired_valid_audit_command(args)
  if args.command == "bootstrap-estimand-sensitivity":
    return run_bootstrap_estimand_sensitivity_command(args)
  if args.command == "random-control":
    return run_random_control_command(args)
  if args.command == "pretraining-label-efficiency":
    return run_pretraining_label_efficiency_command(args)
  if args.command == "pretraining-ami-control":
    return run_pretraining_ami_control_command(args)
  if args.command == "cmax-feasibility":
    return run_cmax_feasibility_command(args)
  if args.command == "cmax-evaluate":
    return run_cmax_evaluate_command(args)
  if args.command == "refit-uncertainty":
    return run_refit_uncertainty_command(args)
  if args.command == "subgroup-multiplicity":
    return run_subgroup_multiplicity_command(args)
  if args.command == "ascertainment-audit":
    return run_ascertainment_audit_command(args)
  if args.command == "history-sensitivity":
    return run_history_sensitivity_command(args)
  if args.command == "compute-scaling":
    return run_compute_scaling_command(args)
  if args.command == "ptbxl-measurement-validity":
    return run_ptbxl_measurement_validity_command(args)
  if args.command == "manuscript-ledger":
    return run_manuscript_ledger_command(args)
  if args.command == "manuscript-figures":
    return run_manuscript_figures_command(args)
  if args.command == "submission-figures":
    return run_submission_figures_command(args)
  if args.command == "compute-benchmark":
    return run_compute_benchmark_command(args)
  if args.command == "machine-baseline":
    return run_machine_baseline_command(args)
  if args.command == "cart-report-feasibility":
    return run_cart_report_feasibility_command(args)
  if args.command == "cart-report-baseline":
    return run_cart_report_evaluation_command(args)
  if args.command == "cart-addon-contrast":
    return run_cart_addon_contrast_command(args)
  if args.command == "code15-recalibrate":
    return run_code15_recalibration_command(args)
  if args.command == "external-code15":
    if args.external_command == "audit":
      return run_external_code15_audit(args)
    if args.external_command == "prepare":
      return run_external_code15_prepare(args)
    if args.external_command == "extract":
      return run_external_code15_extract(args)
    if args.external_command == "transport":
      return run_external_code15_transport(args)
    if args.external_command == "verify-alignment":
      return run_external_code15_alignment_audit(args)
    if args.external_command == "audit-waveforms":
      return run_external_code15_waveform_audit(args)
    if args.external_command == "probes":
      return run_external_code15_probes(args)
    if args.external_command == "heterogeneity":
      return run_external_code15_heterogeneity(args)
    if args.external_command == "deployment":
      return run_external_code15_deployment(args)
    if args.external_command == "failure-audit":
      return run_external_code15_failure_audit(args)
    if args.external_command == "probe-ipcw-fit":
      return run_external_code15_probe_ipcw_fit(args)
    parser.error(f"Unknown external-code15 command: {args.external_command}")
  parser.error(f"Unknown command: {args.command}")
  return 2


if __name__ == "__main__":
  sys.exit(main())
