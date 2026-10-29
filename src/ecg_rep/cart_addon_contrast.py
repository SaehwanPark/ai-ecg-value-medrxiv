"""LM1 addendum: paired cart add-on contrast for B on top of C0+A''.

Round 09 audit ([`docs/audit_20261010/round09.md`](../../../docs/audit_20261010/round09.md),
section 4) identifies the missing informative comparison

    Δ_cart add-on = AUROC(C0 + A'' + B) − AUROC(C0 + A'')

with appropriate paired uncertainty. The committed ledger
(`artifacts/limitation_mitigation/lm1/cart_report_models.json`) already holds both point
metrics, but no paired contrast between them, and the round 09 response explicitly
declined to derive an increment by subtracting stored point metrics.

This module refits the three context-adjusted probes on the identical A''-valid
paired-valid surface under the frozen `cart_report_comparator_v1` vocabulary and reports
the new paired contrast `c_a2_b_minus_c_a2` for AUROC and AUPRC. Every result is gated:

1. every fitted model must produce exactly the same ordered test identifiers and labels;
2. the locked-test population must match the reference ledger by counts AND by an
   order-independent SHA-256 over sorted ``(subject_id, study_id)`` pairs
   (`test_population_sha256`, persisted by ``cart-report-baseline``), which must also
   match the committed nonidentifying aggregate manifest;
3. reproduced point metrics must match both the committed ledger and the committed
   aggregate manifest (determinism gate);
4. the adjacent ledger contrasts (`c_b_minus_c_a2`, `c_a2_b_minus_c_b`) must reproduce
   their committed paired-bootstrap values — always at the historical 1,000 resamples
   (seed 42), independent of the ``--n-bootstraps`` requested for the new contrast;
5. only then is the requested contrast computed with ``--n-bootstraps`` resamples.

Audit 20261015 round 01 findings 1–4 are addressed by gates 1–4, the separated
bootstrap-count semantics, git-commit provenance in the report, and the committed
aggregate manifest `docs/limitation_mitigation/lm1_reference_aggregates.json`.

Analysis status: ``post hoc extension``. The primary B−A result and all committed LM1
artifacts are unchanged; no increment is ever derived from stored point metrics alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
import numpy.typing as npt
import polars as pl

from ecg_rep.cart_report_baseline import (
  REPORT_COLUMNS,
  ReportFeatureBuilder,
  build_cart_report_frame,
  config_sha256,
  freeze_vocabulary,
  load_freeze,
  population_hash,
  vocabulary_hash,
)
from ecg_rep.evaluate import C0_FEATURE_COLS  # pyright: ignore[reportPrivateUsage]
from ecg_rep.machine_baseline import MACHINE_RAW_COLUMNS, machine_feature_columns
from ecg_rep.probe import bootstrap_paired_difference, evaluate_predictions, metric_bundle_to_dict

# Determinism gates: reproduced point metrics must match the committed ledger and the
# committed aggregate manifest to these absolute tolerances before any contrast is
# reported. best_c and feature counts must match exactly.
METRIC_ABS_TOL = 1e-9
BOOTSTRAP_ABS_TOL = 1e-9
# Historical paired-bootstrap convention of the committed LM1 ledger contrasts. Ledger
# verification always uses this count; `--n-bootstraps` applies only to the new
# add-on contrast (audit 20261015 round 01, finding 3).
LEDGER_BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
ADDON_CONTRAST_NAME = "c_a2_b_minus_c_a2"
# Ledger contrasts recomputed here purely to verify the refit+bootstrap pipeline.
REFERENCE_CONTRASTS: tuple[tuple[str, str, str], ...] = (
  ("c_b_minus_c_a2", "LM1_C_A2", "LM1_C_B"),
  ("c_a2_b_minus_c_b", "LM1_C_B", "LM1_C_A2_B"),
)


def _close(actual: float, expected: float, tol: float) -> bool:
  return bool(np.isclose(actual, expected, rtol=0.0, atol=tol))


def verify_ledger_metrics(
  model_name: str,
  metrics: dict[str, Any],
  ledger_entry: dict[str, Any],
  tol: float = METRIC_ABS_TOL,
) -> dict[str, Any]:
  """Compare reproduced test metrics and tuning against a committed ledger entry.

  Returns a verification record; raises ``ValueError`` on any mismatch so a silently
  different refit can never feed the reported contrast.
  """
  checks: dict[str, bool] = {}
  for key in ("auroc", "auprc", "log_loss", "brier", "calibration_intercept", "calibration_slope"):
    checks[key] = _close(
      float(metrics["test_metrics"][key]), float(ledger_entry["test_metrics"][key]), tol
    )
  checks["best_c"] = metrics["best_c"] == ledger_entry["best_c"]
  checks["n_features"] = metrics["n_features"] == ledger_entry["n_features"]
  failed = sorted(k for k, ok in checks.items() if not ok)
  if failed:
    raise ValueError(
      f"Refit determinism gate failed for {model_name} on {failed}: "
      f"reproduced {json.dumps(metrics, sort_keys=True)} does not match committed ledger "
      f"entry {json.dumps(ledger_entry, sort_keys=True)}"
    )
  return {"model": model_name, "checks": checks, "passed": True}


def verify_ledger_contrast(
  contrast_name: str,
  recomputed: dict[str, Any],
  ledger_contrast: dict[str, Any],
  tol: float = BOOTSTRAP_ABS_TOL,
) -> dict[str, Any]:
  """Compare a recomputed paired-bootstrap contrast against its committed ledger row."""
  checks: dict[str, bool] = {}
  for metric in ("auroc", "auprc"):
    for key in ("point_estimate", "ci_lower", "ci_upper"):
      checks[f"{metric}.{key}"] = _close(
        float(recomputed[metric][key]), float(ledger_contrast[metric][key]), tol
      )
  failed = sorted(k for k, ok in checks.items() if not ok)
  if failed:
    raise ValueError(
      f"Bootstrap reproduction gate failed for contrast {contrast_name} on {failed}: "
      f"recomputed {json.dumps(recomputed, sort_keys=True)} does not match committed ledger "
      f"values {json.dumps(ledger_contrast, sort_keys=True)}"
    )
  return {"contrast": contrast_name, "checks": checks, "passed": True}


def _fit_surface_models(
  surface: pl.DataFrame,
  c0_cols: list[str],
  a2: list[str],
  zb: list[str],
) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, dict[str, Any]], pl.DataFrame]:
  """Refit the three context-adjusted probes on the identical surface; return test predictions.

  Every fitted model must produce exactly the same ordered test identifiers and labels;
  any divergence raises before predictions are used (audit 20261015 round 01, finding 1).
  """
  from ecg_rep.evaluate import _fit_probe  # pyright: ignore[reportPrivateUsage]

  specs: dict[str, list[str]] = {
    "LM1_C_A2": [*c0_cols, *a2],
    "LM1_C_B": [*c0_cols, *zb],
    "LM1_C_A2_B": [*c0_cols, *a2, *zb],
  }
  preds: dict[str, npt.NDArray[np.float64]] = {}
  models: dict[str, dict[str, Any]] = {}
  test_ids: pl.DataFrame | None = None
  fitted_rows: dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]] = {}
  for name, cols in specs.items():
    probe, _dev, _val, test = _fit_probe(surface, cols, model_name=name)
    fitted_rows[name] = (test.subject_ids, test.study_ids, test.y)
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
  reference_name, (ref_ids, ref_study, ref_y) = next(iter(fitted_rows.items()))
  for name, (ids, study, labels) in fitted_rows.items():
    same = (
      np.array_equal(ids, ref_ids)
      and np.array_equal(study, ref_study)
      and np.array_equal(labels, ref_y)
    )
    if not same:
      raise ValueError(
        f"Fitted model {name} produced different ordered test identifiers or labels than "
        f"{reference_name}; refusing to combine predictions across divergent populations."
      )
  return preds, models, test_ids


def _paired_contrast(
  y: npt.NDArray[np.int64],
  joined: pl.DataFrame,
  reference: str,
  model: str,
  n_bootstraps: int,
) -> dict[str, Any]:
  """Outcome-stratified paired patient bootstrap for metric(model) − metric(reference)."""

  def col(name: str) -> npt.NDArray[np.float64]:
    return joined[name].to_numpy().astype(np.float64)

  contrast: dict[str, Any] = {"reference": reference, "model": model}
  for metric in ("auroc", "auprc"):
    c = bootstrap_paired_difference(
      y, col(reference), col(model), metric=metric, n_bootstraps=n_bootstraps, seed=BOOTSTRAP_SEED
    )
    contrast[metric] = {
      "point_estimate": c.point_estimate,
      "ci_lower": c.ci_lower,
      "ci_upper": c.ci_upper,
    }
  return contrast


def _addon_results_from_frame(
  frame: pl.DataFrame,
  builder: ReportFeatureBuilder,
  config_version: str,
  config_sha256_value: str,
  ledger: dict[str, Any],
  reference_aggregates: dict[str, Any],
  ledger_path: str,
  reference_aggregates_path: str,
  source_sha: str,
  n_bootstraps_addon: int = LEDGER_BOOTSTRAP_RESAMPLES,
  n_bootstraps_verification: int = LEDGER_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
  """Gate, fit, and compute the add-on contrast; return the results dict (no file writes)."""
  c0_cols = [c for c in C0_FEATURE_COLS if c in frame.columns]
  zb = [c for c in frame.columns if c.startswith("zb_")]
  ap = machine_feature_columns()
  report_cols = builder.feature_columns
  a2 = [*ap, *report_cols]

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

  preds, models, test_ids = _fit_surface_models(surface, c0_cols, a2, zb)
  joined = test_ids.with_columns(*(pl.Series(k, v) for k, v in preds.items()))
  y = joined["y"].to_numpy().astype(np.int64)
  n_test = int(len(y))
  n_events = int(y.sum())
  test_population_sha256 = population_hash(
    test_ids["subject_id"].to_numpy(), test_ids["study_id"].to_numpy()
  )

  # Gate 0 (hard): locked-test counts must match the reference ledger exactly.
  if n_test != ledger["n_test"] or n_events != ledger["n_test_events"]:
    raise ValueError(
      f"Population-identity gate failed: refit surface has {n_test} rows / {n_events} events; "
      f"committed ledger reports {ledger['n_test']} / {ledger['n_test_events']}."
    )
  # Gate 1 (hard): the refit population must be the ledger's patient set by identity,
  # not only by counts. The ledger hash is persisted by `cart-report-baseline`.
  ledger_hash = ledger.get("test_population_sha256")
  if not ledger_hash:
    raise ValueError(
      "Committed ledger lacks test_population_sha256; regenerate it with "
      "`ecg-rep cart-report-baseline` before running the add-on contrast."
    )
  if ledger_hash != test_population_sha256:
    raise ValueError(
      "Population-identity gate failed: refit test population hash "
      f"{test_population_sha256} does not match the committed ledger hash {ledger_hash}."
    )
  # Gate 2 (hard): the committed nonidentifying aggregate manifest must agree.
  manifest_population = reference_aggregates["population"]
  if manifest_population["test_population_sha256"] != test_population_sha256:
    raise ValueError(
      "Population-identity gate failed: refit test population hash "
      f"{test_population_sha256} does not match the committed aggregate manifest hash "
      f"{manifest_population['test_population_sha256']}."
    )
  if (
    manifest_population["n_test"] != n_test
    or manifest_population["n_events"] != n_events
  ):
    raise ValueError(
      "Population-identity gate failed: aggregate manifest counts "
      f"({manifest_population['n_test']} / {manifest_population['n_events']}) do not match "
      f"the refit surface ({n_test} / {n_events})."
    )

  # Gate 3: reproduced point metrics must match the ledger and the manifest.
  verification_models = [
    verify_ledger_metrics(name, models[name], ledger["models"][name]) for name in models
  ]
  verification_manifest_models = [
    verify_ledger_metrics(name, models[name], reference_aggregates["expected_models"][name])
    for name in models
  ]
  # Gate 4: the refit + bootstrap pipeline must reproduce committed contrast values,
  # always at the historical resample count (independent of --n-bootstraps).
  verification_contrasts = []
  for name, ref, new in REFERENCE_CONTRASTS:
    recomputed = _paired_contrast(y, joined, ref, new, n_bootstraps_verification)
    verification_contrasts.append(
      verify_ledger_contrast(name, recomputed, ledger["contrasts"][name])
    )

  # Gated result: the requested nested add-on contrast.
  addon = _paired_contrast(y, joined, "LM1_C_A2", "LM1_C_A2_B", n_bootstraps_addon)

  return {
    "status": "complete",
    "post_hoc": True,
    "analysis_status": "post_hoc_extension",
    "source_commit_git": source_sha,
    "representation": "A_double_cart_report_comparator",
    "config_version": config_version,
    "config_sha256": config_sha256_value,
    "vocabulary_sha256": vocabulary_hash(builder.vocabulary or ()) if builder.vocabulary else None,
    "n_features_a2": len(a2),
    "n_features_report_block": len(report_cols),
    "missing_machine_report_by_split": missing_by_split,
    "n_surface_by_split": n_surface,
    "n_surface_events_by_split": n_surface_events,
    "n_test": n_test,
    "n_test_events": n_events,
    "test_population_sha256": test_population_sha256,
    "primary_paired_valid_population_identical": True,
    "n_bootstraps_addon": n_bootstraps_addon,
    "n_bootstraps_verification": n_bootstraps_verification,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "models": models,
    "verification": {
      "ledger_path": ledger_path,
      "reference_aggregates_path": reference_aggregates_path,
      "metric_abs_tol": METRIC_ABS_TOL,
      "bootstrap_abs_tol": BOOTSTRAP_ABS_TOL,
      "population": {
        "cross_model_ordered_ids_match": True,
        "counts_match_ledger": True,
        "ledger_hash_match": True,
        "manifest_hash_match": True,
        "manifest_counts_match": True,
      },
      "models": verification_models,
      "manifest_models": verification_manifest_models,
      "reference_contrasts": verification_contrasts,
    },
    "contrasts": {
      ADDON_CONTRAST_NAME: {
        "reference": "LM1_C_A2",
        "model": "LM1_C_A2_B",
        "estimand": "AUROC/AUPRC(C0+A''+B) - AUROC/AUPRC(C0+A'')",
        **addon,
      },
    },
  }


def run_cart_addon_contrast(
  cohort_path: Path,
  measurements_path: Path,
  c0_path: Path,
  reps_dir: Path,
  config_path: Path,
  feasibility_path: Path,
  ledger_path: Path,
  reference_aggregates_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = LEDGER_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
  """LM1 addendum: paired add-on contrast AUROC/AUPRC(C0+A''+B) − (C0+A'') with paired
  bootstrap uncertainty, gated on population identity and ledger determinism.
  """
  config = load_freeze(config_path, feasibility_path)
  feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
  if not ledger_path.exists():
    raise FileNotFoundError(
      f"Committed reference ledger not found: {ledger_path}; regenerate it with "
      "`ecg-rep cart-report-baseline` before running the add-on contrast."
    )
  ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
  if not reference_aggregates_path.exists():
    raise FileNotFoundError(
      f"Committed aggregate manifest not found: {reference_aggregates_path}; it is "
      "versioned in the repository and required for the population-identity gate."
    )
  reference_aggregates = json.loads(reference_aggregates_path.read_text(encoding="utf-8"))

  machine = pl.read_csv(
    measurements_path,
    columns=["subject_id", "study_id", *MACHINE_RAW_COLUMNS, *REPORT_COLUMNS],
    infer_schema_length=100_000,
  )
  # Freeze-verify the vocabulary against the committed feasibility hash before any
  # outcome row is touched (derivation is development-only and availability-based).
  cohort_check = pl.read_parquet(cohort_path)
  dev_ids = cohort_check.filter(pl.col("split") == "dev").select(["subject_id", "study_id"])
  dev_reports = machine.join(dev_ids, on=["subject_id", "study_id"], how="inner")
  config = freeze_vocabulary(config, dev_reports, str(feasibility["vocabulary_sha256"]))
  del cohort_check, dev_reports

  frame, builder = build_cart_report_frame(cohort_path, machine, c0_path, reps_dir, config)
  results = _addon_results_from_frame(
    frame,
    builder,
    config.version,
    config_sha256(config_path),
    ledger,
    reference_aggregates,
    str(ledger_path),
    str(reference_aggregates_path),
    source_sha,
    n_bootstraps_addon=n_bootstraps,
    n_bootstraps_verification=LEDGER_BOOTSTRAP_RESAMPLES,
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_addon_outputs(results, output_dir, report_path)
  return results


def _write_addon_outputs(results: dict[str, Any], output_dir: Path, report_path: Path) -> None:
  """Persist the artifact JSON, supplementary CSV, and aggregate markdown report."""
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "cart_addon_contrast.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  _write_contrast_csv(results, report_path)
  _render_addon_report(report_path, results)


def _write_contrast_csv(results: dict[str, Any], report_path: Path) -> None:
  tables_dir = report_path.parent.parent / "manuscript" / "tables"
  tables_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for name, v in results["contrasts"].items():
    for metric in ("auroc", "auprc"):
      rows.append({
        "contrast": name,
        "metric": metric,
        "reference": v["reference"],
        "model": v["model"],
        "point_estimate": v[metric]["point_estimate"],
        "ci_lower": v[metric]["ci_lower"],
        "ci_upper": v[metric]["ci_upper"],
        "n_test": results["n_test"],
        "n_events": results["n_test_events"],
      })
  pl.DataFrame(rows).write_csv(tables_dir / "supp_cart_addon_contrast.csv")


def _render_addon_report(report_path: Path, results: dict[str, Any]) -> None:
  addon = results["contrasts"][ADDON_CONTRAST_NAME]
  verification = results["verification"]
  addon_resamples = results["n_bootstraps_addon"]
  verification_resamples = results["n_bootstraps_verification"]
  count_note = (
    ""
    if addon_resamples == verification_resamples
    else f"\n    The add-on contrast used {addon_resamples} resamples while ledger verification\n"
    f"    used the historical {verification_resamples}; the two counts are reported separately\n"
    "    and the intervals are not pooled."
  )
  lines = [
    "# LM1 cart add-on contrast results (B on top of C0+A'')",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {results['source_commit_git']}",
    f"config_sha256: {results['config_sha256']}",
    "analysis_status: post_hoc_extension",
    "inputs: [configs/cart_report_comparator_v1.yaml, artifacts/limitation_mitigation/lm1/cart_report_models.json, docs/limitation_mitigation/lm1_reference_aggregates.json]",
    "decisions: [refit on identical A''-valid surface; population-identity and ledger determinism gates passed before reporting]",
    "evidence: [cart_addon_contrast.json, supp_cart_addon_contrast.csv]",
    "unresolved: []",
    "verification: [cross-model ordered ids match, population hash matches ledger and committed manifest, reproduced point metrics match ledger and manifest, recomputed reference contrasts match committed bootstrap values]",
    "next_action: manuscript synthesis plan (separate run)",
    "```",
    "",
    "Round 09 audit section 4 recommended the missing nested add-on comparison",
    "AUROC(C0+A''+B) − AUROC(C0+A''). It is computed here with paired uncertainty;",
    "no increment is derived by subtracting stored point metrics.",
    "",
    "## Population",
    "",
    f"- Locked-test rows on the A''-valid paired-valid surface: {results['n_test']:,} "
    f"({results['n_test_events']} deaths)",
    f"- Surface n by split: {results['n_surface_by_split']}",
    f"- Surface events by split: {results['n_surface_events_by_split']}",
    "- Population verified against the ledger and the committed aggregate manifest by counts",
    f" and by order-independent SHA-256 of sorted (subject_id, study_id) pairs: "
    f"{results['test_population_sha256']}.",
    "",
    "## Requested contrast (outcome-stratified paired patient bootstrap, "
    f"{addon_resamples:,} resamples, seed {results['bootstrap_seed']})",
    "",
    "| Metric | Estimate | 95% CI |",
    "|---|---|---|",
    f"| AUROC | {addon['auroc']['point_estimate']:+.4f} | "
    f"{addon['auroc']['ci_lower']:+.4f} to {addon['auroc']['ci_upper']:+.4f} |",
    f"| AUPRC | {addon['auprc']['point_estimate']:+.4f} | "
    f"{addon['auprc']['ci_lower']:+.4f} to {addon['auprc']['ci_upper']:+.4f} |",
    "",
    f"Reference: {addon['reference']} (C0+A''); model: {addon['model']} (C0+A''+B).",
    f"Contrast key: `{ADDON_CONTRAST_NAME}`.{count_note}",
    "Positive values mean adding the frozen D-BETA embedding to C0+A'' increases the metric.",
    "The reverse direction (A'' on top of C0+B, `c_a2_b_minus_c_b`) is already reported in",
    "`supp_cart_report_comparator.csv`; together the two directions describe bidirectional",
    "addition to the combined model, not a replacement comparison. Intervals are exploratory,",
    "nominal 95% intervals from a post hoc analysis and are not multiplicity-adjusted; crossing",
    "zero is not equivalence and a positive point estimate is not established complementarity.",
    "",
    "## Determinism verification",
    "",
    f"- Metric agreement tolerance: {verification['metric_abs_tol']}; "
    f"bootstrap tolerance: {verification['bootstrap_abs_tol']}.",
    "- Every fitted model produced identical ordered test identifiers and labels.",
    "- Reproduced point metrics for "
    + ", ".join(v["model"] for v in verification["models"])
    + " match the committed `cart_report_models.json` entries and the committed aggregate manifest.",
    f"- Recomputed contrasts "
    + ", ".join(v["contrast"] for v in verification["reference_contrasts"])
    + f" reproduce the committed paired-bootstrap values at the historical "
    f"{verification_resamples:,} resamples (seed {results['bootstrap_seed']}).",
    "",
    "All rows are post hoc. The frozen primary B−A result is unchanged.",
    "",
  ]
  report_path.write_text("\n".join(lines), encoding="utf-8")