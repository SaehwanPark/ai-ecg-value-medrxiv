"""Run RQ1–RQ6 scientific estimands from locked-test prediction tables."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import numpy as np
import polars as pl

from ecg_rep.analysis import (
  DEFAULT_CAPACITY_FRACTIONS,
  assign_age_subgroup,
  bidirectional_increments,
  risk_stratified_incremental_value,
  capacity_comparison,
  capacity_evaluation,
  context_attenuation,
  discordance_groups,
  representation_gain,
  residual_risk_gradient,
  selective_a_to_b_cascade,
  spearman_alignment,
  subject_race_from_admissions,
  subgroup_representation_contrasts,
  subgroup_representation_gains,
)
from ecg_rep.cascade import (
  ECDFRankCalibrator,
  bootstrap_primary_operating_point,
  cascade_policy_comparison,
  policy_agreement,
  random_routing_seed_stability,
)
from ecg_rep.probe import bootstrap_paired_difference
from ecg_rep.compute import (
  costs_to_jsonable,
  incremental_extraction_cost,
  load_extraction_costs,
)
from ecg_rep.paths import load_dataset_tables, resolve_data_paths


#: Shared alias for the cascade utilization budgets used across cohorts.
CASCADE_UTILIZATION_FRACTIONS = (0.05, 0.10, 0.20, 0.40, 1.0)


def run_scientific_analysis(
  predictions_ecg_path: Path | None,
  predictions_context_path: Path | None,
  cohort_path: Path | None = None,
  output_dir: Path | None = None,
  reps_dir: Path | None = None,
  admissions_df: pl.DataFrame | None = None,
  n_bootstraps: int = 1000,
  calibration_scores_path: Path | None = None,
) -> dict[str, Any]:
  """Compute prespecified estimands from available locked-test prediction tables."""
  report: dict[str, Any] = {"status": "incomplete", "analyses": {}}

  if predictions_ecg_path is not None and predictions_ecg_path.exists():
    ecg = pl.read_parquet(predictions_ecg_path)
    y = ecg["mortality_30d"].to_numpy().astype(np.int64)
    pa = ecg["p_a_ecg"].to_numpy().astype(np.float64)
    pb = ecg["p_b_ecg"].to_numpy().astype(np.float64)
    sa = ecg["s_a_ecg"].to_numpy().astype(np.float64)
    sb = ecg["s_b_ecg"].to_numpy().astype(np.float64)

    rq1_ecg_headline: dict[str, Any] = {}
    for m in ("auroc", "auprc", "log_loss", "brier"):
      g = representation_gain(y, pa, pb, metric=m)
      b = bootstrap_paired_difference(y, pa, pb, metric=m, n_bootstraps=n_bootstraps, seed=42)
      rq1_ecg_headline[m] = {
        "metric": m,
        "gain": g.gain,
        "score_a": g.score_a,
        "score_b": g.score_b,
        "bootstrap": asdict(b),
      }
    report["analyses"]["rq1_ecg_only_headline"] = rq1_ecg_headline

    report["analyses"]["rq2_alignment_total"] = asdict(spearman_alignment(pa, pb))
    report["analyses"]["rq2_alignment_ecg_specific"] = asdict(spearman_alignment(sa, sb))
    report["analyses"]["rq2_alignment_total_ecg_only"] = asdict(spearman_alignment(pa, pb))
    report["analyses"]["rq2_alignment_ecg_specific_ecg_only"] = asdict(spearman_alignment(sa, sb))
    report["analyses"]["rq2_residual_b_within_a"] = [
      asdict(s) for s in residual_risk_gradient(y, pa, pb)
    ]
    report["analyses"]["rq2_residual_a_within_b"] = [
      asdict(s) for s in residual_risk_gradient(y, pb, pa)
    ]
    report["analyses"]["rq2_discordance"] = [
      asdict(g) for g in discordance_groups(y, pa, pb)
    ]
    report["analyses"]["rq5_capacity_a"] = [
      asdict(capacity_evaluation(y, pa, q)) for q in DEFAULT_CAPACITY_FRACTIONS
    ]
    report["analyses"]["rq5_capacity_b"] = [
      asdict(capacity_evaluation(y, pb, q)) for q in DEFAULT_CAPACITY_FRACTIONS
    ]
    report["analyses"]["rq5_capacity_comparison"] = [
      asdict(capacity_comparison(y, pa, pb, q, n_bootstraps=n_bootstraps))
      for q in DEFAULT_CAPACITY_FRACTIONS
    ]
    report["analyses"]["rq6_cascade"] = [
      asdict(selective_a_to_b_cascade(y, pa, pb, u, capacity_fraction=0.05))
      for u in (0.05, 0.10, 0.20, 0.40, 1.0)
    ]
    # Rank-normalized cascade sensitivity (audit 2026-10-05 item 1): the raw
    # probability-scale blend above is the reference policy; when frozen-probe
    # dev/val calibration scores are available, fit ECDF rank calibrators on
    # those rows only and compare rank_normalized/mixed_reference/random_routing
    # at the same 5% review capacity. Calibrators are never fitted on test rows.
    calibration_path = calibration_scores_path
    if calibration_path is not None and calibration_path.exists():
      calib = pl.read_parquet(calibration_path)
      calib_a = calib["p_a"].to_numpy().astype(np.float64)
      calib_b = calib["p_b"].to_numpy().astype(np.float64)
      policy_rows = cascade_policy_comparison(
        y, pa, pb,
        calibration_scores_a=calib_a,
        calibration_scores_b=calib_b,
        utilization_fractions=CASCADE_UTILIZATION_FRACTIONS,
        capacity_fractions=(0.05,),
      )
      report["analyses"]["rq6_cascade_policies"] = policy_rows
      rank_scores_a = ECDFRankCalibrator().fit(calib_a).transform(pa)
      rank_scores_b = ECDFRankCalibrator().fit(calib_b).transform(pb)
      report["analyses"]["rq6_cascade_random_stability"] = random_routing_seed_stability(
        y, rank_scores_a, rank_scores_b,
        utilizations=CASCADE_UTILIZATION_FRACTIONS,
        capacity_fractions=(0.05,),
        n_seeds=200,
        base_seed=1912,
      )
      report["analyses"]["rq6_cascade_rank_vs_mixed_bootstrap"] = (
        bootstrap_primary_operating_point(
          y, pa, pb,
          ECDFRankCalibrator().fit(calib_a), ECDFRankCalibrator().fit(calib_b),
          n_bootstraps=n_bootstraps, seed=1912, utilization=0.10, capacity_fraction=0.05,
        )
      )
      # Direct mixed-vs-rank selected-patient agreement (audit 2026-10-05
      # audit02): aggregate summaries can coincide while individual
      # prioritization differs; report the overlap explicitly.
      report["analyses"]["rq6_cascade_policy_agreement"] = policy_agreement(
        y, pa, pb, calib_a, calib_b,
        utilization_fractions=CASCADE_UTILIZATION_FRACTIONS,
        capacity_fraction=0.05,
      )
    report["status"] = "partial"

  if predictions_context_path is not None and predictions_context_path.exists():
    ctx = pl.read_parquet(predictions_context_path)
    y = ctx["mortality_30d"].to_numpy().astype(np.int64)
    pa = ctx["p_a"].to_numpy().astype(np.float64)
    pb = ctx["p_b"].to_numpy().astype(np.float64)
    pab = ctx["p_ab"].to_numpy().astype(np.float64)

    rq1_ctx_headline: dict[str, Any] = {}
    for m in ("auroc", "auprc", "log_loss", "brier"):
      g = representation_gain(y, pa, pb, metric=m)
      b = bootstrap_paired_difference(y, pa, pb, metric=m, n_bootstraps=n_bootstraps, seed=42)
      rq1_ctx_headline[m] = {
        "metric": m,
        "gain": g.gain,
        "score_a": g.score_a,
        "score_b": g.score_b,
        "bootstrap": asdict(b),
      }
    report["analyses"]["rq1_context_adjusted_headline"] = rq1_ctx_headline

    report["analyses"]["rq2_bidirectional"] = asdict(
      bidirectional_increments(y, pa, pb, pab, metric="auroc", n_bootstraps=n_bootstraps)
    )
    # Context-adjusted capacity companion (round 05 revisit): the prespecified
    # RQ5 capacity comparison was computed only on ECG-only scores, while the
    # manuscript's primary performance framing is context-adjusted. Report the
    # identical capacity estimand on the context-adjusted frozen probes so the
    # two framings are directly comparable at the same review capacities.
    report["analyses"]["rq5_capacity_comparison_context"] = [
      asdict(capacity_comparison(y, pa, pb, q, n_bootstraps=n_bootstraps))
      for q in DEFAULT_CAPACITY_FRACTIONS
    ]
    # Round 05 (audit 2026-10-16, finding 1): post hoc conventional-risk-stratified
    # incremental value. Tertile cutoffs come from the validation split of the
    # frozen C0+A score table (outcome-free; round 06 reconciliation: the stated
    # protocol is validation-derived, so development rows are excluded); the
    # analysis never uses test outcomes for grouping.
    if calibration_scores_path is not None and calibration_scores_path.exists():
      calib = pl.read_parquet(calibration_scores_path)
      val_risk = calib.filter(pl.col("split") == "val")["p_a"].to_numpy().astype(np.float64)
      if len(val_risk) >= 30:
        cutoffs = [float(q) for q in np.quantile(val_risk, [1 / 3, 2 / 3])]
        report["analyses"]["rq7_risk_strata"] = risk_stratified_incremental_value(
          y, pa, pb, pa, cutoffs,
          capacity_fraction=0.05,
          n_bootstraps=n_bootstraps,
        )
    # Context-adjusted alignment: total risk rho(p_A, p_B) and ECG-specific rho(S_A, S_B)
    report["analyses"]["rq2_alignment_total"] = asdict(spearman_alignment(pa, pb))
    if "s_a_ecg_contrib" in ctx.columns and "s_b_ecg_contrib" in ctx.columns:
      sa_contrib = ctx["s_a_ecg_contrib"].to_numpy().astype(np.float64)
      sb_contrib = ctx["s_b_ecg_contrib"].to_numpy().astype(np.float64)
      report["analyses"]["rq2_alignment_ecg_specific"] = asdict(
        spearman_alignment(sa_contrib, sb_contrib)
      )

    if predictions_ecg_path is not None and predictions_ecg_path.exists():
      ecg = pl.read_parquet(predictions_ecg_path)
      # Align on subject_id for attenuation when both tables exist.
      joined = ecg.join(
        ctx.select(["subject_id", "p_a", "p_b"]),
        on="subject_id",
        how="inner",
      )
      if len(joined) > 0:
        yj = joined["mortality_30d"].to_numpy().astype(np.int64)
        report["analyses"]["rq3_attenuation"] = asdict(
          context_attenuation(
            yj,
            joined["p_a_ecg"].to_numpy().astype(np.float64),
            joined["p_b_ecg"].to_numpy().astype(np.float64),
            joined["p_a"].to_numpy().astype(np.float64),
            joined["p_b"].to_numpy().astype(np.float64),
            metric="auroc",
          )
        )
    report["status"] = "partial"

  if cohort_path is not None and cohort_path.exists():
    cohort = pl.read_parquet(cohort_path)
    # Prefer context-adjusted predictions for primary subgroup heterogeneity
    pred_source = None
    is_context = False
    if predictions_context_path is not None and predictions_context_path.exists():
      pred_source = pl.read_parquet(predictions_context_path)
      is_context = True
    elif predictions_ecg_path is not None and predictions_ecg_path.exists():
      pred_source = pl.read_parquet(predictions_ecg_path)
      is_context = False

    if pred_source is not None:
      joined = pred_source.join(
        cohort.select(["subject_id", "gender", "age_at_ecg"]),
        on="subject_id",
        how="left",
      )
      y = joined["mortality_30d"].to_numpy().astype(np.int64)
      pa = (
        joined["p_a"].to_numpy().astype(np.float64)
        if is_context
        else joined["p_a_ecg"].to_numpy().astype(np.float64)
      )
      pb = (
        joined["p_b"].to_numpy().astype(np.float64)
        if is_context
        else joined["p_b_ecg"].to_numpy().astype(np.float64)
      )
      sex = joined["gender"].fill_null("U").to_numpy()
      ages = joined["age_at_ecg"].to_numpy().astype(np.float64)
      age_labels = assign_age_subgroup(ages)

      for m in ("auroc", "auprc"):
        report["analyses"][f"rq4_subgroup_sex_{m}"] = [
          asdict(g)
          for g in subgroup_representation_gains(
            y, pa, pb, sex, "sex", metric=m, n_bootstraps=n_bootstraps
          )
        ]
        if "F" in sex and "M" in sex:
          report["analyses"][f"rq4_subgroup_sex_contrast_{m}"] = asdict(
            subgroup_representation_contrasts(
              y, pa, pb, sex, "sex", "F", "M", metric=m, n_bootstraps=n_bootstraps
            )
          )
        report["analyses"][f"rq4_subgroup_age_{m}"] = [
          asdict(g)
          for g in subgroup_representation_gains(
            y, pa, pb, age_labels, "age", metric=m, n_bootstraps=n_bootstraps
          )
        ]

      report["analyses"]["rq4_subgroup_sex"] = report["analyses"]["rq4_subgroup_sex_auroc"]
      if "rq4_subgroup_sex_contrast_auroc" in report["analyses"]:
        report["analyses"]["rq4_subgroup_sex_contrast"] = report["analyses"]["rq4_subgroup_sex_contrast_auroc"]
      report["analyses"]["rq4_subgroup_age"] = report["analyses"]["rq4_subgroup_age_auroc"]

      race_source = admissions_df
      if race_source is None:
        try:
          _paths = resolve_data_paths(None, None)
          _tables = load_dataset_tables(_paths)
          race_source = _tables[2]
        except Exception:  # noqa: BLE001 - race subgroup is best-effort when data missing
          race_source = None
      race_map = subject_race_from_admissions(race_source) if race_source is not None else None
      if race_map is not None:
        with_race = joined.join(
          race_map.select(["subject_id", "race_group"]), on="subject_id", how="left"
        )
        y_r = with_race["mortality_30d"].to_numpy().astype(np.int64)
        pa_r = (
          with_race["p_a"].to_numpy().astype(np.float64)
          if is_context
          else with_race["p_a_ecg"].to_numpy().astype(np.float64)
        )
        pb_r = (
          with_race["p_b"].to_numpy().astype(np.float64)
          if is_context
          else with_race["p_b_ecg"].to_numpy().astype(np.float64)
        )
        race_labels = with_race["race_group"].fill_null("unknown_other").to_numpy()
        for m in ("auroc", "auprc"):
          report["analyses"][f"rq4_subgroup_race_{m}"] = [
            asdict(g)
            for g in subgroup_representation_gains(
              y_r, pa_r, pb_r, race_labels, "race_ethnicity", metric=m, n_bootstraps=n_bootstraps
            )
          ]
          if "black" in race_labels and "white" in race_labels:
            report["analyses"][f"rq4_subgroup_race_contrast_{m}"] = asdict(
              subgroup_representation_contrasts(
                y_r,
                pa_r,
                pb_r,
                race_labels,
                "race_ethnicity",
                "black",
                "white",
                metric=m,
                n_bootstraps=n_bootstraps,
              )
            )
        report["analyses"]["rq4_subgroup_race"] = report["analyses"]["rq4_subgroup_race_auroc"]
        if "rq4_subgroup_race_contrast_auroc" in report["analyses"]:
          report["analyses"]["rq4_subgroup_race_contrast"] = report["analyses"]["rq4_subgroup_race_contrast_auroc"]

      # Supplementary ECG-only subgroup gains when context-adjusted is primary
      if is_context and predictions_ecg_path is not None and predictions_ecg_path.exists():
        ecg_supp = pl.read_parquet(predictions_ecg_path)
        joined_ecg = ecg_supp.join(
          cohort.select(["subject_id", "gender", "age_at_ecg"]),
          on="subject_id",
          how="left",
        )
        y_e = joined_ecg["mortality_30d"].to_numpy().astype(np.int64)
        pa_e = joined_ecg["p_a_ecg"].to_numpy().astype(np.float64)
        pb_e = joined_ecg["p_b_ecg"].to_numpy().astype(np.float64)
        sex_e = joined_ecg["gender"].fill_null("U").to_numpy()
        ages_e = joined_ecg["age_at_ecg"].to_numpy().astype(np.float64)

        for m in ("auroc", "auprc"):
          report["analyses"][f"rq4_subgroup_sex_ecg_only_{m}"] = [
            asdict(g)
            for g in subgroup_representation_gains(
              y_e, pa_e, pb_e, sex_e, "sex", metric=m, n_bootstraps=n_bootstraps
            )
          ]
          report["analyses"][f"rq4_subgroup_age_ecg_only_{m}"] = [
            asdict(g)
            for g in subgroup_representation_gains(
              y_e,
              pa_e,
              pb_e,
              assign_age_subgroup(ages_e),
              "age",
              metric=m,
              n_bootstraps=n_bootstraps,
            )
          ]
        report["analyses"]["rq4_subgroup_sex_ecg_only"] = report["analyses"]["rq4_subgroup_sex_ecg_only_auroc"]
        report["analyses"]["rq4_subgroup_age_ecg_only"] = report["analyses"]["rq4_subgroup_age_ecg_only_auroc"]

        if race_map is not None:
          with_race_e = joined_ecg.join(
            race_map.select(["subject_id", "race_group"]), on="subject_id", how="left"
          )
          y_re = with_race_e["mortality_30d"].to_numpy().astype(np.int64)
          pa_re = with_race_e["p_a_ecg"].to_numpy().astype(np.float64)
          pb_re = with_race_e["p_b_ecg"].to_numpy().astype(np.float64)
          race_labels_e = with_race_e["race_group"].fill_null("unknown_other").to_numpy()
          for m in ("auroc", "auprc"):
            report["analyses"][f"rq4_subgroup_race_ecg_only_{m}"] = [
              asdict(g)
              for g in subgroup_representation_gains(
                y_re,
                pa_re,
                pb_re,
                race_labels_e,
                "race_ethnicity",
                metric=m,
                n_bootstraps=n_bootstraps,
              )
            ]
          report["analyses"]["rq4_subgroup_race_ecg_only"] = report["analyses"]["rq4_subgroup_race_ecg_only_auroc"]

  if reps_dir is not None and Path(reps_dir).exists():
    costs = load_extraction_costs(Path(reps_dir))
    if costs:
      report["analyses"]["rq5_extraction_costs"] = costs_to_jsonable(costs)
      by_name = {c.representation: c for c in costs}
      if "ciis_features" in by_name and "dbeta_embedding" in by_name:
        report["analyses"]["rq5_incremental_extraction"] = incremental_extraction_cost(
          by_name["ciis_features"],
          by_name["dbeta_embedding"],
        )

  if report["analyses"]:
    if (
      "rq2_bidirectional" in report["analyses"]
      and "rq2_alignment_total" in report["analyses"]
      and "rq6_cascade" in report["analyses"]
    ):
      report["status"] = "complete"

  if output_dir is not None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scientific_analysis_results.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
  return report
