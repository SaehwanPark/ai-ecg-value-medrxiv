"""Manuscript Figures 1-6 and Tables 2-5 rendered from frozen study artifacts.

Every panel and table row is read from an existing artifact (locked-test predictions,
`scientific_analysis_results.json`, the readiness-slice JSON reports, or the CODE-15%
deployment report). Nothing is fitted or tuned here. The only computation is the
ordinary patient bootstrap of single-model metrics in Table 2, which resamples frozen
locked-test predictions.

Post hoc comparators (the random-encoder control and the cart-measurement baseline
A') are drawn with hollow markers and labeled as such, so readers can separate them
from the pre-specified models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "ecg-rep"
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import polars as pl  # noqa: E402

from ecg_rep.probe import (  # noqa: E402
  compute_auprc,
  compute_auroc,
  compute_binary_log_loss,
  compute_brier_score,
)

METRICS = {
  "auroc": compute_auroc,
  "auprc": compute_auprc,
  "brier": compute_brier_score,
  "log_loss": compute_binary_log_loss,
}

# Frozen locked-test prediction columns -> (model id, label, estimand).
FROZEN_MODELS: tuple[tuple[str, str, str, str], ...] = (
  ("p_a_ecg", "M_A_ECG", "A (CIIS features)", "ecg_only"),
  ("p_b_ecg", "M_B_ECG", "B (D-BETA)", "ecg_only"),
  ("p_c", "M_C", "C0", "context"),
  ("p_a", "M_A", "C0 + A", "context"),
  ("p_b", "M_B", "C0 + B", "context"),
  ("p_ab", "M_AB", "C0 + A + B", "context"),
)

SUBGROUP_LABELS = {
  "F": "Female",
  "M": "Male",
  "18-49": "Age 18-49",
  "50-64": "Age 50-64",
  "65-79": "Age 65-79",
  "80+": "Age 80+",
  "asian": "Asian",
  "black": "Black",
  "hispanic_latino": "Hispanic/Latino",
  "white": "White",
  "unknown_other": "Other/unknown",
}


def _load(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_metric_intervals(
  y: npt.NDArray[np.int64],
  p: npt.NDArray[np.float64],
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, dict[str, float]]:
  """Point estimate and 95% ordinary patient-bootstrap interval for each single-model metric."""
  rng = np.random.default_rng(seed)
  n = len(y)
  draws: dict[str, list[float]] = {m: [] for m in METRICS}
  for _ in range(n_bootstraps):
    idx = rng.integers(0, n, size=n)
    for name, fn in METRICS.items():
      draws[name].append(float(fn(y[idx], p[idx])))
  out: dict[str, dict[str, float]] = {}
  for name, fn in METRICS.items():
    lo, hi = np.quantile(np.asarray(draws[name]), [0.025, 0.975])
    out[name] = {"estimate": float(fn(y, p)), "ci_lower": float(lo), "ci_upper": float(hi)}
  return out


def table2_rows(
  ecg_only: pl.DataFrame,
  context: pl.DataFrame,
  n_bootstraps: int = 1000,
) -> list[dict[str, Any]]:
  """Table 2: locked-test AUROC, AUPRC, Brier, and log loss with 95% CIs per frozen model."""
  rows: list[dict[str, Any]] = []
  for col, model, label, estimand in FROZEN_MODELS:
    frame = ecg_only if col in ecg_only.columns else context
    y = frame["mortality_30d"].to_numpy().astype(np.int64)
    p = frame[col].to_numpy().astype(np.float64)
    stats = bootstrap_metric_intervals(y, p, n_bootstraps=n_bootstraps)
    row: dict[str, Any] = {"model": model, "label": label, "estimand": estimand, "n": len(y), "n_events": int(y.sum())}
    for name, s in stats.items():
      row[name] = s["estimate"]
      row[f"{name}_ci_lower"] = s["ci_lower"]
      row[f"{name}_ci_upper"] = s["ci_upper"]
    rows.append(row)
  return rows


#: Analysis-status taxonomy (round-1 audit item 4). ``post_hoc`` in the emitted
#: table is True for any non-prespecified status, preserving the hollow-marker
#: convention in Figure 2.
ANALYSIS_STATUS_PRIMARY = "prespecified_primary"
ANALYSIS_STATUS_CONTROL = "prespecified_control"
ANALYSIS_STATUS_SENSITIVITY = "post_hoc_sensitivity"
ANALYSIS_STATUS_EXTENSION = "post_hoc_extension"


def _interval_row(
  name: str,
  source: str,
  metric: str,
  point: float,
  lo: float | None,
  hi: float | None,
  analysis_status: str,
  interval_type: str,
) -> dict[str, Any]:
  """One Table 3 row.

  ``interval_type`` names what the interval columns contain so a table renderer
  can never present a seed range as a bootstrap CI (round-3 audit item 3):
  ``bootstrap_95ci`` for outcome-stratified paired-bootstrap limits,
  ``seed_range`` for min/max across random-control seeds, ``none`` when the
  row has a point estimate only.
  """
  if interval_type == "none" and (lo is not None or hi is not None):
    raise ValueError(f"Row {name!r} declares interval_type 'none' but carries interval values.")
  if interval_type in ("bootstrap_95ci", "seed_range") and (lo is None or hi is None):
    raise ValueError(f"Row {name!r} declares interval_type {interval_type!r} but is missing interval values.")
  return {
    "contrast": name,
    "metric": metric,
    "estimate": point,
    "interval_lower": lo,
    "interval_upper": hi,
    "interval_type": interval_type,
    "post_hoc": analysis_status.startswith("post_hoc"),
    "analysis_status": analysis_status,
    "source": source,
  }


def table3_rows(
  analyses: dict[str, Any],
  context_results: dict[str, Any],
  reporting: dict[str, Any],
  random_table: pl.DataFrame,
  machine_contrasts: pl.DataFrame,
) -> list[dict[str, Any]]:
  """Table 3: representation contrasts, bidirectional additions, attenuation, controls, and random-control seed summary."""
  sar = "artifacts/analysis/scientific_analysis_results.json"
  rows: list[dict[str, Any]] = []
  for estimand, key in (("ECG-only", "rq1_ecg_only_headline"), ("Context-adjusted", "rq1_context_adjusted_headline")):
    for metric in ("auroc", "auprc", "brier", "log_loss"):
      h = analyses[key][metric]
      b = h["bootstrap"]
      rows.append(_interval_row(f"{estimand}: B - A", f"{sar}:{key}.{metric}", metric, b["point_estimate"], b["ci_lower"], b["ci_upper"], ANALYSIS_STATUS_PRIMARY, "bootstrap_95ci"))
  bi = analyses["rq2_bidirectional"]
  rows.append(_interval_row("Add B to C0 + A", f"{sar}:rq2_bidirectional", "auroc", bi["add_b_to_a"], bi["add_b_to_a_ci_lower"], bi["add_b_to_a_ci_upper"], ANALYSIS_STATUS_PRIMARY, "bootstrap_95ci"))
  rows.append(_interval_row("Add A to C0 + B", f"{sar}:rq2_bidirectional", "auroc", bi["add_a_to_b"], bi["add_a_to_b_ci_lower"], bi["add_a_to_b_ci_upper"], ANALYSIS_STATUS_PRIMARY, "bootstrap_95ci"))
  att = analyses["rq3_attenuation"]
  rows.append(_interval_row("Attenuation of B - A by C0 (ECG-only gain minus context-adjusted gain)", f"{sar}:rq3_attenuation", "auroc", att["attenuation"], None, None, ANALYSIS_STATUS_PRIMARY, "none"))
  pca = context_results["models"]["M_B_PCA15"]["test_metrics"]["auroc"]
  ma = context_results["models"]["M_A"]["test_metrics"]["auroc"]
  rows.append(_interval_row("Capacity-matched: (C0 + B-PCA15) - (C0 + A)", "artifacts/probe/probe_results_context.json:models", "auroc", pca - ma, None, None, ANALYSIS_STATUS_CONTROL, "none"))
  for key, name in (
    ("ecg_only", "Symmetric ascertainment: ECG-only B - A"),
    ("context_adjusted", "Symmetric ascertainment: context-adjusted B - A"),
  ):
    s = reporting["ascertainment_sensitivity"]["symmetric"][key]["b_minus_a_auroc"]
    rows.append(_interval_row(name, f"artifacts/manuscript/manuscript_reporting.json:ascertainment_sensitivity.symmetric.{key}.b_minus_a_auroc", "auroc", s["point"], s["ci_lower"], s["ci_upper"], ANALYSIS_STATUS_SENSITIVITY, "bootstrap_95ci"))
  # Random-encoder control: seed-level summary only (round-2 audit item 4).
  # The estimate is the mean across the reported seeds and the interval columns
  # span the per-seed range (interval_type=seed_range, not a bootstrap CI);
  # per-seed rows remain in supp_random_encoder_control.csv and
  # 02_random_encoder_control.md.
  for estimand_key, estimand in (("ecg_only", "ECG-only"), ("context_adjusted", "Context-adjusted")):
    sub = random_table.filter(pl.col("estimand") == estimand_key)
    for prefix, name in (("pretrained_minus_random", "pretrained B - random"), ("random_minus_a", "random encoder - A")):
      for metric in ("auroc", "auprc"):
        col = f"{prefix}_{metric}"
        # polars Series.mean()/min()/max() stubs widen to PythonLiteral | None;
        # reduce via numpy on the Float64 column so the scalar type is explicit.
        vals = sub[col].cast(pl.Float64).to_numpy()
        if vals.size == 0:
          raise ValueError(f"No random-control rows for estimand {estimand_key!r}.")
        rows.append(
          _interval_row(
            f"{estimand} random control ({sub.height} seeds): {name}",
            "docs/manuscript/tables/supp_random_encoder_control.csv",
            metric,
            float(np.mean(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            ANALYSIS_STATUS_CONTROL,
            "seed_range",
          )
        )
  labels = {
    "ecg_only_b_minus_ap": "ECG-only: B - A' (cart measurements)",
    "context_b_minus_ap": "Context-adjusted: B - A'",
  }
  for r in machine_contrasts.iter_rows(named=True):
    if r["contrast"] in labels:
      for metric in ("auroc", "auprc"):
        rows.append(_interval_row(labels[r["contrast"]], "docs/manuscript/tables/supp_machine_baseline_contrasts.csv", metric, r[f"{metric}_point_estimate"], r[f"{metric}_ci_lower"], r[f"{metric}_ci_upper"], ANALYSIS_STATUS_EXTENSION, "bootstrap_95ci"))
  return rows


def table4_rows(analyses: dict[str, Any]) -> list[dict[str, Any]]:
  """Table 4: B - A gain by prespecified subgroup, both estimands and both metrics."""
  rows: list[dict[str, Any]] = []
  for variable in ("sex", "age", "race"):
    for estimand, suffix in (("context", ""), ("ecg_only", "_ecg_only")):
      for metric in ("auroc", "auprc"):
        for r in analyses[f"rq4_subgroup_{variable}{suffix}_{metric}"]:
          rows.append({
            "variable": r["subgroup_variable"],
            "level": r["subgroup_level"],
            "label": SUBGROUP_LABELS.get(r["subgroup_level"], r["subgroup_level"]),
            "estimand": estimand,
            "metric": metric,
            "n": r["n"],
            "n_events": r["n_events"],
            "gain_b_minus_a": r["gain"],
            "ci_lower": r["ci_lower"],
            "ci_upper": r["ci_upper"],
          })
  return rows


def table5_rows(analyses: dict[str, Any], benchmark: dict[str, Any], downstream: dict[str, Any]) -> list[dict[str, Any]]:
  """Table 5: controlled extraction and downstream benchmarks, full-cohort extraction logs, storage."""
  empty = {
    "median_ms_per_ecg": None,
    "range_ms_per_ecg": None,
    "ratio_to_ciis": None,
    "peak_rss_mb": None,
    "n_parameters": None,
    "storage_bytes_per_ecg": None,
    "fit_seconds": None,
  }
  rows: list[dict[str, Any]] = []
  for p in benchmark["pathways"]:
    rows.append({
      "source": "controlled_extraction_benchmark_1024_ecgs",
      "pathway": p["pathway"],
      **empty,
      "median_ms_per_ecg": p["median_ms_per_ecg"],
      "range_ms_per_ecg": None if "min_ms_per_ecg" not in p else f"{p['min_ms_per_ecg']:.2f}-{p['max_ms_per_ecg']:.2f}",
      "ratio_to_ciis": p.get("ratio_to_ciis_median"),
      "peak_rss_mb": p["peak_rss_bytes"] / 2**20 if "peak_rss_bytes" in p else None,
      "n_parameters": p.get("n_parameters"),
    })
  for p in downstream["pathways"]:
    rows.append({
      "source": "controlled_downstream_benchmark_paired_valid",
      "pathway": p["pathway"],
      **empty,
      "median_ms_per_ecg": p["predict_us_per_ecg_median"] / 1000.0,
      "n_parameters": p["n_features"] + 1,
      "fit_seconds": p["fit_seconds_median"],
    })
  for c in analyses["rq5_extraction_costs"]:
    rows.append({
      "source": "full_cohort_extraction_log_not_hardware_matched",
      "pathway": f"{c['representation']}_{c['hardware']}",
      **empty,
      "median_ms_per_ecg": c["per_ecg_latency_ms"],
      "storage_bytes_per_ecg": c["storage_bytes"] / c["n_records"],
    })
  return rows


def _save(fig: Any, figures_dir: Path, stem: str) -> list[str]:
  paths = []
  for ext in ("png", "svg"):
    path = figures_dir / f"{stem}.{ext}"
    fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None)
    paths.append(str(path))
  plt.close(fig)
  return paths


def figure1_design(figures_dir: Path, counts: dict[str, int]) -> list[str]:
  """Figure 1: study design schematic with cohort counts read from the frozen cohort.

  Round-2 audit structure: three numbered stages (cohort eligibility and
  splitting, shared representations and modeling, final paired-evaluable
  test population with the explicit 25,760 -> minus-330 -> 25,430 chain),
  plus a clearly separated dashed panel for secondary controls and the
  CODE-15% transport, which uses a different dataset and mortality horizon.
  Round-3 audit: stage 1 additionally shows the outcome-ascertainment
  attrition (161,278 eligible adults -> 129,031 ascertainable cohort),
  stage-3 labels are collision-free, the controls panel is narrower, and
  the PCA control is named without the workload sense of "capacity".
  Round-4 audit: stage-1 boxes sit below the stage heading with a
  consistent gap, the ascertainment exclusion names the eligibility
  criteria, and the controls panel height matches its content.
  Round-11 audit: the side panel becomes "Supporting analyses" and
  distinguishes the additional outcome settings (one-year mortality in
  CODE-15%, exploratory principal-discharge-coded AMI) from the
  previous-training, comparator and resource controls, matching the
  revised three-outcome narrative; the cohort flow is unchanged.
  """
  fig, ax = plt.subplots(figsize=(12, 5.6))
  ax.set_axis_off()

  def box(x: float, y: float, w: float, h: float, text: str, face: str = "#f2f2f2", dashed: bool = False) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="black", lw=1, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

  def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1})

  def stage(x: float, y: float, w: float, h: float, title: str) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#fbfbfb", edgecolor="black", lw=1.2))
    ax.text(x + 0.012, y + h - 0.02, title, ha="left", va="top", fontsize=8.5, weight="bold")

  n_excluded = counts["test"] - counts["test_analytic"]

  # Stage 1: eligibility, outcome-ascertainment attrition, patient-level splitting.
  # Round-4 audit: keep a consistent gap between the stage heading and the
  # first box row (heading top 0.94; row top 0.905) instead of letting the
  # added attrition flow collide with the heading.
  stage(0.0, 0.63, 0.68, 0.33, "1  Cohort eligibility, outcome ascertainment, and splitting")
  box(0.02, 0.78, 0.185, 0.125, f"Eligible adult\nindex ECGs\n{counts['eligible_adults']:,} patients")
  arrow(0.21, 0.8425, 0.40, 0.8425)
  ax.text(0.305, 0.815, f"exclude {counts['ascertainment_excluded']:,} not meeting\noutcome-ascertainment criteria",
          ha="center", va="top", fontsize=6.5)
  box(0.40, 0.78, 0.26, 0.125, f"Outcome-ascertainable cohort\n{counts['cohort']:,} patients")
  arrow(0.53, 0.78, 0.53, 0.762)
  box(0.40, 0.652, 0.26, 0.11, f"Patient-level 60/20/20 split\ndev {counts['dev']:,} / val {counts['val']:,}\ntest {counts['test']:,}")
  arrow(0.53, 0.652, 0.53, 0.613)

  # Stage 2: shared representations and modeling (one probe, no test-set tuning).
  stage(0.0, 0.315, 0.68, 0.295, "2  Shared representations and modeling")
  box(0.02, 0.44, 0.2, 0.115, "A: 15 prespecified\nmorphology measurements\n(standardized extraction)", "#dbe9f6")
  box(0.245, 0.44, 0.185, 0.115, "B: AI-derived\nECG features (D-BETA;\n768 values)", "#fde2c8")
  box(0.455, 0.44, 0.205, 0.115, "C0: age, sex, 31 prior-admission\nElixhauser indicators\n(pre-index)", "#e3f1df")
  for x0 in (0.12, 0.3375, 0.5575):
    arrow(x0, 0.44, x0, 0.427)
  box(0.02, 0.355, 0.64, 0.072, "Logistic regression models using the same training and evaluation procedures\nDevelopment-only standardization · validation log-loss tuning · no test-set tuning")
  ax.text(0.02, 0.322, "MIMIC-IV-ECG also contributed to D-BETA's earlier training.",
          ha="left", va="bottom", fontsize=7, style="italic")
  arrow(0.34, 0.315, 0.34, 0.287)

  # Stage 3: locked test partition and the paired-evaluable test cohort.
  stage(0.0, 0.0, 0.68, 0.285, "3  Locked test partition and paired-evaluable test cohort")
  box(0.02, 0.045, 0.19, 0.17, f"Locked test partition\n{counts['test']:,} patients\n30-day all-cause\nmortality")
  arrow(0.215, 0.13, 0.445, 0.13)
  ax.text(0.33, 0.135, f"exclude {n_excluded:,} patients\nwithout both ECG representations",
          ha="center", va="bottom", fontsize=6.5)
  box(0.45, 0.045, 0.21, 0.17, f"Paired-evaluable test cohort\n{counts['test_analytic']:,} patients / {counts['test_events']:,} deaths\nboth ECG representations valid")

  # Supporting analyses: additional outcome settings and methodological
  # controls, clearly separated from the primary 30-day pipeline.
  # Round-4 audit: panel height matches its content.
  # Round-11 audit: the panel distinguishes the additional outcome settings
  # (one-year mortality in CODE-15%, exploratory principal-discharge-coded AMI)
  # from the previous-training, comparator and resource controls, so the
  # displayed hierarchy matches the revised three-outcome narrative.
  ax.add_patch(Rectangle((0.70, 0.38), 0.30, 0.58, facecolor="#ffffff", edgecolor="black", lw=1, ls="--"))
  ax.text(0.715, 0.955, "Supporting analyses", ha="left", va="top", fontsize=8, weight="bold")
  ax.text(0.715, 0.915, "(additional outcome settings and controls,\nseparate from the primary pipeline)", ha="left", va="top", fontsize=7, style="italic")
  ax.text(
    0.715, 0.845,
    "Additional outcome settings\n"
    "- CODE-15%: 1-year mortality, separate\n"
    "  Brazilian cohort (different horizon)\n"
    "- Exploratory principal-discharge-coded\n"
    "  AMI (prior-training attribution)\n"
    "Previous-training and comparator controls\n"
    "- AI model without prior ECG\n"
    "  training, ten initializations\n"
    "- Cart-global A′ / automated-cart A″\n"
    "- Additional clinical information\n"
    "  incl. laboratory results (post hoc)\n"
    "- Control analysis using 15\n"
    "  AI-derived features\n"
    "Resource and sensitivity analyses\n"
    "- Extraction/storage measurements;\n"
    "  simulated selective routing\n"
    "- Ascertainment sensitivity; refit ranges",
    ha="left", va="top", fontsize=7,
  )
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  return _save(fig, figures_dir, "figure1_study_design")


def figure2_performance(figures_dir: Path, table2: list[dict[str, Any]]) -> list[str]:
  """Figure 2: locked-test AUROC/AUPRC for the six prespecified primary models.

  Round-2 audit: only the prespecified primary combinations (A, B, C0, C0+A,
  C0+B, C0+A+B) remain; the post hoc comparator points (A′, A″ companions)
  and random-encoder initializations moved to Figure 3 and Tables 3-4.
  Round-9 audit (20261018): the embedded footer cross-reference is corrected —
  the comparator and random-encoder results now reside in Supplementary Figure
  S8 (capacity remains main Figure 4) — and the footer adds the plain-language
  key naming what A, B and C0 are for clinician readers.
  """
  frozen = {r["model"]: r for r in table2}
  order: list[tuple[str, str]] = [
    ("A", "M_A_ECG"),
    ("B", "M_B_ECG"),
    ("C0", "M_C"),
    ("C0+A", "M_A"),
    ("C0+B", "M_B"),
    ("C0+A+B", "M_AB"),
  ]
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
  for ax, metric in zip(axes, ("auroc", "auprc")):
    for i, (label, key) in enumerate(order):
      color = "#1f77b4" if i < 2 else "#d62728"
      r = frozen[key]
      ax.errorbar(i, r[metric], yerr=[[r[metric] - r[f"{metric}_ci_lower"]], [r[f"{metric}_ci_upper"] - r[metric]]], fmt="o", color=color, capsize=3)
    ax.axvline(1.5, color="grey", lw=0.6, ls=":")
    ax.set_xticks(range(len(order)), [o[0] for o in order], fontsize=9)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} (left: ECG-only, right: with C0)", fontsize=9)
  fig.text(0.5, -0.06, "Prespecified primary models with 95% patient-bootstrap CIs (1,000 resamples). Blue: ECG-only; red: with basic clinical context (C0).\nKey: A = 15 prespecified ECG morphology measurements; B = AI-derived waveform features (D-BETA); C0 = basic recorded clinical information (age, sex, prior-admission comorbidities).\nPaired-evaluable test cohort, 25,430 patients, 924 deaths. Post hoc comparator and random-encoder points: Supplementary Figure S8; capacity results: Figure 4 and Tables 3-4.", ha="center", fontsize=7.5)
  fig.tight_layout()
  return _save(fig, figures_dir, "figure2_performance")


def figure3_representation_contrasts(
  figures_dir: Path,
  t3: list[dict[str, Any]],
  random_table: pl.DataFrame,
  clinical_tables_dir: Path,
  code_transport: dict[str, Any],
  code_probe_report: dict[str, Any],
  mitigation_table: pl.DataFrame | None = None,
) -> list[str]:
  """Figure 3 (central): representation contrast across contexts, comparators, and cohorts.

  Panel a (MIMIC): each row is its own estimand and is labeled as such —
  B − A ECG-only, B − A with C0, B − A′ with C0 (post hoc), B − A with Cacute
  on the Cacute-defined subset (post hoc), A″/Cmax companions from the unified
  LM9 table, and the prespecified pretrained B − random-encoder control. The rows are separate contrasts, not a
  sequence. Panel b (CODE-15%): the frozen (transported), target-trained, and
  demographics-adjusted B − A contrasts at the 1-year horizon, with per-row
  denominators because the populations differ.
  """
  by_name = {(r["contrast"], r["metric"]): r for r in t3}
  ecg = by_name[("ECG-only: B - A", "auroc")]
  ctx = by_name[("Context-adjusted: B - A", "auroc")]
  ap = by_name[("Context-adjusted: B - A'", "auroc")]
  att = pl.read_csv(clinical_tables_dir / "attenuation_comparison.csv")
  cacute = att.filter(pl.col("metric") == "auroc").row(0, named=True)
  rand = random_table.filter(pl.col("estimand") == "context_adjusted").sort("seed")
  n_seeds = rand.height

  track = next(t for t in code_transport["tracks"] if t["track_name"] == "full_cohort_365d")
  frozen = track["delta"]["delta_auroc_b_minus_a"]
  target = code_probe_report["contrasts"]["B_minus_A_auroc"]
  demo = code_probe_report["contrasts"]["C_demo_B_minus_C_demo_A_auroc"]

  fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
  ax = axes[0]
  locked_denom = "n=25,430 (924 deaths)"
  rows = [
    ("B − A, ECG-only", locked_denom, (ecg["estimate"], ecg["interval_lower"], ecg["interval_upper"]), False),
    ("B − A, +C0", locked_denom, (ctx["estimate"], ctx["interval_lower"], ctx["interval_upper"]), False),
    ("B − A′, +C0", locked_denom, (ap["estimate"], ap["interval_lower"], ap["interval_upper"]), True),
    (
      "B − A, +Cacute",
      f"n={int(cacute['n']):,} ({int(cacute['n_events'])} deaths, Cacute subset)",
      (cacute["delta_cacute"], cacute["delta_cacute_ci_low"], cacute["delta_cacute_ci_high"]),
      True,
    ),
  ]
  if mitigation_table is not None:
    for result_id, label in (
      ("L91_b_minus_a2_ecg_auroc", "B − A″, ECG-only"),
      ("L91_c_b_minus_c_a2_auroc", "B − A″, +C0"),
      ("L93_Cmax+B_minus_Cmax+A_auroc", "B − A, +Cmax (200 resamples)"),
    ):
      result = mitigation_table.filter(pl.col("id") == result_id).row(0, named=True)
      rows.append((
        label, f"n={int(result['n']):,} ({int(result['n_events'])} deaths)",
        (result["estimate"], result["interval_lower"], result["interval_upper"]), True,
      ))
  y = 0.0
  for label, denom, (point, lo, hi), post_hoc in reversed(rows):
    mfc = "white" if post_hoc else "#d62728"
    ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color="#d62728", mfc=mfc, capsize=2)
    ax.annotate(f"{point:.3f}", (point, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_yticks([])
    status = " [post hoc]" if post_hoc else ""
    ax.text(-0.02, y, label + status + "\n" + denom, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7.5, linespacing=1.2, color="grey" if post_hoc else "black")
    y += 1.0
  ctrl_label = "B − random, +C0 [control]"
  ctrl_denom = f"{locked_denom}; {n_seeds} seeds"
  for i, r in enumerate(rand.iter_rows(named=True)):
    point, lo, hi = r["pretrained_minus_random_auroc"], r["pretrained_minus_random_auroc_ci_lower"], r["pretrained_minus_random_auroc_ci_upper"]
    ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="D", color="#7f7f7f", mfc="white", capsize=1.5, ms=4)
    if i == 0:
      ax.text(-0.02, y, ctrl_label + "\n" + ctrl_denom, transform=ax.get_yaxis_transform(), ha="right", va="center",
              fontsize=7.5, linespacing=1.2, color="grey")
  ax.axvline(0, color="black", lw=0.6)
  ax.set_xlim(-0.045, 0.115)
  ax.set_ylim(-0.6, y + 0.4)
  ax.set_xlabel("AUROC difference (95% CI), MIMIC locked test")
  ax.set_title("a  MIMIC: representation contrasts (each row a separate estimand)", fontsize=8.5, loc="left")

  ax = axes[1]
  transport_denom = f"n={track['n']:,} ({track['n_events']:,} events)"
  code_probe_denom = f"n={code_probe_report['n_test']:,} ({code_probe_report['n_events_test']:,} events, CODE locked test)"
  code_rows = [
    ("Frozen probes, 1-year (transported)", transport_denom, (frozen["mean"], frozen["lo95"], frozen["hi95"])),
    ("Target-trained probes, 1-year", code_probe_denom, (target["point_estimate"], target["ci_lower"], target["ci_upper"])),
    ("Demographics-adjusted, 1-year", code_probe_denom, (demo["point_estimate"], demo["ci_lower"], demo["ci_upper"])),
  ]
  y = 0.0
  for label, denom, (point, lo, hi) in reversed(code_rows):
    ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", color="#1f77b4", capsize=2)
    ax.annotate(f"{point:.3f}", (point, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.text(-0.02, y, label + "\n" + denom, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7.5, linespacing=1.2)
    y += 1.0
  ax.axvline(0, color="black", lw=0.6)
  ax.set_xlim(-0.045, 0.115)
  ax.set_ylim(-0.6, y + 0.4)
  ax.set_yticks([])
  ax.set_xlabel("AUROC difference (95% CI), CODE-15% 1-year mortality")
  ax.set_title("b  CODE-15% (pretraining-disjoint, different horizon)", fontsize=8.5, loc="left")
  fig.text(
    0.5,
    -0.07,
    "Prespecified analyses: filled. Post hoc extensions (A′, A″, Cacute, Cmax): hollow. A″ includes cart statements/metadata, not waveform-only.\n"
    "Grey diamonds: pretrained − random per initialization seed. Each row is a separate estimand, not a sequence. Denominators differ where noted:\n"
    "the Cacute row uses the Cacute-defined subset, and panel b's frozen row is a transport population (MIMIC-trained probes, 223,644 records)\n"
    "against the CODE locked test (44,768 records). Panel b is a different-horizon association on a pretraining-disjoint cohort, never external validation.",
    ha="center",
    fontsize=7,
  )
  fig.tight_layout()
  return _save(fig, figures_dir, "figure3_representation_contrasts")


def figure3_patient_heterogeneity(figures_dir: Path, tables_dir: Path, overall: float) -> list[str]:
  """Figure 3 (round 08, audit 20261016): patient-heterogeneity display.

  Panel a prioritizes the C0-adjusted subgroup contrasts (paired patient-bootstrap
  intervals, deaths/patients in tick labels); the ECG-only subgroup contrasts remain
  in Supplementary Figure S4. Panel b plots the committed stratum-level death counts
  identified at the fixed global 5% C0+A/C0+B review lists. Both panels read the
  committed CSVs written by this runner; nothing is fitted or resampled here.
  Round-9 audit (20261018): the panel-b legend moves outside the plotting area
  (it overlapped the higher-stratum bar labels), its entries identify the two
  models in plain language, and the footer states upfront that both bars come
  from the same whole-cohort 5% review lists, not separate within-stratum quotas.
  Round-10 audit (20261018): the legend moves below panel b and the canvas
  narrows, removing the largely unused right-hand region the outside-right
  legend produced and enlarging both panels at publication display width.
  Round-11 audit (20261018): the single-line footer is wrapped to four lines;
  its ~560-character width dominated the tight bounding box (7791x1654-px saved
  canvas) and left unusually large blank areas around both data panels.
  """
  subgroup = pl.read_csv(tables_dir / "table4_subgroup_gain.csv")
  strata = pl.read_csv(tables_dir / "supp_risk_strata.csv")
  ctx = [
    r
    for r in subgroup.to_dicts()
    if r["estimand"] == "context" and r["metric"] == "auroc"
  ]
  strat = [r for r in strata.to_dicts() if r["stratum"] != "all"]
  # Round-10 audit: the round-9 outside-right legend expanded the saved canvas so
  # the panels occupied only ~55% of its width. The legend now sits below panel b
  # and the canvas is narrower, enlarging both panels at publication display size.
  fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.08, 1.0]})
  # Panel a: C0-adjusted subgroup forest (ECG-only contrasts stay in Figure S4).
  for i, r in enumerate(reversed(ctx)):
    ax_a.errorbar(
      r["gain_b_minus_a"],
      i,
      xerr=[[r["gain_b_minus_a"] - r["ci_lower"]], [r["ci_upper"] - r["gain_b_minus_a"]]],
      fmt="o",
      color="#d62728",
      capsize=2,
    )
  ax_a.set_yticks(
    range(len(ctx)),
    [f"{r['label']} ({r['n_events']}/{r['n']:,})" for r in reversed(ctx)],
    fontsize=8,
  )
  ax_a.axvline(0, color="black", lw=0.6)
  ax_a.axvline(overall, color="#d62728", lw=0.6, ls="--")
  ax_a.set_xlabel("AUROC gain, C0+B - C0+A (95% CI)", fontsize=8.5)
  ax_a.set_title("a  C0-adjusted gain by subgroup", fontsize=9, loc="left")
  # Panel b: stratum-level deaths identified at the fixed global 5% lists.
  xs = np.arange(len(strat))
  width = 0.36
  captured_a = [int(r["capacity_events_captured_by_a"]) for r in strat]
  captured_b = [int(r["capacity_events_captured_by_b"]) for r in strat]
  ax_b.bar(xs - width / 2, captured_a, width, color="#9aa5ad", label="C0+A: morphology measurements + clinical information")
  ax_b.bar(xs + width / 2, captured_b, width, color="#d62728", label="C0+B: AI waveform features + clinical information")
  for x, (va, vb) in zip(xs, zip(captured_a, captured_b)):
    ax_b.annotate(f"{va}", (x - width / 2, va), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
    ax_b.annotate(f"{vb}", (x + width / 2, vb), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7)
  ax_b.set_xticks(xs, [f"{r['stratum'].capitalize()}\n({int(r['n_events'])}/{int(r['n']):,})" for r in strat], fontsize=8)
  ax_b.set_ylim(top=max(captured_b) * 1.22)
  ax_b.set_ylabel("Deaths identified on the 5% list", fontsize=8.5)
  ax_b.set_title("b  Same global 5% C0-adjusted review lists", fontsize=9, loc="left")
  ax_b.annotate(
    f"Totals: {sum(captured_a)} vs {sum(captured_b)} (+{sum(captured_b) - sum(captured_a)})",
    (0.02, 0.97),
    xycoords="axes fraction",
    va="top",
    fontsize=8,
  )
  # Round-9 audit: legend moved out of the plotting area (the previous in-axes
  # upper-right placement overlapped the higher-stratum bar value labels).
  # Round-10 audit: it now sits below panel b so the saved canvas stays compact.
  ax_b.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
  # Round-11 audit: the footer was one ~560-character line, so bbox_inches="tight"
  # widened the saved canvas to the text width (7791x1654 px) and left the two data
  # panels occupying only a third of it. Wrapped to four lines within the figure width.
  fig.text(
    0.5,
    -0.24,
    "Panel b: both bars come from the same whole-cohort 5% review lists (1,272 patients each), not separate within-stratum quotas.\n"
    "Grey: the list of the morphology-measurement model with clinical information (C0+A); red: the list of the AI-feature model with the same\n"
    "information (C0+B). Both lists are restricted to validation-derived C0+A risk strata; the models select different patients, and identified\n"
    "deaths measure list enrichment, not deaths prevented. Panel a: tick labels give deaths/patients; dashed line: overall C0-adjusted gain.",
    ha="center",
    fontsize=7,
  )
  fig.tight_layout()
  return _save(fig, figures_dir, "figure3_patient_heterogeneity")


def figure4_overlap(figures_dir: Path, analyses: dict[str, Any]) -> list[str]:
  """Figure 4: discordance event rates and reciprocal residual-risk gradients."""
  fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
  labels = {"concordant_low": "Both low", "a_low_b_high": "A low, B high", "a_high_b_low": "A high, B low", "concordant_high": "Both high"}
  disc = analyses["rq2_discordance"]
  axes[0].bar(range(len(disc)), [100 * d["event_rate"] for d in disc], color=["#9e9e9e", "#fd8d3c", "#6baed6", "#636363"])
  for i, d in enumerate(disc):
    axes[0].text(i, 100 * d["event_rate"] + 0.3, f"{d['n_events']}/{d['n']:,}", ha="center", fontsize=7)
  axes[0].set_xticks(range(len(disc)), [labels[d["label"]] for d in disc], rotation=25, ha="right", fontsize=8)
  axes[0].set_ylabel("30-day mortality (%)")
  axes[0].set_title("a  Discordance (top quintile of each model)", fontsize=9, loc="left")
  for ax, key, title, inner in (
    (axes[1], "rq2_residual_b_within_a", "b  B high vs low within A quartiles", "B"),
    (axes[2], "rq2_residual_a_within_b", "c  A high vs low within B quartiles", "A"),
  ):
    strata = analyses[key]
    x = np.arange(len(strata))
    ax.plot(x, [100 * s["within_model_high_rate"] for s in strata], "o-", label=f"{inner} above within-quartile median")
    ax.plot(x, [100 * s["within_model_low_rate"] for s in strata], "o--", label=f"{inner} at or below median")
    ax.set_xticks(x, [s["stratum_label"] for s in strata])
    ax.set_ylabel("30-day mortality (%)")
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=7)
  fig.text(0.5, -0.04, "ECG-only probes, locked test (25,430 patients). High = at or above the 80th percentile of that model's predicted risk on the test set (descriptive cutpoints).", ha="center", fontsize=7.5)
  fig.tight_layout()
  return _save(fig, figures_dir, "figure4_information_overlap")


def figure5_subgroups(figures_dir: Path, table4: list[dict[str, Any]], overall: float) -> list[str]:
  """Figure 5: forest plot of context-adjusted and ECG-only AUROC gain by subgroup."""
  ctx = [r for r in table4 if r["estimand"] == "context" and r["metric"] == "auroc"]
  ecg = {(r["variable"], r["level"]): r for r in table4 if r["estimand"] == "ecg_only" and r["metric"] == "auroc"}
  fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(ctx) + 1.2))
  for i, r in enumerate(reversed(ctx)):
    ax.errorbar(r["gain_b_minus_a"], i, xerr=[[r["gain_b_minus_a"] - r["ci_lower"]], [r["ci_upper"] - r["gain_b_minus_a"]]], fmt="o", color="#d62728", capsize=2)
    e = ecg[(r["variable"], r["level"])]
    ax.errorbar(e["gain_b_minus_a"], i + 0.25, xerr=[[e["gain_b_minus_a"] - e["ci_lower"]], [e["ci_upper"] - e["gain_b_minus_a"]]], fmt="o", mfc="white", color="#1f77b4", capsize=2)
  ax.set_yticks(range(len(ctx)), [f"{r['label']} ({r['n_events']}/{r['n']:,})" for r in reversed(ctx)], fontsize=8)
  ax.axvline(0, color="black", lw=0.6)
  ax.axvline(overall, color="#d62728", lw=0.6, ls="--")
  ax.set_xlabel("AUROC gain, B - A (95% CI)")
  ax.set_title("Filled: context-adjusted (C0 + B vs C0 + A). Hollow: ECG-only.\nDashed line: overall context-adjusted gain. Labels: deaths/patients.", fontsize=8)
  fig.tight_layout()
  return _save(fig, figures_dir, "figure5_subgroups")


def figure6_capacity(figures_dir: Path, analyses: dict[str, Any]) -> list[str]:
  """Figure 6: deaths captured and PPV at fixed review capacity, ECG-only A vs B.

  Round 05 revisit: the plotted rq5_capacity_comparison values are ECG-only probe
  scores (the pipeline computes capacity only on the ECG-only path); the previous
  "C0 + A"/"C0 + B" labels misattributed them to context-adjusted models. The
  context-adjusted companion (rq5_capacity_comparison_context) is reported in
  Table 4, not plotted here.
  """
  rows = analyses["rq5_capacity_comparison"]
  x = [100 * r["capacity_fraction"] for r in rows]
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
  axes[0].plot(x, [r["capacity_a"]["n_events_captured"] for r in rows], "o-", label="A (ECG-only)")
  b_captured = [r["capacity_b"]["n_events_captured"] for r in rows]
  axes[0].plot(x, b_captured, "o-", label="B (ECG-only)")
  for xi, r in zip(x, rows):
    # Round-4 audit: integer event-count annotations with an en-dash interval.
    axes[0].annotate(f"+{r['delta_events_captured']:.0f} ({r['delta_events_ci_lower']:.0f}–{r['delta_events_ci_upper']:.0f})", (xi, r["capacity_b"]["n_events_captured"]), textcoords="offset points", xytext=(4, 6), fontsize=6.5)
  # Round-3 audit: vertical headroom so the largest-capacity annotation
  # clears the upper frame instead of resting against it.
  axes[0].set_ylim(top=max(b_captured) * 1.14)
  axes[0].set_xlim(0, 25)
  axes[0].set_xlabel("Review capacity (% of patients flagged)")
  axes[0].set_ylabel("Deaths among flagged patients")
  axes[0].set_title("a  Deaths captured (B - A difference, 95% CI)", fontsize=9, loc="left")
  axes[0].legend(fontsize=8)
  axes[1].plot(x, [100 * r["capacity_a"]["ppv"] for r in rows], "o-", label="A (ECG-only)")
  axes[1].plot(x, [100 * r["capacity_b"]["ppv"] for r in rows], "o-", label="B (ECG-only)")
  axes[1].set_xlabel("Review capacity (% of patients flagged)")
  axes[1].set_ylabel("PPV (%)")
  axes[1].set_title("b  Positive predictive value", fontsize=9, loc="left")
  axes[1].legend(fontsize=8)
  fig.text(0.5, -0.04, "Captured deaths indicate enrichment of a review list, not prevention or treatment benefit. Capacity fractions are review budgets, not probability thresholds. Comparison uses ECG-only probe scores; the context-adjusted companion is in Table 4.", ha="center", fontsize=7.5)
  fig.tight_layout()
  return _save(fig, figures_dir, "figure6_capacity")


def _frontier(rows: list[dict[str, Any]], policy: str, capacity: float) -> list[dict[str, Any]]:
  return sorted((r for r in rows if r["policy"] == policy and abs(r["capacity_fraction"] - capacity) < 1e-9), key=lambda r: r["utilization"])


def supp_cascade_frontier(figures_dir: Path, analyses: dict[str, Any], code_deployment: dict[str, Any]) -> list[str]:
  """Aligned count/identity panels from reported policies; no new evaluation or interpolation."""
  fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True, sharey=True)
  panels = (
    (analyses["rq6_cascade_policies"], analyses["rq6_cascade_random_stability"], "MIMIC: 30-day, in-domain; 25,430/924"),
    (code_deployment["operating_rows"], code_deployment["random_routing_seed_stability"], "CODE: 1-year transport; 44,768/577"),
  )
  for column, (rows, stability, title) in enumerate(panels):
    count_ax, identity_ax = axes[0, column], axes[1, column]
    guided = _frontier(rows, "rank_normalized", 0.05)
    full_b = next(r for r in rows if r["policy"] == "pure_B" and abs(r["capacity_fraction"] - 0.05) < 1e-9)
    pure_a = next(r for r in rows if r["policy"] == "pure_A" and abs(r["capacity_fraction"] - 0.05) < 1e-9)
    pts = [g for g in guided if g["utilization"] < 1.0]
    ux = [100 * r["utilization"] for r in pts]
    # Markers only: connecting segments could imply unreported operating points.
    count_ax.plot(ux, [100 * r["relative_capture_ratio"] for r in pts], "s", color="#385f78", label="Guided event count / full-B count")
    identity_ax.plot(ux, [100 * r["recovery_fraction"] for r in pts], "o", color="#182b3a", label="Guided same-event identity recovery")
    for baseline_row, marker, label in ((pure_a, "^", "A-only baseline"), (full_b, "x", "Full-B baseline")):
      for ax, key in ((count_ax, "relative_capture_ratio"), (identity_ax, "recovery_fraction")):
        ax.plot([100 * baseline_row["utilization"]], [100 * baseline_row[key]],
                marker=marker, ls="none", color="black", label=label)
    at20 = next(r for r in guided if abs(r["utilization"] - 0.2) < 1e-9)
    baseline = int(full_b["events_captured"])
    captured = int(at20["events_captured"])
    recovered = int(at20["events_also_captured_by_full_b"])
    missed = int(at20["full_b_events_missed"])
    count_ax.text(0.97, 0.08, f"20% B: {captured}/{baseline} event counts", transform=count_ax.transAxes, ha="right", fontsize=9)
    identity_ax.text(0.97, 0.08, f"20% B: {recovered}/{baseline} identities ({100 * at20['recovery_fraction']:.1f}%)\n{missed} full-B death identities missed", transform=identity_ax.transAxes, ha="right", fontsize=9)
    rand = [u for u in stability["utilizations"] if abs(u["capacity_fraction"] - 0.05) < 1e-9]
    if rand:
      ru = [100 * u["utilization"] for u in rand]
      # These are routing-seed extrema, not patient-bootstrap uncertainty.
      identity_ax.vlines(ru, [100 * u["random_identity_recovery"]["min"] for u in rand],
                         [100 * u["random_identity_recovery"]["max"] for u in rand],
                         color="grey", lw=5, alpha=0.45, label="Random identity min-max (200 seeds)")
    count_ax.set_title(title, fontsize=9, loc="left")
    for ax in (count_ax, identity_ax):
      ax.axhline(100, color="black", lw=0.6, ls=":")
      ax.set_xlim(-3, 103)
      ax.set_ylim(0, 115)
      ax.legend(fontsize=7, loc="center right")
    identity_ax.set_xlabel("B utilization (% receiving D-BETA)")
  axes[0, 0].set_ylabel("Event count (% of full-B count)")
  axes[1, 0].set_ylabel("Same-event identities (% of full-B identities)")
  fig.suptitle("Post hoc harmonized policies: fixed 5% final review capacity", fontsize=12)
  fig.text(0.5, 0.01, "Development/validation ECDF normalization is not mortality-probability calibration. Descriptive policies, not cascade DCA or clinical benefit.\nCount capture and identity loss must be read together. Markers show reported operating points only; n/deaths are cohort-specific.", ha="center", fontsize=8)
  fig.tight_layout(rect=(0, 0.06, 1, 0.96))
  paths = _save(fig, figures_dir, "supp_cascade_frontier")
  for name in paths:
    path = Path(name)
    if path.suffix == ".svg":
      svg = path.read_text(encoding="utf-8")
      end = svg.index(">", svg.index("<svg ")) + 1
      svg = svg[:end] + "\n<title>Post hoc cascade count capture and identity losses</title>\n<desc>Aligned MIMIC 30-day and CODE one-year reported policies at fixed 5% review capacity. At 20% B utilization: MIMIC counts 241/236, identities 189/236, 47 missed; CODE counts 151/155, identities 113/155, 42 missed. Descriptive, not cascade DCA or clinical benefit.</desc>" + svg[end:]
      # Matplotlib path data carries trailing spaces; normalize only this new asset.
      path.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
  return paths


def _cohort_flow_counts(cohort_dir: Path) -> dict[str, int]:
  """Subject-level eligibility counts from the frozen cohort flow report.

  Round-3 audit: Figure 1 stage 1 must show the outcome-ascertainment
  attrition, so the generator needs the waveform-eligible adult count and
  the ascertainment exclusions, not only the ascertainable cohort.
  """
  text = (cohort_dir / "cohort_flow.md").read_text(encoding="utf-8")

  def cell(step: int, column: int) -> int:
    row = next(line for line in text.splitlines() if line.startswith(f"| **{step}** |"))
    value = row.split("|")[column].strip().replace(",", "")
    return int(value)

  eligible_adults = cell(4, 4)
  ascertainment_excluded = cell(6, 6)
  return {"eligible_adults": eligible_adults, "ascertainment_excluded": ascertainment_excluded}


def run_manuscript_figures(
  artifacts_root: Path,
  tables_dir: Path,
  figures_dir: Path,
  clinical_tables_dir: Path = Path("docs/clinical_extensions/tables"),
  n_bootstraps: int = 1000,
) -> dict[str, Any]:
  """Render Figures 1-6 (evidence-ladder reorganized), the round-08 patient-heterogeneity main Figure 3, and Tables 2-5 from frozen artifacts."""
  probe = artifacts_root / "probe"
  analyses = _load(artifacts_root / "analysis" / "scientific_analysis_results.json")["analyses"]
  context_results = _load(probe / "probe_results_context.json")
  reporting = _load(artifacts_root / "manuscript" / "manuscript_reporting.json")
  benchmark = _load(artifacts_root / "compute" / "compute_benchmark.json")
  downstream = _load(artifacts_root / "compute" / "downstream_benchmark.json")
  code_deployment = _load(artifacts_root / "external" / "code15pct" / "deployment" / "deployment_report_365d.json")
  code_transport = _load(artifacts_root / "external" / "code15pct" / "transport" / "transport_results.json")
  code_probe_report = _load(artifacts_root / "external" / "code15pct" / "probes" / "code15_probe_report_365d.json")
  random_table = pl.read_csv(tables_dir / "supp_random_encoder_control.csv")
  machine_contrasts = pl.read_csv(tables_dir / "supp_machine_baseline_contrasts.csv")
  machine_models = pl.read_csv(tables_dir / "supp_machine_baseline_models.csv")
  mitigation_table = pl.read_csv(tables_dir / "table6_limitation_mitigation.csv")
  ecg_only = pl.read_parquet(probe / "test_predictions_ecg_only.parquet")
  context = pl.read_parquet(probe / "test_predictions_context.parquet")
  cohort = pl.read_parquet(artifacts_root / "cohort" / "primary_cohort.parquet")

  tables_dir.mkdir(parents=True, exist_ok=True)
  figures_dir.mkdir(parents=True, exist_ok=True)
  t2 = table2_rows(ecg_only, context, n_bootstraps=n_bootstraps)
  t3 = table3_rows(analyses, context_results, reporting, random_table, machine_contrasts)
  t4 = table4_rows(analyses)
  t5 = table5_rows(analyses, benchmark, downstream)
  for name, rows in (
    ("table2_overall_performance", t2),
    ("table3_incremental_value_and_controls", t3),
    ("table4_subgroup_gain", t4),
    ("table5_computational_cost", t5),
  ):
    pl.DataFrame(rows, infer_schema_length=None).write_csv(tables_dir / f"{name}.csv")

  splits = cohort.group_by("split").len()
  split_n = dict(zip(splits["split"].to_list(), splits["len"].to_list()))
  counts = {
    "cohort": cohort.height,
    "dev": int(split_n.get("dev", 0)),
    "val": int(split_n.get("val", 0)),
    "test": int(split_n.get("test", 0)),
    "test_analytic": context.height,
    "test_events": int(context["mortality_30d"].sum()),
    **_cohort_flow_counts(artifacts_root / "cohort"),
  }
  ctx_overall = analyses["rq1_context_adjusted_headline"]["auroc"]["gain"]
  figures = {
    "figure1": figure1_design(figures_dir, counts),
    "figure2": figure2_performance(figures_dir, t2),
    "figure3": figure3_representation_contrasts(figures_dir, t3, random_table, clinical_tables_dir, code_transport, code_probe_report, mitigation_table),
    "figure3_heterogeneity": figure3_patient_heterogeneity(figures_dir, tables_dir, ctx_overall),
    "figure4": figure4_overlap(figures_dir, analyses),
    "figure5": figure5_subgroups(figures_dir, t4, ctx_overall),
    "figure6": figure6_capacity(figures_dir, analyses),
    "supp_cascade": supp_cascade_frontier(figures_dir, analyses, code_deployment),
  }
  return {"status": "complete", "counts": counts, "figures": figures, "tables": ["table2", "table3", "table4", "table5"]}
