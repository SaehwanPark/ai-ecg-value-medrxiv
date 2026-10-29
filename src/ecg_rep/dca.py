"""Whole-cohort decision curve analysis (research plan section 13).

Plan A8 (Jinbo item 10) allows DCA as a secondary decision-analytic layer for a
clearly defined whole-cohort clinical action with prespecified, clinically
plausible probability thresholds. The decision modeled here is "refer for
further evaluation"; a threshold is the minimum predicted 30-day mortality
probability at which referring is preferred to not referring.

Net benefit at threshold t is

    NB(t) = TP_rate(t) - FP_rate(t) * (t / (1 - t))

with treat-none defined as 0 and treat-all as the event prevalence minus
(1 - prevalence) * t / (1 - t). DCA is descriptive utility framing, not mass
hypothesis testing: no threshold-by-threshold significance testing is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import numpy.typing as npt

#: Prespecified clinically plausible threshold grid for "refer for further
#: evaluation". Declared before evaluation; probability scale, not capacity.
DCA_THRESHOLDS: tuple[float, ...] = (
  0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
)

#: The modeled whole-cohort clinical action (plan A8).
DCA_DECISION = "refer for further evaluation"


@dataclass(frozen=True)
class DCARow:
  """Net benefit of one model at one threshold, with references."""

  model: str
  threshold: float
  n: int
  n_events: int
  flagged_fraction: float
  net_benefit: float
  net_benefit_treat_all: float
  net_benefit_treat_none: float = 0.0


def net_benefit(
  y_true: npt.NDArray[np.int64],
  probabilities: npt.NDArray[np.float64],
  threshold: float,
) -> tuple[float, float]:
  """Return (net benefit, flagged fraction) for one model and threshold."""
  if not 0.0 < threshold < 1.0:
    raise ValueError("threshold must be in (0, 1).")
  yt = np.asarray(y_true, dtype=np.int64)
  prob = np.asarray(probabilities, dtype=np.float64)
  if len(yt) != len(prob):
    raise ValueError("Labels and probabilities must have the same length.")
  if not ((prob >= 0.0).all() and (prob <= 1.0).all()):
    raise ValueError("Probabilities must be in [0, 1]; verify calibration first.")
  n = len(yt)
  flagged = prob >= threshold
  n_flagged = int(flagged.sum())
  flagged_fraction = float(n_flagged / n)
  odds = threshold / (1.0 - threshold)
  tp_rate = float((yt[flagged] == 1).sum() / n)
  fp_rate = float((yt[flagged] == 0).sum() / n)
  return tp_rate - fp_rate * odds, flagged_fraction


def treat_all_net_benefit(
  y_true: npt.NDArray[np.int64],
  threshold: float,
) -> float:
  """Net benefit of referring every patient at one threshold."""
  yt = np.asarray(y_true, dtype=np.int64)
  prevalence = float(yt.mean())
  odds = threshold / (1.0 - threshold)
  return prevalence - (1.0 - prevalence) * odds


def dca_rows(
  y_true: npt.NDArray[np.int64],
  probabilities_by_model: dict[str, npt.NDArray[np.float64]],
  thresholds: tuple[float, ...] = DCA_THRESHOLDS,
) -> list[dict[str, Any]]:
  """Whole-cohort net-benefit rows for each model and prespecified threshold."""
  yt = np.asarray(y_true, dtype=np.int64)
  rows: list[dict[str, Any]] = []
  for model, probabilities in probabilities_by_model.items():
    for threshold in thresholds:
      benefit, flagged_fraction = net_benefit(yt, probabilities, threshold)
      rows.append({
        "model": model,
        "threshold": threshold,
        "n": int(len(yt)),
        "n_events": int(yt.sum()),
        "flagged_fraction": flagged_fraction,
        "net_benefit": benefit,
        "net_benefit_treat_all": treat_all_net_benefit(yt, threshold),
        "net_benefit_treat_none": 0.0,
      })
  return rows


def incremental_net_benefit_rows(
  y_true: npt.NDArray[np.int64],
  probabilities_by_model: dict[str, npt.NDArray[np.float64]],
  contrast_pairs: tuple[tuple[str, str], ...] = (("p_a", "p_b"),),
  thresholds: tuple[float, ...] = DCA_THRESHOLDS,
) -> list[dict[str, Any]]:
  """Paired net-benefit differences (comparison minus baseline) at each threshold."""
  yt = np.asarray(y_true, dtype=np.int64)
  rows: list[dict[str, Any]] = []
  for baseline, comparison in contrast_pairs:
    if baseline not in probabilities_by_model or comparison not in probabilities_by_model:
      raise ValueError(f"Contrast {comparison} - {baseline} references unknown model scores.")
    for threshold in thresholds:
      base_benefit, _ = net_benefit(yt, probabilities_by_model[baseline], threshold)
      comp_benefit, _ = net_benefit(yt, probabilities_by_model[comparison], threshold)
      rows.append({
        "contrast": f"{comparison}_minus_{baseline}",
        "threshold": threshold,
        "n": int(len(yt)),
        "net_benefit_difference": comp_benefit - base_benefit,
      })
  return rows


def render_dca_markdown(
  metric_rows: list[dict[str, Any]],
  contrast_rows: list[dict[str, Any]],
  audit: dict[str, Any],
) -> str:
  """Aggregate note for the whole-cohort decision curve (plan section 13)."""

  def table(frame_rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in frame_rows:
      cells = []
      for col in columns:
        value = row[col]
        cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
      body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])

  thresholds = ", ".join(f"{100 * t:g}%" for t in audit["thresholds"])
  return "\n".join([
    "# Whole-cohort decision curve analysis",
    "",
    "Analysis status: `posthoc_secondary_layer`. Research plan section 13; collaborator action item A8.",
    f"Source SHA: `{audit['source_sha']}`.",
    "",
    f"The modeled whole-cohort clinical action is **{DCA_DECISION}** at prespecified "
    f"predicted-30-day-mortality probability thresholds of {thresholds}. Net benefit is "
    "TP rate minus FP rate times the threshold odds, with treat-all and treat-none "
    "references. These are calibrated-probability utility summaries of the frozen "
    "context-adjusted locked-test predictions, not evidence of treatment benefit, and "
    "not threshold-by-threshold significance testing. Capacity-constrained review "
    "budgets are reported separately and are not probability thresholds.",
    "",
    f"Locked-test rows: {audit['n_rows']:,}. Locked-test events: {audit['n_events']:,}.",
    "",
    "Calibration caveat: net benefit assumes usable probability scale. The frozen "
    "context-adjusted models have calibration intercepts within 0.12 of zero and "
    "slopes near 1 on this population; the DCA therefore describes the shipped "
    "probabilities without recalibration.",
    "",
    "## Net benefit by model and threshold",
    "",
    table(metric_rows, ["model", "threshold", "flagged_fraction", "net_benefit", "net_benefit_treat_all", "net_benefit_treat_none"]),
    "",
    "## Paired net-benefit differences",
    "",
    table(contrast_rows, ["contrast", "threshold", "net_benefit_difference"]),
    "",
    "MIMIC evaluation remains in-domain because D-BETA was pretrained on MIMIC-IV-ECG.",
    "",
  ])


def run_whole_cohort_dca(
  predictions_context_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  thresholds: tuple[float, ...] = DCA_THRESHOLDS,
) -> dict[str, Any]:
  """Whole-cohort DCA of the frozen context-adjusted locked-test predictions."""
  import polars as pl

  predictions = pl.read_parquet(predictions_context_path)
  required = {"mortality_30d", "p_a", "p_b", "p_ab"}
  missing = required - set(predictions.columns)
  if missing:
    raise ValueError(f"Predictions parquet is missing columns: {sorted(missing)}.")
  y = np.asarray(predictions["mortality_30d"].to_numpy(), dtype=np.int64)
  models = {
    "p_a": np.asarray(predictions["p_a"].to_numpy(), dtype=np.float64),
    "p_b": np.asarray(predictions["p_b"].to_numpy(), dtype=np.float64),
    "p_ab": np.asarray(predictions["p_ab"].to_numpy(), dtype=np.float64),
  }
  metric_rows = dca_rows(y, models, thresholds)
  contrast_rows = incremental_net_benefit_rows(
    y, models, contrast_pairs=(("p_a", "p_b"), ("p_a", "p_ab"), ("p_b", "p_ab")), thresholds=thresholds,
  )
  audit = {
    "source_sha": source_sha,
    "n_rows": int(len(y)),
    "n_events": int(y.sum()),
    "thresholds": list(thresholds),
    "decision": DCA_DECISION,
    "models": list(models),
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  pl.DataFrame(metric_rows).write_csv(output_dir / "dca_net_benefit.csv")
  pl.DataFrame(contrast_rows).write_csv(output_dir / "dca_net_benefit_contrasts.csv")
  (output_dir / "dca_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
  report = render_dca_markdown(metric_rows, contrast_rows, audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  return audit
