"""LM4: CODE-15% local recalibration of the transported frozen MIMIC scores.

The frozen MIMIC 30-day probes are miscalibrated on CODE-15% 1-year mortality.
LM4 asks whether that failure is a simple local intercept/scale problem: fit
transparent recalibration maps (R1 intercept-only, R2 intercept-and-slope) on the
CODE **validation** split only, apply them to the untouched CODE locked test once,
and report calibration and proper-scoring changes with paired patient bootstrap.

Analysis status: ``post hoc extension``. The frozen transport analysis is never
modified; recalibrated probabilities are written as new artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from ecg_rep.external.code15_transport import (
  calibration_intercept_slope,
  load_frozen_probes,
  transport_scores,
)

DEFAULT_EPSILON = 1e-6


@dataclass(frozen=True)
class RecalibrationConfig:
  """Frozen LM4 configuration (authoritative copy: configs/code15_recalibration_v1.yaml)."""

  version: str = "code15_recalibration_v1"
  horizon: str = "365d"
  fit_split: str = "val"
  eval_split: str = "test"
  epsilon: float = DEFAULT_EPSILON
  n_val_refit_sensitivity: int = 1000
  seed: int = 1912
  allowed_variants: tuple[str, ...] = ("R0_frozen", "R1_intercept", "R2_intercept_slope")

  def to_dict(self) -> dict[str, Any]:
    return {
      "version": self.version,
      "horizon": self.horizon,
      "fit_split": self.fit_split,
      "eval_split": self.eval_split,
      "epsilon": self.epsilon,
      "n_val_refit_sensitivity": self.n_val_refit_sensitivity,
      "seed": self.seed,
      "allowed_variants": list(self.allowed_variants),
    }


def load_recalibration_config(path: Path) -> RecalibrationConfig:
  """Load the frozen LM4 config; unknown keys are rejected."""
  import yaml

  raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError(f"Recalibration config at {path} is not a mapping.")
  known = set(RecalibrationConfig.__dataclass_fields__)
  unknown = set(raw) - known
  if unknown:
    raise ValueError(f"Unknown recalibration config keys: {sorted(unknown)}")
  return RecalibrationConfig(
    version=str(raw["version"]),
    horizon=str(raw["horizon"]),
    fit_split=str(raw["fit_split"]),
    eval_split=str(raw["eval_split"]),
    epsilon=float(raw["epsilon"]),
    n_val_refit_sensitivity=int(raw["n_val_refit_sensitivity"]),
    seed=int(raw["seed"]),
    allowed_variants=tuple(str(v) for v in raw.get("allowed_variants", ("R0_frozen", "R1_intercept", "R2_intercept_slope"))),
  )


def _clip_logit(p: npt.NDArray[np.floating], epsilon: float) -> npt.NDArray[np.float64]:
  """Clip probabilities for numerical stability with the frozen epsilon."""
  return np.log(np.clip(np.asarray(p, dtype=np.float64), epsilon, 1.0 - epsilon) / (1.0 - np.clip(np.asarray(p, dtype=np.float64), epsilon, 1.0 - epsilon)))


def fit_r1_intercept_only(
  y: npt.NDArray[np.integer],
  p: npt.NDArray[np.floating],
  epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float]:
  """Fit ``logit(Y) = alpha + logit(p)`` with the slope fixed exactly to 1.

  MLE over the single intercept by deterministic scalar Newton/BFGS. Only the
  validation labels and transported probabilities enter; test IDs never do.
  """
  yt = np.asarray(y, dtype=np.float64)
  logit_p = _clip_logit(p, epsilon)
  if len(np.unique(yt)) < 2:
    raise ValueError("R1 requires both outcome classes in the calibration-fit split.")

  def neg_log_likelihood(alpha_arr: npt.NDArray[np.floating]) -> float:
    alpha = float(alpha_arr[0])
    logits = alpha + logit_p
    # Numerically stable log-loss.
    loss = -np.mean(yt * logits - np.logaddexp(0.0, logits))
    return float(loss)

  result = minimize(
    neg_log_likelihood,
    x0=np.asarray([0.0]),
    method="BFGS",
    options={"gtol": 1e-10},
  )
  return {"alpha": float(result.x[0]), "beta": 1.0, "converged": bool(result.success)}


def fit_r2_intercept_and_slope(
  y: npt.NDArray[np.integer],
  p: npt.NDArray[np.floating],
  epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float]:
  """Fit ``logit(Y) = alpha + beta * logit(p)``; no predictor beyond the transported logit."""
  yt = np.asarray(y, dtype=np.int64)
  if len(np.unique(yt)) < 2:
    raise ValueError("R2 requires both outcome classes in the calibration-fit split.")
  logit_p = _clip_logit(p, epsilon).reshape(-1, 1)
  model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000, fit_intercept=True)
  model.fit(logit_p, yt)
  return {
    "alpha": float(np.asarray(model.intercept_).ravel()[0]),
    "beta": float(np.asarray(model.coef_).ravel()[0]),
  }


def apply_recalibration(
  p: npt.NDArray[np.floating],
  params: dict[str, float],
  epsilon: float = DEFAULT_EPSILON,
) -> npt.NDArray[np.float64]:
  """Apply a frozen R1/R2 map; predictions stay in [0, 1] by construction."""
  logits = params["alpha"] + params["beta"] * _clip_logit(p, epsilon)
  return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float64)


def observed_to_expected(y: npt.NDArray[np.integer], p: npt.NDArray[np.floating]) -> float:
  """O/E ratio: observed event fraction over mean predicted risk."""
  yt = np.asarray(y, dtype=np.float64)
  return float(yt.mean() / np.asarray(p, dtype=np.float64).mean())


def decile_calibration(
  y: npt.NDArray[np.integer],
  p: npt.NDArray[np.floating],
) -> list[dict[str, float]]:
  """Observed vs mean predicted risk by decile of the predicted score."""
  yt = np.asarray(y, dtype=np.float64)
  order = np.argsort(np.asarray(p, dtype=np.float64))
  deciles = np.array_split(order, 10)
  rows: list[dict[str, float]] = []
  for i, idx in enumerate(deciles, start=1):
    rows.append(
      {
        "decile": i,
        "n": int(len(idx)),
        "observed_rate": float(yt[idx].mean()),
        "mean_predicted": float(np.asarray(p, dtype=np.float64)[idx].mean()),
      }
    )
  return rows


def paired_bootstrap_score_deltas(
  y: npt.NDArray[np.integer],
  p_frozen: npt.NDArray[np.floating],
  p_recal: npt.NDArray[np.floating],
  n_bootstraps: int = 1000,
  seed: int = 1912,
) -> dict[str, Any]:
  """Paired patient bootstrap for Brier/log-loss changes (recalibrated - frozen).

  The validation-fitted calibration map is held fixed inside every resample; no
  recalibration parameter is refit on locked-test data.
  """
  yt = np.asarray(y, dtype=np.int64)
  pf = np.asarray(p_frozen, dtype=np.float64)
  pr = np.asarray(p_recal, dtype=np.float64)
  rng = np.random.default_rng(seed)
  n = len(yt)
  deltas: dict[str, list[float]] = {"brier": [], "log_loss": []}
  for _ in range(n_bootstraps):
    idx = rng.integers(0, n, size=n)
    y_b = yt[idx]
    deltas["brier"].append(float(np.mean((pr[idx] - y_b) ** 2) - np.mean((pf[idx] - y_b) ** 2)))
    deltas["log_loss"].append(
      float(
        log_loss(y_b, pr[idx], labels=[0, 1])
        - log_loss(y_b, pf[idx], labels=[0, 1])
      )
    )

  def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
      "mean": float(arr.mean()),
      "lo95": float(np.percentile(arr, 2.5)),
      "hi95": float(np.percentile(arr, 97.5)),
    }

  return {
    "delta_brier_recal_minus_frozen": _summary(deltas["brier"]),
    "delta_log_loss_recal_minus_frozen": _summary(deltas["log_loss"]),
    "n_bootstraps": n_bootstraps,
    "seed": seed,
    "calibration_map": "fixed from CODE validation; never refit inside the bootstrap",
  }


def val_refit_sensitivity(
  y_val: npt.NDArray[np.integer],
  p_val: npt.NDArray[np.floating],
  y_test: npt.NDArray[np.integer],
  p_test: npt.NDArray[np.floating],
  config: RecalibrationConfig,
) -> dict[str, Any]:
  """Optional separate sensitivity: bootstrap the CODE validation set, refit R1/R2,
  and evaluate each refit map on the fixed locked test to expose map uncertainty."""
  rng = np.random.default_rng(config.seed)
  n = len(y_val)
  r1_slopes_brier: list[float] = []
  r2_slopes_brier: list[float] = []
  r2_betas: list[float] = []
  for _ in range(config.n_val_refit_sensitivity):
    idx = rng.integers(0, n, size=n)
    if len(np.unique(y_val[idx])) < 2:
      continue
    r1 = fit_r1_intercept_only(y_val[idx], p_val[idx], config.epsilon)
    r2 = fit_r2_intercept_and_slope(y_val[idx], p_val[idx], config.epsilon)
    r2_betas.append(r2["beta"])
    r1_slopes_brier.append(
      float(np.mean((apply_recalibration(p_test, r1, config.epsilon) - y_test) ** 2))
    )
    r2_slopes_brier.append(
      float(np.mean((apply_recalibration(p_test, r2, config.epsilon) - y_test) ** 2))
    )

  def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "lo95": float(np.percentile(arr, 2.5)), "hi95": float(np.percentile(arr, 97.5))}

  return {
    "n_replicates": len(r2_betas),
    "r2_beta_distribution": _summary(r2_betas),
    "r1_test_brier_distribution": _summary(r1_slopes_brier),
    "r2_test_brier_distribution": _summary(r2_slopes_brier),
    "note": "separate from the primary result; the primary map is the single validation fit",
  }


def _score_block(
  y: npt.NDArray[np.integer],
  p_frozen: npt.NDArray[np.floating],
  p_r1: npt.NDArray[np.floating],
  p_r2: npt.NDArray[np.floating],
) -> dict[str, Any]:
  """Discrimination/calibration block for one score under R0/R1/R2 on the locked test."""
  auroc_frozen = float(roc_auc_score(y, p_frozen))
  auroc_r1 = float(roc_auc_score(y, p_r1))
  auroc_r2 = float(roc_auc_score(y, p_r2))
  auprc_frozen = float(average_precision_score(y, p_frozen))
  return {
    "R0_frozen": {
      "auroc": auroc_frozen,
      "auprc": auprc_frozen,
      "brier": float(np.mean((p_frozen - y) ** 2)),
      "log_loss": float(log_loss(y, p_frozen, labels=[0, 1])),
      **calibration_intercept_slope(y, p_frozen),
      "oe_ratio": observed_to_expected(y, p_frozen),
    },
    "R1_intercept": {
      "auroc": auroc_r1,
      "auprc": float(average_precision_score(y, p_r1)),
      "brier": float(np.mean((p_r1 - y) ** 2)),
      "log_loss": float(log_loss(y, p_r1, labels=[0, 1])),
      **calibration_intercept_slope(y, p_r1),
      "oe_ratio": observed_to_expected(y, p_r1),
    },
    "R2_intercept_slope": {
      "auroc": auroc_r2,
      "auprc": float(average_precision_score(y, p_r2)),
      "brier": float(np.mean((p_r2 - y) ** 2)),
      "log_loss": float(log_loss(y, p_r2, labels=[0, 1])),
      **calibration_intercept_slope(y, p_r2),
      "oe_ratio": observed_to_expected(y, p_r2),
    },
    "invariance_check": {
      "auroc_abs_diff_r1": abs(auroc_r1 - auroc_frozen),
      "auroc_abs_diff_r2": abs(auroc_r2 - auroc_frozen),
      "tolerance": 1e-4,
      "note": (
        "AUROC must be unchanged under monotone recalibration when beta > 0. The "
        "logit/expit round trip can flip a handful of near-ties at the last ulp, so "
        "the tolerance accommodates float noise on ~45k heavily tied scores; map "
        "monotonicity is verified separately."
      ),
    },
  }


def run_code15_recalibration(
  features_dir: Path,
  cohort_path: Path,
  model_dir: Path,
  split_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  config: RecalibrationConfig | None = None,
  n_bootstraps: int = 1000,
  with_val_refit_sensitivity: bool = True,
) -> dict[str, Any]:
  """LM4 frozen evaluation: fit R1/R2 on the CODE validation split (1-year endpoint),
  apply to the untouched CODE locked test once, and report calibration outcomes for
  both transported scores A and B."""
  cfg = config or RecalibrationConfig()
  features_a = np.load(features_dir / "ciis_features.npy")
  features_b = np.load(features_dir / "dbeta_embeddings.npy")
  meta = pl.read_parquet(features_dir / "extracted_meta.parquet").sort("exam_id")
  if not (len(meta) == len(features_a) == len(features_b)):
    raise ValueError("Feature/meta row mismatch in the CODE extraction directory.")

  probes = load_frozen_probes(Path(model_dir))
  p_a_all, p_b_all = transport_scores(probes, features_a, features_b)

  cohort = pl.read_parquet(cohort_path)
  label_col = f"y_{cfg.horizon}"
  if label_col not in cohort.columns:
    raise ValueError(f"Cohort parquet missing label column {label_col}.")
  from ecg_rep.external.code15_labels import horizon_label_to_int

  cohort = horizon_label_to_int(cohort, cfg.horizon)
  merged = meta.select(["exam_id", "patient_id"]).join(
    cohort.select(["exam_id", "patient_id", label_col]),
    on=["exam_id", "patient_id"],
    how="left",
    validate="1:1",
  )
  split = pl.read_csv(split_path)
  split_map = dict(zip(split["patient_id"].to_list(), split["split"].to_list(), strict=True))
  pids = merged["patient_id"].to_list()
  labels = merged[label_col].to_list()

  val_mask = np.asarray(
    [split_map.get(pid) == cfg.fit_split and v is not None for pid, v in zip(pids, labels, strict=True)],
    dtype=bool,
  )
  test_mask = np.asarray(
    [split_map.get(pid) == cfg.eval_split and v is not None for pid, v in zip(pids, labels, strict=True)],
    dtype=bool,
  )
  y_val = np.asarray([v for v, keep in zip(labels, val_mask, strict=True) if keep], dtype=np.int64)
  y_test = np.asarray([v for v, keep in zip(labels, test_mask, strict=True) if keep], dtype=np.int64)
  val_idx = np.flatnonzero(val_mask)
  test_idx = np.flatnonzero(test_mask)

  results: dict[str, Any] = {
    "status": "complete",
    "post_hoc": True,
    "analysis_status": "post hoc extension",
    "config": cfg.to_dict(),
    "source_sha": source_sha,
    "endpoint": "1-year mortality (different-horizon association; never 30-day external validation)",
    "calibration_fit_split": cfg.fit_split,
    "evaluation_split": cfg.eval_split,
    "n_val": int(val_mask.sum()),
    "n_val_events": int(y_val.sum()),
    "n_test": int(test_mask.sum()),
    "n_test_events": int(y_test.sum()),
    "scores": {},
  }

  test_predictions: dict[str, npt.NDArray[np.float64]] = {}
  for score_name, p_all in (("A", p_a_all), ("B", p_b_all)):
    p_val = p_all[val_idx]
    p_test = p_all[test_idx]
    r1 = fit_r1_intercept_only(y_val, p_val, cfg.epsilon)
    r2 = fit_r2_intercept_and_slope(y_val, p_val, cfg.epsilon)
    p_r1 = apply_recalibration(p_test, r1, cfg.epsilon)
    p_r2 = apply_recalibration(p_test, r2, cfg.epsilon)
    block: dict[str, Any] = {
      "params": {"R1": r1, "R2": r2},
      "locked_test": _score_block(y_test, p_test, p_r1, p_r2),
      "decile_calibration": {
        "R0_frozen": decile_calibration(y_test, p_test),
        "R2_intercept_slope": decile_calibration(y_test, p_r2),
      },
      "paired_bootstrap_deltas": {
        "R1_minus_frozen": paired_bootstrap_score_deltas(y_test, p_test, p_r1, n_bootstraps=n_bootstraps, seed=cfg.seed),
        "R2_minus_frozen": paired_bootstrap_score_deltas(y_test, p_test, p_r2, n_bootstraps=n_bootstraps, seed=cfg.seed),
      },
    }
    if with_val_refit_sensitivity:
      block["val_refit_sensitivity"] = val_refit_sensitivity(y_val, p_val, y_test, p_test, cfg)
    # Invariance guard: monotone recalibration must not change AUROC when beta > 0.
    # Tolerance covers logit/expit round-trip noise on heavily tied scores.
    unique_sorted_p = np.unique(p_test)
    for variant, params in (("R1", r1), ("R2", r2)):
      mapped = apply_recalibration(unique_sorted_p, params, cfg.epsilon)
      if params["beta"] > 0 and np.any(np.diff(mapped) < -1e-12):
        raise RuntimeError(f"{score_name} {variant}: applied map is not monotone; implementation error.")
      diff = block["locked_test"]["invariance_check"][f"auroc_abs_diff_{variant.lower()}"]
      if params["beta"] > 0 and diff > 1e-4:
        raise RuntimeError(f"{score_name} {variant}: AUROC invariance failed (diff={diff}); implementation error.")
    results["scores"][score_name] = block
    test_predictions[f"p_{score_name}_frozen"] = p_test
    test_predictions[f"p_{score_name}_r1"] = p_r1
    test_predictions[f"p_{score_name}_r2"] = p_r2

  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "recalibration_v1.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  pred_frame = pl.DataFrame({
    "patient_id": [pids[i] for i in test_idx.tolist()],
    "exam_id": [merged["exam_id"][i] for i in test_idx.tolist()],
    "y_365d": y_test.tolist(),
    **{k: v.tolist() for k, v in test_predictions.items()},
  })
  pred_frame.write_parquet(output_dir / "recalibration_test_predictions.parquet")

  tables_dir = report_path.parent.parent / "manuscript" / "tables"
  tables_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for score_name, block in results["scores"].items():
    for variant in ("R0_frozen", "R1_intercept", "R2_intercept_slope"):
      m = block["locked_test"][variant]
      rows.append({
        "score": score_name,
        "variant": variant,
        "auroc": m["auroc"],
        "brier": m["brier"],
        "log_loss": m["log_loss"],
        "calibration_intercept": m["intercept"],
        "calibration_slope": m["slope"],
        "oe_ratio": m["oe_ratio"],
        "n_test": results["n_test"],
        "n_test_events": results["n_test_events"],
      })
  pl.DataFrame(rows).write_csv(tables_dir / "supp_code15_recalibration.csv")
  _render_lm4_report(report_path, results, source_sha)
  return results


def _render_lm4_report(report_path: Path, results: dict[str, Any], source_sha: str) -> None:
  lines = [
    "# CODE-15% local recalibration of transported frozen MIMIC scores (LM4)",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {source_sha}",
    "analysis_status: post_hoc_extension",
    "inputs: [artifacts/external/code15pct/features, artifacts/external/code15pct/cohort, artifacts/probe/models]",
    "decisions: [R1/R2 fitted on the CODE validation split only; locked test evaluated once]",
    "evidence: [artifacts/external/code15/recalibration_v1.json, artifacts/external/code15/recalibration_test_predictions.parquet]",
    "unresolved: []",
    "verification: [AUROC invariance under monotone maps, fixed-map bootstrap, val-refit sensitivity]",
    "next_action: fold into C12/C11 claim wording at LM9 reconciliation",
    "```",
    "",
    f"Endpoint: 1-year mortality — a different-horizon association; this is never",
    "same-endpoint external validation of the MIMIC 30-day model. Calibration maps are",
    f"fitted on the CODE **{results['calibration_fit_split']}** split "
    f"(n={results['n_val']:,}, {results['n_val_events']} events) and evaluated once on the untouched",
    f"CODE locked test (n={results['n_test']:,}, {results['n_test_events']} events).",
    "",
  ]
  for score_name, block in results["scores"].items():
    lines.append(f"## Score {score_name}")
    lines.append("")
    lines.append("| Variant | AUROC | Brier | Log loss | Calib. intercept | Calib. slope | O/E |")
    lines.append("|---|---|---|---|---|---|---|")
    for variant in ("R0_frozen", "R1_intercept", "R2_intercept_slope"):
      m = block["locked_test"][variant]
      lines.append(
        f"| {variant} | {m['auroc']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} | "
        f"{m['intercept']:+.3f} | {m['slope']:.3f} | {m['oe_ratio']:.3f} |"
      )
    for variant in ("R1_minus_frozen", "R2_minus_frozen"):
      d = block["paired_bootstrap_deltas"][variant]
      b = d["delta_brier_recal_minus_frozen"]
      lines.append("")
      lines.append(
        f"{variant}: ΔBrier {b['mean']:+.4f} ({b['lo95']:+.4f} to {b['hi95']:+.4f}); "
        f"Δlog-loss {d['delta_log_loss_recal_minus_frozen']['mean']:+.4f} "
        f"({d['delta_log_loss_recal_minus_frozen']['lo95']:+.4f} to "
        f"{d['delta_log_loss_recal_minus_frozen']['hi95']:+.4f}); paired patient bootstrap, fixed map."
      )
    sensitivity = block.get("val_refit_sensitivity")
    if sensitivity:
      lines += [
        "",
        f"### Validation-refit sensitivity ({sensitivity['n_replicates']} refits)",
        "",
        "CODE validation patients were bootstrapped and R1/R2 refit; the test sample",
        "remained fixed. These are empirical central 95% distributions of map-fitting",
        "variation, not paired test-bootstrap CIs or combined sources of uncertainty.",
        "",
        "| Quantity | Mean | Central 95% |",
        "|---|---|---|",
      ]
      for key, label in (("r2_beta_distribution", "R2 beta"),
                         ("r1_test_brier_distribution", "R1 test Brier"),
                         ("r2_test_brier_distribution", "R2 test Brier")):
        value = sensitivity[key]
        lines.append(f"| {label} | {value['mean']:.6f} | {value['lo95']:.6f} to {value['hi95']:.6f} |")
    else:
      lines += ["", "Validation-refit sensitivity not available in this artifact."]
    lines.append("")
  lines += [
    "Interpretation is bounded by the frozen rules in the LM4 plan: R1 improvements mean a",
    "local baseline-risk offset; R2 beyond R1 means score scale differed; failure of both",
    "supports the need for more extensive local model updating. Recalibration on 1-year",
    "mortality is not external validation.",
    "",
  ]
  report_path.write_text("\n".join(lines), encoding="utf-8")
