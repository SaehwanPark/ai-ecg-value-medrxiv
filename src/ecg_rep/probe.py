"""Ridge-logistic probes with development-only preprocessing and validation tuning.

The shared learner isolates representation information under the research-plan contract:
development-set standardization, validation log-loss C selection, locked test evaluation.
Clinical context ``C0`` is allowed when constructed under the cohort availability ledger.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
import numpy.typing as npt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

DEFAULT_C_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
DEFAULT_PROBE_SEED: int = 42
DEFAULT_PROBE_MAX_ITER: int = 2000
DEFAULT_BOOTSTRAP_SAMPLES: int = 1000


@dataclass(frozen=True)
class ProbeConfig:
  """Immutable ridge-logistic probe configuration."""

  regularization_c_grid: tuple[float, ...] = DEFAULT_C_GRID
  max_iter: int = DEFAULT_PROBE_MAX_ITER
  solver: str = "lbfgs"
  random_state: int = DEFAULT_PROBE_SEED
  standardize: bool = True
  penalty: Literal["l2"] = "l2"


@dataclass(frozen=True)
class ProbeValidationStep:
  """Validation metrics for one candidate regularization strength."""

  c_value: float
  val_log_loss: float
  val_auroc: float
  val_brier: float


@dataclass(frozen=True)
class TrainedProbe:
  """Frozen coefficients and preprocessing learned on development data only."""

  coefficients: tuple[float, ...]
  intercept: float
  best_c: float
  tuning_history: tuple[ProbeValidationStep, ...]
  feature_means: tuple[float, ...] | None
  feature_scales: tuple[float, ...] | None
  standardize: bool
  feature_names: tuple[str, ...] | None = None
  model_name: str | None = None

  def transform(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Apply frozen development-only standardization when enabled."""
    arr = np.asarray(x, dtype=np.float64)
    if not self.standardize or self.feature_means is None or self.feature_scales is None:
      return arr
    means = np.asarray(self.feature_means, dtype=np.float64)
    scales = np.asarray(self.feature_scales, dtype=np.float64)
    return (arr - means) / scales

  def predict_proba(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return positive-class probabilities for rows of ``x``."""
    xt = self.transform(x)
    coef = np.asarray(self.coefficients, dtype=np.float64)
    logits = xt @ coef + self.intercept
    return np.asarray(expit(logits), dtype=np.float64)

  def predict_log_odds(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return linear predictor values (log-odds) for rows of ``x``."""
    xt = self.transform(x)
    coef = np.asarray(self.coefficients, dtype=np.float64)
    return xt @ coef + self.intercept

  def predict_block_contribution(
    self,
    x: npt.NDArray[np.float64],
    start_idx: int,
    end_idx: int | None = None,
  ) -> npt.NDArray[np.float64]:
    """Return linear contribution beta_block^T * \\tilde{z}_block for a subset of features.

    Specifically for context-adjusted probes (Y ~ C + Z), isolates S = beta_Z^T * \\tilde{Z}
    to separate representation-specific contribution from shared clinical context.
    """
    xt = self.transform(x)
    end = end_idx if end_idx is not None else xt.shape[1]
    xt_block = xt[:, start_idx:end]
    coef = np.asarray(self.coefficients[start_idx:end], dtype=np.float64)
    return xt_block @ coef

  def to_dict(self) -> dict[str, Any]:
    """Serialize the probe parameters and metadata into a JSON-compatible dictionary."""
    return {
      "model_name": self.model_name,
      "feature_names": list(self.feature_names) if self.feature_names is not None else None,
      "coefficients": list(self.coefficients),
      "intercept": float(self.intercept),
      "best_c": float(self.best_c),
      "tuning_history": [
        {
          "c_value": float(step.c_value),
          "val_log_loss": float(step.val_log_loss),
          "val_auroc": float(step.val_auroc),
          "val_brier": float(step.val_brier),
        }
        for step in self.tuning_history
      ],
      "feature_means": list(self.feature_means) if self.feature_means is not None else None,
      "feature_scales": list(self.feature_scales) if self.feature_scales is not None else None,
      "standardize": bool(self.standardize),
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> TrainedProbe:
    """Reconstruct a TrainedProbe from a serialized dictionary."""
    history = tuple(
      ProbeValidationStep(
        c_value=float(s["c_value"]),
        val_log_loss=float(s["val_log_loss"]),
        val_auroc=float(s["val_auroc"]),
        val_brier=float(s["val_brier"]),
      )
      for s in data.get("tuning_history", [])
    )
    means = tuple(float(v) for v in data["feature_means"]) if data.get("feature_means") is not None else None
    scales = tuple(float(v) for v in data["feature_scales"]) if data.get("feature_scales") is not None else None
    feat_names = tuple(str(v) for v in data["feature_names"]) if data.get("feature_names") is not None else None
    return cls(
      coefficients=tuple(float(v) for v in data["coefficients"]),
      intercept=float(data["intercept"]),
      best_c=float(data["best_c"]),
      tuning_history=history,
      feature_means=means,
      feature_scales=scales,
      standardize=bool(data.get("standardize", True)),
      feature_names=feat_names,
      model_name=data.get("model_name"),
    )

  def save(self, path: Path | str, format: Literal["auto", "json", "joblib"] = "auto") -> None:
    """Save the trained probe to a JSON file or Joblib pickle archive."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if format == "joblib" or (format == "auto" and ext in {".joblib", ".pkl"}):
      import joblib

      joblib.dump(self, p)
    else:
      p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

  @classmethod
  def load(cls, path: Path | str) -> TrainedProbe:
    """Load a trained probe from a JSON file or Joblib archive."""
    p = Path(path)
    if not p.exists():
      raise FileNotFoundError(f"Trained probe artifact not found at '{p}'.")
    if p.suffix.lower() in {".joblib", ".pkl"}:
      import joblib

      obj = joblib.load(p)
      if isinstance(obj, cls):
        return obj
      if isinstance(obj, dict):
        return cls.from_dict(obj)
      raise TypeError(f"Loaded object from '{p}' is not a TrainedProbe instance (got {type(obj)}).")
    try:
      data = json.loads(p.read_text(encoding="utf-8"))
      return cls.from_dict(data)
    except Exception:
      import joblib

      obj = joblib.load(p)
      if isinstance(obj, cls):
        return obj
      raise


def save_probe_model(
  probe: TrainedProbe,
  output_dir: Path | str,
  model_name: str,
) -> tuple[Path, Path]:
  """Persist a trained probe model as both JSON and Joblib artifacts under output_dir/models."""
  models_dir = Path(output_dir) / "models"
  models_dir.mkdir(parents=True, exist_ok=True)
  json_path = models_dir / f"{model_name}.json"
  joblib_path = models_dir / f"{model_name}.joblib"
  probe.save(json_path, format="json")
  probe.save(joblib_path, format="joblib")
  return json_path, joblib_path


@dataclass(frozen=True)
class MetricBundle:
  """Discrimination and calibration summary for one evaluation split."""

  auroc: float
  auprc: float
  log_loss: float
  brier: float
  calibration_intercept: float
  calibration_slope: float
  n: int
  n_events: int


@dataclass(frozen=True)
class BootstrapDifference:
  """Patient-level bootstrap interval for a paired metric difference."""

  point_estimate: float
  ci_lower: float
  ci_upper: float
  n_bootstraps: int
  seed: int


def compute_binary_log_loss(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
  eps: float = 1e-15,
) -> float:
  """Mean binary cross-entropy with probability clipping."""
  yt = np.asarray(y_true, dtype=np.float64)
  yp = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
  if len(yt) == 0:
    return 0.0
  return float(-np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp)))


def compute_auroc(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_score: npt.NDArray[np.float64] | Sequence[float],
) -> float:
  """AUROC; returns 0.5 when a split lacks both classes."""
  yt = np.asarray(y_true, dtype=np.int64)
  ys = np.asarray(y_score, dtype=np.float64)
  if len(yt) == 0 or len(np.unique(yt)) < 2:
    return 0.5
  return float(roc_auc_score(yt, ys))


def compute_auprc(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_score: npt.NDArray[np.float64] | Sequence[float],
) -> float:
  """Average precision (AUPRC); returns prevalence when only one class exists."""
  yt = np.asarray(y_true, dtype=np.int64)
  ys = np.asarray(y_score, dtype=np.float64)
  if len(yt) == 0:
    return 0.0
  if len(np.unique(yt)) < 2:
    return float(np.mean(yt))
  return float(average_precision_score(yt, ys))


def compute_brier_score(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
) -> float:
  """Brier score for probabilistic predictions."""
  yt = np.asarray(y_true, dtype=np.int64)
  yp = np.asarray(y_prob, dtype=np.float64)
  if len(yt) == 0:
    return 0.0
  return float(brier_score_loss(yt, yp))


def fit_calibration_line(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
  eps: float = 1e-15,
) -> tuple[float, float]:
  """Fit logit(y) ~ intercept + slope * logit(p) on an evaluation split.

  Returns ``(intercept, slope)``. Perfect calibration is intercept 0, slope 1.
  """
  yt = np.asarray(y_true, dtype=np.float64)
  yp = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
  if len(yt) < 2 or len(np.unique(yt)) < 2:
    return 0.0, 1.0
  logit_p = np.log(yp / (1.0 - yp)).reshape(-1, 1)
  model = LogisticRegression(
    C=1e12,
    solver="lbfgs",
    max_iter=1000,
    fit_intercept=True,
  )
  model.fit(logit_p, yt.astype(np.int64))
  intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])
  slope = float(np.asarray(model.coef_, dtype=np.float64).reshape(-1)[0])
  return intercept, slope


def evaluate_predictions(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
) -> MetricBundle:
  """Compute the primary metric bundle for one prediction vector."""
  yt = np.asarray(y_true, dtype=np.int64)
  yp = np.asarray(y_prob, dtype=np.float64)
  intercept, slope = fit_calibration_line(yt, yp)
  return MetricBundle(
    auroc=compute_auroc(yt, yp),
    auprc=compute_auprc(yt, yp),
    log_loss=compute_binary_log_loss(yt, yp),
    brier=compute_brier_score(yt, yp),
    calibration_intercept=intercept,
    calibration_slope=slope,
    n=int(len(yt)),
    n_events=int(yt.sum()),
  )


def fit_logistic_probe(
  x_dev: npt.NDArray[np.float64],
  y_dev: npt.NDArray[np.int64],
  x_val: npt.NDArray[np.float64],
  y_val: npt.NDArray[np.int64],
  config: ProbeConfig | None = None,
  feature_names: Sequence[str] | None = None,
  model_name: str | None = None,
) -> TrainedProbe:
  """Fit ridge-logistic probe; tune C on validation log-loss only.

  Standardization statistics are learned on ``x_dev`` alone. Final-test matrices
  must not be passed here.
  """
  cfg = config or ProbeConfig()
  x_dev_arr = np.asarray(x_dev, dtype=np.float64)
  x_val_arr = np.asarray(x_val, dtype=np.float64)
  y_dev_arr = np.asarray(y_dev, dtype=np.int64)
  y_val_arr = np.asarray(y_val, dtype=np.int64)

  means: tuple[float, ...] | None = None
  scales: tuple[float, ...] | None = None
  if cfg.standardize:
    scaler = StandardScaler()
    x_dev_fit = np.asarray(scaler.fit_transform(x_dev_arr), dtype=np.float64)
    x_val_fit = np.asarray(scaler.transform(x_val_arr), dtype=np.float64)
    mean_arr = np.asarray(scaler.mean_, dtype=np.float64).reshape(-1)
    scale_arr = np.asarray(scaler.scale_, dtype=np.float64).reshape(-1)
    means = tuple(float(v) for v in mean_arr)
    scales = tuple(float(v) for v in scale_arr)
  else:
    x_dev_fit = x_dev_arr
    x_val_fit = x_val_arr

  history: list[ProbeValidationStep] = []
  best_c = cfg.regularization_c_grid[0]
  best_loss = float("inf")
  best_coef: npt.NDArray[np.float64] | None = None
  best_intercept = 0.0

  for c_value in cfg.regularization_c_grid:
    model = LogisticRegression(
      C=c_value,
      solver=cfg.solver,
      max_iter=cfg.max_iter,
      random_state=cfg.random_state,
    )
    model.fit(x_dev_fit, y_dev_arr)
    val_prob = np.asarray(model.predict_proba(x_val_fit)[:, 1], dtype=np.float64)
    val_loss = compute_binary_log_loss(y_val_arr, val_prob)
    step = ProbeValidationStep(
      c_value=float(c_value),
      val_log_loss=val_loss,
      val_auroc=compute_auroc(y_val_arr, val_prob),
      val_brier=compute_brier_score(y_val_arr, val_prob),
    )
    history.append(step)
    if val_loss < best_loss:
      best_loss = val_loss
      best_c = float(c_value)
      best_coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
      best_intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])

  if best_coef is None:
    raise RuntimeError("Probe fitting failed to produce coefficients.")

  return TrainedProbe(
    coefficients=tuple(float(v) for v in best_coef),
    intercept=best_intercept,
    best_c=best_c,
    tuning_history=tuple(history),
    feature_means=means,
    feature_scales=scales,
    standardize=cfg.standardize,
    feature_names=tuple(feature_names) if feature_names is not None else None,
    model_name=model_name,
  )


def bootstrap_paired_difference(
  y_true: npt.NDArray[np.int64],
  score_a: npt.NDArray[np.float64],
  score_b: npt.NDArray[np.float64],
  metric: Literal["auroc", "auprc", "log_loss", "brier"] = "auroc",
  n_bootstraps: int = DEFAULT_BOOTSTRAP_SAMPLES,
  seed: int = DEFAULT_PROBE_SEED,
  stratified: bool = True,
) -> BootstrapDifference:
  """Patient-level bootstrap CI for metric(B) - metric(A) on the same patients.

  With ``stratified=True`` (the study default) each resample draws the
  observed number of deaths and survivors separately, so intervals are
  conditional on the locked-test event count. With ``stratified=False``
  patients are resampled uniformly (ordinary patient bootstrap), which also
  resamples the event count.
  """
  yt = np.asarray(y_true, dtype=np.int64)
  sa = np.asarray(score_a, dtype=np.float64)
  sb = np.asarray(score_b, dtype=np.float64)
  if not (len(yt) == len(sa) == len(sb)):
    raise ValueError("Paired bootstrap requires equal-length arrays.")

  metric_fn = {
    "auroc": compute_auroc,
    "auprc": compute_auprc,
    "log_loss": compute_binary_log_loss,
    "brier": compute_brier_score,
  }[metric]

  point = float(metric_fn(yt, sb) - metric_fn(yt, sa))
  rng = np.random.default_rng(seed)
  diffs: list[float] = []
  pos_idx = np.where(yt == 1)[0]
  neg_idx = np.where(yt == 0)[0]
  n_pos = len(pos_idx)
  n_neg = len(neg_idx)
  n = len(yt)

  if not stratified:
    for _ in range(n_bootstraps):
      idx = rng.integers(0, n, size=n)
      diffs.append(float(metric_fn(yt[idx], sb[idx]) - metric_fn(yt[idx], sa[idx])))
  elif n_pos == 0 or n_neg == 0:
    for _ in range(n_bootstraps):
      idx = rng.integers(0, n, size=n)
      diffs.append(float(metric_fn(yt[idx], sb[idx]) - metric_fn(yt[idx], sa[idx])))
  else:
    for _ in range(n_bootstraps):
      sample_pos = rng.choice(pos_idx, size=n_pos, replace=True)
      sample_neg = rng.choice(neg_idx, size=n_neg, replace=True)
      idx = np.concatenate([sample_pos, sample_neg])
      diffs.append(float(metric_fn(yt[idx], sb[idx]) - metric_fn(yt[idx], sa[idx])))

  lo, hi = np.quantile(np.asarray(diffs, dtype=np.float64), [0.025, 0.975])
  return BootstrapDifference(
    point_estimate=point,
    ci_lower=float(lo),
    ci_upper=float(hi),
    n_bootstraps=n_bootstraps,
    seed=seed,
  )


def stack_feature_blocks(*blocks: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  """Horizontally concatenate feature blocks for ``C``, ``C+A``, ``C+B``, ``C+A+B``."""
  if not blocks:
    raise ValueError("At least one feature block is required.")
  return np.concatenate([np.asarray(block, dtype=np.float64) for block in blocks], axis=1)


def metric_bundle_to_dict(bundle: MetricBundle) -> dict[str, Any]:
  """Serialize a metric bundle for JSON artifacts."""
  return {
    "auroc": bundle.auroc,
    "auprc": bundle.auprc,
    "log_loss": bundle.log_loss,
    "brier": bundle.brier,
    "calibration_intercept": bundle.calibration_intercept,
    "calibration_slope": bundle.calibration_slope,
    "n": bundle.n,
    "n_events": bundle.n_events,
  }
