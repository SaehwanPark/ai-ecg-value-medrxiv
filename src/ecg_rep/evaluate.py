"""Fit and evaluate the locked ridge-logistic model matrix on study artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.decomposition import PCA

from ecg_rep.analysis import representation_gain
from ecg_rep.elixhauser import ELIXHAUSER_FEATURE_NAMES
from ecg_rep.probe import (
  ProbeConfig,
  TrainedProbe,
  bootstrap_paired_difference,
  evaluate_predictions,
  fit_logistic_probe,
  metric_bundle_to_dict,
)
from ecg_rep.representations import verify_representation_artifact_integrity

C0_FEATURE_COLS: tuple[str, ...] = ("age_at_ecg", "sex_female", *ELIXHAUSER_FEATURE_NAMES)


@dataclass(frozen=True)
class SplitMatrices:
  """Feature/label matrices for one model on one split."""

  x: npt.NDArray[np.float64]
  y: npt.NDArray[np.int64]
  subject_ids: npt.NDArray[np.int64]
  study_ids: npt.NDArray[np.int64]


def _split_xy(
  frame: pl.DataFrame,
  feature_cols: list[str],
  split: str,
) -> SplitMatrices:
  sub = frame.filter(pl.col("split") == split)
  x = sub.select(feature_cols).to_numpy().astype(np.float64)
  y_arr = np.asarray(sub.get_column("mortality_30d").to_numpy(), dtype=np.int64)
  ids = np.asarray(sub.get_column("subject_id").to_numpy(), dtype=np.int64)
  study_ids = np.asarray(sub.get_column("study_id").to_numpy(), dtype=np.int64)
  return SplitMatrices(x=x, y=y_arr, subject_ids=ids, study_ids=study_ids)


def _fit_probe(
  frame: pl.DataFrame,
  feature_cols: list[str],
  model_name: str | None = None,
  config: ProbeConfig | None = None,
) -> tuple[TrainedProbe, SplitMatrices, SplitMatrices, SplitMatrices]:
  cfg = config or ProbeConfig()
  dev = _split_xy(frame, feature_cols, "dev")
  val = _split_xy(frame, feature_cols, "val")
  test = _split_xy(frame, feature_cols, "test")
  probe = fit_logistic_probe(
    dev.x, dev.y, val.x, val.y, config=cfg, feature_names=feature_cols, model_name=model_name
  )
  return probe, dev, val, test


def fit_and_eval_model(
  frame: pl.DataFrame,
  feature_cols: list[str],
  model_name: str,
  config: ProbeConfig | None = None,
  models_dir: Path | None = None,
) -> tuple[dict[str, Any], TrainedProbe]:
  """Tune on validation, freeze, evaluate locked test set, optionally saving model artifacts."""
  probe, _dev, _val, test = _fit_probe(frame, feature_cols, model_name=model_name, config=config)
  test_prob = probe.predict_proba(test.x)
  metrics = evaluate_predictions(test.y, test_prob)
  res: dict[str, Any] = {
    "model": model_name,
    "n_features": len(feature_cols),
    "best_c": probe.best_c,
    "test_metrics": metric_bundle_to_dict(metrics),
  }
  if models_dir is not None:
    models_dir.mkdir(parents=True, exist_ok=True)
    json_path = models_dir / f"{model_name}.json"
    joblib_path = models_dir / f"{model_name}.joblib"
    probe.save(json_path, format="json")
    probe.save(joblib_path, format="joblib")
    res["model_artifact_json"] = str(json_path)
    res["model_artifact_joblib"] = str(joblib_path)
  return res, probe


def _join_representation_matrix(
  cohort: pl.DataFrame,
  meta: pl.DataFrame,
  matrix: npt.NDArray[np.float64],
  prefix: str,
  valid_col: str,
) -> pl.DataFrame:
  if matrix.ndim != 2 or matrix.shape[0] != len(meta):
    raise ValueError(f"{prefix} matrix shape incompatible with meta rows.")
  cols = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
  feat = pl.DataFrame(matrix, schema=cols).with_columns(
    meta.select(["subject_id", "study_id", valid_col])
  )
  return cohort.join(feat, on=["subject_id", "study_id"], how="inner")


def run_ecg_only_evaluation(
  cohort_path: Path,
  meta_a_path: Path,
  features_a_path: Path,
  meta_b_path: Path | None = None,
  embeddings_b_path: Path | None = None,
  output_dir: Path | None = None,
  n_bootstraps: int = 200,
  verify_provenance: bool = False,
) -> dict[str, Any]:
  """Evaluate ECG-only A/B probes when representation caches exist."""
  cohort = pl.read_parquet(cohort_path)
  if verify_provenance or (meta_a_path.parent / "ciis_features_provenance.json").exists():
    verify_representation_artifact_integrity(
      reps_dir=meta_a_path.parent,
      cohort=cohort,
      check_b=(meta_b_path is not None and embeddings_b_path is not None and meta_b_path.exists()),
    )
  meta_a = pl.read_parquet(meta_a_path)
  feats_a = np.load(features_a_path)
  frame = _join_representation_matrix(
    cohort, meta_a, feats_a, "za", "representation_a_valid"
  ).filter(pl.col("representation_a_valid") == True)  # noqa: E712
  za_cols = [c for c in frame.columns if c.startswith("za_")]

  models_dir = output_dir / "models" if output_dir is not None else None
  results: dict[str, Any] = {"models": {}, "status": "incomplete"}
  m_a_res, _probe_a_unpaired = fit_and_eval_model(frame, za_cols, "M_A_ECG", models_dir=models_dir)
  results["models"]["M_A_ECG"] = m_a_res

  if (
    meta_b_path is not None
    and embeddings_b_path is not None
    and meta_b_path.exists()
    and embeddings_b_path.exists()
  ):
    meta_b = pl.read_parquet(meta_b_path)
    emb_b = np.load(embeddings_b_path)
    paired = _join_representation_matrix(
      frame, meta_b, emb_b, "zb", "representation_b_valid"
    ).filter(
      (pl.col("representation_a_valid") == True)  # noqa: E712
      & (pl.col("representation_b_valid") == True)  # noqa: E712
    )
    zb_cols = [c for c in paired.columns if c.startswith("zb_")]
    m_b_res, probe_b = fit_and_eval_model(paired, zb_cols, "M_B_ECG", models_dir=models_dir)
    results["models"]["M_B_ECG"] = m_b_res

    # Dimension-matched PCA-15 sensitivity control: fit PCA strictly on development split
    dev_paired = paired.filter(pl.col("split") == "dev")
    n_pca_components = min(15, len(zb_cols), len(dev_paired))
    pca = PCA(n_components=n_pca_components, random_state=42)
    pca.fit(dev_paired.select(zb_cols).to_numpy())
    pca_all = pca.transform(paired.select(zb_cols).to_numpy())
    pca_cols = [f"zb_pca_{i}" for i in range(n_pca_components)]
    paired_with_pca = paired.with_columns(
      [pl.Series(name, pca_all[:, i]) for i, name in enumerate(pca_cols)]
    )
    m_b_pca_res, _probe_b_pca = fit_and_eval_model(
      paired_with_pca, pca_cols, "M_B_PCA15_ECG", models_dir=models_dir
    )
    results["models"]["M_B_PCA15_ECG"] = m_b_pca_res

    probe_a, _da, _va, test_a = _fit_probe(paired, za_cols, model_name="M_A_ECG")
    if models_dir is not None:
      probe_a.save(models_dir / "M_A_ECG.json", format="json")
      probe_a.save(models_dir / "M_A_ECG.joblib", format="joblib")
    test_b = _split_xy(paired, zb_cols, "test")
    pa = probe_a.predict_proba(test_a.x)
    pb = probe_b.predict_proba(test_b.x)

    headline_metrics: dict[str, Any] = {}
    for m in ("auroc", "auprc", "log_loss", "brier"):
      g = representation_gain(test_a.y, pa, pb, metric=m)
      b = bootstrap_paired_difference(
        test_a.y, pa, pb, metric=m, n_bootstraps=n_bootstraps, seed=42
      )
      headline_metrics[m] = {
        "metric": m,
        "gain": g.gain,
        "paired_difference_b_minus_a": g.gain,
        "score_a": g.score_a,
        "score_b": g.score_b,
        "bootstrap": asdict(b),
      }

    gain = representation_gain(test_a.y, pa, pb, metric="auroc")
    boot = headline_metrics["auroc"]["bootstrap"]
    results["rq1_ecg_only"] = {
      "metric": "auroc",
      "gain": gain.gain,
      "score_a": gain.score_a,
      "score_b": gain.score_b,
      "bootstrap": boot,
      "headline_metrics": headline_metrics,
      "n_test": int(len(test_a.y)),
      "n_test_events": int(test_a.y.sum()),
    }
    results["status"] = "complete"
    if output_dir is not None:
      output_dir.mkdir(parents=True, exist_ok=True)
      pl.DataFrame({
        "subject_id": test_a.subject_ids,
        "study_id": test_a.study_ids,
        "mortality_30d": test_a.y,
        "p_a_ecg": pa,
        "p_b_ecg": pb,
        "s_a_ecg": probe_a.predict_log_odds(test_a.x),
        "s_b_ecg": probe_b.predict_log_odds(test_b.x),
      }).write_parquet(output_dir / "test_predictions_ecg_only.parquet")
  else:
    results["unresolved"] = ["D-BETA embeddings unavailable; RQ1 ECG-only gain incomplete"]

  if output_dir is not None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_results.json").write_text(
      json.dumps(results, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
  return results


def run_context_adjusted_evaluation(
  cohort_path: Path,
  c0_path: Path,
  meta_a_path: Path,
  features_a_path: Path,
  meta_b_path: Path,
  embeddings_b_path: Path,
  output_dir: Path | None = None,
  n_bootstraps: int = 200,
  verify_provenance: bool = False,
) -> dict[str, Any]:
  """Evaluate C, C+A, C+B, C+A+B on the paired-valid locked cohort."""
  cohort = pl.read_parquet(cohort_path)
  if verify_provenance or (meta_a_path.parent / "ciis_features_provenance.json").exists():
    verify_representation_artifact_integrity(
      reps_dir=meta_a_path.parent,
      cohort=cohort,
      check_b=True,
    )
  c0 = pl.read_parquet(c0_path)
  meta_a = pl.read_parquet(meta_a_path)
  meta_b = pl.read_parquet(meta_b_path)
  za = np.load(features_a_path)
  zb = np.load(embeddings_b_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]

  frame = cohort.join(
    c0.select(["subject_id", "study_id", *c0_cols]),
    on=["subject_id", "study_id"],
  )
  frame = _join_representation_matrix(frame, meta_a, za, "za", "representation_a_valid")
  frame = _join_representation_matrix(frame, meta_b, zb, "zb", "representation_b_valid")
  frame = frame.filter(
    (pl.col("representation_a_valid") == True)  # noqa: E712
    & (pl.col("representation_b_valid") == True)  # noqa: E712
  )
  za_cols = [c for c in frame.columns if c.startswith("za_")]
  zb_cols = [c for c in frame.columns if c.startswith("zb_")]

  # Dimension-matched PCA-15 sensitivity control: fit PCA strictly on development split
  dev_frame = frame.filter(pl.col("split") == "dev")
  n_pca_components = min(15, len(zb_cols), len(dev_frame))
  pca = PCA(n_components=n_pca_components, random_state=42)
  pca.fit(dev_frame.select(zb_cols).to_numpy())
  pca_all = pca.transform(frame.select(zb_cols).to_numpy())
  pca_cols = [f"zb_pca_{i}" for i in range(n_pca_components)]
  frame_with_pca = frame.with_columns(
    [pl.Series(name, pca_all[:, i]) for i, name in enumerate(pca_cols)]
  )

  models_dir = output_dir / "models" if output_dir is not None else None
  results: dict[str, Any] = {"models": {}, "status": "complete"}
  specs = {
    "M_C": c0_cols,
    "M_A": c0_cols + za_cols,
    "M_B": c0_cols + zb_cols,
    "M_AB": c0_cols + za_cols + zb_cols,
    "M_B_PCA15": c0_cols + pca_cols,
  }
  probes: dict[str, TrainedProbe] = {}
  for name, cols in specs.items():
    source_df = frame_with_pca if name == "M_B_PCA15" else frame
    res, probe = fit_and_eval_model(source_df, cols, name, models_dir=models_dir)
    results["models"][name] = res
    probes[name] = probe

  probe_c = probes["M_C"]
  probe_a = probes["M_A"]
  probe_b = probes["M_B"]
  probe_ab = probes["M_AB"]

  test_ids = _split_xy(frame, c0_cols, "test")
  test_a = _split_xy(frame, c0_cols + za_cols, "test")
  test_b = _split_xy(frame, c0_cols + zb_cols, "test")
  test_ab = _split_xy(frame, c0_cols + za_cols + zb_cols, "test")

  p_c = probe_c.predict_proba(test_ids.x)
  p_a = probe_a.predict_proba(test_a.x)
  p_b = probe_b.predict_proba(test_b.x)
  p_ab = probe_ab.predict_proba(test_ab.x)

  s_a = probe_a.predict_log_odds(test_a.x)
  s_b = probe_b.predict_log_odds(test_b.x)

  # Compute ECG-specific contributions S_A and S_B: isolate beta_Z^T * \tilde{Z}
  # to separate ECG representation signal from shared clinical context C0
  s_a_ecg_contrib = probe_a.predict_block_contribution(test_a.x, start_idx=len(c0_cols))
  s_b_ecg_contrib = probe_b.predict_block_contribution(test_b.x, start_idx=len(c0_cols))
  y_test = test_ids.y

  headline_metrics_ctx: dict[str, Any] = {}
  for m in ("auroc", "auprc", "log_loss", "brier"):
    g = representation_gain(y_test, p_a, p_b, metric=m)
    b = bootstrap_paired_difference(
      y_test, p_a, p_b, metric=m, n_bootstraps=n_bootstraps, seed=42
    )
    headline_metrics_ctx[m] = {
      "metric": m,
      "gain": g.gain,
      "paired_difference_b_minus_a": g.gain,
      "score_a": g.score_a,
      "score_b": g.score_b,
      "bootstrap": asdict(b),
    }

  gain = representation_gain(y_test, p_a, p_b, metric="auroc")
  boot = headline_metrics_ctx["auroc"]["bootstrap"]
  results["rq1_context_adjusted"] = {
    "metric": "auroc",
    "gain": gain.gain,
    "score_a": gain.score_a,
    "score_b": gain.score_b,
    "bootstrap": boot,
    "headline_metrics": headline_metrics_ctx,
  }

  if output_dir is not None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_results_context.json").write_text(
      json.dumps(results, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    pl.DataFrame({
      "subject_id": test_ids.subject_ids,
      "study_id": test_ids.study_ids,
      "mortality_30d": y_test,
      "p_c": p_c,
      "p_a": p_a,
      "p_b": p_b,
      "p_ab": p_ab,
      "s_a": s_a,
      "s_b": s_b,
      "s_a_ecg_contrib": s_a_ecg_contrib,
      "s_b_ecg_contrib": s_b_ecg_contrib,
    }).write_parquet(output_dir / "test_predictions_context.parquet")
  return results


def run_random_control_context(
  cohort_path: Path,
  c0_path: Path,
  meta_random_path: Path,
  embeddings_random_path: Path,
  pair_validity_path: Path,
  reference_context_predictions_path: Path,
  output_dir: Path | None = None,
  n_bootstraps: int = 1000,
) -> dict[str, Any]:
  """Context-adjusted random-encoder control: C0 + random-weight D-BETA embeddings.

  Uses the same paired-valid rows, C0 columns, development standardization, and
  validation log-loss grid as ``M_B``. The fitted ``M_B_RAND`` predictions are contrasted
  with the frozen ``M_A`` and ``M_B`` locked-test predictions, which splits the
  context-adjusted B−A gain into a random-architecture part (``M_B_RAND − M_A``) and a
  pretraining part (``M_B − M_B_RAND``).
  """
  cohort = pl.read_parquet(cohort_path)
  paired = pl.read_parquet(pair_validity_path).filter(pl.col("representation_pair_valid"))
  c0 = pl.read_parquet(c0_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]
  meta_rand = pl.read_parquet(meta_random_path)
  emb_rand = np.load(embeddings_random_path)
  frame = (
    cohort.join(paired.select(["subject_id", "study_id"]), on=["subject_id", "study_id"])
    .join(c0.select(["subject_id", "study_id", *c0_cols]), on=["subject_id", "study_id"])
  )
  frame = _join_representation_matrix(
    frame, meta_rand, emb_rand, "zr", "representation_b_random_valid"
  ).filter(pl.col("representation_b_random_valid") == True)  # noqa: E712
  zr_cols = [c for c in frame.columns if c.startswith("zr_")]

  probe, _dev, _val, test = _fit_probe(frame, c0_cols + zr_cols, model_name="M_B_RAND")
  p_rand = probe.predict_proba(test.x)
  preds = pl.DataFrame({
    "subject_id": test.subject_ids,
    "study_id": test.study_ids,
    "y": test.y,
    "p_rand": p_rand,
  }).join(pl.read_parquet(reference_context_predictions_path), on=["subject_id", "study_id"], how="inner")
  if preds.height != len(test.y):
    raise ValueError(
      f"Frozen context predictions cover {preds.height} of {len(test.y)} locked-test rows."
    )
  y = preds["y"].to_numpy().astype(np.int64)
  pr = preds["p_rand"].to_numpy().astype(np.float64)
  pa = preds["p_a"].to_numpy().astype(np.float64)
  pb = preds["p_b"].to_numpy().astype(np.float64)
  results: dict[str, Any] = {
    "control": "random_encoder_control_context_adjusted",
    "model": "M_B_RAND",
    "n_features": len(c0_cols) + len(zr_cols),
    "best_c": probe.best_c,
    "n_test": int(len(y)),
    "n_test_events": int(y.sum()),
    "test_metrics_random": metric_bundle_to_dict(evaluate_predictions(y, pr)),
  }
  for metric in ("auroc", "auprc"):
    results[f"pretrained_minus_random_{metric}"] = asdict(
      bootstrap_paired_difference(y, pr, pb, metric=metric, n_bootstraps=n_bootstraps, seed=42)
    )
    results[f"random_minus_frozen_a_{metric}"] = asdict(
      bootstrap_paired_difference(y, pa, pr, metric=metric, n_bootstraps=n_bootstraps, seed=42)
    )
  if output_dir is not None:
    output_dir.mkdir(parents=True, exist_ok=True)
    probe.save(output_dir / "M_B_RAND.json", format="json")
    probe.save(output_dir / "M_B_RAND.joblib", format="joblib")
    (output_dir / "random_control_context_results.json").write_text(
      json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
  return results


def dump_context_calibration_scores(
  cohort_path: Path,
  c0_path: Path,
  meta_a_path: Path,
  features_a_path: Path,
  meta_b_path: Path,
  embeddings_b_path: Path,
  models_dir: Path,
  output_path: Path,
  verify_provenance: bool = False,
) -> dict[str, Any]:
  """Score development+validation rows with the FROZEN context-adjusted probes.

  Produces the calibration-score table used by the rank-normalized cascade
  sensitivity (audit 2026-10-05 item 1): ECDF rank calibrators for the MIMIC
  selective cascade must be fitted only on development/validation rows,
  mirroring the CODE-15% deployment analysis. The probes are loaded read-only
  from ``models_dir`` (``M_A``/``M_B``) and are never refit or tuned here;
  dev+val scoring with frozen models is evaluation-only and leakage-safe.
  """
  import joblib  # noqa: PLC0415 — local import mirrors TrainedProbe.load

  cohort = pl.read_parquet(cohort_path)
  if verify_provenance or (meta_a_path.parent / "ciis_features_provenance.json").exists():
    verify_representation_artifact_integrity(
      reps_dir=meta_a_path.parent,
      cohort=cohort,
      check_b=True,
    )
  c0 = pl.read_parquet(c0_path)
  meta_a = pl.read_parquet(meta_a_path)
  meta_b = pl.read_parquet(meta_b_path)
  za = np.load(features_a_path)
  zb = np.load(embeddings_b_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]

  frame = cohort.join(
    c0.select(["subject_id", "study_id", *c0_cols]),
    on=["subject_id", "study_id"],
  )
  frame = _join_representation_matrix(frame, meta_a, za, "za", "representation_a_valid")
  frame = _join_representation_matrix(frame, meta_b, zb, "zb", "representation_b_valid")
  frame = frame.filter(
    (pl.col("representation_a_valid") == True)  # noqa: E712
    & (pl.col("representation_b_valid") == True)  # noqa: E712
  ).filter(pl.col("split").is_in(["dev", "val"]))
  za_cols = [c for c in frame.columns if c.startswith("za_")]
  zb_cols = [c for c in frame.columns if c.startswith("zb_")]

  probe_a = TrainedProbe.load(models_dir / "M_A.joblib")
  probe_b = TrainedProbe.load(models_dir / "M_B.joblib")
  x_a = frame.select(c0_cols + za_cols).to_numpy().astype(np.float64)
  x_b = frame.select(c0_cols + zb_cols).to_numpy().astype(np.float64)

  out = frame.select(["subject_id", "study_id", "split"]).with_columns(
    pl.Series("p_a", probe_a.predict_proba(x_a)),
    pl.Series("p_b", probe_b.predict_proba(x_b)),
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  out.write_parquet(output_path)
  return {
    "n_rows": len(out),
    "n_dev": int((out["split"] == "dev").sum()),
    "n_val": int((out["split"] == "val").sum()),
    "models": {"p_a": str(models_dir / "M_A.joblib"), "p_b": str(models_dir / "M_B.joblib")},
    "models_refit": False,
    "output": str(output_path),
  }


def run_random_dbeta_control(
  cohort_path: Path,
  meta_random_path: Path,
  embeddings_random_path: Path,
  meta_pretrained_path: Path,
  embeddings_pretrained_path: Path,
  output_dir: Path | None = None,
  n_bootstraps: int = 1000,
  pair_validity_path: Path | None = None,
  reference_predictions_path: Path | None = None,
) -> dict[str, Any]:
  """Evaluate architecture-matched random D-BETA encoder vs pretrained D-BETA.

  Assesses whether D-BETA representation gain is attributable to self-supervised
  pretraining rather than high-dimensional random projection capacity.

  ``pair_validity_path`` restricts every split to CIIS/D-BETA paired-valid rows, so the
  locked-test rows match the headline analysis and the pretrained refit can be checked
  against the frozen ``M_B_ECG`` predictions in ``reference_predictions_path``
  (``test_predictions_ecg_only.parquet``). When that file is given, the random encoder is
  also contrasted with the frozen CIIS probe ``M_A_ECG`` on the same patients.
  """
  cohort = pl.read_parquet(cohort_path)
  if pair_validity_path is not None:
    paired = pl.read_parquet(pair_validity_path).filter(pl.col("representation_pair_valid"))
    cohort = cohort.join(paired.select(["subject_id", "study_id"]), on=["subject_id", "study_id"])
  meta_rand = pl.read_parquet(meta_random_path)
  emb_rand = np.load(embeddings_random_path)
  meta_pre = pl.read_parquet(meta_pretrained_path)
  emb_pre = np.load(embeddings_pretrained_path)

  rand_valid_col = (
    "representation_b_random_valid"
    if "representation_b_random_valid" in meta_rand.columns
    else "representation_b_valid"
  )
  frame = _join_representation_matrix(
    cohort, meta_rand, emb_rand, "zb_rand", rand_valid_col
  )
  frame = _join_representation_matrix(
    frame, meta_pre, emb_pre, "zb_pre", "representation_b_valid"
  ).filter(
    (pl.col(rand_valid_col) == True)  # noqa: E712
    & (pl.col("representation_b_valid") == True)  # noqa: E712
  )

  rand_cols = [c for c in frame.columns if c.startswith("zb_rand_")]
  pre_cols = [c for c in frame.columns if c.startswith("zb_pre_")]

  probe_rand, _dr, _vr, test_r = _fit_probe(frame, rand_cols, model_name="M_B_RAND_ECG")
  probe_pre, _dp, _vp, test_p = _fit_probe(frame, pre_cols, model_name="M_B_PRE_ECG")

  pr = probe_rand.predict_proba(test_r.x)
  pp = probe_pre.predict_proba(test_p.x)
  y_test = test_r.y

  gain = representation_gain(y_test, pr, pp, metric="auroc")
  boot = bootstrap_paired_difference(
    y_test, pr, pp, metric="auroc", n_bootstraps=n_bootstraps, seed=42
  )

  boot_auprc = bootstrap_paired_difference(
    y_test, pr, pp, metric="auprc", n_bootstraps=n_bootstraps, seed=42
  )

  results: dict[str, Any] = {
    "control": "random_encoder_control",
    "hypothesis": "pretrained_dbeta_beats_architecture_matched_random_encoder",
    "metric": "auroc",
    "score_random_encoder": gain.score_a,
    "score_pretrained_dbeta": gain.score_b,
    "gain": gain.gain,
    "bootstrap": asdict(boot),
    "bootstrap_auprc": asdict(boot_auprc),
    "test_metrics_random": metric_bundle_to_dict(evaluate_predictions(y_test, pr)),
    "test_metrics_pretrained": metric_bundle_to_dict(evaluate_predictions(y_test, pp)),
    "best_c_random": probe_rand.best_c,
    "best_c_pretrained": probe_pre.best_c,
    "paired_valid_restricted": pair_validity_path is not None,
    "n_test": len(y_test),
    "n_test_events": int(y_test.sum()),
  }

  if reference_predictions_path is not None:
    ref = pl.read_parquet(reference_predictions_path)
    preds = pl.DataFrame({
      "subject_id": test_r.subject_ids,
      "study_id": test_r.study_ids,
      "y": y_test,
      "p_rand": pr,
      "p_pre": pp,
    }).join(ref, on=["subject_id", "study_id"], how="inner")
    if preds.height != len(y_test):
      raise ValueError(
        f"Reference predictions cover {preds.height} of {len(y_test)} locked-test rows; "
        "pass pair_validity_path so both analyses use the same patients."
      )
    y_ref = preds["y"].to_numpy().astype(np.int64)
    p_rand = preds["p_rand"].to_numpy().astype(np.float64)
    p_a_ecg = preds["p_a_ecg"].to_numpy().astype(np.float64)
    results["reproduction_max_abs_diff_vs_frozen_m_b_ecg"] = float(
      np.max(np.abs(preds["p_pre"].to_numpy() - preds["p_b_ecg"].to_numpy()))
    )
    for metric in ("auroc", "auprc"):
      results[f"random_minus_frozen_a_ecg_{metric}"] = asdict(
        bootstrap_paired_difference(
          y_ref, p_a_ecg, p_rand, metric=metric, n_bootstraps=n_bootstraps, seed=42
        )
      )
  if output_dir is not None:
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    probe_rand.save(models_dir / "M_B_RAND_ECG.json", format="json")
    probe_rand.save(models_dir / "M_B_RAND_ECG.joblib", format="joblib")
    probe_pre.save(models_dir / "M_B_PRE_ECG.json", format="json")
    probe_pre.save(models_dir / "M_B_PRE_ECG.joblib", format="joblib")
    results["model_artifact_random_json"] = str(models_dir / "M_B_RAND_ECG.json")
    results["model_artifact_pretrained_json"] = str(models_dir / "M_B_PRE_ECG.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    (output_dir / "random_encoder_control_results.json").write_text(payload, encoding="utf-8")
    (output_dir / "random_dbeta_control_results.json").write_text(payload, encoding="utf-8")
  return results

