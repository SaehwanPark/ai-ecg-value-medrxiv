"""E4: Target-cohort (CODE-15%) linear probe representation replication.

Fits the **same ridge-logistic downstream learners** as the MIMIC analysis
(dev-only standardization, validation log-loss ``C`` selection, locked-test
evaluation) on CODE-15% representations, preserving the unchanged 15 A
features and 768 B features.

Governance (plan Agent E4):

- Primary ECG-only fits: ``A``, ``B``, ``A+B``.
- Separately labeled demographics-adjusted fits: ``C_demo = age + is_male``,
    ``C_demo + A``, ``C_demo + B``, ``C_demo + A+B`` (this is NOT the MIMIC
    ``C0`` historical-Elixhauser block; that variable is unavailable here).
- Dimensionality-matched control ``B_pca15``: PCA-15 fit **only** on CODE
    development features.
- Hyperparameter grids are the MIMIC probe specification. If the optimal
    ``C`` lands on a grid boundary, the predeclared widened grid (below) may
    be used on development/validation only — never after inspecting the
    CODE locked test.
- Locked-test labels are touched exactly once, at final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ecg_rep.external.code15_labels import DAYS_PER_YEAR, HORIZON_DAYS, horizon_label_to_int
from ecg_rep.analysis import fixed_capacity_contrast_rows, fixed_capacity_metric_rows
from ecg_rep.probe import (
  DEFAULT_C_GRID,
  MetricBundle,
  TrainedProbe,
  bootstrap_paired_difference,
  evaluate_predictions,
  fit_logistic_probe,
  stack_feature_blocks,
)

#: Predeclared widened C grid (development/validation-only use, per plan E4).
#: Declared here, before any CODE test inspection, to remove post-hoc freedom.
WIDENED_C_GRID: tuple[float, ...] = (1e-6, 1e-5, *DEFAULT_C_GRID, 1e4, 1e5, 1e6)

#: Demographics control block: age + is_male (separately labeled; not MIMIC C0).
DEMOGRAPHIC_FEATURES: tuple[str, ...] = ("age", "is_male")

#: Fit name -> (family label, uses A, uses B, uses demographics, uses PCA-15).
FIT_SPECS: dict[str, dict[str, Any]] = {
  "A": {"family": "ecg_only", "a": True, "b": False, "demo": False, "pca": False},
  "B": {"family": "ecg_only", "a": False, "b": True, "demo": False, "pca": False},
  "A+B": {"family": "ecg_only", "a": True, "b": True, "demo": False, "pca": False},
  "B_pca15": {"family": "dimensionality_matched", "a": False, "b": False, "demo": False, "pca": True},
  "C_demo": {"family": "demographics_adjusted", "a": False, "b": False, "demo": True, "pca": False},
  "C_demo+A": {"family": "demographics_adjusted", "a": True, "b": False, "demo": True, "pca": False},
  "C_demo+B": {"family": "demographics_adjusted", "a": False, "b": True, "demo": True, "pca": False},
  "C_demo+A+B": {"family": "demographics_adjusted", "a": True, "b": True, "demo": True, "pca": False},
}


def c_at_grid_boundary(probe: TrainedProbe, grid: tuple[float, ...] = DEFAULT_C_GRID) -> bool:
  """Flag when the selected ``C`` sits on either end of the prespecified grid."""
  return probe.best_c in (min(grid), max(grid)) and len(grid) > 1


def boundary_selection_review(
  probe: TrainedProbe,
  grid: tuple[float, ...] = DEFAULT_C_GRID,
) -> dict[str, Any]:
  """Summarize the adjacent validation loss for a boundary-selected ``C``.

  This is a validation-only diagnostic over the already evaluated prespecified
  grid. It does not expand the grid or inspect locked-test outcomes.
  """
  ordered = tuple(sorted(grid))
  boundary: str | None = None
  neighbor: float | None = None
  if len(ordered) > 1 and probe.best_c == ordered[0]:
    boundary, neighbor = "lower", ordered[1]
  elif len(ordered) > 1 and probe.best_c == ordered[-1]:
    boundary, neighbor = "upper", ordered[-2]

  losses = {step.c_value: step.val_log_loss for step in probe.tuning_history}
  selected_loss = losses.get(probe.best_c)
  neighbor_loss = losses.get(neighbor) if neighbor is not None else None
  return {
    "selected_c": probe.best_c,
    "selected_boundary": boundary,
    "selected_val_log_loss": selected_loss,
    "adjacent_inward_c": neighbor,
    "adjacent_inward_val_log_loss": neighbor_loss,
    "adjacent_inward_minus_selected_val_log_loss": (
      neighbor_loss - selected_loss
      if neighbor_loss is not None and selected_loss is not None
      else None
    ),
  }


def assemble_fit_matrix(
  fit_name: str,
  features_a: npt.NDArray[np.float64],
  features_b: npt.NDArray[np.float64],
  demographics: npt.NDArray[np.float64],
  b_pca_components: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
  """Build the design matrix for one fit spec (PCA applied when required).

  ``b_pca_components`` (shape ``(15, 768)``) must come from a PCA fitted on
  development rows only; passing ``None`` for ``B_pca15`` is an error.
  """
  spec = FIT_SPECS[fit_name]
  blocks: list[npt.NDArray[np.float64]] = []
  if spec["a"]:
    blocks.append(features_a)
  if spec["b"]:
    blocks.append(features_b)
  if spec["pca"]:
    if b_pca_components is None:
      raise ValueError(f"Fit '{fit_name}' requires PCA components fitted on dev only.")
    blocks.append(features_b @ b_pca_components.T)
  if spec["demo"]:
    blocks.append(demographics)
  if not blocks:
    raise ValueError(f"Fit '{fit_name}' assembled an empty design matrix.")
  return stack_feature_blocks(*blocks)


def fit_pca15_on_dev(
  features_b_dev: npt.NDArray[np.float64],
  seed: int = 1912,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
  """Fit a 15-component PCA on development B features only (frozen control)."""
  from sklearn.decomposition import PCA

  pca = PCA(n_components=15, random_state=seed)
  pca.fit(np.asarray(features_b_dev, dtype=np.float64))
  info = {
    "n_components": 15,
    "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    "n_dev_rows": int(len(features_b_dev)),
    "seed": seed,
  }
  return np.asarray(pca.components_, dtype=np.float64), info


def fit_code15_probe_set(
  matrices_dev: dict[str, npt.NDArray[np.float64]],
  y_dev: npt.NDArray[np.int64],
  matrices_val: dict[str, npt.NDArray[np.float64]],
  y_val: npt.NDArray[np.int64],
  c_grid: tuple[float, ...] = DEFAULT_C_GRID,
  seed: int = 42,
) -> dict[str, TrainedProbe]:
  """Fit the prespecified probes present in ``matrices_dev`` (dev fit, val tuning only)."""
  from ecg_rep.probe import ProbeConfig

  unknown = set(matrices_dev) - set(FIT_SPECS)
  if unknown:
    raise ValueError(f"Unknown fit names: {sorted(unknown)}.")
  probes: dict[str, TrainedProbe] = {}
  for fit_name in FIT_SPECS:
    if fit_name not in matrices_dev:
      continue
    probes[fit_name] = fit_logistic_probe(
      matrices_dev[fit_name],
      y_dev,
      matrices_val[fit_name],
      y_val,
      config=ProbeConfig(regularization_c_grid=c_grid, random_state=seed),
      model_name=f"CODE15_{fit_name}",
    )
  return probes


def evaluate_code15_probe_set(
  probes: dict[str, TrainedProbe],
  matrices_test: dict[str, npt.NDArray[np.float64]],
  y_test: npt.NDArray[np.int64],
) -> dict[str, MetricBundle]:
  """Locked-test evaluation of every fitted CODE probe (single pass)."""
  bundles: dict[str, MetricBundle] = {}
  for fit_name, probe in probes.items():
    prob = probe.predict_proba(matrices_test[fit_name])
    bundles[fit_name] = evaluate_predictions(y_test, prob)
  return bundles


def paired_incremental_contrasts(
  y_test: npt.NDArray[np.int64],
  prob_by_fit: dict[str, npt.NDArray[np.float64]],
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, Any]:
  """Direct and bidirectional paired AUROC contrasts on the locked test."""
  contrasts: dict[str, Any] = {}
  for base, combined in (("A", "A+B"), ("B", "A+B")):
    diff = bootstrap_paired_difference(
      y_test,
      prob_by_fit[base],
      prob_by_fit[combined],
      metric="auroc",
      n_bootstraps=n_bootstraps,
      seed=seed,
    )
    contrasts[f"{base}_to_{combined}_auroc"] = {
      "point_estimate": diff.point_estimate,
      "ci_lower": diff.ci_lower,
      "ci_upper": diff.ci_upper,
      "n_bootstraps": diff.n_bootstraps,
      "seed": diff.seed,
    }
  for base, added, name in (
    ("A", "B", "B_minus_A_auroc"),
    ("C_demo+A", "C_demo+B", "C_demo_B_minus_C_demo_A_auroc"),
    # Jinbo item 4 (CODE analogue): incremental gains of A and B over C_demo,
    # not only the direct C_demo+B vs C_demo+A contrast.
    ("C_demo", "C_demo+A", "C_demo_add_A_auroc"),
    ("C_demo", "C_demo+B", "C_demo_add_B_auroc"),
    ("C_demo+A", "C_demo+A+B", "C_demo_A_add_B_auroc"),
    ("C_demo+B", "C_demo+A+B", "C_demo_B_add_A_auroc"),
  ):
    diff = bootstrap_paired_difference(
      y_test,
      prob_by_fit[base],
      prob_by_fit[added],
      metric="auroc",
      n_bootstraps=n_bootstraps,
      seed=seed,
    )
    contrasts[name] = {
      "point_estimate": diff.point_estimate,
      "ci_lower": diff.ci_lower,
      "ci_upper": diff.ci_upper,
      "n_bootstraps": diff.n_bootstraps,
      "seed": diff.seed,
    }
  return contrasts


#: Demographics-adjusted fits evaluated at fixed review capacity (plan A3).
DEMOGRAPHICS_CAPACITY_MODELS: tuple[str, ...] = (
  "C_demo", "C_demo+A", "C_demo+B", "C_demo+A+B",
)

#: Fixed-capacity paired contrasts for the demographics-adjusted family.
DEMOGRAPHICS_CAPACITY_CONTRASTS: tuple[tuple[str, str, str], ...] = (
  ("C_demo", "C_demo+A", "add_A_to_C_demo"),
  ("C_demo", "C_demo+B", "add_B_to_C_demo"),
  ("C_demo+A", "C_demo+B", "B_minus_A_under_C_demo"),
  ("C_demo+A", "C_demo+A+B", "add_B_to_C_demo_a"),
  ("C_demo+B", "C_demo+A+B", "add_A_to_C_demo_b"),
)


def build_fixed_capacity_section(
  y_test: npt.NDArray[np.int64],
  prob_by_fit: dict[str, npt.NDArray[np.float64]],
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
  """Fixed-capacity metrics and paired contrasts on the CODE locked test (plan A3).

  Evaluation-only: applied to the already-frozen prespecified fits; no model
  or threshold is selected from these results.
  """
  scores = {name: prob_by_fit[name] for name in DEMOGRAPHICS_CAPACITY_MODELS}
  return {
    "metric_rows": fixed_capacity_metric_rows(y_test, scores, list(DEMOGRAPHICS_CAPACITY_MODELS)),
    "contrast_rows": fixed_capacity_contrast_rows(
      y_test,
      scores,
      DEMOGRAPHICS_CAPACITY_CONTRASTS,
      n_bootstraps=n_bootstraps,
      seed=seed,
    ),
  }


@dataclass(frozen=True)
class ProbeSetReport:
  """Aggregate E4 report (no patient-level values)."""

  horizon: str
  n_dev: int
  n_val: int
  n_test: int
  n_events_test: int
  attrition: dict[str, Any]
  tuning: dict[str, Any]
  pca15_info: dict[str, Any]
  test_bundles: dict[str, dict[str, Any]]
  contrasts: dict[str, Any]
  fixed_capacity: dict[str, list[dict[str, Any]]]
  ipcw_sensitivity: dict[str, Any]


def run_code15_probe_analysis(
  features_a: npt.NDArray[np.float64],
  features_b: npt.NDArray[np.float64],
  meta: Any,
  horizon: str = "365d",
  seed: int = 42,
  n_bootstraps: int = 1000,
) -> ProbeSetReport:
  """Fit and evaluate the full prespecified CODE probe set on one horizon.

  ``meta`` is a ``polars.DataFrame`` with one row per extracted exam
  (sorted by ``exam_id`` and aligned with the feature rows) containing
  ``patient_id``, the horizon label ``y_{horizon}``, ``age``, ``is_male``,
  ``death``, ``timey``, and a ``split`` column from the patient-level assignment.
  """
  import polars as pl

  label_col = f"y_{horizon}"
  required = {"patient_id", "split", "age", "is_male", "death", "timey", label_col}
  missing = required - set(meta.columns)
  if missing:
    raise ValueError(f"Meta frame is missing columns: {sorted(missing)}.")
  meta = horizon_label_to_int(meta, horizon)

  merged = meta.with_row_index("_row")
  labels = merged[label_col].to_list()
  keep_rows = merged.filter(pl.col(label_col).is_not_null())
  n_unknown = int(len(merged) - len(keep_rows))
  split_labels_all = meta["split"].to_list()
  test_all_mask = np.asarray([s == "test" for s in split_labels_all], dtype=bool)
  dev_val_all_mask = np.asarray([s in ("dev", "val") for s in split_labels_all], dtype=bool)

  order = keep_rows["_row"].to_numpy()
  fa = np.asarray(features_a, dtype=np.float64)[order]
  fb = np.asarray(features_b, dtype=np.float64)[order]
  y = np.asarray(keep_rows[label_col].to_list(), dtype=np.int64)
  demo = np.asarray(
    list(
      zip(
        keep_rows["age"].to_list(),
        keep_rows["is_male"].to_list(),
        strict=True,
      ),
    ),
    dtype=np.float64,
  )
  splits = keep_rows["split"].to_list()

  dev_mask = np.asarray([s == "dev" for s in splits], dtype=bool)
  val_mask = np.asarray([s == "val" for s in splits], dtype=bool)
  test_mask = np.asarray([s == "test" for s in splits], dtype=bool)

  pca_components, pca_info = fit_pca15_on_dev(fb[dev_mask], seed=seed)

  matrices = {
    split_name: {
      fit_name: assemble_fit_matrix(fit_name, fa[m], fb[m], demo[m], pca_components)
      for fit_name in FIT_SPECS
    }
    for split_name, m in (("dev", dev_mask), ("val", val_mask), ("test", test_mask))
  }

  probes = fit_code15_probe_set(
    matrices["dev"], y[dev_mask], matrices["val"], y[val_mask], seed=seed,
  )
  tuning = {
    name: {
      "best_c": probe.best_c,
      "best_c_at_default_grid_boundary": c_at_grid_boundary(probe),
      "boundary_selection_review": boundary_selection_review(probe),
      "val_log_loss_at_best_c": min(step.val_log_loss for step in probe.tuning_history)
      if probe.tuning_history
      else None,
    }
    for name, probe in probes.items()
  }

  bundles = evaluate_code15_probe_set(probes, matrices["test"], y[test_mask])
  prob_by_fit = {
    name: probe.predict_proba(matrices["test"][name]) for name, probe in probes.items()
  }
  contrasts = paired_incremental_contrasts(
    y[test_mask], prob_by_fit, n_bootstraps=n_bootstraps, seed=seed,
  )
  fixed_capacity = build_fixed_capacity_section(
    y[test_mask], prob_by_fit, n_bootstraps=n_bootstraps, seed=seed,
  )
  n_dev = int(dev_mask.sum())
  n_val = int(val_mask.sum())
  n_test = int(test_mask.sum())
  n_events_test = int(y[test_mask].sum())
  n_extracted_total = int(len(merged))
  n_analyzed = int(len(y))
  del matrices, fa, fb, demo, y, keep_rows, merged

  # Re-score every extracted test row, including patients censored before the
  # binary horizon, with the probes fitted on complete-case development data.
  features_a_test = np.asarray(features_a[test_all_mask], dtype=np.float64)
  features_b_test = np.asarray(features_b[test_all_mask], dtype=np.float64)
  demographics_all = np.asarray(
    list(
      zip(
        meta.filter(pl.Series(test_all_mask))["age"].to_list(),
        meta.filter(pl.Series(test_all_mask))["is_male"].to_list(),
        strict=True,
      ),
    ),
    dtype=np.float64,
  )
  scores_test_all: dict[str, npt.NDArray[np.float64]] = {}
  for fit_name, probe in probes.items():
    matrix_test_all = assemble_fit_matrix(
      fit_name,
      features_a_test,
      features_b_test,
      demographics_all,
      pca_components,
    )
    scores_test_all[fit_name] = probe.predict_proba(matrix_test_all)
  test_meta = meta.filter(pl.Series(test_all_mask))
  dev_val_meta = meta.filter(pl.Series(dev_val_all_mask))
  follow_up_days = np.asarray(test_meta["timey"].to_list(), dtype=np.float64) * DAYS_PER_YEAR
  event_observed = np.asarray(test_meta["death"].to_list(), dtype=bool)
  from ecg_rep.external.code15_survival import censoring_sensitivity_report

  ipcw_sensitivity = censoring_sensitivity_report(
    reference_times=np.asarray(dev_val_meta["timey"].to_list(), dtype=np.float64) * DAYS_PER_YEAR,
    reference_events=np.asarray(dev_val_meta["death"].to_list(), dtype=bool),
    test_times=follow_up_days,
    test_events=event_observed,
    risk_scores=scores_test_all,
    contrasts={
      "B_minus_A": ("A", "B"),
      "C_demo_B_minus_C_demo_A": ("C_demo+A", "C_demo+B"),
    },
    horizon=HORIZON_DAYS[horizon],
    n_bootstraps=n_bootstraps,
    seed=seed,
  )

  return ProbeSetReport(
    horizon=horizon,
    n_dev=n_dev,
    n_val=n_val,
    n_test=n_test,
    n_events_test=n_events_test,
    attrition={
      "n_extracted_total": n_extracted_total,
      f"n_dropped_unknown_{horizon}": n_unknown,
      "n_analyzed": n_analyzed,
      "n_extracted_test_including_censored": int(test_all_mask.sum()),
      f"n_test_dropped_unknown_{horizon}": int(test_all_mask.sum() - test_mask.sum()),
      "n_test_complete_case": int(test_mask.sum()),
    },
    tuning=tuning,
    pca15_info=pca_info,
    test_bundles={name: dict(vars(bundle)) for name, bundle in bundles.items()},
    contrasts=contrasts,
    fixed_capacity=fixed_capacity,
    ipcw_sensitivity=ipcw_sensitivity,
  )
