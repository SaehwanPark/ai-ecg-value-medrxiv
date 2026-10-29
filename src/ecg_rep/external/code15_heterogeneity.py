"""E5: Heterogeneity, risk discordance and robustness on CODE-15%.

Replicates the MIMIC scientific analyses (residual-risk gradients, four
high/low discordance cells, age/sex subgroup contrasts) on CODE-15% under
the signed G-E1 dual-track decision:

- Score sources: the **unchanged frozen MIMIC probes** (transport track,
    E3) — the same prespecified risk-score source family permitted by plan
    E5. Cutpoints and quantile edges are computed on **development +
    validation rows only** and applied unchanged to the locked test.
- Age groups (<50, 50–64, 65–79, 80+) and sex subgroups report both A and
    B performance, event counts, and B−A differences with patient-level
    bootstrap uncertainty. Published event counts below the suppression
    floor are labeled suppressed; no formal subgroup conclusions are drawn
    from sparse groups.
- No race/ethnicity or comorbidity variables exist in CODE-15%; none are
    invented. The MIMIC Cacute clinical-context attenuation result remains
    MIMIC-only and is never imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ecg_rep.external.code15_labels import horizon_label_to_int
from ecg_rep.external.code15_transport import (
  MIN_CELL_COUNT,
  binary_metrics,
  load_frozen_probes,
  transport_scores,
)

#: Age strata (plan E5: 18–49, 50–64, 65–79, 80+); ages are upper-exclusive.
AGE_EDGES: tuple[float, ...] = (50.0, 65.0, 80.0)
AGE_GROUP_LABELS: tuple[str, ...] = ("<50", "50-64", "65-79", "80+")

#: Prespecified high-risk quantile for the four discordance cells.
HIGH_RISK_QUANTILE = 0.8

#: Number of outer-score quantile strata for residual-risk gradients.
GRADIENT_STRATA = 4


def assign_age_group(age: npt.NDArray[np.floating]) -> npt.NDArray[np.str_]:
  """Map ages to the prespecified strata labels."""
  age = np.asarray(age, dtype=np.float64)
  idx = np.searchsorted(np.asarray(AGE_EDGES), age, side="right")
  return np.asarray(AGE_GROUP_LABELS, dtype=object)[idx].astype(str)


def dev_val_cutpoints(
  scores: npt.NDArray[np.floating],
  high_quantile: float = HIGH_RISK_QUANTILE,
) -> dict[str, Any]:
  """Risk cutpoints computed on development/validation rows only."""
  arr = np.asarray(scores, dtype=np.float64)
  return {
    "high_quantile": high_quantile,
    "high_cutpoint": float(np.quantile(arr, high_quantile)),
    "quantile_edges": np.quantile(arr, np.linspace(0.0, 1.0, GRADIENT_STRATA + 1)).tolist(),
  }


def _stratum_masks(scores: npt.NDArray[np.float64], edges: list[float]) -> list[npt.NDArray[np.bool_]]:
  masks = []
  for i in range(len(edges) - 1):
    lo, hi = edges[i], edges[i + 1]
    if i == len(edges) - 2:
      masks.append((scores >= lo) & (scores <= hi))
    else:
      masks.append((scores >= lo) & (scores < hi))
  return masks


def residual_risk_gradient_prespecified(
  y: npt.NDArray[np.integer],
  outer_scores: npt.NDArray[np.floating],
  inner_scores: npt.NDArray[np.floating],
  edges: list[float],
  min_cell: int = MIN_CELL_COUNT,
) -> list[dict[str, Any]]:
  """Event-rate gradient of ``inner`` within prespecified strata of ``outer``.

  Strata edges come from development/validation scores; the evaluated rows
  are typically the locked test. Within each outer stratum, low/high inner
  risk is split at the median of the inner score *within that stratum*.
  """
  yt = np.asarray(y, dtype=np.int64)
  outer = np.asarray(outer_scores, dtype=np.float64)
  inner = np.asarray(inner_scores, dtype=np.float64)
  rows: list[dict[str, Any]] = []
  for i, mask in enumerate(_stratum_masks(outer, edges)):
    y_s, inner_s = yt[mask], inner[mask]
    if len(y_s) == 0:
      continue
    mid = float(np.median(inner_s))
    low_rate = float(y_s[inner_s <= mid].mean()) if np.any(inner_s <= mid) else None
    high_rate = float(y_s[inner_s > mid].mean()) if np.any(inner_s > mid) else None
    events = int(y_s.sum())
    rows.append(
      {
        "stratum": f"Q{i + 1}",
        "n": int(mask.sum()),
        "n_events": events if events >= min_cell else "suppressed(<11)",
        "event_rate": float(y_s.mean()) if events >= min_cell else "suppressed(<11)",
        "inner_low_rate": low_rate if events >= min_cell else "suppressed(<11)",
        "inner_high_rate": high_rate if events >= min_cell else "suppressed(<11)",
      },
    )
  return rows


def discordance_cells_prespecified(
  y: npt.NDArray[np.integer],
  p_a: npt.NDArray[np.floating],
  p_b: npt.NDArray[np.floating],
  a_cut: float,
  b_cut: float,
  min_cell: int = MIN_CELL_COUNT,
) -> list[dict[str, Any]]:
  """Four high/low cells from development/validation-defined cutpoints."""
  yt = np.asarray(y, dtype=np.int64)
  a_high = np.asarray(p_a) >= a_cut
  b_high = np.asarray(p_b) >= b_cut
  cells: list[dict[str, Any]] = []
  for label, mask in (
    ("concordant_low", (~a_high) & (~b_high)),
    ("a_low_b_high", (~a_high) & b_high),
    ("a_high_b_low", a_high & (~b_high)),
    ("concordant_high", a_high & b_high),
  ):
    n = int(mask.sum())
    events = int(yt[mask].sum())
    cells.append(
      {
        "cell": label,
        "n": n,
        "n_events": events if events >= min_cell else "suppressed(<11)",
        "event_rate": float(events / n) if n and events >= min_cell else "suppressed(<11)",
      },
    )
  return cells


def subgroup_metrics(
  y: npt.NDArray[np.integer],
  group_labels: npt.NDArray[np.str_],
  p_a: npt.NDArray[np.floating],
  p_b: npt.NDArray[np.floating],
  n_bootstraps: int = 1000,
  seed: int = 1912,
  min_cell: int = MIN_CELL_COUNT,
) -> list[dict[str, Any]]:
  """Per-group A/B discrimination with suppressed sparse cells."""
  from ecg_rep.external.code15_transport import paired_bootstrap_delta

  yt = np.asarray(y, dtype=np.int64)
  rows: list[dict[str, Any]] = []
  for group in sorted(set(group_labels.tolist())):
    mask = group_labels == group
    y_g = yt[mask]
    events = int(y_g.sum())
    row: dict[str, Any] = {
      "group": str(group),
      "n": int(mask.sum()),
      "n_events": events if events >= min_cell else "suppressed(<11)",
    }
    if events < min_cell or y_g.min() == y_g.max():
      row.update({"metrics_a": None, "metrics_b": None, "delta_auroc_b_minus_a": None})
    else:
      delta = paired_bootstrap_delta(
        y_g, np.asarray(p_a)[mask], np.asarray(p_b)[mask],
        n_bootstraps=n_bootstraps, seed=seed,
      )
      row.update(
        {
          "metrics_a": binary_metrics(y_g, np.asarray(p_a)[mask]),
          "metrics_b": binary_metrics(y_g, np.asarray(p_b)[mask]),
          "delta_auroc_b_minus_a": delta["delta_auroc_b_minus_a"],
        },
      )
    rows.append(row)
  return rows


def subgroup_interaction_contrasts(
  y: npt.NDArray[np.integer],
  group_labels: npt.NDArray[np.str_],
  p_a: npt.NDArray[np.floating],
  p_b: npt.NDArray[np.floating],
  group_pairs: tuple[tuple[str, str], ...],
  n_bootstraps: int = 1000,
  seed: int = 1912,
  min_cell: int = MIN_CELL_COUNT,
) -> list[dict[str, Any]]:
  """Bootstrap differences between group-specific paired B−A AUROC deltas.

  Each group's patients are resampled independently, stratified by outcome;
  A/B scores stay paired within each patient. The reported contrast is
  ``(AUROC_B - AUROC_A)_left - (AUROC_B - AUROC_A)_right``.
  """
  from sklearn.metrics import roc_auc_score

  yt = np.asarray(y, dtype=np.int64).ravel()
  groups = np.asarray(group_labels).astype(str).ravel()
  score_a = np.asarray(p_a, dtype=np.float64).ravel()
  score_b = np.asarray(p_b, dtype=np.float64).ravel()
  if not (len(yt) == len(groups) == len(score_a) == len(score_b)):
    raise ValueError("Outcome, group, and A/B score arrays must have equal length.")
  if n_bootstraps < 1:
    raise ValueError("At least one bootstrap replicate is required.")

  rng = np.random.default_rng(seed)
  rows: list[dict[str, Any]] = []
  for left_group, right_group in group_pairs:
    left = groups == left_group
    right = groups == right_group
    y_left, y_right = yt[left], yt[right]
    events_left, events_right = int(y_left.sum()), int(y_right.sum())
    valid_left = (
      y_left.size > 0
      and y_left.min() != y_left.max()
      and events_left >= min_cell
      and int(y_left.size - events_left) >= min_cell
    )
    valid_right = (
      y_right.size > 0
      and y_right.min() != y_right.max()
      and events_right >= min_cell
      and int(y_right.size - events_right) >= min_cell
    )
    name = f"{left_group}_minus_{right_group}_delta_auroc"
    base = {
      "contrast": name,
      "left_group": left_group,
      "right_group": right_group,
      "n_left": int(left.sum()),
      "n_events_left": events_left if events_left >= min_cell else "suppressed(<11)",
      "n_right": int(right.sum()),
      "n_events_right": events_right if events_right >= min_cell else "suppressed(<11)",
      "n_bootstraps": n_bootstraps,
      "seed": seed,
    }
    if not valid_left or not valid_right:
      rows.append(
        {
          **base,
          "left_delta_auroc": None,
          "right_delta_auroc": None,
          "interaction_estimate": None,
          "ci_lower": None,
          "ci_upper": None,
          "n_bootstraps_used": 0,
          "status": "suppressed_or_insufficient_events_in_group",
        },
      )
      continue

    left_a, left_b = score_a[left], score_b[left]
    right_a, right_b = score_a[right], score_b[right]
    left_delta = float(roc_auc_score(y_left, left_b) - roc_auc_score(y_left, left_a))
    right_delta = float(roc_auc_score(y_right, right_b) - roc_auc_score(y_right, right_a))
    left_pos, left_neg = np.flatnonzero(y_left == 1), np.flatnonzero(y_left == 0)
    right_pos, right_neg = np.flatnonzero(y_right == 1), np.flatnonzero(y_right == 0)
    boot: list[float] = []
    for _ in range(n_bootstraps):
      left_idx = np.concatenate(
        (
          rng.choice(left_pos, size=len(left_pos), replace=True),
          rng.choice(left_neg, size=len(left_neg), replace=True),
        ),
      )
      right_idx = np.concatenate(
        (
          rng.choice(right_pos, size=len(right_pos), replace=True),
          rng.choice(right_neg, size=len(right_neg), replace=True),
        ),
      )
      left_delta_b = float(
        roc_auc_score(y_left[left_idx], left_b[left_idx])
        - roc_auc_score(y_left[left_idx], left_a[left_idx]),
      )
      right_delta_b = float(
        roc_auc_score(y_right[right_idx], right_b[right_idx])
        - roc_auc_score(y_right[right_idx], right_a[right_idx]),
      )
      boot.append(left_delta_b - right_delta_b)
    bounds = np.percentile(np.asarray(boot, dtype=np.float64), [2.5, 97.5])
    rows.append(
      {
        **base,
        "left_delta_auroc": left_delta,
        "right_delta_auroc": right_delta,
        "interaction_estimate": left_delta - right_delta,
        "ci_lower": float(bounds[0]),
        "ci_upper": float(bounds[1]),
        "bootstrap_mean": float(np.mean(boot)),
        "n_bootstraps_used": len(boot),
        "status": "ok",
      },
    )
  return rows


@dataclass(frozen=True)
class HeterogeneityReport:
  """Aggregate E5 report (no patient-level values)."""

  horizon: str
  attrition: dict[str, Any]
  cutpoints_a: dict[str, Any]
  cutpoints_b: dict[str, Any]
  gradient_b_within_a: list[dict[str, Any]]
  gradient_a_within_b: list[dict[str, Any]]
  discordance_cells: list[dict[str, Any]]
  age_subgroups: list[dict[str, Any]]
  sex_subgroups: list[dict[str, Any]]
  interaction_contrasts: list[dict[str, Any]]


def run_heterogeneity_analysis(
  features_a: npt.NDArray[np.float64],
  features_b: npt.NDArray[np.float64],
  meta: Any,
  model_dir: Any,
  baseline_manifest_path: Any | None = None,
  horizon: str = "365d",
  n_bootstraps: int = 1000,
  seed: int = 1912,
) -> HeterogeneityReport:
  """Run E5 using frozen-probe transport scores on prespecified strata.

  ``meta`` (polars) must be aligned with the feature rows and contain
  ``patient_id``, ``split``, ``age``, ``is_male``, and the horizon label
  ``y_{horizon}``. Cutpoints/edges use dev+val rows; the evaluated rows are
  the locked test.
  """
  import polars as pl

  label_col = f"y_{horizon}"
  required = {"patient_id", "split", "age", "is_male", label_col}
  missing = required - set(meta.columns)
  if missing:
    raise ValueError(f"Meta frame is missing columns: {sorted(missing)}.")
  meta = horizon_label_to_int(meta, horizon)

  probes = load_frozen_probes(model_dir, baseline_manifest_path)
  p_all_a, p_all_b = transport_scores(probes, features_a, features_b)

  labels = meta[label_col].to_list()
  splits = meta["split"].to_list()
  keep = np.asarray([v is not None for v in labels], dtype=bool)
  y = np.asarray([v for v in labels if v is not None], dtype=np.int64)
  pa = p_all_a[keep]
  pb = p_all_b[keep]
  split_kept = [s for s, k in zip(splits, keep, strict=True) if k]
  dev_val_mask = np.asarray([s in ("dev", "val") for s in split_kept], dtype=bool)
  test_mask = np.asarray([s == "test" for s in split_kept], dtype=bool)

  cut_a = dev_val_cutpoints(pa[dev_val_mask])
  cut_b = dev_val_cutpoints(pb[dev_val_mask])

  y_test = y[test_mask]
  age = np.asarray(meta.filter(pl.lit(keep))["age"].to_list(), dtype=np.float64)[test_mask]
  is_male = np.asarray(meta.filter(pl.lit(keep))["is_male"].to_list(), dtype=np.float64)[test_mask]
  age_groups = assign_age_group(age)
  sex_groups = np.where(is_male >= 0.5, "male", "female")

  return HeterogeneityReport(
    horizon=horizon,
    attrition={
      "n_extracted_total": int(len(meta)),
      "n_labeled_complete": int(len(y)),
      f"n_dropped_unknown_{horizon}": int(len(meta) - len(y)),
      "n_dev_val": int(dev_val_mask.sum()),
      "n_test": int(test_mask.sum()),
      "n_events_test": int(y_test.sum()),
    },
    cutpoints_a=cut_a,
    cutpoints_b=cut_b,
    gradient_b_within_a=residual_risk_gradient_prespecified(
      y_test, pa[test_mask], pb[test_mask], cut_a["quantile_edges"],
    ),
    gradient_a_within_b=residual_risk_gradient_prespecified(
      y_test, pb[test_mask], pa[test_mask], cut_b["quantile_edges"],
    ),
    discordance_cells=discordance_cells_prespecified(
      y_test, pa[test_mask], pb[test_mask], cut_a["high_cutpoint"], cut_b["high_cutpoint"],
    ),
    age_subgroups=subgroup_metrics(
      y_test, age_groups, pa[test_mask], pb[test_mask],
      n_bootstraps=n_bootstraps, seed=seed,
    ),
    sex_subgroups=subgroup_metrics(
      y_test, sex_groups,
      pa[test_mask], pb[test_mask],
      n_bootstraps=n_bootstraps, seed=seed,
    ),
    interaction_contrasts=(
      subgroup_interaction_contrasts(
        y_test,
        sex_groups,
        pa[test_mask],
        pb[test_mask],
        (("female", "male"),),
        n_bootstraps=n_bootstraps,
        seed=seed,
      )
      + subgroup_interaction_contrasts(
        y_test,
        age_groups,
        pa[test_mask],
        pb[test_mask],
        tuple((group, "<50") for group in AGE_GROUP_LABELS[1:]),
        n_bootstraps=n_bootstraps,
        seed=seed,
      )
    ),
  )
