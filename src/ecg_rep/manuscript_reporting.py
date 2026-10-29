"""Manuscript reporting layer on frozen artifacts (readiness slice S2).

Produces the items the research plan (§5.1, §9, §20) requires for the manuscript but the
primary pipeline never emitted: Table 1 cohort characteristics, calibration with
uncertainty and decile calibration curves, the canonical CIIS total score as a
descriptive continuity reference, and a symmetric outcome-ascertainment sensitivity.

Everything here is evaluation-only. No probe is fit, no threshold or grid is chosen, and
locked-test outcomes are used only to describe frozen predictions (decisions M2–M4 in
``docs/manuscript/00_readiness_audit.md``). Committed outputs are aggregates with cells
below ``MIN_CELL_COUNT`` suppressed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence
import json

import numpy as np
import numpy.typing as npt
import polars as pl

from ecg_rep.analysis import assign_age_subgroup
from ecg_rep.probe import (
  bootstrap_paired_difference,
  compute_auprc,
  compute_auroc,
  fit_calibration_line,
)

MIN_CELL_COUNT = 11
SUPPRESSED = "<11"

TABLE1_GROUPS: tuple[str, ...] = ("overall", "dev", "val", "test", "test_analytic")

SETTING_LABELS: dict[str, str] = {
  "ED": "Emergency department",
  "inpatient_non_ICU": "Inpatient, non-ICU",
  "ICU": "ICU",
  "other_or_unlinked": "Other or unlinked",
  "ambiguous": "Ambiguous",
}

RACE_LABELS: dict[str, str] = {
  "white": "White",
  "black": "Black",
  "hispanic_latino": "Hispanic/Latino",
  "asian": "Asian",
  "unknown_other": "Other or unknown",
}

AGE_LABELS: dict[str, str] = {
  "18-49": "18–49",
  "50-64": "50–64",
  "65-79": "65–79",
  "80+": "≥80",
}

CONTEXT_MODELS: tuple[tuple[str, str], ...] = (
  ("p_c", "M_C (C0 only)"),
  ("p_a", "M_A (C0 + CIIS)"),
  ("p_b", "M_B (C0 + D-BETA)"),
  ("p_ab", "M_AB (C0 + CIIS + D-BETA)"),
)
ECG_ONLY_MODELS: tuple[tuple[str, str], ...] = (
  ("p_a_ecg", "M_A_ECG (CIIS)"),
  ("p_b_ecg", "M_B_ECG (D-BETA)"),
)


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------


def _logit(p: npt.NDArray[np.float64], eps: float = 1e-15) -> npt.NDArray[np.float64]:
  clipped = np.clip(p, eps, 1.0 - eps)
  return np.log(clipped / (1.0 - clipped))


def calibration_in_the_large(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
  max_iter: int = 50,
  tol: float = 1e-10,
) -> float:
  """Intercept of ``logit(P(y=1)) = a + logit(p)`` with the slope fixed at 1.

  This is the TRIPOD calibration-in-the-large estimate: 0 means the mean predicted risk
  matches the observed rate after accounting for the logit scale. It differs from the
  intercept of the joint (intercept, slope) recalibration fit that the probe reports.
  """
  yt = np.asarray(y_true, dtype=np.float64)
  offset = _logit(np.asarray(y_prob, dtype=np.float64))
  if len(yt) == 0 or len(np.unique(yt)) < 2:
    return 0.0
  a = 0.0
  for _ in range(max_iter):
    mu = 1.0 / (1.0 + np.exp(-(a + offset)))
    grad = float(np.sum(yt - mu))
    hess = float(np.sum(mu * (1.0 - mu)))
    if hess <= 0.0:
      break
    step = grad / hess
    a += step
    if abs(step) < tol:
      break
  return float(a)


@dataclass(frozen=True)
class Interval:
  """Point estimate with a percentile bootstrap interval."""

  point: float
  ci_lower: float
  ci_upper: float


@dataclass(frozen=True)
class CalibrationSummary:
  """Calibration of one frozen prediction vector on the locked test."""

  model: str
  label: str
  n: int
  n_events: int
  observed_rate: float
  mean_predicted: float
  observed_expected_ratio: Interval
  citl_intercept: Interval
  joint_intercept: Interval
  slope: Interval
  n_bootstraps: int
  seed: int
  bootstrap: str = "ordinary_patient"


def _calibration_point(y: npt.NDArray[np.int64], p: npt.NDArray[np.float64]) -> tuple[float, float, float, float]:
  intercept, slope = fit_calibration_line(y, p)
  mean_p = float(np.mean(p))
  oe = float(np.mean(y) / mean_p) if mean_p > 0 else float("nan")
  return oe, calibration_in_the_large(y, p), intercept, slope


def calibration_summary(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
  model: str,
  label: str,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> CalibrationSummary:
  """Observed/expected, calibration-in-the-large, joint intercept, and slope with CIs.

  The bootstrap resamples patients without stratifying on the outcome (decision M3):
  stratification would hold the observed event rate fixed and understate the uncertainty
  of observed/expected and calibration-in-the-large.
  """
  y = np.asarray(y_true, dtype=np.int64)
  p = np.asarray(y_prob, dtype=np.float64)
  point = _calibration_point(y, p)
  rng = np.random.default_rng(seed)
  draws = np.empty((n_bootstraps, 4), dtype=np.float64)
  n = len(y)
  for b in range(n_bootstraps):
    idx = rng.integers(0, n, size=n)
    draws[b] = _calibration_point(y[idx], p[idx])
  lo = np.nanquantile(draws, 0.025, axis=0)
  hi = np.nanquantile(draws, 0.975, axis=0)
  intervals = [Interval(float(point[i]), float(lo[i]), float(hi[i])) for i in range(4)]
  return CalibrationSummary(
    model=model,
    label=label,
    n=int(n),
    n_events=int(y.sum()),
    observed_rate=float(np.mean(y)) if n else float("nan"),
    mean_predicted=float(np.mean(p)) if n else float("nan"),
    observed_expected_ratio=intervals[0],
    citl_intercept=intervals[1],
    joint_intercept=intervals[2],
    slope=intervals[3],
    n_bootstraps=n_bootstraps,
    seed=seed,
  )


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
  """Wilson score interval for a binomial proportion."""
  if n <= 0:
    return float("nan"), float("nan")
  phat = k / n
  denom = 1.0 + z * z / n
  centre = (phat + z * z / (2 * n)) / denom
  half = z * np.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n)) / denom
  return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def calibration_curve_rows(
  y_true: npt.NDArray[np.int64] | Sequence[int],
  y_prob: npt.NDArray[np.float64] | Sequence[float],
  model: str,
  n_bins: int = 10,
) -> list[dict[str, Any]]:
  """Observed versus mean predicted risk by quantile bin of predicted risk.

  Bins with fewer than ``MIN_CELL_COUNT`` events or non-events report the observed
  values as suppressed so a published bin cannot round-trip a small count.
  """
  y = np.asarray(y_true, dtype=np.int64)
  p = np.asarray(y_prob, dtype=np.float64)
  order = np.argsort(p, kind="mergesort")
  rows: list[dict[str, Any]] = []
  for b, idx in enumerate(np.array_split(order, n_bins), start=1):
    n = int(len(idx))
    k = int(y[idx].sum())
    small = k < MIN_CELL_COUNT or (n - k) < MIN_CELL_COUNT
    lo, hi = wilson_interval(k, n)
    rows.append({
      "model": model,
      "bin": b,
      "n": n,
      "mean_predicted": float(np.mean(p[idx])) if n else float("nan"),
      "n_events": SUPPRESSED if small else k,
      "observed_rate": None if small else (k / n if n else float("nan")),
      "observed_ci_lower": None if small else lo,
      "observed_ci_upper": None if small else hi,
    })
  return rows


# -----------------------------------------------------------------------------
# Canonical CIIS continuity reference
# -----------------------------------------------------------------------------


def _interval(boot: Any) -> dict[str, float]:
  return {"point": boot.point_estimate, "ci_lower": boot.ci_lower, "ci_upper": boot.ci_upper}


def _single_metric_interval(
  y: npt.NDArray[np.int64],
  s: npt.NDArray[np.float64],
  metric: str,
  n_bootstraps: int,
  seed: int,
) -> dict[str, float]:
  """Outcome-stratified bootstrap interval for one score's AUROC or AUPRC."""
  fn = compute_auroc if metric == "auroc" else compute_auprc
  rng = np.random.default_rng(seed)
  pos = np.where(y == 1)[0]
  neg = np.where(y == 0)[0]
  vals = np.empty(n_bootstraps, dtype=np.float64)
  for b in range(n_bootstraps):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    vals[b] = fn(y[idx], s[idx])
  lo, hi = np.quantile(vals, [0.025, 0.975])
  return {"point": fn(y, s), "ci_lower": float(lo), "ci_upper": float(hi)}


def canonical_ciis_reference(
  y_true: npt.NDArray[np.int64],
  ciis_score: npt.NDArray[np.float64],
  categories: Sequence[str],
  p_a_ecg: npt.NDArray[np.float64],
  p_b_ecg: npt.NDArray[np.float64],
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, Any]:
  """Discrimination of the unfitted canonical CIIS total score (decision M2).

  The score was built to classify infarction/injury, not to predict mortality, so it is a
  continuity reference against the fitted CIIS-feature probe, not a fair comparator.
  """
  y = np.asarray(y_true, dtype=np.int64)
  s = np.asarray(ciis_score, dtype=np.float64)
  pa = np.asarray(p_a_ecg, dtype=np.float64)
  pb = np.asarray(p_b_ecg, dtype=np.float64)
  out: dict[str, Any] = {
    "n": int(len(y)),
    "n_events": int(y.sum()),
    "auroc": _single_metric_interval(y, s, "auroc", n_bootstraps, seed),
    "auprc": _single_metric_interval(y, s, "auprc", n_bootstraps, seed),
    "probe_a_ecg_minus_score_auroc": _interval(
      bootstrap_paired_difference(y, s, pa, metric="auroc", n_bootstraps=n_bootstraps, seed=seed)
    ),
    "probe_b_ecg_minus_score_auroc": _interval(
      bootstrap_paired_difference(y, s, pb, metric="auroc", n_bootstraps=n_bootstraps, seed=seed)
    ),
    "probe_a_ecg_minus_score_auprc": _interval(
      bootstrap_paired_difference(y, s, pa, metric="auprc", n_bootstraps=n_bootstraps, seed=seed)
    ),
    "n_bootstraps": n_bootstraps,
    "seed": seed,
  }
  cats = np.asarray([str(c) for c in categories])
  rows: list[dict[str, Any]] = []
  for cat in ("normal", "borderline", "possible_injury", "probable_infarction"):
    mask = cats == cat
    n = int(mask.sum())
    k = int(y[mask].sum())
    small = k < MIN_CELL_COUNT or (n - k) < MIN_CELL_COUNT
    rows.append({
      "category": cat,
      "n": n if n >= MIN_CELL_COUNT else SUPPRESSED,
      "n_events": SUPPRESSED if small else k,
      "mortality": None if small else k / n,
    })
  out["by_category"] = rows
  return out


# -----------------------------------------------------------------------------
# Symmetric outcome-ascertainment sensitivity
# -----------------------------------------------------------------------------


def symmetric_ascertainment_expr() -> pl.Expr:
  """True when the 30-day window lies inside the death-record window, whatever the outcome.

  The primary rule treats any observed death as ascertained, while a survivor is kept
  only when ``ECG + 30 days <= last discharge + 1 year``. This expression applies the
  survivor rule to everyone, so retention cannot depend on the outcome (decision M4).
  """
  return (
    pl.col("outcome_ascertainment_end").is_not_null()
    & ((pl.col("ecg_datetime") + pl.duration(days=30)) <= pl.col("outcome_ascertainment_end"))
  )


def _gain_block(
  y: npt.NDArray[np.int64],
  pa: npt.NDArray[np.float64],
  pb: npt.NDArray[np.float64],
  n_bootstraps: int,
  seed: int,
) -> dict[str, Any]:
  return {
    "n": int(len(y)),
    "n_events": int(y.sum()),
    "auroc_a": compute_auroc(y, pa),
    "auroc_b": compute_auroc(y, pb),
    "b_minus_a_auroc": _interval(
      bootstrap_paired_difference(y, pa, pb, metric="auroc", n_bootstraps=n_bootstraps, seed=seed)
    ),
    "b_minus_a_auprc": _interval(
      bootstrap_paired_difference(y, pa, pb, metric="auprc", n_bootstraps=n_bootstraps, seed=seed)
    ),
  }


def ascertainment_sensitivity(
  frame: pl.DataFrame,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, Any]:
  """B−A contrasts on the full locked test and on the symmetric-ascertainment subset.

  ``frame`` holds one row per paired-valid locked-test patient with ``mortality_30d``,
  ``symmetric_ascertainable``, ``p_a``, ``p_b``, ``p_ab``, ``p_a_ecg``, and ``p_b_ecg``.
  """
  out: dict[str, Any] = {}
  sym = frame["symmetric_ascertainable"].to_numpy().astype(bool)
  y_all = frame["mortality_30d"].to_numpy().astype(np.int64)
  dropped = ~sym
  out["n_dropped"] = int(dropped.sum())
  out["n_dropped_events"] = int(y_all[dropped].sum())
  for name, subset in (("primary", np.ones(len(sym), dtype=bool)), ("symmetric", sym)):
    sub = frame.filter(pl.Series(subset))
    y = sub["mortality_30d"].to_numpy().astype(np.int64)
    out[name] = {
      "ecg_only": _gain_block(
        y,
        sub["p_a_ecg"].to_numpy().astype(np.float64),
        sub["p_b_ecg"].to_numpy().astype(np.float64),
        n_bootstraps,
        seed,
      ),
      "context_adjusted": _gain_block(
        y,
        sub["p_a"].to_numpy().astype(np.float64),
        sub["p_b"].to_numpy().astype(np.float64),
        n_bootstraps,
        seed,
      ),
      "add_a_to_b_auroc": _interval(
        bootstrap_paired_difference(
          y,
          sub["p_b"].to_numpy().astype(np.float64),
          sub["p_ab"].to_numpy().astype(np.float64),
          metric="auroc",
          n_bootstraps=n_bootstraps,
          seed=seed,
        )
      ),
    }
  return out


# -----------------------------------------------------------------------------
# Table 1
# -----------------------------------------------------------------------------


def _fmt_count(k: int, n: int) -> str:
  if k < MIN_CELL_COUNT:
    return SUPPRESSED
  return f"{k:,} ({100.0 * k / n:.1f})" if n else f"{k:,}"


def _categorical_cells(values: npt.NDArray[Any], levels: Sequence[str]) -> dict[str, str]:
  """Formatted ``n (%)`` per level with complementary suppression.

  When exactly one level is suppressed, the next-smallest level is also suppressed so the
  hidden count cannot be recovered by subtraction from the group total.
  """
  n = len(values)
  counts = {lvl: int(np.sum(values == lvl)) for lvl in levels}
  if sum(counts.values()) != n:
    unknown = sorted({str(v) for v in values} - set(levels))
    raise ValueError(f"Table 1 levels do not cover the data; unexpected values: {unknown}")
  cells = {lvl: _fmt_count(counts[lvl], n) for lvl in levels}
  hidden = [lvl for lvl in levels if cells[lvl] == SUPPRESSED and counts[lvl] > 0]
  if len(hidden) == 1:
    visible = sorted((counts[lvl], lvl) for lvl in levels if cells[lvl] != SUPPRESSED)
    if visible:
      cells[visible[0][1]] = SUPPRESSED
  return cells


def table1_rows(frame: pl.DataFrame, elix_cols: Sequence[str]) -> list[dict[str, str]]:
  """Cohort characteristics by partition (wide rows: variable, level, one column per group).

  ``frame`` columns: ``split``, ``test_analytic`` (paired-valid locked test), ``age_at_ecg``,
  ``sex_female``, ``race_group``, ``setting``, Elixhauser indicators, and the boolean
  mortality endpoints (null where a horizon is not ascertainable).
  """
  groups: dict[str, pl.DataFrame] = {
    "overall": frame,
    "dev": frame.filter(pl.col("split") == "dev"),
    "val": frame.filter(pl.col("split") == "val"),
    "test": frame.filter(pl.col("split") == "test"),
    "test_analytic": frame.filter(pl.col("test_analytic")),
  }
  rows: list[dict[str, str]] = []

  def add(variable: str, level: str, cells: dict[str, str]) -> None:
    rows.append({"variable": variable, "level": level, **cells})

  add("Patients", "n", {g: f"{df.height:,}" for g, df in groups.items()})

  age_cells_mean: dict[str, str] = {}
  age_cells_med: dict[str, str] = {}
  for g, df in groups.items():
    a = df["age_at_ecg"].to_numpy().astype(np.float64)
    q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
    age_cells_mean[g] = f"{a.mean():.1f} ({a.std(ddof=1):.1f})"
    age_cells_med[g] = f"{med:.0f} [{q1:.0f}–{q3:.0f}]"
  add("Age at ECG, years", "mean (SD)", age_cells_mean)
  add("Age at ECG, years", "median [IQR]", age_cells_med)

  age_band = {g: assign_age_subgroup(df["age_at_ecg"].to_numpy()) for g, df in groups.items()}
  band_cells = {g: _categorical_cells(age_band[g], list(AGE_LABELS)) for g in groups}
  for key, label in AGE_LABELS.items():
    add("Age band, n (%)", label, {g: band_cells[g][key] for g in groups})

  add(
    "Female sex, n (%)",
    "",
    {g: _fmt_count(int(df["sex_female"].sum()), df.height) for g, df in groups.items()},
  )

  race_cells = {
    g: _categorical_cells(df["race_group"].fill_null("unknown_other").to_numpy(), list(RACE_LABELS))
    for g, df in groups.items()
  }
  for key, label in RACE_LABELS.items():
    add("Race and ethnicity, n (%)", label, {g: race_cells[g][key] for g in groups})

  setting_cells = {
    g: _categorical_cells(df["setting"].fill_null("other_or_unlinked").to_numpy(), list(SETTING_LABELS))
    for g, df in groups.items()
  }
  for key, label in SETTING_LABELS.items():
    add("Care setting at ECG, n (%)", label, {g: setting_cells[g][key] for g in groups})

  count_bands = ("0", "1–2", "≥3")
  band_values: dict[str, npt.NDArray[Any]] = {}
  for g, df in groups.items():
    k = df.select(pl.sum_horizontal([pl.col(c).cast(pl.Int64) for c in elix_cols])).to_series().to_numpy()
    band_values[g] = np.where(k == 0, "0", np.where(k <= 2, "1–2", "≥3"))
  count_cells = {g: _categorical_cells(band_values[g], list(count_bands)) for g in groups}
  for band in count_bands:
    add("Prior Elixhauser conditions, n (%)", band, {g: count_cells[g][band] for g in groups})
  for col in elix_cols:
    label = col.removeprefix("elix_").replace("_", " ")
    add(
      "Prior Elixhauser condition, n (%)",
      label,
      {g: _fmt_count(int(df[col].cast(pl.Int64).sum()), df.height) for g, df in groups.items()},
    )

  for col, label in (
    ("mortality_30d", "30-day mortality"),
    ("mortality_90d", "90-day mortality"),
    ("mortality_1yr", "1-year mortality"),
  ):
    cells: dict[str, str] = {}
    for g, df in groups.items():
      known = df.filter(pl.col(col).is_not_null())
      k = int(known[col].cast(pl.Int64).sum())
      cells[g] = (
        f"{_fmt_count(k, known.height)}; ascertainable n={known.height:,}"
        if known.height != df.height
        else _fmt_count(k, known.height)
      )
    add("Outcome, n (%)", label, cells)
  return rows


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  pl.DataFrame(rows, infer_schema_length=None).write_csv(path)


def _plot_calibration(curves: list[dict[str, Any]], labels: dict[str, str], path: Path) -> None:
  import matplotlib  # noqa: PLC0415 — optional plotting dependency

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # noqa: PLC0415

  fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
  panels = (
    ("ECG only", [m for m, _ in ECG_ONLY_MODELS]),
    ("With C0 context", [m for m, _ in CONTEXT_MODELS]),
  )
  for ax, (title, models) in zip(axes, panels):
    top = 0.0
    for model in models:
      pts = [r for r in curves if r["model"] == model and r["observed_rate"] is not None]
      x = np.asarray([r["mean_predicted"] for r in pts])
      yv = np.asarray([r["observed_rate"] for r in pts])
      lo = np.asarray([r["observed_ci_lower"] for r in pts])
      hi = np.asarray([r["observed_ci_upper"] for r in pts])
      ax.errorbar(x, yv, yerr=[yv - lo, hi - yv], marker="o", ms=4, capsize=2, label=labels[model])
      top = max(top, float(np.max(hi)) if len(hi) else 0.0, float(np.max(x)) if len(x) else 0.0)
    lim = top * 1.05
    ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(title)
    ax.set_xlabel("Mean predicted 30-day risk (decile)")
    ax.legend(fontsize=8, loc="upper left")
  axes[0].set_ylabel("Observed 30-day mortality (95% Wilson CI)")
  fig.suptitle("Calibration on the MIMIC locked test (frozen probes, no recalibration)", fontsize=10)
  fig.tight_layout()
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(path, dpi=200)
  fig.savefig(path.with_suffix(".svg"))
  plt.close(fig)


def run_manuscript_reporting(
  cohort_path: Path,
  c0_path: Path,
  setting_path: Path,
  pair_validity_path: Path,
  predictions_context_path: Path,
  predictions_ecg_only_path: Path,
  race_map: pl.DataFrame,
  output_dir: Path,
  tables_dir: Path,
  figures_dir: Path,
  source_sha: str,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, Any]:
  """Build every S2 output and return the machine-readable report.

  ``race_map`` maps ``subject_id`` to ``race_group`` (earliest non-null admission race,
  the same rule the RQ4 subgroup analysis uses).
  """
  cohort = pl.read_parquet(cohort_path)
  c0 = pl.read_parquet(c0_path)
  setting = pl.read_parquet(setting_path).select(["subject_id", "study_id", "setting"])
  pairs = pl.read_parquet(pair_validity_path)
  pred_ctx = pl.read_parquet(predictions_context_path)
  pred_ecg = pl.read_parquet(predictions_ecg_only_path)
  elix_cols = [c for c in c0.columns if c.startswith("elix_")]

  test_ids = pred_ctx.select(["subject_id", "study_id"]).with_columns(pl.lit(True).alias("test_analytic"))
  frame = (
    cohort.select([
      "subject_id", "study_id", "split", "ecg_datetime", "outcome_ascertainment_end",
      "mortality_30d", "mortality_90d", "mortality_1yr",
    ])
    .join(c0.select(["subject_id", "study_id", "age_at_ecg", "sex_female", *elix_cols]), on=["subject_id", "study_id"], how="left")
    .join(setting, on=["subject_id", "study_id"], how="left")
    .join(race_map.select(["subject_id", "race_group"]), on="subject_id", how="left")
    .join(test_ids, on=["subject_id", "study_id"], how="left")
    .with_columns(
      pl.col("test_analytic").fill_null(False),
      symmetric_ascertainment_expr().alias("symmetric_ascertainable"),
    )
  )
  if frame.filter(pl.col("test_analytic")).height != pred_ctx.height:
    raise ValueError("Locked-test predictions do not join one-to-one onto the cohort.")

  t1 = table1_rows(frame, elix_cols)
  _write_csv(t1, tables_dir / "table1_cohort_characteristics.csv")

  test = (
    pred_ctx.join(
      pred_ecg.select(["subject_id", "study_id", "p_a_ecg", "p_b_ecg"]),
      on=["subject_id", "study_id"],
      how="inner",
    )
    .join(
      frame.select(["subject_id", "study_id", "symmetric_ascertainable"]),
      on=["subject_id", "study_id"],
      how="left",
    )
    .join(
      pairs.select(["subject_id", "study_id", "ciis_total_score", "ciis_category"]),
      on=["subject_id", "study_id"],
      how="left",
    )
  )
  if test.height != pred_ctx.height:
    raise ValueError("ECG-only and context-adjusted locked-test predictions are not paired.")
  y = test["mortality_30d"].to_numpy().astype(np.int64)

  labels = dict(CONTEXT_MODELS + ECG_ONLY_MODELS)
  calib: list[CalibrationSummary] = []
  curves: list[dict[str, Any]] = []
  for col, label in ECG_ONLY_MODELS + CONTEXT_MODELS:
    p = test[col].to_numpy().astype(np.float64)
    calib.append(calibration_summary(y, p, col, label, n_bootstraps=n_bootstraps, seed=seed))
    curves.extend(calibration_curve_rows(y, p, col))
  calib_rows = [
    {
      "model": c.label,
      "n": c.n,
      "n_events": c.n_events,
      "observed_rate": c.observed_rate,
      "mean_predicted": c.mean_predicted,
      **{
        f"{name}_{part}": getattr(getattr(c, name), part)
        for name in ("observed_expected_ratio", "citl_intercept", "joint_intercept", "slope")
        for part in ("point", "ci_lower", "ci_upper")
      },
    }
    for c in calib
  ]
  _write_csv(calib_rows, tables_dir / "calibration_summary.csv")
  _write_csv(
    [{**r, "model": labels[r["model"]]} for r in curves],
    tables_dir / "calibration_curve_deciles.csv",
  )
  _plot_calibration(curves, labels, figures_dir / "supp_calibration_curves.png")

  ciis = canonical_ciis_reference(
    y,
    test["ciis_total_score"].to_numpy().astype(np.float64),
    test["ciis_category"].fill_null("missing").to_list(),
    test["p_a_ecg"].to_numpy().astype(np.float64),
    test["p_b_ecg"].to_numpy().astype(np.float64),
    n_bootstraps=n_bootstraps,
    seed=seed,
  )
  _write_csv(ciis["by_category"], tables_dir / "canonical_ciis_by_category.csv")

  asc = ascertainment_sensitivity(test, n_bootstraps=n_bootstraps, seed=seed)
  cohort_asc = frame.group_by(["symmetric_ascertainable", "mortality_30d"]).len().sort(
    ["symmetric_ascertainable", "mortality_30d"]
  )
  asc["cohort_counts"] = cohort_asc.to_dicts()

  report: dict[str, Any] = {
    "status": "complete",
    "analysis_status": "evaluation_only_frozen_predictions",
    "source_sha": source_sha,
    "decisions": ["M2", "M3", "M4"],
    "n_cohort": frame.height,
    "n_test_analytic": test.height,
    "n_test_events": int(y.sum()),
    "calibration": [asdict(c) for c in calib],
    "canonical_ciis": ciis,
    "ascertainment_sensitivity": asc,
    "outputs": {
      "table1": str(tables_dir / "table1_cohort_characteristics.csv"),
      "calibration_summary": str(tables_dir / "calibration_summary.csv"),
      "calibration_curves": str(tables_dir / "calibration_curve_deciles.csv"),
      "calibration_figure": str(figures_dir / "supp_calibration_curves.png"),
      "canonical_ciis_by_category": str(tables_dir / "canonical_ciis_by_category.csv"),
    },
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "manuscript_reporting.json").write_text(
    json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
  )
  return report


PAIRED_VALID_GROUPS: tuple[str, ...] = ("both_valid", "a_only_valid", "b_only_valid", "neither_valid")


def _audit_group(frame: pl.DataFrame, setting_levels: Sequence[str]) -> dict[str, Any]:
  """Aggregate summary for one validity group (small cells suppressed)."""
  n = frame.height
  deaths = int(frame["mortality_30d"].sum()) if n else 0
  age = frame["age_at_ecg"].cast(pl.Float64, strict=False).drop_nulls()
  # polars' Series.mean()/median() stubs widen to PythonLiteral | None; reduce via
  # numpy on the Float64 column so the scalar type is explicit.
  age_array = age.to_numpy()
  age_mean = float(np.mean(age_array)) if age_array.size else None
  age_median = float(np.median(age_array)) if age_array.size else None
  pct_female = round(100.0 * float((frame["gender"] == "F").sum()) / n, 2) if n else None
  row: dict[str, Any] = {
    "n": n,
    "n_deaths_30d": deaths if deaths >= MIN_CELL_COUNT else SUPPRESSED,
    "mortality_rate_30d": round(deaths / n, 5) if n and deaths >= MIN_CELL_COUNT else SUPPRESSED,
    "age_mean": round(age_mean, 2) if age_mean is not None else None,
    "age_median": round(age_median, 2) if age_median is not None else None,
    "pct_female": pct_female,
  }
  for level in setting_levels:
    k = int((frame["setting"] == level).sum()) if n else 0
    row[f"n_setting_{level}"] = k if k >= MIN_CELL_COUNT else SUPPRESSED
  return row


def run_paired_valid_exclusion_audit(
  cohort_path: Path,
  pair_validity_path: Path,
  setting_path: Path,
  output_dir: Path,
  tables_dir: Path,
  source_sha: str,
) -> dict[str, Any]:
  """Characterize locked-test patients excluded from the paired-valid analysis (round-1 audit item 5).

  Creates four groups on the locked test — both representations valid, A-only
  valid, B-only valid, neither valid — and summarizes counts, 30-day mortality,
  age, sex, and care setting. Extraction-error categories are cross-tabulated
  against the validity group (round-2 audit item 3) rather than aggregated over
  all excluded patients, using normalized categories in committed tables.
  Aggregate-only; cells below ``MIN_CELL_COUNT``
  are suppressed. No model is fitted and no frozen prediction is touched.
  """
  cohort = pl.read_parquet(cohort_path).filter(pl.col("split") == "test")
  pairs = pl.read_parquet(pair_validity_path).select(
    ["subject_id", "study_id", "representation_a_valid", "representation_b_valid", "representation_a_error", "representation_b_error"]
  )
  setting = pl.read_parquet(setting_path).select(["subject_id", "study_id", "setting"])
  frame = (
    cohort.select(["subject_id", "study_id", "age_at_ecg", "gender", "mortality_30d"])
    .join(pairs, on=["subject_id", "study_id"], how="left")
    .join(setting, on=["subject_id", "study_id"], how="left")
  )
  if frame.height != cohort.height:
    raise ValueError("Pair-validity or setting table does not join one-to-one onto the locked test.")
  n_test = frame.height
  n_pair_valid = int((frame["representation_a_valid"] & frame["representation_b_valid"]).sum())
  group_expr = (
    pl.when(pl.col("representation_a_valid") & pl.col("representation_b_valid"))
    .then(pl.lit("both_valid"))
    .when(pl.col("representation_a_valid") & ~pl.col("representation_b_valid"))
    .then(pl.lit("a_only_valid"))
    .when(~pl.col("representation_a_valid") & pl.col("representation_b_valid"))
    .then(pl.lit("b_only_valid"))
    .otherwise(pl.lit("neither_valid"))
  )
  frame = frame.with_columns(group_expr.alias("validity_group"))
  setting_levels = sorted(s for s in frame["setting"].unique().to_list() if s is not None)
  rows: list[dict[str, Any]] = []
  summaries: dict[str, Any] = {}
  for group in PAIRED_VALID_GROUPS:
    g = frame.filter(pl.col("validity_group") == group)
    s = _audit_group(g, setting_levels)
    s["pct_of_locked_test"] = round(100.0 * s["n"] / n_test, 2)
    rows.append({"validity_group": group, **s})
    summaries[group] = s
  excluded = frame.filter(pl.col("validity_group") != "both_valid")
  # Round-2 audit item 3: error categories must cross-tab against the validity
  # group — an aggregate over all excluded patients mixes A-side, B-side, and
  # neither-side failures and is uninterpretable. Raw error strings (which can
  # embed local paths) stay in the source parquet; committed tables carry the
  # normalized category.
  def _category_counts(group_frame: pl.DataFrame, side: str) -> list[dict[str, Any]]:
    col = pl.col(f"representation_{side}_error").fill_null("none")
    category = (
      pl.when(col == "none")
      .then(pl.lit("none"))
      .when(col.str.contains("NaN") | col.str.contains("Inf"))
      .then(pl.lit("waveform_nan_inf"))
      .when(col.str.contains("No such file"))
      .then(pl.lit("waveform_file_missing"))
      .when(col.str.contains("R-peaks"))
      .then(pl.lit("ciis_insufficient_rpeaks"))
      .otherwise(pl.lit("other"))
      .alias("error_category")
    )
    return group_frame.select(category).group_by("error_category").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts()

  error_counts_by_group: list[dict[str, Any]] = []
  for group in ("a_only_valid", "b_only_valid", "neither_valid"):
    g = frame.filter(pl.col("validity_group") == group)
    for side in ("a", "b"):
      for c in _category_counts(g, side):
        error_counts_by_group.append({"validity_group": group, "representation": f"{side}_error", "error_category": c["error_category"], "n": c["n"]})
  error_counts: list[dict[str, Any]] = []
  for side in ("a", "b"):
    for c in _category_counts(excluded, side):
      error_counts.append({"representation": f"{side}_error", "error_category": c["error_category"], "n": c["n"]})
  report = {
    "status": "complete",
    "analysis_status": "evaluation_only_frozen_extractions",
    "source_sha": source_sha,
    "n_locked_test": n_test,
    "n_paired_valid": n_pair_valid,
    "n_excluded": n_test - n_pair_valid,
    "groups": rows,
    "excluded_error_categories": error_counts,
    "error_counts_by_group": error_counts_by_group,
    "outputs": {
      "table": str(tables_dir / "supp_mimic_paired_valid_exclusion_audit.csv"),
      "error_categories_by_group_table": str(tables_dir / "supp_paired_valid_error_categories_by_group.csv"),
    },
  }
  _write_csv(rows, tables_dir / "supp_mimic_paired_valid_exclusion_audit.csv")
  _write_csv(error_counts_by_group, tables_dir / "supp_paired_valid_error_categories_by_group.csv")
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "paired_valid_exclusion_audit.json").write_text(
    json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
  )
  return report


#: Contrasts covered by the bootstrap-estimand sensitivity: frozen prediction
#: columns per (estimand, A, B), matching the Table 3 headline rows.
BOOTSTRAP_SENSITIVITY_CONTRASTS: tuple[tuple[str, str, str, str, str], ...] = (
  ("ecg_only", "ECG-only: B - A", "mortality_30d", "p_a_ecg", "p_b_ecg"),
  ("context_adjusted", "Context-adjusted: B - A", "mortality_30d", "p_a", "p_b"),
)

BOOTSTRAP_SENSITIVITY_METRICS: tuple[Literal["auroc", "auprc", "brier", "log_loss"], ...] = (
  "auroc",
  "auprc",
  "brier",
  "log_loss",
)


def run_bootstrap_estimand_sensitivity(
  predictions_ecg_path: Path,
  predictions_context_path: Path,
  analysis_results_path: Path,
  output_dir: Path,
  tables_dir: Path,
  source_sha: str,
  n_bootstraps: int = 1000,
  seed: int = 42,
) -> dict[str, Any]:
  """Compare outcome-stratified and ordinary paired patient bootstrap intervals (round-1 audit item 8).

  For the two primary headline contrasts (ECG-only B-A, context-adjusted B-A)
  and the four reported metrics, this recomputes the frozen outcome-stratified
  interval (conditional on the observed test-set event count) and an ordinary
  unstratified paired patient-bootstrap interval that also resamples the event
  count. No model is fitted; only frozen locked-test prediction columns are
  read. The recomputed stratified intervals are cross-checked against the
  frozen headline values in ``scientific_analysis_results.json``.
  """
  predictions: dict[str, pl.DataFrame] = {
    "ecg_only": pl.read_parquet(predictions_ecg_path),
    "context_adjusted": pl.read_parquet(predictions_context_path),
  }
  frozen = json.loads(analysis_results_path.read_text(encoding="utf-8"))["analyses"]
  frozen_key = {"ecg_only": "rq1_ecg_only_headline", "context_adjusted": "rq1_context_adjusted_headline"}

  rows: list[dict[str, Any]] = []
  max_frozen_gap = 0.0
  for estimand, label, y_col, a_col, b_col in BOOTSTRAP_SENSITIVITY_CONTRASTS:
    frame = predictions[estimand]
    y = frame[y_col].to_numpy().astype(np.int64)
    sa = frame[a_col].to_numpy().astype(np.float64)
    sb = frame[b_col].to_numpy().astype(np.float64)
    for metric in BOOTSTRAP_SENSITIVITY_METRICS:
      strat = bootstrap_paired_difference(y, sa, sb, metric=metric, n_bootstraps=n_bootstraps, seed=seed, stratified=True)
      unstrat = bootstrap_paired_difference(y, sa, sb, metric=metric, n_bootstraps=n_bootstraps, seed=seed, stratified=False)
      fb = frozen[frozen_key[estimand]][metric]["bootstrap"]
      max_frozen_gap = max(
        max_frozen_gap,
        abs(fb["point_estimate"] - strat.point_estimate),
        abs(fb["ci_lower"] - strat.ci_lower),
        abs(fb["ci_upper"] - strat.ci_upper),
      )
      rows.append({
        "estimand": estimand,
        "contrast": label,
        "metric": metric,
        "n": int(frame.height),
        "n_events": int(y.sum()),
        "stratified_point_estimate": strat.point_estimate,
        "stratified_ci_lower": strat.ci_lower,
        "stratified_ci_upper": strat.ci_upper,
        "unstratified_point_estimate": unstrat.point_estimate,
        "unstratified_ci_lower": unstrat.ci_lower,
        "unstratified_ci_upper": unstrat.ci_upper,
        "frozen_point_estimate": fb["point_estimate"],
        "frozen_ci_lower": fb["ci_lower"],
        "frozen_ci_upper": fb["ci_upper"],
        "n_bootstraps": n_bootstraps,
        "seed": seed,
      })
  report = {
    "status": "complete",
    "analysis_status": "post_hoc_sensitivity_frozen_predictions",
    "source_sha": source_sha,
    "n_bootstraps": n_bootstraps,
    "seed": seed,
    "max_abs_gap_recomputed_vs_frozen_stratified": max_frozen_gap,
    "outputs": {"table": str(tables_dir / "supp_bootstrap_estimand_sensitivity.csv")},
  }
  _write_csv(rows, tables_dir / "supp_bootstrap_estimand_sensitivity.csv")
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "bootstrap_estimand_sensitivity.json").write_text(
    json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
  )
  return report
