"""LM5: refit-aware uncertainty for the headline B-A discrimination contrasts.

Three-surface paired patient bootstrap: every replicate bootstraps development,
validation, and locked-test patients with replacement and repeats the entire
probe-fitting process (development-only standardization, independent validation
C tuning on the original grid, dev fit) for the headline models A, B, C0+A,
C0+B. A and B share the identical patient resample within every replicate.

Decomposition (frozen in docs/plan_20261007_ecg-rep-limitation-mitigation/
LM5_refit_aware_uncertainty.md before execution): a training-only variant
(resample dev+val, fixed original test) and a test-only variant (fixed primary
fits, resampled test) identify which surface dominates variation.

Master seed 52026 with deterministic child seeds per replicate (master+r for
the main run and the training-only run; master+10000+r for test-only so the
test resamples differ from the main-run test resamples). Ordinary patient
bootstrap is the unconditional sensitivity target; the manuscript's primary
fixed-model paired-bootstrap interval is compared, not replaced.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ecg_rep.probe import (
  ProbeConfig,
  TrainedProbe,
  compute_auprc,
  compute_auroc,
  compute_binary_log_loss,
  compute_brier_score,
)

MASTER_SEED = 52026
MODEL_KEYS: tuple[str, ...] = ("A", "B", "C0A", "C0B")
MODEL_DISPLAY: dict[str, str] = {"A": "A", "B": "B", "C0A": "C0+A", "C0B": "C0+B"}
CONTRAST_KEYS: tuple[str, ...] = (
  "contrast_ecg_b_minus_a_auroc",
  "contrast_ecg_b_minus_a_auprc",
  "contrast_ctx_b_minus_a_auroc",
  "contrast_ctx_b_minus_a_auprc",
  "contrast_ctx_b_minus_a_log_loss",
  "contrast_ctx_b_minus_a_brier",
)


@dataclass(frozen=True)
class RefitSpec:
  """Frozen model feature lists over one shared column order."""

  all_columns: tuple[str, ...]
  model_columns: dict[str, tuple[str, ...]]
  column_index: dict[str, np.ndarray]

  @classmethod
  def from_frame(cls, frame: pl.DataFrame) -> RefitSpec:
    from ecg_rep.evaluate import C0_FEATURE_COLS  # pyright: ignore[reportPrivateUsage]

    c0_cols = [c for c in C0_FEATURE_COLS if c in frame.columns]
    za_cols = [c for c in frame.columns if c.startswith("za_")]
    zb_cols = [c for c in frame.columns if c.startswith("zb_")]
    all_cols = [*c0_cols, *za_cols, *zb_cols]
    model_columns = {
      "A": tuple(za_cols),
      "B": tuple(zb_cols),
      "C0A": tuple([*c0_cols, *za_cols]),
      "C0B": tuple([*c0_cols, *zb_cols]),
    }
    lookup = {c: i for i, c in enumerate(all_cols)}
    index = {k: np.array([lookup[c] for c in v], dtype=np.intp) for k, v in model_columns.items()}
    return cls(all_columns=tuple(all_cols), model_columns=model_columns, column_index=index)


def build_refit_frame(
  cohort_path: Path,
  c0_path: Path,
  meta_a_path: Path,
  features_a_path: Path,
  meta_b_path: Path,
  embeddings_b_path: Path,
) -> pl.DataFrame:
  """Primary paired-valid surface with C0 + A + B columns; identical to the
  primary context-adjusted evaluation frame (no PCA control here)."""
  from ecg_rep.evaluate import C0_FEATURE_COLS, _join_representation_matrix  # pyright: ignore[reportPrivateUsage]

  cohort = pl.read_parquet(cohort_path)
  c0 = pl.read_parquet(c0_path)
  meta_a = pl.read_parquet(meta_a_path)
  meta_b = pl.read_parquet(meta_b_path)
  za = np.load(features_a_path)
  zb = np.load(embeddings_b_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]
  frame = cohort.join(c0.select(["subject_id", "study_id", *c0_cols]), on=["subject_id", "study_id"])
  frame = _join_representation_matrix(frame, meta_a, za, "za", "representation_a_valid")
  frame = _join_representation_matrix(frame, meta_b, zb, "zb", "representation_b_valid")
  frame = frame.filter(
    (pl.col("representation_a_valid") == True)  # noqa: E712
    & (pl.col("representation_b_valid") == True)  # noqa: E712
  )
  return frame


def _bootstrap_split_indices(sub: pl.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
  """Patient bootstrap with replacement for one split.

  Returns (row-position array into ``sub`` with multiplicity, picked subject
  array). Subjects are sorted for determinism; every row of a picked subject
  is repeated, and the bootstrap has exactly as many rows as the split.
  """
  subject_ids = np.sort(sub["subject_id"].unique().to_numpy())
  rng = np.random.default_rng(seed)
  picks = rng.integers(0, len(subject_ids), size=len(subject_ids))
  picked = subject_ids[picks]
  positions = np.arange(len(sub), dtype=np.int64)
  subject_arr = sub["subject_id"].to_numpy()
  order = np.argsort(subject_arr, kind="stable")
  sorted_subjects = subject_arr[order]
  starts = np.searchsorted(sorted_subjects, picked, side="left")
  ends = np.searchsorted(sorted_subjects, picked, side="right")
  row_idx = np.concatenate([order[s:e] for s, e in zip(starts, ends, strict=True)])
  assert len(row_idx) == len(sub), "bootstrap row count must equal split size"
  return row_idx, picked


def _split_xy(sub: pl.DataFrame, spec: RefitSpec) -> tuple[np.ndarray, np.ndarray]:
  x = sub.select(list(spec.all_columns)).to_numpy().astype(np.float64)
  y = np.asarray(sub["mortality_30d"].to_numpy(), dtype=np.int64)
  return x, y


def fit_refit_probe(
  x_dev: np.ndarray,
  y_dev: np.ndarray,
  x_val: np.ndarray,
  y_val: np.ndarray,
  feature_names: list[str],
  model_name: str = "refit",
  cfg: ProbeConfig | None = None,
) -> TrainedProbe:
  """Single refit through the audited path: dev-only standardization and
  validation-only C tuning on the original grid."""
  from ecg_rep.probe import fit_logistic_probe

  return fit_logistic_probe(
    x_dev, y_dev, x_val, y_val, config=cfg, feature_names=list(feature_names), model_name=model_name,
  )


# Worker globals (initialized per process).
_FRAME: pl.DataFrame | None = None
_SPEC: RefitSpec | None = None


def _init_worker(frame_path: Path, spec: RefitSpec) -> None:
  global _FRAME, _SPEC  # noqa: PLW0603
  _FRAME = pl.read_parquet(frame_path)
  _SPEC = spec


def _resampled_xy(
  frame: pl.DataFrame, spec: RefitSpec, split: str, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Bootstrap one split; return (feature matrix, y, picked subjects)."""
  sub = frame.filter(pl.col("split") == split)
  idx, picked = _bootstrap_split_indices(sub, seed)
  x_all, y = _split_xy(sub, spec)
  return x_all[idx], y[idx], picked


def _fit_scores(
  spec: RefitSpec,
  x_dev: np.ndarray, y_dev: np.ndarray,
  x_val: np.ndarray, y_val: np.ndarray,
  x_test: np.ndarray, y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
  """Refit all four headline models on one resample; score the resampled test."""
  cfg = ProbeConfig()
  out: dict[str, dict[str, float]] = {}
  for key in MODEL_KEYS:
    cols_idx = spec.column_index[key]
    if len(np.unique(y_dev)) < 2 or len(np.unique(y_val)) < 2:
      out[key] = {m: float("nan") for m in ("auroc", "auprc", "log_loss", "brier", "selected_c")}
      continue
    probe = fit_refit_probe(
      x_dev[:, cols_idx], y_dev, x_val[:, cols_idx], y_val,
      feature_names=list(spec.model_columns[key]), model_name=f"refit_{key}", cfg=cfg,
    )
    p_test = probe.predict_proba(x_test[:, cols_idx])
    out[key] = {
      "auroc": float(compute_auroc(y_test, p_test)),
      "auprc": float(compute_auprc(y_test, p_test)),
      "log_loss": float(compute_binary_log_loss(y_test, p_test)),
      "brier": float(compute_brier_score(y_test, p_test)),
      "selected_c": float(probe.best_c),
    }
  return out


def _contrast_rows(scores: dict[str, dict[str, float]]) -> dict[str, float]:
  return {
    "contrast_ecg_b_minus_a_auroc": scores["B"]["auroc"] - scores["A"]["auroc"],
    "contrast_ecg_b_minus_a_auprc": scores["B"]["auprc"] - scores["A"]["auprc"],
    "contrast_ctx_b_minus_a_auroc": scores["C0B"]["auroc"] - scores["C0A"]["auroc"],
    "contrast_ctx_b_minus_a_auprc": scores["C0B"]["auprc"] - scores["C0A"]["auprc"],
    "contrast_ctx_b_minus_a_log_loss": scores["C0B"]["log_loss"] - scores["C0A"]["log_loss"],
    "contrast_ctx_b_minus_a_brier": scores["C0B"]["brier"] - scores["C0A"]["brier"],
  }


def _run_main_replicate(replicate: int, master_seed: int) -> dict[str, Any]:
  """Main three-surface bootstrap replicate (A/B share every resample)."""
  assert _FRAME is not None and _SPEC is not None
  seed = master_seed + replicate
  x_dev, y_dev, p_dev = _resampled_xy(_FRAME, _SPEC, "dev", seed)
  x_val, y_val, p_val = _resampled_xy(_FRAME, _SPEC, "val", seed)
  x_test, y_test, p_test = _resampled_xy(_FRAME, _SPEC, "test", seed)
  digest = hashlib.sha1(  # noqa: S324
    b"".join(p.tobytes() for p in (p_dev, p_val, p_test))
  ).hexdigest()[:12]
  scores = _fit_scores(_SPEC, x_dev, y_dev, x_val, y_val, x_test, y_test)
  row: dict[str, Any] = {"replicate": replicate, "resample_digest": digest}
  for key in MODEL_KEYS:
    for metric, value in scores[key].items():
      row[f"{key}__{metric}"] = value
  row.update(_contrast_rows(scores))
  return row


def _run_training_only_replicate(replicate: int, master_seed: int) -> dict[str, Any]:
  """Training-only variation: resample dev+val, fixed original test."""
  assert _FRAME is not None and _SPEC is not None
  seed = master_seed + replicate
  x_dev, y_dev, _ = _resampled_xy(_FRAME, _SPEC, "dev", seed)
  x_val, y_val, _ = _resampled_xy(_FRAME, _SPEC, "val", seed + 1_000_000)
  sub_test = _FRAME.filter(pl.col("split") == "test")
  x_test, y_test = _split_xy(sub_test, _SPEC)
  scores = _fit_scores(_SPEC, x_dev, y_dev, x_val, y_val, x_test, y_test)
  row: dict[str, Any] = {"replicate": replicate}
  row.update(_contrast_rows(scores))
  return row


def _fixed_probes(frame: pl.DataFrame, spec: RefitSpec) -> dict[str, TrainedProbe]:
  """Deterministic reproduction of the primary fits on the original dev/val."""
  dev = frame.filter(pl.col("split") == "dev")
  val = frame.filter(pl.col("split") == "val")
  x_dev, y_dev = _split_xy(dev, spec)
  x_val, y_val = _split_xy(val, spec)
  probes: dict[str, TrainedProbe] = {}
  for key in MODEL_KEYS:
    cols_idx = spec.column_index[key]
    probes[key] = fit_refit_probe(
      x_dev[:, cols_idx], y_dev, x_val[:, cols_idx], y_val,
      feature_names=list(spec.model_columns[key]), model_name=f"fixed_{key}", cfg=ProbeConfig(),
    )
  return probes


def _run_test_only_replicate(
  replicate: int, probes: dict[str, TrainedProbe], spec: RefitSpec, master_seed: int,
) -> dict[str, Any]:
  """Test-only variation: fixed primary fits, resampled locked test."""
  seed = master_seed + 10_000 + replicate
  x_test, y_test, _ = _resampled_xy(_FRAME, spec, "test", seed)  # type: ignore[arg-type]
  scores: dict[str, dict[str, float]] = {}
  for key in MODEL_KEYS:
    p = probes[key].predict_proba(x_test[:, spec.column_index[key]])
    scores[key] = {
      "auroc": float(compute_auroc(y_test, p)),
      "auprc": float(compute_auprc(y_test, p)),
      "log_loss": float(compute_binary_log_loss(y_test, p)),
      "brier": float(compute_brier_score(y_test, p)),
    }
  row: dict[str, Any] = {"replicate": replicate}
  row.update(_contrast_rows(scores))
  return row


def _contrast_summary(values: np.ndarray) -> dict[str, Any]:
  vals = values[np.isfinite(values)]
  return {
    "median": float(np.median(vals)),
    "pct_2_5": float(np.percentile(vals, 2.5)),
    "pct_97_5": float(np.percentile(vals, 97.5)),
    "mean": float(np.mean(vals)),
    "sd": float(np.std(vals, ddof=1)),
    "fraction_positive": float(np.mean(vals > 0)),
    "n": int(len(vals)),
  }


def summarize_refit(long: pl.DataFrame) -> dict[str, Any]:
  """Summaries for every paired contrast + selected-C distributions."""
  summary: dict[str, Any] = {"contrasts": {}, "selected_c_distribution": {}}
  for contrast in CONTRAST_KEYS:
    vals = long[contrast].to_numpy().astype(np.float64)
    summary["contrasts"][contrast] = _contrast_summary(vals)
  for key in MODEL_KEYS:
    col = f"{key}__selected_c"
    counts = Counter(long[col].drop_nulls().to_numpy().tolist())
    summary["selected_c_distribution"][key] = {
      str(c): int(n) for c, n in sorted(counts.items(), key=lambda kv: float(kv[0]))
    }
  return summary


def _read_primary_interval(path: Path, anchor: str, needle: str) -> dict[str, Any] | None:
  if not path.exists():
    raise FileNotFoundError(f"Primary result file is required for the fixed-model comparison: {path}")
  data = json.loads(path.read_text(encoding="utf-8"))
  block = data.get(anchor, {})
  boot = block.get("bootstrap", {})
  if not boot or block.get("metric") != needle:
    return None
  return {
    "point_estimate": boot.get("point_estimate"),
    "ci_lower": boot.get("ci_lower"),
    "ci_upper": boot.get("ci_upper"),
    "n_bootstraps": boot.get("n_bootstraps"),
    "seed": boot.get("seed"),
    "target": "fixed-model conditional (test-row resampling only)",
  }


def run_decomposition(
  frame: pl.DataFrame,
  frame_path: Path,
  spec: RefitSpec,
  n_replicates: int,
  master_seed: int,
  n_jobs: int,
) -> dict[str, Any]:
  """Frozen two-component decomposition: training-only and test-only variation."""
  out: dict[str, Any] = {}
  with ProcessPoolExecutor(
    max_workers=n_jobs, initializer=_init_worker, initargs=(frame_path, spec),
  ) as pool:
    train_rows = list(
      pool.map(partial(_run_training_only_replicate, master_seed=master_seed), range(n_replicates)),
    )
  out["training_only"] = {
    c: _contrast_summary(pl.DataFrame(train_rows)[c].to_numpy().astype(np.float64))
    for c in CONTRAST_KEYS
  }
  probes = _fixed_probes(frame, spec)
  with ProcessPoolExecutor(
    max_workers=n_jobs, initializer=_init_worker, initargs=(frame_path, spec),
  ) as pool:
    test_rows = list(
      pool.map(
        partial(_run_test_only_replicate, probes=probes, spec=spec, master_seed=master_seed),
        range(n_replicates),
      ),
    )
  out["test_only"] = {
    c: _contrast_summary(pl.DataFrame(test_rows)[c].to_numpy().astype(np.float64))
    for c in CONTRAST_KEYS
  }
  return out


def load_completed_main(path: Path, n_replicates: int) -> pl.DataFrame:
  """Load a finished main bootstrap table. Do not recompute it."""
  long = pl.read_parquet(path).sort("replicate")
  if long.height != n_replicates:
    raise ValueError(f"Expected {n_replicates} main replicates, found {long.height}")
  missing = [name for name in CONTRAST_KEYS if name not in long.columns]
  if missing:
    raise ValueError(f"Main bootstrap table is missing contrasts: {missing}")
  replicates = set(long["replicate"].to_list())
  if replicates != set(range(n_replicates)):
    raise ValueError("Main bootstrap replicates are not 0..n-1")
  return long


def run_refit_uncertainty(
  cohort_path: Path,
  c0_path: Path,
  meta_a_path: Path,
  features_a_path: Path,
  meta_b_path: Path,
  embeddings_b_path: Path,
  output_dir: Path,
  report_path: Path,
  supp_table_path: Path,
  n_replicates: int = 200,
  master_seed: int = MASTER_SEED,
  n_jobs: int = 4,
  with_decomposition: bool = True,
  reuse_main: bool = False,
  primary_ecg_results_path: Path = Path("artifacts/probe/probe_results.json"),
  primary_context_results_path: Path = Path("artifacts/probe/probe_results_context.json"),
) -> dict[str, Any]:
  """LM5 main entry: three-surface refit bootstrap + frozen decomposition."""
  frame = build_refit_frame(cohort_path, c0_path, meta_a_path, features_a_path, meta_b_path, embeddings_b_path)
  spec = RefitSpec.from_frame(frame)
  output_dir.mkdir(parents=True, exist_ok=True)
  frame_path = output_dir / "_refit_frame.parquet"
  frame.write_parquet(frame_path)
  long_path = output_dir / "refit_bootstrap_long.parquet"
  if reuse_main:
    # The main table is the preserved result of a run that died in decomposition.
    long = load_completed_main(long_path, n_replicates)
  else:
    map_fn = partial(_run_main_replicate, master_seed=master_seed)
    with ProcessPoolExecutor(
      max_workers=n_jobs, initializer=_init_worker, initargs=(frame_path, spec),
    ) as pool:
      records = list(pool.map(map_fn, range(n_replicates)))
    long = pl.DataFrame(records).sort("replicate")
    long.write_parquet(long_path)

  summary: dict[str, Any] = {
    "analysis_status": "post_hoc_sensitivity",
    "n_replicates": n_replicates,
    "master_seed": master_seed,
    **summarize_refit(long),
  }
  summary["primary_fixed_model_intervals"] = {
    "ecg_only_b_minus_a_auroc": _read_primary_interval(
      primary_ecg_results_path, "rq1_ecg_only", "auroc",
    ),
    "context_adjusted_b_minus_a_auroc": _read_primary_interval(
      primary_context_results_path, "rq1_context_adjusted", "auroc",
    ),
  }
  if with_decomposition:
    summary["decomposition"] = run_decomposition(
      frame, frame_path, spec, n_replicates, master_seed, n_jobs,
    )

  (output_dir / "refit_bootstrap_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8",
  )
  frame_path.unlink()
  supp_table_path.parent.mkdir(parents=True, exist_ok=True)
  write_supp_table(summary, supp_table_path)
  render_report(report_path, summary, n_replicates=n_replicates, master_seed=master_seed)
  return summary


def write_supp_table(summary: dict[str, Any], path: Path) -> None:
  """Supplementary CSV: refit-aware summaries next to the fixed-model intervals."""
  import csv

  primary = summary.get("primary_fixed_model_intervals", {})
  with path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
      "contrast", "median", "pct_2_5", "pct_97_5", "mean", "sd", "fraction_positive",
      "n_replicates", "fixed_model_point", "fixed_model_ci_lower", "fixed_model_ci_upper",
    ])
    mapping = {
      "contrast_ecg_b_minus_a_auroc": "ecg_only_b_minus_a_auroc",
      "contrast_ctx_b_minus_a_auroc": "context_adjusted_b_minus_a_auroc",
    }
    for contrast in CONTRAST_KEYS:
      s = summary["contrasts"][contrast]
      ref_key = mapping.get(contrast)
      ref = (primary.get(ref_key) or {}) if ref_key else {}
      w.writerow([
        contrast, f'{s["median"]:.4f}', f'{s["pct_2_5"]:.4f}', f'{s["pct_97_5"]:.4f}',
        f'{s["mean"]:.4f}', f'{s["sd"]:.4f}', f'{s["fraction_positive"]:.3f}', s["n"],
        ref.get("point_estimate"), ref.get("ci_lower"), ref.get("ci_upper"),
      ])


def render_report(report_path: Path, summary: dict[str, Any], n_replicates: int, master_seed: int) -> None:
  """Render the LM5 results markdown (auto-generated; headline doc separate)."""
  lines = [
    "# LM5 refit-aware uncertainty: three-surface paired bootstrap",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review, LM9 claim reconciliation",
    "analysis_status: post_hoc_sensitivity",
    f"n_replicates: {n_replicates}",
    f"master_seed: {master_seed}",
    "verification: [full preprocessing+tuning refit inside every replicate; A/B paired at patient level; primary artifacts untouched]",
    "```",
    "",
    "## Paired B-A contrasts under full refit",
    "",
    "| Contrast | Median | 2.5th | 97.5th | Mean | SD | P(>0) |",
    "|---|---|---|---|---|---|---|",
  ]
  for contrast in CONTRAST_KEYS:
    s = summary["contrasts"][contrast]
    lines.append(
      f"| {contrast} | {s['median']:+.4f} | {s['pct_2_5']:+.4f} | {s['pct_97_5']:+.4f} "
      f"| {s['mean']:+.4f} | {s['sd']:.4f} | {s['fraction_positive']:.3f} |"
    )
  primary = summary.get("primary_fixed_model_intervals", {})
  lines += ["", "## Fixed-model primary intervals (comparison; different uncertainty target)", ""]
  for name, ref in primary.items():
    if ref:
      lines.append(
        f"- {name}: {ref['point_estimate']:+.4f} [{ref['ci_lower']:+.4f}, {ref['ci_upper']:+.4f}] "
        f"({ref['n_bootstraps']} resamples, seed {ref['seed']}; {ref['target']})"
      )
  decomp = summary.get("decomposition")
  if decomp:
    lines += ["", "## Decomposition (frozen before execution)", "", "| Component | Contrast | Median | 2.5th | 97.5th | P(>0) |", "|---|---|---|---|---|---|"]
    for component in ("training_only", "test_only"):
      for contrast in CONTRAST_KEYS:
        s = decomp[component][contrast]
        lines.append(
          f"| {component} | {contrast} | {s['median']:+.4f} | {s['pct_2_5']:+.4f} "
          f"| {s['pct_97_5']:+.4f} | {s['fraction_positive']:.3f} |"
        )
  lines += ["", "## Selected-C distribution", ""]
  for key, dist in summary["selected_c_distribution"].items():
    lines.append(f"- {MODEL_DISPLAY[key]}: {dist}")
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
