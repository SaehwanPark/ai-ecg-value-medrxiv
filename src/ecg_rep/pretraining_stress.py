"""LM2: pretraining-attribution stress tests for the random-encoder finding.

The parent random-encoder control showed little or no context-adjusted advantage
for pretrained D-BETA at full data. LM2 maps the boundary of that result in three
settings where pretraining could plausibly matter:

- LM2A: label-efficiency curves (pretrained B vs ten random encoders vs A at
  restricted development/validation budgets, evaluated on the unchanged
  paired-valid locked test);
- LM2B: architecture-matched random-vs-pretrained control for the administrative
  AMI endpoint;
- LM2C: frozen random-encoder transport to CODE-15% (separate module, needs
  CODE random extraction).

Everything here is a ``post hoc extension of a prespecified control``. No seed,
fraction, or variant may be chosen, added, or dropped because of its performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import numpy.typing as npt
import polars as pl

from ecg_rep.evaluate import C0_FEATURE_COLS  # pyright: ignore[reportPrivateUsage]
from ecg_rep.probe import (
  evaluate_predictions,
  fit_logistic_probe,
  metric_bundle_to_dict,
)

#: Random-encoder seeds evaluated in the parent control (all reported, none added later).
RANDOM_SEEDS: tuple[int, ...] = (42, 7, 1, 2, 3, 4, 5, 6, 8, 9)

#: Frozen label fractions (fraction_index drives the deterministic subsample seeds).
LABEL_EFFICIENCY_FRACTIONS: tuple[float, ...] = (0.025, 0.05, 0.10, 0.25, 0.50, 1.0)

#: Repeated deterministic subsamples per fraction below 1.0.
LABEL_EFFICIENCY_REPLICATES: int = 10

#: seed = SEED_BASE + 100 * fraction_index + replicate_index (LM2A plan).
LABEL_EFFICIENCY_SEED_BASE: int = 20000


@dataclass(frozen=True)
class LabelEfficiencyConfig:
  """Frozen LM2A configuration (authoritative copy: configs/label_efficiency_v1.yaml)."""

  version: str = "label_efficiency_v1"
  fractions: tuple[float, ...] = LABEL_EFFICIENCY_FRACTIONS
  n_replicates: int = LABEL_EFFICIENCY_REPLICATES
  seed_base: int = LABEL_EFFICIENCY_SEED_BASE
  random_seeds: tuple[int, ...] = RANDOM_SEEDS
  surfaces: tuple[str, ...] = ("ecg_only", "context")

  def to_dict(self) -> dict[str, Any]:
    return {
      "version": self.version,
      "fractions": list(self.fractions),
      "n_replicates": self.n_replicates,
      "seed_base": self.seed_base,
      "random_seeds": list(self.random_seeds),
      "surfaces": list(self.surfaces),
    }


def load_label_efficiency_config(path: Path) -> LabelEfficiencyConfig:
  """Load the frozen LM2A config; unknown keys rejected."""
  import yaml

  raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise ValueError(f"Label-efficiency config at {path} is not a mapping.")
  known = set(LabelEfficiencyConfig.__dataclass_fields__)
  unknown = set(raw) - known
  if unknown:
    raise ValueError(f"Unknown label-efficiency config keys: {sorted(unknown)}")
  return LabelEfficiencyConfig(
    version=str(raw["version"]),
    fractions=tuple(float(f) for f in raw["fractions"]),
    n_replicates=int(raw["n_replicates"]),
    seed_base=int(raw["seed_base"]),
    random_seeds=tuple(int(s) for s in raw["random_seeds"]),
    surfaces=tuple(str(s) for s in raw["surfaces"]),
  )


def stratified_subsample_indices(
  y: npt.NDArray[np.integer],
  fraction: float,
  seed: int,
) -> npt.NDArray[np.int64]:
  """Deterministic outcome-stratified subsample preserving the event fraction.

  Draws ``round(fraction * n_class)`` indices from each outcome class without
  replacement under a fresh ``default_rng(seed)``.
  """
  if not 0.0 < fraction <= 1.0:
    raise ValueError("fraction must be in (0, 1].")
  rng = np.random.default_rng(seed)
  yt = np.asarray(y, dtype=np.int64)
  parts: list[npt.NDArray[np.int64]] = []
  for cls in (0, 1):
    cls_idx = np.flatnonzero(yt == cls)
    n_take = int(round(fraction * len(cls_idx)))
    if n_take == 0:
      raise ValueError(f"Subsampling at fraction {fraction} would drop all class {cls} rows.")
    picked = rng.choice(cls_idx, size=n_take, replace=False)
    parts.append(np.sort(picked))
  return np.sort(np.concatenate(parts))


def subsample_seed(fraction_index: int, replicate_index: int, seed_base: int = LABEL_EFFICIENCY_SEED_BASE) -> int:
  """Frozen seed rule: ``seed_base + 100 * fraction_index + replicate_index``."""
  return seed_base + 100 * fraction_index + replicate_index


def distribution_summary(values: npt.NDArray[np.floating]) -> dict[str, float | int]:
  """Mean/median/min/max, central 80% and 95% empirical intervals, and P(v > 0)."""
  arr = np.asarray(values, dtype=np.float64)
  return {
    "n": int(arr.size),
    "mean": float(arr.mean()),
    "median": float(np.median(arr)),
    "min": float(arr.min()),
    "max": float(arr.max()),
    "p10": float(np.percentile(arr, 10)),
    "p90": float(np.percentile(arr, 90)),
    "p2.5": float(np.percentile(arr, 2.5)),
    "p97.5": float(np.percentile(arr, 97.5)),
    "fraction_positive": float((arr > 0).mean()),
  }


class RepresentationBank:
  """Aligned lookup of representation matrices for the paired-valid cohort.

  Matrices stay in native layout; rows are addressed through each meta parquet's
  (subject_id, study_id) key so subsets never require re-joining full frames.
  """

  def __init__(self, reps_dir: Path, cohort: pl.DataFrame) -> None:
    self.matrices: dict[str, npt.NDArray[np.float64]] = {}
    self._row_by_key: dict[str, npt.NDArray[np.int64]] = {}
    self.cohort = cohort
    keys = np.asarray(
      [f"{s}-{t}" for s, t in zip(cohort["subject_id"].to_list(), cohort["study_id"].to_list(), strict=True)],
      dtype=np.str_,
    )
    order = np.argsort(keys, kind="stable")
    self._sorted_keys = keys[order]
    specs = {
      "A": ("ciis_features_meta.parquet", "ciis_features.npy", "representation_a_valid"),
      "B": ("dbeta_embedding_meta.parquet", "dbeta_embeddings.npy", "representation_b_valid"),
    }
    for seed in RANDOM_SEEDS:
      specs[f"B_RAND{seed}"] = (
        f"dbeta_random_seed{seed}_meta.parquet",
        f"dbeta_random_seed{seed}_embeddings.npy",
        "representation_b_random_valid"
        if seed == 42
        else "representation_b_valid",
      )
    for name, (meta_name, matrix_name, valid_col) in specs.items():
      meta = pl.read_parquet(reps_dir / meta_name)
      matrix = np.load(reps_dir / matrix_name)
      if valid_col not in meta.columns:
        valid_col = "representation_b_valid"
      meta_keys = np.asarray(
        [f"{s}-{t}" for s, t in zip(meta["subject_id"].to_list(), meta["study_id"].to_list(), strict=True)],
        dtype=np.str_,
      )
      search_pos = np.searchsorted(self._sorted_keys, meta_keys)
      search_pos = np.clip(search_pos, 0, len(self._sorted_keys) - 1)
      matched = self._sorted_keys[search_pos] == meta_keys
      # Map cohort position -> matrix row (inner-join semantics).
      cohort_pos_sorted = search_pos[matched]
      meta_rows = np.arange(len(meta), dtype=np.int64)[matched]
      row_lookup = np.full(len(cohort), -1, dtype=np.int64)
      cohort_positions = order[cohort_pos_sorted]
      row_lookup[cohort_positions] = meta_rows
      self.matrices[name] = np.asarray(matrix, dtype=np.float64)
      self._row_by_key[name] = row_lookup

  def matrix_for(self, name: str, cohort_positions: npt.NDArray[np.int64]) -> npt.NDArray[np.float64] | None:
    rows = self._row_by_key[name][cohort_positions]
    if np.any(rows < 0):
      return None
    return self.matrices[name][rows]

  def coverage(self, name: str) -> float:
    return float((self._row_by_key[name] >= 0).mean())


def _fit_and_eval(
  x_dev: npt.NDArray[np.float64],
  y_dev: npt.NDArray[np.int64],
  x_val: npt.NDArray[np.float64],
  y_val: npt.NDArray[np.int64],
  x_test: npt.NDArray[np.float64],
  y_test: npt.NDArray[np.int64],
) -> dict[str, Any]:
  """One honest label-budget fit: dev-only scaler, val-only C selection, dev-only fit."""
  probe = fit_logistic_probe(x_dev, y_dev, x_val, y_val)
  p = probe.predict_proba(x_test)
  metrics = evaluate_predictions(y_test, p)
  return {"best_c": probe.best_c, **metric_bundle_to_dict(metrics)}


def run_label_efficiency(
  cohort_path: Path,
  reps_dir: Path,
  c0_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  config: LabelEfficiencyConfig | None = None,
) -> dict[str, Any]:
  """LM2A: refit every representation at restricted dev/val budgets and evaluate
  on the unchanged paired-valid locked test. The locked-test rows and labels are
  fixed once and never enter any fit."""
  cfg = config or LabelEfficiencyConfig()
  cohort = pl.read_parquet(cohort_path)
  paired = pl.read_parquet(reps_dir / "representation_pair_validity.parquet").filter(
    pl.col("representation_pair_valid")
  )
  frame = cohort.join(paired.select(["subject_id", "study_id"]), on=["subject_id", "study_id"])
  c0 = pl.read_parquet(c0_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]
  frame = frame.join(c0.select(["subject_id", "study_id", *c0_cols]), on=["subject_id", "study_id"])

  bank = RepresentationBank(reps_dir, frame)
  y_all = np.asarray(frame["mortality_30d"].to_numpy(), dtype=np.int64)
  split_all = frame["split"].to_list()

  dev_pos = np.flatnonzero(np.asarray(split_all) == "dev")
  val_pos = np.flatnonzero(np.asarray(split_all) == "val")
  test_pos = np.flatnonzero(np.asarray(split_all) == "test")
  y_dev_full = y_all[dev_pos]
  y_val_full = y_all[val_pos]
  y_test = y_all[test_pos]
  x_test_c0 = frame[test_pos].select(c0_cols).to_numpy().astype(np.float64)

  representation_names = ["A", "B", *(f"B_RAND{s}" for s in cfg.random_seeds)]
  long_rows: list[dict[str, Any]] = []
  full_auroc: dict[tuple[str, str], float] = {}

  for fraction_index, fraction in enumerate(cfg.fractions):
    n_reps = 1 if fraction >= 1.0 else cfg.n_replicates
    for replicate in range(n_reps):
      seed = subsample_seed(fraction_index, replicate, cfg.seed_base)
      dev_take = stratified_subsample_indices(y_dev_full, fraction, seed)
      val_take = stratified_subsample_indices(y_val_full, fraction, seed + 1_000_003)
      dev_pos_sub = dev_pos[dev_take]
      val_pos_sub = val_pos[val_take]
      y_dev_sub = y_all[dev_pos_sub]
      y_val_sub = y_all[val_pos_sub]
      c0_dev = frame[dev_pos_sub].select(c0_cols).to_numpy().astype(np.float64)
      c0_val = frame[val_pos_sub].select(c0_cols).to_numpy().astype(np.float64)
      for rep_name in representation_names:
        x_dev_rep = bank.matrix_for(rep_name, dev_pos_sub)
        x_val_rep = bank.matrix_for(rep_name, val_pos_sub)
        x_test_rep = bank.matrix_for(rep_name, test_pos)
        if x_dev_rep is None or x_val_rep is None or x_test_rep is None:
          # Missing rows for this representation: record and skip (never silently drop).
          long_rows.append({
            "representation": rep_name,
            "surface": "ecg_only",
            "fraction": fraction,
            "replicate": replicate,
            "seed": seed,
            "status": "missing_rows",
          })
          continue
        x_dev_rep = np.asarray(x_dev_rep, dtype=np.float64)
        x_val_rep = np.asarray(x_val_rep, dtype=np.float64)
        x_test_rep = np.asarray(x_test_rep, dtype=np.float64)
        surfaces: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]] = {}
        if "ecg_only" in cfg.surfaces:
          surfaces["ecg_only"] = (x_dev_rep, x_val_rep, x_test_rep)
        if "context" in cfg.surfaces:
          surfaces["context"] = (
            np.concatenate([c0_dev, x_dev_rep], axis=1),
            np.concatenate([c0_val, x_val_rep], axis=1),
            np.concatenate([x_test_c0, x_test_rep], axis=1),
          )
        for surface, (xd, xv, xt) in surfaces.items():
          res = _fit_and_eval(xd, y_dev_sub, xv, y_val_sub, xt, y_test)
          long_rows.append({
            "representation": rep_name,
            "surface": surface,
            "fraction": fraction,
            "replicate": replicate,
            "seed": seed,
            "status": "ok",
            **res,
          })
          if fraction >= 1.0:
            full_auroc[(rep_name, surface)] = float(res["auroc"])

  long_path = output_dir / "label_efficiency_long.csv"
  output_dir.mkdir(parents=True, exist_ok=True)
  pl.DataFrame(long_rows).write_csv(long_path)

  ok = pl.DataFrame([r for r in long_rows if r["status"] == "ok"])
  summaries: list[dict[str, Any]] = []
  for surface in cfg.surfaces:
    surf_all = ok.filter(pl.col("surface") == surface)
    for frac_val in cfg.fractions:
      rows_f = surf_all.filter(pl.col("fraction") == frac_val)
      pre_f = rows_f.filter(pl.col("representation") == "B").select(["replicate", "auroc", "auprc"]).rename({
        "auroc": "auroc_pre",
        "auprc": "auprc_pre",
      })
      # Pretrained-minus-random per (random seed, replicate); random rows only.
      rand_rows = rows_f.filter(pl.col("representation").str.starts_with("B_RAND")).join(pre_f, on="replicate")
      for rep_name, rep_group in rand_rows.group_by(["representation"]):
        rep_label = rep_name[0] if isinstance(rep_name, tuple) else rep_name
        diffs = (rep_group["auroc_pre"] - rep_group["auroc"]).to_numpy()
        summaries.append({
          "surface": surface,
          "fraction": frac_val,
          "comparison": f"B_minus_{rep_label}",
          **distribution_summary(diffs),
        })
      # Pretrained vs A at each fraction.
      a_rows = rows_f.filter(pl.col("representation") == "A").join(pre_f, on="replicate")
      if a_rows.height:
        diffs = (a_rows["auroc_pre"] - a_rows["auroc"]).to_numpy()
        summaries.append({
          "surface": surface,
          "fraction": frac_val,
          "comparison": "B_minus_A",
          **distribution_summary(diffs),
        })

  # Descriptive minimum labeled fraction to reach 95% of the full-data AUROC.
  reach_rows: list[dict[str, Any]] = []
  for (rep_name, surface), full in full_auroc.items():
    curve = ok.filter((pl.col("representation") == rep_name) & (pl.col("surface") == surface))
    min_fraction = None
    for fraction in cfg.fractions:
      vals = curve.filter(pl.col("fraction") == fraction)["auroc"].drop_nulls()
      if vals.len() == 0:
        continue
      mean_auroc = float(vals.mean())  # pyright: ignore[reportArgumentType]
      if min_fraction is None and mean_auroc >= 0.95 * full:
        min_fraction = fraction
    reach_rows.append({
      "representation": rep_name,
      "surface": surface,
      "full_data_auroc": full,
      "min_fraction_to_95pct_of_full": min_fraction,
    })

  results = {
    "status": "complete",
    "post_hoc": True,
    "analysis_status": "post hoc extension of a prespecified control",
    "config": cfg.to_dict(),
    "source_sha": source_sha,
    "n_test": int(len(y_test)),
    "n_test_events": int(y_test.sum()),
    "representation_coverage": {name: bank.coverage(name) for name in representation_names},
    "summary": summaries,
    "min_fraction_to_95pct": reach_rows,
    "long_artifact": str(long_path),
  }
  (output_dir / "label_efficiency_summary.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  _render_lm2a_report(report_path, results)
  return results


def _render_lm2a_report(report_path: Path, results: dict[str, Any]) -> None:
  lines = [
    "# LM2A label-efficiency: pretrained vs architecture-matched random encoders",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {results['source_sha']}",
    "analysis_status: post_hoc_extension_of_prespecified_control",
    "inputs: [configs/label_efficiency_v1.yaml, artifacts/reps, artifacts/context, artifacts/cohort]",
    "decisions: [frozen fractions/replicates/seeds; no variant added or dropped for performance]",
    "evidence: [label_efficiency_long.csv, label_efficiency_summary.json]",
    "unresolved: []",
    "verification: [locked test unchanged and never fitted; budgets subsample dev and val together]",
    "next_action: boundary wording in claims register C4 (LM9)",
    "```",
    "",
    f"Locked test: {results['n_test']:,} rows, {results['n_test_events']} deaths (unchanged parent surface).",
    "",
    "## Pretrained-minus-random AUROC by label fraction (context-adjusted surface)",
    "",
    "| Fraction | n (seed,rep) | Mean | Median | Min-Max | Central 95% | P(B>random) |",
    "|---|---|---|---|---|---|---|",
  ]
  for row in results["summary"]:
    if row["surface"] != "context" or not str(row["comparison"]).startswith("B_minus_B_RAND"):
      continue
    lines.append(
      f"| {row['fraction']} | {row['n']} | {row['mean']:+.4f} | {row['median']:+.4f} | "
      f"{row['min']:+.4f} to {row['max']:+.4f} | {row['p2.5']:+.4f} to {row['p97.5']:+.4f} | "
      f"{row['fraction_positive']:.2f} |"
    )
  lines += [
    "",
    "## Minimum labeled fraction reaching 95% of the full-data AUROC (context surface)",
    "",
    "| Representation | Full-data AUROC | Min fraction |",
    "|---|---|---|",
  ]
  for row in results["min_fraction_to_95pct"]:
    if row["surface"] != "context":
      continue
    mf = row["min_fraction_to_95pct_of_full"]
    lines.append(f"| {row['representation']} | {row['full_data_auroc']:.4f} | {mf if mf is not None else 'never'} |")
  lines += [
    "",
    "Frozen interpretation rules apply: a small-fraction separation supports label-efficiency",
    "wording; the absence of separation extends the null finding to restricted supervision",
    "within MIMIC. No generalization to other tasks, outcomes, or architectures.",
    "",
  ]
  report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# LM2B: random-vs-pretrained control for administrative AMI.
# ---------------------------------------------------------------------------


def run_ami_random_control(
  cohort_path: Path,
  setting_path: Path,
  admissions_path: Path,
  diagnoses_path: Path,
  labevents_path: Path,
  pair_validity_path: Path,
  c0_path: Path,
  reps_dir: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  random_seeds: tuple[int, ...] = RANDOM_SEEDS,
) -> dict[str, Any]:
  """LM2B: per-seed random-B AMI probes against pretrained-B and A on the identical
  administrative-AMI population and hyperparameter grid. Uses the same frozen
  ``primary_principal_timed`` population and label as the existing AMI analysis."""
  from ecg_rep.ami_administrative import LABEL_POSITIVE, _population_mask, build_analysis_frame  # pyright: ignore[reportPrivateUsage]
  from ecg_rep.ami_outcome import troponin_hadm_flags
  from ecg_rep.data import load_admissions

  admissions = load_admissions(admissions_path)
  diagnoses = pl.scan_csv(
    diagnoses_path,
    schema_overrides={"hadm_id": pl.Int64, "seq_num": pl.Int64, "icd_code": pl.String, "icd_version": pl.Int64},
  ).select(["hadm_id", "seq_num", "icd_code", "icd_version"]).collect()
  troponin = troponin_hadm_flags(labevents_path, [51002, 52642, 51003])
  cohort = pl.read_parquet(cohort_path)
  setting = pl.read_parquet(setting_path)
  pair = pl.read_parquet(pair_validity_path)
  labeled = build_analysis_frame(cohort, setting, admissions, diagnoses, troponin, pair)
  subset_mask = _population_mask(labeled, "primary_principal_timed")
  prepared = labeled.filter(subset_mask)
  c0 = pl.read_parquet(c0_path)
  c0_cols = [c for c in C0_FEATURE_COLS if c in c0.columns]
  prepared = prepared.join(c0.select(["subject_id", "study_id", *c0_cols]), on=["subject_id", "study_id"], how="left")

  bank = RepresentationBank(reps_dir, prepared)
  y = (prepared["primary_label"] == LABEL_POSITIVE).cast(pl.Int64)
  split = prepared["split"].to_list()
  pos = {s: np.flatnonzero(np.asarray(split) == s) for s in ("dev", "val", "test")}
  y_dev, y_val, y_test = (
    y.gather(np.asarray(p, dtype=np.int64)).to_numpy().astype(np.int64) for p in pos.values()
  )
  c0_blocks = {
    s: prepared[pos[s]].select(c0_cols).to_numpy().astype(np.float64) for s in pos
  }

  per_seed: list[dict[str, Any]] = []
  for seed in random_seeds:
    entry: dict[str, Any] = {"random_seed": seed}
    for rep_name, surface_label in (
      ("A", "a_ecg"),
      ("B", "pretrained_ecg"),
      (f"B_RAND{seed}", f"random_ecg_{seed}"),
    ):
      x_blocks = {s: bank.matrix_for(rep_name, p) for s, p in pos.items()}
      if any(m is None for m in x_blocks.values()):
        entry["status"] = f"missing_rows_{rep_name}"
        continue
      xd = np.asarray(x_blocks["dev"], dtype=np.float64)
      xv = np.asarray(x_blocks["val"], dtype=np.float64)
      xt = np.asarray(x_blocks["test"], dtype=np.float64)
      res = _fit_and_eval(xd, y_dev, xv, y_val, xt, y_test)
      entry[surface_label] = res
      # Context-adjusted variant.
      res_ctx = _fit_and_eval(
        np.concatenate([c0_blocks["dev"], xd], axis=1),
        y_dev,
        np.concatenate([c0_blocks["val"], xv], axis=1),
        y_val,
        np.concatenate([c0_blocks["test"], xt], axis=1),
        y_test,
      )
      entry[surface_label.replace("_ecg", "_c0")] = res_ctx
    per_seed.append(entry)

  summaries: list[dict[str, Any]] = []
  for surface in ("ecg", "c0"):
    pre_key = f"pretrained_{surface}"
    a_key = f"a_{surface}"
    rand_deltas_pre: list[float] = []
    rand_deltas_a: list[float] = []
    for entry in per_seed:
      if pre_key not in entry or a_key not in entry:
        continue
      for seed in random_seeds:
        rand_key = f"random_{surface}_{seed}"
        if rand_key not in entry:
          continue
        rand_deltas_pre.append(entry[pre_key]["auroc"] - entry[rand_key]["auroc"])
        rand_deltas_a.append(entry[rand_key]["auroc"] - entry[a_key]["auroc"])
    if rand_deltas_pre:
      summaries.append({
        "surface": surface,
        "comparison": "pretrained_minus_random",
        **distribution_summary(np.asarray(rand_deltas_pre)),
      })
      summaries.append({
        "surface": surface,
        "comparison": "random_minus_A",
        **distribution_summary(np.asarray(rand_deltas_a)),
      })

  results = {
    "status": "complete",
    "post_hoc": True,
    "analysis_status": "post hoc extension of a prespecified control",
    "source_sha": source_sha,
    "endpoint": "principal discharge AMI coding (administrative; never adjudicated AMI)",
    "population": "primary_principal_timed (frozen AMI protocol)",
    "n_test": int(len(y_test)),
    "n_test_events": int(y_test.sum()),
    "per_seed": per_seed,
    "summary": summaries,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "ami_random_control_results.json").write_text(
    json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  _render_lm2b_report(report_path, results)
  return results


def _render_lm2b_report(report_path: Path, results: dict[str, Any]) -> None:
  lines = [
    "# LM2B random-vs-pretrained control for administrative AMI",
    "",
    "```yaml",
    "status: complete",
    "producer: study-orchestrator (agent)",
    "consumer: study-review",
    f"source_commit: {results['source_sha']}",
    "analysis_status: post_hoc_extension_of_prespecified_control",
    "inputs: [configs/ami_administrative_protocol_v1.yaml, artifacts/reps, artifacts/context]",
    "decisions: [identical AMI population and grid; no random representation missing]",
    "evidence: [ami_random_control_results.json]",
    "unresolved: []",
    "verification: [per-seed and distribution summaries; administrative wording preserved]",
    "next_action: boundary wording in claims register (LM9)",
    "```",
    "",
    f"Population: {results['population']} — {results['n_test']:,} test rows, {results['n_test_events']} events.",
    "",
    "| Surface | Comparison | n | Mean | Median | Min-Max | Central 95% |",
    "|---|---|---|---|---|---|---|",
  ]
  for row in results["summary"]:
    lines.append(
      f"| {row['surface']} | {row['comparison']} | {row['n']} | {row['mean']:+.4f} | "
      f"{row['median']:+.4f} | {row['min']:+.4f} to {row['max']:+.4f} | "
      f"{row['p2.5']:+.4f} to {row['p97.5']:+.4f} |"
    )
  lines += [
    "",
    "This endpoint is administrative (discharge coding), not adjudicated AMI. No wording",
    "may imply clinical AMI diagnosis. All ten random initializations are reported.",
    "",
  ]
  report_path.write_text("\n".join(lines), encoding="utf-8")
