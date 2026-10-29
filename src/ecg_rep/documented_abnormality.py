"""A7: documented machine-text abnormality tiers and equal-capacity review policies.

Decision D13 (proxy clinical decision under D11, lexicon approved under G2).
Plan: ``docs/plan_20261005_a6a7/a7-documented-abnormality-policy-plan.md``.

The tiers are built from the frozen lexicon's component flags and nothing else.
A tier is a documentation level, not a diagnosis. An ECG with no documented phrase
is untiered, not normal. Captured deaths are event enrichment, not treatment
benefit, and capacity fractions are review budgets, not probability thresholds.
No probe is refit: every policy reorders the frozen locked-test scores.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import numpy as np
import numpy.typing as npt
import polars as pl

from ecg_rep.analysis import (
  DEFAULT_CAPACITY_FRACTIONS,
  _stratified_bootstrap_indices,
  fixed_capacity_contrast_rows,
  fixed_capacity_metric_rows,
)
from ecg_rep.clinical_setting import PUBLIC_MIN_CELL, wilson_interval
from ecg_rep.ecg_phenotypes import (
  PhenotypeCall,
  PhenotypeDictionary,
  hedged_trigger_kinds,
  load_phenotype_dictionary,
  phenotype_frame,
)
from ecg_rep.mortality_sensitivity import assert_frozen_lexicon, public_suppression

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# Tier level L: the policy score ``L + p`` sorts by tier first, then by risk.
LEVEL_TIER1 = 3
LEVEL_TIER2 = 2
LEVEL_TIER3 = 1
LEVEL_UNTIERED = 0
TIER_NAMES: dict[int, str] = {
  LEVEL_TIER1: "tier1_definite_emergency",
  LEVEL_TIER2: "tier2_hedged_machine_alert",
  LEVEL_TIER3: "tier3_documented_major",
  LEVEL_UNTIERED: "untiered",
}
# Documented non-emergency major findings (D13). Isolated RBBB and suggested
# pericarditis are excluded on purpose and reported for transparency only.
TIER3_FLAGS: tuple[str, ...] = (
  "atrial_fibrillation",
  "atrial_flutter",
  "left_bundle_branch_block",
  "lvh_with_strain",
  "old_or_undetermined_infarct",
  "paced_rhythm",
  "prolonged_qt",
  "unsustained_ventricular_tachycardia",
)
REPORT_ONLY_FLAGS: tuple[str, ...] = ("right_bundle_branch_block", "suggested_pericarditis")

POLICIES: tuple[str, ...] = ("A", "B", "overt_then_A", "doc_then_A", "doc_then_B")
CONTRASTS: tuple[tuple[str, str, str], ...] = (
  ("doc_then_A", "B", "B_minus_doc_then_A"),
  ("A", "doc_then_A", "doc_then_A_minus_A"),
  ("B", "doc_then_B", "doc_then_B_minus_B"),
  ("A", "overt_then_A", "overt_then_A_minus_A"),
)
PRIMARY_CONTRAST = "B_minus_doc_then_A"
PRIMARY_CAPACITY = 0.05
N_BOOTSTRAPS = 200
SEED = 42
REVIEW_SEED = 20261005
SCORE_SOURCES: dict[str, tuple[str, str]] = {
  "context_adjusted_frozen": ("p_a", "p_b"),
  "ecg_only_frozen": ("p_a_ecg", "p_b_ecg"),
}
GROUP_PARTITION = ("tier1", "tier2", "tier3", "untiered")
GROUP_UNION = ("tier1_2", "tier1_2_3")
GROUP_FLAGS = (*TIER3_FLAGS, *REPORT_ONLY_FLAGS)


def policy_spec() -> dict[str, Any]:
  """Everything the locked run depends on besides the frozen scores and lexicon."""
  return {
    "analysis": "A7_documented_abnormality_policy",
    "decision": "D13",
    "tier_levels": {name: level for level, name in TIER_NAMES.items()},
    "tier3_flags": list(TIER3_FLAGS),
    "report_only_flags": list(REPORT_ONLY_FLAGS),
    "policies": list(POLICIES),
    "contrasts": [list(item) for item in CONTRASTS],
    "capacity_fractions": list(DEFAULT_CAPACITY_FRACTIONS),
    "primary_capacity": PRIMARY_CAPACITY,
    "score_sources": {key: list(value) for key, value in SCORE_SOURCES.items()},
    "n_bootstraps": N_BOOTSTRAPS,
    "seed": SEED,
    "tier1_suppression_pooling": "report_within_tier1_2",
  }


def policy_spec_hash() -> str:
  """Hash of the frozen A7 specification (not the lexicon file, which has its own hash)."""
  text = json.dumps(policy_spec(), sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


FROZEN_POLICY_SPEC_HASH = "f5a80b027bb2a9a55a86e3b50c7dd1b8f1ed7d2e1ac6a7e9cc215aad73185357"


def assert_frozen_policy_spec() -> str:
  """Refuse the locked run when the tier or policy definitions changed after the freeze."""
  current = policy_spec_hash()
  if current != FROZEN_POLICY_SPEC_HASH:
    raise RuntimeError(
      "The A7 policy specification no longer matches the freeze "
      f"{FROZEN_POLICY_SPEC_HASH} (current {current}). Refusing to run on the locked test."
    )
  return current


def _split_flags(value: str | None) -> frozenset[str]:
  if not value:
    return frozenset()
  return frozenset(part for part in str(value).split("|") if part)


def assign_tier_from_flags(
  definite_overt: Iterable[str],
  uncertain_overt: Iterable[str],
  other_major: Iterable[str],
) -> int:
  """Tier level from component flags. Tier 1 beats Tier 2 beats Tier 3."""
  if any(True for _ in definite_overt):
    return LEVEL_TIER1
  if any(True for _ in uncertain_overt):
    return LEVEL_TIER2
  if any(flag in TIER3_FLAGS for flag in other_major):
    return LEVEL_TIER3
  return LEVEL_UNTIERED


def assign_tier(call: PhenotypeCall) -> int:
  """Tier level of one classified machine report."""
  return assign_tier_from_flags(call.definite_overt, call.uncertain_overt, call.other_major)


def tier_levels(phenotypes: pl.DataFrame) -> IntArray:
  """Tier level for each row of ``phenotype_frame`` output (pipe-joined flag columns).

  Rows with null flag columns, such as a study with no machine report, are untiered.
  """
  levels = [
    assign_tier_from_flags(
      _split_flags(row["definite_overt"]),
      _split_flags(row["uncertain_overt"]),
      _split_flags(row["other_major"]),
    )
    for row in phenotypes.select(["definite_overt", "uncertain_overt", "other_major"]).iter_rows(named=True)
  ]
  return np.asarray(levels, dtype=np.int64)


def flag_membership(other_major: Sequence[str | None]) -> dict[str, BoolArray]:
  """Non-exclusive presence of each reported flag."""
  sets = [_split_flags(value) for value in other_major]
  return {
    flag: np.asarray([flag in flags for flags in sets], dtype=np.bool_)
    for flag in GROUP_FLAGS
  }


def priority_scores(level: IntArray | Sequence[int], p: FloatArray | Sequence[float]) -> FloatArray:
  """``L + p``: lexicographic by tier, then by risk. ``p`` must be a probability below 1."""
  lv = np.asarray(level, dtype=np.float64)
  prob = np.asarray(p, dtype=np.float64)
  if len(lv) != len(prob):
    raise ValueError("Tier levels and risk scores must describe the same rows.")
  if np.any(~np.isfinite(prob)) or np.any(prob < 0.0) or np.any(prob >= 1.0):
    raise ValueError("Risk scores must lie in [0, 1) so a tier always outranks risk.")
  return lv + prob


def policy_scores(level: IntArray, p_a: FloatArray, p_b: FloatArray) -> dict[str, FloatArray]:
  """The five single-score policies of the A7 plan."""
  tier1 = (np.asarray(level) == LEVEL_TIER1).astype(np.float64)
  return {
    "A": np.asarray(p_a, dtype=np.float64),
    "B": np.asarray(p_b, dtype=np.float64),
    "overt_then_A": priority_scores(tier1, p_a),
    "doc_then_A": priority_scores(level, p_a),
    "doc_then_B": priority_scores(level, p_b),
  }


def group_masks(level: IntArray, flags: Mapping[str, BoolArray]) -> dict[str, tuple[str, BoolArray]]:
  """P1 groups as ``name -> (family, mask)``. Partition rows are mutually exclusive."""
  lv = np.asarray(level)
  masks: dict[str, tuple[str, BoolArray]] = {
    "tier1": ("partition", lv == LEVEL_TIER1),
    "tier2": ("partition", lv == LEVEL_TIER2),
    "tier3": ("partition", lv == LEVEL_TIER3),
    "untiered": ("partition", lv == LEVEL_UNTIERED),
    "tier1_2": ("union", lv >= LEVEL_TIER2),
    "tier1_2_3": ("union", lv >= LEVEL_TIER3),
  }
  for flag in GROUP_FLAGS:
    masks[flag] = ("flag", np.asarray(flags[flag], dtype=np.bool_))
  return masks


def flagged_indices(scores: FloatArray, capacity_fraction: float) -> IntArray:
  """Top-capacity patients, with the same rounding as ``analysis.capacity_evaluation``."""
  n = len(scores)
  n_flagged = max(1, int(round(n * capacity_fraction)))
  return np.asarray(np.argsort(-np.asarray(scores, dtype=np.float64))[:n_flagged], dtype=np.int64)


def _ci(values: FloatArray) -> tuple[float | None, float | None]:
  finite = values[np.isfinite(values)]
  if finite.size == 0:
    return None, None
  return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def tier_ppv_table(
  y: IntArray,
  level: IntArray,
  flags: Mapping[str, BoolArray],
  risk_scores: Mapping[str, FloatArray],
  score_source: str,
  n_bootstraps: int = N_BOOTSTRAPS,
  seed: int = SEED,
) -> pl.DataFrame:
  """P1: PPV and sensitivity of each documented group next to the same-size top-n by A and B.

  ``risk_scores`` maps a label (``A``, ``B``) to its score. The bootstrap resamples
  patients stratified by outcome with the probes frozen; the top-n size follows the
  group size in each resample.
  """
  groups = group_masks(level, flags)
  names = list(groups)
  total_events = int(y.sum())
  labels = list(risk_scores)
  point: dict[str, dict[str, Any]] = {}
  for name, (family, mask) in groups.items():
    n = int(mask.sum())
    events = int(y[mask].sum())
    ci_low, ci_high = wilson_interval(events, n)
    row: dict[str, Any] = {
      "score_source": score_source,
      "group": name,
      "family": family,
      "n": n,
      "n_events": events,
      "event_prevalence": (events / n) if n else None,
      "ppv_ci_low": ci_low if n else None,
      "ppv_ci_high": ci_high if n else None,
      "sensitivity": (events / total_events) if total_events else None,
    }
    for label in labels:
      order = np.argsort(-risk_scores[label])
      top = int(y[order[:n]].sum()) if n else 0
      row[f"top_n_events_{label}"] = top if n else None
      row[f"top_n_ppv_{label}"] = (top / n) if n else None
      row[f"delta_top_n_minus_group_ppv_{label}"] = ((top / n) - (events / n)) if n else None
    point[name] = row

  rng = np.random.default_rng(seed)
  draws = np.full((n_bootstraps, len(names), len(labels)), np.nan, dtype=np.float64)
  for b in range(n_bootstraps):
    idx = _stratified_bootstrap_indices(y, rng)
    yb = y[idx]
    cums: list[FloatArray] = []
    for label in labels:
      order = np.argsort(-risk_scores[label][idx])
      cums.append(np.cumsum(yb[order]).astype(np.float64))
    for g, name in enumerate(names):
      mask = groups[name][1][idx]
      n = int(mask.sum())
      if n == 0:
        continue
      group_ppv = float(yb[mask].sum()) / n
      for s, cum in enumerate(cums):
        draws[b, g, s] = float(cum[n - 1]) / n - group_ppv
  for g, name in enumerate(names):
    for s, label in enumerate(labels):
      low, high = _ci(draws[:, g, s])
      point[name][f"delta_top_n_minus_group_ppv_{label}_ci_low"] = low if point[name]["n"] else None
      point[name][f"delta_top_n_minus_group_ppv_{label}_ci_high"] = high if point[name]["n"] else None
  frame = pl.DataFrame([point[name] for name in names])
  return frame.with_columns(
    pl.lit(n_bootstraps).alias("n_bootstraps"),
    pl.lit(False).alias("models_refit"),
  )


_P1_IDENTITY = ("score_source", "group", "family", "n", "n_events")


def public_tier_table(table: pl.DataFrame) -> pl.DataFrame:
  """Apply D8 to a P1 table.

  The four partition rows (tier 1/2/3/untiered) get complementary suppression, so
  a hidden Tier 1 also hides the next-smallest cell. Pooled and flag rows are
  hidden when their own n or deaths are under the public minimum. Every derived
  column of a hidden row is blanked, including the same-size top-n comparison,
  because its size would reveal the hidden denominator.
  """
  value_cols = tuple(col for col in table.columns if col not in _P1_IDENTITY and col not in {
    "n_bootstraps", "models_refit",
  })
  rows: list[pl.DataFrame] = []
  for source in table["score_source"].unique(maintain_order=True).to_list():
    sub = table.filter(pl.col("score_source") == source)
    part = sub.filter(pl.col("family") == "partition")
    # ``public_suppression`` already blanks event_prevalence; pass only the other derived columns.
    extra_cols = tuple(col for col in value_cols if col != "event_prevalence")
    pub_part = public_suppression(part, ("score_source",), metric_cols=extra_cols).with_columns(
      [
        pl.when(pl.col("suppress_status").is_in(["primary", "complement"]))
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in value_cols
      ]
    )
    other = sub.filter(pl.col("family") != "partition")
    small = (pl.col("n") < PUBLIC_MIN_CELL) | ((pl.col("n_events") > 0) & (pl.col("n_events") < PUBLIC_MIN_CELL))
    other_pub = other.with_columns(
      pl.when(small).then(pl.lit("primary")).otherwise(pl.lit("ok")).alias("suppress_status")
    ).with_columns(
      [
        pl.when(pl.col("suppress_status") == "primary").then(None).otherwise(pl.col(col)).alias(col)
        for col in ("n", "n_events", *value_cols)
      ]
    )
    rows.append(pl.concat([pub_part, other_pub.select(pub_part.columns)]))
  combined = pl.concat(rows)
  order = {name: i for i, name in enumerate([*GROUP_PARTITION, *GROUP_UNION, *GROUP_FLAGS])}
  return combined.sort(
    [pl.col("score_source"), pl.col("group").replace_strict(order, default=len(order))]
  )


def policy_capacity_table(
  y: IntArray,
  level: IntArray,
  scores_by_policy: Mapping[str, FloatArray],
  score_source: str,
  capacity_fractions: tuple[float, ...] = DEFAULT_CAPACITY_FRACTIONS,
) -> pl.DataFrame:
  """P2 per-policy capacity metrics plus overlap with B and B-identity recovery."""
  scores = dict(scores_by_policy)
  base_rows = fixed_capacity_metric_rows(y, scores, list(scores), capacity_fractions)
  rows: list[dict[str, Any]] = []
  for row in base_rows:
    fraction = float(row["capacity_fraction"])
    policy = str(row["model"])
    flagged = set(flagged_indices(scores[policy], fraction).tolist())
    b_flagged = set(flagged_indices(scores["B"], fraction).tolist())
    b_events = {i for i in b_flagged if y[i] == 1}
    union = flagged | b_flagged
    out = dict(row)
    out["score_source"] = score_source
    out["policy"] = out.pop("model")
    out["jaccard_with_B"] = (len(flagged & b_flagged) / len(union)) if union else None
    out["b_identity_recovery"] = (
      len({i for i in b_events if i in flagged}) / len(b_events) if b_events else None
    )
    rows.append(out)
  return pl.DataFrame(rows)


SLOT_GROUPS: tuple[tuple[str, int, int | None], ...] = (
  # Tier 1 is only ever reported pooled with Tier 2 (D8 pooling fixed before counts).
  ("tier1_2_pooled", LEVEL_TIER2, None),
  (TIER_NAMES[LEVEL_TIER3], LEVEL_TIER3, LEVEL_TIER3),
  (TIER_NAMES[LEVEL_UNTIERED], LEVEL_UNTIERED, LEVEL_UNTIERED),
)


def policy_slot_table(
  y: IntArray,
  level: IntArray,
  scores_by_policy: Mapping[str, FloatArray],
  score_source: str,
  capacity_fractions: tuple[float, ...] = DEFAULT_CAPACITY_FRACTIONS,
) -> pl.DataFrame:
  """Review slots (and deaths) each policy spends on each tier group.

  Tier 1 and Tier 2 are pooled. Publishing Tier 2 slots next to the pooled Tier 1∪2
  size in the P1 table would otherwise reveal a Tier 1 count under the public minimum.
  """
  rows: list[dict[str, Any]] = []
  for policy, scores in scores_by_policy.items():
    for fraction in capacity_fractions:
      idx = flagged_indices(scores, fraction)
      for name, low, high in SLOT_GROUPS:
        in_group = (level[idx] >= low) if high is None else (level[idx] == low)
        n = int(in_group.sum())
        events = int(y[idx][in_group].sum())
        rows.append({
          "score_source": score_source,
          "policy": policy,
          "capacity_fraction": fraction,
          "tier": name,
          "n": n,
          "n_events": events,
          "event_prevalence": (events / n) if n else None,
        })
  return pl.DataFrame(rows)


def public_slot_table(slots: pl.DataFrame) -> pl.DataFrame:
  """D8 for the slot table: each (source, policy, capacity) is one partition of the flagged list."""
  return public_suppression(slots, ("score_source", "policy", "capacity_fraction"), metric_cols=())


def public_capacity_table(table: pl.DataFrame) -> pl.DataFrame:
  """D8 for capacity metrics: blank a row whose flagged n or captured deaths are under the minimum."""
  small = (pl.col("n_flagged") < PUBLIC_MIN_CELL) | (
    (pl.col("n_events_captured") > 0) & (pl.col("n_events_captured") < PUBLIC_MIN_CELL)
  )
  derived = [
    col for col in table.columns
    if col not in {"score_source", "policy", "capacity_fraction", "n_events_total"}
  ]
  return table.with_columns(
    pl.when(small).then(pl.lit("primary")).otherwise(pl.lit("ok")).alias("suppress_status")
  ).with_columns(
    [pl.when(pl.col("suppress_status") == "primary").then(None).otherwise(pl.col(col)).alias(col) for col in derived]
  )


def public_contrast_table(table: pl.DataFrame) -> pl.DataFrame:
  """Blank a contrast whose own captured deaths fall under the public minimum."""
  small = (
    ((pl.col("events_base") > 0) & (pl.col("events_base") < PUBLIC_MIN_CELL))
    | ((pl.col("events_comparison") > 0) & (pl.col("events_comparison") < PUBLIC_MIN_CELL))
  )
  derived = [
    col for col in table.columns
    if col not in {"score_source", "contrast", "capacity_fraction", "n_bootstraps", "models_refit_inside_bootstrap"}
  ]
  return table.with_columns(
    pl.when(small).then(pl.lit("primary")).otherwise(pl.lit("ok")).alias("suppress_status")
  ).with_columns(
    [pl.when(pl.col("suppress_status") == "primary").then(None).otherwise(pl.col(col)).alias(col) for col in derived]
  )


def development_tier_audit(
  phenotypes: pl.DataFrame,
  reports: pl.DataFrame,
  dictionary: PhenotypeDictionary,
) -> dict[str, Any]:
  """Tier sizes and overlaps on development reports. No outcomes, no scores.

  Tiers may be revised on these counts only, and only before the locked run.
  """
  level = tier_levels(phenotypes)
  flags = flag_membership(phenotypes["other_major"].to_list())
  n = phenotypes.height
  tier_counts = {TIER_NAMES[lvl]: int((level == lvl).sum()) for lvl in TIER_NAMES}
  tier3_any = np.logical_or.reduce([flags[f] for f in TIER3_FLAGS])
  n_tier3_flags = np.sum([flags[f].astype(np.int64) for f in TIER3_FLAGS], axis=0)
  cooccurrence = {
    a: {b: int((flags[a] & flags[b]).sum()) for b in TIER3_FLAGS if b != a}
    for a in TIER3_FLAGS
  }
  tier2_ids = set(phenotypes.filter(pl.Series(level == LEVEL_TIER2))["study_id"].to_list())
  uncertainty_n = 0
  temporality_n = 0
  both_n = 0
  report_cols = [f"report_{i}" for i in range(18)]
  for record in reports.select(["study_id", *report_cols]).iter_rows(named=True):
    if record["study_id"] not in tier2_ids:
      continue
    unc, temp = hedged_trigger_kinds([record[col] for col in report_cols], dictionary)
    uncertainty_n += int(unc)
    temporality_n += int(temp)
    both_n += int(unc and temp)
  n_tier2 = len(tier2_ids)
  return {
    "split_policy": "development_only",
    "joined_to_mortality": False,
    "joined_to_model_scores": False,
    "lexicon_version": dictionary.version,
    "lexicon_hash": dictionary.rule_hash,
    "n_development": n,
    "tier_counts": tier_counts,
    "tier3_flag_counts": {f: int(flags[f].sum()) for f in TIER3_FLAGS},
    "report_only_flag_counts": {f: int(flags[f].sum()) for f in REPORT_ONLY_FLAGS},
    "tier3_flag_cooccurrence": cooccurrence,
    "n_tier3_with_two_or_more_flags": int((n_tier3_flags >= 2).sum()),
    "n_tier3_flag_in_tier1_or_2": int((tier3_any & (level >= LEVEL_TIER2)).sum()),
    "n_rbbb_only_untiered": int((flags["right_bundle_branch_block"] & (level == LEVEL_UNTIERED)).sum()),
    "n_pericarditis_only_untiered": int((flags["suggested_pericarditis"] & (level == LEVEL_UNTIERED)).sum()),
    "n_missing_report": int((phenotypes["phenotype_reason"] == "missing_report").sum()),
    "tier2_reports": n_tier2,
    "tier2_uncertainty_wording": uncertainty_n,
    "tier2_temporality_wording": temporality_n,
    "tier2_both_wordings": both_n,
    "tier2_temporality_share": (temporality_n / n_tier2) if n_tier2 else None,
    "policy_spec_hash": policy_spec_hash(),
  }


def draw_review_sample(
  phenotypes: pl.DataFrame,
  reports: pl.DataFrame,
  seed: int = REVIEW_SEED,
  n_tier2: int = 100,
  n_tier3: int = 150,
  n_untiered: int = 100,
) -> pl.DataFrame:
  """Blinded development sample for Jennifer Lewey's post-hoc review.

  All Tier 1, ``n_tier2`` Tier 2, ``n_tier3`` Tier 3 stratified by flag (rarest flag
  first), and ``n_untiered`` untiered reports. Columns are report text, tier call,
  flags, and sampling probability. No scores, mortality, or concordance group.
  """
  rng = np.random.default_rng(seed)
  level = tier_levels(phenotypes)
  flags = flag_membership(phenotypes["other_major"].to_list())
  ids = np.asarray(phenotypes["study_id"].to_numpy(), dtype=np.int64)
  chosen: dict[int, tuple[str, int, int]] = {}

  def _draw(stratum: str, pool: IntArray, k: int) -> None:
    pool = np.asarray([i for i in pool if int(ids[i]) not in chosen], dtype=np.int64)
    take = min(k, len(pool))
    if take == 0:
      return
    picked = rng.choice(pool, size=take, replace=False)
    for i in picked:
      chosen[int(ids[i])] = (stratum, len(pool), take)

  _draw("tier1_all", np.where(level == LEVEL_TIER1)[0], int((level == LEVEL_TIER1).sum()))
  _draw("tier2_random", np.where(level == LEVEL_TIER2)[0], n_tier2)
  per_flag = math.ceil(n_tier3 / len(TIER3_FLAGS))
  for flag in sorted(TIER3_FLAGS, key=lambda f: int((flags[f] & (level == LEVEL_TIER3)).sum())):
    _draw(f"tier3_{flag}", np.where(flags[flag] & (level == LEVEL_TIER3))[0], per_flag)
  _draw("untiered_random", np.where(level == LEVEL_UNTIERED)[0], n_untiered)

  report_cols = [f"report_{i}" for i in range(18)]
  text_by_id: dict[int, str] = {}
  for record in reports.select(["study_id", *report_cols]).iter_rows(named=True):
    sid = int(record["study_id"])
    if sid in chosen:
      text_by_id[sid] = " / ".join(str(record[col]) for col in report_cols if record[col])
  rows: list[dict[str, Any]] = []
  for pos in range(phenotypes.height):
    sid = int(ids[pos])
    if sid not in chosen:
      continue
    stratum, pool_n, take = chosen[sid]
    rows.append({
      "study_id": sid,
      "report_text": text_by_id.get(sid, ""),
      "tier_call": TIER_NAMES[int(level[pos])],
      "definite_overt": phenotypes["definite_overt"][pos] or "",
      "uncertain_overt": phenotypes["uncertain_overt"][pos] or "",
      "other_major": phenotypes["other_major"][pos] or "",
      "sampling_stratum": stratum,
      "stratum_pool_n": pool_n,
      "stratum_n_sampled": take,
      "sampling_probability": take / pool_n,
    })
  frame = pl.DataFrame(rows)
  shuffled = rng.permutation(frame.height)
  return frame[shuffled].with_row_index("review_id", offset=1)


def _fmt(value: Any, digits: int = 3) -> str:
  if value is None:
    return "suppressed"
  if isinstance(value, float):
    return f"{value:.{digits}f}"
  return str(value)


def _md_table(frame: pl.DataFrame, columns: Sequence[str], digits: int = 3) -> str:
  header = "| " + " | ".join(columns) + " |"
  rule = "| " + " | ".join("---" for _ in columns) + " |"
  body = [
    "| " + " | ".join(_fmt(row[col], digits) for col in columns) + " |"
    for row in frame.select(list(columns)).iter_rows(named=True)
  ]
  return "\n".join([header, rule, *body])


def _interval_reading(low: float | None, high: float | None) -> str:
  if low is None or high is None:
    return "suppressed"
  if low > 0:
    return "interval above zero"
  if high < 0:
    return "interval below zero"
  return "interval includes zero"


def _applied_reading(contrasts: pl.DataFrame, tier_table: pl.DataFrame, source: str) -> list[str]:
  """Map the 5% intervals onto the pre-written interpretation guide."""
  sub = contrasts.filter((pl.col("score_source") == source) & (pl.col("capacity_fraction") == PRIMARY_CAPACITY))

  def _row(name: str) -> dict[str, Any] | None:
    rows = sub.filter(pl.col("contrast") == name)
    return rows.row(0, named=True) if rows.height else None

  lines: list[str] = []
  primary = _row(PRIMARY_CONTRAST)
  a_to_doc = _row("doc_then_A_minus_A")
  b_to_doc = _row("doc_then_B_minus_B")
  if primary and primary["delta_events_ci_low"] is not None:
    if primary["delta_events_ci_low"] > 0:
      lines.append(
        "- B captures more deaths than documented abnormalities plus C0+A at equal capacity: "
        "D-BETA's prioritization advantage is not recoverable from documented abnormalities plus CIIS morphology."
      )
    elif primary["delta_events_ci_high"] is not None and primary["delta_events_ci_high"] < 0:
      lines.append("- The documentation-first policy captures more deaths than B at equal capacity.")
    else:
      lines.append(
        "- The B minus doc_then_A interval includes zero: the machine-read documentation reproduces much of B's tail advantage."
      )
  if a_to_doc and a_to_doc["delta_events_ci_high"] is not None:
    if a_to_doc["delta_events_ci_high"] < 0:
      lines.append(
        "- Putting documented abnormalities first captures fewer deaths than A alone: "
        "documentation does not add to A at this capacity and displaces higher-risk patients."
      )
    elif a_to_doc["delta_events_ci_low"] is not None and a_to_doc["delta_events_ci_low"] > 0:
      lines.append("- Putting documented abnormalities first adds deaths to A.")
    else:
      lines.append("- Putting documented abnormalities first neither clearly adds to nor subtracts from A.")
  if b_to_doc and b_to_doc["delta_events_ci_high"] is not None and b_to_doc["delta_events_ci_high"] < 0:
    lines.append("- Forcing documented abnormalities to the front displaces higher-risk patients that B would have selected.")
  tier3 = tier_table.filter((pl.col("score_source") == source) & (pl.col("group") == "tier3"))
  if tier3.height:
    row = tier3.row(0, named=True)
    low = row["delta_top_n_minus_group_ppv_B_ci_low"]
    if low is not None:
      if low > 0:
        lines.append(
          "- At the size of the Tier 3 group, a B-ranked list has higher PPV than the documented Tier 3 list "
          f"(difference {_fmt(row['delta_top_n_minus_group_ppv_B'])}, interval above zero)."
        )
      else:
        lines.append("- At the size of the Tier 3 group, the B-ranked list PPV interval includes or is below the Tier 3 PPV.")
  return lines or ["- No contrast was publishable at this capacity."]


def render_policy_markdown(
  tier_table: pl.DataFrame,
  capacity: pl.DataFrame,
  contrasts: pl.DataFrame,
  slots: pl.DataFrame,
  audit: dict[str, Any],
  dev_audit: dict[str, Any] | None,
) -> str:
  """Result document with the proxy-decision wording and the pre-written interpretation guide."""
  primary_cap = PRIMARY_CAPACITY
  ctx = "context_adjusted_frozen"
  prim = contrasts.filter(
    (pl.col("score_source") == ctx) & (pl.col("capacity_fraction") == primary_cap)
  )

  def _contrast(name: str) -> dict[str, Any]:
    rows = prim.filter(pl.col("contrast") == name)
    return rows.row(0, named=True) if rows.height else {}

  readings = []
  for name, text in (
    ("B_minus_doc_then_A", "B minus doc_then_A (primary)"),
    ("doc_then_A_minus_A", "doc_then_A minus A"),
    ("doc_then_B_minus_B", "doc_then_B minus B"),
    ("overt_then_A_minus_A", "overt_then_A minus A (negative control)"),
  ):
    row = _contrast(name)
    if not row:
      continue
    readings.append(
      f"- {text}: {_fmt(row['delta_events_captured'])} deaths "
      f"(95% CI {_fmt(row['delta_events_ci_low'])} to {_fmt(row['delta_events_ci_high'])}); "
      f"{_interval_reading(row['delta_events_ci_low'], row['delta_events_ci_high'])}."
    )
  applied = _applied_reading(contrasts, tier_table, ctx)
  s1_rows = contrasts.filter(
    (pl.col("score_source") == "ecg_only_frozen") & (pl.col("capacity_fraction") == primary_cap)
  )
  tier_cols = [
    "group", "family", "n", "n_events", "event_prevalence", "ppv_ci_low", "ppv_ci_high", "sensitivity",
    "top_n_ppv_A", "delta_top_n_minus_group_ppv_A", "delta_top_n_minus_group_ppv_A_ci_low",
    "delta_top_n_minus_group_ppv_A_ci_high", "top_n_ppv_B", "delta_top_n_minus_group_ppv_B",
    "delta_top_n_minus_group_ppv_B_ci_low", "delta_top_n_minus_group_ppv_B_ci_high",
  ]
  cap_cols = [
    "policy", "capacity_fraction", "n_flagged", "n_events_captured", "sensitivity", "ppv",
    "number_needed_to_evaluate", "false_positives_per_event", "jaccard_with_B", "b_identity_recovery",
  ]
  contrast_cols = [
    "contrast", "capacity_fraction", "events_base", "events_comparison", "delta_events_captured",
    "delta_events_ci_low", "delta_events_ci_high",
  ]
  slot_cols = ["policy", "capacity_fraction", "tier", "n", "n_events"]
  lines = [
    "# A7: documented-abnormality policy at equal review capacity",
    "",
    "Analysis status: `posthoc_frozen_extension`. Proxy clinical decision D13 (governance D11): "
    "tier set defined by the study team as a proxy clinical decision; clinical review post hoc.",
    f"Lexicon: `{audit['lexicon_version']}`, hash `{audit['lexicon_hash']}` (unchanged). "
    f"Policy specification hash: `{audit['policy_spec_hash']}`. Source SHA: `{audit['source_sha']}`.",
    f"Locked-test patients: {audit['n_locked_test']:,}; deaths: {audit['n_events']:,}. "
    f"Bootstraps: {audit['n_bootstraps']} stratified patient resamples, probes frozen, seed {audit['seed']}.",
    "",
    "This run used frozen locked-test scores. No probe was refit. Machine text is not adjudication, "
    "and an ECG with no documented phrase is not a normal ECG. Captured deaths are event enrichment, "
    "not treatment benefit; capacity fractions are review budgets, not probability thresholds. "
    "D-BETA was pretrained on MIMIC-IV-ECG waveforms and the machine text comes from the same tracings, "
    "so overlap between B and documented abnormalities is expected and is not a mechanism. "
    "This is an in-domain MIMIC analysis, not external validation.",
    "",
    "## Tiers (decision D13)",
    "",
    "| Tier | Rule |",
    "| --- | --- |",
    "| Tier 1, definite emergency | `definite_overt` non-empty (acute STEMI, ventricular tachycardia, ventricular fibrillation, complete AV block) |",
    "| Tier 2, hedged machine alert | `uncertain_overt` non-empty and no Tier 1. A workflow trigger, not a diagnosis |",
    f"| Tier 3, documented major | any of {', '.join(f'`{f}`' for f in TIER3_FLAGS)}, and no Tier 1 or 2 |",
    "| Untiered | everything else, including missing reports |",
    "",
    "Isolated right bundle branch block and suggested pericarditis are excluded from Tier 3 and "
    "reported for transparency only. Within a tier, patients are ordered by C0+A risk (`p_a`).",
    "",
  ]
  if dev_audit is not None:
    lines += [
      "## Development tier audit (no outcomes, no scores)",
      "",
      f"Development reports: {dev_audit['n_development']:,}. Tier counts: "
      + ", ".join(f"{k} {v:,}" for k, v in dev_audit["tier_counts"].items()) + ".",
      f"Missing machine reports: {dev_audit['n_missing_report']:,}. "
      f"Tier 3 reports with two or more Tier 3 flags: {dev_audit['n_tier3_with_two_or_more_flags']:,}. "
      f"Isolated RBBB left untiered: {dev_audit['n_rbbb_only_untiered']:,}; "
      f"isolated suggested pericarditis left untiered: {dev_audit['n_pericarditis_only_untiered']:,}.",
      f"Tier 2 reports: {dev_audit['tier2_reports']:,}. Hedging wording present: "
      f"uncertainty {dev_audit['tier2_uncertainty_wording']:,}, temporality {dev_audit['tier2_temporality_wording']:,} "
      f"(temporality share {_fmt(dev_audit['tier2_temporality_share'])}), both {dev_audit['tier2_both_wordings']:,}.",
      "",
    ]
  lines += [
    "## P1: PPV of documented abnormalities versus the same number of top-ranked patients",
    "",
    "Context-adjusted scores. `delta_top_n_minus_group_ppv_*` is the PPV of the same number of "
    "patients ranked highest by that score minus the group PPV, with a bootstrap interval. "
    "Tier 1 is reported within Tier 1∪2 when its own cell is suppressed under D8 "
    "(this pooling was fixed before counts were seen). A suppressed cell has fewer than 11 patients "
    "or 1 to 10 deaths, or is the complementary cell that would let a total reveal it.",
    "",
    _md_table(tier_table.filter(pl.col("score_source") == ctx), tier_cols),
    "",
    "## P2: equal-capacity policies",
    "",
    f"Primary capacity is {primary_cap:.1%}. `b_identity_recovery` is the share of deaths captured by B "
    "that the policy also captures.",
    "",
    _md_table(capacity.filter(pl.col("score_source") == ctx), cap_cols),
    "",
    "### Paired contrasts on deaths captured",
    "",
    _md_table(contrasts.filter(pl.col("score_source") == ctx), contrast_cols),
    "",
    f"### Reading at {primary_cap:.1%} capacity (context-adjusted)",
    "",
    *readings,
    "",
    "Pre-written interpretation (applied by reading the intervals above, not retuned):",
    "",
    "- `B − doc_then_A` interval above zero: D-BETA's prioritization advantage is not recoverable from "
    "documented abnormalities plus CIIS morphology.",
    "- Interval includes zero and `doc_then_A > A`: much of B's tail advantage over A is reproducible "
    "from what the machine read already documents.",
    "- `doc_then_B < B`: forcing documented abnormalities to the front displaces higher-risk patients B would have selected.",
    "- Top-n PPV by `p_b` above the Tier 3 PPV: at equal size, a B-ranked list is more death-enriched than a documented-abnormality list.",
    "",
    "### Applied reading (computed from the intervals, not retuned)",
    "",
    *applied,
    "",
    "### Review slots by tier group",
    "",
    "Tier 1 is pooled with Tier 2 here for the same D8 reason as in P1.",
    "",
    _md_table(slots.filter(pl.col("score_source") == ctx), slot_cols),
    "",
    "## S1: ECG-only scores",
    "",
    "The same P1 and P2 tables with ECG-only `p_a_ecg` and `p_b_ecg` are in "
    "`tables/documented_policy_*.csv` (rows with `score_source = ecg_only_frozen`). "
    f"Contrasts at {primary_cap:.1%} capacity:",
    "",
    _md_table(s1_rows, contrast_cols),
    "",
    "## Limits",
    "",
    "- Machine text is a documentation proxy, not a clinician reading. Hedged language is a workflow "
    "trigger, not a diagnosis.",
    "- Tier 3 members are common, so Tier 3 behaves like a broad enrichment filter, not a sharp alert.",
    "- Jennifer Lewey's post-hoc review covers tier membership, the RBBB and pericarditis exclusions, "
    "the hedged-alert treatment, and the blinded development sample. A revision is run as a labelled v2 "
    "next to this result.",
    "",
  ]
  return "\n".join(lines)


def run_development_audit(
  cohort_path: Path,
  measurements_path: Path,
  dictionary_path: Path,
  output_dir: Path,
  source_sha: str,
) -> dict[str, Any]:
  """Development-only tier audit plus the blinded review sample. Joins no outcomes or scores."""
  dictionary = load_phenotype_dictionary(dictionary_path)
  assert_frozen_lexicon(dictionary_path)
  cohort = pl.read_parquet(cohort_path)
  if "split" not in cohort.columns:
    raise ValueError("Cohort is missing split. Refusing to classify an unlabeled population.")
  dev_ids = cohort.filter(pl.col("split") == "dev").select(["study_id"])
  columns = [f"report_{i}" for i in range(18)]
  reports = (
    pl.scan_csv(measurements_path, infer_schema_length=500)
    .select(["study_id", *columns])
    .join(dev_ids.lazy(), on="study_id", how="inner")
    .collect()
  )
  if reports.height != dev_ids.height:
    raise ValueError(f"Development audit is missing {dev_ids.height - reports.height} machine-report rows.")
  phenotypes = phenotype_frame(reports, dictionary)
  audit = development_tier_audit(phenotypes, reports, dictionary)
  audit["source_sha"] = source_sha
  sample = draw_review_sample(phenotypes, reports)
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "development_tier_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  sample.write_parquet(output_dir / "review_sample_blinded.parquet")
  sample.write_csv(output_dir / "review_sample_blinded.csv")
  audit["n_review_sample"] = sample.height
  return audit


def run_documented_policy(
  predictions_context_path: Path,
  predictions_ecg_path: Path,
  measurements_path: Path,
  dictionary_path: Path,
  dev_audit_path: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = N_BOOTSTRAPS,
  seed: int = SEED,
) -> dict[str, Any]:
  """Locked-test run. Refuses a changed lexicon, a changed A7 specification, or changed run settings."""
  lexicon_hash = assert_frozen_lexicon(dictionary_path)
  spec_hash = assert_frozen_policy_spec()
  if n_bootstraps != N_BOOTSTRAPS or seed != SEED:
    raise RuntimeError("n_bootstraps and seed are part of the frozen A7 specification.")
  dictionary = load_phenotype_dictionary(dictionary_path)
  context = pl.read_parquet(predictions_context_path)
  ecg = pl.read_parquet(predictions_ecg_path).select(["study_id", "p_a_ecg", "p_b_ecg"])
  if context["study_id"].n_unique() != context.height:
    raise ValueError("Context predictions are not unique by study_id.")
  columns = [f"report_{i}" for i in range(18)]
  reports = (
    pl.scan_csv(measurements_path, infer_schema_length=500)
    .select(["study_id", *columns])
    .join(context.select("study_id").lazy(), on="study_id", how="inner")
    .collect()
  )
  phenotypes = phenotype_frame(reports, dictionary)
  base = (
    context.join(phenotypes, on="study_id", how="left")
    .join(ecg, on="study_id", how="inner")
  )
  if base.height != context.height:
    raise ValueError("Joining machine text or ECG-only scores changed the locked-test row count.")
  y = np.asarray(base["mortality_30d"].to_numpy(), dtype=np.int64)
  level = tier_levels(base)
  flags = flag_membership(base["other_major"].to_list())

  tier_frames: list[pl.DataFrame] = []
  capacity_frames: list[pl.DataFrame] = []
  slot_frames: list[pl.DataFrame] = []
  contrast_frames: list[pl.DataFrame] = []
  for offset, (source, (col_a, col_b)) in enumerate(SCORE_SOURCES.items()):
    p_a = np.asarray(base[col_a].to_numpy(), dtype=np.float64)
    p_b = np.asarray(base[col_b].to_numpy(), dtype=np.float64)
    scores = policy_scores(level, p_a, p_b)
    tier_frames.append(tier_ppv_table(
      y, level, flags, {"A": p_a, "B": p_b}, source, n_bootstraps, seed + 100 * offset,
    ))
    capacity_frames.append(policy_capacity_table(y, level, scores, source))
    slot_frames.append(policy_slot_table(y, level, scores, source))
    contrast_frames.append(
      pl.DataFrame(fixed_capacity_contrast_rows(
        y, scores, CONTRASTS, DEFAULT_CAPACITY_FRACTIONS, n_bootstraps, seed + 100 * offset,
      )).with_columns(pl.lit(source).alias("score_source"))
    )
  tier_raw = pl.concat(tier_frames)
  capacity_raw = pl.concat(capacity_frames)
  slots_raw = pl.concat(slot_frames)
  contrast_raw = pl.concat(contrast_frames)
  tier_public = public_tier_table(tier_raw)
  capacity_public = public_capacity_table(capacity_raw)
  contrast_public = public_contrast_table(contrast_raw)
  slots_public = public_slot_table(slots_raw)

  dev_audit: dict[str, Any] | None = None
  if dev_audit_path.exists():
    loaded: dict[str, Any] = json.loads(dev_audit_path.read_text(encoding="utf-8"))
    if loaded.get("lexicon_hash") != lexicon_hash:
      raise RuntimeError("The development tier audit was produced under a different lexicon hash.")
    dev_audit = loaded
  audit = {
    "analysis_status": "posthoc_frozen_extension",
    "decision_ids": "D1, D8, D11, D13",
    "lexicon_version": dictionary.version,
    "lexicon_hash": lexicon_hash,
    "policy_spec_hash": spec_hash,
    "source_sha": source_sha,
    "n_locked_test": base.height,
    "n_events": int(y.sum()),
    "n_missing_report": int((base["phenotype_reason"] == "missing_report").sum()),
    "tier_counts": {TIER_NAMES[lvl]: int((level == lvl).sum()) for lvl in TIER_NAMES},
    "n_bootstraps": n_bootstraps,
    "seed": seed,
    "models_refit": False,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  base.select(["study_id", "subject_id", "mortality_30d"]).with_columns(
    pl.Series("tier_level", level),
    pl.Series("tier", [TIER_NAMES[int(v)] for v in level]),
  ).write_parquet(output_dir / "locked_test_tiers.parquet")
  (output_dir / "documented_policy_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  report = render_policy_markdown(tier_public, capacity_public, contrast_public, slots_public, audit, dev_audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  tables = report_path.parent / "tables"
  tables.mkdir(parents=True, exist_ok=True)
  drop = ["suppress_status"]
  tier_public.drop(drop).write_csv(tables / "documented_policy_tier_ppv.csv")
  capacity_public.drop(drop).write_csv(tables / "documented_policy_capacity.csv")
  contrast_public.drop(drop).write_csv(tables / "documented_policy_contrasts.csv")
  slots_public.drop(drop).write_csv(tables / "documented_policy_slots.csv")
  return audit
