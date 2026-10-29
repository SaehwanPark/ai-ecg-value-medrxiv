"""A6b: locked-test analysis of post-ECG next-step evaluation by risk group.

Decision D12 (proxy clinical decision under D11). Runs once against the frozen
``configs/next_step_outcomes_v1.yaml``. Clinicians never saw the A or B scores, so
concordance between risk and later actions is observational: it does not show that
the ECG prompted an action, that the action was appropriate, or that a prompt would
have changed an outcome. "No action observed" is not "no action taken". Probes are
frozen; no model is refit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import numpy.typing as npt
import polars as pl

from ecg_rep.analysis import DEFAULT_CAPACITY_FRACTIONS, _stratified_bootstrap_indices
from ecg_rep.cacute import INCLUDED_SETTINGS
from ecg_rep.clinical_setting import wilson_interval
from ecg_rep.mortality_sensitivity import (
  RISK_GROUPS,
  apply_frozen_risk_groups,
  frozen_risk_cuts,
  public_suppression,
)
from ecg_rep.next_step import (
  NextStepConfig,
  NextStepSources,
  assert_frozen_config,
  build_outcome_frame,
  death_in_window_expr,
  died_without_action,
  ensure_caches,
  load_sources,
)
from ecg_rep.probe import bootstrap_paired_difference, compute_auroc

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

PRIMARY_CAPACITY = 0.05
VARIANT_PRIMARY = "primary"
VARIANT_EXCLUDE_DEATH = "exclude_died_in_window_no_action"
VARIANTS = (VARIANT_PRIMARY, VARIANT_EXCLUDE_DEATH)
SCORE_SOURCES: dict[str, tuple[str, str]] = {
  "context_adjusted_frozen": ("p_a", "p_b"),
  "ecg_only_frozen": ("p_a_ecg", "p_b_ecg"),
}
DISCORDANT_LOW_A = "a_low_b_high"
DISCORDANT_HIGH_A = "a_high_b_low"
CELL_KEY = ("score_source", "variant", "outcome_id", "population", "setting")


@dataclass(frozen=True)
class Cell:
  """One outcome by setting by population cell over the fixed locked-test subset."""

  outcome_id: str
  population: str
  setting: str
  is_primary: bool
  eligible: BoolArray
  action: BoolArray
  died_no_action: BoolArray

  def mask(self, variant: str) -> BoolArray:
    """Eligible patients; the sensitivity variant also drops deaths with no observed action."""
    if variant == VARIANT_EXCLUDE_DEATH:
      return self.eligible & ~self.died_no_action
    return self.eligible


def add_death_columns(frame: pl.DataFrame, config: NextStepConfig) -> pl.DataFrame:
  """Per-outcome ``dna_<id>``: died inside that outcome's window with no observed action."""
  out = frame
  for spec in config.outcomes:
    action = pl.col(f"act_{spec.outcome_id}").fill_null(False)
    out = out.with_columns(
      died_without_action(action, death_in_window_expr(spec.window_hours)).alias(f"dna_{spec.outcome_id}")
    )
  return out


def build_cells(frame: pl.DataFrame, config: NextStepConfig) -> list[Cell]:
  """The outcome by setting by population cells, matching the feasibility report."""
  cells: list[Cell] = []
  setting = frame["setting"].to_numpy()
  naive = frame["lab_naive"].fill_null(False).to_numpy().astype(bool)
  prior = frame["troponin_in_prior_window"].fill_null(False).to_numpy().astype(bool)
  for spec in config.outcomes:
    eid = spec.outcome_id
    elig = frame[f"elig_{eid}"].fill_null(False).to_numpy().astype(bool)
    act = frame[f"act_{eid}"].fill_null(False).to_numpy().astype(bool)
    dna = frame[f"dna_{eid}"].fill_null(False).to_numpy().astype(bool)
    for name in spec.settings:
      in_setting = elig & (setting == name)
      if eid == "N1":
        pops = [("lab_naive", True, in_setting & naive), ("all_eligible", False, in_setting)]
      elif eid == "N2":
        pops = [
          ("all_eligible", True, in_setting),
          ("troponin_in_prior_window", False, in_setting & prior),
          ("no_prior_troponin", False, in_setting & ~prior),
        ]
      else:
        pops = [("all_eligible", True, in_setting)]
      for population, primary, mask in pops:
        cells.append(Cell(eid, population, name, primary, mask, act, dna))
  return cells


def group_cell_rows(
  cells: list[Cell],
  y: IntArray,
  groups: npt.NDArray[np.str_],
  source: str,
  variant: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
  """P1 action rates by concordance group and P2 mortality by group and action (raw counts)."""
  p1: list[dict[str, Any]] = []
  p2: list[dict[str, Any]] = []
  for cell in cells:
    mask = cell.mask(variant)
    for group in RISK_GROUPS:
      gm = mask & (groups == group)
      n = int(gm.sum())
      acted = int((gm & cell.action).sum())
      low, high = wilson_interval(acted, n)
      base = {
        "score_source": source, "variant": variant, "outcome_id": cell.outcome_id,
        "population": cell.population, "setting": cell.setting, "is_primary": cell.is_primary,
        "risk_group": group,
      }
      p1.append({
        **base, "n": n, "n_action": acted,
        "action_rate": (acted / n) if n else None,
        "action_rate_ci_low": low if n else None,
        "action_rate_ci_high": high if n else None,
      })
      for label, sel in (("action", gm & cell.action), ("no_action", gm & ~cell.action)):
        m = int(sel.sum())
        d = int((sel & (y == 1)).sum())
        dl, dh = wilson_interval(d, m)
        p2.append({
          **base, "action_status": label, "n": m, "n_deaths": d,
          "death_rate": (d / m) if m else None,
          "death_rate_ci_low": dl if m else None,
          "death_rate_ci_high": dh if m else None,
        })
  return pl.DataFrame(p1), pl.DataFrame(p2)


def discordant_contrast_rows(
  cells: list[Cell],
  groups: npt.NDArray[np.str_],
  source: str,
  variant: str,
  n_bootstraps: int,
  seed: int,
) -> pl.DataFrame:
  """Headline: action rate in A-low/B-high minus A-high/B-low, with a patient bootstrap interval."""
  rng = np.random.default_rng(seed)
  rows: list[dict[str, Any]] = []
  for cell in cells:
    ix = np.where(cell.mask(variant))[0]
    g = groups[ix]
    a = cell.action[ix]
    low_m = g == DISCORDANT_LOW_A
    high_m = g == DISCORDANT_HIGH_A
    low = float(a[low_m].mean()) if low_m.any() else None
    high = float(a[high_m].mean()) if high_m.any() else None
    diff = (low - high) if (low is not None and high is not None) else None
    draws: list[float] = []
    if diff is not None:
      for _ in range(n_bootstraps):
        pick = rng.integers(0, len(ix), size=len(ix))
        pl_m, ph_m, pa = low_m[pick], high_m[pick], a[pick]
        if pl_m.any() and ph_m.any():
          draws.append(float(pa[pl_m].mean() - pa[ph_m].mean()))
    ci_low = float(np.percentile(draws, 2.5)) if draws else None
    ci_high = float(np.percentile(draws, 97.5)) if draws else None
    rows.append({
      "score_source": source, "variant": variant, "outcome_id": cell.outcome_id,
      "population": cell.population, "setting": cell.setting, "is_primary": cell.is_primary,
      "rate_a_low_b_high": low, "rate_a_high_b_low": high,
      "delta_rate_low_a_minus_high_a": diff,
      "delta_ci_low": ci_low, "delta_ci_high": ci_high,
      "n_bootstraps": n_bootstraps, "models_refit_inside_bootstrap": False,
    })
  return pl.DataFrame(rows)


def _capacity_size(n: int, fraction: float) -> int:
  return max(1, int(round(n * fraction)))


def unevaluated_capacity_rows(
  cells: list[Cell],
  y: IntArray,
  scores: dict[str, FloatArray],
  source: str,
  variant: str,
  fractions: tuple[float, ...],
  n_bootstraps: int,
  seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
  """P3: flagged patients without the action, and deaths among them, at fixed capacity.

  Capacity is a review budget on the whole fixed locked-test subset (rank by ``p_a``
  and separately by ``p_b``); each cell then counts its own flagged patients. The
  B minus A difference is a paired bootstrap that re-ranks inside every resample.
  """
  n = len(y)
  eligible = np.stack([cell.mask(variant) for cell in cells], axis=1).astype(np.int64)
  unevaluated = np.stack(
    [cell.mask(variant) & ~cell.action for cell in cells], axis=1
  ).astype(np.int64)
  deaths = unevaluated * y[:, None]
  sizes = [_capacity_size(n, f) for f in fractions]

  def _counts(idx: IntArray, label: str) -> tuple[IntArray, IntArray, IntArray]:
    order = np.argsort(-scores[label][idx])
    flag_e = np.zeros((len(fractions), len(cells)), dtype=np.int64)
    flag_u = np.zeros_like(flag_e)
    flag_d = np.zeros_like(flag_e)
    for c, k in enumerate(sizes):
      top = idx[order[:k]]
      flag_e[c] = eligible[top].sum(axis=0)
      flag_u[c] = unevaluated[top].sum(axis=0)
      flag_d[c] = deaths[top].sum(axis=0)
    return flag_e, flag_u, flag_d

  full = np.arange(n)
  point = {label: _counts(full, label) for label in ("A", "B")}
  rng = np.random.default_rng(seed)
  boot_deaths = {label: np.zeros((n_bootstraps, len(fractions), len(cells))) for label in ("A", "B")}
  boot_unev = {label: np.zeros((n_bootstraps, len(fractions), len(cells))) for label in ("A", "B")}
  for b in range(n_bootstraps):
    idx = _stratified_bootstrap_indices(y, rng)
    for label in ("A", "B"):
      _, flag_u, flag_d = _counts(idx, label)
      boot_deaths[label][b] = flag_d
      boot_unev[label][b] = flag_u
  metric_rows: list[dict[str, Any]] = []
  contrast_rows: list[dict[str, Any]] = []
  for j, cell in enumerate(cells):
    for c, fraction in enumerate(fractions):
      base = {
        "score_source": source, "variant": variant, "outcome_id": cell.outcome_id,
        "population": cell.population, "setting": cell.setting, "is_primary": cell.is_primary,
        "capacity_fraction": fraction,
      }
      for label in ("A", "B"):
        flag_e, flag_u, flag_d = point[label]
        metric_rows.append({
          **base, "score": label,
          "n_flagged_eligible": int(flag_e[c, j]),
          "n_flagged_unevaluated": int(flag_u[c, j]),
          "deaths_among_unevaluated": int(flag_d[c, j]),
        })
      diff_d = point["B"][2][c, j] - point["A"][2][c, j]
      diff_u = point["B"][1][c, j] - point["A"][1][c, j]
      contrast_rows.append({
        **base,
        "deaths_unevaluated_A": int(point["A"][2][c, j]),
        "deaths_unevaluated_B": int(point["B"][2][c, j]),
        "delta_deaths_unevaluated_b_minus_a": int(diff_d),
        "delta_deaths_ci_low": float(np.percentile(boot_deaths["B"][:, c, j] - boot_deaths["A"][:, c, j], 2.5)),
        "delta_deaths_ci_high": float(np.percentile(boot_deaths["B"][:, c, j] - boot_deaths["A"][:, c, j], 97.5)),
        "n_unevaluated_A": int(point["A"][1][c, j]),
        "n_unevaluated_B": int(point["B"][1][c, j]),
        "delta_n_unevaluated_b_minus_a": int(diff_u),
        "n_bootstraps": n_bootstraps, "models_refit_inside_bootstrap": False,
      })
  return pl.DataFrame(metric_rows), pl.DataFrame(contrast_rows)


def alignment_rows(
  cells: list[Cell],
  scores: dict[str, FloatArray],
  source: str,
  n_bootstraps: int,
  seed: int,
) -> pl.DataFrame:
  """S1: how well each frozen score ranks the action itself inside a cell (descriptive alignment)."""
  rows: list[dict[str, Any]] = []
  for j, cell in enumerate(cells):
    ix = np.where(cell.eligible)[0]
    act = cell.action[ix].astype(np.int64)
    base: dict[str, Any] = {
      "score_source": source, "outcome_id": cell.outcome_id, "population": cell.population,
      "setting": cell.setting, "is_primary": cell.is_primary, "n": int(len(ix)), "n_action": int(act.sum()),
      "auroc_a": None, "auroc_b": None, "delta_auroc_b_minus_a": None,
      "delta_auroc_ci_low": None, "delta_auroc_ci_high": None,
    }
    if act.sum() >= 11 and (len(act) - act.sum()) >= 11:
      sa, sb = scores["A"][ix], scores["B"][ix]
      diff = bootstrap_paired_difference(act, sa, sb, "auroc", n_bootstraps, seed + j)
      base.update({
        "auroc_a": float(compute_auroc(act, sa)), "auroc_b": float(compute_auroc(act, sb)),
        "delta_auroc_b_minus_a": diff.point_estimate,
        "delta_auroc_ci_low": diff.ci_lower, "delta_auroc_ci_high": diff.ci_upper,
      })
    rows.append(base)
  return pl.DataFrame(rows)


def _blank(frame: pl.DataFrame, hidden: pl.Expr, columns: list[str]) -> pl.DataFrame:
  return frame.with_columns(
    [pl.when(hidden).then(None).otherwise(pl.col(col)).alias(col) for col in columns]
  )


def public_p1(p1: pl.DataFrame) -> pl.DataFrame:
  """D8 for action rates. Four groups partition each cell, so complementary suppression applies.

  A group is also hidden when fewer than 11 of its patients lack the action, because the
  published denominator and action count would reveal that count by subtraction.
  """
  work = p1.with_columns(
    pl.min_horizontal(pl.col("n_action"), pl.col("n") - pl.col("n_action")).alias("n_events"),
    pl.col("action_rate").alias("event_prevalence"),
  )
  shown = public_suppression(work, CELL_KEY, metric_cols=())
  status = shown["suppress_status"]
  out = p1.with_columns(status.alias("suppress_status"))
  hidden = pl.col("suppress_status").is_in(["primary", "complement"])
  return _blank(out, hidden, ["n", "n_action", "action_rate", "action_rate_ci_low", "action_rate_ci_high"])


def public_p2(p2: pl.DataFrame) -> pl.DataFrame:
  """D8 for mortality by group and action: deaths under 11, or a denominator under 11, are hidden."""
  work = p2.with_columns(
    pl.col("n_deaths").alias("n_events"),
    pl.col("death_rate").alias("event_prevalence"),
  )
  shown = public_suppression(work, (*CELL_KEY,), metric_cols=())
  out = p2.with_columns(shown["suppress_status"].alias("suppress_status"))
  hidden = pl.col("suppress_status").is_in(["primary", "complement"])
  return _blank(out, hidden, ["n", "n_deaths", "death_rate", "death_rate_ci_low", "death_rate_ci_high"])


def public_contrast(contrast: pl.DataFrame, p1_public: pl.DataFrame) -> pl.DataFrame:
  """Hide the headline contrast when either discordant group's action rate is hidden."""
  hidden_groups = (
    p1_public.filter(
      pl.col("risk_group").is_in([DISCORDANT_LOW_A, DISCORDANT_HIGH_A])
      & pl.col("suppress_status").is_in(["primary", "complement"])
    ).select([*CELL_KEY]).unique()
  ).with_columns(pl.lit(True).alias("_hidden"))
  joined = contrast.join(hidden_groups, on=list(CELL_KEY), how="left")
  hide = pl.col("_hidden").fill_null(False)
  cols = [
    "rate_a_low_b_high", "rate_a_high_b_low", "delta_rate_low_a_minus_high_a", "delta_ci_low", "delta_ci_high",
  ]
  return _blank(joined, hide, cols).drop("_hidden")


def _small(n: pl.Expr, events: pl.Expr) -> pl.Expr:
  return (n < 11) | ((events > 0) & (events < 11))


def public_capacity(metrics: pl.DataFrame, contrasts: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
  """D8 for P3 rows. A B-minus-A difference is hidden when either side is hidden."""
  hide = _small(pl.col("n_flagged_unevaluated"), pl.col("deaths_among_unevaluated")) | (
    pl.col("n_flagged_eligible") < 11
  )
  hidden_metrics = metrics.with_columns(hide.alias("_hide"))
  public_metrics = _blank(
    hidden_metrics, pl.col("_hide"), ["n_flagged_eligible", "n_flagged_unevaluated", "deaths_among_unevaluated"]
  ).drop("_hide")
  contrast_hide = (
    _small(pl.col("n_unevaluated_A"), pl.col("deaths_unevaluated_A"))
    | _small(pl.col("n_unevaluated_B"), pl.col("deaths_unevaluated_B"))
  )
  public_contrasts = _blank(
    contrasts.with_columns(contrast_hide.alias("_hide")),
    pl.col("_hide"),
    [
      "deaths_unevaluated_A", "deaths_unevaluated_B", "delta_deaths_unevaluated_b_minus_a",
      "delta_deaths_ci_low", "delta_deaths_ci_high", "n_unevaluated_A", "n_unevaluated_B",
      "delta_n_unevaluated_b_minus_a",
    ],
  ).drop("_hide")
  return public_metrics, public_contrasts


def public_alignment(alignment: pl.DataFrame) -> pl.DataFrame:
  """Hide alignment rows with fewer than 11 patients on either side of the action."""
  hide = (pl.col("n") < 11) | (pl.col("n_action") < 11) | ((pl.col("n") - pl.col("n_action")) < 11)
  return _blank(
    alignment.with_columns(hide.alias("_hide")),
    pl.col("_hide"),
    ["n", "n_action", "auroc_a", "auroc_b", "delta_auroc_b_minus_a", "delta_auroc_ci_low", "delta_auroc_ci_high"],
  ).drop("_hide")


def _fmt(value: Any, digits: int = 3) -> str:
  if value is None:
    return "suppressed"
  if isinstance(value, float):
    return f"{value:.{digits}f}"
  return str(value)


def _md_table(frame: pl.DataFrame, columns: list[str]) -> str:
  header = "| " + " | ".join(columns) + " |"
  rule = "| " + " | ".join("---" for _ in columns) + " |"
  body = [
    "| " + " | ".join(_fmt(row[col]) for col in columns) + " |"
    for row in frame.select(columns).iter_rows(named=True)
  ]
  return "\n".join([header, rule, *body])


def _contrast_label(low: float | None, high: float | None) -> str:
  if low is None or high is None:
    return "suppressed"
  if high < 0:
    return "A-low/B-high evaluated less often"
  if low > 0:
    return "A-low/B-high evaluated more often"
  return "similar (interval includes zero)"


def _unevaluated_label(low: float | None, high: float | None) -> str:
  if low is None or high is None:
    return "suppressed"
  if low > 0:
    return "B flags more unevaluated deaths"
  if high < 0:
    return "A flags more unevaluated deaths"
  return "similar (interval includes zero)"


def _applied_reading(contrast: pl.DataFrame, p3: pl.DataFrame, s1: pl.DataFrame) -> list[str]:
  """Map the observed counts onto the pre-written guide without adding new claims."""
  more = contrast.filter(pl.col("reading") == "A-low/B-high evaluated more often").height
  less = contrast.filter(pl.col("reading") == "A-low/B-high evaluated less often").height
  b_more = p3.filter(pl.col("reading") == "B flags more unevaluated deaths").height
  a_more = p3.filter(pl.col("reading") == "A flags more unevaluated deaths").height
  p3_shown = p3.filter(pl.col("delta_deaths_ci_low").is_not_null()).height
  s1_shown = s1.filter(pl.col("delta_auroc_ci_low").is_not_null())
  s1_b = s1_shown.filter(pl.col("delta_auroc_ci_low") > 0).height
  s1_a = s1_shown.filter(pl.col("delta_auroc_ci_high") < 0).height
  lines = [
    f"- P1: patients that only B ranks high were evaluated less often than patients that only A ranks high in {less} cells "
    f"and more often in {more}. {'The first row of the guide is not supported: ' if less == 0 else ''}"
    "B-only high-risk patients are not being passed over by the observed workup in these outcomes.",
    f"- P3 at {PRIMARY_CAPACITY:.1%} capacity: B flags more unevaluated deaths than A in {b_more} of {p3_shown} unsuppressed cells, "
    f"A flags more in {a_more}; the rest are similar.",
    f"- S1: B aligns better with the observed action than A in {s1_b} of {s1_shown.height} unsuppressed cells, A aligns better in {s1_a}.",
    "- Taken together this fits the third and second guide rows more than the first: much of what B adds is visible in who received "
    "follow-up, so these data do not show B-only high-risk patients being evaluated less. They cannot show that follow-up was appropriate or that it changed outcomes.",
  ]
  return lines


def render_results_markdown(
  tables: dict[str, pl.DataFrame],
  audit: dict[str, Any],
) -> str:
  """Result document with the proxy-decision wording and the pre-written interpretation guide."""
  ctx = "context_adjusted_frozen"
  prim = VARIANT_PRIMARY
  contrast = tables["contrast"].filter(
    (pl.col("score_source") == ctx) & (pl.col("variant") == prim) & pl.col("is_primary")
  ).with_columns(
    pl.struct(["delta_ci_low", "delta_ci_high"]).map_elements(
      lambda s: _contrast_label(s["delta_ci_low"], s["delta_ci_high"]), return_dtype=pl.String
    ).alias("reading")
  )
  key_cells = tables["p2"].filter(
    (pl.col("score_source") == ctx) & (pl.col("variant") == prim) & pl.col("is_primary")
    & (pl.col("risk_group") == DISCORDANT_LOW_A) & (pl.col("action_status") == "no_action")
  )
  p3 = tables["p3_contrast"].filter(
    (pl.col("score_source") == ctx) & (pl.col("variant") == prim) & pl.col("is_primary")
    & (pl.col("capacity_fraction") == PRIMARY_CAPACITY)
  ).with_columns(
    pl.struct(["delta_deaths_ci_low", "delta_deaths_ci_high"]).map_elements(
      lambda s: _unevaluated_label(s["delta_deaths_ci_low"], s["delta_deaths_ci_high"]), return_dtype=pl.String
    ).alias("reading")
  )
  s1 = tables["alignment"].filter((pl.col("score_source") == ctx) & pl.col("is_primary"))
  p1_rates = tables["p1"].filter(
    (pl.col("score_source") == ctx) & (pl.col("variant") == prim) & pl.col("is_primary")
  )
  n_cells = contrast.height
  n_supp = contrast.filter(pl.col("delta_ci_low").is_null()).height
  n_similar = contrast.filter(
    pl.col("delta_ci_low").is_not_null() & (pl.col("delta_ci_low") <= 0) & (pl.col("delta_ci_high") >= 0)
  ).height
  n_less = contrast.filter(pl.col("delta_ci_high") < 0).height
  n_more = contrast.filter(pl.col("delta_ci_low") > 0).height
  lines = [
    "# A6: post-ECG next-step evaluation by A/B risk group",
    "",
    "Analysis status: `posthoc_frozen_extension`. Proxy clinical decision D12 (governance D11): "
    "outcome set defined by the study team as a proxy clinical decision; clinical review post hoc.",
    f"Config: `{audit['config_version']}`, sha256 `{audit['config_hash']}` (frozen after development-only feasibility). "
    f"Source SHA: `{audit['source_sha']}`.",
    f"Population: Cacute fixed locked-test subset (paired-valid, non-ambiguous), {audit['n_fixed_test']:,} patients, "
    f"{audit['n_fixed_deaths']:,} deaths. ICU index ECGs ({audit['n_icu_in_fixed']:,}) are excluded from every action outcome and only counted. "
    f"Concordance groups use the frozen D3 cutoffs (80th percentile of the full {audit['n_paired_valid_test']:,}-patient paired-valid "
    f"locked test): context-adjusted A {audit['context_a_cut']:.4f}, B {audit['context_b_cut']:.4f}. "
    f"{audit['n_bootstraps']} patient bootstraps, probes frozen, seed {audit['seed']}.",
    "",
    "Clinicians did not see the A or B scores. Concordance between risk and later actions is observational: it does not show "
    "that the ECG prompted the action, that the action was appropriate, or that a prompt would have changed an outcome. "
    "\"No action observed\" is not \"no action taken\": care outside BIDMC, transfers out, goals-of-care decisions, and death "
    "before the window closed all produce an unobserved action. Actions are downstream process outcomes, never predictors, "
    "and they are separate from Cacute, which uses only pre-ECG information. This is an in-domain MIMIC analysis, not external "
    "validation, and settings are never pooled into one action rate. All seven outcome rows are reported whether or not "
    "they show a difference. No p-values; intervals are descriptive.",
    "",
    "A suppressed cell has fewer than 11 patients, or 1 to 10 events or non-events, or is the complementary cell that would "
    "let a published total reveal one (D8).",
    "",
    "## Headline: action rate in A-low/B-high minus A-high/B-low (P1)",
    "",
    "Primary populations: N1 is the lab-naive population; all other outcomes use every eligible patient. "
    "A negative difference means patients that only B ranks high were evaluated less often than patients that only A ranks high.",
    "",
    _md_table(
      contrast,
      ["outcome_id", "population", "setting", "rate_a_low_b_high", "rate_a_high_b_low", "delta_rate_low_a_minus_high_a",
       "delta_ci_low", "delta_ci_high", "reading"],
    ),
    "",
    f"Of {n_cells} primary cells: A-low/B-high evaluated less often in {n_less}, more often in {n_more}, "
    f"similar in {n_similar}, suppressed in {n_supp}.",
    "",
    "## Action rate by concordance group (P1, primary populations)",
    "",
    _md_table(
      p1_rates,
      ["outcome_id", "population", "setting", "risk_group", "n", "n_action", "action_rate", "action_rate_ci_low",
       "action_rate_ci_high"],
    ),
    "",
    "## Mortality among patients without the action: A-low/B-high (P2)",
    "",
    "Full P2 (both action states, all four groups) is in `tables/next_step_p2_mortality.csv`.",
    "",
    _md_table(
      key_cells,
      ["outcome_id", "population", "setting", "n", "n_deaths", "death_rate", "death_rate_ci_low", "death_rate_ci_high"],
    ),
    "",
    f"## Unevaluated deaths at {PRIMARY_CAPACITY:.1%} capacity (P3)",
    "",
    "Capacity is a review budget on the whole fixed subset, ranked by `p_a` and separately by `p_b`; each cell counts its own flagged "
    "patients without the action and the deaths among them. Other capacities are in `tables/next_step_p3_unevaluated_contrasts.csv`.",
    "",
    _md_table(
      p3,
      ["outcome_id", "population", "setting", "deaths_unevaluated_A", "deaths_unevaluated_B",
       "delta_deaths_unevaluated_b_minus_a", "delta_deaths_ci_low", "delta_deaths_ci_high", "n_unevaluated_A",
       "n_unevaluated_B", "reading"],
    ),
    "",
    "## Workflow alignment: AUROC of each score for the action itself (S1)",
    "",
    "A higher B AUROC for an action means B's ranking is closer to what clinicians already did. Frozen probes predict mortality, "
    "not actions; this is descriptive alignment only.",
    "",
    _md_table(
      s1,
      ["outcome_id", "population", "setting", "n", "n_action", "auroc_a", "auroc_b", "delta_auroc_b_minus_a",
       "delta_auroc_ci_low", "delta_auroc_ci_high"],
    ),
    "",
    "## Sensitivity analyses",
    "",
    "S2 (ECG-only scores and ECG-only D3 cutoffs) and S3 (dropping patients who died inside the window with no observed action) "
    "repeat P1 to P3. They are in `tables/next_step_*.csv` (columns `score_source` and `variant`).",
    f"S3 drops, in the primary outcome windows, from {audit['s3_min_dropped']:,} to {audit['s3_max_dropped']:,} patients across cells.",
    "",
    "## Pre-written interpretation guide (applied by reading the intervals, not retuned)",
    "",
    "| Pattern | Reading |",
    "| --- | --- |",
    "| A-low/B-high has lower action rates than A-high/B-low, and higher mortality without the action | B identifies risk that observed workup did not track. This is a hypothesis for a prospective prompt study, not evidence of missed care |",
    "| Action rates are similar across discordant groups | B's extra signal is not visible in which patients received evaluation; its value is in ranking, not in matching workflow |",
    "| B predicts actions better than A (S1) and P2/P3 show few unevaluated deaths | Much of B's prognostic advantage overlaps information that clinicians acted on anyway |",
    "| Mostly suppressed cells in a setting | Report as underpowered in that setting; do not pool settings to manufacture a contrast |",
    "",
    "## Applied reading (counts from the tables above; descriptive)",
    "",
    *_applied_reading(contrast, p3, s1),
    "",
    "## Limits",
    "",
    "- Single-center capture: actions outside BIDMC are invisible, most of all for other/unlinked patients.",
    "- No ED module: ED orders that do not become labs, ECGs, or admissions (for example ED imaging) are not observed. `poe` covers admitted care only.",
    "- Troponin and labs are often protocol-ordered at triage, so N1 and N2 partly reflect protocols rather than ECG-driven judgment. "
    "That is why N1 is analyzed in the lab-naive population and N2 is stratified by prior troponin.",
    "- The frozen probes predict mortality, not actions; S1 is descriptive alignment only.",
    "- Jennifer Lewey's post-hoc review covers the outcome table, the windows, the ICU exclusion, the lab-naive primary population "
    "for N1, and the feasibility drops (none were made). Any revisions she makes would run as `next_step_outcomes_v2` next to this result.",
    "",
  ]
  return "\n".join(lines)


def run_next_step(
  *,
  cohort_path: Path,
  setting_path: Path,
  pair_validity_path: Path,
  predictions_context_path: Path,
  predictions_ecg_path: Path,
  config_path: Path,
  labevents_path: Path,
  poe_path: Path,
  record_list_path: Path,
  transfers_path: Path,
  admissions_df: pl.DataFrame,
  icustays_path: Path,
  cache_dir: Path,
  output_dir: Path,
  report_path: Path,
  source_sha: str,
  n_bootstraps: int = 200,
  seed: int = 42,
) -> dict[str, Any]:
  """Locked-test run. Refuses a config that is not the frozen one."""
  config = assert_frozen_config(config_path)
  cohort = pl.read_parquet(cohort_path).select([
    "study_id", "subject_id", "ecg_datetime", "split", "mortality_30d", "first_hosp_death_dt", "dod_date",
  ])
  setting = pl.read_parquet(setting_path).select(["study_id", "setting", "hadm_id"])
  pair = pl.read_parquet(pair_validity_path).filter(pl.col("representation_pair_valid") == True)  # noqa: E712
  context = pl.read_parquet(predictions_context_path).select(["study_id", "mortality_30d", "p_a", "p_b"])
  ecg = pl.read_parquet(predictions_ecg_path).select(["study_id", "p_a_ecg", "p_b_ecg"])
  paired_valid = context.join(ecg, on="study_id", how="inner")
  if paired_valid.height != context.height:
    raise ValueError("ECG-only predictions do not cover the context-adjusted locked test.")
  test_ids = cohort.filter(pl.col("split") == "test").select("study_id")
  if paired_valid.join(test_ids, on="study_id", how="anti").height:
    raise ValueError("Prediction rows include study_ids outside the locked test split.")
  if paired_valid.join(pair.select("study_id"), on="study_id", how="anti").height:
    raise ValueError("Prediction rows include pairs that are not representation-valid.")

  cuts = {
    source: frozen_risk_cuts(
      np.asarray(paired_valid[col_a].to_numpy(), dtype=np.float64),
      np.asarray(paired_valid[col_b].to_numpy(), dtype=np.float64),
    )
    for source, (col_a, col_b) in SCORE_SOURCES.items()
  }
  index = (
    cohort.join(paired_valid.select("study_id", "p_a", "p_b", "p_a_ecg", "p_b_ecg"), on="study_id", how="inner")
    .join(setting, on="study_id", how="left")
  )
  if index["setting"].null_count():
    raise ValueError("Some locked-test index ECGs have no setting assignment.")
  n_ambiguous = int(index.filter(pl.col("setting") == "ambiguous").height)
  index = index.filter(pl.col("setting").is_in(list(INCLUDED_SETTINGS)))
  y = np.asarray(index["mortality_30d"].cast(pl.Int64).to_numpy(), dtype=np.int64)
  context_y = paired_valid.join(index.select("study_id"), on="study_id", how="inner")["mortality_30d"]
  if int(context_y.sum()) != int(y.sum()):
    raise ValueError("Cohort and prediction files disagree on 30-day mortality.")

  subject_ids = [int(v) for v in index["subject_id"].unique().to_list()]
  all_subjects = [int(v) for v in cohort["subject_id"].unique().to_list()]
  ensure_caches(
    all_subjects, config, labevents_path=labevents_path, poe_path=poe_path, cache_dir=cache_dir,
  )
  sources: NextStepSources = load_sources(
    subject_ids, config, cache_dir=cache_dir, record_list_path=record_list_path,
    transfers_path=transfers_path, admissions_df=admissions_df, icustays_path=icustays_path,
  )
  frame = build_outcome_frame(index, config, **sources.frame_kwargs())
  frame = add_death_columns(
    frame.join(
      index.select("study_id", "first_hosp_death_dt", "dod_date"), on="study_id", how="left",
    ),
    config,
  )
  # Row order of ``frame`` can differ from ``index`` after asof joins; align everything by study_id.
  frame = frame.join(index.select("study_id", "p_a", "p_b", "p_a_ecg", "p_b_ecg", "mortality_30d"), on="study_id", how="left")
  y = np.asarray(frame["mortality_30d"].cast(pl.Int64).to_numpy(), dtype=np.int64)
  cells = build_cells(frame, config)

  p1_parts: list[pl.DataFrame] = []
  p2_parts: list[pl.DataFrame] = []
  contrast_parts: list[pl.DataFrame] = []
  p3_parts: list[pl.DataFrame] = []
  p3c_parts: list[pl.DataFrame] = []
  align_parts: list[pl.DataFrame] = []
  for s_offset, (source, (col_a, col_b)) in enumerate(SCORE_SOURCES.items()):
    scores = {
      "A": np.asarray(frame[col_a].to_numpy(), dtype=np.float64),
      "B": np.asarray(frame[col_b].to_numpy(), dtype=np.float64),
    }
    groups = apply_frozen_risk_groups(scores["A"], scores["B"], *cuts[source])
    align_parts.append(alignment_rows(cells, scores, source, n_bootstraps, seed + 1000 * s_offset))
    for v_offset, variant in enumerate(VARIANTS):
      offset = seed + 100 * s_offset + 10 * v_offset
      p1_raw, p2_raw = group_cell_rows(cells, y, groups, source, variant)
      p1_parts.append(p1_raw)
      p2_parts.append(p2_raw)
      contrast_parts.append(discordant_contrast_rows(cells, groups, source, variant, n_bootstraps, offset))
      metrics, contrasts = unevaluated_capacity_rows(
        cells, y, scores, source, variant, DEFAULT_CAPACITY_FRACTIONS, n_bootstraps, offset + 5,
      )
      p3_parts.append(metrics)
      p3c_parts.append(contrasts)

  p1 = public_p1(pl.concat(p1_parts))
  p2 = public_p2(pl.concat(p2_parts))
  contrast = public_contrast(pl.concat(contrast_parts), p1)
  p3_metrics, p3_contrasts = public_capacity(pl.concat(p3_parts), pl.concat(p3c_parts))
  alignment = public_alignment(pl.concat(align_parts))

  dropped_counts = [
    int(cell.eligible.sum() - cell.mask(VARIANT_EXCLUDE_DEATH).sum()) for cell in cells if cell.is_primary
  ]
  audit = {
    "analysis_status": "posthoc_frozen_extension",
    "decision_ids": "D3, D8, D11, D12",
    "config_version": config.version,
    "config_hash": config.config_hash,
    "source_sha": source_sha,
    "n_paired_valid_test": paired_valid.height,
    "n_fixed_test": frame.height,
    "n_fixed_deaths": int(y.sum()),
    "n_ambiguous_excluded": n_ambiguous,
    "n_icu_in_fixed": int(frame.filter(pl.col("setting") == "ICU").height),
    "context_a_cut": cuts["context_adjusted_frozen"][0],
    "context_b_cut": cuts["context_adjusted_frozen"][1],
    "ecg_a_cut": cuts["ecg_only_frozen"][0],
    "ecg_b_cut": cuts["ecg_only_frozen"][1],
    "n_bootstraps": n_bootstraps,
    "seed": seed,
    "s3_min_dropped": min(dropped_counts) if dropped_counts else 0,
    "s3_max_dropped": max(dropped_counts) if dropped_counts else 0,
    "models_refit": False,
  }
  tables = {
    "p1": p1, "p2": p2, "contrast": contrast, "p3": p3_metrics, "p3_contrast": p3_contrasts, "alignment": alignment,
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  frame.select([
    "study_id", "subject_id", "setting",
    *[f"elig_{s.outcome_id}" for s in config.outcomes],
    *[f"act_{s.outcome_id}" for s in config.outcomes],
    *[f"dna_{s.outcome_id}" for s in config.outcomes],
    "lab_naive", "troponin_in_prior_window",
  ]).write_parquet(output_dir / "locked_test_actions.parquet")
  (output_dir / "next_step_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  report = render_results_markdown(tables, audit)
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(report, encoding="utf-8")
  doc_tables = report_path.parent / "tables"
  doc_tables.mkdir(parents=True, exist_ok=True)
  p1.drop("suppress_status").write_csv(doc_tables / "next_step_p1_action_rates.csv")
  contrast.write_csv(doc_tables / "next_step_p1_discordant_contrast.csv")
  p2.drop("suppress_status").write_csv(doc_tables / "next_step_p2_mortality.csv")
  p3_metrics.write_csv(doc_tables / "next_step_p3_unevaluated.csv")
  p3_contrasts.write_csv(doc_tables / "next_step_p3_unevaluated_contrasts.csv")
  alignment.write_csv(doc_tables / "next_step_s1_alignment.csv")
  return audit
