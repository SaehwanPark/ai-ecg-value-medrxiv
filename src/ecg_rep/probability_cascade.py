"""Patient-wise absolute-risk routing for the post-hoc ECG cascade extension.

The policy in this module is separate from the cohort-relative ranking policies
in :mod:`ecg_rep.cascade`. Routing is a function of one validated Stage A score
and a frozen policy; it never reads or ranks other patients. Raw selective scores
are diagnostic outputs only and are not asserted to be calibrated risks.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Sequence, cast

import numpy as np
import yaml

ModelScope = Literal["ecg_only", "context_adjusted"]
RouteMode = Literal["threshold", "none", "all"]
RouteStatus = Literal["route_to_b", "retain_a", "unable_to_route"]
ReferenceSplit = Literal["dev", "val"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POLICY_FIELDS = frozenset({
  "version",
  "model_scope",
  "reference_dataset",
  "outcome_horizon_days",
  "route_mode",
  "tau_r",
  "inclusive",
  "model_hash_a",
  "model_hash_b",
  "feature_schema_hash_a",
  "feature_schema_hash_b",
  "calibration_id",
  "missing_score_policy",
  "reference_split",
  "reference_source_sha256",
  "target_utilization",
})


def _is_sha256(value: object) -> bool:
  return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: str, name: str) -> None:
  if not _is_sha256(value):
    raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest.")


def _probability(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a finite probability in [0, 1].")
  result = float(value)
  if not math.isfinite(result) or result < 0.0 or result > 1.0:
    raise ValueError(f"{name} must be a finite probability in [0, 1].")
  return result


@dataclass(frozen=True)
class RoutingPolicy:
  """Immutable, provenance-bound patient-wise routing policy.

  ``tau_r`` is the Stage A cutoff for invoking B, not a clinical action
  threshold. Only threshold policies carry a fitted cutoff and reference split;
  the 0%/100% comparators use explicit ``none``/``all`` modes.
  """

  version: str
  model_scope: ModelScope
  reference_dataset: str
  outcome_horizon_days: int
  route_mode: RouteMode
  tau_r: float | None
  inclusive: bool
  model_hash_a: str
  model_hash_b: str
  feature_schema_hash_a: str
  feature_schema_hash_b: str
  calibration_id: str | None = None
  missing_score_policy: Literal["unable_to_route"] = "unable_to_route"
  reference_split: ReferenceSplit | None = None
  reference_source_sha256: str | None = None
  target_utilization: float | None = None

  def __post_init__(self) -> None:
    if self.version != "probability_cascade_v1":
      raise ValueError("Unsupported probability-cascade policy version.")
    if self.model_scope not in ("ecg_only", "context_adjusted"):
      raise ValueError("model_scope must be ecg_only or context_adjusted.")
    if not isinstance(self.reference_dataset, str) or not self.reference_dataset:
      raise ValueError("reference_dataset must be a non-empty dataset/site identifier.")
    if (
      isinstance(self.outcome_horizon_days, bool)
      or not isinstance(self.outcome_horizon_days, int)
      or self.outcome_horizon_days <= 0
    ):
      raise ValueError("outcome_horizon_days must be a positive integer.")
    if not isinstance(self.inclusive, bool):
      raise ValueError("inclusive must be a bool.")
    for name, value in (
      ("model_hash_a", self.model_hash_a),
      ("model_hash_b", self.model_hash_b),
      ("feature_schema_hash_a", self.feature_schema_hash_a),
      ("feature_schema_hash_b", self.feature_schema_hash_b),
    ):
      _require_sha256(value, name)
    if self.missing_score_policy != "unable_to_route":
      raise ValueError("Only the explicit unable_to_route missing-score policy is supported.")
    if self.calibration_id is not None and (not isinstance(self.calibration_id, str) or not self.calibration_id):
      raise ValueError("calibration_id must be a non-empty string or None.")

    if self.route_mode == "threshold":
      if self.tau_r is None:
        raise ValueError("Threshold routing requires tau_r.")
      object.__setattr__(self, "tau_r", _probability(self.tau_r, "tau_r"))
      if self.tau_r <= 0.0 or self.tau_r >= 1.0:
        raise ValueError("tau_r must be strictly between 0 and 1.")
      if self.reference_split not in ("dev", "val"):
        raise ValueError("Threshold policy must record a dev or val reference split.")
      if self.reference_source_sha256 is None:
        raise ValueError("Threshold policy must record its reference score-source SHA-256.")
      _require_sha256(self.reference_source_sha256, "reference_source_sha256")
      if (
        self.target_utilization is None
        or isinstance(self.target_utilization, bool)
        or not isinstance(self.target_utilization, (int, float))
        or not math.isfinite(float(self.target_utilization))
        or not 0.0 < self.target_utilization < 1.0
      ):
        raise ValueError("Threshold policy must record target_utilization in (0, 1).")
    elif self.route_mode in ("none", "all"):
      if self.tau_r is not None:
        raise ValueError("route_mode none/all must not use a numeric tau_r sentinel.")
      if (
        self.reference_split is not None
        or self.reference_source_sha256 is not None
        or self.target_utilization is not None
      ):
        raise ValueError("route_mode none/all do not have fitted reference thresholds.")
    else:
      raise ValueError("route_mode must be threshold, none, or all.")

  @property
  def policy_id(self) -> str:
    """Return a content-derived identity that changes whenever policy changes."""
    canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
    return "prob-cascade-v1-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

  def to_dict(self) -> dict[str, Any]:
    """Return a detached serialization mapping including the immutable ID."""
    return {**asdict(self), "policy_id": self.policy_id}

  @classmethod
  def from_dict(cls, raw: dict[str, Any]) -> RoutingPolicy:
    """Validate a JSON/YAML mapping and verify any serialized policy ID."""
    unknown = set(raw) - _POLICY_FIELDS - {"policy_id"}
    if unknown:
      raise ValueError(f"Unknown routing-policy fields: {sorted(unknown)}")
    payload = {key: value for key, value in raw.items() if key != "policy_id"}
    policy = cls(**cast(Any, payload))
    stored_id = raw.get("policy_id")
    if stored_id is not None and stored_id != policy.policy_id:
      raise ValueError("Serialized policy_id does not match the policy contents.")
    return policy


@dataclass(frozen=True)
class StageAScore:
  """Stage A score with the metadata needed to reject incompatible inputs."""

  probability: float | None
  model_scope: str
  dataset_id: str
  outcome_horizon_days: int
  model_hash: str
  feature_schema_hash: str
  input_sha256: str
  subject_id: int | None = None
  study_id: int | None = None


@dataclass(frozen=True)
class StageBScore:
  """Stage B score keyed to the same ECG input and frozen model provenance."""

  probability: float
  model_scope: str
  dataset_id: str
  outcome_horizon_days: int
  model_hash: str
  feature_schema_hash: str
  input_sha256: str
  subject_id: int | None = None
  study_id: int | None = None


@dataclass(frozen=True)
class RoutingDecision:
  """Explicit route result; unavailable inputs are not treated as low risk."""

  status: RouteStatus
  route_to_b: bool | None
  stage_a_probability: float | None
  input_sha256: str | None
  policy_id: str
  reason: str | None = None


@dataclass(frozen=True)
class ThresholdFit:
  """Reference-only numerical cutoff and realized utilization under inclusive ties."""

  target_utilization: float
  tau_r: float
  reference_split: ReferenceSplit
  reference_source_sha256: str
  n_reference: int
  n_routed_reference: int
  realized_reference_utilization: float
  n_tied_at_threshold: int
  quantile_method: Literal["linear"] = "linear"
  inclusive: Literal[True] = True


@dataclass(frozen=True)
class ReplayRow:
  """Policy output for one ECG, excluding B scores on non-routed assignments.

  ``stage_b_computed`` indicates a routed B score was supplied/consumed in this
  replay. It is not runtime invocation or resource telemetry.
  """

  policy_id: str
  input_sha256: str | None
  route_status: RouteStatus
  route_to_b: bool | None
  stage_a_probability: float | None
  stage_b_computed: bool
  stage_b_probability: float | None
  raw_selective_mixture_probability: float | None
  stage_b_status: Literal["not_requested", "computed", "unavailable"]
  probability_status: Literal["diagnostic_only_not_validated"] = "diagnostic_only_not_validated"
  subject_id: int | None = None
  study_id: int | None = None


@dataclass(frozen=True)
class CounterfactualBReference:
  """Separate audit-only B score record for retrospective counterfactual analysis."""

  policy_id: str
  input_sha256: str
  stage_b_probability: float
  model_hash_b: str
  feature_schema_hash_b: str
  dataset_id: str
  subject_id: int | None = None
  study_id: int | None = None


@dataclass(frozen=True)
class ReplayBatch:
  """Patient-policy assignments plus an optional, separate full-B audit table."""

  rows: tuple[ReplayRow, ...]
  counterfactual_b_reference: tuple[CounterfactualBReference, ...] | None


@dataclass(frozen=True)
class LegacyComparatorResult:
  """Read-only result from a historical cohort-relative cascade comparator."""

  policy_name: Literal["mixed_reference", "rank_normalized"]
  route_mask: tuple[bool, ...]
  blended_scores: tuple[float, ...]


def validate_stage_a(score: StageAScore, policy: RoutingPolicy) -> float:
  """Validate score range, model/scope/horizon/schema and input identity."""
  probability = _probability(score.probability, "Stage A score")
  _require_sha256(score.input_sha256, "input_sha256")
  if (
    isinstance(score.outcome_horizon_days, bool)
    or not isinstance(score.outcome_horizon_days, int)
    or score.outcome_horizon_days <= 0
  ):
    raise ValueError("Stage A outcome horizon must be a positive integer.")
  if score.model_scope != policy.model_scope:
    raise ValueError("Stage A model scope does not match routing policy.")
  if score.dataset_id != policy.reference_dataset:
    raise ValueError("Stage A dataset does not match the policy reference dataset.")
  if score.outcome_horizon_days != policy.outcome_horizon_days:
    raise ValueError("Stage A outcome horizon does not match routing policy.")
  if score.model_hash != policy.model_hash_a:
    raise ValueError("Stage A model hash does not match routing policy.")
  if score.feature_schema_hash != policy.feature_schema_hash_a:
    raise ValueError("Stage A feature schema hash does not match routing policy.")
  return probability


def validate_stage_b_reference(score: StageBScore, policy: RoutingPolicy) -> float:
  """Validate B score range and frozen provenance independent of routing."""
  probability = _probability(score.probability, "Stage B score")
  _require_sha256(score.input_sha256, "Stage B input_sha256")
  if (
    isinstance(score.outcome_horizon_days, bool)
    or not isinstance(score.outcome_horizon_days, int)
    or score.outcome_horizon_days <= 0
  ):
    raise ValueError("Stage B outcome horizon must be a positive integer.")
  if score.model_scope != policy.model_scope:
    raise ValueError("Stage B model scope does not match routing policy.")
  if score.dataset_id != policy.reference_dataset:
    raise ValueError("Stage B dataset does not match the policy reference dataset.")
  if score.outcome_horizon_days != policy.outcome_horizon_days:
    raise ValueError("Stage B outcome horizon does not match routing policy.")
  if score.model_hash != policy.model_hash_b:
    raise ValueError("Stage B model hash does not match routing policy.")
  if score.feature_schema_hash != policy.feature_schema_hash_b:
    raise ValueError("Stage B feature schema hash does not match routing policy.")
  return probability


def validate_stage_b(score: StageBScore, stage_a: StageAScore, policy: RoutingPolicy) -> float:
  """Validate B provenance and ensure it describes the same ECG as Stage A."""
  probability = validate_stage_b_reference(score, policy)
  if score.input_sha256 != stage_a.input_sha256:
    raise ValueError("Stage A and Stage B input hashes do not match.")
  if score.subject_id != stage_a.subject_id or score.study_id != stage_a.study_id:
    raise ValueError("Stage A and Stage B patient/ECG keys do not match.")
  return probability


def route(score_a: StageAScore, policy: RoutingPolicy) -> bool:
  """Return whether this one patient is routed to B under a frozen policy.

  Invalid/missing score metadata raises ``ValueError``. The production boundary
  should use :func:`decide_route` to turn such failures into an explicit
  ``unable_to_route`` state instead of interpreting them as low risk.
  """
  probability = validate_stage_a(score_a, policy)
  if policy.route_mode == "none":
    return False
  if policy.route_mode == "all":
    return True
  assert policy.tau_r is not None
  if policy.inclusive:
    return probability >= policy.tau_r
  return probability > policy.tau_r


def route_many(scores_a: Sequence[StageAScore], policy: RoutingPolicy) -> tuple[bool, ...]:
  """Apply exactly the single-case route semantics without batch ranking."""
  return tuple(route(score_a, policy) for score_a in scores_a)


def route_none(policy: RoutingPolicy) -> RoutingPolicy:
  """Return an explicit no-B comparator without a numeric threshold sentinel."""
  return RoutingPolicy(
    version=policy.version,
    model_scope=policy.model_scope,
    reference_dataset=policy.reference_dataset,
    outcome_horizon_days=policy.outcome_horizon_days,
    route_mode="none",
    tau_r=None,
    inclusive=policy.inclusive,
    model_hash_a=policy.model_hash_a,
    model_hash_b=policy.model_hash_b,
    feature_schema_hash_a=policy.feature_schema_hash_a,
    feature_schema_hash_b=policy.feature_schema_hash_b,
    calibration_id=policy.calibration_id,
    missing_score_policy=policy.missing_score_policy,
  )


def route_all(policy: RoutingPolicy) -> RoutingPolicy:
  """Return an explicit all-B comparator without a numeric threshold sentinel."""
  return RoutingPolicy(
    version=policy.version,
    model_scope=policy.model_scope,
    reference_dataset=policy.reference_dataset,
    outcome_horizon_days=policy.outcome_horizon_days,
    route_mode="all",
    tau_r=None,
    inclusive=policy.inclusive,
    model_hash_a=policy.model_hash_a,
    model_hash_b=policy.model_hash_b,
    feature_schema_hash_a=policy.feature_schema_hash_a,
    feature_schema_hash_b=policy.feature_schema_hash_b,
    calibration_id=policy.calibration_id,
    missing_score_policy=policy.missing_score_policy,
  )


def legacy_top_u_route(scores_a: Sequence[float], utilization: float) -> tuple[bool, ...]:
  """Read-only adapter matching the historical stable cohort-relative top-u rule.

  This is a retrospective comparator, not a prospective per-patient policy:
  adding patients can change existing routes, and equal scores use input row
  order just as ``cascade_policy_comparison`` does.
  """
  if isinstance(utilization, bool) or not isinstance(utilization, (int, float)):
    raise ValueError("utilization must be in [0, 1].")
  u = float(utilization)
  if not math.isfinite(u) or not 0.0 <= u <= 1.0:
    raise ValueError("utilization must be in [0, 1].")
  scores = tuple(_probability(score, "Stage A score") for score in scores_a)
  if not scores:
    return ()
  n_route = max(1, int(round(len(scores) * u)))
  selected = set(sorted(range(len(scores)), key=lambda index: -scores[index])[:n_route])
  return tuple(index in selected for index in range(len(scores)))


def legacy_dev_val_ecdf(
  scores: Iterable[float],
  reference_scores: Iterable[float],
) -> tuple[float, ...]:
  """Read-only adapter to the old dev/val ECDF blending transform.

  Returned percentiles are ranking values only, never absolute mortality risks
  or routing cutoffs.
  """
  from ecg_rep.cascade import ECDFRankCalibrator

  calibrator = ECDFRankCalibrator().fit(np.asarray(tuple(reference_scores), dtype=np.float64))
  transformed = calibrator.transform(np.asarray(tuple(scores), dtype=np.float64))
  return tuple(float(value) for value in transformed)


def legacy_mixed_reference_comparator(
  scores_a: Sequence[float],
  scores_b: Sequence[float],
  utilization: float,
) -> LegacyComparatorResult:
  """Replay the historical raw-probability mix after retrospective top-u routing."""
  a = tuple(_probability(score, "Stage A score") for score in scores_a)
  b = tuple(_probability(score, "Stage B score") for score in scores_b)
  if len(a) != len(b):
    raise ValueError("A and B legacy comparator scores must have equal lengths.")
  mask = legacy_top_u_route(a, utilization)
  return LegacyComparatorResult(
    policy_name="mixed_reference",
    route_mask=mask,
    blended_scores=tuple(b_score if routed else a_score for a_score, b_score, routed in zip(a, b, mask, strict=True)),
  )


def legacy_rank_normalized_comparator(
  scores_a: Sequence[float],
  scores_b: Sequence[float],
  reference_scores_a: Sequence[float],
  reference_scores_b: Sequence[float],
  utilization: float,
) -> LegacyComparatorResult:
  """Replay the old top-u plus dev/val ECDF score-blending comparator."""
  a = tuple(_probability(score, "Stage A score") for score in scores_a)
  b = tuple(_probability(score, "Stage B score") for score in scores_b)
  if len(a) != len(b):
    raise ValueError("A and B legacy comparator scores must have equal lengths.")
  mask = legacy_top_u_route(a, utilization)
  rank_a = legacy_dev_val_ecdf(a, reference_scores_a)
  rank_b = legacy_dev_val_ecdf(b, reference_scores_b)
  return LegacyComparatorResult(
    policy_name="rank_normalized",
    route_mask=mask,
    blended_scores=tuple(b_score if routed else a_score for a_score, b_score, routed in zip(rank_a, rank_b, mask, strict=True)),
  )


def decide_route(score_a: StageAScore | None, policy: RoutingPolicy) -> RoutingDecision:
  """Return an explicit unavailable state for absent or invalid Stage A input."""
  if score_a is None:
    return RoutingDecision("unable_to_route", None, None, None, policy.policy_id, "missing_stage_a_score")
  try:
    probability = validate_stage_a(score_a, policy)
    selected = route(score_a, policy)
  except (TypeError, ValueError) as exc:
    return RoutingDecision("unable_to_route", None, None, score_a.input_sha256, policy.policy_id, str(exc))
  return RoutingDecision(
    "route_to_b" if selected else "retain_a",
    selected,
    probability,
    score_a.input_sha256,
    policy.policy_id,
  )


def fit_reference_cutoff(
  scores_a: Sequence[float],
  target_utilization: float,
  reference_split: str,
  reference_source_sha256: str | None = None,
) -> ThresholdFit:
  """Fit an absolute A-risk cutoff from one dev/val score distribution only.

  The empirical ``linear`` quantile at ``1 - target_utilization`` is frozen;
  actual reference utilization uses the same inclusive ``score >= tau_r`` rule
  used at inference. Ties can therefore make realized use exceed the target.
  Test rows are rejected by name before any score transformation.
  """
  if reference_split not in ("dev", "val"):
    raise ValueError("Routing cutoffs may be fitted on dev or val scores, never test.")
  if isinstance(target_utilization, bool) or not isinstance(target_utilization, (int, float)):
    raise ValueError("target_utilization must be in (0, 1).")
  target = float(target_utilization)
  if not math.isfinite(target) or not 0.0 < target < 1.0:
    raise ValueError("target_utilization must be in (0, 1).")
  scores = tuple(_probability(score, "Stage A reference score") for score in scores_a)
  if not scores:
    raise ValueError("At least one Stage A reference score is required.")
  source_sha256 = (
    reference_source_sha256
    if reference_source_sha256 is not None
    else hashlib.sha256(json.dumps(sorted(scores), separators=(",", ":")).encode("utf-8")).hexdigest()
  )
  _require_sha256(source_sha256, "reference_source_sha256")

  ordered = sorted(scores)
  position = (len(ordered) - 1) * (1.0 - target)
  lower = math.floor(position)
  upper = math.ceil(position)
  weight = position - lower
  threshold = ordered[lower] + (ordered[upper] - ordered[lower]) * weight
  if not 0.0 < threshold < 1.0:
    raise ValueError("Fitted tau_r is outside (0, 1); inspect scores and use a governed policy.")
  n_routed = sum(score >= threshold for score in scores)
  tied = sum(score == threshold for score in scores)
  return ThresholdFit(
    target_utilization=target,
    tau_r=threshold,
    reference_split=cast(ReferenceSplit, reference_split),
    reference_source_sha256=source_sha256,
    n_reference=len(scores),
    n_routed_reference=n_routed,
    realized_reference_utilization=n_routed / len(scores),
    n_tied_at_threshold=tied,
  )


def replay_scores(
  scores_a: Sequence[StageAScore | None],
  policy: RoutingPolicy,
  scores_b: Sequence[StageBScore | None] | None = None,
  *,
  allow_precomputed_b_reference: bool = False,
) -> ReplayBatch:
  """Replay a frozen policy without leaking non-routed B scores into its output.

  A full-cohort precomputed B table is accepted only with the explicit audit
  flag. Routed B scores appear in the assignment rows; all supplied B scores
  are separately returned as an audit reference, never copied into a
  non-routed assignment. ``stage_b_computed`` reports score availability/use
  in this replay, not on-demand invocation or compute cost.
  """
  if scores_b is not None and not allow_precomputed_b_reference:
    raise ValueError("Precomputed B scores require allow_precomputed_b_reference=True.")
  if scores_b is not None and len(scores_a) != len(scores_b):
    raise ValueError("Stage A and Stage B score rows must have equal lengths.")

  b_rows = scores_b if scores_b is not None else (None,) * len(scores_a)
  rows: list[ReplayRow] = []
  b_reference: list[CounterfactualBReference] = []
  for score_a, score_b in zip(scores_a, b_rows, strict=True):
    decision = decide_route(score_a, policy)
    if decision.status == "unable_to_route":
      if score_b is not None:
        probability_b = validate_stage_b_reference(score_b, policy)
        if score_a is not None:
          if score_a.input_sha256 != score_b.input_sha256:
            raise ValueError("Stage A and Stage B input hashes do not match.")
          if score_a.subject_id != score_b.subject_id or score_a.study_id != score_b.study_id:
            raise ValueError("Stage A and Stage B patient/ECG keys do not match.")
        b_reference.append(CounterfactualBReference(
          policy_id=policy.policy_id,
          input_sha256=score_b.input_sha256,
          stage_b_probability=probability_b,
          model_hash_b=score_b.model_hash,
          feature_schema_hash_b=score_b.feature_schema_hash,
          dataset_id=score_b.dataset_id,
          subject_id=score_b.subject_id,
          study_id=score_b.study_id,
        ))
      rows.append(ReplayRow(
        policy_id=policy.policy_id,
        input_sha256=(
          score_a.input_sha256
          if score_a is not None and _is_sha256(score_a.input_sha256)
          else None
        ),
        route_status="unable_to_route",
        route_to_b=None,
        stage_a_probability=None,
        stage_b_computed=False,
        stage_b_probability=None,
        raw_selective_mixture_probability=None,
        stage_b_status="not_requested",
        subject_id=score_a.subject_id if score_a is not None else None,
        study_id=score_a.study_id if score_a is not None else None,
      ))
      continue

    assert score_a is not None
    probability_a = cast(float, decision.stage_a_probability)
    probability_b: float | None = None
    b_computed = False
    if score_b is not None:
      probability_b = validate_stage_b(score_b, score_a, policy)
      b_reference.append(CounterfactualBReference(
        policy_id=policy.policy_id,
        input_sha256=score_b.input_sha256,
        stage_b_probability=probability_b,
        model_hash_b=score_b.model_hash,
        feature_schema_hash_b=score_b.feature_schema_hash,
        dataset_id=score_b.dataset_id,
        subject_id=score_b.subject_id,
        study_id=score_b.study_id,
      ))

    if decision.route_to_b:
      b_computed = score_b is not None
      mixture = probability_b
      # Public row status uses the route decision; B availability is represented
      # by stage_b_computed and the absent mixture when the requested score failed.
      route_status: RouteStatus = "route_to_b"
    else:
      mixture = probability_a
      route_status = "retain_a"

    rows.append(ReplayRow(
      policy_id=policy.policy_id,
      input_sha256=score_a.input_sha256,
      route_status=route_status,
      route_to_b=decision.route_to_b,
      stage_a_probability=probability_a,
      stage_b_computed=b_computed,
      stage_b_probability=probability_b if decision.route_to_b else None,
      raw_selective_mixture_probability=mixture,
      stage_b_status=("computed" if b_computed else "unavailable")
      if decision.route_to_b
      else "not_requested",
      subject_id=score_a.subject_id,
      study_id=score_a.study_id,
    ))
  return ReplayBatch(
    rows=tuple(rows),
    counterfactual_b_reference=tuple(b_reference) if allow_precomputed_b_reference else None,
  )


def save_policy(policy: RoutingPolicy, path: Path) -> None:
  """Write a new immutable JSON/YAML policy file; never overwrite a version."""
  suffix = path.suffix.lower()
  payload = policy.to_dict()
  if suffix == ".json":
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
  elif suffix in (".yaml", ".yml"):
    text = yaml.safe_dump(payload, sort_keys=True)
  else:
    raise ValueError("Policy path must use .json, .yaml, or .yml extension.")
  with path.open("x", encoding="utf-8") as stream:
    stream.write(text)
  path.chmod(0o600)


def load_policy(path: Path) -> RoutingPolicy:
  """Load and validate an immutable policy from JSON or YAML."""
  suffix = path.suffix.lower()
  text = path.read_text(encoding="utf-8")
  if suffix == ".json":
    raw = json.loads(text)
  elif suffix in (".yaml", ".yml"):
    raw = yaml.safe_load(text)
  else:
    raise ValueError("Policy path must use .json, .yaml, or .yml extension.")
  if not isinstance(raw, dict):
    raise ValueError("Serialized routing policy must be a mapping.")
  return RoutingPolicy.from_dict(cast(dict[str, Any], raw))
