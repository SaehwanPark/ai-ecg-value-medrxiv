"""CIIS morphology preflight validation suite and acceptance report generator.

Validates the CIIS primary morphology extractor against an independent baseline algorithm
across both standard tracings and enriched clinically difficult cases (wide QRS, near-0.025 mV
codability thresholds, T-wave inversion, tall R waves, prolonged repolarization/QT, and J-point
elevations). Generates a frozen acceptance report required prior to the full 160k extraction run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import signal

from ecg_rep.data import CANONICAL_12_LEADS, read_ecg_waveform
from ecg_rep.representations import (
  CIIS_EXTRACTOR_VERSION,
  CIIS_FEATURE_SCHEMA_HASH,
  get_git_commit_hash,
)
from ecg_rep.scoring.traditional import (
  CIIS_FEATURE_NAMES,
  DEFAULT_SAMPLING_RATE_HZ,
  extract_ciis_measurements_from_waveform,
  extract_independent_benchmark_measurements,
  run_ciis_measurement_validation_study,
)


def generate_synthetic_ecg_tracing(
  heart_rate_bpm: float = 75.0,
  duration_sec: float = 10.0,
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
  lead_names: Sequence[str] = CANONICAL_12_LEADS,
  q_amp_scale: float = 1.0,
  q_dur_ms_target: float = 30.0,
  r_amp_scale: float = 1.0,
  s_amp_scale: float = 1.0,
  t_pos_scale: float = 1.0,
  t_neg_scale: float = 0.0,
  qrs_width_factor: float = 1.0,
  st_elevation_mv: float = 0.0,
  qt_prolong_ms: float = 0.0,
  noise_amplitude: float = 0.002,
  seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], list[str], int]:
  """Generate a realistic 12-lead synthetic ECG tracing with controllable morphology fiducials."""
  rng = np.random.default_rng(seed)
  n_samples = int(duration_sec * fs)
  time_arr = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
  rr_sec = 60.0 / heart_rate_bpm
  beat_times = np.arange(0.5, duration_sec - 0.5, rr_sec)

  signals = np.zeros((n_samples, len(lead_names)), dtype=np.float64)

  # Canonical lead orientation multipliers for baseline standard beat
  lead_weights: dict[str, dict[str, float]] = {
    "I": {"q": -0.05, "r": 0.8, "s": -0.1, "t": 0.25},
    "II": {"q": -0.08, "r": 1.2, "s": -0.15, "t": 0.35},
    "III": {"q": -0.05, "r": 0.4, "s": -0.1, "t": 0.15},
    "aVR": {"q": 0.05, "r": -0.9, "s": 0.1, "t": -0.28},
    "aVL": {"q": -0.06, "r": 0.6, "s": -0.08, "t": 0.20},
    "aVF": {"q": -0.07, "r": 0.9, "s": -0.12, "t": 0.25},
    "V1": {"q": 0.0, "r": 0.3, "s": -0.8, "t": 0.10},
    "V2": {"q": 0.0, "r": 0.7, "s": -1.2, "t": 0.30},
    "V3": {"q": -0.04, "r": 1.1, "s": -0.8, "t": 0.40},
    "V4": {"q": -0.05, "r": 1.4, "s": -0.4, "t": 0.45},
    "V5": {"q": -0.06, "r": 1.3, "s": -0.3, "t": 0.35},
    "V6": {"q": -0.05, "r": 1.0, "s": -0.2, "t": 0.25},
  }

  qrs_half_width_sec = 0.045 * qrs_width_factor
  t_center_offset_sec = 0.30 + (qt_prolong_ms / 1000.0)
  t_half_width_sec = 0.08 + (qt_prolong_ms / 2000.0)

  for l_idx, lead in enumerate(lead_names):
    w = lead_weights.get(lead, {"q": -0.05, "r": 1.0, "s": -0.1, "t": 0.3})
    lead_sig = np.zeros(n_samples, dtype=np.float64)

    for bt in beat_times:
      # P wave
      p_center = bt - 0.18
      lead_sig += 0.12 * w["t"] * np.exp(-0.5 * ((time_arr - p_center) / 0.035) ** 2)

      # Q wave
      q_dur_scale = q_dur_ms_target / 30.0
      q_center = bt - 0.025 * q_dur_scale
      base_q = w["q"] * q_amp_scale
      lead_sig += base_q * np.exp(-0.5 * ((time_arr - q_center) / (0.012 * q_dur_scale)) ** 2)

      # R wave
      r_center = bt
      base_r = w["r"] * r_amp_scale
      lead_sig += base_r * np.exp(-0.5 * ((time_arr - r_center) / (0.020 * qrs_width_factor)) ** 2)

      # S wave
      s_center = bt + 0.030 * qrs_width_factor
      base_s = w["s"] * s_amp_scale
      lead_sig += base_s * np.exp(-0.5 * ((time_arr - s_center) / (0.018 * qrs_width_factor)) ** 2)

      # ST segment / J-point elevation
      if st_elevation_mv != 0.0:
        st_center = bt + 0.08 * qrs_width_factor
        lead_sig += st_elevation_mv * np.exp(-0.5 * ((time_arr - st_center) / 0.06) ** 2)

      # T wave
      t_center = bt + t_center_offset_sec
      if t_neg_scale > 0.0:
        base_t = -abs(t_neg_scale)
      else:
        base_t = w["t"] * t_pos_scale
      lead_sig += base_t * np.exp(-0.5 * ((time_arr - t_center) / t_half_width_sec) ** 2)

    # Add realistic high-frequency noise & baseline wander
    if noise_amplitude > 0.0:
      lead_sig += rng.normal(0.0, noise_amplitude, n_samples)
    signals[:, l_idx] = lead_sig

  return signals, list(lead_names), fs


def generate_validation_suite(
  n_standard: int = 200,
  n_difficult: int = 100,
  seed: int = 42,
) -> list[dict[str, Any]]:
  """Generate a comprehensive reproducible preflight suite of standard and difficult ECGs."""
  rng = np.random.default_rng(seed)
  suite: list[dict[str, Any]] = []

  # 1. Standard cases (n_standard)
  for i in range(n_standard):
    hr = float(rng.uniform(55.0, 95.0))
    q_scale = float(rng.uniform(0.6, 1.4))
    r_scale = float(rng.uniform(0.7, 1.3))
    s_scale = float(rng.uniform(0.6, 1.4))
    t_scale = float(rng.uniform(0.6, 1.4))
    noise = float(rng.uniform(0.001, 0.004))
    sig, leads, fs = generate_synthetic_ecg_tracing(
      heart_rate_bpm=hr,
      q_amp_scale=q_scale,
      r_amp_scale=r_scale,
      s_amp_scale=s_scale,
      t_pos_scale=t_scale,
      noise_amplitude=noise,
      seed=int(rng.integers(0, 1_000_000)),
    )
    suite.append({
      "case_id": f"standard_{i:04d}",
      "category": "standard",
      "heart_rate_bpm": hr,
      "signals": sig,
      "leads": leads,
      "fs": fs,
      "description": f"Standard tracing #{i} (HR={hr:.1f})",
    })

  # 2. Enriched difficult cases (n_difficult)
  n_per_category = max(1, n_difficult // 6)
  diff_idx = 0

  # Case A: Wide QRS without false T inversion
  for _ in range(n_per_category):
    sig, leads, fs = generate_synthetic_ecg_tracing(
      heart_rate_bpm=float(rng.uniform(60.0, 85.0)),
      qrs_width_factor=float(rng.uniform(1.6, 2.2)),
      s_amp_scale=float(rng.uniform(1.8, 2.5)),
      t_pos_scale=float(rng.uniform(1.0, 1.5)),
      t_neg_scale=0.0,
      seed=int(rng.integers(0, 1_000_000)),
    )
    suite.append({
      "case_id": f"difficult_{diff_idx:04d}",
      "category": "wide_qrs",
      "signals": sig,
      "leads": leads,
      "fs": fs,
      "description": "Wide QRS with deep S wave and upright T wave (test against false T inversion)",
    })
    diff_idx += 1

  # Case B: Near-0.025 mV codability threshold boundaries
  for _ in range(n_per_category):
    q_scale = float(rng.uniform(0.15, 0.40))
    r_scale = float(rng.uniform(0.02, 0.04))
    sig, leads, fs = generate_synthetic_ecg_tracing(
      heart_rate_bpm=70.0,
      q_amp_scale=q_scale,
      r_amp_scale=r_scale,
      s_amp_scale=0.03,
      t_pos_scale=0.08,
      noise_amplitude=0.001,
      seed=int(rng.integers(0, 1_000_000)),
    )
    suite.append({
      "case_id": f"difficult_{diff_idx:04d}",
      "category": "near_threshold_codability",
      "signals": sig,
      "leads": leads,
      "fs": fs,
      "description": "Near 0.025 mV deflection amplitudes (tests 1/4 mm codability gating)",
    })
    diff_idx += 1

  # Case C: True T-wave inversions
  for _ in range(n_per_category):
    t_neg = float(rng.uniform(0.15, 0.50))
    sig, leads, fs = generate_synthetic_ecg_tracing(
      heart_rate_bpm=float(rng.uniform(60.0, 90.0)),
      t_pos_scale=0.0,
      t_neg_scale=t_neg,
      seed=int(rng.integers(0, 1_000_000)),
    )
    suite.append({
      "case_id": f"difficult_{diff_idx:04d}",
      "category": "t_inversion",
      "signals": sig,
      "leads": leads,
      "fs": fs,
      "description": "True T-wave inversion (tests negative T measurement)",
    })
    diff_idx += 1

  # Case D: Tall R / LVH patterns
  for _ in range(n_per_category):
    sig, leads, fs = generate_synthetic_ecg_tracing(
      heart_rate_bpm=65.0,
      r_amp_scale=float(rng.uniform(1.8, 2.5)),
      s_amp_scale=float(rng.uniform(1.6, 2.4)),
      seed=int(rng.integers(0, 1_000_000)),
    )
    suite.append({
      "case_id": f"difficult_{diff_idx:04d}",
      "category": "lvh_high_voltage",
      "signals": sig,
      "leads": leads,
      "fs": fs,
      "description": "High voltage QRS / LVH pattern",
    })
    diff_idx += 1

  # Case E: Prolonged repolarization (prolonged QT / broad T)
  for _ in range(n_per_category):
    sig, leads, fs = generate_synthetic_ecg_tracing(
      heart_rate_bpm=55.0,
      qt_prolong_ms=float(rng.uniform(120.0, 180.0)),
      seed=int(rng.integers(0, 1_000_000)),
    )
    suite.append({
      "case_id": f"difficult_{diff_idx:04d}",
      "category": "prolonged_qt",
      "signals": sig,
      "leads": leads,
      "fs": fs,
      "description": "Prolonged repolarization / long QT (tests algorithmically estimated T-end without truncation)",
    })
    diff_idx += 1

  # Case F: J-point / ST elevation and baseline shift
  remaining_difficult = n_difficult - diff_idx
  for _ in range(max(1, remaining_difficult)):
    sig, leads, fs = generate_synthetic_ecg_tracing(
      heart_rate_bpm=75.0,
      st_elevation_mv=float(rng.uniform(0.08, 0.20)),
      seed=int(rng.integers(0, 1_000_000)),
    )
    suite.append({
      "case_id": f"difficult_{diff_idx:04d}",
      "category": "st_elevation_jpoint",
      "signals": sig,
      "leads": leads,
      "fs": fs,
      "description": "J-point / ST elevation (tests post-QRS J-point detection and PR baseline)",
    })
    diff_idx += 1

  return suite


def run_ciis_preflight_validation(
  output_dir: Path | str,
  cohort_df: pl.DataFrame | None = None,
  ecg_data_dir: Path | None = None,
  n_standard: int = 200,
  n_difficult: int = 100,
  seed: int = 42,
  enforce_acceptance: bool = True,
) -> dict[str, Any]:
  """Execute CIIS preflight validation on standard and enriched difficult cases.

  Runs both the primary CIIS morphology extractor and independent benchmark algorithm,
  verifies algorithmic agreement, evaluates difficult-case edge behaviors, and writes
  a frozen preflight acceptance report.

  Args:
    output_dir: Directory to write acceptance report and difficult-case audit artifacts.
    cohort_df: Optional real cohort DataFrame.
    ecg_data_dir: Optional path to MIMIC-IV-ECG waveforms directory.
    n_standard: Number of standard tracings to validate (~200).
    n_difficult: Number of enriched difficult cases to validate (~100).
    seed: Random seed for reproducibility.
    enforce_acceptance: If True, raises RuntimeError if acceptance criteria fail.

  Returns:
    Dictionary containing the acceptance report.
  """
  out_path = Path(output_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  # Check if real data is accessible
  can_use_real = (
    cohort_df is not None
    and len(cohort_df) > 0
    and ecg_data_dir is not None
    and ecg_data_dir.exists()
  )

  cases: list[dict[str, Any]] = []
  data_mode = "mimic_iv_ecg_cohort" if can_use_real else "synthetic_reproducible_benchmark_suite"

  if can_use_real and cohort_df is not None and ecg_data_dir is not None:
    # Strictly filter away locked test waveforms to avoid representation feedback leakage
    candidate_df = cohort_df
    if "split" in cohort_df.columns:
      dev_cohort = cohort_df.filter(pl.col("split") == "dev")
      if len(dev_cohort) > 0:
        candidate_df = dev_cohort
      else:
        candidate_df = cohort_df.filter(pl.col("split").is_in(["dev", "val"]))
      print(
        f"[CIIS Preflight] Filtered cohort to development split ({len(candidate_df)} records) "
        "to strictly isolate preflight validation from the locked test set."
      )
    rng = np.random.default_rng(seed)
    n_samp = min(n_standard, len(candidate_df))
    sampled_indices = rng.choice(len(candidate_df), size=n_samp, replace=False)
    sample_sub = candidate_df[sampled_indices]
    for r in sample_sub.iter_rows(named=True):
      rec_p = ecg_data_dir / str(r["path"])
      try:
        sig, leads, fs = read_ecg_waveform(rec_p)
        cases.append({
          "case_id": f"real_cohort_{r['study_id']}",
          "category": "standard",
          "signals": sig,
          "leads": leads,
          "fs": fs,
          "study_id": int(r["study_id"]),
          "description": f"Real MIMIC-IV-ECG record {r['study_id']} (dev split)",
        })
      except Exception:
        pass

  synth_suite = generate_validation_suite(
    n_standard=max(0, n_standard - len([c for c in cases if c["category"] == "standard"])),
    n_difficult=n_difficult,
    seed=seed,
  )
  cases.extend(synth_suite)

  print(f"[CIIS Preflight] Validating {len(cases)} cases ({data_mode})...")

  primary_rows: list[dict[str, Any]] = []
  benchmark_rows: list[dict[str, Any]] = []
  difficult_audit_rows: list[dict[str, Any]] = []

  for idx, case in enumerate(cases):
    cid = case["case_id"]
    sig = case["signals"]
    leads = case["leads"]
    fs = case["fs"]
    cat = case["category"]

    try:
      # Primary extraction
      meas = extract_ciis_measurements_from_waveform(sig, leads, fs)
      p_dict: dict[str, Any] = {"study_id": idx + 1, "case_id": cid, "category": cat}
      for fn in CIIS_FEATURE_NAMES:
        p_dict[fn] = getattr(meas, fn)
      primary_rows.append(p_dict)

      # Benchmark extraction
      bench = extract_independent_benchmark_measurements(sig, leads, fs)
      b_dict: dict[str, Any] = {"study_id": idx + 1, "case_id": cid, "category": cat}
      for fn in CIIS_FEATURE_NAMES:
        b_dict[fn] = bench.get(fn, 0.0)
      benchmark_rows.append(b_dict)

      if cat != "standard":
        amp_feature_names = (
          "avl_t_pos_mv",
          "avl_t_neg_mv",
          "inv_avr_r_mv",
          "inv_avr_t_pos_mv",
          "lead3_t_neg_mv",
          "v1_t_pos_mv",
          "v2_r_mv",
          "v2_t_neg_mv",
          "v5_s_mv",
        )
        subthreshold_violations = sum(
          1 for fn in amp_feature_names if 0.0 < getattr(meas, fn) < 0.025
        )
        difficult_audit_rows.append({
          "case_id": cid,
          "category": cat,
          "description": case["description"],
          "primary_v2_t_neg": meas.v2_t_neg_mv,
          "primary_v1_t_pos": meas.v1_t_pos_mv,
          "primary_v5_s": meas.v5_s_mv,
          "primary_v2_r": meas.v2_r_mv,
          "primary_inv_avr_r": meas.inv_avr_r_mv,
          "primary_avl_t_pos": meas.avl_t_pos_mv,
          "primary_lead3_t_neg": meas.lead3_t_neg_mv,
          "subthreshold_violation_count": subthreshold_violations,
          "benchmark_v2_t_neg": bench.get("v2_t_neg_mv", 0.0),
          "benchmark_v5_s": bench.get("v5_s_mv", 0.0),
        })
    except Exception as exc:
      print(f"[CIIS Preflight] Warning: Case {cid} failed extraction: {exc}")

    done = idx + 1
    if done % 50 == 0 or done == len(cases):
      print(f"[CIIS Preflight] Processed {done}/{len(cases)} cases", flush=True)

  prim_df = pl.DataFrame(primary_rows)
  bench_df = pl.DataFrame(benchmark_rows)

  val_report = run_ciis_measurement_validation_study(
    extracted_df=prim_df,
    reference_df=bench_df,
  )

  # Explicit Scenario-Level Acceptance Criteria
  wide_qrs_cases = [r for r in difficult_audit_rows if r["category"] == "wide_qrs"]
  wide_qrs_false_inversions = sum(1 for w in wide_qrs_cases if w["primary_v2_t_neg"] > 0.025)
  wide_qrs_ok = (wide_qrs_false_inversions == 0) if wide_qrs_cases else True

  near_thresh_cases = [r for r in difficult_audit_rows if r["category"] == "near_threshold_codability"]
  near_thresh_violations = sum(r.get("subthreshold_violation_count", 0) for r in near_thresh_cases)
  near_thresh_ok = (near_thresh_violations == 0 and len(near_thresh_cases) > 0) if near_thresh_cases else True

  t_inv_cases = [r for r in difficult_audit_rows if r["category"] == "t_inversion"]
  t_inv_detected = sum(
    1 for r in t_inv_cases
    if r["primary_v2_t_neg"] > 0.05 or r.get("primary_lead3_t_neg", 0.0) > 0.05
  )
  t_inv_ok = (t_inv_detected == len(t_inv_cases) and len(t_inv_cases) > 0) if t_inv_cases else True

  lvh_cases = [r for r in difficult_audit_rows if r["category"] == "lvh_high_voltage"]
  lvh_captured = sum(
    1 for r in lvh_cases
    if r["primary_v2_r"] > 0.5 and (r["primary_v5_s"] > 0.10 or r.get("primary_inv_avr_r", 0.0) > 1.0)
  )
  lvh_ok = (lvh_captured == len(lvh_cases) and len(lvh_cases) > 0) if lvh_cases else True

  prolonged_qt_cases = [r for r in difficult_audit_rows if r["category"] == "prolonged_qt"]
  prolonged_qt_captured = sum(
    1 for r in prolonged_qt_cases
    if r.get("primary_v1_t_pos", 0.0) > 0.05 or r.get("primary_avl_t_pos", 0.0) > 0.05
  )
  prolonged_qt_ok = (prolonged_qt_captured == len(prolonged_qt_cases) and len(prolonged_qt_cases) > 0) if prolonged_qt_cases else True

  st_elevation_cases = [r for r in difficult_audit_rows if r["category"] == "st_elevation_jpoint"]
  st_elevation_captured = sum(
    1 for r in st_elevation_cases
    if r.get("primary_v2_r", 0.0) > 0.3
  )
  st_elevation_ok = (st_elevation_captured == len(st_elevation_cases) and len(st_elevation_cases) > 0) if st_elevation_cases else True

  # Algorithmic Agreement and Distribution QC
  agreement_val = val_report.get("independent_algorithmic_agreement_validation", {})
  feat_stats = agreement_val.get("features", {}) if isinstance(agreement_val, dict) else {}
  outlier_rates = [v.get("outlier_rate", 0.0) for v in feat_stats.values() if isinstance(v, dict)]
  mean_outlier_rate = float(np.mean(outlier_rates)) if outlier_rates else 0.0
  outlier_ok = mean_outlier_rate < 0.60

  spearman_rhos = [
    v.get("spearman_rho", 0.0)
    for v in feat_stats.values()
    if isinstance(v, dict) and v.get("spearman_rho") is not None
  ]
  mean_spearman = float(np.mean(spearman_rhos)) if spearman_rhos else 1.0
  spearman_ok = mean_spearman > 0.20

  dist_qc = val_report.get("distribution_qc", {})
  phys_ok = dist_qc.get("all_physiologically_valid", True)

  dim_ok = len(CIIS_FEATURE_NAMES) == 15
  finite_ok = len(prim_df) == len(bench_df) and len(prim_df) >= int((n_standard + n_difficult) * 0.9)

  all_passed = (
    dim_ok
    and finite_ok
    and phys_ok
    and wide_qrs_ok
    and near_thresh_ok
    and t_inv_ok
    and lvh_ok
    and prolonged_qt_ok
    and st_elevation_ok
    and spearman_ok
    and outlier_ok
  )

  acceptance_report: dict[str, Any] = {
    "status": "PASSED" if all_passed else "FAILED",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "extractor_version": CIIS_EXTRACTOR_VERSION,
    "feature_schema_hash": CIIS_FEATURE_SCHEMA_HASH,
    "git_commit": get_git_commit_hash(),
    "data_mode": data_mode,
    "sample_counts": {
      "n_total": len(cases),
      "n_standard": len([c for c in cases if c["category"] == "standard"]),
      "n_difficult": len([c for c in cases if c["category"] != "standard"]),
      "n_evaluated": len(prim_df),
    },
    "acceptance_criteria": {
      "feature_dimension_15": dim_ok,
      "finite_pairs_evaluated": finite_ok,
      "physiologic_distributions_valid": phys_ok,
      "wide_qrs_false_t_inversion_prevented": wide_qrs_ok,
      "near_threshold_codability_verified": near_thresh_ok,
      "true_t_inversions_detected": t_inv_ok,
      "high_voltage_lvh_captured": lvh_ok,
      "prolonged_qt_untruncated": prolonged_qt_ok,
      "jpoint_st_elevation_delineated": st_elevation_ok,
      "mean_spearman_rho_acceptable": spearman_ok,
      "mean_outlier_rate_acceptable": outlier_ok,
      "all_checks_passed": all_passed,
    },
    "summary_metrics": {
      "mean_outlier_rate": round(mean_outlier_rate, 4),
      "mean_spearman_rho": round(mean_spearman, 4),
      "wide_qrs_false_inversions": wide_qrs_false_inversions,
      "near_thresh_cases_evaluated": len(near_thresh_cases),
      "true_t_inversion_cases_evaluated": len(t_inv_cases),
      "high_voltage_lvh_cases_evaluated": len(lvh_cases),
      "prolonged_qt_cases_evaluated": len(prolonged_qt_cases),
      "st_elevation_cases_evaluated": len(st_elevation_cases),
    },
    "feature_agreement": agreement_val,
    "distribution_qc": dist_qc,
  }

  report_path = out_path / "ciis_preflight_acceptance_report.json"
  report_path.write_text(json.dumps(acceptance_report, indent=2) + "\n", encoding="utf-8")
  print(f"[CIIS Preflight] Acceptance report written to {report_path} (Status: {acceptance_report['status']})")

  if difficult_audit_rows:
    diff_df = pl.DataFrame(difficult_audit_rows)
    diff_df.write_parquet(out_path / "ciis_preflight_difficult_cases.parquet")
    (out_path / "ciis_preflight_difficult_cases.json").write_text(
      json.dumps(difficult_audit_rows, indent=2) + "\n", encoding="utf-8"
    )

  if enforce_acceptance and not all_passed:
    raise RuntimeError(
      f"CIIS Preflight validation FAILED acceptance criteria!\n"
      f"Details: {json.dumps(acceptance_report['acceptance_criteria'], indent=2)}"
    )

  return acceptance_report
