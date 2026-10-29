"""Traditional ECG risk scoring models, focusing on the Cardiac Infarction/Injury Score (CIIS).

Authoritative references:
- Rautaharju PM, Warren JW, Jain U, Wolf HK, Nielsen CL. Cardiac infarction injury score:
  an electrocardiographic coding scheme for ischemic heart disease. Circulation. 1981;64(2):249-256.
- Dekker JM, Schouten EG, Pool J, Kok FJ. Cardiac Infarction Injury Score predicts
  cardiovascular mortality in apparently healthy men and women. Br Heart J. 1994;72(1):39-44.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast
import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.signal as signal
from scipy.stats import pearsonr, spearmanr

from ecg_rep.data import CANONICAL_12_LEADS, DEFAULT_SAMPLING_RATE_HZ


class CIISCategory(str, Enum):
  """Published clinical risk categories for CIIS."""

  NORMAL = "normal"  # CIIS < 10
  BORDERLINE = "borderline"  # 10 <= CIIS < 15
  POSSIBLE_INJURY = "possible_injury"  # 15 <= CIIS < 20
  PROBABLE_INFARCTION = "probable_infarction"  # CIIS >= 20


@dataclass(frozen=True)
class LeadFeatures:
  """Morphologic measurements extracted for a single ECG lead."""

  q_duration_ms: float
  q_amplitude_mv: float
  r_amplitude_mv: float
  s_amplitude_mv: float
  t_pos_amplitude_mv: float
  t_neg_amplitude_mv: float
  qr_ratio: float


@dataclass(frozen=True)
class CIISMeasurements:
  """Complete set of measurements required to compute CIIS.

  These fields are the conventional ECG representation ``Z_A`` for the fair
  ridge-logistic probe (CIIS-feature linear probe).

  Authoritative mapping from Rautaharju et al. Circulation 1981;64(2):249-256 Table 3
  and ARIC Protocol 5 Appendix L Table 3 (NCBI phd004495.1):
    Item 1:  Lead aVL Q duration           -> avl_q_duration_ms
    Item 2:  Lead aVL T amplitude          -> avl_t_pos_mv, avl_t_neg_mv
    Item 3:  Lead -aVR R amplitude         -> inv_avr_r_mv
    Item 4:  Lead -aVR T amplitude         -> inv_avr_t_pos_mv
    Item 5:  Lead II / aVF Q/R ratio       -> lead2_qr_ratio, avf_qr_ratio
    Item 6:  Lead III or -aVL Q duration   -> lead3_q_duration_ms, inv_avl_q_duration_ms
    Item 7:  Lead III T amplitude          -> lead3_t_neg_mv
    Item 8:  Lead V1 T amplitude           -> v1_t_pos_mv
    Item 9:  Lead V2 R amplitude           -> v2_r_mv
    Item 10: Lead V2 T amplitude           -> v2_t_neg_mv
    Item 11: Lead V3 Q/R ratio             -> v3_qr_ratio
    Item 12: Lead V5 S amplitude           -> v5_s_mv

  Note on literature terminology ('15-feature classifier vs 12-item visual checklist'):
    The original CIIS publication (Rautaharju et al. Circulation 1981;64(2):249-256)
    developed an optimal 15-feature computer classifier combining eight binary, three
    ternary, and four continuous features (Table 2). The authors subsequently simplified that into
    a practical 12-item visual scoring checklist (Table 3). Evaluating all 12 items
    of this visual checklist requires 15 continuous morphologic measurements across
    the standard 12 leads (e.g. Lead II and aVF Q/R ratios; aVL, -aVL, and Lead III Q durations;
    upright vs inverted T wave amplitudes in aVL, -aVR, III, V1, V2; V5 S amplitude). In this study,
    Representation A (Z_A in R^15) represents these 15 underlying continuous morphologic measurements,
    enabling the downstream ridge-logistic probe to learn optimal mortality weights
    directly from the continuous waveform measurements rather than discrete point bins.
  """

  avl_q_duration_ms: float
  avl_t_pos_mv: float
  avl_t_neg_mv: float
  inv_avr_r_mv: float  # R amplitude in -aVR lead
  inv_avr_t_pos_mv: float  # Positive T amplitude in -aVR lead
  lead2_qr_ratio: float
  avf_qr_ratio: float
  lead3_q_duration_ms: float
  inv_avl_q_duration_ms: float  # Q duration in -aVL lead (initial R duration in +aVL)
  lead3_t_neg_mv: float
  v1_t_pos_mv: float
  v2_r_mv: float
  v2_t_neg_mv: float
  v3_qr_ratio: float
  v5_s_mv: float


CIIS_FEATURE_NAMES: tuple[str, ...] = (
  "avl_q_duration_ms",
  "avl_t_pos_mv",
  "avl_t_neg_mv",
  "inv_avr_r_mv",
  "inv_avr_t_pos_mv",
  "lead2_qr_ratio",
  "avf_qr_ratio",
  "lead3_q_duration_ms",
  "inv_avl_q_duration_ms",
  "lead3_t_neg_mv",
  "v1_t_pos_mv",
  "v2_r_mv",
  "v2_t_neg_mv",
  "v3_qr_ratio",
  "v5_s_mv",
)

CIIS_PHYSIOLOGIC_BOUNDS: dict[str, tuple[float, float]] = {
  "avl_q_duration_ms": (0.0, 150.0),
  "avl_t_pos_mv": (0.0, 3.0),
  "avl_t_neg_mv": (0.0, 3.0),
  "inv_avr_r_mv": (0.0, 5.0),
  "inv_avr_t_pos_mv": (0.0, 3.0),
  "lead2_qr_ratio": (0.0, 10.0),
  "avf_qr_ratio": (0.0, 10.0),
  "lead3_q_duration_ms": (0.0, 150.0),
  "inv_avl_q_duration_ms": (0.0, 150.0),
  "lead3_t_neg_mv": (0.0, 3.0),
  "v1_t_pos_mv": (0.0, 3.0),
  "v2_r_mv": (0.0, 6.0),
  "v2_t_neg_mv": (0.0, 3.0),
  "v3_qr_ratio": (0.0, 10.0),
  "v5_s_mv": (0.0, 5.0),
}


def measurements_to_feature_vector(measurements: CIISMeasurements) -> npt.NDArray[np.float64]:
  """Convert CIIS measurements into a fixed-order float feature vector ``Z_A``."""
  return np.asarray(
    [getattr(measurements, name) for name in CIIS_FEATURE_NAMES],
    dtype=np.float64,
  )


@dataclass(frozen=True)
class CIISItemScores:
  """Itemized points contributing to the total CIIS."""

  item1_avl_q_duration: float
  item2_avl_t_amplitude: float
  item3_avr_r_amplitude: float
  item4_avr_t_amplitude: float
  item5_lead2_avf_qr_ratio: float
  item6_lead3_avl_q_duration: float
  item7_lead3_t_amplitude: float
  item8_v1_t_amplitude: float
  item9_v2_r_amplitude: float
  item10_v2_t_amplitude: float
  item11_v3_qr_ratio: float
  item12_v5_s_amplitude: float


@dataclass(frozen=True)
class CIISScoreResult:
  """Complete score output including continuous score, category, and diagnostic metadata."""

  is_valid: bool
  total_score: float | None = None
  category: CIISCategory | None = None
  item_scores: CIISItemScores | None = None
  measurements: CIISMeasurements | None = None
  error_message: str | None = None


# -----------------------------------------------------------------------------
# Pure Item Scoring Functions
# Calibration standard: 1 mV = 10 mm (0.1 mV = 1 mm; 0.025 mV = 0.25 mm)
# -----------------------------------------------------------------------------


def score_item1_avl_q_duration(q_dur_ms: float) -> float:
  """Score Item 1: Lead aVL Q-wave duration."""
  if q_dur_ms <= 0.0:
    return 5.0
  if q_dur_ms <= 15.0:
    return 1.0
  if q_dur_ms <= 25.0:
    return 3.0
  if q_dur_ms <= 35.0:
    return 9.0
  if q_dur_ms <= 45.0:
    return 10.0
  return 12.0


def score_item2_avl_t_amplitude(t_pos_mv: float, t_neg_mv: float) -> float:
  """Score Item 2: Lead aVL T-wave amplitude."""
  if t_neg_mv > 0.025:
    # Inverted T wave: 3 points + 2.0 points per mm (20 points per mV)
    t_neg_mm = t_neg_mv * 10.0
    return 3.0 + 2.0 * t_neg_mm
  if t_pos_mv <= 0.05:  # <= 0.5 mm flat or low positive
    return 3.0
  if t_pos_mv >= 0.30:  # >= 3.0 mm tall positive
    return 3.0
  return 0.0


def score_item3_avr_r_amplitude(inv_avr_r_mv: float) -> float:
  """Score Item 3: Lead -aVR R-wave amplitude (subtract 1 point per mm of R)."""
  r_mm = inv_avr_r_mv * 10.0
  return -1.0 * r_mm


def score_item4_avr_t_amplitude(inv_avr_t_pos_mv: float) -> float:
  """Score Item 4: Lead -aVR positive T-wave amplitude."""
  t_mm = inv_avr_t_pos_mv * 10.0
  if t_mm <= 0.5:
    return 6.0
  if t_mm <= 1.5:
    return 3.0
  if t_mm <= 2.5:
    return 0.0
  if t_mm <= 3.5:
    return -2.0
  if t_mm <= 4.5:
    return -5.0
  return -5.0 - 2.0 * (t_mm - 4.0)


def score_item5_lead2_avf_qr_ratio(lead2_qr: float, avf_qr: float) -> float:
  """Score Item 5: Largest Q/R ratio in Lead II or aVF (threshold >= 1/20 = 0.05)."""
  if max(lead2_qr, avf_qr) >= 0.05:
    return 12.0
  return 0.0


def score_item6_lead3_avl_q_duration(lead3_q_dur_ms: float, inv_avl_q_dur_ms: float) -> float:
  """Score Item 6: Duration of Q in Lead III or -aVL >= 40 ms."""
  if max(lead3_q_dur_ms, inv_avl_q_dur_ms) >= 40.0:
    return 5.0
  return 0.0


def score_item7_lead3_t_amplitude(lead3_t_neg_mv: float) -> float:
  """Score Item 7: Amplitude of inverted T in Lead III > 1.0 mm (0.10 mV)."""
  if lead3_t_neg_mv > 0.10:
    return 5.0
  return 0.0


def score_item8_v1_t_amplitude(v1_t_pos_mv: float) -> float:
  """Score Item 8: Amplitude of positive T in Lead V1 > 2.0 mm (0.2 mV)."""
  if v1_t_pos_mv > 0.20:
    return 5.0
  return 0.0


def score_item9_v2_r_amplitude(v2_r_mv: float) -> float:
  """Score Item 9: Amplitude of R in Lead V2 < 3.0 mm (0.3 mV) or >= 14.0 mm (1.4 mV)."""
  if v2_r_mv < 0.30 or v2_r_mv >= 1.40:
    return 5.0
  return 0.0


def score_item10_v2_t_amplitude(v2_t_neg_mv: float) -> float:
  """Score Item 10: Amplitude of inverted T in Lead V2 >= 0.25 mm (0.025 mV)."""
  if v2_t_neg_mv >= 0.025:
    return 5.0
  return 0.0


def score_item11_v3_qr_ratio(v3_qr: float) -> float:
  """Score Item 11: Largest Q/R ratio in Lead V3 > 1/20 (0.05)."""
  if v3_qr > 0.05:
    return 9.0
  return 0.0


def score_item12_v5_s_amplitude(v5_s_mv: float) -> float:
  """Score Item 12: Amplitude of S in Lead V5 < 2.0 mm (0.2 mV)."""
  if v5_s_mv < 0.20:
    return 5.0
  return 0.0


def classify_ciis_category(ciis_score: float) -> CIISCategory:
  """Classify continuous CIIS into published clinical risk categories.

  Args:
    ciis_score: Continuous CIIS total score.

  Returns:
    CIISCategory enum member.
  """
  if ciis_score < 10.0:
    return CIISCategory.NORMAL
  if ciis_score < 15.0:
    return CIISCategory.BORDERLINE
  if ciis_score < 20.0:
    return CIISCategory.POSSIBLE_INJURY
  return CIISCategory.PROBABLE_INFARCTION


def compute_ciis_from_measurements(measurements: CIISMeasurements) -> CIISScoreResult:
  """Compute CIIS score and itemized breakdown from explicit ECG measurements.

  Args:
    measurements: CIISMeasurements dataclass.

  Returns:
    CIISScoreResult with continuous score, category, and item breakdown.
  """
  item1 = score_item1_avl_q_duration(measurements.avl_q_duration_ms)
  item2 = score_item2_avl_t_amplitude(measurements.avl_t_pos_mv, measurements.avl_t_neg_mv)
  item3 = score_item3_avr_r_amplitude(measurements.inv_avr_r_mv)
  item4 = score_item4_avr_t_amplitude(measurements.inv_avr_t_pos_mv)
  item5 = score_item5_lead2_avf_qr_ratio(measurements.lead2_qr_ratio, measurements.avf_qr_ratio)
  item6 = score_item6_lead3_avl_q_duration(
    measurements.lead3_q_duration_ms, measurements.inv_avl_q_duration_ms
  )
  item7 = score_item7_lead3_t_amplitude(measurements.lead3_t_neg_mv)
  item8 = score_item8_v1_t_amplitude(measurements.v1_t_pos_mv)
  item9 = score_item9_v2_r_amplitude(measurements.v2_r_mv)
  item10 = score_item10_v2_t_amplitude(measurements.v2_t_neg_mv)
  item11 = score_item11_v3_qr_ratio(measurements.v3_qr_ratio)
  item12 = score_item12_v5_s_amplitude(measurements.v5_s_mv)

  item_scores = CIISItemScores(
    item1_avl_q_duration=item1,
    item2_avl_t_amplitude=item2,
    item3_avr_r_amplitude=item3,
    item4_avr_t_amplitude=item4,
    item5_lead2_avf_qr_ratio=item5,
    item6_lead3_avl_q_duration=item6,
    item7_lead3_t_amplitude=item7,
    item8_v1_t_amplitude=item8,
    item9_v2_r_amplitude=item9,
    item10_v2_t_amplitude=item10,
    item11_v3_qr_ratio=item11,
    item12_v5_s_amplitude=item12,
  )

  total = (
    item1
    + item2
    + item3
    + item4
    + item5
    + item6
    + item7
    + item8
    + item9
    + item10
    + item11
    + item12
  )
  category = classify_ciis_category(total)

  return CIISScoreResult(
    is_valid=True,
    total_score=float(total),
    category=category,
    item_scores=item_scores,
    measurements=measurements,
    error_message=None,
  )


# -----------------------------------------------------------------------------
# Deterministic Waveform Feature Extraction & Delineation
# -----------------------------------------------------------------------------


def filter_ecg_signal(
  signal_1d: npt.NDArray[np.float64],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> npt.NDArray[np.float64]:
  """Apply zero-phase high-pass filter to remove baseline drift without phase distortion.

  Args:
    signal_1d: 1D NumPy array representing single lead waveform in mV.
    fs: Sampling frequency in Hz.

  Returns:
    Filtered 1D array.
  """
  cutoff_hz = 0.67
  nyq = fs / 2.0
  b, a = cast(
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    signal.butter(2, cutoff_hz / nyq, btype="highpass", output="ba"),
  )
  filtered: npt.NDArray[np.float64] = signal.filtfilt(b, a, signal_1d)
  return filtered


def detect_qrs_peaks(
  lead_signal: npt.NDArray[np.float64],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> npt.NDArray[np.int64]:
  """Detect QRS R-peaks using an energy-derivative envelope.

  Args:
    lead_signal: 1D signal (preferably Lead II or vector magnitude).
    fs: Sampling rate in Hz.

  Returns:
    Array of integer sample indices corresponding to detected R-peaks.
  """
  if len(lead_signal) < fs:
    return np.empty(0, dtype=np.int64)

  # Bandpass 5-18 Hz for QRS energy
  nyq = fs / 2.0
  b, a = cast(
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    signal.butter(2, [5.0 / nyq, 18.0 / nyq], btype="bandpass", output="ba"),
  )
  bp_sig: npt.NDArray[np.float64] = signal.filtfilt(b, a, lead_signal)

  diff_sig = np.diff(bp_sig, prepend=bp_sig[0])
  squared = diff_sig**2
  win_len = max(1, int(0.10 * fs))  # 100 ms moving average
  integrated: npt.NDArray[np.float64] = np.convolve(
    squared, np.ones(win_len) / win_len, mode="same"
  )

  mean_val = float(np.mean(integrated))
  std_val = float(np.std(integrated))
  threshold = mean_val + 0.4 * std_val
  min_distance = int(0.28 * fs)  # 280 ms refractory window

  peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)
  if len(peaks) == 0:
    return np.empty(0, dtype=np.int64)

  # Refine peak to local extreme in raw signal
  refined: list[int] = []
  search_win = int(0.08 * fs)
  for p in peaks:
    start = max(0, int(p) - search_win)
    end = min(len(lead_signal), int(p) + search_win)
    if end > start:
      sub = lead_signal[start:end]
      local_idx = int(np.argmax(np.abs(sub)))
      refined.append(start + local_idx)

  return np.asarray(refined, dtype=np.int64)


def extract_median_beat(
  signals_12lead: npt.NDArray[np.float64],
  r_peaks: npt.NDArray[np.int64],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> tuple[npt.NDArray[np.float64], int]:
  """Extract a time-aligned representative median beat across all beats in the recording.

  Args:
    signals_12lead: 2D array [samples, 12].
    r_peaks: Array of R-peak sample indices.
    fs: Sampling rate in Hz.

  Returns:
    Tuple of (median_beat [beat_samples, 12], r_peak_index_in_beat).
  """
  pre_samples = int(0.25 * fs)  # 250 ms before R
  post_samples = int(0.45 * fs)  # 450 ms after R
  total_samples = pre_samples + post_samples

  valid_beats: list[npt.NDArray[np.float64]] = []
  n_samples = signals_12lead.shape[0]

  for r in r_peaks:
    start = int(r) - pre_samples
    end = int(r) + post_samples
    if start >= 0 and end <= n_samples:
      beat = signals_12lead[start:end, :]
      valid_beats.append(beat)

  if not valid_beats:
    raise ValueError("No valid complete beats could be extracted within recording boundaries.")

  stacked = np.stack(valid_beats, axis=0)  # [n_beats, total_samples, 12]
  median_beat: npt.NDArray[np.float64] = np.median(stacked, axis=0)
  return median_beat, pre_samples


def extract_lead_features(
  lead_signal: npt.NDArray[np.float64],
  r_idx: int,
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> LeadFeatures:
  """Extract fiducials and amplitudes from a single-lead median beat.

  Args:
    lead_signal: 1D array representing the median beat for one lead in mV.
    r_idx: Sample index of the R-peak in lead_signal.
    fs: Sampling rate in Hz.

  Returns:
    LeadFeatures dataclass.
  """
  n_samples = len(lead_signal)

  # 1. Isoelectric PR baseline estimation:
  # Search within the PR segment window [r_idx - 200 ms, r_idx - 40 ms] for the
  # lowest-variance isoelectric window (free of P-wave and QRS upstroke).
  pr_win = max(2, int(0.02 * fs))
  pr_search_start = max(0, r_idx - int(0.20 * fs))
  pr_search_end = max(pr_win + 1, r_idx - int(0.04 * fs))

  best_var = float("inf")
  baseline = float(lead_signal[0]) if len(lead_signal) > 0 else 0.0
  if pr_search_end > pr_search_start + pr_win:
    for i in range(pr_search_start, pr_search_end - pr_win):
      sub = lead_signal[i:i + pr_win]
      v = float(np.var(sub))
      if v < best_var:
        best_var = v
        baseline = float(np.median(sub))
  else:
    fallback_end = max(1, r_idx - int(0.04 * fs))
    baseline = float(np.median(lead_signal[:fallback_end]))

  zeroed = lead_signal - baseline

  # 2. QRS onset: search backwards from R peak for deflection from PR baseline
  qrs_search_start = max(0, r_idx - int(0.14 * fs))
  qrs_onset = qrs_search_start
  if r_idx > qrs_search_start:
    for i in range(r_idx - int(0.02 * fs), qrs_search_start, -1):
      if abs(zeroed[i]) < 0.025:
        qrs_onset = i
        break

  # 3. Q-wave detection (between qrs_onset and R peak)
  # Per Rautaharju 1981 Table 3 & ARIC Protocol 5 Appendix L:
  # Minimum codable deflection is 1/4 standard mm (0.025 mV = 25 uV).
  q_dur_ms = 0.0
  q_amp_mv = 0.0
  if r_idx > qrs_onset:
    q_window = zeroed[qrs_onset:r_idx]
    min_q = float(np.min(q_window))
    if min_q <= -0.025:  # Significant negative deflection (>= 0.025 mV / 25 uV per ARIC / CIIS rule)
      q_amp_mv = float(np.abs(min_q))
      min_q_idx = qrs_onset + int(np.argmin(q_window))
      q_end_idx = r_idx
      for i in range(min_q_idx, r_idx):
        if zeroed[i] >= -0.005:  # return to isoelectric baseline
          q_end_idx = i
          break
      q_dur_ms = float((q_end_idx - qrs_onset) / fs * 1000.0)

  # 4. R-wave detection (positive deflection at/around r_idx)
  # Must meet 1/4 standard mm (0.025 mV = 25 uV) codability threshold.
  r_search_start = max(0, r_idx - int(0.025 * fs))
  r_search_end = min(n_samples, r_idx + int(0.035 * fs))
  max_r = float(np.max(zeroed[r_search_start:r_search_end]))
  r_amp_mv = float(max_r) if max_r >= 0.025 else 0.0
  r_actual_idx = r_search_start + int(np.argmax(zeroed[r_search_start:r_search_end]))

  # 5. S-wave detection (negative deflection following R peak within 180 ms)
  # Per ARIC CIIS coding definitions, S is defined after a qualifying positive R deflection
  # (>= 0.025 mV) and must meet the 1/4 mm (0.025 mV) codability gate.
  s_search_start = r_actual_idx
  s_search_end = min(n_samples, r_actual_idx + int(0.18 * fs))
  s_amp_mv = 0.0
  s_trough_idx = r_actual_idx
  if r_amp_mv >= 0.025 and s_search_end > s_search_start:
    min_s = float(np.min(zeroed[s_search_start:s_search_end]))
    if min_s <= -0.025:
      s_amp_mv = float(np.abs(min_s))
      s_trough_idx = s_search_start + int(np.argmin(zeroed[s_search_start:s_search_end]))

  # 6. QRS offset (J-point) delineation
  # Search window extends to encompass wide QRS variants (up to 240 ms post R)
  j_search_end = min(n_samples, max(r_idx + int(0.24 * fs), s_trough_idx + int(0.16 * fs)))
  j_idx = s_trough_idx + int(0.04 * fs)
  if j_search_end > s_trough_idx + 1:
    post_s = zeroed[s_trough_idx:j_search_end]
    post_slopes = np.abs(np.diff(post_s))
    max_post_slope = float(np.max(post_slopes)) if len(post_slopes) > 0 else 1.0
    j_thresh = max(0.010, 0.20 * max_post_slope)
    found_j = False
    for i in range(1, len(post_slopes)):
      val = zeroed[s_trough_idx + i]
      if val >= -0.05 or (s_amp_mv > 0 and val >= -s_amp_mv * 0.15):
        if post_slopes[i] < j_thresh or val >= -0.020:
          j_idx = s_trough_idx + i
          found_j = True
          break
    if not found_j:
      best_i = int(np.argmax(post_s))
      j_idx = s_trough_idx + best_i

  # 7. Q/R ratio: computed when R is a qualifying deflection (>= 0.025 mV).
  # If R is absent (< 0.025 mV) and Q is present (QS complex), ratio represents infinity (capped at 10.0).
  # If both absent, ratio is 0.0.
  qr_ratio = (
    min(10.0, float(q_amp_mv / r_amp_mv))
    if r_amp_mv >= 0.025
    else (10.0 if q_amp_mv >= 0.025 else 0.0)
  )

  # 8. ST-T repolarization window and algorithmically estimated T-end delineation:
  # Per Rautaharju 1981 / ARIC rules: repolarization window starts at J + 80 ms,
  # and extends to the algorithmically estimated end of the T wave (T_end).
  t_start = min(n_samples - 1, j_idx + int(0.08 * fs))
  # Ensure t_start has cleared the depolarization recovery in wide QRS
  while (
    t_start < n_samples - 1
    and zeroed[t_start] < -0.025
    and (zeroed[t_start + 1] - zeroed[t_start]) > 0.005
  ):
    t_start += 1

  t_search_max = min(n_samples, qrs_onset + int(0.60 * fs))

  # Fallback if beat template is short
  if t_search_max <= t_start + int(0.04 * fs):
    t_start = min(n_samples - 1, r_idx + int(0.12 * fs))
    t_search_max = min(n_samples, r_idx + int(0.48 * fs))

  t_pos_amp_mv = 0.0
  t_neg_amp_mv = 0.0

  if t_search_max > t_start + 1:
    rep_segment = zeroed[t_start:t_search_max]
    max_rep = float(np.max(rep_segment))
    min_rep = float(np.min(rep_segment))

    has_pos_t = max_rep >= 0.025
    pos_peak_idx = t_start + int(np.argmax(rep_segment)) if has_pos_t else t_start

    # True negative T-wave deflection:
    # Requires a true negative deflection/trough in the repolarization segment.
    # If upright T wave exists, negative deflection requires a distinct local trough
    # (not just the start boundary ascending toward the upright T peak).
    neg_peak_idx = t_start + int(np.argmin(rep_segment))
    has_neg_t = False
    if min_rep <= -0.025:
      if neg_peak_idx > t_start:
        has_neg_t = True
      elif not has_pos_t:
        has_neg_t = True

    if has_pos_t or has_neg_t:
      last_peak_idx = max(
        pos_peak_idx if has_pos_t else t_start,
        neg_peak_idx if has_neg_t else t_start,
      )

      # Delineate T_end from last peak forward to baseline return or slope flattening
      t_end_idx = t_search_max
      for idx in range(last_peak_idx + 1, t_search_max):
        val = zeroed[idx]
        if abs(val) <= 0.015:
          t_end_idx = idx
          break
        if idx > last_peak_idx + 2 and idx < t_search_max - 1:
          local_slope = abs(zeroed[idx + 1] - zeroed[idx - 1]) / 2.0
          if local_slope < 0.003 and abs(val) < 0.025:
            t_end_idx = idx
            break
      t_end = max(t_start + int(0.04 * fs), t_end_idx)
    else:
      t_end = min(t_search_max, t_start + int(0.20 * fs))

    t_window = zeroed[t_start:t_end]
    if len(t_window) > 0:
      win_max = float(np.max(t_window))
      if win_max >= 0.025:
        t_pos_amp_mv = win_max
      if has_neg_t:
        if neg_peak_idx > t_start and neg_peak_idx < t_end:
          t_neg_amp_mv = float(abs(zeroed[neg_peak_idx]))
        else:
          win_min = float(np.min(t_window))
          if win_min <= -0.025:
            t_neg_amp_mv = float(abs(win_min))

  return LeadFeatures(
    q_duration_ms=q_dur_ms,
    q_amplitude_mv=q_amp_mv,
    r_amplitude_mv=r_amp_mv,
    s_amplitude_mv=s_amp_mv,
    t_pos_amplitude_mv=t_pos_amp_mv,
    t_neg_amplitude_mv=t_neg_amp_mv,
    qr_ratio=qr_ratio,
  )


def extract_ciis_measurements_from_waveform(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> CIISMeasurements:
  """Extract all 12 CIIS measurements from a 12-lead ECG waveform array.

  Args:
    signal_array: 2D array of shape [sig_len, n_sig].
    lead_names: List of lead names matching columns of signal_array.
    fs: Sampling rate in Hz.

  Returns:
    CIISMeasurements dataclass.
  """
  lead_map = {name: i for i, name in enumerate(lead_names)}
  for req in CANONICAL_12_LEADS:
    if req not in lead_map:
      raise ValueError(f"Required lead {req} not found in lead names: {lead_names}")

  # Filter each lead
  filtered_signals = np.zeros_like(signal_array)
  for i in range(signal_array.shape[1]):
    filtered_signals[:, i] = filter_ecg_signal(signal_array[:, i], fs)

  # Detect R-peaks using Lead II
  lead2_idx = lead_map["II"]
  r_peaks = detect_qrs_peaks(filtered_signals[:, lead2_idx], fs)
  if len(r_peaks) < 2:
    # Fallback to V4/V5 if Lead II has low amplitude
    v5_idx = lead_map["V5"]
    r_peaks = detect_qrs_peaks(filtered_signals[:, v5_idx], fs)
    if len(r_peaks) < 2:
      raise ValueError(f"Insufficient R-peaks detected for median beat extraction: {len(r_peaks)}")

  # Extract median beat
  median_beat, r_idx = extract_median_beat(filtered_signals, r_peaks, fs)

  # Extract features for all leads
  feats: dict[str, LeadFeatures] = {}
  for lead_name, col_idx in lead_map.items():
    feats[lead_name] = extract_lead_features(median_beat[:, col_idx], r_idx, fs)

  # Extract inverted aVR features (lead -aVR)
  inv_avr_signal = -1.0 * median_beat[:, lead_map["aVR"]]
  inv_avr_feats = extract_lead_features(inv_avr_signal, r_idx, fs)

  # Extract inverted aVL features (lead -aVL)
  inv_avl_signal = -1.0 * median_beat[:, lead_map["aVL"]]
  inv_avl_feats = extract_lead_features(inv_avl_signal, r_idx, fs)

  avl = feats["aVL"]
  lead2 = feats["II"]
  lead3 = feats["III"]
  avf = feats["aVF"]
  v1 = feats["V1"]
  v2 = feats["V2"]
  v3 = feats["V3"]
  v5 = feats["V5"]

  return CIISMeasurements(
    avl_q_duration_ms=avl.q_duration_ms,
    avl_t_pos_mv=avl.t_pos_amplitude_mv,
    avl_t_neg_mv=avl.t_neg_amplitude_mv,
    inv_avr_r_mv=inv_avr_feats.r_amplitude_mv,
    inv_avr_t_pos_mv=inv_avr_feats.t_pos_amplitude_mv,
    lead2_qr_ratio=lead2.qr_ratio,
    avf_qr_ratio=avf.qr_ratio,
    lead3_q_duration_ms=lead3.q_duration_ms,
    inv_avl_q_duration_ms=inv_avl_feats.q_duration_ms,
    lead3_t_neg_mv=lead3.t_neg_amplitude_mv,
    v1_t_pos_mv=v1.t_pos_amplitude_mv,
    v2_r_mv=v2.r_amplitude_mv,
    v2_t_neg_mv=v2.t_neg_amplitude_mv,
    v3_qr_ratio=v3.qr_ratio,
    v5_s_mv=v5.s_amplitude_mv,
  )


def score_ecg_waveform(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> CIISScoreResult:
  """Calculate CIIS continuous score and risk category directly from a 12-lead ECG waveform.

  Args:
    signal_array: 2D array of shape [sig_len, 12] in mV.
    lead_names: Sequence of lead names matching array channels.
    fs: Sampling frequency in Hz.

  Returns:
    CIISScoreResult with total_score, category, item_scores, measurements, and status.
  """
  if np.isnan(signal_array).any() or np.isinf(signal_array).any():
    return CIISScoreResult(
      is_valid=False,
      error_message="Waveform contains NaN or Infinite values.",
    )

  if signal_array.shape[0] < fs:
    return CIISScoreResult(
      is_valid=False,
      error_message=f"Waveform duration too short: {signal_array.shape[0]} samples at {fs} Hz.",
    )

  try:
    measurements = extract_ciis_measurements_from_waveform(signal_array, lead_names, fs)
    return compute_ciis_from_measurements(measurements)
  except Exception as exc:
    return CIISScoreResult(
      is_valid=False,
      error_message=f"Waveform scoring failure: {exc}",
    )


def compute_cornell_voltage(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> float:
  """Calculate Cornell Voltage index (R in aVL + S in V3 in mV) from a 12-lead ECG waveform.

  Cornell voltage is a classical electrophysiologic criterion for left ventricular hypertrophy
  and myocardial strain (normal cutoff: <= 2.8 mV in men, <= 2.0 mV in women).

  Args:
    signal_array: 2D array of shape [sig_len, 12] in mV.
    lead_names: Sequence of lead names matching channels.
    fs: Sampling rate in Hz.

  Returns:
    Cornell voltage sum in mV (R_aVL + S_V3).
  """
  lead_map = {name: i for i, name in enumerate(lead_names)}
  for req in ("aVL", "V3"):
    if req not in lead_map:
      raise ValueError(f"Required lead {req} not found in lead names: {lead_names}")

  # Filter signals
  filtered_signals = np.zeros_like(signal_array)
  for i in range(signal_array.shape[1]):
    filtered_signals[:, i] = filter_ecg_signal(signal_array[:, i], fs)

  lead2_idx = lead_map.get("II", 0)
  r_peaks = detect_qrs_peaks(filtered_signals[:, lead2_idx], fs)
  if len(r_peaks) < 2:
    v5_idx = lead_map.get("V5", 0)
    r_peaks = detect_qrs_peaks(filtered_signals[:, v5_idx], fs)
    if len(r_peaks) < 2:
      raise ValueError(f"Insufficient R-peaks detected for Cornell voltage: {len(r_peaks)}")

  median_beat, r_idx = extract_median_beat(filtered_signals, r_peaks, fs)
  avl_feat = extract_lead_features(median_beat[:, lead_map["aVL"]], r_idx, fs)
  v3_feat = extract_lead_features(median_beat[:, lead_map["V3"]], r_idx, fs)
  return float(avl_feat.r_amplitude_mv + v3_feat.s_amplitude_mv)


def compute_sokolow_lyon_voltage(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> float:
  """Calculate Sokolow-Lyon Voltage index (S in V1 + max(R in V5, R in V6) in mV).

  Classical voltage criterion for left ventricular hypertrophy (normal cutoff: <= 3.5 mV).

  Args:
    signal_array: 2D array of shape [sig_len, 12] in mV.
    lead_names: Sequence of lead names matching channels.
    fs: Sampling rate in Hz.

  Returns:
    Sokolow-Lyon voltage sum in mV (S_V1 + max(R_V5, R_V6)).
  """
  lead_map = {name: i for i, name in enumerate(lead_names)}
  for req in ("V1", "V5", "V6"):
    if req not in lead_map:
      raise ValueError(f"Required lead {req} not found in lead names: {lead_names}")

  filtered_signals = np.zeros_like(signal_array)
  for i in range(signal_array.shape[1]):
    filtered_signals[:, i] = filter_ecg_signal(signal_array[:, i], fs)

  lead2_idx = lead_map.get("II", 0)
  r_peaks = detect_qrs_peaks(filtered_signals[:, lead2_idx], fs)
  if len(r_peaks) < 2:
    v5_idx = lead_map.get("V5", 0)
    r_peaks = detect_qrs_peaks(filtered_signals[:, v5_idx], fs)
    if len(r_peaks) < 2:
      raise ValueError(f"Insufficient R-peaks detected for Sokolow-Lyon voltage: {len(r_peaks)}")

  median_beat, r_idx = extract_median_beat(filtered_signals, r_peaks, fs)
  v1_feat = extract_lead_features(median_beat[:, lead_map["V1"]], r_idx, fs)
  v5_feat = extract_lead_features(median_beat[:, lead_map["V5"]], r_idx, fs)
  v6_feat = extract_lead_features(median_beat[:, lead_map["V6"]], r_idx, fs)
  return float(v1_feat.s_amplitude_mv + max(v5_feat.r_amplitude_mv, v6_feat.r_amplitude_mv))


def score_simplified_ischemic_ecg(
  measurements: CIISMeasurements,
) -> float:
  """Calculate simplified 4-item ischemic score based on key CIIS components.

  Includes:
  1. aVL Q-wave duration
  2. Largest Q/R ratio in Lead II or aVF
  3. V2 T-wave inversion amplitude
  4. V5 S-wave amplitude

  Args:
    measurements: CIISMeasurements instance.

  Returns:
    Continuous simplified ischemic score points.
  """
  item1 = score_item1_avl_q_duration(measurements.avl_q_duration_ms)
  item5 = score_item5_lead2_avf_qr_ratio(measurements.lead2_qr_ratio, measurements.avf_qr_ratio)
  item10 = score_item10_v2_t_amplitude(measurements.v2_t_neg_mv)
  item12 = score_item12_v5_s_amplitude(measurements.v5_s_mv)
  return float(item1 + item5 + item10 + item12)


def validate_measurements_against_machine_measurements(
  extracted_df: pl.DataFrame,
  machine_df: pl.DataFrame,
) -> dict[str, Any]:
  """Quantitatively validate extracted features against MIMIC machine_measurements.csv.

  Matches recordings on study_id and calculates Pearson correlation, Spearman
  correlation, and Mean Absolute Error (MAE) across shared clinical intervals.
  """
  joined = extracted_df.join(machine_df, on="study_id", how="inner")
  metrics: dict[str, Any] = {
    "n_overlapping": len(joined),
    "features_evaluated": {},
  }
  if len(joined) < 2:
    return metrics

  candidate_pairs = [
    ("rr_interval_ms", "rr_interval"),
    ("qrs_duration_ms", "qrs_duration"),
    ("qrs_onset_ms", "qrs_onset"),
    ("qrs_end_ms", "qrs_end"),
    ("p_onset_ms", "p_onset"),
    ("p_end_ms", "p_end"),
    ("t_end_ms", "t_end"),
  ]

  for ext_col, mach_col in candidate_pairs:
    if ext_col in joined.columns and mach_col in joined.columns:
      valid = joined.filter(pl.col(ext_col).is_not_null() & pl.col(mach_col).is_not_null())
      if len(valid) >= 2:
        x = valid[ext_col].to_numpy().astype(np.float64)
        y = valid[mach_col].to_numpy().astype(np.float64)
        mae = float(np.mean(np.abs(x - y)))
        pr_res: Any = pearsonr(x, y)
        sr_res: Any = spearmanr(x, y)
        pr_val = float(getattr(pr_res, "statistic", pr_res[0]))
        sr_val = float(getattr(sr_res, "statistic", sr_res[0]))
        metrics["features_evaluated"][f"{ext_col}_vs_{mach_col}"] = {
          "n": len(valid),
          "pearson_r": pr_val,
          "spearman_r": sr_val,
          "mae": mae,
        }

  return metrics


def validate_ciis_feature_distributions(ciis_df: pl.DataFrame) -> dict[str, Any]:
  """Validate distributions and physiologic plausibility of the 14 CIIS continuous features.

  MIMIC-IV-ECG machine_measurements.csv contains global fiducial intervals
  (rr_interval, qrs_duration, etc.) rather than lead-specific wave amplitudes
  or continuous CIIS components (e.g. avl_q_duration, v5_s_mv). This validator
  audits the actual Z_A feature matrix against domain-physiologic plausibility
  bounds to identify extraction artifacts, clipping, or impossible values.

  Args:
    ciis_df: Polars DataFrame containing CIIS feature columns.

  Returns:
    Dict containing summary statistics per feature and an overall
    all_physiologically_valid boolean.
  """
  n_total = len(ciis_df)
  feature_stats: dict[str, Any] = {}
  all_valid = True

  for feat, (min_bound, max_bound) in CIIS_PHYSIOLOGIC_BOUNDS.items():
    if feat not in ciis_df.columns:
      continue
    col = ciis_df.get_column(feat)
    non_null = col.drop_nulls()
    arr = non_null.to_numpy().astype(np.float64)
    # Filter out non-finite values (NaN/Inf) since failed extractions are represented as NaN
    arr = arr[np.isfinite(arr)]
    n_valid = len(arr)
    if n_valid == 0:
      feature_stats[feat] = {
        "n_valid": 0,
        "valid_fraction": 0.0,
        "physiologically_valid": False,
      }
      all_valid = False
      continue

    c_min = float(np.min(arr))
    c_max = float(np.max(arr))
    c_median = float(np.median(arr))
    p1 = float(np.percentile(arr, 1.0))
    p99 = float(np.percentile(arr, 99.0))
    within_bounds = (arr >= min_bound) & (arr <= max_bound)
    fraction_in_bounds = float(np.mean(within_bounds))
    is_feat_valid = fraction_in_bounds >= 0.99 and c_min >= 0.0
    if not is_feat_valid:
      all_valid = False

    feature_stats[feat] = {
      "n_valid": n_valid,
      "valid_fraction": float(n_valid / n_total) if n_total > 0 else 0.0,
      "min": c_min,
      "p1": p1,
      "median": c_median,
      "p99": p99,
      "max": c_max,
      "physiologic_min": min_bound,
      "physiologic_max": max_bound,
      "fraction_in_bounds": fraction_in_bounds,
      "physiologically_valid": is_feat_valid,
    }

  return {
    "n_total": n_total,
    "all_physiologically_valid": all_valid,
    "features": feature_stats,
  }


DEFAULT_CIIS_MEASUREMENT_TOLERANCES: dict[str, float] = {
  "avl_q_duration_ms": 10.0,
  "avl_t_pos_mv": 0.10,
  "avl_t_neg_mv": 0.10,
  "inv_avr_r_mv": 0.15,
  "inv_avr_t_pos_mv": 0.10,
  "lead2_qr_ratio": 0.05,
  "avf_qr_ratio": 0.05,
  "lead3_q_duration_ms": 10.0,
  "inv_avl_q_duration_ms": 10.0,
  "lead3_t_neg_mv": 0.10,
  "v1_t_pos_mv": 0.10,
  "v2_r_mv": 0.20,
  "v2_t_neg_mv": 0.10,
  "v3_qr_ratio": 0.05,
  "v5_s_mv": 0.15,
}


def validate_ciis_measurements_against_reference(
  extracted_df: pl.DataFrame,
  reference_df: pl.DataFrame,
  id_col: str = "study_id",
  tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
  """Quantitatively validate continuous Z_A morphology features against independent reference values.

  Calculates sample size, Mean Absolute Error (MAE), mean bias (extracted - reference),
  error standard deviation, Pearson r, Spearman rank correlation rho, outlier/failure
  rate exceeding tolerance, and records top extreme disagreement cases.

  Args:
    extracted_df: DataFrame containing extracted continuous features.
    reference_df: DataFrame containing reference/benchmark measurements.
    id_col: Common identifier column (default: "study_id").
    tolerances: Per-feature absolute tolerance dictionary.

  Returns:
    Dict containing detailed per-feature validation statistics and extreme disagreements.
  """
  tol_map = tolerances or DEFAULT_CIIS_MEASUREMENT_TOLERANCES
  if id_col not in extracted_df.columns or id_col not in reference_df.columns:
    return {"n_overlapping": 0, "features": {}}

  joined = extracted_df.join(reference_df, on=id_col, how="inner", suffix="_ref")
  n_overlapping = len(joined)
  results: dict[str, Any] = {
    "n_overlapping": n_overlapping,
    "features": {},
  }

  if n_overlapping < 2:
    return results

  shared_features = [
    col
    for col in CIIS_FEATURE_NAMES
    if col in extracted_df.columns and (f"{col}_ref" in joined.columns or col in reference_df.columns)
  ]

  for feat in shared_features:
    ref_col = f"{feat}_ref" if f"{feat}_ref" in joined.columns else feat
    sub = joined.filter(pl.col(feat).is_not_null() & pl.col(ref_col).is_not_null())
    if len(sub) < 2:
      continue

    raw_x = sub[feat].to_numpy().astype(np.float64)
    raw_y = sub[ref_col].to_numpy().astype(np.float64)
    finite_mask = np.isfinite(raw_x) & np.isfinite(raw_y)
    x = raw_x[finite_mask]
    y = raw_y[finite_mask]
    n_pairs = len(x)
    if n_pairs < 2:
      continue

    diff = x - y
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    bias = float(np.mean(diff))
    error_std = float(np.std(diff))

    pr_val = 0.0
    sr_val = 0.0
    if np.std(x) > 1e-9 and np.std(y) > 1e-9:
      pr_res: Any = pearsonr(x, y)
      sr_res: Any = spearmanr(x, y)
      pr_val = float(getattr(pr_res, "statistic", pr_res[0]))
      sr_val = float(getattr(sr_res, "statistic", sr_res[0]))

    tol = tol_map.get(feat, 0.10)
    outliers = abs_diff > tol
    outlier_rate = float(np.mean(outliers))

    top_k = min(5, n_pairs)
    top_idx = np.argsort(-abs_diff)[:top_k]
    all_sids = sub[id_col].to_list()
    valid_sids = [sid for sid, m in zip(all_sids, finite_mask) if m]
    extreme_cases = [
      {
        "study_id": int(valid_sids[i]),
        "extracted": float(x[i]),
        "reference": float(y[i]),
        "difference": float(diff[i]),
        "abs_error": float(abs_diff[i]),
      }
      for i in top_idx
    ]

    results["features"][feat] = {
      "n": n_pairs,
      "mae": mae,
      "bias": bias,
      "error_std": error_std,
      "pearson_r": pr_val,
      "spearman_rho": sr_val,
      "tolerance": tol,
      "outlier_rate": outlier_rate,
      "extreme_disagreements": extreme_cases,
    }

  return results


def extract_independent_benchmark_measurements(
  signal_array: npt.NDArray[np.float64],
  lead_names: Sequence[str],
  fs: int = DEFAULT_SAMPLING_RATE_HZ,
) -> dict[str, float]:
  """Independent baseline signal-processing measurement extractor for CIIS morphology.

  Uses scipy.signal peak finding on bandpass-filtered leads rather than the
  primary template-matching algorithm, providing an independent algorithmic benchmark.
  """
  name_to_idx = {name.upper(): i for i, name in enumerate(lead_names)}
  nyq = 0.5 * fs
  b, a = cast(
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    signal.butter(2, [0.5 / nyq, 40.0 / nyq], btype="bandpass", output="ba"),
  )
  filtered = signal.filtfilt(b, a, signal_array, axis=0)

  results: dict[str, float] = {}
  inv_avr = -filtered[:, name_to_idx["AVR"]] if "AVR" in name_to_idx else np.zeros(len(filtered))
  inv_avl = -filtered[:, name_to_idx["AVL"]] if "AVL" in name_to_idx else np.zeros(len(filtered))

  def get_lead_fiducials(sig: npt.NDArray[np.float64]) -> tuple[float, float, float, float, float, float]:
    peaks, _ = signal.find_peaks(sig, distance=int(0.5 * fs), prominence=0.2)
    if len(peaks) == 0:
      return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    r_idx = int(peaks[len(peaks) // 2])
    raw_r = float(sig[r_idx])
    r_amp = raw_r if raw_r >= 0.025 else 0.0

    q_window = int(0.06 * fs)
    q_start = max(0, r_idx - q_window)
    q_sub = sig[q_start:r_idx]
    if len(q_sub) > 0 and np.min(q_sub) <= -0.025:
      q_amp = float(-np.min(q_sub))
      q_dur = float((r_idx - (q_start + np.argmin(q_sub))) * 1000.0 / fs)
    else:
      q_amp = 0.0
      q_dur = 0.0

    s_window = int(0.08 * fs)
    s_end = min(len(sig), r_idx + s_window)
    s_sub = sig[r_idx:s_end]
    s_amp = (
      float(-np.min(s_sub))
      if len(s_sub) > 0 and np.min(s_sub) <= -0.025 and r_amp >= 0.025
      else 0.0
    )

    t_start = min(len(sig), r_idx + int(0.14 * fs))
    t_end = min(len(sig), r_idx + int(0.48 * fs))
    t_sub = sig[t_start:t_end]
    raw_t_pos = float(np.max(t_sub)) if len(t_sub) > 0 else 0.0
    raw_t_neg = float(-np.min(t_sub)) if len(t_sub) > 0 and np.min(t_sub) < 0 else 0.0
    t_pos = raw_t_pos if raw_t_pos >= 0.025 else 0.0
    t_neg = raw_t_neg if raw_t_neg >= 0.025 else 0.0

    return q_dur, q_amp, r_amp, s_amp, t_pos, t_neg

  if "AVL" in name_to_idx:
    q_dur, _, _, _, t_pos, t_neg = get_lead_fiducials(filtered[:, name_to_idx["AVL"]])
    results["avl_q_duration_ms"] = q_dur
    results["avl_t_pos_mv"] = t_pos
    results["avl_t_neg_mv"] = t_neg

  if len(inv_avr) > 0:
    _, _, r_amp, _, t_pos, _ = get_lead_fiducials(inv_avr)
    results["inv_avr_r_mv"] = r_amp
    results["inv_avr_t_pos_mv"] = t_pos

  if "II" in name_to_idx:
    _, q_amp, r_amp, _, _, _ = get_lead_fiducials(filtered[:, name_to_idx["II"]])
    results["lead2_qr_ratio"] = (
      min(10.0, float(q_amp / r_amp)) if r_amp >= 0.025 else (10.0 if q_amp >= 0.025 else 0.0)
    )

  if "AVF" in name_to_idx:
    _, q_amp, r_amp, _, _, _ = get_lead_fiducials(filtered[:, name_to_idx["AVF"]])
    results["avf_qr_ratio"] = (
      min(10.0, float(q_amp / r_amp)) if r_amp >= 0.025 else (10.0 if q_amp >= 0.025 else 0.0)
    )

  if "III" in name_to_idx:
    q_dur, _, _, _, _, t_neg = get_lead_fiducials(filtered[:, name_to_idx["III"]])
    results["lead3_q_duration_ms"] = q_dur
    results["lead3_t_neg_mv"] = t_neg

  if len(inv_avl) > 0:
    q_dur, _, _, _, _, _ = get_lead_fiducials(inv_avl)
    results["inv_avl_q_duration_ms"] = q_dur

  if "V1" in name_to_idx:
    _, _, _, _, t_pos, _ = get_lead_fiducials(filtered[:, name_to_idx["V1"]])
    results["v1_t_pos_mv"] = t_pos

  if "V2" in name_to_idx:
    _, _, r_amp, _, _, t_neg = get_lead_fiducials(filtered[:, name_to_idx["V2"]])
    results["v2_r_mv"] = r_amp
    results["v2_t_neg_mv"] = t_neg

  if "V3" in name_to_idx:
    _, q_amp, r_amp, _, _, _ = get_lead_fiducials(filtered[:, name_to_idx["V3"]])
    results["v3_qr_ratio"] = (
      min(10.0, float(q_amp / r_amp)) if r_amp >= 0.025 else (10.0 if q_amp >= 0.025 else 0.0)
    )

  if "V5" in name_to_idx:
    _, _, _, s_amp, _, _ = get_lead_fiducials(filtered[:, name_to_idx["V5"]])
    results["v5_s_mv"] = s_amp

  return results


def run_ciis_measurement_validation_study(
  extracted_df: pl.DataFrame,
  machine_measurements_df: pl.DataFrame | None = None,
  reference_df: pl.DataFrame | None = None,
  id_col: str = "study_id",
) -> dict[str, Any]:
  """Execute full multi-tier measurement validation for Representation A (Z_A).

  Includes:
  1. Internal distribution plausibility audit across physiologic bounds.
  2. Independent morphology feature validation (MAE, bias, rank correlation, outlier rate, extreme disagreements) if reference_df is provided.
  3. Machine interval correlation (e.g. Marquette 12SL) if machine_measurements_df is provided.
  """
  report: dict[str, Any] = {
    "distribution_qc": validate_ciis_feature_distributions(extracted_df),
  }
  if reference_df is not None and len(reference_df) > 0:
    agreement_res = validate_ciis_measurements_against_reference(
      extracted_df=extracted_df,
      reference_df=reference_df,
      id_col=id_col,
    )
    report["independent_algorithmic_agreement_validation"] = agreement_res
    report["independent_reference_validation"] = agreement_res  # backward compatibility
  if machine_measurements_df is not None and len(machine_measurements_df) > 0:
    report["machine_measurements_validation"] = validate_measurements_against_machine_measurements(
      extracted_df=extracted_df,
      machine_df=machine_measurements_df,
    )
  return report




