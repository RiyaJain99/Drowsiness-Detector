"""Windowed behavioural features.

THE PARITY CONTRACT: ``src/lib/mlFeatures.ts`` is a line-for-line port of this
file. Change one and you must change the other, then regenerate the fixture with
``python -m drowsyguard_ml.parity``; vitest and pytest both fail on any drift.
Keep to plain loops and operations that have an exact TypeScript equivalent.

All features are finite by construction (absent events yield 0), so no imputer
has to be shipped to the browser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_FEATURE_CONFIG, FeatureConfig

FEATURE_NAMES: List[str] = [
    # Eye opening
    "ear_abs_mean", "ear_rel_mean", "ear_rel_std", "ear_rel_p10", "ear_rel_p50", "ear_asym_mean",
    # PERCLOS
    "perclos", "perclos_deep", "perclos_recent",
    # Blinks
    "blink_rate", "blink_dur_mean_ms", "blink_dur_max_ms", "blink_amp_mean", "blink_open_vel_mean",
    "blink_interval_cv",
    # Eye closure duration
    "long_closure_rate", "closure_max_ms", "closure_mean_ms",
    # Mouth
    "mar_rel_mean", "mar_std", "mar_rel_max", "yawn_rate", "yawn_frac",
    # Head pose and nodding
    "pitch_rel_mean", "pitch_std", "pitch_rel_min", "yaw_std", "roll_std", "head_speed_mean",
    "nod_rate", "head_down_frac",
    # Temporal behaviour
    "ear_rel_slope", "perclos_trend", "blink_rate_trend", "pitch_trend",
    # Signal quality
    "visible_frac",
]


@dataclass(frozen=True)
class Calibration:
    ear: float
    mar: float
    pitch: float
    yaw: float
    roll: float


def trimmed_mean(values: np.ndarray, trim: float) -> float:
    """Same as ``trimmedMean`` in geometry.ts."""
    s = np.sort(values)
    cut = int(math.floor(len(s) * trim))
    kept = s[cut: len(s) - cut] if len(s) - 2 * cut > 0 else s
    return float(np.mean(kept))


def calibrate(frames: Dict[str, np.ndarray], cfg: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> Optional[Calibration]:
    """Baseline from the first ``calibration_s`` of the series, as the app does at start-up."""
    t = frames["t_ms"]
    if len(t) == 0:
        return None
    mask = (t < t[0] + cfg.calibration_s * 1000) & ~np.isnan(frames["ear"])
    if mask.sum() < cfg.calibration_min_frames:
        return None
    ear = trimmed_mean(frames["ear"][mask], cfg.calibration_trim)
    if ear < cfg.calibration_min_ear:
        return None
    return Calibration(
        ear=ear,
        mar=trimmed_mean(frames["mar"][mask], cfg.calibration_trim),
        pitch=trimmed_mean(frames["pitch"][mask], cfg.calibration_trim),
        yaw=trimmed_mean(frames["yaw"][mask], cfg.calibration_trim),
        roll=trimmed_mean(frames["roll"][mask], cfg.calibration_trim),
    )


def _runs(flags: List[bool]) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    runs = []
    start = -1
    for i, flag in enumerate(flags):
        if flag and start < 0:
            start = i
        elif not flag and start >= 0:
            runs.append((start, i - 1))
            start = -1
    if start >= 0:
        runs.append((start, len(flags) - 1))
    return runs


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _percentile(sorted_values: List[float], q: float) -> float:
    """Linear interpolation; matches numpy's default."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _slope_per_minute(ts: List[float], ys: List[float]) -> float:
    n = len(ts)
    if n < 2:
        return 0.0
    mt, my = _mean(ts), _mean(ys)
    num = sum((t - mt) * (y - my) for t, y in zip(ts, ys))
    den = sum((t - mt) ** 2 for t in ts)
    return 0.0 if den == 0 else num / den * 60000.0


def window_features(
    t: List[float],
    ear: List[float],
    ear_l: List[float],
    ear_r: List[float],
    mar: List[float],
    pitch: List[float],
    yaw: List[float],
    roll: List[float],
    cal: Calibration,
    cfg: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> Optional[List[float]]:
    """Features for one window, in ``FEATURE_NAMES`` order, or None if unusable.

    Inputs are parallel lists with NaN where no face was found.
    """
    n = len(t)
    if n < 2:
        return None
    period = 1000.0 / cfg.sample_hz
    valid = [not math.isnan(v) for v in ear]
    idx = [i for i in range(n) if valid[i]]
    visible = len(idx) / n
    span_ms = t[n - 1] - t[0]
    if visible < cfg.min_visible_frac or span_ms < cfg.min_window_s * 1000 - 2 * period:
        return None
    minutes = span_ms * visible / 60000.0
    half_minutes = minutes / 2
    t_mid = (t[0] + t[n - 1]) / 2
    t_recent = t[n - 1] - cfg.recent_s * 1000

    rel = [ear[i] / cal.ear if valid[i] else 0.0 for i in range(n)]
    rel_valid = [rel[i] for i in idx]
    closed = [valid[i] and rel[i] < cfg.closure_ratio for i in range(n)]
    deep = [valid[i] and rel[i] < cfg.deep_closure_ratio for i in range(n)]

    def closed_share(lo: float, hi: float) -> float:
        total = shut = 0
        for i in idx:
            if lo <= t[i] <= hi:
                total += 1
                shut += 1 if closed[i] else 0
        return shut / total * 100 if total else 0.0

    # Closures, split into blinks and long closures by the app's own limits.
    closure_durs: List[float] = []
    blink_durs: List[float] = []
    blink_amps: List[float] = []
    blink_vels: List[float] = []
    blink_starts: List[float] = []
    long_count = 0
    for a, b in _runs(closed):
        dur = t[b] - t[a] + period
        closure_durs.append(dur)
        if dur >= cfg.long_closure_ms:
            long_count += 1
        if dur <= cfg.blink_max_ms:
            lowest = a
            for i in range(a, b + 1):
                if rel[i] < rel[lowest]:
                    lowest = i
            blink_durs.append(dur)
            blink_starts.append(t[a])
            blink_amps.append(1 - rel[lowest])
            if b + 1 < n and valid[b + 1] and t[b + 1] > t[lowest]:
                blink_vels.append((rel[b + 1] - rel[lowest]) / ((t[b + 1] - t[lowest]) / 1000))

    intervals = [blink_starts[i + 1] - blink_starts[i] for i in range(len(blink_starts) - 1)]
    interval_mean = _mean(intervals)
    blinks_first = sum(1 for s in blink_starts if s < t_mid)
    blinks_second = len(blink_starts) - blinks_first

    # Mouth
    mar_valid = [mar[i] for i in idx]
    mar_rel = [v - cal.mar for v in mar_valid]
    yawn_limit = max(cfg.yawn_ratio, cal.mar + cfg.yawn_margin)
    yawn_ms = 0.0
    yawns = 0
    for a, b in _runs([valid[i] and mar[i] > yawn_limit for i in range(n)]):
        dur = t[b] - t[a] + period
        if dur >= cfg.yawn_hold_ms:
            yawns += 1
            yawn_ms += dur

    # Head
    pitch_rel = [pitch[i] - cal.pitch if valid[i] else 0.0 for i in range(n)]
    pitch_valid = [pitch_rel[i] for i in idx]
    nods = 0
    for a, b in _runs([valid[i] and pitch_rel[i] <= -cfg.nod_degrees for i in range(n)]):
        if t[b] - t[a] + period >= cfg.nod_hold_ms:
            nods += 1
    down = sum(1 for i in idx if pitch_rel[i] <= -cfg.head_down_degrees)
    speeds: List[float] = []
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if valid[i] and valid[i - 1] and 0 < dt <= cfg.head_speed_max_gap_ms:
            step = math.sqrt(
                (pitch[i] - pitch[i - 1]) ** 2 + (yaw[i] - yaw[i - 1]) ** 2 + (roll[i] - roll[i - 1]) ** 2
            )
            speeds.append(step / (dt / 1000))

    first_pitch = [pitch_rel[i] for i in idx if t[i] < t_mid]
    second_pitch = [pitch_rel[i] for i in idx if t[i] >= t_mid]
    sorted_rel = sorted(rel_valid)

    return [
        _mean([ear[i] for i in idx]),
        _mean(rel_valid),
        _std(rel_valid),
        _percentile(sorted_rel, 0.10),
        _percentile(sorted_rel, 0.50),
        _mean([abs(ear_l[i] - ear_r[i]) / ear[i] if ear[i] > 0 else 0.0 for i in idx]),
        closed_share(t[0], t[n - 1]),
        sum(1 for i in idx if deep[i]) / len(idx) * 100,
        closed_share(t_recent, t[n - 1]),
        len(blink_durs) / minutes,
        _mean(blink_durs),
        max(blink_durs) if blink_durs else 0.0,
        _mean(blink_amps),
        _mean(blink_vels),
        _std(intervals) / interval_mean if len(intervals) >= 2 and interval_mean > 0 else 0.0,
        long_count / minutes,
        max(closure_durs) if closure_durs else 0.0,
        _mean(closure_durs),
        _mean(mar_rel),
        _std(mar_valid),
        max(mar_rel),
        yawns / minutes,
        yawn_ms / (span_ms * visible),
        _mean(pitch_valid),
        _std(pitch_valid),
        min(pitch_valid),
        _std([yaw[i] for i in idx]),
        _std([roll[i] for i in idx]),
        _mean(speeds),
        nods / minutes,
        down / len(idx),
        _slope_per_minute([t[i] for i in idx], rel_valid),
        closed_share(t_mid, t[n - 1]) - closed_share(t[0], t_mid - 1e-9),
        (blinks_second - blinks_first) / half_minutes,
        _mean(second_pitch) - _mean(first_pitch),
        visible,
    ]


SIGNALS = ("ear", "ear_l", "ear_r", "mar", "pitch", "yaw", "roll")


def video_windows(frames: pd.DataFrame, cfg: FeatureConfig = DEFAULT_FEATURE_CONFIG) -> Iterator[dict]:
    """Slide over one video. Windows grow to ``window_s`` exactly as live warm-up does."""
    t_all = frames["t_ms"].to_numpy()
    if len(t_all) == 0:
        return
    keep = (t_all >= cfg.trim_start_s * 1000) & (t_all <= t_all[-1] - cfg.trim_end_s * 1000)
    trimmed = frames.loc[keep]
    arrays = {c: trimmed[c].to_numpy(dtype=np.float64) for c in ("t_ms",) + SIGNALS}
    labels = trimmed["frame_label"].to_numpy() if "frame_label" in trimmed else None
    t = arrays["t_ms"]
    if len(t) == 0:
        return
    cal = calibrate(arrays, cfg)
    if cal is None:
        return

    end = t[0] + cfg.min_window_s * 1000
    while end <= t[-1] + 1e-6:
        lo = np.searchsorted(t, end - cfg.window_s * 1000, side="left")
        hi = np.searchsorted(t, end, side="right")
        cols = {c: arrays[c][lo:hi].tolist() for c in ("t_ms",) + SIGNALS}
        vector = window_features(
            cols["t_ms"], cols["ear"], cols["ear_l"], cols["ear_r"], cols["mar"],
            cols["pitch"], cols["yaw"], cols["roll"], cal, cfg,
        )
        if vector is not None:
            row = dict(zip(FEATURE_NAMES, vector))
            row["t_end_s"] = end / 1000
            if labels is not None:
                # Per-frame datasets: the window takes its majority label.
                values, counts = np.unique(labels[lo:hi], return_counts=True)
                row["label"] = int(values[np.argmax(counts)])
            yield row
        end += cfg.stride_s * 1000
