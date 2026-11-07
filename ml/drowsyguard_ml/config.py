"""Every constant shared between training and live inference.

``FeatureConfig`` is written into ``public/ml/model.json`` and the browser reads
it from there, so a model always runs with the settings it was trained with.
Values that have a counterpart in ``src/lib/config.ts`` use the same number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
DATA_DIR = ML_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
TMP_DIR = DATA_DIR / "tmp"
ARTIFACTS_DIR = ML_ROOT / "artifacts"
PUBLIC_ML_DIR = REPO_ROOT / "public" / "ml"
# The same landmarker the browser loads, so landmarks mean the same thing in both.
LANDMARKER_PATH = REPO_ROOT / "public" / "mediapipe" / "face_landmarker.task"


@dataclass(frozen=True)
class FeatureConfig:
    # Frames are thinned to this rate in both training and live use, so a
    # 30 fps webcam and a 24 fps dataset video produce comparable series.
    sample_hz: float = 15.0
    # Longest frame the extractor feeds the landmarker; the app asks for 960x720.
    max_frame_side: int = 960

    window_s: float = 60.0  # matches perclosWindowMs
    min_window_s: float = 30.0  # first prediction, and YawDD's shortest usable clip
    stride_s: float = 10.0  # training only; live re-evaluates more often
    recent_s: float = 15.0
    min_visible_frac: float = 0.5

    calibration_s: float = 4.0  # matches calibrationMs
    calibration_min_frames: int = 30
    calibration_min_ear: float = 0.12
    calibration_trim: float = 0.2

    closure_ratio: float = 0.78  # matches closureRatio
    deep_closure_ratio: float = 0.6
    blink_max_ms: float = 400.0  # matches blinkMaxMs
    long_closure_ms: float = 500.0  # matches microsleepMs
    yawn_ratio: float = 0.55  # matches yawnRatio
    yawn_margin: float = 0.25  # matches yawnMarMargin
    yawn_hold_ms: float = 900.0  # matches yawnHoldMs
    nod_degrees: float = 18.0  # matches nodPitchDegrees
    nod_hold_ms: float = 800.0  # matches nodHoldMs
    head_down_degrees: float = 8.0
    head_speed_max_gap_ms: float = 200.0

    # Dataset videos open and close with the participant reaching for the
    # camera. A live driver is told to look ahead instead, so drop those seconds.
    trim_start_s: float = 5.0
    trim_end_s: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_FEATURE_CONFIG = FeatureConfig()
