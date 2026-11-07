"""Per-frame facial measurements from MediaPipe landmarks.

Mirrors ``src/lib/geometry.ts`` and ``src/lib/mlFeatures.ts``. One deliberate
difference from the dashboard's EAR: ``y`` is rescaled by the aspect ratio first.
Training videos are 16:9, 4:3 and portrait, and normalised coordinates would
give the same eye a different EAR in each.
"""
from __future__ import annotations

import math

import numpy as np

RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_VERTICAL = (13, 14)
MOUTH_HORIZONTAL = (61, 291)
FOREHEAD = 10
CHIN = 152
EYE_OUTER = (33, 263)

DEG = 180.0 / math.pi


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(p: np.ndarray, idx: tuple) -> float:
    p1, p2, p3, p4, p5, p6 = (p[i] for i in idx)
    width = _dist(p1, p4)
    return 0.0 if width == 0 else (_dist(p2, p6) + _dist(p3, p5)) / (2 * width)


def mouth_aspect_ratio(p: np.ndarray) -> float:
    width = _dist(p[MOUTH_HORIZONTAL[0]], p[MOUTH_HORIZONTAL[1]])
    return 0.0 if width == 0 else _dist(p[MOUTH_VERTICAL[0]], p[MOUTH_VERTICAL[1]]) / width


def head_pose(p: np.ndarray) -> tuple:
    """(pitch, yaw, roll) in degrees from aspect-corrected 3-D landmarks."""
    a, b = p[EYE_OUTER[0]], p[EYE_OUTER[1]]
    top, bottom = p[FOREHEAD], p[CHIN]
    rx, ry, rz = b - a
    ux, uy, uz = top - bottom
    pitch = math.atan2(uz, math.hypot(ux, uy)) * DEG
    yaw = math.atan2(rz, math.hypot(rx, ry)) * DEG
    roll = math.atan2(ry, abs(rx)) * DEG
    return pitch, yaw, roll


def measure(landmarks: np.ndarray, aspect: float) -> dict:
    """``landmarks``: (478, 3) normalised MediaPipe output; ``aspect``: width/height."""
    p = landmarks.astype(np.float64).copy()
    p[:, 1] /= aspect or 1.0
    left = eye_aspect_ratio(p, LEFT_EYE)
    right = eye_aspect_ratio(p, RIGHT_EYE)
    pitch, yaw, roll = head_pose(p)
    return {
        "ear": (left + right) / 2,
        "ear_l": left,
        "ear_r": right,
        "mar": mouth_aspect_ratio(p),
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }
