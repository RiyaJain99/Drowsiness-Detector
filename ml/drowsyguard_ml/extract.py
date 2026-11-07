"""Video file -> per-frame measurement table. No pixels are kept."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions, vision

from .config import DEFAULT_FEATURE_CONFIG, LANDMARKER_PATH, FeatureConfig
from .geometry import measure

COLUMNS = ["t_ms", "face", "ear", "ear_l", "ear_r", "mar", "pitch", "yaw", "roll"]


def _landmarker() -> vision.FaceLandmarker:
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"{LANDMARKER_PATH} is missing; run `npm run setup` first.")
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
    )


def extract_video(
    path: Path,
    cfg: FeatureConfig = DEFAULT_FEATURE_CONFIG,
    save_frames_dir: Optional[Path] = None,
    save_every_s: float = 10.0,
) -> tuple:
    """Return ``(frames, info)``. Rows with no detected face carry NaN measurements.

    ``save_frames_dir`` optionally writes sampled JPEGs. It is off by default:
    24 of RLDD's 60 participants did not agree to have their faces published.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    period_ms = 1000.0 / cfg.sample_hz
    next_t = 0.0
    next_save = 0.0
    index = 0
    last_ts = -1
    rows = []
    shape = None

    with _landmarker() as landmarker:
        while cap.grab():
            pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = pos if pos > 0 or index == 0 else index * 1000.0 / fps
            index += 1
            if t + 1e-6 < next_t:
                continue
            # Advance on the grid; after a long frame gap, resume from now.
            next_t = max(next_t + period_ms, t)
            ok, frame = cap.retrieve()
            if not ok:
                continue

            h, w = frame.shape[:2]
            scale = cfg.max_frame_side / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            shape = (w, h)

            ts = int(t)
            if ts <= last_ts:  # MediaPipe requires strictly increasing timestamps
                ts = last_ts + 1
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            if result.face_landmarks:
                pts = np.array([(l.x, l.y, l.z) for l in result.face_landmarks[0]])
                m = measure(pts, w / h)
                rows.append((t, 1, m["ear"], m["ear_l"], m["ear_r"], m["mar"], m["pitch"], m["yaw"], m["roll"]))
            else:
                rows.append((t, 0) + (float("nan"),) * 7)

            if save_frames_dir is not None and t >= next_save:
                save_frames_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_frames_dir / f"{int(t):09d}.jpg"), frame)
                next_save = t + save_every_s * 1000
    cap.release()

    frames = pd.DataFrame(rows, columns=COLUMNS)
    info = {
        "source_fps": float(fps),
        "width": shape[0] if shape else 0,
        "height": shape[1] if shape else 0,
        "n_samples": int(len(frames)),
        "duration_s": float(frames.t_ms.iloc[-1] / 1000) if len(frames) else 0.0,
        "face_frac": float(frames.face.mean()) if len(frames) else 0.0,
    }
    return frames, info
