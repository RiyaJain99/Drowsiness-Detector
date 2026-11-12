"""Run the saved joblib model outside the browser.

    python -m drowsyguard_ml.inference path/to/video.mp4

Importable too: ``DrowsinessClassifier().predict(vector)``. The live dashboard
uses the exported copy of the same model; this module is for batch scoring and
is what a local API would wrap if a Python service were ever added.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np

from .config import ARTIFACTS_DIR, FeatureConfig
from .features import FEATURE_NAMES, video_windows


@dataclass(frozen=True)
class Prediction:
    label: str
    probability_drowsy: float
    confidence: float


class DrowsinessClassifier:
    def __init__(self, path: Path = ARTIFACTS_DIR / "model.joblib"):
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist; train a model first.")
        bundle = joblib.load(path)
        if bundle["feature_names"] != FEATURE_NAMES:
            raise ValueError("Model was trained on a different feature set than this code computes.")
        self.model = bundle["model"]
        self.columns = [FEATURE_NAMES.index(c) for c in bundle.get("columns", FEATURE_NAMES)]
        self.threshold = bundle["threshold"]
        self.feature_config = FeatureConfig(**bundle["feature_config"])

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        """``vectors``: full feature vectors in FEATURE_NAMES order."""
        full = np.asarray(vectors, dtype=np.float32)
        return self.model.predict_proba(full[:, self.columns])[:, 1]

    def predict(self, vector: Sequence[float]) -> Prediction:
        p = float(self.predict_proba(np.asarray(vector)[None, :])[0])
        drowsy = p >= self.threshold
        return Prediction("drowsy" if drowsy else "alert", p, p if drowsy else 1 - p)

    def predict_video(self, path: Path) -> List[dict]:
        from .extract import extract_video

        frames, _ = extract_video(path, self.feature_config)
        rows = list(video_windows(frames, self.feature_config))
        if not rows:
            return []
        proba = self.predict_proba(np.array([[r[n] for n in FEATURE_NAMES] for r in rows]))
        return [{"t_end_s": r["t_end_s"], "probability_drowsy": float(p)} for r, p in zip(rows, proba)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    classifier = DrowsinessClassifier()
    rows = classifier.predict_video(args.video)
    if not rows:
        raise SystemExit("No usable windows: the clip is too short or no face could be calibrated.")
    for row in rows:
        verdict = "drowsy" if row["probability_drowsy"] >= classifier.threshold else "alert"
        print(f"{row['t_end_s']:7.1f}s  {verdict:6s}  P(drowsy)={row['probability_drowsy']:.3f}")
    print(f"mean P(drowsy) = {np.mean([r['probability_drowsy'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
