"""Generate the Python/TypeScript parity fixture.

    python -m drowsyguard_ml.parity [--real path/to/video.parquet]

Writes ``src/lib/__tests__/fixtures/feature-parity.json``: measurement series and
the features Python computes from them. ``mlFeatures.test.ts`` must reproduce
them. This is a test fixture and is never used for training or evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_FEATURE_CONFIG, REPO_ROOT
from .features import FEATURE_NAMES, SIGNALS, calibrate, window_features
from .geometry import measure

OUT = REPO_ROOT / "src" / "lib" / "__tests__" / "fixtures" / "feature-parity.json"


class Lcg:
    """Tiny deterministic generator, so the fixture is stable across numpy versions."""

    def __init__(self, seed: int):
        self.state = seed

    def next(self) -> float:
        self.state = (1103515245 * self.state + 12345) % 2147483648
        return self.state / 2147483648


def scripted_series(seconds: int = 70) -> Dict[str, List[float]]:
    """Exercises every branch: blinks, a long closure, a yawn, a nod and a tracking gap."""
    rng = Lcg(7)
    period = 1000 / 15
    out: Dict[str, List[float]] = {k: [] for k in ("t_ms",) + SIGNALS}
    for i in range(seconds * 15):
        t = i * period
        s = t / 1000
        ear = 0.30 + (rng.next() - 0.5) * 0.01
        if s > 4 and s % 4 < 0.2:
            ear = 0.10
        if 30 < s < 31.5:
            ear = 0.08
        mar = 0.05 + (rng.next() - 0.5) * 0.01
        if 40 < s < 42:
            mar = 0.8
        pitch = 2 + (rng.next() - 0.5)
        if 50 < s < 51.5:
            pitch = -25
        gap = 58 < s < 60
        nan = float("nan")
        values = {
            "t_ms": t,
            "ear": nan if gap else ear,
            "ear_l": nan if gap else ear * 1.03,
            "ear_r": nan if gap else ear * 0.97,
            "mar": nan if gap else mar,
            "pitch": nan if gap else pitch,
            "yaw": nan if gap else 3 + math.sin(s),
            "roll": nan if gap else -1 + math.cos(s / 2),
        }
        for key, value in values.items():
            out[key].append(value)
    return out


def _case(name: str, series: Dict[str, List[float]]) -> Optional[Dict]:
    cfg = DEFAULT_FEATURE_CONFIG
    arrays = {k: np.array(v, dtype=np.float64) for k, v in series.items()}
    cal = calibrate(arrays, cfg)
    if cal is None:
        return None
    t = arrays["t_ms"]
    keep = t > t[-1] - cfg.window_s * 1000
    cols = {k: v[keep].tolist() for k, v in arrays.items()}
    features = window_features(cols["t_ms"], cols["ear"], cols["ear_l"], cols["ear_r"], cols["mar"],
                               cols["pitch"], cols["yaw"], cols["roll"], cal, cfg)
    json_safe = {k: [None if math.isnan(x) else x for x in v] for k, v in series.items()}
    return {"name": name, "series": json_safe, "calibration": cal.__dict__, "features": features}


def scripted_landmarks() -> Dict:
    rng = Lcg(11)
    points = [[rng.next(), rng.next(), (rng.next() - 0.5) * 0.2] for _ in range(478)]
    return {"points": points, "aspect": 4 / 3, "measured": measure(np.array(points), 4 / 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, help="also include 70 s of a real extracted measurement series")
    args = parser.parse_args()
    cases = [_case("scripted", scripted_series())]
    if args.real:
        frames = pd.read_parquet(args.real)
        frames = frames[(frames.t_ms >= 60000) & (frames.t_ms < 130000)]
        cases.append(_case("real", {c: frames[c].round(6).tolist() for c in ("t_ms",) + SIGNALS}))
    document = {"feature_names": FEATURE_NAMES, "feature_config": DEFAULT_FEATURE_CONFIG.to_dict(),
                "cases": [c for c in cases if c], "landmarks": scripted_landmarks()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document))
    print(f"wrote {OUT} with {len(document['cases'])} cases")


if __name__ == "__main__":
    main()
