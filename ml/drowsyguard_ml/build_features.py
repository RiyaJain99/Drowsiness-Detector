"""Stage 2: cached per-frame tables -> one labelled window table.

    python -m drowsyguard_ml.build_features

Writes ``ml/data/features.parquet``. Cheap to re-run; no video access needed.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, DEFAULT_FEATURE_CONFIG, FRAMES_DIR
from .features import FEATURE_NAMES, video_windows
from .splits import assert_subject_independent, assign_folds, partition

FEATURES_PATH = DATA_DIR / "features.parquet"


def _one(path: str) -> list:
    source = Path(path)
    meta = json.loads(source.with_suffix(".json").read_text())
    frames = pd.read_parquet(source)
    rows = []
    for row in video_windows(frames, DEFAULT_FEATURE_CONFIG):
        rows.append({
            **row,
            "label": row.get("label", meta["label"]),
            "dataset": meta["dataset"],
            "subject_key": f"{meta['dataset']}:{meta['subject']}",
            "video_key": f"{meta['dataset']}:{meta['subject']}:{meta['video_id']}",
            "fold": meta["fold"],
        })
    return rows


def build(frames_dir: Path = FRAMES_DIR, workers: int = 4) -> pd.DataFrame:
    paths = sorted(str(p) for p in frames_dir.rglob("*.parquet"))
    if not paths:
        raise SystemExit(f"No extracted videos in {frames_dir}. Run drowsyguard_ml.prepare first.")
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for chunk in pool.map(_one, paths, chunksize=2) for r in chunk]
    table = pd.DataFrame(rows)
    table["cv_fold"] = assign_folds(table)
    table["partition"] = partition(table["cv_fold"])
    assert_subject_independent(table)
    missing = set(FEATURE_NAMES) - set(table.columns)
    if missing:
        raise AssertionError(f"missing features: {missing}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    table = build(workers=args.workers)
    table.to_parquet(FEATURES_PATH, index=False)
    summary = table.groupby(["partition", "label"]).agg(
        windows=("label", "size"), videos=("video_key", "nunique"), subjects=("subject_key", "nunique")
    )
    print(summary.to_string())
    print(f"\nwrote {FEATURES_PATH}")


if __name__ == "__main__":
    main()
