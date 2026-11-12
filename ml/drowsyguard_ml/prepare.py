"""Stage 1: turn dataset videos into cached per-frame tables.

    python -m drowsyguard_ml.prepare --dataset rldd --root /path/to/RLDD
    python -m drowsyguard_ml.prepare --dataset rldd --stream --subjects-per-fold 6

Each finished video becomes ``ml/data/frames/<dataset>/<subject>/<video>.parquet``
plus a ``.json`` sidecar, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Optional

from .config import DATA_DIR, FRAMES_DIR, TMP_DIR
from .datasets import load_items
from .datasets.base import VideoItem


def output_path(item: VideoItem) -> Path:
    return FRAMES_DIR / item.dataset / item.subject / f"{item.video_id}.parquet"


def _process(video: str, out: str, meta: dict, save_frames: Optional[str], frame_labels) -> dict:
    """Runs in a worker process; imports MediaPipe there, not in the parent."""
    from .extract import extract_video

    frames, info = extract_video(Path(video), save_frames_dir=Path(save_frames) if save_frames else None)
    if frame_labels is not None:
        fps = info["source_fps"] or 30.0
        idx = (frames.t_ms / 1000 * fps).round().astype(int).clip(0, len(frame_labels) - 1)
        frames["frame_label"] = frame_labels[idx.to_numpy()]
    else:
        frames["frame_label"] = meta["label"]
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    frames.to_parquet(tmp, index=False)
    tmp.rename(target)
    record = {**meta, **info}
    target.with_suffix(".json").write_text(json.dumps(record, indent=1))
    return record


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(items: list, workers: int, max_on_disk: int, save_frames: bool) -> int:
    todo = [i for i in items if not output_path(i).exists()]
    log(f"{len(items)} videos selected, {len(items) - len(todo)} already done, {len(todo)} to process")
    failures = 0
    finished = 0
    downloads, extractions = {}, {}
    with ThreadPoolExecutor(1) as fetcher, ProcessPoolExecutor(workers) as pool:
        queue = list(todo)
        while queue or downloads or extractions:
            # One download at a time: the line is the limit, not the connection count.
            while queue and not downloads and len(extractions) < max_on_disk:
                item = queue.pop(0)
                downloads[fetcher.submit(item.fetch)] = item
            done, _ = wait(list(downloads) + list(extractions), return_when=FIRST_COMPLETED)
            for future in done:
                if future in downloads:
                    item = downloads.pop(future)
                    try:
                        path = future.result()
                    except Exception as err:  # noqa: BLE001 - one bad video must not end the run
                        failures += 1
                        log(f"FETCH FAILED {item.key}: {err}")
                        if item.temporary:
                            shutil.rmtree(TMP_DIR / f"{item.dataset}-{item.subject}-{item.video_id}", ignore_errors=True)
                        continue
                    meta = {"dataset": item.dataset, "subject": item.subject, "video_id": item.video_id,
                            "label": item.label, "fold": item.fold, **item.meta}
                    frames_dir = str(DATA_DIR / "frames_jpg" / item.key) if save_frames else None
                    job = pool.submit(_process, str(path), str(output_path(item)), meta, frames_dir, item.frame_labels)
                    extractions[job] = (item, path)
                else:
                    item, path = extractions.pop(future)
                    try:
                        record = future.result()
                        finished += 1
                        log(f"done {item.key} {record['duration_s']:.0f}s face={record['face_frac']:.2f} "
                            f"({finished}/{len(todo)})")
                    except Exception as err:  # noqa: BLE001
                        failures += 1
                        log(f"EXTRACT FAILED {item.key}: {err}")
                    if item.temporary:
                        shutil.rmtree(Path(path).parent, ignore_errors=True)
    log(f"finished: {finished} ok, {failures} failed")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["rldd", "nthu", "yawdd"])
    parser.add_argument("--root", type=Path, help="folder holding the unpacked dataset")
    parser.add_argument("--stream", action="store_true", help="RLDD only: fetch videos one by one, delete after use")
    parser.add_argument("--subjects-per-fold", type=int, help="RLDD only: limit to N subjects in each official fold")
    parser.add_argument("--include-low-vigilant", action="store_true", help="RLDD only: also process label 5 videos")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-on-disk", type=int, default=3, help="downloaded videos held at once when streaming")
    parser.add_argument("--save-frames", action="store_true", help="also keep sampled JPEGs (see extract.py)")
    args = parser.parse_args()

    if not args.stream and args.root is None:
        parser.error("give --root, or --stream for RLDD")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    items = load_items(args.dataset, args.root, args.stream, args.subjects_per_fold, args.include_low_vigilant)
    if not items:
        raise SystemExit(f"No videos found for {args.dataset}. Check the folder layout in ml/README.md.")
    raise SystemExit(1 if run(items, args.workers, args.max_on_disk, args.save_frames) else 0)


if __name__ == "__main__":
    main()
