"""UTA Real-Life Drowsiness Dataset (Ghoddoosian, Galib, Athitsos; CVPRW 2019).

60 participants, roughly 10 minutes each of alert (0), low vigilant (5) and
drowsy (10), laid out as ``FoldN_partM/<subject>/<label>[_k].<ext>``. Labels are
the participant's own report of their predominant state for the whole video.
"""
from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Iterable, List, Optional

from ..config import TMP_DIR
from .base import ALERT, DROWSY, LOW_VIGILANT, VideoItem
from .remote_zip import fetch_member, list_members

NAME = "rldd"
LABELS = {"0": ALERT, "5": LOW_VIGILANT, "10": DROWSY}
EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
PATTERN = re.compile(r"Fold(\d)[^/]*/(\d+)/(0|5|10)(?:_(\d+))?\.\w+$", re.IGNORECASE)

# The public folder linked from https://sites.google.com/view/utarldd/home
DRIVE_FILES = {
    "Fold1_part1": "16x8afSOwfbdGAjUbfRFvwGj1ie8Lf5Do",
    "Fold1_part2": "1qPoJ06jeznytP_BGJ25hxdQy7eYTTjyk",
    "Fold2_part1": "12uVMVLv_kRXzQb4QDdru9RK3G9hzfpPQ",
    "Fold2_part2": "1nYGUpqE3x5ZeRLgDeMlUN4K_ccPjM2pe",
    "Fold3_part1": "1-JhE5TfkNyr4UbQk6PSkkpI7z-SChkWS",
    "Fold3_part2": "1LZU5KfJkFMj2pIkGwb3RjHDazoUfxSYQ",
    "Fold4_part1": "1IYKdrjvawf90bG_cfeoWA4rZSymllFbD",
    "Fold4_part2": "1zJlASBwABBgVSULQLiia0R_Eqar8_B3D",
    "Fold5_part1": "16w30wEgcAbaUr4gx9o4sl_2TbIunD5yA",
    "Fold5_part2": "1loOQE1-hoNM5dIOjXHI2Ss5NjjoPZcX6",
}


def drive_url(file_id: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _parse(path: str) -> Optional[tuple]:
    match = PATTERN.search(path.replace("\\", "/"))
    if not match or "__MACOSX" in path:
        return None
    fold, subject, label, part = match.groups()
    return int(fold) - 1, subject.zfill(2), label, part


def _video_id(label: str, part: Optional[str]) -> str:
    return f"{label}_{part}" if part else label


def scan_local(root: Path, labels: Iterable[str]) -> List[VideoItem]:
    """Videos already on disk, from unpacked zips."""
    items = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        parsed = _parse(path.relative_to(root).as_posix())
        if not parsed or parsed[2] not in labels:
            continue
        fold, subject, label, part = parsed
        items.append(
            VideoItem(NAME, subject, _video_id(label, part), LABELS[label], fold=fold,
                      fetch=lambda p=path: p, size_bytes=path.stat().st_size)
        )
    return items


def _download(url: str, info, dest: Path) -> Path:
    fetch_member(url, info, dest)
    return dest


def scan_remote(labels: Iterable[str]) -> List[VideoItem]:
    """Every video in the public archives, without downloading any of them."""
    items = []
    for archive, file_id in DRIVE_FILES.items():
        url = drive_url(file_id)
        for info in list_members(url):
            parsed = _parse(info.filename)
            if info.is_dir() or not parsed or parsed[2] not in labels:
                continue
            fold, subject, label, part = parsed
            vid = _video_id(label, part)
            dest = TMP_DIR / f"{NAME}-{subject}-{vid}" / f"video{Path(info.filename).suffix.lower()}"
            items.append(
                VideoItem(NAME, subject, vid, LABELS[label], fold=fold, temporary=True,
                          fetch=partial(_download, url, info, dest), size_bytes=info.compress_size,
                          meta={"archive": archive, "member": info.filename})
            )
    return items


def balanced_subset(items: List[VideoItem], subjects_per_fold: Optional[int]) -> List[VideoItem]:
    """Queue subjects smallest-download-first, round-robin across folds.

    Interrupting the queue at any point therefore leaves every official fold
    about equally represented, with each subject's videos complete. Ordering by
    size does skew a partial run towards lower-bitrate recordings; ``None`` keeps
    all subjects.
    """
    subjects = {}
    for item in items:
        subjects.setdefault((item.fold, item.subject), []).append(item)
    by_fold = {}
    for (fold, _), videos in sorted(subjects.items(), key=lambda kv: sum(v.size_bytes for v in kv[1])):
        by_fold.setdefault(fold, []).append(videos)

    ordered = []
    rank = 0
    while subjects_per_fold is None or rank < subjects_per_fold:
        row = [fold[rank] for fold in by_fold.values() if rank < len(fold)]
        if not row:
            break
        ordered.extend(video for videos in row for video in videos)
        rank += 1
    return ordered
