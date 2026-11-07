"""NTHU Driver Drowsiness Detection dataset (access by request from NTHU CVLab).

Expected layout, as distributed::

    <root>/<subject>/<scenario>/<behaviour>.avi
    <root>/<subject>/<scenario>/<subject>_<behaviour>_drowsiness.txt

The text file holds one 0/1 digit per frame. Drowsiness here is acted, not
spontaneous, which is why RLDD stays the primary dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import ALERT, DROWSY, VideoItem

NAME = "nthu"


def _labels_for(video: Path) -> Optional[np.ndarray]:
    matches = sorted(video.parent.glob(f"*{video.stem}*drowsiness*.txt"))
    if not matches:
        return None
    digits = [c for c in matches[0].read_text().strip() if c in "01"]
    if not digits:
        return None
    return np.where(np.array(digits) == "1", DROWSY, ALERT).astype(np.int8)


def scan_local(root: Path) -> List[VideoItem]:
    items = []
    for video in sorted(root.rglob("*.avi")) + sorted(root.rglob("*.mp4")):
        labels = _labels_for(video)
        if labels is None:
            continue  # unannotated test videos cannot be used for supervised training
        rel = video.relative_to(root).parts
        items.append(
            VideoItem(NAME, rel[-3] if len(rel) >= 3 else video.parent.name,
                      "-".join(p.replace(" ", "_") for p in rel[-2:]).rsplit(".", 1)[0],
                      None, fetch=lambda p=video: p, size_bytes=video.stat().st_size, frame_labels=labels)
        )
    return items
