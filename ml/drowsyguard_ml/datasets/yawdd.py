"""YawDD (Abtahi et al., MMSys 2014): short clips of drivers yawning, talking or neither.

File names look like ``12-FemaleNoGlasses-Yawning.avi`` (mirror camera) or
``12-Female.avi`` style dash clips; only names stating a behaviour are used.

YawDD labels a *behaviour*, not a fatigue level, so treating "Yawning" as drowsy
is a weak label. It is an auxiliary source and never the test set on its own.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .base import ALERT, DROWSY, VideoItem

NAME = "yawdd"
PATTERN = re.compile(r"^(\d+)-(Female|Male)[A-Za-z]*-(Normal|Talking|Yawning)", re.IGNORECASE)


def scan_local(root: Path) -> List[VideoItem]:
    items = []
    for video in sorted(root.rglob("*.avi")) + sorted(root.rglob("*.mp4")):
        match = PATTERN.match(video.name)
        if not match:
            continue
        number, sex, behaviour = match.groups()
        items.append(
            VideoItem(NAME, f"{sex.lower()}{int(number):02d}", video.stem.replace(" ", "_"),
                      DROWSY if behaviour.lower() == "yawning" else ALERT,
                      fetch=lambda p=video: p, size_bytes=video.stat().st_size)
        )
    return items
