"""Common shape for every dataset adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Canonical state ids. Each adapter maps its own labels onto these.
ALERT, LOW_VIGILANT, DROWSY = 0, 1, 2


@dataclass
class VideoItem:
    dataset: str
    subject: str  # unique within the dataset; never crosses a split
    video_id: str
    # Whole-video label, or None when frame_labels supplies one per frame.
    label: Optional[int]
    # Subject-level fold id. RLDD's official folds; None lets splits.py assign one.
    fold: Optional[int] = None
    # Returns the path to a playable local file, fetching it first if needed.
    fetch: Callable = None
    # True when `fetch` produced a temporary copy that may be deleted afterwards.
    temporary: bool = False
    size_bytes: int = 0
    # Per-frame labels (NTHU-DDD), indexed by source frame number.
    frame_labels: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.subject}/{self.video_id}"
