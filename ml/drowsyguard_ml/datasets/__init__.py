"""Dataset adapters. Each yields ``VideoItem``s; nothing else knows a dataset's layout."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from . import nthu_ddd, uta_rldd, yawdd
from .base import VideoItem


def load_items(
    dataset: str,
    root: Optional[Path],
    stream: bool = False,
    subjects_per_fold: Optional[int] = None,
    include_low_vigilant: bool = False,
) -> List[VideoItem]:
    if dataset == "rldd":
        labels = ("0", "5", "10") if include_low_vigilant else ("0", "10")
        items = uta_rldd.scan_remote(labels) if stream else uta_rldd.scan_local(root, labels)
        return uta_rldd.balanced_subset(items, subjects_per_fold)
    if stream:
        raise ValueError("--stream is only available for rldd")
    if dataset == "nthu":
        return nthu_ddd.scan_local(root)
    if dataset == "yawdd":
        return yawdd.scan_local(root)
    raise ValueError(f"unknown dataset {dataset}")
