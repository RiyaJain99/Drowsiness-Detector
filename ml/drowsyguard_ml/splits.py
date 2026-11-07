"""Subject-independent splits. A subject's windows never straddle two partitions."""
from __future__ import annotations

import hashlib
from typing import Dict

import pandas as pd

N_FOLDS = 5
VAL_FOLD = 3  # RLDD Fold4
TEST_FOLD = 4  # RLDD Fold5


def _hashed_fold(subject_key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{subject_key}".encode()).hexdigest()
    return int(digest, 16) % N_FOLDS


def assign_folds(table: pd.DataFrame, seed: int = 13) -> pd.Series:
    """Fold per row: RLDD's official folds where present, otherwise a stable hash of the subject."""
    def fold_of(row) -> int:
        if pd.notna(row.fold):
            return int(row.fold)
        return _hashed_fold(row.subject_key, seed)

    subjects = table.drop_duplicates("subject_key")
    mapping: Dict[str, int] = {r.subject_key: fold_of(r) for r in subjects.itertuples()}
    return table["subject_key"].map(mapping)


def partition(cv_fold: pd.Series) -> pd.Series:
    return cv_fold.map(lambda f: "test" if f == TEST_FOLD else "val" if f == VAL_FOLD else "train")


def assert_subject_independent(table: pd.DataFrame) -> None:
    """Raises if any subject appears in more than one partition."""
    spread = table.groupby("subject_key")["partition"].nunique()
    leaked = spread[spread > 1]
    if len(leaked):
        raise AssertionError(f"subjects in several partitions: {list(leaked.index)}")
