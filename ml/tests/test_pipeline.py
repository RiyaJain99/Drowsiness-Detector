"""Run with: cd ml && .venv/bin/python -m pytest -q

Series here are test inputs only. They exercise logic and are never saved as data.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from drowsyguard_ml.config import DEFAULT_FEATURE_CONFIG as CFG
from drowsyguard_ml.datasets import nthu_ddd, uta_rldd, yawdd
from drowsyguard_ml.datasets.base import VideoItem
from drowsyguard_ml.export import _boosted, _forest, evaluate
from drowsyguard_ml.features import FEATURE_NAMES, Calibration, _percentile, trimmed_mean, window_features
from drowsyguard_ml.parity import OUT, scripted_series
from drowsyguard_ml.splits import TEST_FOLD, assert_subject_independent, assign_folds, partition

CAL = Calibration(ear=0.3, mar=0.05, pitch=0.0, yaw=0.0, roll=0.0)
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def series(seconds=60, ear=lambda s: 0.3, mar=lambda s: 0.05, pitch=lambda s: 0.0):
    t = [i * 1000 / 15 for i in range(seconds * 15)]
    e = [ear(x / 1000) for x in t]
    return (t, e, e, e, [mar(x / 1000) for x in t], [pitch(x / 1000) for x in t], [0.0] * len(t), [0.0] * len(t))


def test_open_eyes_have_no_events():
    f = window_features(*series(), CAL)
    assert f[IDX["perclos"]] == 0 and f[IDX["blink_rate"]] == 0 and f[IDX["closure_max_ms"]] == 0
    assert f[IDX["ear_rel_mean"]] == pytest.approx(1)


def test_perclos_matches_closed_share():
    f = window_features(*series(ear=lambda s: 0.1 if s < 15 else 0.3), CAL)
    assert f[IDX["perclos"]] == pytest.approx(25, abs=0.2)
    assert f[IDX["perclos_trend"]] == pytest.approx(-50, abs=0.5)


def test_blinks_counted_per_minute():
    f = window_features(*series(ear=lambda s: 0.1 if s % 5 < 0.2 else 0.3), CAL)
    assert f[IDX["blink_rate"]] == pytest.approx(12, abs=0.3)
    assert f[IDX["long_closure_rate"]] == 0


def test_long_closure_is_not_a_blink():
    f = window_features(*series(ear=lambda s: 0.1 if 20 < s < 22 else 0.3), CAL)
    assert f[IDX["blink_rate"]] == 0
    assert f[IDX["long_closure_rate"]] == pytest.approx(1, abs=0.05)
    assert f[IDX["closure_max_ms"]] == pytest.approx(2000, abs=100)


def test_yawn_needs_the_hold_time():
    short = window_features(*series(mar=lambda s: 0.8 if 20 < s < 20.5 else 0.05), CAL)
    held = window_features(*series(mar=lambda s: 0.8 if 20 < s < 22 else 0.05), CAL)
    assert short[IDX["yawn_rate"]] == 0
    assert held[IDX["yawn_rate"]] == pytest.approx(1, abs=0.05)


def test_nod_needs_the_hold_time():
    held = window_features(*series(pitch=lambda s: -25 if 20 < s < 22 else 0), CAL)
    assert held[IDX["nod_rate"]] == pytest.approx(1, abs=0.05)
    assert held[IDX["pitch_rel_min"]] == -25


def test_tracking_gap_is_not_a_closure():
    t, e, l, r, m, p, y, ro = series()
    nan = float("nan")
    for i in range(300, 400):
        e[i] = l[i] = r[i] = m[i] = p[i] = y[i] = ro[i] = nan
    f = window_features(t, e, l, r, m, p, y, ro, CAL)
    assert f[IDX["perclos"]] == 0
    assert f[IDX["visible_frac"]] == pytest.approx(800 / 900)
    assert all(math.isfinite(v) for v in f)


def test_mostly_missing_window_is_rejected():
    t, e, l, r, m, p, y, ro = series()
    nan = float("nan")
    for i in range(0, 600):
        e[i] = nan
    assert window_features(t, e, l, r, m, p, y, ro, CAL) is None


def test_short_window_is_rejected():
    assert window_features(*series(seconds=20), CAL) is None


def test_helpers_match_numpy():
    values = [3.0, 1.0, 2.0, 9.0, 4.0]
    assert _percentile(sorted(values), 0.1) == pytest.approx(np.percentile(values, 10))
    assert trimmed_mean(np.array([0.0, 1, 1, 1, 100]), 0.2) == 1


def test_feature_vector_length():
    assert len(window_features(*series(), CAL)) == len(FEATURE_NAMES)


def test_parity_fixture_is_current():
    fixture = json.loads(OUT.read_text())
    assert fixture["feature_names"] == FEATURE_NAMES
    assert fixture["feature_config"] == CFG.to_dict()
    scripted = scripted_series()
    assert fixture["cases"][0]["series"]["t_ms"] == scripted["t_ms"]


def _table():
    rows = []
    for subject in range(20):
        for label in (0, 2):
            rows.append({"subject_key": f"rldd:{subject:02d}", "fold": subject % 5, "label": label})
            rows.append({"subject_key": f"yawdd:m{subject:02d}", "fold": None, "label": label})
    return pd.DataFrame(rows)


def test_official_folds_are_kept_and_subjects_never_leak():
    table = _table()
    table["cv_fold"] = assign_folds(table)
    table["partition"] = partition(table["cv_fold"])
    rldd = table[table.subject_key.str.startswith("rldd")]
    assert (rldd.cv_fold == rldd.fold).all()
    assert (table[table.cv_fold == TEST_FOLD].partition == "test").all()
    assert_subject_independent(table)


def test_leak_detector_fires():
    table = pd.DataFrame({"subject_key": ["a", "a"], "partition": ["train", "test"]})
    with pytest.raises(AssertionError):
        assert_subject_independent(table)


def test_rldd_paths_parse():
    assert uta_rldd._parse("Fold1_part1/06/10.mp4") == (0, "06", "10", None)
    assert uta_rldd._parse("Fold5_part1/49/10_2.mp4") == (4, "49", "10", "2")
    assert uta_rldd._parse("Fold1_part1/03/0.MOV") == (0, "03", "0", None)
    assert uta_rldd._parse("__MACOSX/Fold1_part1/03/0.MOV") is None


def test_balanced_subset_is_round_robin():
    items = [VideoItem("rldd", f"{f}{s}", "0", 0, fold=f, size_bytes=s + 1) for f in range(5) for s in range(4)]
    picked = uta_rldd.balanced_subset(items, 2)
    assert [i.fold for i in picked[:5]] == [0, 1, 2, 3, 4]
    assert len(picked) == 10


def test_nthu_and_yawdd_layouts(tmp_path):
    video = tmp_path / "001" / "glasses" / "yawning.avi"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    (video.parent / "001_yawning_drowsiness.txt").write_text("0011")
    (item,) = nthu_ddd.scan_local(tmp_path)
    assert item.subject == "001" and item.frame_labels.tolist() == [0, 0, 2, 2]

    other = tmp_path / "y"
    other.mkdir()
    (other / "12-FemaleNoGlasses-Yawning.avi").write_bytes(b"x")
    (other / "12-FemaleNoGlasses-Talking.avi").write_bytes(b"x")
    (other / "notes.avi").write_bytes(b"x")
    labels = {i.video_id.split("-")[-1]: i.label for i in yawdd.scan_local(other)}
    assert labels == {"Yawning": 2, "Talking": 0}


def _xy():
    grid = np.linspace(0, 1, 200, dtype=np.float32)
    X = np.stack([grid, grid[::-1] ** 2], axis=1)
    return X, (grid > 0.5).astype(int)


def test_forest_export_reproduces_sklearn():
    X, y = _xy()
    model = RandomForestClassifier(n_estimators=7, max_depth=4, random_state=0).fit(X, y)
    assert np.allclose(evaluate(_forest(model), X), model.predict_proba(X)[:, 1], atol=1e-5)


def test_xgboost_export_reproduces_xgboost():
    X, y = _xy()
    model = XGBClassifier(n_estimators=9, max_depth=3).fit(X, y)
    assert np.allclose(evaluate(_boosted(model, X), X), model.predict_proba(X)[:, 1], atol=1e-5)
