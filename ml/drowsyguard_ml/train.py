"""Stage 3: train, compare, evaluate and save.

    python -m drowsyguard_ml.train

Protocol, all subject-independent:
  1. Fit every candidate on the train subjects, score it on the val subjects.
  2. Pick the best by validation ROC-AUC, and its decision threshold by val F1.
  3. Refit that configuration on train + val.
  4. Score the test subjects once. Nothing is tuned after this point.
  5. Cross-subject check: rerun the chosen configuration leaving out each
     official fold in turn, with its threshold fixed from step 2.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .build_features import FEATURES_PATH
from .config import ARTIFACTS_DIR, DEFAULT_FEATURE_CONFIG
from .datasets.base import ALERT, DROWSY, LOW_VIGILANT
from .evaluate import best_f1_threshold, binary_metrics, per_subject, summarise, video_level
from .export import export_model
from .features import FEATURE_NAMES
from .splits import N_FOLDS, assert_subject_independent

SEED = 13
RF_GRID = [
    {"max_depth": 8, "min_samples_leaf": 10},
    {"max_depth": 10, "min_samples_leaf": 5},
    {"max_depth": 12, "min_samples_leaf": 20},
]
XGB_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 150},
]


def make_model(kind: str, params: Dict, n_jobs: int):
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", max_features="sqrt",
            random_state=SEED, n_jobs=n_jobs, **params,
        )
    return XGBClassifier(
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
        eval_metric="logloss", random_state=SEED, n_jobs=n_jobs, **params,
    )


def load_table(low_vigilant: str) -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise SystemExit("ml/data/features.parquet is missing. Run prepare, then build_features.")
    table = pd.read_parquet(FEATURES_PATH)
    if low_vigilant == "drop":
        table = table[table.label != LOW_VIGILANT]
    table = table.assign(y=table.label.isin([DROWSY, LOW_VIGILANT]).astype(int)).reset_index(drop=True)
    assert set(table.label.unique()) <= {ALERT, LOW_VIGILANT, DROWSY}
    assert_subject_independent(table)
    return table


def rule_baseline(frame: pd.DataFrame) -> Dict:
    """The app's existing PERCLOS limit, scored on the same windows, for an honest comparison."""
    limit = 15.0  # drowsyPerclos in src/lib/config.ts
    score = frame.perclos.to_numpy()
    metrics = binary_metrics(frame.y.to_numpy(), score / 100.0, limit / 100.0)
    return {"rule": f"PERCLOS >= {limit:.0f}%", **metrics}


def describe(frame: pd.DataFrame) -> Dict:
    return {
        "subjects": int(frame.subject_key.nunique()),
        "videos": int(frame.video_key.nunique()),
        "windows": int(len(frame)),
        "drowsy_windows": int(frame.y.sum()),
        "alert_windows": int((1 - frame.y).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--low-vigilant", choices=["drop", "drowsy"], default="drop",
                        help="RLDD's middle class is ambiguous; dropped unless it was extracted and asked for")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--min-subjects-per-partition", type=int, default=3)
    args = parser.parse_args()

    table = load_table(args.low_vigilant)
    parts = {name: table[table.partition == name] for name in ("train", "val", "test")}
    for name, frame in parts.items():
        count = frame.subject_key.nunique()
        if count < args.min_subjects_per_partition or frame.y.nunique() < 2:
            raise SystemExit(f"Not enough data to train honestly: {name} has {count} subjects, "
                             f"classes {sorted(frame.y.unique())}.")
    print({k: describe(v) for k, v in parts.items()})

    X = lambda f: f[FEATURE_NAMES].to_numpy(dtype=np.float32)  # noqa: E731
    candidates: List[Dict] = []
    for kind, grid in (("random_forest", RF_GRID), ("xgboost", XGB_GRID)):
        for params in grid:
            start = time.time()
            model = make_model(kind, params, args.jobs).fit(X(parts["train"]), parts["train"].y)
            val_proba = model.predict_proba(X(parts["val"]))[:, 1]
            threshold = best_f1_threshold(parts["val"].y.to_numpy(), val_proba)
            metrics = binary_metrics(parts["val"].y.to_numpy(), val_proba, threshold)
            candidates.append({"kind": kind, "params": params, "threshold": threshold,
                               "fit_seconds": round(time.time() - start, 2), "val": metrics})
            print(f"{kind:14s} {params} val AUC {metrics['roc_auc']:.3f} F1 {metrics['f1']:.3f} @ {threshold}")

    best = max(candidates, key=lambda c: c["val"]["roc_auc"])
    print(f"selected {best['kind']} {best['params']}")

    fit_frame = pd.concat([parts["train"], parts["val"]])
    final = make_model(best["kind"], best["params"], args.jobs).fit(X(fit_frame), fit_frame.y)
    test = parts["test"]
    test_proba = final.predict_proba(X(test))[:, 1]
    threshold = best["threshold"]
    test_metrics = binary_metrics(test.y.to_numpy(), test_proba, threshold)
    print(f"TEST window AUC {test_metrics['roc_auc']:.3f} acc {test_metrics['accuracy']:.3f} F1 {test_metrics['f1']:.3f}")

    folds = []
    for fold in range(N_FOLDS):
        held = table[table.cv_fold == fold]
        rest = table[table.cv_fold != fold]
        if held.empty or held.y.nunique() < 2:
            continue
        assert not set(held.subject_key) & set(rest.subject_key)
        model = make_model(best["kind"], best["params"], args.jobs).fit(X(rest), rest.y)
        proba = model.predict_proba(X(held))[:, 1]
        folds.append({"fold": fold + 1, "subjects": int(held.subject_key.nunique()),
                      "window": binary_metrics(held.y.to_numpy(), proba, threshold),
                      "video": video_level(held, proba, threshold)})

    names = dict(zip(FEATURE_NAMES, final.feature_importances_.tolist()))
    importance = sorted(({"feature": k, "importance": v} for k, v in names.items()),
                        key=lambda r: r["importance"], reverse=True)

    report = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "datasets": sorted(table.dataset.unique().tolist()),
        "label_scheme": {"alert": "RLDD 0", "drowsy": "RLDD 10", "low_vigilant": args.low_vigilant},
        "data": {**{k: describe(v) for k, v in parts.items()}, "all": describe(table)},
        "model": {"kind": best["kind"], "params": best["params"], "threshold": threshold},
        "candidates": candidates,
        "test": {
            "window": test_metrics,
            "video": video_level(test, test_proba, threshold),
            "per_subject": per_subject(test, test_proba, threshold),
            "rule_baseline": rule_baseline(test),
        },
        "cross_subject": {
            "protocol": "Leave one official fold out; configuration and threshold fixed from train/val. "
                        "The test fold is included here, so this confirms stability and is not a second "
                        "untouched estimate.",
            "folds": folds,
            "window": {m: summarise([f["window"][m] for f in folds])
                       for m in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            "video": {m: summarise([f["video"][m] for f in folds])
                      for m in ("accuracy", "precision", "recall", "f1", "roc_auc")},
        },
        "feature_importance": importance,
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "kind": best["kind"], "feature_names": FEATURE_NAMES, "threshold": threshold,
                 "feature_config": DEFAULT_FEATURE_CONFIG.to_dict(), "trained_at": report["trained_at"]},
                ARTIFACTS_DIR / "model.joblib")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(report, indent=1))
    (ARTIFACTS_DIR / "preprocessing.json").write_text(json.dumps(
        {"feature_config": DEFAULT_FEATURE_CONFIG.to_dict(), "landmarker": "face_landmarker.task (float16/1)",
         "splits": table.groupby("partition").subject_key.unique().map(sorted).map(list).to_dict()}, indent=1))
    (ARTIFACTS_DIR / "feature_metadata.json").write_text(json.dumps(
        {"feature_names": FEATURE_NAMES, "dtype": "float32", "order_matters": True}, indent=1))

    export_model(final, best["kind"], threshold, report, X(test), test_proba)
    print(f"saved model and metrics to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
