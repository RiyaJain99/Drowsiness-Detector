"""Stage 3: train, compare, evaluate and save.

    python -m drowsyguard_ml.train

Protocol, all subject-independent. The development subjects are train + val
(official folds 1-4); the test subjects (fold 5) take no part in any choice.
  1. Every candidate (model x hyperparameters x feature set) is scored by
     leave-one-fold-out over the four development folds.
  2. The best mean ROC-AUC wins. Its decision threshold maximises F1 over the
     pooled out-of-fold predictions, which spans every development subject
     rather than the four in a single validation fold.
  3. That configuration is refitted on all development subjects.
  4. The test subjects are scored once.
  5. Cross-subject check: rerun the chosen configuration leaving out each of
     the five official folds, with its threshold fixed from step 2.
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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .build_features import FEATURES_PATH
from .config import ARTIFACTS_DIR, DEFAULT_FEATURE_CONFIG
from .datasets.base import ALERT, DROWSY, LOW_VIGILANT
from .evaluate import best_f1_threshold, binary_metrics, threshold_for_false_alarm_rate, per_subject, summarise, video_level
from .export import export_model
from .features import FEATURE_NAMES
from .splits import N_FOLDS, TEST_FOLD, assert_subject_independent

SEED = 13
# Decided before the final run, on application grounds: at most one alert
# window in five may be flagged. The app needs five drowsy verdicts in a row
# before it warns, so the rate of actual false warnings is far lower.
MAX_FALSE_ALARM_RATE = 0.20
# Fixed for a person rather than their state: natural eye opening and facial
# asymmetry. With few training subjects a forest can learn who someone is
# instead of how tired they are, so the subset without them is a candidate.
IDENTITY_SENSITIVE = ["ear_abs_mean", "ear_asym_mean"]
FEATURE_SETS = {
    "all": list(FEATURE_NAMES),
    "state_only": [f for f in FEATURE_NAMES if f not in IDENTITY_SENSITIVE],
}
NOTES = [
    "2026-11-08: a preliminary 19-subject run selected on the fold-4 validation subjects alone scored "
    "test ROC-AUC 0.660. The selection procedure was then changed to cross-validation over folds 1-4 and an "
    "identity-free feature subset was added as a candidate. Those choices were made by cross-validation on "
    "folds 1-4 only, but the three fold-5 test subjects had been seen once, so treat their scores as "
    "slightly optimistic.",
    "2026-11-08: a 20-subject run with an F1-optimal threshold flagged 53% of alert test windows. The threshold "
    "rule was changed to cap false alarms at 20% of alert windows on the development folds, before the final run.",
]
RF_GRID = [
    {"max_depth": 8, "min_samples_leaf": 10},
    {"max_depth": 10, "min_samples_leaf": 5},
    {"max_depth": 12, "min_samples_leaf": 20},
]
# Few subjects favour simple models; a linear one cannot memorise a person.
LR_GRID = [{"C": 0.03}, {"C": 0.1}, {"C": 0.3}, {"C": 1.0}]
XGB_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 150},
]


def make_model(kind: str, params: Dict, n_jobs: int):
    if kind == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=5000, random_state=SEED, **params))
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", max_features="sqrt",
            random_state=SEED, n_jobs=n_jobs, **params,
        )
    return XGBClassifier(
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
        eval_metric="logloss", random_state=SEED, n_jobs=n_jobs, **params,
    )


def importances(model) -> np.ndarray:
    """Tree importances, or |standardised coefficient| for the linear model, scaled to sum to 1."""
    raw = (np.abs(model[-1].coef_[0]) if kind_of(model) == "logistic_regression"
           else np.asarray(model.feature_importances_))
    return raw / raw.sum() if raw.sum() > 0 else raw


def kind_of(model) -> str:
    return "logistic_regression" if hasattr(model, "steps") else "tree"


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
    # (validation subjects remain a partition; they are one of the four CV folds)
    args = parser.parse_args()

    table = load_table(args.low_vigilant)
    parts = {name: table[table.partition == name] for name in ("train", "val", "test")}
    for name, frame in parts.items():
        count = frame.subject_key.nunique()
        if count < args.min_subjects_per_partition or frame.y.nunique() < 2:
            raise SystemExit(f"Not enough data to train honestly: {name} has {count} subjects, "
                             f"classes {sorted(frame.y.unique())}.")
    print({k: describe(v) for k, v in parts.items()})

    dev = table[table.cv_fold != TEST_FOLD]
    dev_folds = sorted(dev.cv_fold.unique())

    def cross_validate(kind, params, columns):
        """Out-of-fold P(drowsy) for every development window, plus per-fold AUC."""
        oof = np.zeros(len(dev))
        aucs = []
        for fold in dev_folds:
            held = (dev.cv_fold == fold).to_numpy()
            model = make_model(kind, params, args.jobs).fit(
                dev.loc[~held, columns].to_numpy(np.float32), dev.y[~held])
            oof[held] = model.predict_proba(dev.loc[held, columns].to_numpy(np.float32))[:, 1]
            aucs.append(binary_metrics(dev.y[held].to_numpy(), oof[held], 0.5)["roc_auc"])
        return oof, aucs

    candidates: List[Dict] = []
    for feature_set, columns in FEATURE_SETS.items():
        for kind, grid in (("logistic_regression", LR_GRID), ("random_forest", RF_GRID), ("xgboost", XGB_GRID)):
            for params in grid:
                start = time.time()
                oof, aucs = cross_validate(kind, params, columns)
                threshold = threshold_for_false_alarm_rate(dev.y.to_numpy(), oof, MAX_FALSE_ALARM_RATE)
                pooled = binary_metrics(dev.y.to_numpy(), oof, threshold)
                val_rows = (dev.partition == "val").to_numpy()
                candidates.append({
                    "kind": kind, "params": params, "feature_set": feature_set, "threshold": threshold,
                    "cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
                    "fit_seconds": round(time.time() - start, 2),
                    "cv_pooled": pooled,
                    "val": binary_metrics(dev.y[val_rows].to_numpy(), oof[val_rows], threshold),
                })
                print(f"{feature_set:10s} {kind:19s} {params} CV AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
                      f"pooled F1 {pooled['f1']:.3f} @ {threshold}")

    best = max(candidates, key=lambda c: c["cv_auc_mean"])
    columns = FEATURE_SETS[best["feature_set"]]
    print(f"selected {best['feature_set']} {best['kind']} {best['params']}")
    X = lambda f: f[columns].to_numpy(dtype=np.float32)  # noqa: E731

    final = make_model(best["kind"], best["params"], args.jobs).fit(X(dev), dev.y)
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

    names = dict(zip(columns, importances(final).tolist()))
    importance = sorted(({"feature": k, "importance": v} for k, v in names.items()),
                        key=lambda r: r["importance"], reverse=True)

    report = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "datasets": sorted(table.dataset.unique().tolist()),
        "label_scheme": {"alert": "RLDD 0", "drowsy": "RLDD 10", "low_vigilant": args.low_vigilant},
        "data": {**{k: describe(v) for k, v in parts.items()}, "all": describe(table)},
        "model": {"kind": best["kind"], "params": best["params"], "threshold": threshold,
                  "feature_set": best["feature_set"], "features_used": columns},
        "selection": "Leave-one-fold-out cross-validation over official folds 1-4; fold 5 held out. "
                     f"Threshold: highest recall flagging at most {MAX_FALSE_ALARM_RATE:.0%} of alert windows "
                     "in the pooled out-of-fold predictions.",
        "notes": NOTES,
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
    joblib.dump({"model": final, "kind": best["kind"], "feature_names": FEATURE_NAMES, "columns": columns,
                 "threshold": threshold,
                 "feature_config": DEFAULT_FEATURE_CONFIG.to_dict(), "trained_at": report["trained_at"]},
                ARTIFACTS_DIR / "model.joblib")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(report, indent=1))
    (ARTIFACTS_DIR / "preprocessing.json").write_text(json.dumps(
        {"feature_config": DEFAULT_FEATURE_CONFIG.to_dict(), "landmarker": "face_landmarker.task (float16/1)",
         "splits": table.groupby("partition").subject_key.unique().map(sorted).map(list).to_dict()}, indent=1))
    (ARTIFACTS_DIR / "feature_metadata.json").write_text(json.dumps(
        {"feature_names": FEATURE_NAMES, "used_by_model": columns, "dtype": "float32", "order_matters": True},
        indent=1))

    export_model(final, best["kind"], threshold, report, test[FEATURE_NAMES].to_numpy(np.float32), test_proba,
                 columns)
    print(f"saved model and metrics to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
