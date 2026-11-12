"""Write the trained model in the form the browser evaluates.

Trees become flat arrays that ``src/lib/mlModel.ts`` walks directly, which keeps
inference under a millisecond with no runtime dependency or server. Export fails
unless a Python re-implementation of that walk reproduces ``predict_proba``.
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from .config import DEFAULT_FEATURE_CONFIG, PUBLIC_ML_DIR
from .features import FEATURE_NAMES

SCHEMA = 1


def _forest(model) -> Dict:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        value = tree.value[:, 0, :]
        positive = value[:, 1] / value.sum(axis=1)
        trees.append({
            "f": tree.feature.tolist(),  # negative marks a leaf
            "t": tree.threshold.tolist(),
            "l": tree.children_left.tolist(),
            "r": tree.children_right.tolist(),
            "v": [round(float(v), 6) for v in positive],
        })
    # sklearn casts inputs to float32 and goes left on x <= threshold.
    return {"combine": "mean", "strict": False, "bias": 0.0, "trees": trees}


def _boosted(model, sample: np.ndarray) -> Dict:
    # save_raw, not trees_to_dataframe: the latter rounds splits to 6 digits,
    # which sends inputs near a split down the wrong branch.
    raw = json.loads(bytes(model.get_booster().save_raw("json")))
    trees = []
    for tree in raw["learner"]["gradient_booster"]["model"]["trees"]:
        left, right = tree["left_children"], tree["right_children"]
        conditions = [float(np.float32(c)) for c in tree["split_conditions"]]
        leaf = [c < 0 for c in left]
        trees.append({
            "f": [-2 if is_leaf else f for is_leaf, f in zip(leaf, tree["split_indices"])],
            "t": [0.0 if is_leaf else c for is_leaf, c in zip(leaf, conditions)],
            "l": left,
            "r": right,
            "v": [c if is_leaf else 0.0 for is_leaf, c in zip(leaf, conditions)],
        })
    payload = {"combine": "logistic", "strict": True, "bias": 0.0, "trees": trees}
    # The intercept is measured rather than decoded from xgboost's config format.
    margins = model.predict(sample, output_margin=True)
    offsets = margins - _raw_scores(payload, sample)
    if np.ptp(offsets) > 1e-3:
        raise AssertionError("xgboost intercept is not constant; the tree dump is wrong")
    payload["bias"] = float(np.median(offsets))
    return payload


def _linear(pipeline) -> Dict:
    """Fold the scaler into the weights: P = sigmoid(bias + w . x) on raw features."""
    scaler, model = pipeline[0], pipeline[-1]
    coef = model.coef_[0] / scaler.scale_
    bias = float(model.intercept_[0] - np.sum(coef * scaler.mean_))
    return {"combine": "linear", "strict": False, "bias": bias, "trees": [], "weights": coef.tolist()}


def _walk(tree: Dict, row: np.ndarray, strict: bool) -> float:
    i = 0
    while tree["f"][i] >= 0:
        # float() first: NumPy 2 would otherwise round the threshold down to float32.
        x = float(row[tree["f"][i]])
        go_left = x < tree["t"][i] if strict else x <= tree["t"][i]
        i = tree["l"][i] if go_left else tree["r"][i]
    return tree["v"][i]


def _raw_scores(payload: Dict, X: np.ndarray) -> np.ndarray:
    return np.array([sum(_walk(t, row, payload["strict"]) for t in payload["trees"]) for row in X])


def evaluate(payload: Dict, X: np.ndarray) -> np.ndarray:
    """Reference implementation of ``evaluateModel`` in mlModel.ts."""
    if payload["combine"] == "linear":
        z = X.astype(np.float64) @ np.array(payload["weights"]) + payload["bias"]
        return 1 / (1 + np.exp(-z))
    raw = _raw_scores(payload, X)
    if payload["combine"] == "mean":
        return raw / len(payload["trees"])
    return 1 / (1 + np.exp(-(raw + payload["bias"])))


def _remap(payload: Dict, columns: List[str]) -> Dict:
    """Point tree splits at positions in the full feature vector the browser computes."""
    position = [FEATURE_NAMES.index(c) for c in columns]
    if payload["combine"] == "linear":
        full = [0.0] * len(FEATURE_NAMES)
        for weight, index in zip(payload["weights"], position):
            full[index] = weight
        payload["weights"] = full
        return payload
    for tree in payload["trees"]:
        tree["f"] = [position[f] if f >= 0 else f for f in tree["f"]]
    return payload


def export_model(model, kind: str, threshold: float, report: Dict, X_check: np.ndarray, expected: np.ndarray,
                 columns: List[str] = FEATURE_NAMES) -> None:
    """``X_check`` holds full feature vectors; the model may use a subset (``columns``)."""
    sample = X_check[: min(len(X_check), 500)]
    subset = sample[:, [FEATURE_NAMES.index(c) for c in columns]]
    if kind == "logistic_regression":
        payload = _linear(model)
    elif kind == "random_forest":
        payload = _forest(model)
    else:
        payload = _boosted(model, subset)
    payload = _remap(payload, columns)
    drift = float(np.max(np.abs(evaluate(payload, sample) - expected[: len(sample)])))
    if drift > 1e-4:
        raise AssertionError(f"exported model disagrees with the trained one by {drift}")

    PUBLIC_ML_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": SCHEMA,
        "kind": kind,
        "trained_at": report["trained_at"],
        "threshold": threshold,
        "feature_names": FEATURE_NAMES,
        "feature_config": DEFAULT_FEATURE_CONFIG.to_dict(),
        **payload,
    }
    (PUBLIC_ML_DIR / "model.json").write_text(json.dumps(document, separators=(",", ":")))
    (PUBLIC_ML_DIR / "metrics.json").write_text(json.dumps(report, indent=1))
    # Real test rows with the trained model's outputs; vitest replays them through the TS walker.
    check = {"rows": sample[:100].astype(float).tolist(), "expected": expected[:100].tolist()}
    (PUBLIC_ML_DIR / "model-check.json").write_text(json.dumps(check))
