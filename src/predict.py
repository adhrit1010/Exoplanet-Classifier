"""
Inference utilities for the exoplanet classifier.

Loads the serialized ``Pipeline(preprocessor, classifier)`` and offers a friendly
API for the Streamlit app, batch scoring, and the CLI:

    python -m src.predict --input new_kois.csv --output scored.csv

Because the leakage-drop, feature engineering, imputation and scaling all live
*inside* the saved pipeline, callers only need to supply raw KOI columns - any
subset is accepted; missing fields are imputed exactly as during training.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config
from .utils import INT_TO_LABEL, load_json


def load_model(path=config.MODEL_PATH):
    """Load the trained pipeline from disk."""
    return joblib.load(path)


def load_metadata(path=config.METADATA_PATH) -> dict:
    """Load the model metadata sidecar (feature list, metrics, params)."""
    return load_json(path)


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """Score a raw KOI ``DataFrame``.

    Returns a frame with the predicted disposition, per-class probabilities, a
    confidence score (max probability) and a simple 95% confidence interval on
    that probability via a normal approximation across the ensemble where
    available (falls back to a point estimate otherwise).
    """
    proba = model.predict_proba(df)
    pred_int = proba.argmax(axis=1)

    out = pd.DataFrame(
        proba, columns=[f"p_{c}" for c in config.CLASS_ORDER], index=df.index
    )
    out.insert(0, "predicted_disposition", [INT_TO_LABEL[i] for i in pred_int])
    out["confidence"] = proba.max(axis=1)

    lo, hi = _proba_interval(model, df, pred_int)
    if lo is not None:
        out["confidence_low"] = lo
        out["confidence_high"] = hi
    return out


def predict_one(model, features: dict) -> dict:
    """Score a single KOI passed as a ``{column: value}`` dict (Streamlit form)."""
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0]
    pred_int = int(proba.argmax())
    return {
        "predicted_disposition": INT_TO_LABEL[pred_int],
        "confidence": float(proba.max()),
        "probabilities": {c: float(p) for c, p in zip(config.CLASS_ORDER, proba)},
    }


def _proba_interval(model, df: pd.DataFrame, pred_int: np.ndarray):
    """Per-tree probability spread -> rough 95% CI for ensemble models.

    For a forest/booster we read each estimator's vote; the std across trees gives
    an uncertainty band on the winning-class probability. Returns ``(None, None)``
    when the model does not expose per-estimator predictions.
    """
    clf = model.named_steps["classifier"]
    pre = model.named_steps["preprocessor"]
    estimators = getattr(clf, "estimators_", None)
    if estimators is None or not hasattr(clf, "predict_proba"):
        return None, None
    try:
        X_trans = pre.transform(df)
        # Random/Extra forests store per-tree estimators with predict_proba.
        per_tree = []
        for est in np.asarray(estimators).ravel():
            if hasattr(est, "predict_proba"):
                per_tree.append(est.predict_proba(X_trans))
        if not per_tree:
            return None, None
        stack = np.stack(per_tree, axis=0)  # (n_trees, n_samples, n_classes)
        winning = stack[:, np.arange(stack.shape[1]), pred_int]  # (n_trees, n_samples)
        mean = winning.mean(axis=0)
        std = winning.std(axis=0)
        return np.clip(mean - 1.96 * std, 0, 1), np.clip(mean + 1.96 * std, 0, 1)
    except Exception:  # noqa: BLE001
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score new KOIs with the saved model.")
    parser.add_argument("--input", required=True, help="CSV of raw KOI rows.")
    parser.add_argument("--output", default=str(config.PREDICTIONS_DIR / "scored.csv"))
    args = parser.parse_args()

    model = load_model()
    df = pd.read_csv(args.input, low_memory=False)
    scored = predict_dataframe(model, df)
    scored.to_csv(args.output)
    print(f"Scored {len(df):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
