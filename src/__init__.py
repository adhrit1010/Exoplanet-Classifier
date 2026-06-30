"""
Exoplanet Classifier - leakage-safe ML pipeline for NASA Kepler KOI dispositions.

Public modules
--------------
config              Paths, random seed, target, and the data-leakage policy.
preprocessing       Loading + leakage-safe preprocessing pipeline.
feature_engineering Physically-motivated derived features.
models              Model zoo + Optuna search spaces.
train               End-to-end training orchestrator (``python -m src.train``).
evaluate            Metrics and all diagnostic plots.
explain             SHAP explainability.
predict             Inference API + CLI (``python -m src.predict``).
"""

from . import config  # noqa: F401

__all__ = [
    "config",
    "preprocessing",
    "feature_engineering",
    "feature_selection",
    "models",
    "train",
    "evaluate",
    "explain",
    "predict",
    "report",
    "extensions",
    "utils",
]

__version__ = "1.0.0"
