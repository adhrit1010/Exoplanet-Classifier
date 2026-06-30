"""
Explainable-AI layer (SHAP) for the exoplanet classifier.

The shipped model is a two-step ``Pipeline(preprocessor, classifier)``. SHAP must
see the *transformed, named* feature matrix, so every helper here first pushes the
raw frame through the preprocessor, then explains the underlying tree model.

Outputs (saved to ``outputs/plots``):
* ``shap_global_bar.png``      - mean |SHAP| global importance.
* ``shap_summary_<class>.png`` - beeswarm summary for each disposition.
* ``shap_waterfall.png``       - local explanation for one example.
* native ``feature_importance.png`` - the model's own impurity/gain importance.

Robustness note: SHAP's multi-class output shape changed across versions, so we
normalize everything to a ``list[np.ndarray]`` (one array per class) up front.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .preprocessing import get_feature_names
from .utils import set_plot_style


def _split_pipeline(model):
    """Return ``(preprocessor, classifier)`` from the shipped pipeline."""
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    return pre, clf


def _transform(model, X: pd.DataFrame) -> pd.DataFrame:
    """Push raw frame through the preprocessor -> named, transformed DataFrame."""
    pre, _ = _split_pipeline(model)
    arr = pre.transform(X)
    names = get_feature_names(pre)
    return pd.DataFrame(arr, columns=names, index=X.index)


def _normalize_shap(shap_values, n_classes: int) -> list[np.ndarray]:
    """Coerce any SHAP multi-class output into ``list[(n_samples, n_features)]``."""
    if isinstance(shap_values, list):
        return shap_values
    arr = np.asarray(shap_values)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        return [arr[:, :, k] for k in range(arr.shape[-1])]
    return [arr]  # binary / single-output fallback


def compute_shap(model, X: pd.DataFrame, max_samples: int = 1000):
    """Compute SHAP values for a sample of ``X``.

    Returns
    -------
    explainer : shap.TreeExplainer
    shap_list : list[np.ndarray]   one (n, n_features) array per class
    X_trans   : DataFrame          the transformed feature matrix explained
    """
    import shap

    X_sample = X.sample(min(max_samples, len(X)), random_state=config.RANDOM_STATE)
    X_trans = _transform(model, X_sample)
    _, clf = _split_pipeline(model)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_trans)
    shap_list = _normalize_shap(shap_values, len(config.CLASS_ORDER))
    return explainer, shap_list, X_trans


def plot_global_importance(
    shap_list, X_trans, path: Path = config.PLOTS_DIR / "shap_global_bar.png", top_n: int = 20
) -> Path:
    """Bar chart of mean |SHAP| across all classes (global feature importance)."""
    import matplotlib.pyplot as plt

    set_plot_style()
    # Average absolute SHAP across classes, then across samples.
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_list], axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    names = np.asarray(X_trans.columns)[order]
    vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(order))))
    ax.barh(range(len(order)), vals[::-1], color="#06d6a0")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Global feature importance (SHAP)")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_class_summaries(shap_list, X_trans, out_dir: Path = config.PLOTS_DIR) -> list[Path]:
    """Beeswarm summary plot per disposition class."""
    import matplotlib.pyplot as plt
    import shap

    set_plot_style()
    paths: list[Path] = []
    for k, name in enumerate(config.CLASS_ORDER):
        if k >= len(shap_list):
            break
        plt.figure()
        shap.summary_plot(shap_list[k], X_trans, show=False, max_display=15, plot_size=(8, 6))
        plt.title(f"SHAP summary - class: {name}")
        p = out_dir / f"shap_summary_{name.replace(' ', '_').lower()}.png"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        plt.close()
        paths.append(p)
    return paths


def plot_local_waterfall(
    explainer, shap_list, X_trans, sample_idx: int = 0,
    class_idx: int | None = None, path: Path = config.PLOTS_DIR / "shap_waterfall.png",
) -> Path:
    """Waterfall explanation for a single prediction.

    If ``class_idx`` is None, explain the class with the largest |SHAP| mass for
    that row (i.e. the class the model leaned toward).
    """
    import matplotlib.pyplot as plt
    import shap

    set_plot_style()
    if class_idx is None:
        class_idx = int(np.argmax([np.abs(sv[sample_idx]).sum() for sv in shap_list]))

    base = explainer.expected_value
    base_val = base[class_idx] if np.ndim(base) > 0 else base

    expl = shap.Explanation(
        values=shap_list[class_idx][sample_idx],
        base_values=base_val,
        data=X_trans.iloc[sample_idx].values,
        feature_names=list(X_trans.columns),
    )
    plt.figure()
    shap.plots.waterfall(expl, max_display=14, show=False)
    plt.title(f"Local explanation - predicted toward {config.CLASS_ORDER[class_idx]}")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    return path


def plot_native_importance(
    model, path: Path = config.PLOTS_DIR / "feature_importance.png", top_n: int = 20
) -> Path | None:
    """Plot the model's native importance (gain/impurity) if available."""
    import matplotlib.pyplot as plt

    set_plot_style()
    pre, clf = _split_pipeline(model)
    if not hasattr(clf, "feature_importances_"):
        return None
    names = np.asarray(get_feature_names(pre))
    imp = np.asarray(clf.feature_importances_)
    order = np.argsort(imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(order))))
    ax.barh(range(len(order)), imp[order][::-1], color="#118ab2")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[order][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Native model feature importance")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_permutation_importance(
    model, X: pd.DataFrame, y, path: Path = config.PLOTS_DIR / "permutation_importance.png",
    top_n: int = 20, n_repeats: int = 10,
) -> Path:
    """Model-agnostic permutation importance on the held-out set.

    Works for any estimator (including HistGradientBoosting, which exposes no
    native ``feature_importances_``). Importance = drop in weighted F1 when a
    feature's column is shuffled. ``n_jobs=1`` avoids the joblib/loky deadlock
    seen on this platform. Also writes the full ranking to a CSV next to the PNG.
    """
    import matplotlib.pyplot as plt
    from sklearn.inspection import permutation_importance

    set_plot_style()
    # Permute the *transformed* model features (same space as SHAP), so the
    # importance vector aligns with the final feature names. n_jobs=1 sidesteps
    # the joblib/loky deadlock seen on this platform.
    pre, clf = _split_pipeline(model)
    X_trans = pre.transform(X)
    names = np.asarray(get_feature_names(pre))
    result = permutation_importance(
        clf, X_trans, y, scoring="f1_weighted", n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    imp = result.importances_mean
    ser = pd.Series(imp, index=names).sort_values(ascending=False)
    ser.to_csv(config.REPORTS_DIR / "permutation_importance.csv", header=["importance"])

    order = np.argsort(imp)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(order))))
    ax.barh(range(len(order)), imp[order][::-1],
            xerr=result.importances_std[order][::-1], color="#118ab2")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[order][::-1])
    ax.set_xlabel("Drop in weighted F1 when shuffled")
    ax.set_title("Permutation feature importance (test set)")
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_shap(
    model, X: pd.DataFrame, y=None, max_samples: int = 1000
) -> dict[str, object]:
    """Compute SHAP once and emit every explainability artifact.

    If the classifier has no native ``feature_importances_`` (e.g.
    HistGradientBoosting) and labels ``y`` are supplied, a permutation-importance
    plot is produced as ``feature_importance.png`` so that artifact always
    reflects the shipped model rather than a stale earlier run.
    """
    explainer, shap_list, X_trans = compute_shap(model, X, max_samples=max_samples)
    native = plot_native_importance(model)
    if native is None and y is not None:
        # Fall back to permutation importance, written to the same artifact path.
        native = plot_permutation_importance(
            model, X, y, path=config.PLOTS_DIR / "feature_importance.png"
        )
    outputs = {
        "global_bar": plot_global_importance(shap_list, X_trans),
        "class_summaries": plot_class_summaries(shap_list, X_trans),
        "waterfall": plot_local_waterfall(explainer, shap_list, X_trans),
        "native_importance": native,
    }
    if y is not None:
        outputs["permutation_importance"] = plot_permutation_importance(model, X, y)
    return outputs
