"""
Evaluation and visualization for the exoplanet classifier.

Provides:
* ``compute_metrics``        - the per-model scorecard (accuracy, weighted
                               precision/recall/F1, macro/weighted ROC-AUC OvR,
                               train & predict time).
* a family of ``plot_*``     - confusion matrix, ROC (OvR), precision-recall
                               (OvR), learning curve, validation curve, calibration
                               curve, class distribution, and the model-comparison
                               bar chart.

All plotting helpers save a PNG to ``outputs/plots`` and return the path so the
notebook, report and Streamlit app can reference identical figures.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import label_binarize

from . import config
from .utils import INT_TO_LABEL, PALETTE, set_plot_style

CLASS_INTS = list(range(len(config.CLASS_ORDER)))
CLASS_NAMES = config.CLASS_ORDER


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    train_time: float | None = None,
) -> dict[str, float]:
    """Return a scorecard dict for a fitted ``model`` on the test set.

    ROC-AUC is computed One-vs-Rest (weighted), the appropriate multi-class
    formulation for an imbalanced 3-class target.
    """
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        try:
            metrics["roc_auc_ovr_weighted"] = roc_auc_score(
                y_test, proba, multi_class="ovr", average="weighted"
            )
        except ValueError:
            metrics["roc_auc_ovr_weighted"] = np.nan
    else:
        metrics["roc_auc_ovr_weighted"] = np.nan

    if train_time is not None:
        metrics["train_time_s"] = train_time
    metrics["predict_time_s"] = predict_time
    return metrics


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
def plot_class_distribution(y, path: Path = config.PLOTS_DIR / "class_distribution.png") -> Path:
    """Bar chart of the target class balance."""
    import matplotlib.pyplot as plt

    set_plot_style()
    counts = pd.Series(y).value_counts().reindex(config.CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{v:,}\n{v/len(y):.1%}",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Target class distribution - koi_disposition")
    ax.set_ylabel("Count")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    model, X_test, y_test, path: Path = config.PLOTS_DIR / "confusion_matrix.png"
) -> Path:
    """Row-normalized confusion matrix with raw counts annotated."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_plot_style()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_INTS)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm).astype(object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n{cm_norm[i, j]:.1%}"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="mako", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                vmin=0, vmax=1, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (row-normalized)")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_roc_curves(
    model, X_test, y_test, path: Path = config.PLOTS_DIR / "roc_curves.png"
) -> Path:
    """One-vs-Rest ROC curve per class."""
    import matplotlib.pyplot as plt

    set_plot_style()
    proba = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=CLASS_INTS)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in INT_TO_LABEL.items():
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        auc = roc_auc_score(y_bin[:, i], proba[:, i])
        ax.plot(fpr, tpr, color=PALETTE[name], lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pr_curves(
    model, X_test, y_test, path: Path = config.PLOTS_DIR / "pr_curves.png"
) -> Path:
    """One-vs-Rest precision-recall curve per class."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score

    set_plot_style()
    proba = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=CLASS_INTS)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in INT_TO_LABEL.items():
        precision, recall, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
        ap = average_precision_score(y_bin[:, i], proba[:, i])
        ax.plot(recall, precision, color=PALETTE[name], lw=2, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (One-vs-Rest)")
    ax.legend(loc="lower left")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_learning_curve(
    estimator, X, y, path: Path = config.PLOTS_DIR / "learning_curve.png"
) -> Path:
    """Train/validation score vs. training-set size (bias-variance diagnostic)."""
    import matplotlib.pyplot as plt

    set_plot_style()
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=config.CV_FOLDS, scoring=config.PRIMARY_METRIC,
        train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1, random_state=config.RANDOM_STATE,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_scores.mean(axis=1), "o-", color=PALETTE["CONFIRMED"], label="Train")
    ax.fill_between(sizes, train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15, color=PALETTE["CONFIRMED"])
    ax.plot(sizes, val_scores.mean(axis=1), "o-", color=PALETTE["FALSE POSITIVE"], label="CV")
    ax.fill_between(sizes, val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15, color=PALETTE["FALSE POSITIVE"])
    ax.set_xlabel("Training examples")
    ax.set_ylabel(f"{config.PRIMARY_METRIC}")
    ax.set_title("Learning curve")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_validation_curve(
    estimator, X, y, param_name: str, param_range,
    path: Path = config.PLOTS_DIR / "validation_curve.png",
) -> Path:
    """Score vs. a single hyperparameter (over/under-fitting diagnostic)."""
    import matplotlib.pyplot as plt

    set_plot_style()
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=config.CV_FOLDS, scoring=config.PRIMARY_METRIC, n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(param_range, train_scores.mean(axis=1), "o-", color=PALETTE["CONFIRMED"], label="Train")
    ax.plot(param_range, val_scores.mean(axis=1), "o-", color=PALETTE["FALSE POSITIVE"], label="CV")
    ax.set_xlabel(param_name)
    ax.set_ylabel(config.PRIMARY_METRIC)
    ax.set_title(f"Validation curve - {param_name}")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration_curve(
    model, X_test, y_test, path: Path = config.PLOTS_DIR / "calibration_curve.png"
) -> Path:
    """Reliability diagram (One-vs-Rest) - are predicted probabilities honest?"""
    import matplotlib.pyplot as plt

    set_plot_style()
    proba = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=CLASS_INTS)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in INT_TO_LABEL.items():
        CalibrationDisplay.from_predictions(
            y_bin[:, i], proba[:, i], n_bins=10, ax=ax, name=name, color=PALETTE[name]
        )
    ax.set_title("Calibration curves (One-vs-Rest)")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_model_comparison(
    comparison: pd.DataFrame, path: Path = config.PLOTS_DIR / "model_comparison.png"
) -> Path:
    """Horizontal bar chart ranking models by weighted F1 (with ROC-AUC overlay)."""
    import matplotlib.pyplot as plt

    set_plot_style()
    df = comparison.sort_values("f1_weighted")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(df))))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["f1_weighted"], height=0.4, color=PALETTE["CONFIRMED"], label="F1 (weighted)")
    if "roc_auc_ovr_weighted" in df:
        ax.barh(y + 0.2, df["roc_auc_ovr_weighted"], height=0.4, color=ACCENT_BLUE, label="ROC-AUC (OvR)")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlim(min(0.5, df["f1_weighted"].min() - 0.05), 1.0)
    ax.set_xlabel("Score")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)
    return path


ACCENT_BLUE = "#118ab2"
