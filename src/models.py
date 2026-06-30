"""
Model zoo and Optuna search spaces for the exoplanet classifier.

Keeping the catalogue of estimators in one place means ``train.py`` stays a thin
orchestrator and the Streamlit "model comparison" page can describe each model
from the same source of truth.

For every model we record, in ``MODEL_NOTES``, the trade-offs the brief asks us to
discuss (advantages, disadvantages, interpretability, expected performance).
"""

from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config

RS = config.RANDOM_STATE


# ------------------------------------------------------------
# Model zoo - default-ish hyperparameters for the head-to-head comparison
# ------------------------------------------------------------
def get_model_zoo(include_slow: bool = True) -> dict[str, object]:
    """Return ``{name: fresh_estimator}`` for the comparison table.

    Parameters
    ----------
    include_slow : bool
        If False, omit the estimators that scale poorly to ~7.5k samples
        (SVM, KNN, MLP). Useful for a quick smoke run.
    """
    zoo: dict[str, object] = {
        # sklearn >=1.7 uses multinomial automatically for multiclass targets.
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=RS),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, random_state=RS
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=RS
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=RS
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
        # Regularized + early-stopped defaults so a final model cloned from this
        # base (then given the tuned params) matches the configuration searched.
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=RS,
        ),
    }

    # Optional third-party gradient boosters (imported lazily so a missing
    # library never breaks the whole zoo).
    try:
        from xgboost import XGBClassifier

        zoo["XGBoost"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RS,
        )
    except ImportError:  # pragma: no cover
        pass

    try:
        from lightgbm import LGBMClassifier

        zoo["LightGBM"] = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=RS,
            verbose=-1,
        )
    except ImportError:  # pragma: no cover
        pass

    try:
        from catboost import CatBoostClassifier

        zoo["CatBoost"] = CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            random_state=RS,
            verbose=False,
            allow_writing_files=False,
        )
    except ImportError:  # pragma: no cover
        pass

    if include_slow:
        zoo["Support Vector Machine"] = SVC(
            C=3.0, kernel="rbf", probability=True, random_state=RS
        )
        zoo["K-Nearest Neighbors"] = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
        zoo["MLP Neural Network"] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            alpha=1e-3,
            max_iter=400,
            early_stopping=True,
            random_state=RS,
        )

    return zoo


# ------------------------------------------------------------
# Optuna search spaces for the strongest model families
# ------------------------------------------------------------
def build_tuned_model(name: str, trial) -> object:
    """Instantiate ``name`` with hyperparameters sampled from ``trial``.

    Supported: Random Forest, Extra Trees, Hist Gradient Boosting, XGBoost,
    LightGBM, CatBoost. These are the families worth the optimization budget.
    """
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            n_jobs=-1,
            random_state=RS,
        )
    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            n_jobs=-1,
            random_state=RS,
        )
    if name == "Hist Gradient Boosting":
        # Regularization-first space + early stopping to control the train/test
        # gap: shallow trees (small max_leaf_nodes), large min_samples_leaf,
        # meaningful L2, column subsampling, and a validation-based stop.
        return HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_iter=trial.suggest_int("max_iter", 300, 900, step=100),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 31),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 200),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-2, 30.0, log=True),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RS,
        )
    if name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RS,
        )
    if name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 300, 1000, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 150),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            n_jobs=-1,
            random_state=RS,
            verbose=-1,
        )
    if name == "CatBoost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 300, 900, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_state=RS,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"No Optuna search space defined for model {name!r}")


TUNABLE: set[str] = {
    "Random Forest",
    "Extra Trees",
    "Hist Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
}


# ------------------------------------------------------------
# Qualitative notes (used in the report and the Streamlit comparison page)
# ------------------------------------------------------------
MODEL_NOTES: dict[str, dict[str, str]] = {
    "Logistic Regression": {
        "advantages": "Fast, fully interpretable coefficients, strong baseline.",
        "disadvantages": "Linear decision boundary; underfits complex interactions.",
        "interpretability": "High",
        "expected": "Moderate",
    },
    "Decision Tree": {
        "advantages": "Human-readable rules, captures non-linearities.",
        "disadvantages": "High variance, prone to overfitting alone.",
        "interpretability": "High",
        "expected": "Moderate",
    },
    "Random Forest": {
        "advantages": "Robust, handles mixed scales, low tuning effort.",
        "disadvantages": "Larger memory, less interpretable than a single tree.",
        "interpretability": "Medium",
        "expected": "High",
    },
    "Extra Trees": {
        "advantages": "Even lower variance than RF, very fast to train.",
        "disadvantages": "Extra randomness can cost a little bias.",
        "interpretability": "Medium",
        "expected": "High",
    },
    "Gradient Boosting": {
        "advantages": "Strong accuracy, sequential error-correction.",
        "disadvantages": "Slow to train, sensitive to hyperparameters.",
        "interpretability": "Medium",
        "expected": "High",
    },
    "Hist Gradient Boosting": {
        "advantages": "Histogram boosting; fast and native NaN handling.",
        "disadvantages": "Fewer knobs than XGBoost/LightGBM.",
        "interpretability": "Medium",
        "expected": "High",
    },
    "XGBoost": {
        "advantages": "Regularized boosting, excellent tabular performance.",
        "disadvantages": "Many hyperparameters; needs tuning.",
        "interpretability": "Medium (SHAP-friendly)",
        "expected": "Very High",
    },
    "LightGBM": {
        "advantages": "Leaf-wise growth, very fast, great on tabular data.",
        "disadvantages": "Can overfit small data without regularization.",
        "interpretability": "Medium (SHAP-friendly)",
        "expected": "Very High",
    },
    "CatBoost": {
        "advantages": "Ordered boosting, strong defaults, robust.",
        "disadvantages": "Slower training than LightGBM.",
        "interpretability": "Medium (SHAP-friendly)",
        "expected": "Very High",
    },
    "Support Vector Machine": {
        "advantages": "Effective in high-dimensional margins.",
        "disadvantages": "Scales poorly (O(n^2)); slow probability calibration.",
        "interpretability": "Low",
        "expected": "Moderate",
    },
    "K-Nearest Neighbors": {
        "advantages": "Simple, non-parametric.",
        "disadvantages": "Slow prediction, sensitive to scaling & curse of dim.",
        "interpretability": "Low",
        "expected": "Moderate",
    },
    "MLP Neural Network": {
        "advantages": "Learns complex interactions.",
        "disadvantages": "Needs scaling/tuning, less interpretable, data-hungry.",
        "interpretability": "Low",
        "expected": "High",
    },
}
