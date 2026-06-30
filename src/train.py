"""
Training orchestrator for the exoplanet classifier.

Running ``python -m src.train`` reproduces the entire modeling pipeline end to
end and writes every artifact the brief asks for:

    Phase 5  Model zoo comparison        -> outputs/reports/model_comparison.csv
    Phase 4  Class-imbalance comparison   -> outputs/reports/imbalance_comparison.csv
    Phase 6  Optuna hyperparameter search -> models/best_params.json
    Phase 7  Final evaluation + plots     -> outputs/plots/*.png
    Phase 8  SHAP explainability          -> outputs/plots/shap_*.png
             Saved model + metadata       -> models/exoplanet_classifier.pkl

Everything is leakage-safe: the test split is held out before any fitting, model
selection / tuning use cross-validation on the training split only, and all
preprocessing is fitted inside the CV folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

# Keep console logging crash-proof on code pages that can't encode unicode
# (e.g. Windows cp1252) - unencodable characters are replaced, never raised.
try:  # pragma: no cover - environment dependent
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import os

import joblib
import pandas as pd
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE

# Escape hatch for platforms where joblib's loky (process) backend deadlocks
# (observed on some Windows setups after worker pools are killed). Set
# ``EXOPLANET_BACKEND=threading`` to run all CV under the threading backend.
if os.environ.get("EXOPLANET_BACKEND", "").lower() == "threading":
    joblib.parallel_backend("threading").__enter__()
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline

from . import config, evaluate
from .models import TUNABLE, build_tuned_model, get_model_zoo
from .preprocessing import build_preprocessor, get_feature_names, load_dataset
from .utils import decode_labels, encode_labels, save_json

CV = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
SCORING = {
    "accuracy": "accuracy",
    "precision_weighted": "precision_weighted",
    "recall_weighted": "recall_weighted",
    "f1_weighted": "f1_weighted",
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_sampler(strategy: str):
    """Return an imblearn sampler for a resampling ``strategy`` (or None)."""
    rs = config.RANDOM_STATE
    return {
        "none": None,
        "class_weight": None,
        "smote": SMOTE(random_state=rs),
        "adasyn": ADASYN(random_state=rs),
        "random_over": RandomOverSampler(random_state=rs),
        "random_under": RandomUnderSampler(random_state=rs),
    }[strategy]


def _apply_class_weight(clf, strategy: str):
    """Set ``class_weight='balanced'`` when the estimator supports it."""
    clf = clone(clf)
    if strategy == "class_weight" and "class_weight" in clf.get_params():
        clf.set_params(class_weight="balanced")
    return clf


def _make_estimator(clf, strategy: str = "none"):
    """Compose preprocessing + optional sampler + classifier into one estimator.

    The preprocessing steps are *flattened* (not nested as a sub-Pipeline) because
    ``imblearn.Pipeline`` forbids a Pipeline as an intermediate step. The final
    classifier step is always named ``"classifier"`` so hyperparameter paths like
    ``classifier__n_estimators`` work for validation curves and Optuna.
    """
    pre_steps = build_preprocessor(scale=True).steps  # flatten to (name, transformer)
    clf = _apply_class_weight(clf, strategy)
    sampler = _get_sampler(strategy)
    if sampler is None:
        return SkPipeline([*pre_steps, ("classifier", clf)])
    return ImbPipeline([*pre_steps, ("sampler", sampler), ("classifier", clf)])


# ------------------------------------------------------------
# Phase 5 - model zoo comparison (cross-validated on the training split)
# ------------------------------------------------------------
def compare_models(X_train, y_train, include_slow: bool = True) -> pd.DataFrame:
    """Cross-validate every model in the zoo and return a ranked scorecard."""
    zoo = get_model_zoo(include_slow=include_slow)
    rows = {}
    for name, clf in zoo.items():
        est = _make_estimator(clf, "none")
        print(f"  - CV evaluating {name} ...", flush=True)
        cv = cross_validate(
            est, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1,
            return_train_score=False, error_score="raise",
        )
        row = {f"{m}": cv[f"test_{m}"].mean() for m in SCORING}
        row["f1_weighted_std"] = cv["test_f1_weighted"].std()
        row["fit_time_s"] = cv["fit_time"].mean()
        row["score_time_s"] = cv["score_time"].mean()
        rows[name] = row
    df = pd.DataFrame(rows).T.sort_values("f1_weighted", ascending=False)
    return df


# ------------------------------------------------------------
# Phase 4 - class-imbalance strategy comparison
# ------------------------------------------------------------
def compare_imbalance(X_train, y_train, base_clf) -> tuple[pd.DataFrame, str]:
    """Compare resampling / weighting strategies for a fixed model family."""
    strategies = ["none", "class_weight", "smote", "adasyn", "random_over", "random_under"]
    rows = {}
    for strat in strategies:
        # Skip class_weight if the chosen model does not support it.
        if strat == "class_weight" and "class_weight" not in clone(base_clf).get_params():
            continue
        est = _make_estimator(base_clf, strat)
        print(f"  - CV imbalance strategy: {strat} ...", flush=True)
        cv = cross_validate(est, X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
        rows[strat] = {
            "f1_weighted": cv["test_f1_weighted"].mean(),
            "f1_weighted_std": cv["test_f1_weighted"].std(),
            "recall_weighted": cv["test_recall_weighted"].mean(),
            "roc_auc_ovr_weighted": cv["test_roc_auc_ovr_weighted"].mean(),
        }
    df = pd.DataFrame(rows).T.sort_values("f1_weighted", ascending=False)
    best = df.index[0]
    return df, best


# ------------------------------------------------------------
# Phase 6 - Optuna hyperparameter optimization
# ------------------------------------------------------------
def tune_with_optuna(X_train, y_train, model_name: str, strategy: str, n_trials: int):
    """Optimize ``model_name`` for weighted F1 using Optuna; return best params."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        clf = build_tuned_model(model_name, trial)
        est = _make_estimator(clf, strategy)
        scores = cross_validate(
            est, X_train, y_train, cv=CV, scoring="f1_weighted", n_jobs=-1,
            return_train_score=True,
        )
        # Optimize for *generalization*: validation F1 penalized by the train/val
        # gap, so the search prefers configurations that do not overfit.
        val = scores["test_score"].mean()
        gap = scores["train_score"].mean() - val
        return val - 0.5 * max(0.0, gap)

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  > best CV f1_weighted = {study.best_value:.4f}")
    return study.best_params, study


# ------------------------------------------------------------
# Final model assembly  (leakage-safe: resampling only affects training data)
# ------------------------------------------------------------
def build_final_model(X_train, y_train, clf, strategy: str) -> SkPipeline:
    """Fit preprocessor -> (optional) resample -> classifier; ship preprocessor+clf.

    The returned estimator is a clean ``Pipeline(preprocessor, classifier)`` with
    NO sampler, so prediction never resamples - exactly what we want in production.
    """
    pre = build_preprocessor(scale=True)
    clf = _apply_class_weight(clf, strategy)

    X_proc = pre.fit_transform(X_train, y_train)
    sampler = _get_sampler(strategy)
    if sampler is not None:
        X_res, y_res = sampler.fit_resample(X_proc, y_train)
    else:
        X_res, y_res = X_proc, y_train

    clf.fit(X_res, y_res)
    return SkPipeline([("preprocessor", pre), ("classifier", clf)])


# ------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------
def main(quick: bool = False, n_trials: int = 40, include_slow: bool = True) -> None:
    """Run the full training pipeline and persist all artifacts."""
    print("Loading dataset ...")
    X, y = load_dataset()
    y_int = encode_labels(y)
    print(f"  dataset: {X.shape[0]:,} rows x {X.shape[1]} raw columns")
    print(f"  class balance: {pd.Series(y).value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=y_int,
    )
    print(f"  train={len(X_train):,}  test={len(X_test):,}")

    # Phase 5 - model zoo comparison.
    print("\n[Phase 5] Model comparison (5-fold CV on train) ...")
    comparison = compare_models(X_train, y_train, include_slow=include_slow)
    comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv")
    evaluate.plot_model_comparison(comparison)
    print(comparison.round(4).to_string())

    best_name = comparison.index[0]
    # Optuna spaces only exist for tree families; tune the best *tunable* model.
    tunable_ranked = [n for n in comparison.index if n in TUNABLE]
    tune_name = tunable_ranked[0] if tunable_ranked else best_name
    print(f"\nBest overall: {best_name} | tuning target: {tune_name}")

    base_clf = get_model_zoo(include_slow=include_slow)[tune_name]

    # Phase 4 - imbalance strategy.
    print("\n[Phase 4] Class-imbalance strategy comparison ...")
    imb_table, best_strategy = compare_imbalance(X_train, y_train, base_clf)
    imb_table.to_csv(config.REPORTS_DIR / "imbalance_comparison.csv")
    print(imb_table.round(4).to_string())
    print(f"  > chosen imbalance strategy: {best_strategy}")

    # Phase 6 - Optuna.
    print(f"\n[Phase 6] Optuna optimization of {tune_name} ({n_trials} trials) ...")
    best_params, study = tune_with_optuna(
        X_train, y_train, tune_name, best_strategy, n_trials=n_trials
    )
    save_json(best_params, config.MODELS_DIR / "best_params.json")
    study.trials_dataframe().to_csv(config.REPORTS_DIR / "optuna_trials.csv", index=False)
    print(f"  best params: {best_params}")

    # Build & fit final model on the full training split.
    print("\nFitting final model on full training split ...")
    tuned_clf = clone(base_clf).set_params(**best_params)
    final_model = build_final_model(X_train, y_train, tuned_clf, best_strategy)

    # Phase 7 - evaluation on the held-out test set.
    print("\n[Phase 7] Final evaluation on held-out test set ...")
    final_metrics = evaluate.compute_metrics(final_model, X_test, y_test)
    print("  " + json.dumps({k: round(float(v), 4) for k, v in final_metrics.items()}))

    evaluate.plot_class_distribution(decode_labels(y_int))
    evaluate.plot_confusion_matrix(final_model, X_test, y_test)
    evaluate.plot_roc_curves(final_model, X_test, y_test)
    evaluate.plot_pr_curves(final_model, X_test, y_test)
    try:
        evaluate.plot_calibration_curve(final_model, X_test, y_test)
    except Exception as exc:  # noqa: BLE001 - calibration is "if appropriate"
        print(f"  (calibration skipped: {exc})")

    # Learning / validation curves use a leakage-safe CV estimator.
    cv_est = _make_estimator(tuned_clf, best_strategy)
    try:
        evaluate.plot_learning_curve(cv_est, X_train, y_train)
        n_est_param = (
            "classifier__n_estimators"
            if "n_estimators" in clone(tuned_clf).get_params()
            else "classifier__max_iter"
            if "max_iter" in clone(tuned_clf).get_params()
            else None
        )
        if n_est_param:
            evaluate.plot_validation_curve(
                cv_est, X_train, y_train, n_est_param, [100, 200, 400, 600, 800]
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  (curve plots skipped: {exc})")

    # Phase 8 - SHAP.
    print("\n[Phase 8] SHAP explainability ...")
    try:
        from . import explain

        explain.generate_all_shap(final_model, X_test, y=y_test, max_samples=800)
        print("  SHAP + permutation importance saved.")
    except Exception as exc:  # noqa: BLE001
        print(f"  (SHAP skipped: {exc})")

    # Persist model + metadata.
    print("\nSaving model and metadata ...")
    joblib.dump(final_model, config.MODEL_PATH)
    feature_names = get_feature_names(final_model.named_steps["preprocessor"])
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": tune_name,
        "best_overall_in_comparison": best_name,
        "imbalance_strategy": best_strategy,
        "best_params": best_params,
        "random_state": config.RANDOM_STATE,
        "class_order": config.CLASS_ORDER,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "test_metrics": {k: float(v) for k, v in final_metrics.items()},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    save_json(metadata, config.METADATA_PATH)

    # Save test-set predictions with confidence.
    _save_predictions(final_model, X_test, y_test)

    # Bonus: experiment tracking (no-op if MLflow is not installed).
    _log_mlflow(metadata, comparison)

    print(f"\nDone. Model -> {config.MODEL_PATH}")
    print(f"Test weighted-F1 = {final_metrics['f1_weighted']:.4f} | "
          f"ROC-AUC(OvR) = {final_metrics['roc_auc_ovr_weighted']:.4f}")


def _log_mlflow(metadata: dict, comparison: pd.DataFrame) -> None:
    """Log the run to MLflow if it is installed (bonus: experiment tracking).

    Import-guarded so the pipeline never depends on MLflow being present.
    """
    try:  # pragma: no cover - optional dependency
        import mlflow
    except ImportError:
        return
    try:
        mlflow.set_experiment("exoplanet-classifier")
        with mlflow.start_run(run_name=metadata["model_name"]):
            mlflow.log_params(
                {
                    "model_name": metadata["model_name"],
                    "imbalance_strategy": metadata["imbalance_strategy"],
                    "n_features": metadata["n_features"],
                    "random_state": metadata["random_state"],
                    **{f"hp_{k}": v for k, v in metadata["best_params"].items()},
                }
            )
            mlflow.log_metrics({k: float(v) for k, v in metadata["test_metrics"].items()})
            comparison.to_csv(config.REPORTS_DIR / "model_comparison.csv")
            mlflow.log_artifact(str(config.REPORTS_DIR / "model_comparison.csv"))
            if config.MODEL_PATH.exists():
                mlflow.log_artifact(str(config.MODEL_PATH))
        print("  MLflow run logged.")
    except Exception as exc:  # noqa: BLE001
        print(f"  (MLflow logging skipped: {exc})")


def _save_predictions(model, X_test, y_test) -> None:
    """Write held-out test predictions + class probabilities + confidence."""
    proba = model.predict_proba(X_test)
    pred_int = proba.argmax(axis=1)
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in config.CLASS_ORDER], index=X_test.index)
    out.insert(0, "predicted", decode_labels(pred_int))
    out.insert(0, "actual", decode_labels(y_test))
    out["confidence"] = proba.max(axis=1)
    out["correct"] = out["actual"] == out["predicted"]
    out.to_csv(config.PREDICTIONS_DIR / "test_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the exoplanet classifier.")
    parser.add_argument("--quick", action="store_true", help="Fast smoke run.")
    parser.add_argument("--trials", type=int, default=40, help="Optuna trials.")
    parser.add_argument("--no-slow", action="store_true", help="Skip SVM/KNN/MLP.")
    args = parser.parse_args()
    main(
        quick=args.quick,
        n_trials=15 if args.quick else args.trials,
        include_slow=not args.no_slow,
    )
