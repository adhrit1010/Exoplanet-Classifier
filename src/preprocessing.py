"""
Loading and preprocessing for the KOI exoplanet data.

Two main functions:

* ``load_dataset``       reads the CSV, drops exact-duplicate rows, separates the
                         target, and returns a clean ``(X, y)`` pair.
* ``build_preprocessor`` builds the leakage-safe pipeline (drop leakage columns,
                         add engineered features, prune columns, impute, scale) as
                         one sklearn estimator.

Anything that learns from the data (median fill values, scaler statistics, which
columns survive) is fitted inside the pipeline. That way, when it runs inside
cross-validation, those numbers come from the training folds only and never leak
from the test fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .feature_engineering import FeatureEngineer


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------
def load_dataset(
    path=config.RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the KOI table and return a clean ``(X, y)`` pair.

    Parameters
    ----------
    path : str or Path
        Location of ``KOI_Cumulative_clean.csv``.

    Returns
    -------
    X : DataFrame
        All columns except the target and known-leakage/identifier columns.
        (Leakage columns are *also* dropped defensively inside the pipeline.)
    y : Series
        The ``koi_disposition`` target as an ordered categorical.
    """
    df = pd.read_csv(path, low_memory=False)

    # Exact-duplicate rows carry no information and would inflate CV scores.
    df = df.drop_duplicates().reset_index(drop=True)

    if config.TARGET not in df.columns:
        raise KeyError(f"Target column {config.TARGET!r} not found in {path}")

    # Rows with no label cannot be used for supervised learning.
    df = df[df[config.TARGET].notna()].reset_index(drop=True)

    y = pd.Categorical(
        df[config.TARGET], categories=config.CLASS_ORDER, ordered=True
    )
    y = pd.Series(y, name=config.TARGET)

    X = df.drop(columns=[config.TARGET])
    return X, y


# ------------------------------------------------------------
# Leakage drop  (stateless, but lives in the pipeline so the app can't forget it)
# ------------------------------------------------------------
class LeakageDropper(BaseEstimator, TransformerMixin):
    """Remove identifier, vetting-conclusion and metadata columns.

    Implemented as a transformer (rather than a one-off ``df.drop``) so the rule
    travels *inside* the serialized pipeline - the Streamlit app and any future
    batch-prediction job get the exact same leakage protection for free.
    """

    def __init__(self, drop_cols=None) -> None:
        self.drop_cols = list(config.DROP_COLS) if drop_cols is None else drop_cols

    def fit(self, X: pd.DataFrame, y=None) -> "LeakageDropper":  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        present = [c for c in self.drop_cols if c in X.columns]
        # Also drop the target if it ever sneaks in.
        if config.TARGET in X.columns:
            present.append(config.TARGET)
        return X.drop(columns=present, errors="ignore").copy()


# ------------------------------------------------------------
# Column pruning  (fitted on TRAIN folds only)
# ------------------------------------------------------------
class ColumnPruner(BaseEstimator, TransformerMixin):
    """Coerce to numeric, then drop empty / near-constant / over-missing columns.

    Parameters
    ----------
    missing_threshold : float
        Drop columns whose missing fraction exceeds this value.
    near_constant_threshold : float
        Drop columns where a single value accounts for more than this fraction of
        the non-null entries (no usable variance).

    The surviving column set is learned at ``fit`` time and reused at
    ``transform`` time, guaranteeing identical geometry between train and test.
    """

    def __init__(
        self,
        missing_threshold: float = config.MISSING_DROP_THRESHOLD,
        near_constant_threshold: float = config.NEAR_CONSTANT_THRESHOLD,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.near_constant_threshold = near_constant_threshold

    @staticmethod
    def _to_numeric(X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Force every column to numeric; unparseable strings become NaN."""
        return X.apply(pd.to_numeric, errors="coerce")

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnPruner":  # noqa: N803
        Xn = self._to_numeric(X)
        keep: list[str] = []
        for col in Xn.columns:
            s = Xn[col]
            missing_frac = s.isna().mean()
            if missing_frac > self.missing_threshold:
                continue
            non_null = s.dropna()
            if non_null.empty:
                continue
            # Dominant-value fraction (near-constant detector).
            top_frac = non_null.value_counts(normalize=True).iloc[0]
            if top_frac >= self.near_constant_threshold:
                continue
            keep.append(col)
        self.columns_ = keep
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        Xn = self._to_numeric(X)
        # Re-add any column the model expects but the caller omitted (app case).
        for col in self.columns_:
            if col not in Xn.columns:
                Xn[col] = np.nan
        return Xn[self.columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


# ------------------------------------------------------------
# Correlation pruning  (fitted on TRAIN folds only) - a real feature-selection step
# ------------------------------------------------------------
class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop one column from every near-duplicate (|corr| > threshold) pair.

    Many KOI columns are redundant - most obviously the ``*_err1`` / ``*_err2``
    uncertainty pairs (typically |corr| ~ 1). Removing one of each pair shrinks
    the feature space, reduces variance/overfitting, and makes SHAP importances
    less diluted across collinear copies. The decision (which columns to keep) is
    learned on the training folds only, so it is leakage-safe.

    For each correlated pair the column with the **higher missing fraction** is
    dropped (ties broken by name) - we prefer to keep the better-measured column.
    """

    def __init__(self, threshold: float = config.CORR_DROP_THRESHOLD) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "CorrelationPruner":  # noqa: N803
        corr = X.corr(numeric_only=True).abs()
        missing = X.isna().mean()
        cols = list(corr.columns)
        upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        to_drop: set[str] = set()
        for i, a in enumerate(cols):
            if a in to_drop:
                continue
            for j, b in enumerate(cols):
                if j <= i or b in to_drop:
                    continue
                if upper[i, j] and corr.iat[i, j] > self.threshold:
                    # Drop whichever of the pair is more poorly measured.
                    drop = b if missing[b] >= missing[a] else a
                    to_drop.add(drop)
        self.columns_ = [c for c in cols if c not in to_drop]
        self.dropped_ = sorted(to_drop)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        for col in self.columns_:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


# ------------------------------------------------------------
# Assembled preprocessor
# ------------------------------------------------------------
def build_preprocessor(scale: bool = True) -> Pipeline:
    """Return the full leakage-safe preprocessing pipeline.

    Steps
    -----
    1. ``LeakageDropper``   - strip identifiers / vetting conclusions / metadata.
    2. ``FeatureEngineer``  - append physically-motivated derived features.
    3. ``ColumnPruner``     - numeric coercion + drop empty/constant/over-missing.
    4. ``CorrelationPruner``- drop one of each near-duplicate (|corr|>0.97) pair.
    5. ``SimpleImputer``    - median imputation (robust to the heavy KOI skew).
    6. ``StandardScaler``   - optional; on for linear/SVM/KNN/MLP, harmless to trees.

    Parameters
    ----------
    scale : bool
        Whether to standardize features. Tree ensembles ignore monotonic scaling,
        so this is left on by default to keep one preprocessor for all models.
    """
    steps = [
        ("leakage_drop", LeakageDropper()),
        ("feature_engineering", FeatureEngineer()),
        ("prune", ColumnPruner()),
        ("corr_prune", CorrelationPruner()),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Recover the final feature names produced by a fitted ``build_preprocessor``."""
    last = "corr_prune" if "corr_prune" in preprocessor.named_steps else "prune"
    return list(preprocessor.named_steps[last].get_feature_names_out())
