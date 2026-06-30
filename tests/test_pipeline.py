"""
Smoke tests for the leakage-safe preprocessing pipeline.

These are intentionally lightweight (no model training) so they run in seconds in
CI, yet they assert the property that matters most for this project: **no leakage
column ever reaches the model**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.feature_engineering import FeatureEngineer
from src.preprocessing import (
    LeakageDropper,
    build_preprocessor,
    get_feature_names,
    load_dataset,
)
from src.utils import decode_labels, encode_labels


@pytest.fixture(scope="module")
def data():
    X, y = load_dataset()
    return X.sample(min(800, len(X)), random_state=0), y.loc[
        X.sample(min(800, len(X)), random_state=0).index
    ]


def test_target_separated_and_labeled():
    X, y = load_dataset()
    assert config.TARGET not in X.columns
    assert set(pd.unique(y)).issubset(set(config.CLASS_ORDER))
    assert y.isna().sum() == 0


def test_leakage_dropper_removes_all_banned_columns():
    X, _ = load_dataset()
    dropped = LeakageDropper().fit_transform(X)
    for col in config.DROP_COLS:
        assert col not in dropped.columns, f"{col} leaked through!"
    # The most dangerous leaks specifically.
    for col in ["kepler_name", "koi_pdisposition", "koi_score",
                "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]:
        assert col not in dropped.columns


def test_feature_engineer_adds_columns():
    X, _ = load_dataset()
    fe = FeatureEngineer().fit(X)
    out = fe.transform(X)
    assert out.shape[1] > X.shape[1]
    for col in ["duty_cycle", "depth_snr", "teq_class", "period_class"]:
        assert col in out.columns


def test_preprocessor_outputs_clean_numeric_matrix(data):
    X, y = data
    pre = build_preprocessor(scale=True)
    Xt = pre.fit_transform(X, encode_labels(y))
    assert np.isfinite(Xt).all(), "preprocessor produced NaN/inf"
    names = get_feature_names(pre)
    assert len(names) == Xt.shape[1]
    # No leakage column should survive into the feature names.
    for col in config.DROP_COLS:
        assert col not in names


def test_label_roundtrip():
    codes = encode_labels(config.CLASS_ORDER)
    assert list(codes) == list(range(len(config.CLASS_ORDER)))
    assert list(decode_labels(codes)) == config.CLASS_ORDER


def test_preprocessor_handles_partial_input(data):
    """A single row with only a few columns (the Streamlit case) must score."""
    X, y = data
    pre = build_preprocessor(scale=True)
    pre.fit(X, encode_labels(y))
    partial = pd.DataFrame([{"koi_period": 10.0, "koi_depth": 500.0, "koi_prad": 2.0}])
    out = pre.transform(partial)
    assert out.shape[0] == 1
    assert np.isfinite(out).all()
