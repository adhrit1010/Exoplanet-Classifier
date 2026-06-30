"""
Feature engineering for the KOI exoplanet classifier.

This module exposes a single scikit-learn compatible transformer,
``FeatureEngineer``, that derives *scientifically meaningful* features from the
raw transit and stellar measurements. It is deliberately defensive: every
feature is only created when its source columns are present, so the exact same
object can transform the full training frame **and** a single-row dictionary
coming from the Streamlit form.

Design principles
-----------------
* Stateless physics. Every engineered feature is a deterministic function of a
  single row (ratios, logs, bins). It therefore cannot leak information across
  the train/test boundary - there are no fitted statistics here.
* All-numeric output. Ordered categorical bins (e.g. temperature class) are
  emitted as integer codes so the downstream ``ColumnTransformer`` only ever
  sees numbers. This keeps the pipeline simple and SHAP-friendly.
* No target usage. The transformer never sees ``koi_disposition``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN (not inf) on divide-by-zero."""
    out = numerator / denominator.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Append derived astrophysical features to a KOI DataFrame.

    Parameters
    ----------
    drop_source : bool, default=False
        If True, drop a few raw columns once a strictly more informative derived
        version exists (kept False by default so tree models can use both).

    Notes
    -----
    The transformer is a no-op ``fit`` (it learns nothing), which is exactly what
    makes it leakage-safe.
    """

    def __init__(self, drop_source: bool = False) -> None:
        self.drop_source = drop_source

    # ``fit`` learns nothing - the engineering is pure per-row physics.
    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":  # noqa: N803
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return a copy of ``X`` with engineered columns appended."""
        df = X.copy()
        cols = set(df.columns)

        def has(*names: str) -> bool:
            return all(n in cols for n in names)

        # -- 1. Log transforms of strongly right-skewed positive quantities ----
        # Orbital period, transit depth, insolation, planet radius and signal
        # strength span several orders of magnitude; logs linearize them and
        # stabilize variance for the linear / distance-based models.
        for col in [
            "koi_period", "koi_depth", "koi_insol", "koi_prad",
            "koi_model_snr", "koi_max_mult_ev", "koi_num_transits", "koi_sma",
        ]:
            if col in cols:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

        # -- 2. Transit-shape ratios (separate planets from eclipsing binaries) -
        # Transit "duty cycle": fraction of the orbit spent in transit. Grazing
        # eclipses and stellar binaries sit at anomalous values.
        if has("koi_duration", "koi_period"):
            df["duty_cycle"] = _safe_div(df["koi_duration"], df["koi_period"] * 24.0)

        # Depth significance: a real planet produces a depth that is large
        # relative to its measurement error. Low significance -> likely noise.
        if has("koi_depth", "koi_depth_err1"):
            df["depth_snr"] = _safe_div(df["koi_depth"], df["koi_depth_err1"].abs())

        # Signal-to-noise normalized per transit event.
        if has("koi_model_snr", "koi_num_transits"):
            df["snr_per_transit"] = _safe_div(
                df["koi_model_snr"], np.sqrt(df["koi_num_transits"].clip(lower=1))
            )

        # Consistency between fitted radius ratio and observed depth
        # (depth ≈ (Rp/Rs)^2). Large mismatch is a classic false-positive tell.
        if has("koi_depth", "koi_ror"):
            df["depth_ror_consistency"] = _safe_div(
                df["koi_depth"], (df["koi_ror"] ** 2) * 1.0e6
            )

        # -- 3. Centroid-offset significance (background eclipsing binaries) ---
        # When the photo-centre shifts during transit, the signal comes from a
        # nearby star, not the target -> false positive. These are RAW measured
        # offsets (not the robovetter flag), so using them is fair game.
        for base in ["koi_dicco_msky", "koi_dikco_msky", "koi_fwm_sra", "koi_fwm_sdec"]:
            if has(base, f"{base}_err"):
                df[f"{base}_sig"] = _safe_div(df[base].abs(), df[f"{base}_err"].abs())

        # -- 4. Stellar / planetary physical ratios ---------------------------
        # Equilibrium temperature relative to the host star's effective temp.
        if has("koi_teq", "koi_steff"):
            df["teq_steff_ratio"] = _safe_div(df["koi_teq"], df["koi_steff"])

        # Planet radius expressed in stellar radii (sanity vs koi_ror).
        if has("koi_prad", "koi_srad"):
            df["prad_per_srad"] = _safe_div(df["koi_prad"], df["koi_srad"])

        # -- 5. Ordered categorical bins (emitted as integer codes) ------------
        # Equilibrium-temperature class - physically interpretable buckets.
        if "koi_teq" in cols:
            df["teq_class"] = pd.cut(
                df["koi_teq"],
                bins=[-np.inf, 300, 600, 1000, 2000, np.inf],
                labels=[0, 1, 2, 3, 4],  # cold -> temperate -> warm -> hot -> ultra-hot
            ).astype("float")

        # Orbital-period regime.
        if "koi_period" in cols:
            df["period_class"] = pd.cut(
                df["koi_period"],
                bins=[-np.inf, 1, 10, 100, 365, np.inf],
                labels=[0, 1, 2, 3, 4],
            ).astype("float")

        # Planet-size class (sub-Earth -> Earth -> super-Earth -> Neptune -> Jupiter).
        if "koi_prad" in cols:
            df["prad_class"] = pd.cut(
                df["koi_prad"],
                bins=[-np.inf, 0.8, 1.25, 2.0, 6.0, np.inf],
                labels=[0, 1, 2, 3, 4],
            ).astype("float")

        # Rough habitable-zone flag: insolation within ~0.25-2 × Earth's.
        if "koi_insol" in cols:
            df["in_habitable_zone"] = (
                df["koi_insol"].between(0.25, 2.0)
            ).astype("float")

        if self.drop_source:
            for col in ["koi_sma"]:
                if col in df.columns:
                    df = df.drop(columns=col)

        # Replace any infinities introduced above; imputation happens downstream.
        df = df.replace([np.inf, -np.inf], np.nan)
        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        """Expose output names for sklearn / SHAP introspection."""
        return np.asarray(getattr(self, "feature_names_out_", []), dtype=object)
