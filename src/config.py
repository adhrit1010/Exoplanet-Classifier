"""
Project settings, kept in one place.

This holds the things every other file needs to agree on: where the data and
outputs live, the random seed, the target column, and the list of columns I
refuse to feed the model (the data-leakage policy). Keeping them here means the
notebook, the training script, and the dashboard all stay in sync.
"""

from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
RANDOM_STATE: int = 42  # set EVERYWHERE (splits, models, resamplers, Optuna)

# ------------------------------------------------------------
# Paths (resolved relative to the project root, so the code runs from anywhere)
# ------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"

RAW_DATA_PATH: Path = DATA_DIR / "KOI_Cumulative_clean.csv"
MODEL_PATH: Path = MODELS_DIR / "exoplanet_classifier.pkl"
METADATA_PATH: Path = MODELS_DIR / "model_metadata.json"

for _d in (MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, REPORTS_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------
TARGET: str = "koi_disposition"
CLASS_ORDER: list[str] = ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]

# ------------------------------------------------------------
# Data-leakage policy (the part I care about most)
# ------------------------------------------------------------
# The model should decide from real measurements, not from columns that already
# hold the answer the vetting team reached. So I drop three kinds of columns:
#
#   1. Identifiers          - names/ids the model could just memorize.
#   2. Vetting conclusions  - basically the label in disguise.
#   3. Provenance/metadata  - bookkeeping text, not physics.
#
# The sneaky ones worth spelling out:
#   - kepler_name      only confirmed planets ever get a name, so keeping it would
#                      let the model "predict" CONFIRMED just by checking for a name.
#   - koi_pdisposition the Kepler pipeline's own CANDIDATE/FALSE-POSITIVE call.
#   - koi_fpflag_*     the four robovetter flags. koi_pdisposition is FALSE POSITIVE
#                      whenever any flag is set, so these are the conclusion, not raw
#                      data. I drop them and let the model learn from the raw
#                      measurements behind them (centroid offsets, secondary-eclipse
#                      depth, odd/even significance, transit shape).
#   - koi_comment      notes like "DEEP_V_SHAPED" that literally say why it failed.
#   - koi_score        already removed from this file; I never recreate it.
# ------------------------------------------------------------

IDENTIFIER_COLS: list[str] = [
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",
]

# Columns that encode the vetting *decision* (label leakage).
LEAKAGE_COLS: list[str] = [
    "koi_pdisposition",   # pipeline disposition (the answer minus CONFIRMED)
    "koi_fpflag_nt",      # robovetter: not transit-like
    "koi_fpflag_ss",      # robovetter: stellar eclipse / significant secondary
    "koi_fpflag_co",      # robovetter: centroid offset
    "koi_fpflag_ec",      # robovetter: ephemeris match / contamination
    "koi_vet_stat",       # vetting status
    "koi_vet_date",       # vetting date
    "koi_disp_prov",      # disposition provenance
    "koi_comment",        # free-text vetting comments  <- strong leak
    "koi_score",          # disposition confidence (removed in this file; guard anyway)
]

# Non-physical bookkeeping / provenance / model-name strings.
METADATA_COLS: list[str] = [
    "koi_fittype",        # fit method (e.g. LS+MCMC)
    "koi_limbdark_mod",   # limb-darkening model citation
    "koi_trans_mod",      # transit model citation
    "koi_parm_prov",      # parameter provenance
    "koi_sparprov",       # stellar parameter provenance
    "koi_tce_delivname",  # TCE delivery name
    "koi_quarters",       # 32-char observed-quarter bitmask (string)
    "koi_datalink_dvr",   # link to DV report PDF
    "koi_datalink_dvs",   # link to DV summary PDF
]

# Convenience: every column that must never reach the model.
DROP_COLS: list[str] = IDENTIFIER_COLS + LEAKAGE_COLS + METADATA_COLS

# ------------------------------------------------------------
# Cleaning thresholds
# ------------------------------------------------------------
MISSING_DROP_THRESHOLD: float = 0.60   # drop columns that are >60% empty
NEAR_CONSTANT_THRESHOLD: float = 0.999  # drop columns where one value dominates
CORR_DROP_THRESHOLD: float = 0.97       # one of each near-duplicate pair is dropped

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
PRIMARY_METRIC: str = "f1_weighted"  # multi-class -> weighted F1 (per the brief)
