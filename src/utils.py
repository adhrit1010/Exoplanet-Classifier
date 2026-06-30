"""Small shared helpers: label encoding, plotting style, timing, and IO."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import config

# Use the non-interactive Agg backend for scripts/CLI/training, where an
# interactive Tk backend crashes ("Tcl_AsyncDelete: ... wrong thread") in any
# process that uses threads (e.g. joblib's threading backend). Inside a Jupyter
# kernel we leave the inline backend untouched so notebook figures still render.
# Safe because pyplot is only imported lazily (after this) inside plot helpers.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")


# ------------------------------------------------------------
# Label encoding (stable, human-readable mapping shared everywhere)
# ------------------------------------------------------------
LABEL_TO_INT: dict[str, int] = {name: i for i, name in enumerate(config.CLASS_ORDER)}
INT_TO_LABEL: dict[int, str] = {i: name for name, i in LABEL_TO_INT.items()}


def encode_labels(y) -> np.ndarray:
    """Map disposition strings to stable integer codes (per ``CLASS_ORDER``)."""
    return np.asarray(pd.Series(y).map(LABEL_TO_INT).to_numpy(), dtype=int)


def decode_labels(codes) -> np.ndarray:
    """Inverse of :func:`encode_labels`."""
    return np.asarray(pd.Series(np.asarray(codes)).map(INT_TO_LABEL).to_numpy())


# ------------------------------------------------------------
# Timing
# ------------------------------------------------------------
@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    elapsed = {"value": 0.0}
    try:
        yield lambda: time.perf_counter() - start
    finally:
        elapsed["value"] = time.perf_counter() - start


# ------------------------------------------------------------
# Plot styling - one consistent, presentation-ready look
# ------------------------------------------------------------
PALETTE: dict[str, str] = {
    "FALSE POSITIVE": "#ef476f",
    "CANDIDATE": "#ffd166",
    "CONFIRMED": "#06d6a0",
}
ACCENT = "#118ab2"


def set_plot_style() -> None:
    """Apply a clean, dark-friendly matplotlib/seaborn theme used in all plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.family": "DejaVu Sans",
        }
    )


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------
def save_json(obj, path: Path) -> None:
    """Write ``obj`` to ``path`` as pretty JSON (numpy-aware)."""

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    Path(path).write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def load_json(path: Path):
    """Read JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
