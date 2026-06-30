"""
Exoplanet Classifier - interactive Streamlit dashboard.

Pages
-----
1. Overview            project framing + headline test metrics.
2. EDA                 interactive Plotly exploration of the KOI dataset.
3. Feature Explorer    pick any two features, colour by disposition.
4. Predict             manual KOI form -> disposition, probabilities, SHAP.
5. Model Comparison    cross-validated scoreboard + model trade-off notes.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Make ``src`` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.models import MODEL_NOTES  # noqa: E402
from src.predict import load_metadata, load_model, predict_one  # noqa: E402
from src.utils import PALETTE  # noqa: E402

st.set_page_config(
    page_title="Exoplanet Classifier",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DATA_PATH, low_memory=False)


@st.cache_resource(show_spinner=False)
def get_model():
    try:
        return load_model(), load_metadata()
    except Exception:  # noqa: BLE001 - model not trained yet
        return None, None


@st.cache_data(show_spinner=False)
def get_bonus():
    """Load the complementary binary / validation models + leakage benchmark."""
    import json

    out = {}
    for key, fn in [("binary", "binary_metadata.json"),
                    ("validation", "validation_metadata.json")]:
        p = config.MODELS_DIR / fn
        if p.exists():
            out[key] = json.loads(p.read_text(encoding="utf-8"))
    lk = config.REPORTS_DIR / "leakage_benchmark.csv"
    out["leakage"] = pd.read_csv(lk, index_col=0) if lk.exists() else None
    return out


def _plot(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"`{path.name}` not found - run `python -m src.train` to generate it.")


def _render_shap(model, values: dict) -> None:
    """Show the top SHAP contributions for a single manual prediction."""
    try:
        import shap

        from src.preprocessing import get_feature_names

        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]
        X = pd.DataFrame([values])
        X_trans = pd.DataFrame(pre.transform(X), columns=get_feature_names(pre))

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_trans)
        pred_int = int(model.predict_proba(X)[0].argmax())

        arr = np.asarray(sv)
        contrib = arr[:, :, pred_int][0] if arr.ndim == 3 else (
            sv[pred_int][0] if isinstance(sv, list) else arr[0]
        )
        s = (
            pd.Series(contrib, index=X_trans.columns)
            .sort_values(key=np.abs, ascending=False)
            .head(12)
        )
        fig = px.bar(
            x=s.values[::-1], y=s.index[::-1], orientation="h",
            color=s.values[::-1], color_continuous_scale="RdBu",
            labels={"x": "SHAP value (impact on prediction)", "y": ""},
            title=f"Why {config.CLASS_ORDER[pred_int]}? Top feature contributions",
        )
        fig.update_layout(template="plotly_dark", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"(SHAP explanation unavailable: {exc})")


# Headline interpretable features for the manual form (model imputes the rest).
FORM_FEATURES = [
    ("koi_period", "Orbital period (days)", 0.5, 700.0),
    ("koi_duration", "Transit duration (hrs)", 0.1, 20.0),
    ("koi_depth", "Transit depth (ppm)", 1.0, 50000.0),
    ("koi_prad", "Planet radius (R⊕)", 0.1, 40.0),
    ("koi_ror", "Planet/star radius ratio", 0.001, 0.5),
    ("koi_impact", "Impact parameter", 0.0, 2.0),
    ("koi_model_snr", "Transit SNR", 1.0, 2000.0),
    ("koi_num_transits", "Number of transits", 1.0, 1000.0),
    ("koi_max_mult_ev", "Max MES (multi-event)", 1.0, 2000.0),
    ("koi_teq", "Equilibrium temp (K)", 100.0, 3000.0),
    ("koi_insol", "Insolation (Earth flux)", 0.01, 10000.0),
    ("koi_steff", "Stellar Teff (K)", 3000.0, 8000.0),
    ("koi_slogg", "Stellar log g", 3.0, 5.0),
    ("koi_srad", "Stellar radius (R☉)", 0.1, 10.0),
    ("koi_dicco_msky", "Centroid offset (arcsec)", 0.0, 5.0),
]


# ------------------------------------------------------------
# Sidebar / navigation
# ------------------------------------------------------------
st.sidebar.title("🪐 Exoplanet Classifier")
st.sidebar.caption("NASA Kepler KOI disposition predictor")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA", "Feature Explorer", "Predict", "Model Comparison"],
)
model, meta = get_model()
if meta:
    st.sidebar.success(
        f"Model: {meta['model_name']}\n\n"
        f"Test weighted-F1: {meta['test_metrics']['f1_weighted']:.3f}"
    )
else:
    st.sidebar.warning("No trained model found.\nRun `python -m src.train`.")


# ------------------------------------------------------------
# 1. Overview
# ------------------------------------------------------------
if page == "Overview":
    st.title("Exoplanet Classification with NASA Kepler Data")
    st.markdown(
        """
        This dashboard predicts the **disposition** of a Kepler Object of Interest
        (KOI) - `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE` - from the physical
        properties of its transit signal and host star.

        The model is deliberately **leakage-safe**: it never sees the vetting
        pipeline's own conclusions (`koi_pdisposition`, the `koi_fpflag_*` flags,
        `koi_score`, free-text comments) or identifiers like `kepler_name`. It
        reasons only from raw measurements a scientist would trust.
        """
    )
    if meta:
        m = meta["test_metrics"]
        st.subheader(f"Primary model · {meta['model_name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Weighted F1", f"{m['f1_weighted']:.3f}")
        c2.metric("Accuracy", f"{m['accuracy']:.3f}")
        c3.metric("ROC-AUC (OvR)", f"{m['roc_auc_ovr_weighted']:.3f}")
        c4.metric("Features", f"{meta['n_features']}")
        st.caption(
            f"Imbalance strategy: {meta['imbalance_strategy']} · "
            f"train={meta['n_train']:,} / test={meta['n_test']:,} · "
            f"train/test F1 gap: {meta.get('overfit_gap', float('nan')):.3f} (low = generalizes well)"
        )

        # All three leakage-free framings - including the binary models that clear 0.90+.
        bonus = get_bonus()
        rows = [{
            "Model": "3-class · CONFIRMED / CANDIDATE / FALSE POSITIVE",
            "Train F1": meta.get("train_f1_weighted", float("nan")),
            "Test F1": m["f1_weighted"], "Test acc": m["accuracy"],
            "ROC-AUC": m["roc_auc_ovr_weighted"],
        }]
        if bonus.get("binary"):
            bm = bonus["binary"]["metrics"]
            rows.append({"Model": "Binary · REAL vs FALSE POSITIVE", "Train F1": bm["train_f1"],
                         "Test F1": bm["f1"], "Test acc": bm["accuracy"], "ROC-AUC": bm["roc_auc"]})
        if bonus.get("validation"):
            vm = bonus["validation"]["metrics"]
            rows.append({"Model": "Validation · CONFIRMED vs FALSE POSITIVE", "Train F1": vm["train_f1"],
                         "Test F1": vm["f1"], "Test acc": vm["accuracy"], "ROC-AUC": vm["roc_auc"]})
        st.markdown("**All three leakage-free framings** (the two binary models clear 0.90 on train *and* test):")
        st.dataframe(
            pd.DataFrame(rows).set_index("Model").style.format("{:.3f}")
            .background_gradient(subset=["Test F1"], cmap="Greens"),
            use_container_width=True,
        )
    df = get_data()
    counts = df[config.TARGET].value_counts().reindex(config.CLASS_ORDER)
    fig = px.bar(
        counts, color=counts.index, color_discrete_map=PALETTE,
        labels={"value": "Count", "index": "Disposition"},
        title="Target class distribution",
    )
    fig.update_layout(showlegend=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    df = get_data()
    st.markdown(f"**Dataset:** {df.shape[0]:,} KOIs × {df.shape[1]} columns.")

    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Saved figures"])

    with tab1:
        feat = st.selectbox(
            "Feature",
            ["koi_period", "koi_depth", "koi_prad", "koi_model_snr", "koi_teq",
             "koi_insol", "koi_duration", "koi_steff"],
        )
        logx = st.checkbox("Log-scale x", value=True)
        plot_df = df[[feat, config.TARGET]].dropna()
        fig = px.violin(
            plot_df, x=config.TARGET, y=feat, color=config.TARGET,
            color_discrete_map=PALETTE, box=True, points=False,
            category_orders={config.TARGET: config.CLASS_ORDER},
        )
        if logx:
            fig.update_yaxes(type="log")
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Violin plots reveal how each physical quantity separates the classes "
            "- e.g. false positives often show extreme depths and radii."
        )

    with tab2:
        n = st.slider("Top-N most-variable numeric features", 8, 30, 16)
        num = df.select_dtypes("number")
        top = num.var().sort_values(ascending=False).head(n).index
        corr = num[top].corr()
        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto", title="Correlation heatmap",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Highly correlated pairs are pruned during preprocessing.")

    with tab3:
        _plot(config.PLOTS_DIR / "class_distribution.png", "Class distribution")
        _plot(config.PLOTS_DIR / "confusion_matrix.png", "Confusion matrix")
        _plot(config.PLOTS_DIR / "roc_curves.png", "ROC curves (OvR)")
        _plot(config.PLOTS_DIR / "pr_curves.png", "Precision-Recall curves (OvR)")


# ------------------------------------------------------------
# 3. Feature Explorer
# ------------------------------------------------------------
elif page == "Feature Explorer":
    st.title("Interactive Feature Explorer")
    df = get_data()
    numeric = sorted(df.select_dtypes("number").columns)
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", numeric, index=numeric.index("koi_period") if "koi_period" in numeric else 0)
    y = c2.selectbox("Y axis", numeric, index=numeric.index("koi_prad") if "koi_prad" in numeric else 1)
    logx = c1.checkbox("Log X", value=True)
    logy = c2.checkbox("Log Y", value=True)

    sample = df[[x, y, config.TARGET]].dropna()
    if len(sample) > 4000:
        sample = sample.sample(4000, random_state=config.RANDOM_STATE)
    fig = px.scatter(
        sample, x=x, y=y, color=config.TARGET, color_discrete_map=PALETTE,
        opacity=0.6, category_orders={config.TARGET: config.CLASS_ORDER},
        log_x=logx, log_y=logy, title=f"{y} vs {x}",
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# 4. Predict
# ------------------------------------------------------------
elif page == "Predict":
    st.title("Predict a KOI Disposition")
    if model is None:
        st.error("No trained model available. Run `python -m src.train` first.")
        st.stop()

    df = get_data()
    medians = df.median(numeric_only=True)

    st.markdown("Enter the KOI's measured properties - unspecified fields are imputed.")
    with st.form("predict_form"):
        cols = st.columns(3)
        values: dict[str, float] = {}
        for i, (key, label, lo, hi) in enumerate(FORM_FEATURES):
            default = float(medians.get(key, (lo + hi) / 2))
            default = float(np.clip(default, lo, hi))
            values[key] = cols[i % 3].number_input(
                label, min_value=float(lo), max_value=float(hi), value=default
            )
        submitted = st.form_submit_button("🔭 Classify", use_container_width=True)

    if submitted:
        result = predict_one(model, values)
        pred = result["predicted_disposition"]
        st.markdown(
            f"### Prediction: "
            f"<span style='color:{PALETTE[pred]}'>{pred}</span>",
            unsafe_allow_html=True,
        )
        st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.1%}")

        proba = result["probabilities"]
        fig = px.bar(
            x=list(proba.keys()), y=list(proba.values()),
            color=list(proba.keys()), color_discrete_map=PALETTE,
            labels={"x": "Disposition", "y": "Probability"},
            category_orders={"x": config.CLASS_ORDER},
        )
        fig.update_layout(template="plotly_dark", showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        _render_shap(model, values)


# ------------------------------------------------------------
# 5. Model Comparison
# ------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    comp_path = config.REPORTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path, index_col=0)
        st.dataframe(
            comp.style.format("{:.4f}").background_gradient(
                subset=["f1_weighted"], cmap="Greens"
            ),
            use_container_width=True,
        )
        _plot(config.PLOTS_DIR / "model_comparison.png", "Model comparison")
    else:
        st.info("Run `python -m src.train` to generate the comparison table.")

    # Bonus models + the documented leakage ceiling.
    bonus = get_bonus()
    if bonus.get("binary") or bonus.get("validation"):
        st.subheader("Bonus models - leakage-free framings")
        rows = []
        if bonus.get("binary"):
            bm = bonus["binary"]["metrics"]
            rows.append({"Model": "Binary · REAL vs FALSE POSITIVE", "Train F1": bm["train_f1"],
                         "Test F1": bm["f1"], "Accuracy": bm["accuracy"], "ROC-AUC": bm["roc_auc"]})
        if bonus.get("validation"):
            vm = bonus["validation"]["metrics"]
            rows.append({"Model": "Validation · CONFIRMED vs FALSE POSITIVE", "Train F1": vm["train_f1"],
                         "Test F1": vm["f1"], "Accuracy": vm["accuracy"], "ROC-AUC": vm["roc_auc"]})
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"),
                     use_container_width=True)
        lk = bonus.get("leakage")
        if lk is not None:
            st.caption("Leakage ceiling - the score the dropped 'answer-key' flags would buy "
                       "(excluded on purpose for a trustworthy model):")
            st.dataframe(lk.style.format("{:.3f}"), use_container_width=True)

    st.subheader("Model trade-offs")
    notes = pd.DataFrame(MODEL_NOTES).T
    st.dataframe(notes, use_container_width=True)
