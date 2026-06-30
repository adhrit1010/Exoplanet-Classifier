# Executive Report - Exoplanet Classification (NASA Kepler KOI)

**Author:** Adhrit Chakraborty · **Model:** Hist Gradient Boosting · **Generated from live artifacts**

## Problem
NASA's *Kepler* mission flagged thousands of periodic dips in stellar brightness as
candidate transiting planets - **Kepler Objects of Interest (KOIs)**. Each must be
dispositioned as a **CONFIRMED** planet, a **CANDIDATE**, or a **FALSE POSITIVE**
(eclipsing binaries, instrumental artifacts, background contamination). Manual
vetting is slow and subjective. This project delivers a leakage-safe, explainable
classifier that reproduces those dispositions from physical measurements alone,
triaging which signals deserve scarce telescope follow-up.

## Data & EDA
The KOI Cumulative table holds **9,564** labelled objects across ~140
columns spanning transit, host-star and centroid diagnostics. EDA drove every downstream
choice via three facts: **class imbalance** (FALSE POSITIVE ~51%, CONFIRMED ~29%,
CANDIDATE ~21%); **heavy right-skew** over orders of magnitude (period, depth,
insolation); and a **cluster of near-empty columns**. Violin/scatter plots confirmed the
classes are genuinely separable - false positives skew to star-sized radii and depths.

## Cleaning & the Leakage Firewall
The most important decision was **refusing to cheat**. Columns that silently encode the
answer - `kepler_name` (only confirmed planets are named), `koi_pdisposition`, the four
`koi_fpflag_*` robovetter *conclusions*, `koi_score`, free-text `koi_comment` - are
removed **inside** the serialized pipeline so they can never leak. The model instead
learns from the **raw measurements** those flags were derived from (centroid offsets,
secondary-eclipse depth, odd/even significance, transit shape). After pruning
empty/constant/over-missing columns, median imputation, scaling and correlation pruning,
**91 model features** remain. Re-introducing the `koi_fpflag_*` flags would lift test weighted F1 from 0.858 to **0.947** (a **+0.089 leakage premium**) - a shortcut we deliberately forgo.

## Feature Engineering & Selection
We added physically-motivated features: log transforms of skewed quantities; the
transit **duty cycle**; **depth signal-to-noise**; **centroid-offset significance**
(a background-binary tell); and ordered **temperature / period / radius classes** plus
a habitable-zone flag. An **ablation** (3-fold CV) measures their effect honestly: weighted F1 is **0.853** with engineered features vs **0.848** without - a **+0.005** weighted-F1 improvement. They remain informative (SHAP ranks `duty_cycle` among the top features) and interpretable, which is why they are retained.

Selection is explicit, not assumed: a **CorrelationPruner** drops one of every
near-duplicate (|corr| > 0.97) pair - mostly the redundant `*_err1`/`*_err2`
uncertainty twins - and a **Mutual Information** ranking
(`outputs/reports/mutual_information.csv`) plus **permutation importance**
(`outputs/reports/permutation_importance.csv`) audit which features actually carry signal.

## Model Selection
Twelve models were benchmarked under identical preprocessing with stratified 5-fold
cross-validation on the **training split only**. Gradient-boosted trees dominated;
the top performers were **Hist Gradient Boosting (F1=0.859), LightGBM (F1=0.858), XGBoost (F1=0.856)**. In honesty, the leading boosters sit within
**~0.002** (one CV standard deviation) of one another, so the choice is not
statistically decisive - Hist Gradient Boosting is selected as the marginal CV leader and one of
the fastest to train. Tree ensembles suit this heterogeneous, non-linear tabular data.

## Class Imbalance
We compared *no resampling*, *class weighting*, *SMOTE*, *ADASYN*, *random over-* and
*under-sampling* by cross-validated weighted F1. All six landed within one CV standard
deviation of each other - i.e. **resampling gave no measurable weighted-F1 benefit** on
these boosters. We therefore ship **class_weight** (rather than oversampling): it up-weights
the minority CANDIDATE class to improve its recall *without* duplicating rows, which also
avoids the extra memorization that random oversampling can induce.

## Hyperparameter Tuning
**Optuna** (TPE, **30 trials**) tuned Hist Gradient Boosting over a deliberately
**regularization-first** space - shallow trees, large `min_samples_leaf`, L2, feature
subsampling, **early stopping**. Critically, the objective **penalized the
train/validation gap**, so the search actively preferred configurations that generalize
rather than the highest raw CV score. (Zoo compared at 5-fold; tuning at 3-fold.)

## Evaluation
On the **held-out 1,913-row test set** the final model achieves:

| Metric | Score |
|---|---|
| Accuracy | 84.9% |
| Precision (weighted) | 86.5% |
| Recall (weighted) | 84.9% |
| **F1 (weighted)** | **85.4%** |
| F1 (macro) | 83.1% |
| ROC-AUC (OvR, weighted) | 0.964 |


Per-class test F1 (the number weighted-F1 hides):

| Class | F1 |
|---|---|
| FALSE POSITIVE | 0.889 |
| CANDIDATE | 0.701 |
| CONFIRMED | 0.903 |

Train weighted F1 is **0.885** vs test **0.854** (gap **0.031**); the regularization-first search (shallow trees, large `min_samples_leaf`, L2, column subsampling, early stopping) keeps this gap in check rather than the much larger gap an unregularized booster produces.

A complementary **leakage-free binary model** (REAL vs FALSE POSITIVE - the challenge's stated mission) reaches **F1 0.905, ROC-AUC 0.971** (`outputs/reports/bonus_analysis.md`).

Confusion-matrix, ROC, precision-recall, learning, validation and calibration curves
are saved under `outputs/plots/`.

## Explainability
**SHAP** (global + per-class + local) and **permutation importance** agree on the top
drivers: the **MES**, the engineered **duty cycle**, **centroid-offset** measurements,
**planet radius/depth** and **transit count** - the quantities a human vetter weighs.
Honesty caveat: a stellar *uncertainty* column ranks high, likely catalog/provenance
signal rather than physics, and is flagged for future pruning.

## Error Analysis
Residual errors concentrate at the **CANDIDATE ↔ CONFIRMED** boundary and at lower
confidence - the *honest* failure mode, since a candidate is merely a not-yet-confirmed
planet. FALSE POSITIVE is separated cleanly. With vetting columns removed there is no
detectable leakage; the modest train-vs-test gap and the learning curve
(`outputs/plots/learning_curve.png`) confirm the model generalizes rather than memorizes.

## Scientific Impact, Limitations & Future Work
**Impact:** an explainable triage layer that prioritizes the most planet-like candidates
and flags likely impostors, saving scarce follow-up time. **Limitations:** labels carry
vetting bias; CANDIDATE is inherently fuzzy; the model is Kepler-specific; a stellar
uncertainty column still ranks high and warrants pruning. **Future work:** isotonic
calibration and per-class threshold tuning to lift CANDIDATE, conformal prediction for
guaranteed coverage, and cross-mission transfer to TESS (TOI).

## Deployment Strategy
The shipped artifact is a single `Pipeline(preprocessor, classifier)` accepting raw KOI
rows (any column subset) and returning a disposition, probabilities and a confidence
band. It is served via the **Streamlit dashboard** and a `python -m src.predict` CLI,
containerized with **Docker**, tested/linted in **CI**, and optionally tracked with
**MLflow**. Retraining is one command - fully reproducible end to end.
