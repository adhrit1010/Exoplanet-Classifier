# Bonus Analysis - Binary Model & Leakage Ceiling
## Leakage-free binary model - REAL vs FALSE POSITIVE

The challenge's stated mission is to *"separate real exoplanet candidates from noise and false signals"* - a **binary** task. Collapsing CONFIRMED + CANDIDATE -> **REAL** vs **FALSE POSITIVE** turns the intrinsically fuzzy 3-class problem into a well-posed one, with **no leakage**.

| Metric | Test score |
|---|---|
| Accuracy | 0.906 |
| Precision | 0.907 |
| Recall | 0.903 |
| **F1** | **0.905** |
| ROC-AUC | 0.971 |
| Avg precision | 0.970 |
| CV F1 (5-fold) | 0.901 ± 0.008 |
| Train F1 / gap | 0.950 / 0.045 |

Figures: `outputs/plots/binary_confusion_matrix.png`, `outputs/plots/binary_roc.png`. Model: `models/exoplanet_binary.pkl`.

## Planet-validation model - CONFIRMED vs FALSE POSITIVE (>0.95)

Restricting to the two *decided* classes (excluding the ~2k undecided CANDIDATEs) yields a well-posed, genuinely separable problem - *"is this a validated planet or a false positive?"* - and is the **only leakage-free framing that clears 0.95 on both train and test**:

| Metric | Test score |
|---|---|
| Accuracy | 0.974 |
| Precision | 0.967 |
| Recall | 0.960 |
| **F1** | **0.964** |
| ROC-AUC | 0.997 |
| CV F1 (5-fold) | 0.965 ± 0.006 |
| **Train F1 / Test F1** | **0.994 / 0.964** (gap 0.031) |

Trade-off: it answers a narrower question than the 3-class model, so it is a *complementary* result, not a replacement. Model: `models/exoplanet_validation.pkl`.

## Leakage ceiling - why we exclude the robovetter flags

The posted rules do not ban any column, but the `koi_fpflag_*` / `koi_pdisposition` columns *are* the vetting conclusion. Training the 3-class model **with** them re-introduced shows the score they would buy - the **leakage premium** - which is exactly why they are excluded from the shipped model:

| Configuration | Train F1 | Test F1 | Test Acc |
|---|---|---|---|
| Leakage-free (shipped) | 0.953 | 0.858 | 0.861 |
| WITH flags (leakage) | 0.985 | 0.947 | 0.947 |

**Leakage premium: +0.089 test weighted F1.** We forgo it: a model a scientist can trust beats a higher number it can't.
