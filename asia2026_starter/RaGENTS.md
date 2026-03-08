For scp command, have of the format scp -r ppurkayastha@euler.ethz.ch <List of files> Desktop/<Exact file name csv>


# How to implement local testing

Yes — you can (and should) build **strong local validation** even when the final Kaggle test labels are hidden. The fix is to make every pipeline produce **OOF (out-of-fold) predictions on the training set** and score them on the **same cells Kaggle likely scores** (the **imputed-only** cells), plus add a “mask-and-reconstruct” proxy for imputation models like SAITS.

Also: don’t rely on public LB for iteration; Kaggle explicitly uses a public/private split and warns about public-LB overfitting.

Below is a **drop-in validation scheme for Tasks 1–4** (your Stage 1–4).

---

## Common evaluation harness you should add once (used by Tasks 1–4)

### A) Define the two masks (this is everything)

For each target column `c`:

- **Observed mask** (copy-through territory): `obs = X_features[c].notna()`
- **Imputed-only mask** (what matters): `imp = X_features[c].isna()`

Then compute RMSE **only on `imp`**:

RMSEimputed-only=1∣imp∣∑(i,c)∈imp(yi,c−y^i,c)2\text{RMSE}_{\text{imputed-only}} = \sqrt{\frac{1}{|\text{imp}|}\sum_{(i,c)\in \text{imp}} (y_{i,c}-\hat y_{i,c})^2}

RMSEimputed-only=∣imp∣1(i,c)∈imp∑(yi,c−y^i,c)2

This aligns much better with leaderboard RMSE in your own experiments.

### B) Always output these artifacts per run

- `oof_predictions_train.parquet` (or `.npz`) ← per row × target, from CV
- `predictions_test.csv` ← Kaggle submission
- `run_summary.json` with:
    - `rmse_sensory_imputed_only`
    - `rmse_all_imputed_only`
    - (optional) per-time and per-target breakdown

### C) Use one split strategy everywhere (stable + honest)

1. `StratifiedKFold`like behavior by time:
    - do `KFold(n_splits=5, shuffle=True, random_state=42)` **but** ensure each fold has similar time distribution (you can implement by splitting within each time bucket then concatenating).

This reduces leakage and makes OOF closer to Kaggle.

---

# Task 1 — TabPFN (discrete / seed ensemble / probability averaging)

### What to validate locally

**OOF imputed-only RMSE** for your final “seedbag + prob-avg” logic.

### How (exact)

1. Build a **5-fold splitter** (as per common harness).
2. For each fold:
    - Train your TabPFN discrete model(s) on fold-train.
    - Predict fold-val **for all targets**.
    - Apply copy-through on fold-val:
        - If `X_val[c]` observed → set `pred_val[c] = X_val[c]`.
    - Store into `OOF[val_idx, :] = pred_val`.
3. After all folds:
    - Compute `rmse_sensory_imputed_only` on OOF using `imp = X_train_features.isna()`.
4. Separately (for the submission):
    - Train full-train models for each seed (or reuse your full-train seedbag runner) and write `predictions_test.csv`.

### Extra local proxy (fast, catches SAITS-style wiring bugs too)

**Mask-and-reconstruct** on train:

- Randomly hide 10% of **observed** sensory cells in `X_train_features` (only sensory columns).
- Run your inference pipeline (same as test inference).
- RMSE on only the artificially hidden cells.
    
    This is a good sanity proxy when you change preprocessing/mapping.
    

---

# Task 2 — CatBoost “blocks” (or any tree partner)

### What to validate locally

OOF imputed-only RMSE. Do **not** look at train-fit error.

### How (exact)

1. Use the same 5-fold split.
2. For each fold:
    - Train your 4 CatBoost block models on fold-train labels.
        - If you use multi-target: `loss_function="MultiRMSE"` is correct. (Kaggle docs about RMSE + overfitting; and CatBoost multi-regression docs are the right reference if you want to cite in your write-up; you already know it works.)
    - Predict fold-val for all sensory targets.
    - Apply copy-through on fold-val.
    - Write into OOF.
3. Compute `rmse_sensory_imputed_only`.

### Why this is necessary

Your catastrophic public score means the current CatBoost model is either:

- not using enough signal, or
- misaligned with the “imputed-only” objective.
    
    OOF imputed-only RMSE will tell you immediately if it’s actually learning anything useful.
    

---

# Task 3 — Blending/stacking (TabPFN + CatBoost + optional SAITS)

### What to validate locally

Blend weights must be fitted on **OOF predictions**, not on full-train predictions (otherwise leakage).

### How (exact)

1. Produce OOF predictions for each component model you want to blend:
    - `OOF_tabpfn`
    - `OOF_catboost`
    - (optional) `OOF_saits`
2. Fit blend weight(s) on **imputed-only cells only**:
    - simplest: global alpha in {0.6, 0.7, 0.8, 0.9}
    - compute RMSE_imputed_only for each alpha and pick best
3. For test submission:
    - blend the corresponding **test** predictions with the chosen alpha
    - apply copy-through on test at the end

This is the standard “don’t overfit public LB” discipline Kaggle warns about.

---

# Task 4 — SAITS (structure-aware imputer)

SAITS is an imputer; you can validate it very well locally if you set up the mask and loss correctly. SAITS is trained with a mask-aware objective and is meant to be evaluated on missing positions.

### What to validate locally (two tests)

### Test 4A: Supervised OOF (best)

Use full train labels as ground truth:

1. For each fold:
    - Inputs: `X_val_partial` = the expedited sensory grid (with NaNs) + motor/meta/time
    - Targets: `Y_val_full` = full sensory labels
    - Loss/metric: compute only on `imp = X_val_partial is NaN` (imputed-only)
2. Store predictions and compute OOF `rmse_sensory_imputed_only`.

### Test 4B: Mask-and-reconstruct (fast sanity)

1. Take train features.
2. Randomly hide 10–20% of *observed* sensory cells.
3. Run SAITS.
4. RMSE on the artificially hidden cells.
    
    If this is bad, your dermatome/channel mapping or mask logic is wrong.