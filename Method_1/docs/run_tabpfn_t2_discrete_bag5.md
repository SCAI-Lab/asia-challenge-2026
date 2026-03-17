# run_tabpfn_t2_discrete_bag5.py

## Purpose

This script is the **Method 1 base model**. It trains the main Track 2 predictor using a **5-fold bagged discrete TabPFN formulation**.

The design choice is:

- treat the sensory outputs as discrete states,
- let TabPFN model nonlinear tabular interactions,
- produce soft expected values rather than hard snapped labels,
- and preserve any observed follow-up targets by copy-through.

## Data situation this script is solving

From the bundled Track 2 data:

- training rows: **931**
- test rows: **252**
- targets: **112**
- mean test target-feature missingness: **0.858**
- targets with test missingness >= 0.85: **80**
- fully missing test targets: **24**

This is exactly why the script avoids naive regression-on-all-columns:

- the follow-up target columns themselves appear inside the feature matrix with many missing values,
- categorical and numeric metadata both matter,
- and leakage must be prevented whenever a target column exists as a feature.

## Core model design

### 1) discrete modeling
Each sensory target is modeled as a discrete 3-state problem with target classes:

- `0`
- `1`
- `2`

`anyana` is handled as a binary target with classes:

- `0`
- `1`

The script then converts class probabilities into expected values instead of hard class labels.

### 2) per-target leakage masking
For each target column, if the same-named feature exists after preprocessing, the script masks that transformed feature before fitting / predicting that target.

This is one of the most important design choices in the entire repo:
- it prevents trivial copy leakage,
- while still allowing copy-through to be applied explicitly after prediction.

### 3) preprocessing
The preprocessor is a `ColumnTransformer` with:

- categorical branch:
  - constant-fill imputation to `"MISSING"`
  - one-hot encoding
- numeric branch:
  - median imputation
  - missing-indicator expansion
  - standard scaling

That gives the model both:
- the imputed values,
- and the information that the value was originally missing.

### 4) bag5
The script uses:

- `KFold(n_splits=5, shuffle=True, random_state=42)`

For each fold:
- fit on train split,
- predict validation split for OOF,
- predict test set,
- average the test predictions across folds.

This reduces variance and provides useful out-of-fold diagnostics.

## Why this base model made sense

The choice of a discrete TabPFN base is justified by the task structure itself:

- the outputs are low-cardinality states,
- there are many nonlinear interactions between motor, sensory, metadata, and missingness,
- and the sample size is large enough for a strong prior model like TabPFN to be useful, but still small enough that classical huge-parameter deep setups are not obviously superior.

In other words, this script tries to solve the difficult part - the global tabular prediction problem - without immediately hard-coding spinal-cord rules into training.

## Safety / validity steps

After prediction, the script does two non-negotiable operations:

### copy-through
If a target value is already observed in the follow-up features, the final submission value is set to that observed value.

### clipping
- sensory targets are clipped to `[0, 2]`
- `anyana` is clipped to `[0, 1]`

These keep the outputs valid and consistent with the competition target space.

## Environment

Main requirements:
- Python 3.11
- CUDA GPU
- `tabpfn==6.4.1`
- `torch==2.4.1+cu121`
- `scikit-learn==1.8.0`
- `pandas==2.3.3`
- `numpy==2.4.3`

The script also forces single-threaded worker settings for stability:
- `TABPFN_NUM_WORKERS=1`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`

## How to run

```bash
python Method_1/code/run_tabpfn_t2_discrete_bag5.py --data-root data --run-root runs
```

Or via Slurm:

```bash
sbatch Method_1/slurm/23_track2_tabpfn_discrete_bag5.sbatch
```

## Notes / release-quality cautions

- this script depends on authorized TabPFN model access,
- the token must not be embedded in a public README,
- the script is the correct base for the later Method 1 post-processing chain,
- and the model logic itself looks internally consistent with the task.

No major algorithmic red flag stands out here from the bundled code.
