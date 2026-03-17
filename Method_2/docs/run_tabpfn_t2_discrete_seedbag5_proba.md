# run_tabpfn_t2_discrete_seedbag5_proba.py

## Purpose

This script is the full implementation of **Method 2**.

It trains a **seedbag5 probability-averaged discrete TabPFN ensemble** for Track 2.

Instead of using fold-averaged models as the final inference engine, it trains multiple full-data models with different random seeds and averages their class probabilities.

## Why this script exists alongside Method 1

Method 1 and Method 2 are intentionally different kinds of robustness:

- **Method 1** emphasizes explicit structural corrections after a bagged base model,
- **Method 2** emphasizes a cleaner full-train ensemble with probability averaging and minimal manual intervention.

This gives two distinct solution families from the same data and overall model class.

## Exact configuration in the bundled code

- seed ensemble: **[11, 22, 33, 44, 55]**
- CV seed: **42**
- CV folds: **5**
- default TabPFN estimator count: **8** when supported
- device: **CUDA**
- target family:
  - sensory targets as 3-class discrete problems
  - `anyana` as binary

## What the script does

### 1) preprocess once
It builds the feature space once using the shared TabPFN discrete preprocessor:
- categorical -> constant fill + one-hot encode
- numeric -> median impute + missing indicators + standard scaling

### 2) target-wise leakage masking
For each target, the transformed same-named numeric feature is masked before fit/predict if present.

This is the same crucial anti-leakage idea as in the Method 1 base model.

### 3) train one model per seed
For each seed in `[11, 22, 33, 44, 55]`:
- set NumPy / Torch RNG state,
- fit a TabPFN classifier for each target,
- predict class probabilities on validation/test.

### 4) probability averaging
For each target:
- accumulate probability vectors across seeds,
- average them,
- convert the averaged probabilities to the final expected target value.

This is a principled ensemble strategy for low-cardinality outputs.

### 5) safety post-processing
Finally the script:
- copies through any observed follow-up targets,
- clips sensory outputs to `[0, 2]`,
- clips `anyana` to `[0, 1]`.

## Why seedbag, not just one seed

Using multiple seeds gives:
- lower variance than a single fit,
- full-data training for every member of the ensemble,
- and diversity without changing the data partition itself.

This is especially useful when the target space is discrete and probability mass can shift slightly between seeds.

## Why probability averaging, not mean-of-predictions

Averaging probabilities preserves calibration information longer. For example:
- two models that split mass differently between `1` and `2` may have similar expectations,
- but the probability average is a better representation of ensemble uncertainty than averaging already-decoded expectations.

That is why the script accumulates full class probabilities before projecting to the expected value.

## CV and diagnostics

This script can also run CV (`--do-cv 1 --n-splits 5`) and writes:

- fold metrics,
- overall CV metrics,
- weighted imputed-only WRMSE diagnostics,
- OOF artifact files.

That makes it useful both as an inference script and as an evaluation script.

## Environment

Main requirements:
- Python 3.11
- CUDA GPU
- `tabpfn==6.4.1`
- `torch==2.4.1+cu121`
- `scikit-learn==1.8.0`
- `pandas==2.3.3`
- `numpy==2.4.3`

Operational notes recorded for this method:
- about **3 hours** on RTX 4090
- about **6 hours** on RTX 2080

## How to run

```bash
python Method_2/code/run_tabpfn_t2_discrete_seedbag5_proba.py --data-root data --run-root runs --do-cv 1 --n-splits 5
```

Or:

```bash
sbatch Method_2/slurm/23b_track2_tabpfn_discrete_seedbag5_proba.sbatch
```

## Readme-level judgment

The script is internally consistent and represents a clean alternate final method. The main non-modeling caution is again packaging hygiene:
- do not expose the HF token,
- and make sure the repo docs name this script correctly in the final public bundle.
