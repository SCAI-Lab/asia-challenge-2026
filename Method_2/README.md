# Method 2 - TabPFN discrete seedbag5-proba

## Executive summary

Method 2 is the second final Track 2 method preserved in this bundle.

Instead of a fold-bagged model plus post-processing chain, this method keeps the modeling pipeline much simpler:

- train the same general discrete TabPFN target formulation,
- but fit **multiple full-train models with different random seeds**,
- average the **class probabilities** across seeds,
- convert those averaged probabilities into expected sensory values,
- then apply only the standard safety steps:
  - copy-through of observed follow-up values,
  - clipping to legal target ranges.

This gives a strong alternative to Method 1 because it changes the bias-variance trade-off:

- **Method 1** = more structure, more explicit post-processing, more handcrafted corrections,
- **Method 2** = less hand-tuning, more full-data ensembling, probability-level averaging.

## Why this method exists

Bagged CV models and full-train seed ensembles are not the same thing.

Method 2 was kept because it gives a genuinely different hedge:

- every seed model sees the **full training set**,
- randomness is diversified through seed changes rather than fold subsampling,
- averaging is done at the **probability** level rather than on already-collapsed expected values,
- and there is no extra Track-1-derived anchor chain layered on top.

That makes Method 2 a valuable second solution family even when Method 1 is stronger on some observed scores.

## Core configuration

From the code in `Method_2/code/run_tabpfn_t2_discrete_seedbag5_proba.py`:

- seeds: **[11, 22, 33, 44, 55]**
- CV seed: **42**
- CV splits: **5**
- default TabPFN estimators: **8** (when supported by the installed version)
- device: **CUDA**
- probability averaging across seeds, then expected-value projection to target scale

## Why probability averaging

For discrete sensory targets, averaging probabilities is cleaner than averaging already-decoded class expectations from heterogeneous models.

The method does:

1. obtain class probabilities for each seed,
2. align them to the fixed target classes,
3. average the full probability vectors,
4. convert the averaged vector to an expected target value.

This is often a better ensemble operator for low-cardinality targets because it preserves uncertainty structure longer.

## Environment and runtime

Operational notes recorded for this method:

- primary GPU: **RTX 4090 (24 GB)**
- runtime on RTX 4090: **~3 hours**
- reproduced on **RTX 2080** in **~6 hours**
- requires the same TabPFN/CUDA environment as Method 1 base model

## Bundled final output

- `data/submissions/predictions_test__t2_discrete_seedbag5_proba__20260224_141548__job58350309__01n1mgw8.csv`

## What this method is *not*

This method does **not** include the Method 1 post-processing chain in the shipped bundle.

That is intentional. It was preserved as the cleaner alternative family:
- same task,
- same data,
- same overall discrete target philosophy,
- but fewer handcrafted structural interventions.

## How to reproduce Method 2

### Recommended
```bash
sbatch Method_2/slurm/23b_track2_tabpfn_discrete_seedbag5_proba.sbatch
```

### Direct python
```bash
python Method_2/code/run_tabpfn_t2_discrete_seedbag5_proba.py --data-root data --run-root runs --do-cv 1 --n-splits 5
```

## Things to note for release and verification

- do not publish an HF token,
- document the need for authorized TabPFN weight access,
- keep the exact seed list unchanged for faithful reproduction,
- keep the copy-through + clipping steps unchanged, because they are part of the final prediction contract.

## Read next

- `docs/run_tabpfn_t2_discrete_seedbag5_proba.md`
