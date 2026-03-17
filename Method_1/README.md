# Method 1 - TabPFN discrete bag5 -> hedge pairwise shrink -> anchor correction -> extend observed anchors

## Executive summary

Method 1 is the main **structured refinement chain** in this Track 2 bundle.

It starts from a strong **TabPFN discrete bag5** base model and then adds three increasingly targeted post-processing stages:

1. **base model**: learn a robust tabular predictor over all targets using TabPFN in a 5-fold bagged setup,
2. **hedge pairwise shrink**: softly smooth highly missing symmetric pairs toward each other when both sides are missing,
3. **anchor correction**: use observed paired follow-up targets to correct one-sided missing targets with hand-selected rules,
4. **extend observed anchors**: apply a final, small, upward-only adjustment on a narrow set of targets where the anchor suggests the previous stage is still too low.

The philosophy is simple:

- let the base model do the heavy lifting,
- use ISNCSCI structure only in **small, conservative** ways,
- and reserve stronger post-processing for target/source pairs where the train data gives a compelling reason.

## Inputs and outputs

### Inputs
- `Track_2/data/features_train_2.csv`
- `Track_2/data/features_test_2.csv`
- `Track_2/data/labels_train_2.csv`
- `Track_2/data/metadata_train_2.csv`
- `Track_2/data/metadata_test_2.csv`

## Why this method exists

Track 2 is not a generic tabular task. In the bundled data:

- the mean target-feature missingness in test is **0.858**,
- **80** targets have test missingness at least `0.85`,
- and **24** targets are fully missing in test.

At the same time, the data contains a lot of useful structure:
- left/right sensory symmetry,
- LT/PP within-level agreement,
- partial follow-up observations in upper/cervical regions,
- and baseline + motor context already provided by the track.

So Method 1 does **not** try to replace the learned model with hard rules.  
It uses rules only where the train-set structure is unusually strong.

## Why the base is discrete bag5

The base script treats the sensory outputs as discrete states and uses TabPFN as a **classifier-style estimator** on each target. This is well matched to the target space:

- sensory outputs are `0/1/2`,
- `anyana` is binary,
- and the model should respect the discrete nature of the labels while still producing soft expected values.

The bagging choice is:

- 5 folds,
- KFold with seed `42`,
- average test predictions across folds.

The point of bag5 is not to maximize individuality of each fold model. It is to reduce variance and smooth out fold idiosyncrasies before any structural post-processing.

## Why the three post-processing stages are ordered this way

The order is deliberate.

### Stage 1: pairwise hedge first
This is the broadest but weakest adjustment. It only nudges symmetric / modality-paired values toward each other, and only under strong gating. It is safe enough to run before anchor rules.

### Stage 2: anchor correction second
Once the pairwise jitter is reduced, targeted one-sided corrections become more reliable. This stage is stronger than stage 1 because it can move one target toward a learned conditional anchor.

### Stage 3: extend observed anchors last
This is the narrowest and smallest stage. It only fires when:
- the target is missing,
- the source is observed and reasonably strong,
- and the current prediction is still below the anchor by a margin.

This makes it a cleanup step, not a replacement for the earlier stages.

## What actually changed across the chain

Using the submission files included in the bundle:

- **base -> hedge**
  - changed cells: **18744**
  - affected rows: **252**
  - mean abs change on changed cells: **0.002352**
  - max abs change: **0.039429**

- **hedge -> anchor**
  - changed cells: **1775**
  - affected rows: **252**
  - mean abs change on changed cells: **0.030209**
  - max abs change: **0.862269**

- **anchor -> extend**
  - changed cells: **57**
  - affected rows: **48**
  - mean abs change on changed cells: **0.031674**
  - max abs change: **0.194050**

This is a healthy pattern:
- stage 1 is broad but tiny,
- stage 2 is targeted and meaningfully larger,
- stage 3 is very selective.

## Environment and runtime

Operational notes for this method:

- primary GPU: **RTX 4090 (24 GB)**
- typical runtime recorded for this method: **~1 hour**
- reproduced on **RTX 2080** in **~2 hours**
- base TabPFNv2.5 Classifier model run uses conda / CUDA environment
- post-processing scripts only need Python + pandas/numpy and can be run in a lighter venv

## How to run the pipelines

From the top-level `Track_2/` folder:

### Method 1
```bash
python Method_1/code/run_tabpfn_t2_discrete_bag5.py
python Method_1/code/run_t2_hedge_pairwise_shrink.py --base-csv <Output of previous run>
python Method_1/code/run_t2_anchor_correction.py --submission <Output of previous run>
python Method_1/code/run_t2_extend_obs_anchor.py --submission <Output of previous run>
```
