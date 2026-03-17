# run_t2_anchor_correction.py

## Purpose

This script is the **second post-processing layer** in Method 1.

It starts from the pairwise-hedged submission and applies a curated set of **one-sided conditional anchors**:

- if a target is missing in the test follow-up features,
- and a paired source target is observed,
- then the prediction is pulled toward `E[target | source]` estimated from the train labels.

This script is where the more explicit **Track-1-inspired but Track-2-applied anatomical heuristics** enter the pipeline.

## Why the idea made sense

The bundled Track 2 test matrix contains a very asymmetric upper block:

- some right-side upper sensory columns are observed much more often,
- while their paired left or PP counterparts remain heavily missing.

That makes one-sided target-source rules attractive in the upper cervical / upper thoracic region.

In practice, the biggest trigger counts in the bundled data are:

| target   | source   |   weight |   test_rows_triggered |   corr |   p_equal |   anchor_rmse |
|:---------|:---------|---------:|----------------------:|-------:|----------:|--------------:|
| c2ppr    | c2ltr    |     0.65 |                   206 |  0.744 |     0.997 |         0.073 |
| c2ltl    | c2ltr    |     0.7  |                   205 |  1     |     1     |         0     |
| c3ppr    | c3ltr    |     0.65 |                   154 |  0.891 |     0.986 |         0.118 |
| c3ltl    | c3ltr    |     0.7  |                   151 |  0.815 |     0.986 |         0.141 |
| c4ltl    | c4ltr    |     0.7  |                   123 |  0.784 |     0.972 |         0.197 |
| c4ppr    | c4ltr    |     0.6  |                   119 |  0.846 |     0.971 |         0.185 |
| c7ltl    | c7ltr    |     0.45 |                    80 |  0.89  |     0.887 |         0.381 |
| c7ppr    | c7ltr    |     0.2  |                    79 |  0.813 |     0.813 |         0.528 |
| c8ltl    | c8ltr    |     0.45 |                    71 |  0.915 |     0.876 |         0.355 |
| c8ppr    | c8ltr    |     0.25 |                    70 |  0.869 |     0.844 |         0.454 |
| t1ltl    | t1ltr    |     0.45 |                    65 |  0.896 |     0.884 |         0.391 |
| t1ppr    | t1ltr    |     0.25 |                    64 |  0.879 |     0.852 |         0.434 |
| t2ltl    | t2ltr    |     0.45 |                    63 |  0.903 |     0.89  |         0.383 |
| t2ppr    | t2ltr    |     0.25 |                    63 |  0.883 |     0.845 |         0.434 |
| t3ppr    | t3ltr    |     0.28 |                    59 |  0.862 |     0.845 |         0.467 |

## Interpreting the rule design

The rule list is concentrated in three zones:

1. **ultra-safe upper cervical**
   - `c2ltl <- c2ltr`
   - `c3ltl <- c3ltr`
   - `c4ltl <- c4ltr`
   - `c2/3/4 ppr <- corresponding right LT`
   - plus `c2/3/4 ppl <- corresponding left LT`

2. **mid-to-upper transition block**
   - `c7ltl <- c7ltr`
   - `c8ltl <- c8ltr`
   - `t1ltl <- t1ltr`
   - `t2ltl <- t2ltr`
   - `t3ltl <- t3ltr`
   - `t4ltl <- t4ltr`

3. **parallel PP-right rules**
   - `c7/8/t1/t2/t3/t4 ppr <- right LT`

This matches the actual follow-up observation pattern in Track 2 much better than a broad all-target anchor scheme.

## Why each weight was chosen

The weights in the shipped code are not arbitrary. They follow the train-data reliability of each target-source relationship.

### Very high weights (`0.60` to `0.70`)
Used where the source-target mapping is almost deterministic in the train labels and triggers a lot in test.

Examples from the bundle:
- `c2ltl <- c2ltr`, weight `0.70`
  - correlation: **1.000**
  - equality: **1.000**
  - anchor RMSE: **0.000**
  - test rows triggered: **205**

- `c2ppr <- c2ltr`, weight `0.65`
  - correlation: **0.744**
  - equality: **0.997**
  - anchor RMSE: **0.073**
  - test rows triggered: **206**

- `c3ltl <- c3ltr`, weight `0.70`
  - correlation: **0.815**
  - equality: **0.986**
  - anchor RMSE: **0.141**
  - test rows triggered: **151**

These are the kinds of rules that justify strong anchoring.

### Medium weights (`0.45` to `0.50`)
Used where the relationship is still useful, but no longer nearly deterministic.

Examples:
- `c7ltl <- c7ltr`, weight `0.45`
  - correlation: **0.890**
  - equality: **0.887**
  - anchor RMSE: **0.381**
  - test rows triggered: **80**

- `t3ltl <- t3ltr`, weight `0.50`
  - correlation: **0.901**
  - equality: **0.888**
  - anchor RMSE: **0.386**
  - test rows triggered: **57**

These are helpful but clearly noisier, so the anchor is blended rather than trusted outright.

### Lower weights (`0.20` to `0.28`)
Used for the PP-right rules in the noisier transition region.

Examples:
- `c7ppr <- c7ltr`, weight `0.20`
- `t4ppr <- t4ltr`, weight `0.20`

These target-source pairs still carry signal, but the train-set conditional anchor is much less clean, so only a light nudge is used.

## Why this translated from Track 1

This code was intentionally based on Track-1-style observations:
- upper cervical one-sided structure was very reliable,
- directly observed paired sources could be used as strong anchors,
- and soft blending was safer than hard replacement.

What translated well to Track 2 was specifically:
- **right-LT -> left-LT** in the upper block,
- **right-LT -> right-PP** in the upper block,
- and selected **left-LT -> left-PP** rules near the very top.

What did **not** get generalized into this file was a broad all-level anchor sweep. The bundle correctly keeps the rules hand-selected.

## What the script changed

Comparing the shipped hedge and anchor submissions:

- changed cells: **1775**
- affected rows: **252**
- mean abs change on changed cells: **0.030209**
- max abs change: **0.862269**

That is materially larger than the hedge step, which is expected:
- the hedge step is smoothing,
- this step is targeted correction.

## Important implementation note

The exact code in this bundle learns anchors using:

```python
mapping = ytr.groupby(source)[target].mean().to_dict()
```

where both `target` and `source` come from `labels_train_2.csv`.

That means the mapping is a **label-space conditional mean**, not a `features_train_2` missingness-conditioned mapping. This matches the actual shipped code and should be documented exactly that way in a reproduction package.

## How to run

```bash
python Method_1/code/run_t2_anchor_correction.py --submission Method_1/data/submissions/hedge_pairwise_shrink__t2_hedge_pairwise_shrink__20260315_213234__jobnojid__yp2096zc.csv --features-test data/features_test_2.csv --labels-train data/labels_train_2.csv --run-root runs
```

Or:

```bash
sbatch Method_1/slurm/t2_anchor_correction.sbatch
```

## Readme-level judgment

For the exact code bundled here, this stage is coherent and defensible. The main thing to document clearly is that the anchor maps are learned from **label-label conditional means**, because that is the exact implementation actually used.
