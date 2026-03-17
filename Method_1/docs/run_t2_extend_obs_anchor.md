# run_t2_extend_obs_anchor.py

## Purpose

This script is the **third and final post-processing layer** in Method 1.

It does **not** introduce a broad new correction family.  
Instead, it makes a **small, upward-only refinement** on a narrow rule list when the previous prediction still looks too low relative to an observed source anchor.

This is the cleanup step at the end of Method 1.

## Why this exists

After the anchor-correction stage, some targets can still remain slightly conservative in rows where:

- the target is missing,
- a related source is observed,
- the source is at least moderate (`>= 1`),
- and the current prediction is below the learned anchor by a margin.

The final script only fixes that exact situation.

## Exact gating logic

A rule fires only if all of the following hold:

1. `target` is missing in `features_test_2.csv`
2. `source` is observed
3. `source >= 1`
4. `current_prediction < anchor - 0.02`

Then the script raises the target toward the anchor:

- full rule weight if `source >= 2`
- half weight if `source == 1`

This is an important design choice: the script does **not** pull values downward, and it trusts stronger observed sources more than ambiguous intermediate ones.

## Rule list and why it is narrow

The included rules are:

| target   | source   |   weight |   test_rows_triggered |   corr |   p_equal |   anchor_rmse |
|:---------|:---------|---------:|----------------------:|-------:|----------:|--------------:|
| c2ppl    | c2ltr    |     0.18 |                   202 |  0.848 |     0.997 |         0.046 |
| c3ppl    | c3ltr    |     0.14 |                   148 |  0.756 |     0.976 |         0.198 |
| c4ppl    | c4ltr    |     0.12 |                   119 |  0.68  |     0.952 |         0.275 |
| c5ltl    | c5ltr    |     0.1  |                   109 |  0.849 |     0.918 |         0.319 |
| c6ltl    | c6ltr    |     0.16 |                    83 |  0.88  |     0.897 |         0.366 |
| c6ppl    | c6ltr    |     0.08 |                    83 |  0.808 |     0.842 |         0.493 |
| c6ppr    | c6ltr    |     0.12 |                    80 |  0.85  |     0.868 |         0.438 |
| c7ppl    | c7ltr    |     0.1  |                    80 |  0.796 |     0.792 |         0.539 |
| c8ppl    | c8ltr    |     0.16 |                    71 |  0.853 |     0.821 |         0.478 |
| t1ppl    | t1ltr    |     0.16 |                    64 |  0.839 |     0.81  |         0.499 |
| t2ppl    | t2ltr    |     0.16 |                    62 |  0.839 |     0.809 |         0.501 |
| t3ppl    | t3ltr    |     0.08 |                    56 |  0.826 |     0.795 |         0.523 |

This rule list is concentrated in the upper/transition region and on `ppl` / related targets, because that is where the previous stages can still under-call relative to the observed paired source.

Examples of high-volume rules:
- `c2ppl <- c2ltr`, weight `0.18`, triggered on **202** rows
- `c3ppl <- c3ltr`, weight `0.14`, triggered on **148** rows
- `c4ppl <- c4ltr`, weight `0.12`, triggered on **119** rows

Examples of more selective cleanup rules:
- `c6ltl <- c6ltr`
- `c6ppr <- c6ltr`
- `t3ppl <- t3ltr`

## Why the weights are smaller here

This is the final stage, so the weights are intentionally modest:
- max weight in this script is only `0.18`,
- several rules are `0.08` to `0.12`,
- and there is an additional `MARGIN = 0.02` gate before any update happens.

That makes this step a *refinement*, not a re-anchoring of the whole submission.

## What changed in practice

Comparing the shipped anchor and final extend submissions:

- changed cells: **57**
- affected rows: **48**
- mean abs change on changed cells: **0.031674**
- max abs change: **0.194050**

This is exactly the intended signature of the script:
- tiny number of changes,
- small magnitudes,
- final polish only.

## Why this step is reasonable

This stage bakes in three conservative ideas:

1. **observed source must exist**
2. **source must already indicate at least some preserved function (`>=1`)**
3. **only raise, never force a downward correction**

That makes it much safer than a symmetric "re-anchoring" pass.

## Important implementation note

Unlike the previous anchor stage, this script builds the source map using:

```python
d = pd.DataFrame({"y": ytr[target], "x": Xtr[source]}).dropna()
maps[(target, source)] = d.groupby("x")["y"].mean().to_dict()
```

So this stage uses:
- `target` from `labels_train_2.csv`,
- `source` from `features_train_2.csv`.

That matches the exact shipped code and should be documented precisely.

## How to run

```bash
python Method_1/code/run_t2_extend_obs_anchor.py --submission Method_1/data/submissions/anchor_correction__t2_anchor_correction__20260315_222231__jobnojid__4chrmkaj.csv --features-test data/features_test_2.csv --features-train data/features_train_2.csv --labels-train data/labels_train_2.csv --run-root runs
```

Or:

```bash
sbatch Method_1/slurm/t2_extend_obs_anchor.sbatch
```

## Readme-level judgment

This is a tidy final-stage refinement. It is narrow, conservative, and consistent with the rest of Method 1. I do not see a logic red flag in the exact shipped implementation.
