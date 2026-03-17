# run_t2_hedge_pairwise_shrink.py

## Purpose

This script is the **first post-processing layer** in Method 1.

It starts from the base TabPFN bag5 submission and applies **very small, data-driven pairwise shrinkage** only when:

- both values in the pair are missing in the test follow-up features,
- and both targets have very high test missingness.

The goal is not to overwrite the model.  
The goal is to reduce obvious pairwise noise in exactly those regions where the model has the least direct follow-up evidence.

## Why this exists

Track 2 contains many dermatome-side/modality pairs with strong structural coupling.

From the train labels included in the bundle:

- mean left/right equality across sensory pairs: about **0.900**
- mean LT-vs-PP equality across within-side pairs: about **0.859**

And in test:
- many of these same targets are missing at rates above `0.85`,
- while a lower block is completely missing for all 252 rows.

So instead of hand-snapping left and right together, this script performs a **soft convex shrink toward the pair average**.

## The exact gating logic

A pair is modified only if:

1. its train-set pair agreement is high enough to yield a positive shrink weight,
2. each target in the pair has test missingness at least `0.85`,
3. and both cells are missing for that row.

That means the script never touches:
- observed follow-up cells,
- low-missingness pairs,
- or pairs whose train agreement is too weak.

## Weight formulas

### Left/right pairs
For `(left, right)`:
```python
w_lr = clip((p_equal - 0.85) / 0.15, 0.0, 1.0) * 0.25
```

### LT-PP pairs
For `(lt, pp)`:
```python
w_ltpp = clip((p_equal - 0.82) / 0.18, 0.0, 1.0) * 0.15
```

These formulas were chosen to make the hedge intentionally mild:
- LR can move more than LT/PP,
- neither family can dominate the base model,
- and mediocre pairs simply receive near-zero weight.

## Strongest applied pairs in this bundle

The highest-weight / highest-coverage applied pairs from the actual data are:

| pair          | type   |   weight |   p_equal |   trigger_rows |
|:--------------|:-------|---------:|----------:|---------------:|
| c2ppl|c2ppr   | LR     |    0.243 |     0.996 |            230 |
| s3ltl|s3ltr   | LR     |    0.155 |     0.943 |            252 |
| s2ltl|s2ltr   | LR     |    0.126 |     0.926 |            252 |
| l4ltl|l4ltr   | LR     |    0.125 |     0.925 |            252 |
| s1ltl|s1ltr   | LR     |    0.116 |     0.919 |            252 |
| l3ltl|l3ltr   | LR     |    0.114 |     0.918 |            252 |
| l5ltl|l5ltr   | LR     |    0.114 |     0.918 |            252 |
| l2ltl|l2ltr   | LR     |    0.101 |     0.911 |            250 |
| l1ltl|l1ltr   | LR     |    0.1   |     0.91  |            247 |
| t12ltl|t12ltr | LR     |    0.096 |     0.908 |            247 |
| s3ppl|s3ppr   | LR     |    0.091 |     0.904 |            252 |
| t10ltl|t10ltr | LR     |    0.089 |     0.903 |            241 |
| t9ltl|t9ltr   | LR     |    0.089 |     0.903 |            233 |
| t11ltl|t11ltr | LR     |    0.085 |     0.901 |            241 |
| t8ltl|t8ltr   | LR     |    0.076 |     0.896 |            227 |
| t12ppl|t12ppr | LR     |    0.058 |     0.885 |            247 |
| s2ppl|s2ppr   | LR     |    0.057 |     0.884 |            252 |
| t11ppl|t11ppr | LR     |    0.057 |     0.884 |            247 |
| c8ppl|c8ppr   | LR     |    0.057 |     0.884 |            238 |
| l1ppl|l1ppr   | LR     |    0.055 |     0.883 |            249 |

Interpretation:

- `c2ppl|c2ppr` gets the largest weight because train agreement is almost perfect and both sides are heavily missing in test.
- the fully missing lower block (`l3/l4/l5/s1/s2/s3`) also gets meaningful LR smoothing because both sides are always missing and left-right agreement is high.
- LT/PP shrinkage exists, but with smaller maximum weights.

## What the script changed in practice

Comparing the shipped base and hedge submissions:

- changed cells: **18744**
- affected rows: **252**
- mean abs change on changed cells: **0.002352**
- max abs change: **0.039429**

This is exactly the intended behavior:
- very broad coverage,
- but very small magnitude.

## Why this stage was chosen before anchors

This stage is low risk because it only assumes:
- sensory symmetry,
- within-level modality agreement,
- and that the base model's two paired predictions should not drift too far apart when the data offers no direct observation.

That makes it the right first hedge before stronger one-sided anchor rules are introduced.

## Safety behavior

After shrinkage, the script:

- clips predictions to valid ranges,
- copies through any observed test values,
- fills rare NaNs from the baseline / medians,
- writes a run summary with the exact applied weight maps.

## How to run

```bash
python Method_1/code/run_t2_hedge_pairwise_shrink.py --base-csv Method_1/data/submissions/t2_tabpfn_25_discrete_bag5__20260214_014329__job57139435__on5h0rmx.csv --data-root data --run-root runs
```

Or:

```bash
sbatch Method_1/slurm/t2_hedge_pairwise_shrink.sbatch
```

## Readme-level judgment

For the exact code in this bundle, this is a sensible conservative hedge and I do not see a major red flag in its logic. The main thing to remember is that it is designed to be **small**, not to generate the whole solution by itself.
