Patch plan (Track 1 TabImpute) — tasks and subtasks

This is written so an agent can execute it.
Make sure that each and every step has a progress bar of sorts that can be live tracked. Use tqdm.

Task T1-TI-1 — Redefine the imputation unit: from “dataset split” to “context + query batch”

Where: your Track-1 runner script (e.g., run_track1_tabimpute.py or equivalent)

Subtask T1-TI-1.1 — Define canonical column sets

Implement a function that returns these lists in a deterministic order:

id_col = "ID"

target_cols: columns you must predict (exactly the columns in labels_train_1 excluding ID)

covariate_cols: all non-target usable inputs, e.g.

motor columns

time

metadata columns after encoding

any other non-target features you intentionally keep

Why: You must guarantee stable column ordering across:

building matrices

reconstructing predictions

writing Kaggle submission

Subtask T1-TI-1.2 — Build FULL context rows from training

Create context_df with shape (n_context, len(covariate_cols)+len(target_cols)):

covariates from merged train features+metadata

targets from labels_train_1[target_cols] (fully observed)

Rules:

Context targets must contain no NaNs.

Context should be sampled deterministically (seeded).

Context should be stratified by time if possible (see below).

Why: This gives TabImpute real target distributions and prevents “statistically dead” columns.

Subtask T1-TI-1.3 — Build query rows for test batch (partial targets)

For each test batch:

covariates from merged features_test_1 + metadata_test_1

targets taken from features_test_1[target_cols] (partial; NaNs where unobserved)

Important: Do NOT fill missing targets in query rows with mean/0 before TabImpute. Let TabImpute see NaNs.

Subtask T1-TI-1.4 — Single-call imputation per batch: table = [context ; query]

For each batch:

table = concat(context_df, query_df)

table_imputed = tabimpute.impute(table_numpy)

extract query_pred = table_imputed[context_rows:, target_block]

Subtask T1-TI-1.5 — Copy-through overwrite (mandatory)

After you get query_pred:

For each target cell that was observed in the query input (i.e., original features_test_1[target_col] not NaN), overwrite prediction with the observed value.

Why: It prevents the imputer from “changing what was measured,” and it reduces RMSE.

Subtask T1-TI-1.6 — Clip to valid ranges (mandatory)

Sensory targets: clip to [0, 2]

Binary targets (if any): clip to [0, 1]

Do not globally round.

Task T1-TI-2 — Make chunking safe and stable (4090-friendly)

Goal: avoid OOM without breaking the matrix completion regime.

Subtask T1-TI-2.1 — Use “fixed context, small query batch”

Set defaults:

n_context = 128 (start here)

batch_size = 4 (start here)

Make sure that each and every step has a progress bar of sorts that can be live tracked. Use tqdm.

If OOM:

reduce batch_size first

then reduce n_context

Subtask T1-TI-2.2 — Stratify the context by time

Instead of randomly sampling context rows:

sample ~equal counts per time value (1/4/8/16)

if some time buckets are small, fill remainder from others

Why: Track 1 distribution changes with time; stable context prevents drift.

Subtask T1-TI-2.3 — Keep the same context for every batch

Do not resample context per test batch.
Do not shuffle context per batch.
Only the query rows change.

Why: It stabilizes normalization and outputs.


Task T1-TI-4 — Add deterministic calibration (optional but often huge)

TabImpute outputs can have systematic bias (e.g., too close to 0 or too close to 2).

Subtask T1-TI-4.1 — Fit per-target linear correction on CV

For each target column j:

Fit y_true = a_j * y_pred + b_j on validation predictions (only imputed-only cells)

Apply correction to test predictions

Clip again

Subtask T1-TI-4.2 — If per-target is too heavy, do per-block calibration

Calibrate 5 blocks separately:

LT_left

LT_right

PP_left

PP_right

any binary column


Task T1-TI-5 — Harden output correctness (submission killers)

These are “small bugs that make you last.”

Subtask T1-TI-5.1 — Enforce exact submission schema

Always generate the submission dataframe by:

loading labels_test_1_dummy.csv

filling columns in its exact order

writing CSV with that column order

Subtask T1-TI-5.2 — Assert no NaNs in output

Before saving:

assert not df.isna().any().any()

Subtask T1-TI-5.3 — Assert value ranges

sensory columns within [-0.01, 2.01] after clipping

binary within [-0.01, 1.01]

4) Exhaustive “possible changes” list for Track 1 TabImpute (everything worth trying)
A) Table construction choices

Increase/decrease n_context

Context sampling:

uniform by time

biased toward rare injury severities (if you can proxy severity)

Covariate set:

motor-only + time + minimal metadata (often better than “everything”)

add missingness-summary features (counts of observed sensory values per region)

B) Inference batching knobs

batch_size (query rows per call)

keep context fixed across calls

keep dtype consistent (float32 recommended unless you know what you’re doing)

C) Preprocessing

Encode categorical metadata consistently (train-fit encoder, then transform test)

Normalize covariates (optional; do not normalize targets yourself—TabImpute handles internal scaling)

D) Postprocessing

clipping (mandatory)

“soft snapping” only when close to {0,1,2} (optional)

per-target/per-block calibration (often high ROI)

E) Evaluation alignment

score only imputed-only cells

use fold-level context+query, not “train alone / val alone”