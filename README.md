# 1) Track 2 : Final methods, Environment, and Reproducibility notes


This README pack documents the **two Track 2 methods that were actually used as final competition submissions**.

| Rank within our Track 2 submissions | Submission file | Public RMSE |
|---|---|---:|
| 1 | `Discrete Bag 5 with Post-processing Submission (Method 1)` | **0.38056** |
| 2 | `Seed Bag 5 Submission (Method 2)` | **0.38514** |

This package includes:

- one overall README
- one method-level README for **Method 1: Discrete TabPFN bag5 with Post-processing**
- one method-level README for **Method 2: discrete TabPFN seedbag5 probability ensemble**

> Please note that due to time and submission constraints, we were not able to try out the Post processings on Method 2

## 2) Environment and compute assumptions

We used the following environment, libraries and tools. 

- Python 3.11-class runtime
- CUDA-enabled PyTorch (`torch`)
- `tabpfn` (We used Pretrained TabPFNv2.5 Classifier weights)
- `numpy`, `pandas`, `scikit-learn`
- `huggingface_hub`
- standard library modules used in the scripts
- enough GPU memory for TabPFN inference/training loops

> The TabPFN-2.5 pretrained weights used in this work are distributed via Hugging Face under a gated, non-commercial license and can be found [here](https://huggingface.co/Prior-Labs/tabpfn_2_5). Earlier TabPFN-2 weights are available under a more permissive license. 
>
> Please note that access to TabPFN-2.5 may require approval from Hugging Face and acceptance of its license terms prior to download.

For this project, we used:

- **GPU**: NVIDIA GeForce RTX 4090
- **Method 1 runtime budget**: about **1 hour**
- **Method 2 runtime budget**: about **3 hours**

> We also reproduced the same on NVIDIA GeForce RTX 2080 in **about 2 and 6 hours**, respectively.

An exact version of the packages used is described in`requirements.txt`, though it is not a compulsion.


## 3) How to run the pipelines

From the repository root `asia-challenge-2026/`.

The scripts expect this default hierarchy unless you pass explicit flags:

- `asia-challenge-2026/data/` for the Track 2 CSV inputs
- `asia-challenge-2026/files/` for baseline handoff CSVs and saved submissions
- `asia-challenge-2026/runs/` for run directories and summaries

### Method 1 main path
```bash
python Method_1/scripts/run_t2_method1_pipeline.py
```

### Method 1 manual chain
```bash
python Method_1/scripts/run_tabpfn_t2_discrete_bag5.py
python Method_1/scripts/run_t2_hedge_pairwise_shrink.py --base-csv <Output of previous run>
python Method_1/scripts/run_t2_anchor_correction.py --base-cv <Output of previous run>
python Method_1/scripts/run_t2_extend_obs_anchor.py --base-cv <Output of previous run>
```

### Method 2
```bash
python Method_2/scripts/run_tabpfn_t2_discrete_seedbag5_proba.py --do-cv 1 --n-splits 5
```

## 4) Where to look for what

- **You want the big picture** -> read this file, then `Method_1/README.md`, then `Method_2/README.md`.
- **You want the exact code** -> open the corresponding files under `Method_1/scripts/` or `Method_2/scripts/`.
- **You want the final CSVs** -> see `Method_1/data/submissions/` and `Method_2/data/submissions/`.
- **You want support utilities** -> see `utils/`.

## 5) Where results live

The default run output root is `asia-challenge-2026/runs/`.

Each run gets its own folder named with a generated run id, typically:

```text
<method>__<timestamp>__job<slurm_job_id_or_nojid>__<random_suffix>
```

Inside each run folder, the main result file is:

```text
predictions_test.csv
```

Some runs also write extra artifacts such as:

- `run_summary.json`
- `cv_metrics.json`
- `weighted_oof.json`
- `oof_predictions_train.npz`

For Method 1, the pipeline also uses stage folders like `00_discrete_bag/`, `01_pairwise_shrink/`, `02_anchor_correction/`, and `03_extend_obs_anchor/` under the pipeline run directory.

For Method 2, the final predictions are written directly to:

```text
asia-challenge-2026/runs/<run_id>/predictions_test.csv
```

If you need the exact final output path for any run, the corresponding `run_summary.json` records it.

## 6) Recommended reading order

1. This overview,
2. `Method_1/README.md`,
3. `Method_2/README.md`
