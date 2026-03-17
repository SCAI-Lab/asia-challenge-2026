# 1) README - Track 2 final methods, environment, compliance, and reproducibility notes


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

From the top-level `Track_2/` folder:

### Method 1
```bash
python Method_1/code/run_tabpfn_t2_discrete_bag5.py
python Method_1/code/run_t2_hedge_pairwise_shrink.py --base-csv <Output of previous run>
python Method_1/code/run_t2_anchor_correction.py --submission <Output of previous run>
python Method_1/code/run_t2_extend_obs_anchor.py --submission <Output of previous run>
```

### Method 2
```bash
python Method_2/code/run_tabpfn_t2_discrete_seedbag5_proba.py --do-cv 1 --n-splits 5
```

## Where to look for what

- **You want the big picture** -> read this file, then `Method_1/README.md`, then `Method_2/README.md`.
- **You want the exact code** -> open the corresponding files under `Method_1/code/` or `Method_2/code/`.
- **You want the final CSVs** -> see `Method_1/data/submissions/` and `Method_2/data/submissions/`.
- **You want support utilities** -> see `utils/`.

## 4) Recommended reading order

1. This overview,
2. `Method_1/README.md`,
3. `Method_2/README.md`



