# ASIA 2026 Data Science Challenge — Track 2

## Longitudinal reconstruction of full ISNCSCI sensory examinations

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/manuscript-in%20preparation-6F42C1)](#publication-status)
[![Use](https://img.shields.io/badge/use-research%20only-B54708)](#clinical-use-and-limitations)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21744231.svg)](https://doi.org/10.5281/zenodo.21744231)

- **Challenge:** [ASIA Data Challenge](https://asia-spinalinjury.org/asia-data-challenge/)
- **Paper:** in preparation; public link and DOI forthcoming
- **Archived code release:** [Zenodo](https://doi.org/10.5281/zenodo.21744231)

This repository contains the SCAI Lab winning solution for **Track 2** of the
2026 American Spinal Injury Association (ASIA) Data Science Challenge. Track 2
asked teams to reconstruct omitted follow-up sensory findings from a complete
baseline International Standards for Neurological Classification of Spinal
Cord Injury (ISNCSCI) examination and an expedited follow-up examination.

## TL;DR

- **What we did:** Reconstructed the missing follow-up sensory grid using a
  target-wise TabPFN v2.5 classifier, while retaining all observed follow-up
  measurements unchanged.
- **Why:** Sensory grades are ordered clinical states, and the examination has
  meaningful longitudinal and anatomical structure that a general-purpose
  tabular model does not enforce by itself.
- **What worked:** Five-fold probability averaging provided the selected base
  model; small, auditable anatomical corrections then improved the final
  challenge submission.
- **Main result:** The selected pipeline achieved **0.38056 public RMSE** and
  **0.40706 private RMSE**, ranking first in Track 2. Lower RMSE is better.
- **Takeaway:** Cautiously injecting domain-specific structure may help
  contextualize broadly pretrained tabular models for clinical data, but this
  finding remains challenge-specific and requires prospective external
  validation.

## Method at a glance

```text
Complete baseline ISNCSCI
            +
Expedited follow-up ISNCSCI
            +
Clinical metadata
            │
            ▼
Target-wise leakage masking
            │
            ▼
Five-fold TabPFN classification
            │
            ▼
Probability averaging → expected sensory score
            │
            ├── observed follow-up value → copy unchanged
            │
            └── omitted value → conservative anatomical post-processing
                                  1. pairwise shrinkage
                                  2. anchor correction
                                  3. extended anchor correction
```

Sensory outcomes were represented as the ordered classes `0`, `1`, and `2`;
binary anal sensation was handled as a two-class target. For each sensory
target, the corresponding follow-up feature was masked before fitting to
prevent target leakage. Class probabilities were converted to a continuous
expected score for RMSE evaluation.

The selected implementation used shuffled row-wise five-fold cross-validation
with `random_state=42`. Each round used approximately 80% of development rows
for training and 20% for validation, with every row served once for validation, and
the five test-prediction sets were averaged.

## Results

### Final submissions

| Pipeline | Description | Public RMSE | Private RMSE |
|---|---|---:|---:|
| **Method 1 — selected** | Five-fold discrete TabPFN ensemble plus sequential anatomical post-processing | **0.38056** | **0.40706** |
| Method 2 | Five-seed probability ensemble without the Method 1 post-processing chain | 0.38514 | 0.41142 |

### Development-set model comparison

We reported the following performance across all sensory entries and on the
subset of cells that required reconstruction:

| Model formulation | Validation | All-entry RMSE | Reconstructed-cell RMSE |
|---|---:|---:|---:|
| Direct TabPFN regression | 3-fold | 0.393 ± 0.010 | 0.422 ± 0.010 |
| Single discrete TabPFN classifier | 3-fold | 0.377 ± 0.010 | 0.405 ± 0.011 |
| **Discrete TabPFN ensemble** | **5-fold** | **0.374 ± 0.015** | **0.402 ± 0.015** |

Values after `±` are the sample standard deviation of fold-specific RMSE. Post-processing was deterministic and evaluated using fixed public/private
leaderboard partitions.

## Included methods

### Method 1: selected Track 2 submission

[Method 1 details](method_1/README.md) ·
[pipeline entry point](method_1/scripts/run_pipeline.py)

Method 1 combines the five-fold discrete TabPFN base with three conservative
post-processing stages:

1. **Pairwise hedge shrink** softly reduces large disagreements between
   left/right or light-touch/pin-prick pairs when both values are missing.
2. **Anchor correction** moves a missing prediction toward a training-derived
   conditional mean associated with a related observed follow-up value.
3. **Extended anchor correction** applies a final, narrowly gated correction where
   a nearby observed value indicates that the current prediction may be too
   low.

All three stages operate only on unobserved target cells. Recorded follow-up
measurements are copied through and are never overwritten.

### Method 2: secondary submission

[Method 2 details](method_2/README.md) ·
[script](method_2/scripts/train_seed_ensemble.py)

Method 2 trains five full-development-set models with seeds
`11, 22, 33, 44, 55`, averages their class probabilities, and converts the
result to expected sensory scores. The Method 1 post-processing chain was not
applied to this submission.

## Repository structure

```text
asia-challenge-2026/
├── method_1/
│   ├── README.md
│   ├── scripts/                 # Selected five-fold model and post-processing
│   └── data/submissions/
│       ├── base_model_submission.csv
│       ├── pairwise_shrinkage_submission.csv
│       ├── anchor_corrected_submission.csv
│       └── final_submission.csv
├── method_2/
│   ├── README.md
│   ├── scripts/                 # Five-seed probability ensemble
│   └── data/submissions/
│       └── seed_ensemble_submission.csv
├── utils/                       # Data loading, metrics, and shared utilities
├── requirements.txt             # Recorded Python environment
└── README.md
```

The repository preserves the final submission CSVs. Run summaries, fold-level
metrics, out-of-fold predictions, and stage summaries are generated when the
pipelines are rerun under `runs/`; historical run directories are not included
in the current repository snapshot.

## Data access and expected files

The de-identified challenge data are not redistributed in this repository.
Access is governed by the challenge data-use conditions. Place the authorized
Track 2 files in `data/`:

```text
data/
├── features_train_2.csv
├── features_test_2.csv
├── labels_train_2.csv
├── labels_test_2_dummy.csv
├── metadata_train_2.csv
└── metadata_test_2.csv
```

The scripts merge features and metadata by `ID`.

## Environment

The recorded environment used:

- Python 3.11
- CUDA-enabled PyTorch
- `tabpfn==6.4.1` with pretrained **TabPFN v2.5 classifier weights**
- `numpy`, `pandas`, `scikit-learn`, and `huggingface_hub`
- NVIDIA GeForce RTX 4090 for the primary runs

The pinned package snapshot is in [`requirements.txt`](requirements.txt).
Install a PyTorch build compatible with the local CUDA driver, then install the
remaining recorded dependencies in an isolated environment.

This release specifically used **TabPFN v2.5**. The v2.5 weights are not redistributed in this repository. They are obtained separately from
[Prior Labs on Hugging Face](https://huggingface.co/Prior-Labs/tabpfn_2_5) and
are subject to their own gated, non-commercial terms. Access may require
approval and acceptance of those terms.

Approximate recorded runtimes were:

| Hardware | Method 1 | Method 2 |
|---|---:|---:|
| NVIDIA RTX 4090 | ~1 hour | ~3 hours |
| NVIDIA RTX 2080 | ~2 hours | ~6 hours |

## Running the code

Run commands from the repository root.

### Selected Method 1 pipeline

```bash
python method_1/scripts/run_pipeline.py \
  --data-root data \
  --run-root runs
```

The pipeline executes the five-fold base model and all three post-processing
stages. Its final submission is written to:

```text
runs/<pipeline_run_id>/predictions_test.csv
```

### Method 2

```bash
python method_2/scripts/train_seed_ensemble.py \
  --data-root data \
  --run-root runs \
  --do-cv 1 \
  --n-splits 5
```

When cross-validation is enabled, the run directory also contains
`cv_metrics.json`, `weighted_oof.json`, and
`oof_predictions_train.npz`.

## Clinical use and limitations

This code is for research and benchmarking. Reconstructed values are estimates,
not recorded neurological findings, and should not replace a clinically
indicated full ISNCSCI examination. The challenge used one clinical-trial
dataset and an evaluation-specific missingness pattern. Generalization across
centres, injury groups, examination time points, and routine-care missingness
has not yet been established.

Any downstream use should:

- clearly label imputed values and preserve their provenance
- retain observed clinical measurements unchanged
- assess clinically consequential errors, not only aggregate RMSE, and
- undergo external and prospective validation before clinical deployment.

## Publication status

An ASIA Data Science Challenge manuscript is in preparation for
**Topics in Spinal Cord Injury Rehabilitation**, with publication targeted for
the ASIA 2027 cycle. The manuscript has not yet been assigned a public DOI.
Frozen software releases are preserved on Zenodo; cite the version-specific
DOI associated with the exact release used in the manuscript.

## Citation

Citation metadata, including the author ORCID, are provided in
[`CITATION.cff`](CITATION.cff). For re-use and re-distribution, cite the
version-specific Zenodo release rather than the mutable repository branch.

```text
Purkayastha, Partha Sarathi. ASIA 2026 Track 2: Longitudinal ISNCSCI
Sensory Reconstruction. Zenodo, 2026.
ORCID: https://orcid.org/0009-0007-8879-7622
```


## Licensing

The repository code is released under the [MIT License](LICENSE):

```text
Copyright (c) 2026 Partha Sarathi Purkayastha
```

The license permits use, modification, and redistribution while requiring the
copyright and permission notice to be retained. The separately obtained
TabPFN v2.5 model weights remain subject to the Prior Labs terms linked above.

## Acknowledgments

We thank the American Spinal Injury Association, the ASIA Engineering and Data
Science Committee, the challenge organizers, the Sygen data contributors, and
the individuals whose de-identified examinations made this benchmark possible.
