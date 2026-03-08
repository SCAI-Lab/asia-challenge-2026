# ASIA 2026 (Track 1 + Track 2) - Slurm-first Starter (Baselines + TabPFN 2.5)

## At a glance (tasks, data, status)
- Track 1 task: predict target columns for the test set. Sensory targets are scored in [0,2] and anyana in [0,1]. If a target column is already observed in features, the code copies it through and only imputes missing values.
- Track 2 task: same target set, plus track2 features include baseline motor columns (w1_*) that are used by KNN baselines.
- Data layout (staged under `data/staged/`):
  ```
  data/raw_zips/Share_Track1.zip
  data/raw_zips/Share_Track2.zip

  data/staged/track1/
    features_train_1.csv
    labels_train_1.csv
    metadata_train_1.csv
    features_test_1.csv
    metadata_test_1.csv
    labels_test_1_dummy.csv

  data/staged/track2/
    features_train_2.csv
    labels_train_2.csv
    metadata_train_2.csv
    features_test_2.csv
    metadata_test_2.csv
    labels_test_2_dummy.csv
  ```
- Current repo status (this checkout): `runs/` already contains TabPFN outputs (t1 `tabpfn_25`, t1 `tabpfn_25_t1safe`, t1 `tabpfn_25_t1safe_discrete`, t2 `tabpfn_25`, t2 `tabpfn_25_discrete`). `runs_saits_pypots/` contains PyPOTS SAITS runs. `slurm_logs/` has previous job logs; `venv/` and cache dirs are present and can be regenerated.

## Dataset overview
### Track 1
| File | Split | Rows | Cols | What it contains |
|---|---|---|---|---|
| `features_train_1.csv` | Train | 1694 | 136 | Motor scores (20), partial sensory columns (0/1/2 with NaNs), time, vaccd, and other exam fields |
| `labels_train_1.csv` | Train | 1694 | 113 | Targets: full sensory vector (all sensory targets + anyana) + ID |
| `metadata_train_1.csv` | Train | 1694 | 14 | Demographics / clinical metadata (categorical + numeric) |
| `features_test_1.csv` | Test | 292 | 136 | Same schema as features_train_1.csv, with NaNs where sensory not observed |
| `metadata_test_1.csv` | Test | 292 | 14 | Same schema as metadata_train_1.csv |
| `labels_test_1_dummy.csv` | Test template | 292 | 113 | Submission template: ID + all target columns in correct order |

### Track 2
| File | Split | Rows | Cols | What it contains |
|---|---|---|---|---|
| `features_train_2.csv` | Train | 931 | 270 | Baseline full exam (w1_* block) + follow-up expedited exam block (partial sensory + motor + time) |
| `labels_train_2.csv` | Train | 931 | 113 | Targets: full follow-up sensory vector + ID |
| `metadata_train_2.csv` | Train | 931 | 14 | Demographics / clinical metadata |
| `features_test_2.csv` | Test | 252 | 270 | Same schema as train features; baseline full + follow-up partial |
| `metadata_test_2.csv` | Test | 252 | 14 | Same schema as train metadata |
| `labels_test_2_dummy.csv` | Test template | 252 | 113 | Submission template: ID + all target columns in correct order |

## Track 1 missingness plots (bar)
Train:
![Track 1 missingness (train)](runs/t1_missingness_report__20260306_164147__jobnojid__dzzo0rot/missingness_train_bar.png)

Test:
![Track 1 missingness (test)](runs/t1_missingness_report__20260306_164147__jobnojid__dzzo0rot/missingness_test_bar.png)

## Quick start (Slurm)
```bash
bash scripts/submit_all.sh
```
That submits:
- `slurm/00_setup_all.sbatch` (CPU) - setup venv, stage data, prefetch TabPFN regressor weights
- baselines for both tracks (CPU)
- `tabpfn_25` for both tracks (GPU)

If you want safe/discrete TabPFN, seedbagging, CatBoost, SAITS, or sweeps, use the specific sbatch files listed below.

## Core code map (repo structure + functions)
- `src/asia2026/run.py`: shared runner for baselines and `tabpfn_25` (CV, copy-through, clipping, artifacts)
- `src/asia2026/baselines.py`: `baseline_time_mean`, `baseline_strat_time_mean`, `baseline_knn15`
- `src/asia2026/tabpfn_model.py`: `tabpfn_25` regressor
- `src/asia2026/tabpfn_model_discrete.py`: classifier-as-regressor for sensory/anyana
- `src/asia2026/tabpfn_model_t1.py`: track1 safe TabPFN (reduced ensemble, PCA fallback)
- `src/asia2026/tabpfn_model_t1_discrete.py`: track1 safe discrete TabPFN
- `src/asia2026/data.py`: track loading, metadata merge, motor/meta column inference
- `src/asia2026/metrics.py`: R2/MAE/RMSE (all, sensory, and imputed-only)
- `src/asia2026/utils.py`: run IDs, config, JSON helpers

## Methods (both tracks)
Baselines (via `python -m asia2026.run --method ...`):
- `baseline_time_mean`: copy-through observed targets else mean(label | time).
- `baseline_strat_time_mean`: mean(label | time + metadata strata) with backoff to time-only if a group is small.
- `baseline_knn15`: fallback to time-mean, then replace the 15 always-missing targets via KNN on motor + metadata (k=25, distance weighting).

TabPFN methods:
- `tabpfn_25` (Track 1 and Track 2): per-target TabPFNRegressor with copy-through.
- `tabpfn_25_t1safe` (Track 1): safer TabPFN with single-threaded preprocessing and PCA fallback.
- `tabpfn_25_t1safe_discrete` (Track 1): safe + classifier-as-regressor for sensory/anyana.
- `tabpfn_25_discrete` (Track 2): classifier-as-regressor for sensory/anyana.

Additional approaches implemented in scripts:
- TabPFN seedbagging/overlays for track 1 and track 2 (discrete, seedbag5, overlay OOF).
- CatBoost multi-target blocks + residual variants (Track 1).
- SAITS (custom + PyPOTS) for Track 1.
- TabImpute / TabImpute+ (Track 1 + Track 2).
- Blending TabPFN + CatBoost predictions.

## Safe fixes (TabPFN)
- Enforces single-threaded preprocessing (TABPFN_NUM_WORKERS and OMP/MKL/OPENBLAS/NUMEXPR set to 1).
- Uses a reduced ensemble (n_ensemble_configurations=1, n_estimators=1, n_jobs=1 when supported).
- Falls back to PCA if preprocessing fails (ARPACK issues) in the track1 safe variants.

## Sweeps
- KNN sweep: `scripts/tune_knn_sweep.py` (slurm/30, slurm/31) explores k in `--k-list` and `--weights`.
- Stratified mean sweep: `scripts/tune_strat_sweep.py` (slurm/32, slurm/33) tries time + sex + age_bin + bmi_bin strata with varying `min_group` thresholds (defaults: track1 50/100, track2 30/60).

## Environments (venvs)
- `venv`
- `venv_saits_pypots`
- `venv_saits_virt`
- `venv_tabimpute`
- `venv_tabimpute_plus`

## Scripts by approach
### Setup and environment
- `scripts/00_env.sh`
- `scripts/10_setup_project.sh`
- `scripts/11_create_venv.sh`
- `scripts/18_install_catboost.sh`
- `scripts/19_setup_saits_env.sh`
- `scripts/20_setup_catboost_residual_env.sh`
- `scripts/99_fix_and_verify_envs.sh`
- `scripts/install_tabpfn_extensions.sh`
- `scripts_saits_pypots/00_env_saits_pypots.sh`
- `scripts_saits_pypots/00_setup_saits_pypots_env.sh`

### Data staging and unpack
- `scripts/12_stage_data.sh`
- `scripts/unpack_data.py`

### Weight prefetch / model assets
- `scripts/13_prefetch_tabpfn.sh`
- `scripts/14_prefetch_tabpfn_classifier.sh`
- `scripts/17_prefetch_tabpfn_v2_regressor.sh`
- `scripts/15_prefetch_tabimpute.sh`
- `scripts/16_prefetch_tabimpute_plus.sh`

### TabPFN (track 1, safe/discrete/seedbag/overlay)
- `scripts/run_tabpfn_t1_safe.py`
- `scripts/run_tabpfn_t1_safe_discrete.py`
- `scripts/run_tabpfn_t1safe_bag5.py`
- `scripts/run_tabpfn_t1safe_discrete_bag5.py`
- `scripts/run_tabpfn_t1_safe_discrete_oof.py`
- `scripts/run_tabpfn_t1_safe_discrete_overlay.py`
- `scripts/run_tabpfn_t1_safe_discrete_overlay_oof.py`
- `scripts/run_t1_discrete_seedbag5_proba.py`
- `scripts/run_t1_discrete_seedbag5_proba_isncsci.py`
- `scripts/run_t1_discrete_seedbag_proba_custom.py`
- `scripts/run_t1_autotabpfn_overlay_T24.py`

### TabPFN (track 2 discrete/seedbag)
- `scripts/run_tabpfn_t2_discrete.py`
- `scripts/run_tabpfn_t2_discrete_bag5.py`
- `scripts/run_tabpfn_t2_discrete_seedbag5_proba.py`

### CatBoost blocks
- `scripts/run_t1_catboost_blocks.py`

### CatBoost residual blocks
- `scripts/train_t1_catboost_residual_blocks.py`
- `scripts/train_t1_catboost_residual_blocks_lvl1.py`

### Blending / ensembling
- `scripts/blend_t1_predictions.py`
- `scripts/predict_t1_tabpfn_plus_catboost.py`
- `scripts/predict_t1_tabpfn_plus_catboost_lvl1.py`

### SAITS (custom)
- `scripts/run_t1_saits.py`
- `scripts/debug_saits_coverage_t1.py`
- `scripts/debug_saits_mapping_t1.py`
- `scripts/debug_saits_roundtrip_t1.py`

### SAITS (PyPOTS)
- `scripts_saits_pypots/run_t1_saits_pypots.py`
- `scripts_saits_pypots/run_t1_saits_pypots_cv.py`

### TabImpute
- `scripts/run_tabimpute_t1.py`
- `scripts/run_tabimpute_t2.py`

### Postprocessing
- `scripts/postprocess_discrete_greedy.py`
- `scripts/postprocess_discrete_prior_time_soft.py`

### Sweeps
- `scripts/tune_knn_sweep.py`
- `scripts/tune_strat_sweep.py`

### Reporting and metrics
- `scripts/report_pred_metrics_generic.py`
- `scripts/report_pred_metrics_t1.py`
- `scripts/report_run_metrics.py`
- `scripts/report_t1_missingness.py`
- `scripts/compare_t1_discrete_overlay_metrics.py`
- `scripts/compare_t1_discrete_overlay_oof_metrics.py`

### Orchestration
- `scripts/submit_all.sh`

### Root helper scripts
- `bash.sh`

## Slurm batch files by approach
### Setup and data
- `slurm/00_setup_all.sbatch`

### Baselines
- `slurm/10_track1_baselines.sbatch`
- `slurm/20_track2_baselines.sbatch`

### TabPFN base
- `slurm/11_track1_tabpfn.sbatch`
- `slurm/21_track2_tabpfn.sbatch`

### TabPFN safe/discrete
- `slurm/12_track1_tabpfn_safe.sbatch`
- `slurm/13_track1_tabpfn_safe_discrete.sbatch`
- `slurm/22_track2_tabpfn_discrete.sbatch`

### TabPFN bagging/seedbag/overlay/OOF
- `slurm/14_track1_tabpfn_safe_bag5.sbatch`
- `slurm/15_track1_tabpfn_safe_discrete_bag5.sbatch`
- `slurm/16_track1_tabpfn_safe_discrete_seedbag5_proba.sbatch`
- `slurm/16b_track1_tabpfn_seedbag5_proba_isncsci.sbatch`
- `slurm/16b_track1_tabpfn_seedbag5_proba_isncsci_smoke.sbatch`
- `slurm/16c_track1_tabpfn_seedbag_proba_custom.sbatch`
- `slurm/16d_t1_autotabpfn_overlay_T24.sbatch`
- `slurm/18_track1_tabpfn_safe_discrete_overlay.sbatch`
- `slurm/24_track1_tabpfn_safe_discrete_oof.sbatch`
- `slurm/25_track1_tabpfn_safe_discrete_overlay_oof.sbatch`
- `slurm/23_track2_tabpfn_discrete_bag5.sbatch`
- `slurm/23b_track2_tabpfn_discrete_seedbag5_proba.sbatch`
- `slurm/23b_track2_tabpfn_discrete_seedbag5_proba_smoke.sbatch`
- `slurm/28_t1_autotabpfn_overlay_T24.sbatch`

### CatBoost blocks
- `slurm/17_t1_catboost_blocks.sbatch`
- `slurm/17_t1_catboost_blocks_cpu.sbatch`
- `slurm/tmp_t1_catboost_blocks_smoke.sbatch`

### CatBoost residual blocks
- `slurm/24_t1_catboost_residual_smoke.sbatch`
- `slurm/25_t1_catboost_residual_full.sbatch`
- `slurm/26_t1_catboost_residual_smoke_lvl1.sbatch`
- `slurm/27_t1_catboost_residual_full_lvl1.sbatch`

### Blending
- `slurm/18_t1_blend_predictions.sbatch`

### SAITS (custom/PyPOTS)
- `slurm/19_t1_saits.sbatch`
- `slurm/19_t1_saits_pypots_smoke.sbatch`
- `slurm/20_t1_saits_pypots_full.sbatch`
- `slurm/21_t1_saits_pypots_cv_smoke.sbatch`
- `slurm/22_t1_saits_pypots_cv_full.sbatch`

### Sweeps
- `slurm/30_t1_knn_sweep.sbatch`
- `slurm/31_t2_knn_sweep.sbatch`
- `slurm/32_t1_strat_sweep.sbatch`
- `slurm/33_t2_strat_sweep.sbatch`

### TabImpute
- `slurm/40_t1_tabimpute.sbatch`
- `slurm/41_t1_tabimpute_plus.sbatch`
- `slurm/42_t2_tabimpute.sbatch`
- `slurm/43_t2_tabimpute_plus.sbatch`

## Outputs and run IDs
- Run ID format: `t<track>_<method>__YYYYMMDD_HHMMSS__job<SLURM_JOB_ID>__<rand8>`
- Each run directory under `runs/` or `runs_saits_pypots/` contains:
  - `config.json`
  - `predictions_test.csv`
  - `run_summary.json`
  - `cv_metrics.json` (only when `--do-cv 1`)
- Slurm logs are written to `slurm_logs/` under the submit directory.

## Manual CLI examples
```bash
python -m asia2026.run \
  --track 1 \
  --method baseline_knn15 \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --do-cv 1 \
  --n-splits 5
```

```bash
python scripts/run_tabpfn_t2_discrete.py \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --do-cv 1 \
  --n-splits 3
```

## Paths and environment
- `ASIA2026_ROOT` defaults to the repo root. Override it if you want everything to live on scratch.
- Set `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`) before running prefetch scripts or TabPFN jobs that need weights.
- `HF_HUB_OFFLINE=1` by default for compute jobs; prefetch scripts set it to 0.

## Job summary (last 500 successful runs with CV metrics)
- Successful = run directory has `run_summary.json` in `runs/` or `runs_saits_pypots/`.
- Only rows with `cv_metrics.json` and the required RMSE/R2 metrics are included (no NA rows).
- wRMSE uses sensory RMSE as the weighted proxy (all + imputed-only).
- Sorted by most recent timestamp (run summary `finished_utc`, then run ID timestamp, then mtime).

### Track 1
| Job ID | Task | Config | RMSE all | RMSE imputed-only | wRMSE all | wRMSE imputed-only | R2 all | R2 imputed-only |
|---|---|---|---|---|---|---|---|---|
| `58155071` | Top-K overlay (K=15, LB 0.41479) | `runs/t1_tabpfn_t1safe_discrete_overlay_oof__20260222_211655__job58155071__go6jgpqf/config.json` | 0.508720 | 0.546508 | 0.510936 | 0.547553 | 0.685813 | 0.602527 |
| `58152606` | Top-K overlay (K=30, LB 0.41494) | `runs/t1_tabpfn_t1safe_discrete_overlay_oof__20260222_200243__job58152606__xmlagr4h/config.json` | 0.512232 | 0.550281 | 0.514460 | 0.551329 | 0.681534 | 0.597115 |
| `58152512` | tabpfn_25_t1safe_discrete_oof (track 1) | `runs/t1_tabpfn_25_t1safe_discrete_oof__20260222_195410__job58152512__wxrdebfu/config.json` | 0.498184 | 0.535189 | 0.500356 | 0.536214 | 0.698764 | 0.618912 |
| `58147968` | tabpfn_25_t1safe_discrete (track 1) | `runs/t1_tabpfn_25_t1safe_discrete__20260222_174033__job58147968__89r81hqg/config.json` | 0.498184 | 0.535189 | 0.500356 | 0.536214 | 0.698764 | 0.618912 |
| `57223107` | tabpfn_25_t1safe_discrete_bag5 (track 1) | `runs/t1_tabpfn_25_t1safe_discrete_bag5__20260215_094024__job57223107__ptj7733i/config.json` | 0.498184 | 0.535189 | 0.500356 | 0.536214 | 0.698764 | 0.618912 |
| `57223106` | tabpfn_25_t1safe_bag5 (track 1) | `runs/t1_tabpfn_25_t1safe_bag5__20260215_094016__job57223106__3xbhmlf7/config.json` | 0.452248 | 0.485841 | 0.454232 | 0.486785 | 0.751755 | 0.685949 |
| `57139434` | tabpfn_25_t1safe_discrete_bag5 (track 1) | `runs/t1_tabpfn_25_t1safe_discrete_bag5__20260214_012837__job57139434__qw4i8vad/config.json` | 0.497244 | 0.534180 | 0.499410 | 0.535201 | 0.699900 | 0.620348 |
| `57139176` | tabpfn_25_t1safe_bag5 (track 1) | `runs/t1_tabpfn_25_t1safe_bag5__20260214_010603__job57139176__thotnlfv/config.json` | 0.452248 | 0.485841 | 0.454232 | 0.486785 | 0.751755 | 0.685949 |
| `57017218` | tabpfn_25_t1safe_discrete (track 1) | `runs/t1_tabpfn_25_t1safe_discrete__20260212_211649__job57017218__44tae8c8/config.json` | 0.497244 | 0.534180 | 0.499410 | 0.535201 | 0.699900 | 0.620348 |
| `56991810` | tabpfn_25_t1safe (track 1) | `runs/t1_tabpfn_25_t1safe__20260212_165824__job56991810__gyjcrfx2/config.json` | 0.496018 | 0.532862 | 0.497524 | 0.533179 | 0.701378 | 0.622218 |

### Track 1 CatBoost + lambda sweeps
Rows are from `t1_tabpfn_plus_catboost*` run summaries joined with CatBoost residual CV metrics when available.

| Job ID | CSV | Alphas | RMSE all | RMSE imputed-only | wRMSE all | wRMSE imputed-only | R2 all | R2 imputed-only |
|---|---|---|---|---|---|---|---|---|
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda-0.10.csv` | -0.10 | NA | 0.534111 | 0.510041 | 0.533705 | NA | 0.620446 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda-0.20.csv` | -0.20 | NA | 0.536545 | 0.512604 | 0.536157 | NA | 0.616978 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda-0.30.csv` | -0.30 | NA | 0.539119 | 0.515314 | 0.538750 | NA | 0.613294 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.00.csv` | 0.00 | NA | 0.531860 | 0.507669 | 0.531438 | NA | 0.623639 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.03.csv` | 0.03 | NA | 0.531336 | 0.507118 | 0.530910 | NA | 0.624379 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.05.csv` | 0.05 | NA | 0.530829 | 0.506583 | 0.530400 | NA | 0.625095 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.07.csv` | 0.07 | NA | 0.530339 | 0.506067 | 0.529906 | NA | 0.625787 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.12.csv` | 0.12 | NA | 0.529410 | 0.505087 | 0.528970 | NA | 0.627098 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.15.csv` | 0.15 | NA | 0.528970 | 0.504624 | 0.528527 | NA | 0.627717 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.17.csv` | 0.17 | NA | 0.528548 | 0.504178 | 0.528101 | NA | 0.628311 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.40.csv` | 0.40 | NA | 0.525510 | 0.500974 | 0.525041 | NA | 0.632571 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.50.csv` | 0.50 | NA | 0.524606 | 0.500020 | 0.524130 | NA | 0.633835 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.60.csv` | 0.60 | NA | 0.523975 | 0.499355 | 0.523495 | NA | 0.634714 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.70.csv` | 0.70 | NA | 0.523617 | 0.498976 | 0.523135 | NA | 0.635214 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.75.csv` | 0.75 | NA | 0.523538 | 0.498893 | 0.523055 | NA | 0.635324 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.80.csv` | 0.80 | NA | 0.523524 | 0.498878 | 0.523041 | NA | 0.635343 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.85.csv` | 0.85 | NA | 0.523573 | 0.498930 | 0.523090 | NA | 0.635275 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.90.csv` | 0.90 | NA | 0.523683 | 0.499046 | 0.523201 | NA | 0.635121 |
| `59367879` | `runs/t1_tabpfn_plus_catboost__20260306_203831__job59367879__uvs6ejpr/predictions_test_lambda0.95.csv` | 0.95 | NA | 0.523852 | 0.499224 | 0.523371 | NA | 0.634887 |
| `58178475` | `runs/t1_tabpfn_plus_catboost__20260223_095812__job58178475__7mgv0ck4/predictions_test_lambda0.40.csv` | 0.40 | NA | NA | NA | NA | NA | NA |
| `58178475` | `runs/t1_tabpfn_plus_catboost__20260223_095812__job58178475__7mgv0ck4/predictions_test_lambda0.50.csv` | 0.50 | NA | NA | NA | NA | NA | NA |
| `58156982` | `runs/t1_tabpfn_plus_catboost__20260223_004340__job58156982__td6tfwz7/predictions_test_lambda0.10.csv` | 0.10 | NA | 0.523530 | NA | NA | NA | 0.635335 |
| `58156982` | `runs/t1_tabpfn_plus_catboost__20260223_004340__job58156982__td6tfwz7/predictions_test_lambda0.20.csv` | 0.20 | NA | 0.516022 | NA | NA | NA | 0.645719 |
| `58156982` | `runs/t1_tabpfn_plus_catboost__20260223_004340__job58156982__td6tfwz7/predictions_test_lambda0.30.csv` | 0.30 | NA | 0.509364 | NA | NA | NA | 0.654802 |
| `58155590` | `runs/t1_tabpfn_plus_catboost__20260222_215100__job58155590__3xe06dyl/predictions_test_lambda0.10.csv` | 0.10 | NA | NA | NA | NA | NA | NA |
| `58155590` | `runs/t1_tabpfn_plus_catboost__20260222_215100__job58155590__3xe06dyl/predictions_test_lambda0.20.csv` | 0.20 | NA | NA | NA | NA | NA | NA |
| `58155590` | `runs/t1_tabpfn_plus_catboost__20260222_215100__job58155590__3xe06dyl/predictions_test_lambda0.30.csv` | 0.30 | NA | NA | NA | NA | NA | NA |
| `58156958` | `runs/t1_tabpfn_plus_catboost__20260222_222934__job58156958__3fa8tmib/predictions_test_lambda0.10.csv` | 0.10 | NA | 0.550054 | NA | NA | NA | 0.610168 |
| `58156958` | `runs/t1_tabpfn_plus_catboost__20260222_222934__job58156958__3fa8tmib/predictions_test_lambda0.20.csv` | 0.20 | NA | 0.550034 | NA | NA | NA | 0.610197 |
| `58156958` | `runs/t1_tabpfn_plus_catboost__20260222_222934__job58156958__3fa8tmib/predictions_test_lambda0.30.csv` | 0.30 | NA | 0.550198 | NA | NA | NA | 0.609965 |
| `58211415` | `runs/t1_tabpfn_plus_catboost__20260223_123946__job58211415__3m88vmio/predictions_test_lambda0.00.csv` | 0.00 | NA | 0.531860 | NA | NA | NA | 0.623639 |
| `58211415` | `runs/t1_tabpfn_plus_catboost__20260223_123946__job58211415__3m88vmio/predictions_test_lambda0.03.csv` | 0.03 | NA | 0.529704 | NA | NA | NA | 0.626684 |
| `58211415` | `runs/t1_tabpfn_plus_catboost__20260223_123946__job58211415__3m88vmio/predictions_test_lambda0.05.csv` | 0.05 | NA | 0.527597 | NA | NA | NA | 0.629646 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda-0.10.csv` | -0.10 | NA | 0.533291 | 0.509177 | 0.532878 | NA | 0.621610 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda-0.20.csv` | -0.20 | NA | 0.534847 | 0.510816 | 0.534444 | NA | 0.619399 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda-0.30.csv` | -0.30 | NA | 0.536500 | 0.512557 | 0.536109 | NA | 0.617042 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.00.csv` | 0.00 | NA | 0.531860 | 0.507669 | 0.531438 | NA | 0.623639 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.03.csv` | 0.03 | NA | 0.531527 | 0.507319 | 0.531103 | NA | 0.624109 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.05.csv` | 0.05 | NA | 0.531206 | 0.506980 | 0.530780 | NA | 0.624563 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.07.csv` | 0.07 | NA | 0.530895 | 0.506653 | 0.530467 | NA | 0.625002 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.12.csv` | 0.12 | NA | 0.530306 | 0.506032 | 0.529874 | NA | 0.625834 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.15.csv` | 0.15 | NA | 0.530028 | 0.505739 | 0.529594 | NA | 0.626226 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.17.csv` | 0.17 | NA | 0.529760 | 0.505457 | 0.529324 | NA | 0.626604 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.40.csv` | 0.40 | NA | 0.527841 | 0.503432 | 0.527392 | NA | 0.629305 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.50.csv` | 0.50 | NA | 0.527269 | 0.502830 | 0.526817 | NA | 0.630107 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.60.csv` | 0.60 | NA | 0.526871 | 0.502410 | 0.526416 | NA | 0.630665 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.70.csv` | 0.70 | NA | 0.526645 | 0.502171 | 0.526188 | NA | 0.630983 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.75.csv` | 0.75 | NA | 0.526593 | 0.502117 | 0.526136 | NA | 0.631055 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.80.csv` | 0.80 | NA | 0.526583 | 0.502106 | 0.526126 | NA | 0.631069 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.85.csv` | 0.85 | NA | 0.526611 | 0.502136 | 0.526154 | NA | 0.631029 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.90.csv` | 0.90 | NA | 0.526678 | 0.502207 | 0.526222 | NA | 0.630936 |
| `59368035` | `runs/t1_tabpfn_plus_catboost__20260306_202102__job59368035__gh6azxhl/predictions_test_lambda0.95.csv` | 0.95 | NA | 0.526780 | 0.502314 | 0.526324 | NA | 0.630792 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda-0.10.csv` | -0.10 | NA | 0.534486 | 0.510432 | 0.534082 | NA | 0.619912 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda-0.20.csv` | -0.20 | NA | 0.537358 | 0.513450 | 0.536973 | NA | 0.615817 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda-0.30.csv` | -0.30 | NA | 0.540431 | 0.516678 | 0.540066 | NA | 0.611411 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.00.csv` | 0.00 | NA | 0.531860 | 0.507669 | 0.531438 | NA | 0.623639 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.03.csv` | 0.03 | NA | 0.531250 | 0.507028 | 0.530824 | NA | 0.624501 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.05.csv` | 0.05 | NA | 0.530660 | 0.506407 | 0.530230 | NA | 0.625334 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.07.csv` | 0.07 | NA | 0.530090 | 0.505807 | 0.529656 | NA | 0.626139 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.12.csv` | 0.12 | NA | 0.529008 | 0.504668 | 0.528567 | NA | 0.627663 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.15.csv` | 0.15 | NA | 0.528497 | 0.504130 | 0.528052 | NA | 0.628383 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.17.csv` | 0.17 | NA | 0.528005 | 0.503613 | 0.527557 | NA | 0.629074 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.40.csv` | 0.40 | NA | 0.524482 | 0.499902 | 0.524009 | NA | 0.634008 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.50.csv` | 0.50 | NA | 0.523437 | 0.498801 | 0.522958 | NA | 0.635464 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.60.csv` | 0.60 | NA | 0.522716 | 0.498040 | 0.522231 | NA | 0.636469 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.70.csv` | 0.70 | NA | 0.522313 | 0.497616 | 0.521826 | NA | 0.637028 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.75.csv` | 0.75 | NA | 0.522230 | 0.497528 | 0.521743 | NA | 0.637143 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.80.csv` | 0.80 | NA | 0.522224 | 0.497521 | 0.521736 | NA | 0.637153 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.85.csv` | 0.85 | NA | 0.522292 | 0.497592 | 0.521804 | NA | 0.637058 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.90.csv` | 0.90 | NA | 0.522432 | 0.497740 | 0.521946 | NA | 0.636863 |
| `59368082` | `runs/t1_tabpfn_plus_catboost__20260306_204751__job59368082__4fxaj58i/predictions_test_lambda0.95.csv` | 0.95 | NA | 0.522642 | 0.497962 | 0.522158 | NA | 0.636570 |
| `59367803` | `runs/t1_tabpfn_plus_catboost__20260306_201751__job59367803__e4vheyal/predictions_test_lambda0.60.csv` | 0.60 | NA | 0.526695 | 0.502225 | 0.526235 | NA | 0.630912 |
| `59367803` | `runs/t1_tabpfn_plus_catboost__20260306_201751__job59367803__e4vheyal/predictions_test_lambda0.70.csv` | 0.70 | NA | 0.526484 | 0.502002 | 0.526022 | NA | 0.631208 |
| `59367803` | `runs/t1_tabpfn_plus_catboost__20260306_201751__job59367803__e4vheyal/predictions_test_lambda0.75.csv` | 0.75 | NA | 0.526446 | 0.501961 | 0.525984 | NA | 0.631261 |
| `59367803` | `runs/t1_tabpfn_plus_catboost__20260306_201751__job59367803__e4vheyal/predictions_test_lambda0.80.csv` | 0.80 | NA | 0.526452 | 0.501968 | 0.525990 | NA | 0.631252 |
| `59367803` | `runs/t1_tabpfn_plus_catboost__20260306_201751__job59367803__e4vheyal/predictions_test_lambda0.85.csv` | 0.85 | NA | 0.526502 | 0.502021 | 0.526040 | NA | 0.631183 |
| `59367803` | `runs/t1_tabpfn_plus_catboost__20260306_201751__job59367803__e4vheyal/predictions_test_lambda0.90.csv` | 0.90 | NA | 0.526595 | 0.502119 | 0.526134 | NA | 0.631053 |
| `59367803` | `runs/t1_tabpfn_plus_catboost__20260306_201751__job59367803__e4vheyal/predictions_test_lambda0.95.csv` | 0.95 | NA | 0.526729 | 0.502260 | 0.526269 | NA | 0.630865 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda-0.10.csv` | -0.10 | NA | 0.534431 | 0.510375 | 0.534027 | NA | 0.619991 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda-0.20.csv` | -0.20 | NA | 0.537234 | 0.513323 | 0.536850 | NA | 0.615994 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda-0.30.csv` | -0.30 | NA | 0.540220 | 0.516462 | 0.539858 | NA | 0.611713 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.00.csv` | 0.00 | NA | 0.531860 | 0.507669 | 0.531438 | NA | 0.623639 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.03.csv` | 0.03 | NA | 0.531262 | 0.507041 | 0.530836 | NA | 0.624483 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.05.csv` | 0.05 | NA | 0.530684 | 0.506432 | 0.530253 | NA | 0.625301 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.07.csv` | 0.07 | NA | 0.530125 | 0.505843 | 0.529690 | NA | 0.626090 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.12.csv` | 0.12 | NA | 0.529063 | 0.504725 | 0.528621 | NA | 0.627586 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.15.csv` | 0.15 | NA | 0.528561 | 0.504196 | 0.528115 | NA | 0.628292 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.17.csv` | 0.17 | NA | 0.528078 | 0.503688 | 0.527629 | NA | 0.628971 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.40.csv` | 0.40 | NA | 0.524601 | 0.500024 | 0.524127 | NA | 0.633841 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.50.csv` | 0.50 | NA | 0.523561 | 0.498927 | 0.523079 | NA | 0.635292 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.60.csv` | 0.60 | NA | 0.522833 | 0.498160 | 0.522346 | NA | 0.636305 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.70.csv` | 0.70 | NA | 0.522417 | 0.497722 | 0.521927 | NA | 0.636883 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.75.csv` | 0.75 | NA | 0.522323 | 0.497622 | 0.521832 | NA | 0.637015 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.80.csv` | 0.80 | NA | 0.522300 | 0.497598 | 0.521809 | NA | 0.637046 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.85.csv` | 0.85 | NA | 0.522349 | 0.497650 | 0.521858 | NA | 0.636979 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.90.csv` | 0.90 | NA | 0.522469 | 0.497776 | 0.521979 | NA | 0.636812 |
| `59367942` | `runs/t1_tabpfn_plus_catboost__20260306_204332__job59367942__gr13uaz3/predictions_test_lambda0.95.csv` | 0.95 | NA | 0.522656 | 0.497974 | 0.522168 | NA | 0.636551 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda-0.10.csv` | -0.10 | NA | 0.533361 | NA | NA | NA | 0.621510 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda-0.20.csv` | -0.20 | NA | 0.534984 | NA | NA | NA | 0.619203 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda-0.30.csv` | -0.30 | NA | 0.536703 | NA | NA | NA | 0.616753 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.00.csv` | 0.00 | NA | 0.531860 | NA | NA | NA | 0.623639 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.03.csv` | 0.03 | NA | 0.531511 | NA | NA | NA | 0.624131 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.05.csv` | 0.05 | NA | 0.531174 | NA | NA | NA | 0.624608 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.07.csv` | 0.07 | NA | 0.530849 | NA | NA | NA | 0.625067 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.12.csv` | 0.12 | NA | 0.530233 | NA | NA | NA | 0.625938 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.15.csv` | 0.15 | NA | 0.529942 | NA | NA | NA | 0.626348 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.17.csv` | 0.17 | NA | 0.529662 | NA | NA | NA | 0.626742 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.40.csv` | 0.40 | NA | 0.527671 | NA | NA | NA | 0.629544 |
| `59350171` | `runs/t1_tabpfn_plus_catboost__20260306_180125__job59350171__b0xodch1/predictions_test_lambda0.50.csv` | 0.50 | NA | 0.527089 | NA | NA | NA | 0.630360 |
| `58211689` | `runs/t1_tabpfn_plus_catboost__20260223_124249__job58211689__sxeuk34b/predictions_test_lambda-0.30.csv` | -0.30 | NA | 0.560775 | NA | NA | NA | 0.581603 |
| `58211689` | `runs/t1_tabpfn_plus_catboost__20260223_124249__job58211689__sxeuk34b/predictions_test_lambda0.40.csv` | 0.40 | NA | 0.503617 | NA | NA | NA | 0.662549 |
| `58211689` | `runs/t1_tabpfn_plus_catboost__20260223_124249__job58211689__sxeuk34b/predictions_test_lambda0.50.csv` | 0.50 | NA | 0.498741 | NA | NA | NA | 0.669051 |
| `58211421` | `runs/t1_tabpfn_plus_catboost__20260223_123915__job58211421__8mdlwukb/predictions_test_lambda-0.10.csv` | -0.10 | NA | NA | NA | NA | NA | NA |
| `58211421` | `runs/t1_tabpfn_plus_catboost__20260223_123915__job58211421__8mdlwukb/predictions_test_lambda-0.20.csv` | -0.20 | NA | NA | NA | NA | NA | NA |
| `58211421` | `runs/t1_tabpfn_plus_catboost__20260223_123915__job58211421__8mdlwukb/predictions_test_lambda0.17.csv` | 0.17 | NA | NA | NA | NA | NA | NA |
| `58206982` | `runs/t1_tabpfn_plus_catboost__20260223_094521__job58206982__5nc5u3ab/predictions_test_lambda0.10.csv` | 0.10 | NA | 0.550052 | NA | NA | NA | 0.610172 |
| `58206982` | `runs/t1_tabpfn_plus_catboost__20260223_094521__job58206982__5nc5u3ab/predictions_test_lambda0.20.csv` | 0.20 | NA | 0.550039 | NA | NA | NA | 0.610190 |
| `58206982` | `runs/t1_tabpfn_plus_catboost__20260223_094521__job58206982__5nc5u3ab/predictions_test_lambda0.30.csv` | 0.30 | NA | 0.550217 | NA | NA | NA | 0.609938 |
| `58155622` | `runs/t1_tabpfn_plus_catboost__20260222_221551__job58155622__g0s63x42/predictions_test_lambda0.10.csv` | 0.10 | NA | NA | NA | NA | NA | NA |
| `58155622` | `runs/t1_tabpfn_plus_catboost__20260222_221551__job58155622__g0s63x42/predictions_test_lambda0.20.csv` | 0.20 | NA | NA | NA | NA | NA | NA |
| `58155622` | `runs/t1_tabpfn_plus_catboost__20260222_221551__job58155622__g0s63x42/predictions_test_lambda0.30.csv` | 0.30 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.05.csv` | -0.05 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.10.csv` | -0.10 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.15.csv` | -0.15 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.20.csv` | -0.20 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.25.csv` | -0.25 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.30.csv` | -0.30 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.35.csv` | -0.35 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.40.csv` | -0.40 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.45.csv` | -0.45 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda-0.50.csv` | -0.50 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.00.csv` | 0.00 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.01.csv` | 0.01 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.03.csv` | 0.03 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.04.csv` | 0.04 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.05.csv` | 0.05 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.07.csv` | 0.07 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.09.csv` | 0.09 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.11.csv` | 0.11 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.12.csv` | 0.12 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.14.csv` | 0.14 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.15.csv` | 0.15 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.17.csv` | 0.17 | NA | NA | NA | NA | NA | NA |
| `58207476` | `runs/t1_tabpfn_plus_catboost__20260223_120145__job58207476__hcxmg2i8/predictions_test_lambda0.19.csv` | 0.19 | NA | NA | NA | NA | NA | NA |
| `58211417` | `runs/t1_tabpfn_plus_catboost__20260223_123915__job58211417__nrr2si3v/predictions_test_lambda0.07.csv` | 0.07 | NA | 0.525536 | NA | NA | NA | 0.632534 |
| `58211417` | `runs/t1_tabpfn_plus_catboost__20260223_123915__job58211417__nrr2si3v/predictions_test_lambda0.12.csv` | 0.12 | NA | 0.521575 | NA | NA | NA | 0.638054 |
| `58211417` | `runs/t1_tabpfn_plus_catboost__20260223_123915__job58211417__nrr2si3v/predictions_test_lambda0.15.csv` | 0.15 | NA | 0.519672 | NA | NA | NA | 0.640690 |

### Track 1 SAITS experiments
Only rows with imputed-only metrics and RMSE <= 0.6 are shown (custom `metrics` or PyPOTS `oof_metrics`).

| Job ID | Method | Run | Source | Config | RMSE all (imputed-only) | RMSE sensory (imputed-only) | MAE all (imputed-only) | MAE sensory (imputed-only) |
|---|---|---|---|---|---|---|---|---|
| `58213859` | t1_saits_pypots_cv | `t1_saits_pypots_cv__20260223_104348__job58213859__oprrjlx5` | pypots | n_splits=5, seed=42, smoke=False | 0.551660 | 0.552687 | 0.260587 | 0.261510 |
| `57967685` | t1_saits | `t1_saits__20260220_193211__job57967685__ioewsj05` | custom | mask_frac=0.1, epochs=50, lr=0.001, batch_size=128 | 0.543858 | 0.543619 | 0.359613 | 0.358632 |

### Track 1 TabImpute results (train metrics)
TabImpute runs log training-set metrics in `run_summary.json` (no CV metrics available).

| Job ID | Method | Run | RMSE all | RMSE imputed-only | wRMSE all | wRMSE imputed-only | R2 all | R2 imputed-only |
|---|---|---|---|---|---|---|---|---|
| `57231698` | tabimpute_plus | `t1_tabimpute_plus__20260215_141121__job57231698__64b57wfa` | 1.312310 | 1.409788 | 1.317682 | 1.412115 | -1.090257 | -1.644348 |
| `57301611` | tabimpute | `t1_tabimpute__20260216_103554__job57301611__h52c9wg5` | 0.585436 | 0.628922 | 0.586928 | 0.628991 | 0.584007 | 0.473735 |
| `57301610` | tabimpute | `t1_tabimpute__20260216_103553__job57301610__cj3cxd7i` | 0.564268 | 0.591175 | 0.562476 | 0.592347 | 0.613546 | 0.535010 |
| `57301608` | tabimpute | `t1_tabimpute__20260216_103546__job57301608__mwz10cuh` | 0.552025 | 0.575455 | 0.549871 | 0.576593 | 0.630133 | 0.559410 |
| `57301609` | tabimpute | `t1_tabimpute__20260216_103544__job57301609__fla7k8s8` | 0.576702 | 0.619539 | 0.578295 | 0.619738 | 0.596327 | 0.489320 |
| `57301606` | tabimpute | `t1_tabimpute__20260216_103532__job57301606__cidtv32v` | 0.536841 | 0.555379 | 0.534538 | 0.556475 | 0.650201 | 0.589616 |
| `57301607` | tabimpute | `t1_tabimpute__20260216_103528__job57301607__6h4l7unx` | 0.586368 | 0.629924 | 0.587834 | 0.629962 | 0.582681 | 0.472057 |
| `57301605` | tabimpute | `t1_tabimpute__20260216_103502__job57301605__ust3zl0v` | 0.618527 | 0.664471 | 0.620087 | 0.664526 | 0.535651 | 0.412561 |
| `57301604` | tabimpute | `t1_tabimpute__20260216_103441__job57301604__ukjif23g` | 0.595658 | 0.619022 | 0.594024 | 0.620252 | 0.569353 | 0.490173 |
| `57301597` | tabimpute | `t1_tabimpute__20260216_095415__job57301597__zj7fbgmw` | 0.584161 | 0.627553 | 0.585767 | 0.627747 | 0.585817 | 0.476024 |
| `57301596` | tabimpute | `t1_tabimpute__20260216_095217__job57301596__0hu4e0bg` | 0.558925 | 0.585753 | 0.557024 | 0.586913 | 0.620829 | 0.543500 |
| `57301595` | tabimpute | `t1_tabimpute__20260216_095217__job57301595__d5427bzf` | 0.562093 | 0.603846 | 0.563782 | 0.604186 | 0.616519 | 0.514865 |
| `57301594` | tabimpute | `t1_tabimpute__20260216_095217__job57301594__gxmux89d` | 0.535241 | 0.557373 | 0.532905 | 0.558473 | 0.652283 | 0.586664 |
| `57301593` | tabimpute | `t1_tabimpute__20260216_095050__job57301593__n671uiko` | 0.575486 | 0.618233 | 0.576979 | 0.618328 | 0.598028 | 0.491472 |
| `57301592` | tabimpute | `t1_tabimpute__20260216_095026__job57301592__obon7bw2` | 0.524112 | 0.546225 | 0.521652 | 0.547302 | 0.666593 | 0.603033 |
| `57301591` | tabimpute | `t1_tabimpute__20260216_094957__job57301591__oze2o1ci` | 0.599505 | 0.644036 | 0.601408 | 0.644508 | 0.563773 | 0.448137 |
| `57301590` | tabimpute | `t1_tabimpute__20260216_094957__job57301590__j2k8vjoi` | 0.569908 | 0.594443 | 0.568077 | 0.595622 | 0.605782 | 0.529855 |
| `57300643` | tabimpute | `t1_tabimpute__20260216_090907__job57300643__zcopa50e` | 0.577004 | 0.604136 | 0.575215 | 0.605335 | 0.595904 | 0.514398 |
| `57300646` | tabimpute | `t1_tabimpute__20260216_090906__job57300646__qqkd4s15` | 0.619380 | 0.665387 | 0.620875 | 0.665370 | 0.534370 | 0.410939 |
| `57300645` | tabimpute | `t1_tabimpute__20260216_090906__job57300645__tfy46pae` | 0.598205 | 0.625601 | 0.596656 | 0.626846 | 0.565663 | 0.479277 |
| `57300644` | tabimpute | `t1_tabimpute__20260216_090906__job57300644__m5tnsr8u` | 0.592610 | 0.636630 | 0.593960 | 0.636526 | 0.573749 | 0.460757 |
| `57300636` | tabimpute | `t1_tabimpute__20260216_090457__job57300636__bhjx1wz7` | 0.610562 | 0.655914 | 0.612103 | 0.655969 | 0.547534 | 0.427593 |
| `57300635` | tabimpute | `t1_tabimpute__20260216_090415__job57300635__c5sc2d8t` | 0.585285 | 0.611386 | 0.583690 | 0.612601 | 0.584222 | 0.502673 |
| `57300634` | tabimpute | `t1_tabimpute__20260216_090415__job57300634__fn8qt461` | 0.587523 | 0.631165 | 0.588925 | 0.631131 | 0.581035 | 0.469975 |
| `57300633` | tabimpute | `t1_tabimpute__20260216_090415__job57300633__azylnhnj` | 0.569324 | 0.596097 | 0.567509 | 0.597279 | 0.606589 | 0.527235 |
| `57230081` | tabimpute | `t1_tabimpute__20260215_132614__job57230081__r3xs93h5` | 1.129977 | 1.213912 | 1.133536 | 1.214772 | -0.549768 | -0.960584 |

### Track 2
| Job ID | Task | Config | RMSE all | RMSE imputed-only | wRMSE all | wRMSE imputed-only | R2 all | R2 imputed-only |
|---|---|---|---|---|---|---|---|---|
| `58350309` | t2_discrete_seedbag5_proba (track 2) | `runs/t2_discrete_seedbag5_proba__20260224_141548__job58350309__01n1mgw8/config.json` | 0.372493 | 0.400723 | 0.374079 | 0.401465 | 0.832670 | 0.789681 |
| `57223104` | tabpfn_25_discrete_bag5 (track 2) | `runs/t2_tabpfn_25_discrete_bag5__20260215_094004__job57223104__vi6m4eqc/config.json` | 0.372493 | 0.400723 | 0.374079 | 0.401465 | 0.832670 | 0.789681 |
| `57139435` | tabpfn_25_discrete_bag5 (track 2) | `runs/t2_tabpfn_25_discrete_bag5__20260214_014329__job57139435__on5h0rmx/config.json` | 0.372527 | 0.400759 | 0.374112 | 0.401501 | 0.832640 | 0.789643 |
| `57017219` | tabpfn_25_discrete (track 2) | `runs/t2_tabpfn_25_discrete__20260212_211823__job57017219__6627w2wm/config.json` | 0.375374 | 0.403823 | 0.376985 | 0.404583 | 0.830071 | 0.786415 |
| `56990783` | tabpfn_25 (track 2) | `runs/t2_tabpfn_25__20260212_164757__job56990783__mv5wjg9b/config.json` | 0.391249 | 0.420901 | 0.392862 | 0.421624 | 0.815395 | 0.767968 |

### Track 2 TabImpute results (train metrics)
TabImpute runs log training-set metrics in `run_summary.json` (no CV metrics available).

| Job ID | Method | Run | RMSE all | RMSE imputed-only | wRMSE all | wRMSE imputed-only | R2 all | R2 imputed-only |
|---|---|---|---|---|---|---|---|---|
| `57231699` | tabimpute_plus | `t2_tabimpute_plus__20260215_141129__job57231699__7ney72c2` | 1.086743 | 1.169103 | 1.091316 | 1.171211 | -0.424264 | -0.790174 |
| `57230421` | tabimpute | `t2_tabimpute__20260215_133449__job57230421__feajml9k` | 1.006981 | 1.083297 | 1.009775 | 1.083699 | -0.222869 | -0.537039 |
## AutoTabPFN overlay (T24) job summary
| Job ID | Config | Result | Outputs | Leaderboard RMSE |
|---|---|---|---|---|
| `59446549` | `PRESETS=best_quality`, `MAX_TIME=1500`, `N_ENSEMBLE_MODELS=15` | Incomplete: did not finish 24 targets; time-limit exceeded mid-loop | Partial models only; no `run_summary.json` or `predictions_test.csv` in `runs/t1_autotabpfn_overlay_T24__20260307_152430__job59446549__p0hsy6mr/` | NA |
| `59446552` | `PRESETS=medium_quality`, `MAX_TIME=1500`, `N_ENSEMBLE_MODELS=20` | Success: completed 24 targets and wrote summary | `runs/t1_autotabpfn_overlay_T24__20260307_152429__job59446552__bjdt3a0j/` (`run_summary.json`, `predictions_test.csv`) | 0.41601 |
| `59446553` | `PRESETS=medium_quality`, `MAX_TIME=600`, `N_ENSEMBLE_MODELS=10` | Success: completed 24 targets and wrote summary | `runs/t1_autotabpfn_overlay_T24__20260307_152429__job59446553__uud0c503/` (`run_summary.json`, `predictions_test.csv`) | NA |
| `59446862` | `PRESETS=best_quality`, `MAX_TIME=1200`, `N_ENSEMBLE_MODELS=8` | Success: completed 24 targets (DyStack sub-fit first) | `runs/t1_autotabpfn_overlay_T24__20260307_153016__job59446862__f6i3ztt2/` (`run_summary.json`, `predictions_test.csv`) | 0.41594 |
| `59448473` | `PRESETS=medium_quality`, `MAX_TIME=900`, `N_ENSEMBLE_MODELS=10` | Crash: no models trained for one target (`RuntimeError: No models were trained successfully during fit()`) | Run dir exists; no `run_summary.json` or `predictions_test.csv` in `runs/t1_autotabpfn_overlay_T24__20260307_155639__job59448473__acpjkleb/` | NA |
