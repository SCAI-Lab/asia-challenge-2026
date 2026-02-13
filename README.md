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
- Current repo status (this checkout): `runs/` already contains TabPFN outputs (t1 `tabpfn_25`, t1 `tabpfn_25_t1safe`, t1 `tabpfn_25_t1safe_discrete`, t2 `tabpfn_25`, t2 `tabpfn_25_discrete`). Baseline and sweep runs are not present yet. `slurm_logs/` has previous job logs; `venv/` and cache dirs are present and can be regenerated.

## Quick start (Slurm)
```bash
bash scripts/submit_all.sh
```
That submits:
- `slurm/00_setup_all.sbatch` (CPU) - setup venv, stage data, prefetch TabPFN regressor weights
- baselines for both tracks (CPU)
- `tabpfn_25` for both tracks (GPU)

If you want safe/discrete TabPFN or sweeps, use the specific sbatch files listed below.

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

## Safe fixes (TabPFN)
- Enforces single-threaded preprocessing (TABPFN_NUM_WORKERS and OMP/MKL/OPENBLAS/NUMEXPR set to 1).
- Uses a reduced ensemble (n_ensemble_configurations=1, n_estimators=1, n_jobs=1 when supported).
- Falls back to PCA if preprocessing fails (ARPACK issues) in the track1 safe variants.

## Sweeps
- KNN sweep: `scripts/tune_knn_sweep.py` (slurm/30, slurm/31) explores k in `--k-list` and `--weights`.
- Stratified mean sweep: `scripts/tune_strat_sweep.py` (slurm/32, slurm/33) tries time + sex + age_bin + bmi_bin strata with varying `min_group` thresholds (defaults: track1 50/100, track2 30/60).

## Scripts (what each does)
- `scripts/00_env.sh`: defines paths (ASIA2026_ROOT, data/staged, runs, caches) and HF token variables.
- `scripts/10_setup_project.sh`: creates the directory layout under ASIA2026_ROOT.
- `scripts/11_create_venv.sh`: loads modules, creates venv, installs torch + requirements.
- `scripts/12_stage_data.sh`: unpacks Share_Track1.zip and Share_Track2.zip into data/staged/track1 and data/staged/track2.
- `scripts/13_prefetch_tabpfn.sh`: prefetches TabPFN regressor weights (requires network).
- `scripts/14_prefetch_tabpfn_classifier.sh`: prefetches TabPFN classifier weights (needed for discrete variants).
- `scripts/run_tabpfn_t1_safe.py`: Track 1 safe TabPFN run.
- `scripts/run_tabpfn_t1_safe_discrete.py`: Track 1 safe discrete TabPFN run.
- `scripts/run_tabpfn_t2_discrete.py`: Track 2 discrete TabPFN run.
- `scripts/tune_knn_sweep.py`: KNN sweep.
- `scripts/tune_strat_sweep.py`: stratified-mean sweep.
- `scripts/report_run_metrics.py`: aggregate metrics across run directories.
- `scripts/submit_all.sh`: submits setup + baselines + `tabpfn_25` jobs only.

## Slurm batch files (by track)
- `slurm/00_setup_all.sbatch`: setup venv, stage data, prefetch regressor weights.
- `slurm/10_track1_baselines.sbatch` and `slurm/20_track2_baselines.sbatch`: `baseline_time_mean`, `baseline_strat_time_mean`, `baseline_knn15`.
- `slurm/11_track1_tabpfn.sbatch` and `slurm/21_track2_tabpfn.sbatch`: `tabpfn_25` regressor.
- `slurm/12_track1_tabpfn_safe.sbatch`: `tabpfn_25_t1safe`.
- `slurm/13_track1_tabpfn_safe_discrete.sbatch`: `tabpfn_25_t1safe_discrete`.
- `slurm/22_track2_tabpfn_discrete.sbatch`: `tabpfn_25_discrete`.
- `slurm/30_t1_knn_sweep.sbatch` and `slurm/31_t2_knn_sweep.sbatch`: KNN sweeps.
- `slurm/32_t1_strat_sweep.sbatch` and `slurm/33_t2_strat_sweep.sbatch`: stratified-mean sweeps.

## Outputs and run IDs
- Run ID format: `t<track>_<method>__YYYYMMDD_HHMMSS__job<SLURM_JOB_ID>__<rand8>`
- Each run directory under `runs/` contains:
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
