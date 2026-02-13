# ASIA 2026 (Track 1 + Track 2) — Slurm-first Starter (Baselines + TabPFN 2.5)

This starter pack is designed for **ETH Euler / Slurm** style clusters where:

* **All heavy I/O must go to `$SCRATCH`** (not `$HOME`) and scratch is **auto-purged** after ~2 weeks.
* Compute nodes often require **`module load eth_proxy`** to reach external services (pip/HF). 

It provides **four runnable methods for each track**:

1. `baseline_time_mean`  
   **Copy observed else time-mean**.
2. `baseline_strat_time_mean`  
   **Copy observed else time-mean stratified by metadata** (backoff to time-only).
3. `baseline_knn15`  
   **Copy observed else time-mean**, but for the **15 always-missing targets**, use **KNN on motor+metadata**.
4. `tabpfn_25`  
   **TabPFN 2.5** per-target regression (GPU recommended).

Outputs include:

* Slurm logs: `.out` and `.err` (written to `./slurm_logs/` in the directory you submit from)
* A unique run folder per execution under `$SCRATCH/Acads/asia2026/runs/<RUN_ID>/`
  * `metrics.json`, `config.json`, `predictions_test.csv` (Kaggle submission format)
  * optional `oof_predictions.parquet` when `--do-cv` is enabled

---

## 0) What you run (minimal)

From the unpacked starter directory:

```bash
bash scripts/submit_all.sh
```

That script submits:

1) `slurm/00_setup_all.sbatch` (CPU) — creates venv on `$SCRATCH`, installs deps, stages data, prefetches TabPFN weights.

2) Four experiment jobs (GPU for TabPFN; CPU for baselines).

If you only want a single run, use:

```bash
sbatch slurm/10_track1_baselines.sbatch
sbatch slurm/20_track2_baselines.sbatch
sbatch slurm/11_track1_tabpfn.sbatch
sbatch slurm/21_track2_tabpfn.sbatch
```

---

## 1) Where things go on `$SCRATCH`

The setup job builds a project root:

```text
$SCRATCH/Acads/asia2026/
  repo/                 # copy of this starter pack
  data/
    raw_zips/           # bundled Share_Track1.zip + Share_Track2.zip
    track1/             # extracted CSVs
    track2/             # extracted CSVs
  venv/                 # python venv
  hf_home/              # HF cache + tokens (if any)
  xdg_cache/
  pip_cache/
  runs/<RUN_ID>/        # per-run artifacts (metrics/config/submissions)
  logs/                 # reserved (your Slurm logs go to ./slurm_logs in the submit directory)
```

---

## 2) Data structure expected

After setup, you will have:

```text
$SCRATCH/Acads/asia2026/data/track1/
  features_train_1.csv
  labels_train_1.csv
  metadata_train_1.csv
  features_test_1.csv
  metadata_test_1.csv
  labels_test_1_dummy.csv

$SCRATCH/Acads/asia2026/data/track2/
  features_train_2.csv
  labels_train_2.csv
  metadata_train_2.csv
  features_test_2.csv
  metadata_test_2.csv
  labels_test_2_dummy.csv
```

---

## 3) Reproducibility + run IDs

Each job creates a unique `RUN_ID`:

```
YYYYMMDD_HHMMSS__job<SLURM_JOB_ID>__<8char_random>
```

All results go into:

```text
$SCRATCH/Acads/asia2026/runs/<RUN_ID>/
```

---

## 4) CLI (advanced)

You can also run manually (inside the venv):

```bash
python -m asia2026.run \
  --track 1 \
  --method baseline_knn15 \
  --data-root "$SCRATCH/Acads/asia2026/data" \
  --run-root "$SCRATCH/Acads/asia2026/runs" \
  --do-cv 1 \
  --n-splits 5
```
