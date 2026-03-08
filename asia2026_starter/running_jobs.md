 scp -r ppurkayastha@euler.ethz.ch:/<Addrss? Destop/<ID based name?>>            
             
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)

          57917303 gpuhe.24h asia2026 ppurkaya  R       9:27      1 eu-g6-053 Stage 1 no log

57944819   gpu.24h asia2026 ppurkaya  R      16:29      1 eu-lo-g3-047 Stage 2 GPU
          57944792 gpuhe.24h asia2026 ppurkaya  R      15:45      1 eu-g6-006 Stage 2 CPU
         
           57950829 gpuhe.24h asia2026 ppurkaya PD       0:00      1 (Resources) Stage 1 logging


           57951842 gpuhe.24h asia2026 ppurkaya PD       0:00      1 (Resources) Stage 4 SAITS


58116257 gpuhe.24h asia2026 ppurkaya  R       1:20      1 eu-g6-079

58131847 gpuhe.24h asia2026 ppurkaya  R       2:39      1 eu-g6-078
58131844 gpuhe.24h asia2026 ppurkaya  R       2:49      1 eu-g6-079
          58156982 -> Main CatBoost



          58178475 gpuhe.24h asia2026 ppurkaya  R      44:23      1 eu-g6-077 0.4,0.5


        58211689 
          58211415 
          58211417 
          58211421 

58204043 Cat correction level 1 full
   58155622 -> Base
58156982 -> Cat


======

58350163 -> Track 1 Added features 
58350309 -> Track 2 Seedbag



(venv) ppurkayastha@eu-login-39:/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter$ sbatch --export=ALL,TOPK_TARGETS=15,TOPK_SOURCE=test slurm/25_t1_catboost_residual_full.sbatch
Submitted batch job 59367803
(venv) ppurkayastha@eu-login-39:/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter$ sbatch --export=ALL,TOPK_TARGETS=24,TOPK_SOURCE=test slurm/25_t1_catboost_residual_full.sbatch
Submitted batch job 59367879
(venv) ppurkayastha@eu-login-39:/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter$ sbatch --export=ALL,TOPK_TARGETS=32,TOPK_SOURCE=test slurm/25_t1_catboost_residual_full.sbatch
Submitted batch job 59367942
(venv) ppurkayastha@eu-login-39:/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter$ sbatch --export=ALL,TOPK_TARGETS=15,TOPK_SOURCE=train slurm/25_t1_catboost_residual_full.sbatch
Submitted batch job 59368035
(venv) ppurkayastha@eu-login-39:/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter$ sbatch --export=ALL,TOPK_TARGETS=35,TOPK_SOURCE=train slurm/25_t1_catboost_residual_full.sbatch
Submitted batch job 59368082



cd /cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter
PRESETS=best_quality MAX_TIME=1500 N_ENSEMBLE_MODELS=15 sbatch slurm/28_t1_autotabpfn_overlay_T24.sbatch

cd /cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter
PRESETS=best_quality MAX_TIME=1200 N_ENSEMBLE_MODELS=8 sbatch slurm/28_t1_autotabpfn_overlay_T24.sbatch

cd /cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter
PRESETS=medium_quality MAX_TIME=1500 N_ENSEMBLE_MODELS=20 sbatch slurm/28_t1_autotabpfn_overlay_T24.sbatch

cd /cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter
PRESETS=medium_quality MAX_TIME=600 N_ENSEMBLE_MODELS=10 sbatch slurm/28_t1_autotabpfn_overlay_T24.sbatch

cd /cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter
PRESETS=best_quality MAX_TIME=1200 N_ENSEMBLE_MODELS=8 sbatch slurm/28_t1_autotabpfn_overlay_T24.sbatch

cd /cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter
PRESETS=medium_quality MAX_TIME=900 N_ENSEMBLE_MODELS=10 sbatch slurm/28_t1_autotabpfn_overlay_T24.sbatch


Submitted batch job 59446550-> Crashed



Submitted batch job 59446549
Submitted batch job 59446552
Submitted batch job 59446553
Submitted batch job 59446862
Submitted batch job 59448473



cd /cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter
BASELINE_CSV=/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter/runs/t1_discrete_seedbag5_proba__20260222_120028__job58116257__zkw773g3/predictions_test.csv \
PRESETS=best_quality \
MAX_TIME=1800 \
N_ENSEMBLE_MODELS=12 \
N_JOBS=8 \
RAISE_ON_NO_MODELS_FITTED=0 \
sbatch slurm/16d_t1_autotabpfn_overlay_T24.sbatch
Submitted batch job 59503392



TabPFN + CatBoost all lambdas -> Boosting
15 models full local TabPFN -> Bagging
SAITS -> Standalone poor results so skip

=======

AutoTabPFN
BNN
TabCSDI
Only on TopK 2-3 models + classifier
