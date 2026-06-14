@echo off
REM ============================================================================
REM Local Batch Script for GNN Baseline Experiments (Windows)
REM Fallback option if SLURM cluster unavailable
REM
REM WARNING: This will run sequentially on your GTX 1660 Ti
REM Estimated time: 2-3 days for all 135 experiments
REM
REM Author: Sapta (林恩)
REM Date: 2026-04-01
REM ============================================================================

echo ====================================================================
echo GNN BASELINE EXPERIMENTS - LOCAL EXECUTION
echo ====================================================================
echo.
echo This script will run 135 experiments sequentially:
echo - 3 models (GIN, GCN, GAT)
echo - 9 datasets (ESOL, FreeSolv, Lipo, BACE, BBBP, HIV, ClinTox, Tox21, SIDER)
echo - 5 seeds (0, 1, 2, 3, 4)
echo.
echo Estimated total time on GTX 1660 Ti: 2-3 days
echo.
echo Press Ctrl+C to cancel, or
pause

REM Activate conda environment
call conda activate molprop

REM Create directories
if not exist "baselines\saved_models" mkdir baselines\saved_models
if not exist "baselines\results" mkdir baselines\results
if not exist "logs" mkdir logs

REM Define data directory (UPDATE THIS PATH)
set DATA_DIR=D:\molprop_project\data\moleculenet

REM ============================================================================
REM EXPERIMENT LOOP
REM ============================================================================

set MODELS=GIN GCN GAT
set DATASETS=ESOL FreeSolv Lipo BACE BBBP HIV ClinTox Tox21 SIDER
set SEEDS=0 1 2 3 4

echo.
echo Starting experiments...
echo Start time: %date% %time%
echo.

set /a TOTAL_JOBS=0
set /a COMPLETED_JOBS=0

REM Count total jobs
for %%M in (%MODELS%) do (
    for %%D in (%DATASETS%) do (
        for %%S in (%SEEDS%) do (
            set /a TOTAL_JOBS+=1
        )
    )
)

echo Total jobs to run: %TOTAL_JOBS%
echo.

REM Run experiments
for %%M in (%MODELS%) do (
    for %%D in (%DATASETS%) do (
        for %%S in (%SEEDS%) do (
            set /a COMPLETED_JOBS+=1
            
            echo ====================================================================
            echo Job !COMPLETED_JOBS! / %TOTAL_JOBS%
            echo Model: %%M ^| Dataset: %%D ^| Seed: %%S
            echo Time: %date% %time%
            echo ====================================================================
            
            python gnn_baselines.py ^
                --model %%M ^
                --dataset %%D ^
                --seed %%S ^
                --data_dir %DATA_DIR% ^
                --save_dir baselines/saved_models ^
                --results_file baselines/results/gnn_results.json ^
                --hidden_dim 300 ^
                --num_layers 3 ^
                --dropout 0.0 ^
                --epochs 100 ^
                --batch_size 64 ^
                --lr 1e-3 ^
                --patience 20 ^
                > logs/%%M_%%D_seed%%S.log 2>&1
            
            if errorlevel 1 (
                echo ERROR: Job failed! Check logs/%%M_%%D_seed%%S.log
                pause
            ) else (
                echo SUCCESS: Completed %%M on %%D ^(seed %%S^)
            )
            echo.
        )
    )
)

echo ====================================================================
echo ALL EXPERIMENTS COMPLETED
echo End time: %date% %time%
echo Total jobs: %TOTAL_JOBS%
echo ====================================================================
echo.
echo Results saved to: baselines\results\gnn_results.json
echo.
pause
