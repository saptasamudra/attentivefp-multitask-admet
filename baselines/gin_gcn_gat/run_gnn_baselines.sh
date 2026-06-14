#!/bin/bash
#SBATCH --job-name=gnn_baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gnn_%A_%a.out
#SBATCH --error=logs/gnn_%A_%a.err
#SBATCH --array=0-134  # 3 models × 9 datasets × 5 seeds = 135 jobs

# ============================================================================
# SLURM Job Array Script for GNN Baseline Experiments
# Following Professor's Phase 1 Priority ① Requirements
#
# Job array breakdown:
# - 3 models (GIN, GCN, GAT)
# - 9 datasets (ESOL, FreeSolv, Lipo, BACE, BBBP, HIV, ClinTox, Tox21, SIDER)
# - 5 seeds (0, 1, 2, 3, 4)
# - Total: 135 jobs
#
# Author: Sapta (林恩)
# Date: 2026-04-01
# Lab: SAMLab, Guizhou University
# ============================================================================

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate molprop

# Create logs directory
mkdir -p logs

# Define arrays
MODELS=("GIN" "GCN" "GAT")
DATASETS=("ESOL" "FreeSolv" "Lipo" "BACE" "BBBP" "HIV" "ClinTox" "Tox21" "SIDER")
SEEDS=(0 1 2 3 4)

# Calculate indices
NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_SEEDS=${#SEEDS[@]}

# Get current job configuration
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_SEEDS)))
DATASET_IDX=$(((SLURM_ARRAY_TASK_ID / NUM_SEEDS) % NUM_DATASETS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "========================================="
echo "SLURM Job ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Run experiment
python gnn_baselines.py \
    --model $MODEL \
    --dataset $DATASET \
    --seed $SEED \
    --data_dir /path/to/moleculenet/data \
    --save_dir ./baselines/saved_models \
    --results_file ./baselines/results/gnn_results.json \
    --hidden_dim 300 \
    --num_layers 3 \
    --dropout 0.0 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --patience 20

echo "Job completed successfully!"
