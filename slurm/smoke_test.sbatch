#!/bin/bash
#SBATCH --job-name=smoke-qwen
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/smoke_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/smoke_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "=== GPU info ==="
nvidia-smi
echo "=================="

# Activate env
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity

# HF + CUDA env
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false

# Run
cd /work/anujs/savana/aicity-track2
python scripts/smoke_test.py
