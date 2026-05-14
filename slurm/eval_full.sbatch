#!/bin/bash
#SBATCH --job-name=eval-full
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/eval_full_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/eval_full_%j.err

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false

cd /work/anujs/savana/aicity-track2
python scripts/eval_zeroshot.py \
  --data-root /ptmp/anujs/savana/aicity-data/synwts \
  --split val \
  --max-scenarios 48 \
  --out-dir /work/anujs/savana/aicity-track2/outputs/zeroshot_full
