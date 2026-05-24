#!/bin/bash
#SBATCH --job-name=eval-ft
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/eval_ft_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/eval_ft_%A_%a.err

set -euo pipefail
echo "=== FT eval shard $SLURM_ARRAY_TASK_ID on $(hostname) ==="
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false

cd /work/anujs/savana/aicity-track2
python scripts/eval_zeroshot.py \
  --data-root /ptmp/anujs/savana/aicity-data/synwts \
  --split val \
  --adapter-path /ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756 \
  --num-fewshot 0 \
  --scenario-slice ${SLURM_ARRAY_TASK_ID}/8 \
  --out-dir /work/anujs/savana/aicity-track2/outputs/ft_v1
