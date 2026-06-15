#!/bin/bash
#SBATCH --job-name=eval-ftv2
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-7
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/eval_ftv2_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/eval_ftv2_%A_%a.err

set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
module load ffmpeg/7.1-gvzmeku 2>/dev/null || true
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /work/anujs/savana/aicity-track2
python scripts/eval_zeroshot.py \
  --split val \
  --adapter-path /ptmp/anujs/savana/aicity-outputs/lora_v2/v0-20260523-212509/checkpoint-753 \
  --clips-root /ptmp/anujs/savana/aicity-data/synwts_clips \
  --num-fewshot 0 \
  --scenario-slice ${SLURM_ARRAY_TASK_ID}/8 \
  --out-dir /work/anujs/savana/aicity-track2/outputs/ft_v2
