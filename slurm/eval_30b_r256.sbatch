#!/bin/bash
#SBATCH --job-name=eval-30br256
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/eval_30br256_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/eval_30br256_%A_%a.err

set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /work/anujs/savana/aicity-track2
python scripts/eval_zeroshot.py \
  --split val \
  --model-path /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-30B-A3B-Instruct \
  --adapter-path /ptmp/anujs/savana/aicity-outputs/lora_30b_r256/v0-20260526-160342/checkpoint-756 \
  --num-fewshot 0 \
  --scenario-slice ${SLURM_ARRAY_TASK_ID}/8 \
  --out-dir /work/anujs/savana/aicity-track2/outputs/ft_30b_r256
