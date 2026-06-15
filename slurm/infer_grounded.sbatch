#!/bin/bash
#SBATCH --job-name=infer-grnd
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/infer_grnd_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/infer_grnd_%A_%a.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export MAX_PIXELS=602112
cd /work/anujs/savana/aicity-track2

# Prefer the final checkpoint-540; else fall back to the latest checkpoint-* present.
ADP=$(ls -d /ptmp/anujs/savana/aicity-outputs/lora_grounded/v*/checkpoint-540 2>/dev/null | tail -1 || true)
if [ -z "${ADP:-}" ]; then
  ADP=$(ls -d /ptmp/anujs/savana/aicity-outputs/lora_grounded/v*/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
fi
if [ -z "${ADP:-}" ]; then echo "FATAL: no checkpoint found under lora_grounded/v*/"; exit 1; fi
echo "adapter=$ADP  shard=${SLURM_ARRAY_TASK_ID}/4  $(date)"

python scripts/infer_grounded.py \
  --adapter-path "$ADP" \
  --out-dir /ptmp/anujs/savana/aicity-outputs/infer_grounded_v6 \
  --shard ${SLURM_ARRAY_TASK_ID}/4 \
  --max-pixels 602112
echo "shard ${SLURM_ARRAY_TASK_ID}/4 DONE $(date)"
