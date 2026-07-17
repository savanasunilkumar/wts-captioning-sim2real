#!/bin/bash
#SBATCH --job-name=v12-vqa
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/v12vqa_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/v12vqa_%A_%a.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
cd /work/anujs/savana/aicity-track2
ADP=$(ls -d /ptmp/anujs/savana/aicity-outputs/lora_32b_s2vqa/v*/checkpoint-* | sort -V | tail -1)
echo "adapter=$ADP"
python scripts/infer_realtest.py --task vqa --shard ${SLURM_ARRAY_TASK_ID}/4 \
  --model-path /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-32B-s1cap-merged \
  --adapter-path "$ADP" \
  --out-dir /ptmp/anujs/savana/aicity-outputs/v12_vqa
