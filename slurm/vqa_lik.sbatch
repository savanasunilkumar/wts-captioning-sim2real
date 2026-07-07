#!/bin/bash
#SBATCH --job-name=vqa-lik
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/vqalik_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/vqalik_%A_%a.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
cd /work/anujs/savana/aicity-track2
python scripts/vqa_likelihood.py --shard ${SLURM_ARRAY_TASK_ID}/2
echo "shard ${SLURM_ARRAY_TASK_ID}/2 DONE"
