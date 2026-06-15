#!/bin/bash
#SBATCH --job-name=fusion-vqa
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/fusion_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/fusion_%A_%a.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export MAX_PIXELS=230400
cd /work/anujs/savana/aicity-track2
python scripts/infer_fusion.py --shard ${SLURM_ARRAY_TASK_ID}/8 --out-dir /ptmp/anujs/savana/aicity-outputs/fusion_vqa
echo "shard ${SLURM_ARRAY_TASK_ID}/8 DONE"
