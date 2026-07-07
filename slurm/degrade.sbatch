#!/bin/bash
#SBATCH --job-name=degrade
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/degrade_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/degrade_%A_%a.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
cd /work/anujs/savana/aicity-track2
python scripts/degrade_videos.py --list-file /ptmp/anujs/savana/aicity-data/degrade_list.txt --shard ${SLURM_ARRAY_TASK_ID}/16
