#!/bin/bash
#SBATCH --job-name=dl-32b
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/dl_32b_slurm.log
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/dl_32b_slurm.log
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
echo "download starting on $(hostname) $(date)"
hf download Qwen/Qwen3-VL-32B-Instruct --local-dir /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-32B-Instruct
echo "download COMPLETE $(date)"
du -sh /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-32B-Instruct
