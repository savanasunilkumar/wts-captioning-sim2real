#!/bin/bash
#SBATCH --job-name=merge-s1
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/merge_s1.log
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/merge_s1.log
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
ADP=$(ls -d /ptmp/anujs/savana/aicity-outputs/lora_32b_s1cap/v*/checkpoint-* | sort -V | tail -1)
echo "merging $ADP"
swift export --adapters "$ADP" --merge_lora true \
  --output_dir /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-32B-s1cap-merged
echo "MERGED"
