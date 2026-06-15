#!/bin/bash -l
#SBATCH --job-name=inf-vqa-v5
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=anujs-scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/inf_vqa_v5_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/inf_vqa_v5_%A_%a.err
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export FPS_MAX_FRAMES=32
cd /work/anujs/savana/aicity-track2
python scripts/infer_realtest.py --task vqa --shard ${SLURM_ARRAY_TASK_ID}/16 \
  --adapter-path /ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756 \
  --fps 2.0 --max-pixels 921600 \
  --out-dir /ptmp/anujs/savana/aicity-outputs/realtest_v5_vqa_hires
