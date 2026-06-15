#!/bin/bash -l
#SBATCH --job-name=vqa-sweep
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=anujs-scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/vqa_sweep_%x_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/vqa_sweep_%x_%j.err
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export FPS_MAX_FRAMES=${FPSMAXFRAMES:-16}
cd /work/anujs/savana/aicity-track2
python scripts/vqa_testtime.py --tag "$TAG" --fps "$FPS" --max-pixels "$MAXPIX" --votes "$VOTES" --max-scenarios 8
