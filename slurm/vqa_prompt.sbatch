#!/bin/bash -l
#SBATCH --job-name=vqa-prompt
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=anujs-scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/vqa_prompt_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/vqa_prompt_%j.err
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
cd /work/anujs/savana/aicity-track2
python scripts/vqa_prompt_test.py --max-scenarios 12
