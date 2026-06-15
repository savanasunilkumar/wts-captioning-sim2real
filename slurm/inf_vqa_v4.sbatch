#!/bin/bash -l
#SBATCH --job-name=inf-vqa-v4
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/inf_vqa_v4_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/inf_vqa_v4_%A_%a.err
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
cd /work/anujs/savana/aicity-track2
ADAPTER=$(ls -dv /ptmp/anujs/savana/aicity-outputs/lora_v4/v*/checkpoint-* 2>/dev/null | tail -1)
echo "Using adapter: $ADAPTER"
python scripts/infer_realtest.py --task vqa --shard ${SLURM_ARRAY_TASK_ID}/16 --adapter-path "$ADAPTER" --out-dir /ptmp/anujs/savana/aicity-outputs/realtest_v4_vqa
