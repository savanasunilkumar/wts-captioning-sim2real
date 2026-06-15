#!/bin/bash -l
#SBATCH --job-name=inf-cap-v4
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/inf_cap_v4_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/inf_cap_v4_%A_%a.err
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate aicity
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
cd /work/anujs/savana/aicity-track2
ADAPTER=$(ls -dv /ptmp/anujs/savana/aicity-outputs/lora_v4/v*/checkpoint-* 2>/dev/null | tail -1)
echo "Using adapter: $ADAPTER"
python scripts/infer_realtest.py --task captions --shard ${SLURM_ARRAY_TASK_ID}/8 --adapter-path "$ADAPTER" --out-dir /ptmp/anujs/savana/aicity-outputs/realtest_v4_cap
