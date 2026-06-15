#!/bin/bash -l
#SBATCH --job-name=cosmos-batch
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=anujs-scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/cosmos_batch_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/cosmos_batch_%A_%a.err
module load apptainer/1.4.0-py311-cdh2uiv
apptainer exec --nv --bind /ptmp:/ptmp /ptmp/anujs/savana/aicity-data/cosmos_base.sif bash -c "
source /ptmp/anujs/savana/aicity-data/cosmos-transfer2.5/.venv/bin/activate
export LD_LIBRARY_PATH=/ptmp/anujs/savana/cosmos_ffmpeg6/env/lib:\$LD_LIBRARY_PATH
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export COSMOS_EXPERIMENTAL_CHECKPOINTS=1
cd /ptmp/anujs/savana/aicity-data/cosmos-transfer2.5
python /work/anujs/savana/aicity-track2/scripts/cosmos_batch_augment.py --shard \${SLURM_ARRAY_TASK_ID}/8
echo DONE rc=\$?
"
