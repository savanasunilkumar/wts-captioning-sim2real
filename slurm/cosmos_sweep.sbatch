#!/bin/bash -l
#SBATCH --job-name=cosmos-sweep
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=anujs-scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/cosmos_sweep_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/cosmos_sweep_%j.err
module load apptainer/1.4.0-py311-cdh2uiv
apptainer exec --nv --bind /ptmp:/ptmp /ptmp/anujs/savana/aicity-data/cosmos_base.sif bash -c "
source /ptmp/anujs/savana/aicity-data/cosmos-transfer2.5/.venv/bin/activate
export LD_LIBRARY_PATH=/ptmp/anujs/savana/cosmos_ffmpeg6/env/lib:\$LD_LIBRARY_PATH
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
cd /ptmp/anujs/savana/aicity-data/cosmos-transfer2.5
python examples/inference.py -i $SPEC -o $OUTDIR --model $MODEL --offload-guardrail-models
echo DONE rc=\$?
"
