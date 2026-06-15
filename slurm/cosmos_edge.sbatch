#!/bin/bash -l
#SBATCH --job-name=cosmos-edge
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/cosmos_edge_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/cosmos_edge_%j.err
module load apptainer/1.4.0-py311-cdh2uiv
apptainer exec --nv --bind /ptmp:/ptmp /ptmp/anujs/savana/aicity-data/cosmos_base.sif bash -c '
source /ptmp/anujs/savana/aicity-data/cosmos-transfer2.5/.venv/bin/activate
export LD_LIBRARY_PATH=/ptmp/anujs/savana/cosmos_ffmpeg6/env/lib:$LD_LIBRARY_PATH
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
cd /ptmp/anujs/savana/aicity-data/cosmos-transfer2.5
echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== RUN ==="
python examples/inference.py -i /ptmp/anujs/savana/aicity-data/synwts_test_spec.json -o /ptmp/anujs/savana/aicity-data/cosmos_test_out --model edge --offload-guardrail-models
echo "=== DONE rc=$? ==="
'
