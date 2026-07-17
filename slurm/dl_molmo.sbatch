#!/bin/bash
#SBATCH --job-name=dl-molmo
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/dl_molmo.log
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
hf download allenai/Molmo2-4B --local-dir /ptmp/anujs/savana/aicity-data/weights/Molmo2-4B
hf download allenai/Molmo2-8B --local-dir /ptmp/anujs/savana/aicity-data/weights/Molmo2-8B
echo "DONE"
