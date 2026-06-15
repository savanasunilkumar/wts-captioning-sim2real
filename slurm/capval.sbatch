#!/bin/bash
#SBATCH --job-name=capval
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/capval_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/capval_%A_%a.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export MAX_PIXELS="${MAXPIX:-230400}"
cd /work/anujs/savana/aicity-track2
ADP="${ADAPTER:-/ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756}"
PROMPT_ARG=""
[ -n "${PROMPT_PY:-}" ] && PROMPT_ARG="--prompt-py ${PROMPT_PY}"
VIEW_ARG=""
[ -n "${VIEW:-}" ] && VIEW_ARG="--view ${VIEW}"
OUTDIR="${OUTDIR:-/ptmp/anujs/savana/aicity-outputs/capval_v1}"
echo "adapter=$ADP outdir=$OUTDIR promptpy=${PROMPT_PY:-none} view=${VIEW:-vehicle} maxpix=${MAXPIX:-230400} fps=${FPS:-1.0} shard=${SLURM_ARRAY_TASK_ID}/4"
python scripts/val_caption_infer.py --shard ${SLURM_ARRAY_TASK_ID}/4 --adapter-path "$ADP" --out-dir "$OUTDIR" $PROMPT_ARG $VIEW_ARG --max-pixels ${MAXPIX:-230400} --fps ${FPS:-1.0}
echo "shard ${SLURM_ARRAY_TASK_ID}/4 DONE"
