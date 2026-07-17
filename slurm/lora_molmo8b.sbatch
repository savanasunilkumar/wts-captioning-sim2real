#!/bin/bash
#SBATCH --job-name=molmo8b
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/molmo8b_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/molmo8b_%j.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export TOKENIZERS_PARALLELISM=false
cd /work/anujs/savana/aicity-track2
DATASET="${DATASET:-/ptmp/anujs/savana/aicity-data/sft/tpv_vqa.jsonl}"
STEPS_ARG=""
[ -n "${MAX_STEPS:-}" ] && STEPS_ARG="--max_steps ${MAX_STEPS}"
NPROC="${NPROC:-4}"
NPROC_PER_NODE=$NPROC swift sft \
  --model /ptmp/anujs/savana/aicity-data/weights/Molmo2-8B \
  --model_type molmo2 \
  --dataset "$DATASET" \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --target_modules all-linear \
  --freeze_vit true \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --gradient_checkpointing true \
  --logging_steps 5 \
  --save_steps 200 \
  --save_total_limit 2 \
  --dataloader_num_workers 4 \
  --output_dir "${OUTDIR:-/ptmp/anujs/savana/aicity-outputs/lora_molmo8b}" \
  $STEPS_ARG
echo "DONE $(date)"
