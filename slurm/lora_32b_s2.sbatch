#!/bin/bash
#SBATCH --job-name=lora-32b
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=40:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/lora_32b_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/lora_32b_%j.err
set -euo pipefail
echo "=== LoRA 32B on $(hostname) $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export VIDEO_MAX_PIXELS=100352
export FPS=1.0
export FPS_MAX_FRAMES=16
cd /work/anujs/savana/aicity-track2
DATASET="${DATASET:-/ptmp/anujs/savana/aicity-data/sft/train_plus_val.jsonl}"
STEPS_ARG=""
[ -n "${MAX_STEPS:-}" ] && STEPS_ARG="--max_steps ${MAX_STEPS}"
NPROC_PER_NODE=4 swift sft \
  --model /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-32B-s1cap-merged \
  --dataset "$DATASET" \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --target_modules all-linear \
  --freeze_vit true \
  --torch_dtype bfloat16 \
  --deepspeed zero3 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --gradient_checkpointing true \
  --logging_steps 5 \
  --save_steps 150 \
  --save_total_limit 3 \
  --dataloader_num_workers 4 \
  --output_dir "${OUTDIR:-/ptmp/anujs/savana/aicity-outputs/lora_32b}" \
  $STEPS_ARG
echo "DONE $(date)"
