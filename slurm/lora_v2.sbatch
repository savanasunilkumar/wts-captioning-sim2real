#!/bin/bash
#SBATCH --job-name=lora-v2
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/lora_v2_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/lora_v2_%j.err

set -euo pipefail
echo "=== LoRA v2 (clipped, 1 GPU) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VIDEO_MAX_PIXELS=100352
export FPS=1.0
export FPS_MAX_FRAMES=16

cd /work/anujs/savana/aicity-track2
swift sft \
  --model /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct \
  --dataset /ptmp/anujs/savana/aicity-data/sft/train_clipped.jsonl \
  --tuner_type lora \
  --lora_rank 64 \
  --lora_alpha 128 \
  --target_modules all-linear \
  --freeze_vit true \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --gradient_checkpointing true \
  --logging_steps 5 \
  --save_steps 200 \
  --save_total_limit 3 \
  --dataloader_num_workers 4 \
  --output_dir /ptmp/anujs/savana/aicity-outputs/lora_v2
