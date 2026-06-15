#!/bin/bash
#SBATCH --job-name=lora-cap
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/lora_cap_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/lora_cap_%j.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export MAX_PIXELS=230400
cd /work/anujs/savana/aicity-track2
NPROC_PER_NODE=1 swift sft \
  --model /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct \
  --dataset /ptmp/anujs/savana/aicity-data/sft/train_cap_only.jsonl \
  --tuner_type lora --lora_rank 128 --lora_alpha 256 \
  --target_modules all-linear --freeze_vit true \
  --torch_dtype bfloat16 --num_train_epochs 4 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --logging_steps 5 --save_steps 200 --save_total_limit 3 \
  --dataloader_num_workers 8 \
  --output_dir /ptmp/anujs/savana/aicity-outputs/lora_cap
echo "DONE"
