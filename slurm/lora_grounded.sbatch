#!/bin/bash -l
#SBATCH --job-name=lora-grnd
#SBATCH --partition=nova
#SBATCH --qos=normal
#SBATCH --account=anujs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/lora_grnd_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/lora_grnd_%j.err
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export MAX_PIXELS=602112
cd /work/anujs/savana/aicity-track2
swift sft \
  --model /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct \
  --dataset /ptmp/anujs/savana/aicity-data/sft/train_grounded.jsonl \
  --tuner_type lora --lora_rank 64 --lora_alpha 128 \
  --target_modules all-linear --freeze_vit true \
  --torch_dtype bfloat16 --num_train_epochs 3 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 --warmup_ratio 0.03 --gradient_checkpointing true \
  --logging_steps 5 --save_steps 300 --save_total_limit 2 --dataloader_num_workers 8 \
  --output_dir /ptmp/anujs/savana/aicity-outputs/lora_grounded
