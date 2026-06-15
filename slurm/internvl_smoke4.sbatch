#!/bin/bash -l
#SBATCH --job-name=ivl-smoke4
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=anujs-scavenger
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/ivl_smoke4_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/ivl_smoke4_%j.err
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export VIDEO_MAX_PIXELS=100352
export FPS=1.0
export FPS_MAX_FRAMES=16
cd /work/anujs/savana/aicity-track2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NPROC_PER_NODE=4 swift sft \
  --model /ptmp/anujs/savana/aicity-data/weights/InternVL3_5-38B \
  --dataset /ptmp/anujs/savana/aicity-data/sft/train.jsonl \
  --tuner_type lora --lora_rank 64 --lora_alpha 128 \
  --target_modules all-linear --freeze_vit true \
  --torch_dtype bfloat16 --deepspeed zero3 \
  --max_steps 20 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --gradient_checkpointing true \
  --logging_steps 1 --save_steps 1000 --dataloader_num_workers 4 \
  --output_dir /ptmp/anujs/savana/aicity-outputs/internvl_smoke
