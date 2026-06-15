#!/bin/bash -l
#SBATCH --job-name=ivl-q
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=anujs-scavenger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/ivl_q_%j.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/ivl_q_%j.err
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
export HF_HOME=/ptmp/anujs/savana/aicity-data/weights/.hf_cache
export TOKENIZERS_PARALLELISM=false
export NUM_FRAMES=16
cd /work/anujs/savana/aicity-track2
swift sft \
  --model /ptmp/anujs/savana/aicity-data/weights/InternVL3_5-38B-HF \
  --dataset /ptmp/anujs/savana/aicity-data/sft/train.jsonl \
  --tuner_type lora --lora_rank 64 --lora_alpha 128 \
  --target_modules all-linear --freeze_vit true \
  --quant_method bnb --quant_bits 4 --bnb_4bit_compute_dtype bfloat16 \
  --torch_dtype bfloat16 --max_length 16384 --max_steps 20 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --gradient_checkpointing true \
  --logging_steps 1 --save_steps 1000 --dataloader_num_workers 4 \
  --output_dir /ptmp/anujs/savana/aicity-outputs/ivl_q_smoke
