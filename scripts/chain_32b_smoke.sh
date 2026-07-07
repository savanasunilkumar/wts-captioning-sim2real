#!/bin/bash
# Wait for 32B download to complete, then submit the smoke test.
W=/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-32B-Instruct
LOG=/work/anujs/savana/aicity-track2/logs/dl_32b.log
for i in $(seq 1 240); do
  if grep -q "Fetching 26 files: 100%" "$LOG" 2>/dev/null && ! ls "$W"/.cache/huggingface/download/*.incomplete >/dev/null 2>&1; then
    SZ=$(du -sb "$W" | cut -f1)
    if [ "$SZ" -gt 60000000000 ]; then
      echo "$(date) download complete ($SZ bytes) — submitting smoke"
      sbatch --export=ALL,MAX_STEPS=15,DATASET='/ptmp/anujs/savana/aicity-data/sft/train_plus_val.jsonl#96',OUTDIR=/ptmp/anujs/savana/aicity-outputs/lora_32b_smoke \
        /work/anujs/savana/aicity-track2/slurm/lora_32b.sbatch
      exit 0
    fi
  fi
  sleep 30
done
echo "$(date) TIMEOUT waiting for download"
