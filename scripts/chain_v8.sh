#!/bin/bash
for i in $(seq 1 240); do
  CK=$(ls -d /ptmp/anujs/savana/aicity-outputs/lora_32b/v*/checkpoint-754 2>/dev/null | tail -1)
  if [ -n "$CK" ] && [ -f "$CK/adapter_model.safetensors" ]; then
    sleep 60  # let the save finish
    V=$(sbatch --parsable /work/anujs/savana/aicity-track2/slurm/infer_32b_vqa.sbatch)
    C=$(sbatch --parsable --dependency=afterany:$V /work/anujs/savana/aicity-track2/slurm/infer_32b_cap.sbatch)
    echo "$(date) checkpoint ready -> vqa=$V captions=$C (chained)"
    exit 0
  fi
  sleep 60
done
echo "timeout"
