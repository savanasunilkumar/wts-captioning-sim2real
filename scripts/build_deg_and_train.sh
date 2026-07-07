#!/bin/bash
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
python3 - <<'PY'
import json, random
SRC='/ptmp/anujs/savana/aicity-data/synwts/data/videos'
DEG='/ptmp/anujs/savana/aicity-data/synwts_degraded/videos'
from pathlib import Path
rng=random.Random(7)
vids=sorted({v for line in open('/ptmp/anujs/savana/aicity-data/sft/train_plus_val.jsonl') for v in json.loads(line).get('videos',[])})
keep_clean={v for v in vids if rng.random()<0.2}
n_deg=n_clean=n_missing=0
with open('/ptmp/anujs/savana/aicity-data/sft/train_plus_val_deg.jsonl','w') as f:
    for line in open('/ptmp/anujs/savana/aicity-data/sft/train_plus_val.jsonl'):
        d=json.loads(line); out=[]
        for v in d.get('videos',[]):
            dv=v.replace(SRC,DEG)
            if v in keep_clean: out.append(v); n_clean+=1
            elif Path(dv).exists() and Path(dv).stat().st_size>10000: out.append(dv); n_deg+=1
            else: out.append(v); n_missing+=1
        d['videos']=out
        f.write(json.dumps(d)+'\n')
print(f'deg={n_deg} clean={n_clean} missing_fallback={n_missing}')
PY
T=$(sbatch --parsable --export=ALL,DATASET=/ptmp/anujs/savana/aicity-data/sft/train_plus_val_deg.jsonl,OUTDIR=/ptmp/anujs/savana/aicity-outputs/lora_32b_deg /work/anujs/savana/aicity-track2/slurm/lora_32b.sbatch)
echo "retrain job: $T"
cat > /tmp/v9_vqa.sbatch <<VEOF
#!/bin/bash
#SBATCH --job-name=v9-vqa
#SBATCH --partition=nova
#SBATCH --account=anujs
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/work/anujs/savana/aicity-track2/logs/v9vqa_%A_%a.out
#SBATCH --error=/work/anujs/savana/aicity-track2/logs/v9vqa_%A_%a.err
set -euo pipefail
source /work/anujs/savana/miniforge3/etc/profile.d/conda.sh
conda activate swift-train
cd /work/anujs/savana/aicity-track2
ADP=\$(ls -d /ptmp/anujs/savana/aicity-outputs/lora_32b_deg/v*/checkpoint-754 2>/dev/null | tail -1)
[ -z "\$ADP" ] && ADP=\$(ls -d /ptmp/anujs/savana/aicity-outputs/lora_32b_deg/v*/checkpoint-* | sort -V | tail -1)
echo "adapter=\$ADP"
python scripts/infer_realtest.py --task vqa --shard \${SLURM_ARRAY_TASK_ID}/4 \
  --model-path /ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-32B-Instruct \
  --adapter-path "\$ADP" \
  --out-dir /ptmp/anujs/savana/aicity-outputs/v9_vqa
VEOF
V=$(sbatch --parsable --dependency=afterok:$T /tmp/v9_vqa.sbatch)
echo "v9 VQA inference chained: $V (after $T)"
