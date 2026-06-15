"""Filter the joint SynWTS train JSONL down to VQA-only examples, so we can train
a VQA-specialized LoRA (separate-task training; +8.6% VQA in the dual-model ablation)."""
import json
ORIG = "/ptmp/anujs/savana/aicity-data/sft/train.jsonl"
OUT  = "/ptmp/anujs/savana/aicity-data/sft/train_vqa_only.jsonl"
rows = [json.loads(l) for l in open(ORIG) if l.strip()]
vqa, cap = [], 0
for ex in rows:
    user = ex["messages"][0]["content"]
    asst = ex["messages"][1]["content"].strip()
    is_vqa = ("Answer with a single letter" in user) or (len(asst) == 1 and asst.lower() in "abcd")
    if is_vqa:
        vqa.append(ex)
    else:
        cap += 1
with open(OUT, "w") as f:
    for ex in vqa:
        f.write(json.dumps(ex) + "\n")
# sanity: answer-letter distribution
from collections import Counter
dist = Counter(ex["messages"][1]["content"].strip().lower()[:1] for ex in vqa)
print(f"total {len(rows)} -> VQA-only {len(vqa)}  (caption examples dropped: {cap})")
print("answer distribution:", dict(dist))
print("wrote", OUT)
