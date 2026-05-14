"""Merge per-shard submission JSONs into final files + local VQA accuracy."""
from __future__ import annotations
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--in-dir", required=True)
p.add_argument("--out-dir", default=None)
args = p.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir) if args.out_dir else in_dir / "merged"
out_dir.mkdir(parents=True, exist_ok=True)

shards = sorted(in_dir.glob("shard_*"))
print(f"Found {len(shards)} shards: {[s.name for s in shards]}")

captions, vqa, vqa_gt = {}, [], []
for s in shards:
    cp = s / "submission_captions.json"
    vp = s / "submission_vqa.json"
    gp = s / "vqa_gt.json"
    if cp.exists(): captions.update(json.loads(cp.read_text()))
    if vp.exists(): vqa.extend(json.loads(vp.read_text()))
    if gp.exists(): vqa_gt.extend(json.loads(gp.read_text()))

(out_dir / "submission_captions.json").write_text(json.dumps(captions, indent=2))
(out_dir / "submission_vqa.json").write_text(json.dumps(vqa, indent=2))
if vqa_gt: (out_dir / "vqa_gt.json").write_text(json.dumps(vqa_gt, indent=2))

print(f"\n=== Merged into {out_dir} ===")
print(f"  Captions: {len(captions)} scenarios, {sum(len(v) for v in captions.values())} segments")
print(f"  VQA: {len(vqa)} questions")
if vqa_gt:
    gt_map = {g['id']: g['correct'] for g in vqa_gt}
    correct = sum(1 for x in vqa if gt_map.get(x['id']) == x['correct'])
    acc = correct / len(vqa_gt) * 100 if vqa_gt else 0
    print(f"  Local VQA accuracy: {correct}/{len(vqa_gt)} = {acc:.2f}%")
