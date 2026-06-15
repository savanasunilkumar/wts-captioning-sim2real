"""Build a MIXED training set: original SynWTS examples + augmented copies.

For every example whose video has a Cosmos-augmented version in synwts_aug_flat,
emit a duplicate example with the video path swapped to the augmented clip.
Mixed = originals + augmented copies (model sees both synthetic and real-looking).
"""
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="/ptmp/anujs/savana/aicity-data/sft/train.jsonl")
    ap.add_argument("--aug-flat", default="/ptmp/anujs/savana/aicity-data/synwts_aug_flat")
    ap.add_argument("--out-mixed", default="/ptmp/anujs/savana/aicity-data/sft/train_mixed.jsonl")
    ap.add_argument("--out-aug", default="/ptmp/anujs/savana/aicity-data/sft/train_aug_only.jsonl")
    args = ap.parse_args()

    aug_dir = Path(args.aug_flat)
    orig = [json.loads(l) for l in open(args.orig) if l.strip()]
    aug_examples = []
    for ex in orig:
        vids = ex.get("videos", [])
        if not vids:
            continue
        stem = Path(vids[0]).stem            # e.g. 20230707_12_SN17_T1_vehicle_view
        aug = aug_dir / f"{stem}.mp4"
        if aug.exists():
            aug_examples.append({"messages": ex["messages"], "videos": [str(aug)]})

    with open(args.out_aug, "w") as f:
        for ex in aug_examples:
            f.write(json.dumps(ex) + "\n")
    with open(args.out_mixed, "w") as f:
        for ex in orig:
            f.write(json.dumps(ex) + "\n")
        for ex in aug_examples:
            f.write(json.dumps(ex) + "\n")

    uniq_aug_vids = len({e["videos"][0] for e in aug_examples})
    print(f"original examples : {len(orig)}")
    print(f"augmented copies  : {len(aug_examples)}  (from {uniq_aug_vids} augmented videos)")
    print(f"mixed total       : {len(orig) + len(aug_examples)}")
    print(f"wrote: {args.out_mixed}")

if __name__ == "__main__":
    main()
