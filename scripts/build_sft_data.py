"""Build ms-swift SFT JSONL from SynWTS train (both views, full video, plain prompts)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.wts_dataset import WTSDataset

CAP_PROMPT = """You are a traffic safety analyst. Watch this video segment depicting the {phase_name} phase of a pedestrian-vehicle traffic event.

Provide TWO captions in this exact format:
PEDESTRIAN: <pedestrian's position relative to vehicle, attention/line of sight, body action, appearance, and environment>
VEHICLE: <vehicle's position relative to pedestrian, field of view, action and speed, and environment>

Output only those two labeled lines."""

VQA_PHASED = """You are a traffic safety analyst. Watch this video ({phase_name} phase).

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Answer with a single letter only."""

VQA_ENV = """You are a traffic safety analyst. Watch this pedestrian-vehicle traffic video.

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Answer with a single letter only."""

def pick_video(view, vids):
    if view in vids and vids[view]:
        return vids[view][0]
    for v in vids.values():
        if v:
            return v[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ptmp/anujs/savana/aicity-data/synwts")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="/ptmp/anujs/savana/aicity-data/sft/train.jsonl")
    args = ap.parse_args()

    ds = WTSDataset(args.data_root, split=args.split)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cap = n_vqa = 0
    with open(out_path, "w") as f:
        for sid in ds.scenarios():
            caps = ds.load_captions(sid)
            vqas = ds.load_vqa(sid)
            vids = ds.get_video_paths(sid)

            # Captions: both views
            for cv in caps:
                video = pick_video(cv.view, vids)
                if not video:
                    continue
                for seg in cv.segments:
                    target = f"PEDESTRIAN: {seg.pedestrian}\nVEHICLE: {seg.vehicle}"
                    prompt = CAP_PROMPT.format(phase_name=seg.phase_name)
                    f.write(json.dumps({
                        "messages": [
                            {"role": "user", "content": f"<video>{prompt}"},
                            {"role": "assistant", "content": target},
                        ],
                        "videos": [str(video)],
                    }) + "\n")
                    n_cap += 1

            # VQA: all views
            for q in vqas:
                if not q.correct:
                    continue
                opts = {k: q.options.get(k, "") for k in ("a", "b", "c", "d")}
                if q.view == "environment":
                    video = pick_video("overhead_view", vids)
                    prompt = VQA_ENV.format(question=q.question, **opts)
                else:
                    video = pick_video(q.view, vids)
                    prompt = VQA_PHASED.format(phase_name=q.phase_name or "", question=q.question, **opts)
                if not video:
                    continue
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"<video>{prompt}"},
                        {"role": "assistant", "content": q.correct},
                    ],
                    "videos": [str(video)],
                }) + "\n")
                n_vqa += 1

    print(f"Wrote {n_cap} caption + {n_vqa} VQA = {n_cap + n_vqa} examples")
    print(f"Output: {out_path}")

if __name__ == "__main__":
    main()
