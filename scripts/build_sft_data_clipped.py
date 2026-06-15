"""Build ms-swift SFT JSONL using per-phase clips, verifying each is decord-readable."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.wts_dataset import WTSDataset
import decord

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

_cache = {}
def clip_ok(p: Path) -> bool:
    key = str(p)
    if key in _cache:
        return _cache[key]
    ok = False
    try:
        if p.exists() and p.stat().st_size > 0:
            vr = decord.VideoReader(key)
            ok = len(vr) > 0
    except Exception:
        ok = False
    _cache[key] = ok
    return ok

def phase_clip(clips_root, split, sid, view, phase_num):
    p = Path(clips_root) / split / sid / view / f"phase_{phase_num}.mp4"
    return p if clip_ok(p) else None

def pick_full(view, vids):
    if view in vids and vids[view]:
        return vids[view][0]
    for v in vids.values():
        if v:
            return v[0]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ptmp/anujs/savana/aicity-data/synwts")
    ap.add_argument("--clips-root", default="/ptmp/anujs/savana/aicity-data/synwts_clips")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="/ptmp/anujs/savana/aicity-data/sft/train_clipped.jsonl")
    args = ap.parse_args()

    ds = WTSDataset(args.data_root, split=args.split)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    n_cap = n_vqa = n_skip = 0
    for idx, sid in enumerate(ds.scenarios()):
        pass
    with open(out_path, "w") as f:
        for sid in ds.scenarios():
            caps = ds.load_captions(sid)
            vqas = ds.load_vqa(sid)
            vids = ds.get_video_paths(sid)
            for cv in caps:
                for seg in cv.segments:
                    clip = phase_clip(args.clips_root, args.split, sid, cv.view, seg.phase_num)
                    if clip is None:
                        n_skip += 1; continue
                    target = f"PEDESTRIAN: {seg.pedestrian}\nVEHICLE: {seg.vehicle}"
                    f.write(json.dumps({"messages": [
                        {"role": "user", "content": f"<video>{CAP_PROMPT.format(phase_name=seg.phase_name)}"},
                        {"role": "assistant", "content": target}], "videos": [str(clip)]}) + "\n")
                    n_cap += 1
            for q in vqas:
                if not q.correct:
                    continue
                opts = {k: q.options.get(k, "") for k in ("a", "b", "c", "d")}
                if q.view == "environment":
                    video = pick_full("overhead_view", vids)
                    prompt = VQA_ENV.format(question=q.question, **opts)
                else:
                    video = phase_clip(args.clips_root, args.split, sid, q.view, q.phase_num)
                    prompt = VQA_PHASED.format(phase_name=q.phase_name or "", question=q.question, **opts)
                if video is None:
                    n_skip += 1; continue
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": f"<video>{prompt}"},
                    {"role": "assistant", "content": q.correct}], "videos": [str(video)]}) + "\n")
                n_vqa += 1
    print(f"Wrote {n_cap} caption + {n_vqa} VQA = {n_cap+n_vqa} ({n_skip} skipped as unreadable)")

if __name__ == "__main__":
    main()
