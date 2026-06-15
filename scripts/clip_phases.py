"""Clip SynWTS videos to per-phase segments via ffmpeg mpeg4 re-encode (parallel)."""
from __future__ import annotations
import argparse, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.wts_dataset import WTSDataset

def do_clip(job):
    src, start, end, dst = job
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return True
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(src), "-ss", str(start), "-to", str(end),
           "-c:v", "mpeg4", "-q:v", "4", "-an", str(dst)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ptmp/anujs/savana/aicity-data/synwts")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-root", default="/ptmp/anujs/savana/aicity-data/synwts_clips")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ds = WTSDataset(args.data_root, split=args.split)
    out_root = Path(args.out_root) / args.split
    jobs = []
    for sid in ds.scenarios():
        caps = ds.load_captions(sid)
        vids = ds.get_video_paths(sid)
        for cv in caps:
            srcs = vids.get(cv.view, [])
            if not srcs:
                continue
            for seg in cv.segments:
                dst = out_root / sid / cv.view / f"phase_{seg.phase_num}.mp4"
                jobs.append((str(srcs[0]), seg.start_time, seg.end_time, str(dst)))
    print(f"{args.split}: {len(jobs)} clips, {args.workers} workers...")
    with Pool(args.workers) as p:
        results = p.map(do_clip, jobs)
    print(f"DONE {args.split}: {sum(results)} ok, {len(results)-sum(results)} fail")

if __name__ == "__main__":
    main()
