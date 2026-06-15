"""Batch Cosmos-Transfer augmentation of SynWTS train videos (distilled edge).

Enumerates training videos, shards by SLURM array index, writes one spec per
video, then calls examples/inference.py ONCE with all of this shard's specs
(model loads a single time, all clips run sequentially). Outputs land flat in
OUT_FLAT/<stem>.mp4 where <stem> = e.g. 20230707_12_SN17_T1_vehicle_view.

Engine (validated): edge/distilled, num_steps=4, guidance=3, control_weight=0.5.
Resumable: skips any clip whose output already exists.
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

VIDEOS_ROOT = "/ptmp/anujs/savana/aicity-data/synwts/data/videos/train"
PROMPT      = "/ptmp/anujs/savana/aicity-data/synwts_realism_prompt.txt"
OUT_FLAT    = "/ptmp/anujs/savana/aicity-data/synwts_aug_flat"
SPEC_DIR    = "/ptmp/anujs/savana/aicity-data/aug_specs"
REPO        = "/ptmp/anujs/savana/aicity-data/cosmos-transfer2.5"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n: take every nth video")
    ap.add_argument("--view", default="vehicle_view")
    ap.add_argument("--guidance", type=float, default=3)
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--control-weight", type=float, default=0.5)
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/"))

    vids = sorted(Path(VIDEOS_ROOT).glob(f"*/{args.view}/*.mp4"))
    shard = vids[i::n]
    print(f"[shard {i}/{n}] {len(shard)}/{len(vids)} {args.view} videos", flush=True)

    Path(SPEC_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUT_FLAT).mkdir(parents=True, exist_ok=True)

    specs, todo = [], []
    for v in shard:
        name = v.stem
        if (Path(OUT_FLAT) / f"{name}.mp4").exists():
            print(f"  skip (done): {name}", flush=True); continue
        spec = {
            "name": name,
            "prompt_path": PROMPT,
            "video_path": str(v),
            "guidance": args.guidance,
            "num_steps": args.num_steps,
            "edge": {"control_path": str(v), "control_weight": args.control_weight},
        }
        sp = Path(SPEC_DIR) / f"{name}.json"
        sp.write_text(json.dumps(spec))
        specs.append(str(sp)); todo.append(name)

    if not specs:
        print("  nothing to do (all done)", flush=True); return
    print(f"  augmenting {len(specs)}: {todo[:3]}{'...' if len(todo)>3 else ''}", flush=True)

    cmd = ["python", "examples/inference.py", "-i", *specs,
           "-o", OUT_FLAT, "--model=edge/distilled",
           "--offload-guardrail-models", "--keep-going"]
    subprocess.run(cmd, cwd=REPO, check=False)

    done = sum(1 for nm in todo if (Path(OUT_FLAT) / f"{nm}.mp4").exists())
    print(f"[shard {i}/{n}] produced {done}/{len(specs)} outputs", flush=True)

if __name__ == "__main__":
    main()
