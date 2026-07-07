"""Real-ESRGAN-style photometric degradation of SynWTS training videos (sim2real).

Per video: sample ONE consistent degradation chain (like a real camera) —
gamma/brightness/contrast/saturation shift -> Gaussian blur -> downscale+upscale
(bilinear/bicubic) -> Gaussian noise -> per-frame JPEG round-trip -> low-bitrate
H.264 encode. Strengths biased toward components that SURVIVE the model's
~100k-pixel preprocessing (color response, blur/softness, codec artifacts).

Writes a mirror tree under --out-root preserving relative paths. Sharded.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import decord
decord.bridge.set_bridge('native')
import imageio

def sample_chain(rng):
    return {
        "gamma":   rng.uniform(0.75, 1.35),
        "bright":  rng.uniform(0.85, 1.15),
        "contrast":rng.uniform(0.80, 1.20),
        "sat":     rng.uniform(0.70, 1.15),
        "blur":    rng.uniform(0.4, 2.2),
        "scale":   rng.uniform(0.45, 0.85),
        "interp_dn": rng.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]),
        "interp_up": rng.choice([Image.BILINEAR, Image.BICUBIC]),
        "noise":   rng.uniform(2, 14),
        "jpeg_q":  rng.randint(30, 70),
        "crf":     rng.randint(28, 34),
    }

def degrade_frame(im, p):
    im = ImageEnhance.Brightness(im).enhance(p["bright"])
    im = ImageEnhance.Contrast(im).enhance(p["contrast"])
    im = ImageEnhance.Color(im).enhance(p["sat"])
    a = np.asarray(im).astype(np.float32) / 255.0
    a = np.power(np.clip(a, 0, 1), p["gamma"])
    im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(p["blur"]))
    W, H = im.size
    im = im.resize((max(2,int(W*p["scale"])), max(2,int(H*p["scale"]))), p["interp_dn"]).resize((W, H), p["interp_up"])
    a = np.asarray(im).astype(np.float32)
    a = a + np.random.randn(*a.shape).astype(np.float32) * p["noise"]
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=p["jpeg_q"]); buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-file", required=True, help="txt file of absolute video paths")
    ap.add_argument("--src-root", default="/ptmp/anujs/savana/aicity-data/synwts/data/videos")
    ap.add_argument("--out-root", default="/ptmp/anujs/savana/aicity-data/synwts_degraded/videos")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/"))
    vids = [l.strip() for l in open(args.list_file) if l.strip()][i::n]
    done = skip = fail = 0
    for k, vp in enumerate(vids):
        rel = str(Path(vp).relative_to(args.src_root))
        out = Path(args.out_root) / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 10000:
            skip += 1; continue
        rng = random.Random(hash((args.seed, rel)) & 0xffffffff)
        p = sample_chain(rng)
        np.random.seed(hash((args.seed, rel, 1)) & 0xffffffff)
        try:
            vr = decord.VideoReader(vp)
            fps = float(vr.get_avg_fps()) or 30.0
            w = imageio.get_writer(str(out), fps=fps, codec="libx264",
                                   ffmpeg_params=["-crf", str(p["crf"]), "-pix_fmt", "yuv420p"],
                                   macro_block_size=2)
            for fi in range(len(vr)):
                fr = Image.fromarray(vr[fi].asnumpy()).convert("RGB")
                w.append_data(degrade_frame(fr, p))
            w.close()
            done += 1
        except Exception as ex:
            fail += 1
            print(f"FAIL {rel}: {ex}", flush=True)
        if (done + skip + fail) % 10 == 0:
            print(f"[{i}/{n}] {done+skip+fail}/{len(vids)} (done={done} skip={skip} fail={fail})", flush=True)
    print(f"shard {i}/{n} FINISHED: done={done} skip={skip} fail={fail}", flush=True)

if __name__ == "__main__":
    main()
