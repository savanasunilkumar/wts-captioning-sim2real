"""Spatially-grounded keyframe rendering for WTS (TrafficInternVL recipe):
pick best view per phase (max pedestrian bbox area), draw pedestrian=GREEN /
vehicle=BLUE boxes onto the keyframe, crop to the union of boxes * 1.5
(context-preserving). Foundation for the box-grounding VQA pipeline.

--test renders ONE example to a PNG so we can eyeball box placement before scaling.
"""
from __future__ import annotations
import argparse, json, glob, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import decord
decord.bridge.set_bridge('native')

def load_phase_boxes(bbox_json):
    """{phase_number: (image_id, [x,y,w,h])} — one representative (middle) box per phase."""
    d = json.loads(Path(bbox_json).read_text())
    byp = {}
    for a in d.get("annotations", []):
        p = str(a.get("phase_number"))
        byp.setdefault(p, []).append((a["image_id"], a["bbox"]))
    return {p: lst[len(lst)//2] for p, lst in byp.items()}   # middle frame of the phase

def bbox_to_video(bbox_json, videos_root, split):
    """Map a bbox json path to its source video .mp4."""
    name = Path(bbox_json).name.replace("_bbox.json", ".mp4")
    scen = Path(bbox_json).parts[-3]
    view = Path(bbox_json).parts[-2]
    cand = Path(videos_root) / split / scen / view / name
    if cand.exists():
        return cand
    # fallback: any mp4 in that scenario/view whose stem matches the bbox stem
    hits = list((Path(videos_root) / split / scen / view).glob("*.mp4"))
    stem = Path(bbox_json).name.replace("_bbox.json", "")
    for h in hits:
        if h.stem == stem or stem in h.stem:
            return h
    return hits[0] if hits else None

def union_crop(boxes, W, H, scale=1.5):
    xs = [b[0] for b in boxes] + [b[0]+b[2] for b in boxes]
    ys = [b[1] for b in boxes] + [b[1]+b[3] for b in boxes]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    w, h = (x1-x0)*scale, (y1-y0)*scale
    return (max(0,int(cx-w/2)), max(0,int(cy-h/2)), min(W,int(cx+w/2)), min(H,int(cy+h/2)))

def draw_box(draw, box, color, label):
    x, y, w, h = box
    draw.rectangle([x, y, x+w, y+h], outline=color, width=4)
    draw.text((x, max(0,y-12)), label, fill=color)

def render_keyframe(ped_json, veh_json, videos_root, split, phase, out_png):
    pb = load_phase_boxes(ped_json)
    vb = load_phase_boxes(veh_json) if veh_json else {}
    if phase not in pb:
        return f"phase {phase} not in pedestrian boxes (have {list(pb)})"
    img_id, ped_box = pb[phase]
    video = bbox_to_video(ped_json, videos_root, split)
    if video is None:
        return "no video found"
    vr = decord.VideoReader(str(video))
    idx = min(img_id, len(vr)-1)
    frame = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
    W, H = frame.size
    draw = ImageDraw.Draw(frame)
    draw_box(draw, ped_box, (0,255,0), "pedestrian")
    boxes = [ped_box]
    if phase in vb:
        draw_box(draw, vb[phase][1], (0,128,255), "vehicle")
        boxes.append(vb[phase][1])
    crop = frame.crop(union_crop(boxes, W, H, 1.5))
    frame.save(out_png)
    crop.save(out_png.replace(".png", "_crop.png"))
    return f"OK video={video.name} frame={idx}/{len(vr)} ped={ped_box} veh={vb.get(phase,['',''])[1] if phase in vb else None} size={W}x{H}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ptmp/anujs/savana/aicity-data/synwts")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--phase", default="3")
    ap.add_argument("--out", default="/ptmp/anujs/savana/aicity-data/grounded_test.png")
    args = ap.parse_args()
    ped_root = f"{args.data_root}/data/annotations/bbox_annotated/pedestrian/{args.split}"
    scen_glob = args.scenario if args.scenario else "*"
    # only consider views where BOTH ped+veh boxes exist (-> overhead); pick max pedestrian-area example (best-view)
    cands = []
    for pj in glob.glob(f"{ped_root}/{scen_glob}/*/*_bbox.json"):
        if "._" in pj:
            continue
        vj = pj.replace("/pedestrian/", "/vehicle/")
        if not Path(vj).exists():
            continue  # skip vehicle_view (no vehicle box) -> keeps overhead
        pb = load_phase_boxes(pj)
        for ph, (iid, box) in pb.items():
            area = box[2] * box[3]
            if area >= 100:  # skip degenerate tiny boxes
                cands.append((area, pj, vj, ph))
        if args.scenario is None and len(cands) > 200:
            break
    if not cands:
        print("NO viable overhead ped+veh example with a non-degenerate box found"); sys.exit(1)
    cands.sort(reverse=True)
    area, ped_json, veh_json, phase = cands[0]
    print(f"best-view pick: area={area:.0f}px phase={phase}")
    print(f"ped_json: {ped_json}")
    print(f"veh_json: {veh_json}")
    print(render_keyframe(ped_json, veh_json, f"{args.data_root}/data/videos", args.split, phase, args.out))
    print(f"saved: {args.out} (+ _crop.png)")

if __name__ == "__main__":
    main()
