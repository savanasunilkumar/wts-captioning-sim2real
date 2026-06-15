"""Build spatially-grounded IMAGE VQA training data from SynWTS (TrafficInternVL recipe).

For each (scenario, phase): pick best view (max pedestrian bbox area among views that
have BOTH pedestrian+vehicle boxes -> overhead), render a keyframe with pedestrian=GREEN
/ vehicle=BLUE boxes drawn in, plus a context crop (union x1.5). Pair every VQA question
for that scenario/phase with the grounded image(s) + a role-aware prompt + GT answer.

Output: grounded images under <out_img_dir>, and ms-swift JSONL (images, not video):
  {"messages":[{"role":"user","content":"<image><image>{role_prompt}{Q}{opts}"},
               {"role":"assistant","content":"<letter>"}],
   "images":[global_png, crop_png]}

--max-scenarios N for a small pilot before the full run; --dump-samples K saves K example
grounded PNGs for eyeballing (incl. late phases) without writing JSONL.
"""
from __future__ import annotations
import argparse, json, glob, sys
from pathlib import Path
from PIL import Image, ImageDraw
import decord
decord.bridge.set_bridge('native')
sys.path.insert(0, "/work/anujs/savana/aicity-track2/scripts")
from wts_dataset import WTSDataset

ROLE_PROMPT = ("In this image the pedestrian is outlined in a GREEN box and the vehicle in a BLUE box. "
               "Use their boxes to reason about spatial relationships (relative position, orientation, distance). ")

_P2N = {"pre-recognition":"0","prerecognition":"0","pre_recognition":"0","recognition":"1",
        "judgment":"2","judgement":"2","action":"3","avoidance":"4"}
def to_phase_num(q):
    raw = getattr(q, "phase_number", None)
    if raw is None: raw = getattr(q, "phase_name", None)
    if raw is None: return None
    s = str(raw).strip().lower()
    return s if s in {"0","1","2","3","4"} else _P2N.get(s)

def phase_boxes(bbox_json):
    d = json.loads(Path(bbox_json).read_text())
    byp = {}
    for a in d.get("annotations", []):
        byp.setdefault(str(a.get("phase_number")), []).append((a["image_id"], a["bbox"]))
    return {p: lst[len(lst)//2] for p, lst in byp.items()}

def video_for(bbox_json, videos_root, split):
    scen, view = Path(bbox_json).parts[-3], Path(bbox_json).parts[-2]
    stem = Path(bbox_json).name.replace("_bbox.json", "")
    d = Path(videos_root)/split/scen/view
    cand = d/f"{stem}.mp4"
    if cand.exists(): return cand
    hits = list(d.glob("*.mp4"))
    for h in hits:
        if h.stem == stem: return h
    return hits[0] if hits else None

def best_view(ped_root, scenario, phase):
    """Among overhead views with both boxes, pick the one with max pedestrian area for this phase."""
    best = None
    for pj in glob.glob(f"{ped_root}/{scenario}/*/*_bbox.json"):
        if "._" in pj: continue
        vj = pj.replace("/pedestrian/", "/vehicle/")
        if not Path(vj).exists(): continue
        pb = phase_boxes(pj)
        if phase not in pb: continue
        _, box = pb[phase]
        area = box[2]*box[3]
        if best is None or area > best[0]:
            best = (area, pj, vj)
    return best  # (area, ped_json, veh_json) or None

def render(ped_json, veh_json, videos_root, split, phase, gpath, cpath):
    pb, vb = phase_boxes(ped_json), phase_boxes(veh_json)
    if phase not in pb: return False
    iid, ped = pb[phase]
    video = video_for(ped_json, videos_root, split)
    if video is None: return False
    vr = decord.VideoReader(str(video))
    fr = Image.fromarray(vr[min(iid, len(vr)-1)].asnumpy()).convert("RGB")
    W, H = fr.size
    dr = ImageDraw.Draw(fr)
    dr.rectangle([ped[0],ped[1],ped[0]+ped[2],ped[1]+ped[3]], outline=(0,255,0), width=5)
    boxes=[ped]
    if phase in vb:
        v=vb[phase][1]; dr.rectangle([v[0],v[1],v[0]+v[2],v[1]+v[3]], outline=(0,128,255), width=5); boxes.append(v)
    xs=[b[0] for b in boxes]+[b[0]+b[2] for b in boxes]; ys=[b[1] for b in boxes]+[b[1]+b[3] for b in boxes]
    cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2; w=(max(xs)-min(xs))*1.5; h=(max(ys)-min(ys))*1.5
    fr.crop((max(0,int(cx-w/2)),max(0,int(cy-h/2)),min(W,int(cx+w/2)),min(H,int(cy+h/2)))).save(cpath)
    fr.save(gpath)
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ptmp/anujs/savana/aicity-data/synwts")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out-img", default="/ptmp/anujs/savana/aicity-data/grounded/train_imgs")
    ap.add_argument("--out-jsonl", default="/ptmp/anujs/savana/aicity-data/sft/train_grounded.jsonl")
    ap.add_argument("--max-scenarios", type=int, default=0)
    ap.add_argument("--dump-samples", type=int, default=0)
    args=ap.parse_args()
    ped_root=f"{args.data_root}/data/annotations/bbox_annotated/pedestrian/{args.split}"
    videos_root=f"{args.data_root}/data/videos"
    Path(args.out_img).mkdir(parents=True, exist_ok=True)
    ds=WTSDataset(args.data_root, args.split)
    sids=ds.scenarios()
    if args.max_scenarios: sids=sids[:args.max_scenarios]

    rows=[]; rendered={}; n_q=0; n_norender=0; dumped=0
    for sid in sids:
        for q in ds.load_vqa(sid):
            if q.correct is None: continue
            phase=to_phase_num(q)
            if phase is None: continue
            key=(sid,phase)
            if key not in rendered:
                bv=best_view(ped_root, sid, phase)
                if bv is None: rendered[key]=None
                else:
                    g=f"{args.out_img}/{sid}_p{phase}.png"; c=f"{args.out_img}/{sid}_p{phase}_crop.png"
                    ok=render(bv[1], bv[2], videos_root, args.split, phase, g, c)
                    rendered[key]=(g,c) if ok else None
                    if ok and dumped < args.dump_samples:
                        dumped+=1; print(f"sample {dumped}: {sid} p{phase} area={bv[0]:.0f} -> {g}", flush=True)
            imgs=rendered[key]
            if imgs is None: n_norender+=1; continue
            opts={k:q.options.get(k,"") for k in ("a","b","c","d")}
            prompt=ROLE_PROMPT+f"{q.question}\n(a) {opts['a']}\n(b) {opts['b']}\n(c) {opts['c']}\n(d) {opts['d']}\nAnswer with a single letter only."
            rows.append({"messages":[{"role":"user","content":"<image><image>"+prompt},
                                      {"role":"assistant","content":q.correct}],
                         "images":[imgs[0],imgs[1]]})
            n_q+=1
        if args.dump_samples and dumped>=args.dump_samples and args.max_scenarios==0 and not args.out_jsonl:
            break
    if args.dump_samples and not rows:
        print(f"dumped {dumped} sample images only"); return
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl,"w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print(f"scenarios={len(sids)} grounded_keyframes={sum(1 for v in rendered.values() if v)} "
          f"VQA_examples={n_q} (no-render skipped={n_norender}) -> {args.out_jsonl}", flush=True)

if __name__=="__main__":
    main()
