"""Grounded VQA inference on the REAL WTS test set — in-memory, crash-hardened.

Per VQA question: locate the video + its pedestrian/vehicle bbox jsons, render a keyframe
with pedestrian=GREEN / vehicle=BLUE boxes + union crop (x1.5) entirely IN MEMORY (no PNG
round-trip -> dodges PIL's '_idat'/'tile cannot extend' bugs), feed [global, crop] with a
role-aware prompt to the grounded LoRA, parse the letter. Robust: a bad frame / crop / gen
degrades to a plain frame or 'a' instead of killing the shard. Pin the job to a100 — this
env's cu130 torch has no v100/sm_70 kernels. Output: <out>/shard_i_of_n/submission_vqa.json
"""
from __future__ import annotations
import argparse, json, glob, re
from pathlib import Path
import torch
from PIL import Image, ImageDraw
import decord
decord.bridge.set_bridge('native')
from transformers import AutoProcessor, AutoModelForImageTextToText

TEST = "/ptmp/anujs/savana/aicity-data/wts_real_test/unpacked/WTS_DATASET_PUBLIC_TEST"
BBOX_ROOT = "/ptmp/anujs/savana/aicity-data/wts_real_test/bbox_unpacked"
ROLE_PROMPT = ("In this image the pedestrian is outlined in a GREEN box and the vehicle in a BLUE box. "
               "Use their boxes to reason about spatial relationships (relative position, orientation, distance). ")
PLAIN_PROMPT = "You are a traffic safety analyst. Look at this scene. "
_P2N = {"pre-recognition":"0","prerecognition":"0","pre_recognition":"0","recognition":"1",
        "judgment":"2","judgement":"2","action":"3","avoidance":"4"}
def to_num(raw):
    s=str(raw).strip().lower()
    return s if s in {"0","1","2","3","4"} else _P2N.get(s)

def build_index(root, pat):
    idx={}
    for p in glob.glob(f"{root}/{pat}", recursive=True):
        if "._" in p: continue
        idx.setdefault(Path(p).name, p)
    return idx

def build_bbox_indices(root):
    allb=[p for p in glob.glob(f"{root}/**/*_bbox.json", recursive=True) if "._" not in p]
    ped={Path(p).name:p for p in allb if "/pedestrian/" in p or "BDD" in p}  # BDD = pedestrian-only
    veh={Path(p).name:p for p in allb if "/vehicle/" in p}
    return ped, veh

def phase_box(bbox_json, phase):
    """(image_id,[x,y,w,h]) for this phase; else the NEAREST phase's box so we stay grounded."""
    try: d=json.loads(Path(bbox_json).read_text())
    except Exception: return None
    byp={}
    for a in d.get("annotations",[]):
        if not a.get("bbox"): continue
        byp.setdefault(to_num(a.get("phase_number")),[]).append((a["image_id"], a["bbox"]))
    if not byp: return None
    if phase in byp:
        rows=byp[phase]
    else:
        keys=[k for k in byp if k is not None and str(k).isdigit()]
        if not keys: return None
        try: tgt=int(phase)
        except Exception: tgt=0
        rows=byp[min(keys, key=lambda k: abs(int(k)-tgt))]
    return rows[len(rows)//2] if rows else None

def parse_letter(t):
    m=re.search(r"\b([abcd])\b", t.lower()); return m.group(1) if m else "a"

def render(video_path, ped_json, veh_json, phase):
    """Return (kind, global_img, crop_img); kind in 'grounded'|'plain'|None. All in-memory."""
    if video_path is None or not Path(video_path).exists(): return (None, None, None)
    try:
        vr=decord.VideoReader(str(video_path)); n=len(vr)
        pb = phase_box(ped_json, phase) if ped_json else None
        iid = min(max(int(pb[0] if pb else n//2), 0), n-1)
        fr=Image.fromarray(vr[iid].asnumpy()).convert("RGB"); W,H=fr.size
        if pb is None:
            return ("plain", fr, fr)
        dr=ImageDraw.Draw(fr); ped=pb[1]
        dr.rectangle([ped[0],ped[1],ped[0]+ped[2],ped[1]+ped[3]], outline=(0,255,0), width=5)
        boxes=[ped]
        vb = phase_box(veh_json, phase) if veh_json else None
        if vb:
            v=vb[1]; dr.rectangle([v[0],v[1],v[0]+v[2],v[1]+v[3]], outline=(0,128,255), width=5); boxes.append(v)
        xs=[b[0] for b in boxes]+[b[0]+b[2] for b in boxes]; ys=[b[1] for b in boxes]+[b[1]+b[3] for b in boxes]
        cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2; w=(max(xs)-min(xs))*1.5; h=(max(ys)-min(ys))*1.5
        x0=max(0, min(W-2, int(cx-w/2))); y0=max(0, min(H-2, int(cy-h/2)))
        x1=max(x0+1, min(W, int(cx+w/2))); y1=max(y0+1, min(H, int(cy+h/2)))
        return ("grounded", fr, fr.crop((x0,y0,x1,y1)))
    except Exception as ex:
        print(f"  render-fail ({Path(video_path).name if video_path else '?'}): {ex}", flush=True)
        return (None, None, None)

def gen(model, proc, imgs, prompt, maxpix):
    content=[{"type":"image","image":im,"max_pixels":maxpix} for im in imgs]+[{"type":"text","text":prompt}]
    msgs=[{"role":"user","content":content}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        ims,_=process_vision_info(msgs)
        inp=proc(text=[text],images=ims,return_tensors="pt",padding=True).to(model.device)
    except Exception:
        inp=proc(text=[text],images=list(imgs),return_tensors="pt",padding=True).to(model.device)
    with torch.inference_mode():
        out=model.generate(**inp,max_new_tokens=4,do_sample=False)
    return parse_letter(proc.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vqa-file", default="/ptmp/anujs/savana/aicity-data/wts_real_test/WTS_VQA_PUBLIC_TEST.json")
    ap.add_argument("--model-path", default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--max-pixels", type=int, default=602112)
    args=ap.parse_args()
    i,n=map(int,args.shard.split("/")); out=Path(args.out_dir)/f"shard_{i}_of_{n}"; out.mkdir(parents=True,exist_ok=True)

    print("indexing videos + bboxes ...", flush=True)
    vid_idx=build_index(TEST,"**/*.mp4")
    ped_idx,veh_idx=build_bbox_indices(BBOX_ROOT)
    print(f"videos={len(vid_idx)} ped_bbox={len(ped_idx)} veh_bbox={len(veh_idx)}", flush=True)

    proc=AutoProcessor.from_pretrained(args.model_path)
    model=AutoModelForImageTextToText.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to("cuda").eval()
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,args.adapter_path).merge_and_unload()
    print("model+adapter loaded", flush=True)

    data=json.loads(Path(args.vqa_file).read_text()); data=data[i::n]
    sub=[]; ng=npl=nf=gf=0
    for ei,e in enumerate(data):
        vname=e["videos"][0] if e.get("videos") else None
        vpath=vid_idx.get(vname) if vname else None
        stem=vname[:-4] if vname else ""
        pj=ped_idx.get(f"{stem}_bbox.json"); vj=veh_idx.get(f"{stem}_bbox.json")
        for ph in e.get("event_phase",[]):
            phase=to_num(ph["labels"][0])
            kind,gimg,cimg=render(vpath,pj,vj,phase)
            for q in ph.get("conversations",[]):
                qid=q["id"]; opts={k:q.get(k,"") for k in ("a","b","c","d")}
                if kind is None: sub.append({"id":qid,"correct":"a"}); nf+=1; continue
                if kind=="grounded": pr=ROLE_PROMPT; ng+=1
                else: pr=PLAIN_PROMPT; npl+=1
                prompt=pr+f"{q['question']}\n(a) {opts['a']}\n(b) {opts['b']}\n(c) {opts['c']}\n(d) {opts['d']}\nAnswer with a single letter only."
                try:
                    a=gen(model,proc,[gimg,cimg],prompt,args.max_pixels)
                except Exception as ex:
                    a="a"; gf+=1
                    if gf<=5: print(f"  gen-fail qid={qid}: {ex}", flush=True)
                sub.append({"id":qid,"correct":a})
        if (ei+1)%20==0: print(f"  {ei+1}/{len(data)} entries, {len(sub)} ans (grounded={ng} plain={npl} novid={nf} genfail={gf})", flush=True)
    (out/"submission_vqa.json").write_text(json.dumps(sub,indent=2))
    print(f"shard {i}/{n}: {len(sub)} answers (grounded={ng} plain={npl} no-video={nf} gen-fail={gf}) -> {out}", flush=True)

if __name__=="__main__":
    main()
