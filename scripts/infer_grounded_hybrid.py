"""Corrected grounded VQA — hybrid with a fallback answer set (default: v1 banked).

Fixes vs the failed v6 inference:
  1. BEST-VIEW: for each (scenario, phase) pick the overhead view exactly like training
     (among views with BOTH ped+veh bbox jsons, max pedestrian box area at that phase).
  2. EXACT-PHASE ONLY: no nearest-phase fallback. If the phase has no box -> fallback.
  3. FALLBACK = a prior submission's answers (v1 by default), so worst case == fallback.
Only grounded questions hit the GPU; everything else copies the fallback answer.
Output: <out>/shard_i_of_n/submission_vqa.json  (covers ALL question ids in the shard)
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
_P2N = {"pre-recognition":"0","prerecognition":"0","pre_recognition":"0","recognition":"1",
        "judgment":"2","judgement":"2","action":"3","avoidance":"4"}
def to_num(raw):
    s=str(raw).strip().lower()
    return s if s in {"0","1","2","3","4"} else _P2N.get(s)

def phase_boxes(bbox_json):
    try: d=json.loads(Path(bbox_json).read_text())
    except Exception: return {}
    byp={}
    for a in d.get("annotations",[]):
        if not a.get("bbox"): continue
        byp.setdefault(to_num(a.get("phase_number")),[]).append((a["image_id"],a["bbox"]))
    return {p:l[len(l)//2] for p,l in byp.items() if p is not None}

def parse_letter(t):
    m=re.search(r"\b([abcd])\b",t.lower()); return m.group(1) if m else None

def build_indices():
    ped=[p for p in glob.glob(BBOX_ROOT+"/**/pedestrian/**/*_bbox.json",recursive=True) if "._" not in p]
    by_scen={}
    for pj in ped:
        parts=Path(pj).parts
        try: scen=parts[-3]
        except Exception: continue
        vj=pj.replace("/pedestrian/","/vehicle/")
        if Path(vj).exists():                      # training's requirement: both boxes exist
            by_scen.setdefault(scen,[]).append((pj,vj))
    vids={}
    for v in glob.glob(TEST+"/**/*.mp4",recursive=True):
        if "._" in v: continue
        vids.setdefault(Path(v).name,v)
    return by_scen,vids

def best_view(cands,phase):
    best=None
    for pj,vj in cands:
        pb=phase_boxes(pj)
        if phase not in pb: continue               # EXACT phase only
        iid,box=pb[phase]; area=box[2]*box[3]
        if area<100: continue
        if best is None or area>best[0]: best=(area,pj,vj,pb)
    return best

def render(pj,vj,pb,phase,vids):
    stem=Path(pj).name.replace("_bbox.json","")
    vp=vids.get(stem+".mp4")
    if vp is None:
        sib=sorted(Path(pj).parent.name+"/" for _ in [0])  # no-op guard
        cand=[v for n,v in vids.items() if n.startswith(stem)]
        vp=cand[0] if cand else None
    if vp is None: return None
    try:
        vr=decord.VideoReader(vp); n=len(vr)
        iid,ped=pb[phase]; iid=min(max(int(iid),0),n-1)
        fr=Image.fromarray(vr[iid].asnumpy()).convert("RGB"); W,H=fr.size
        dr=ImageDraw.Draw(fr)
        dr.rectangle([ped[0],ped[1],ped[0]+ped[2],ped[1]+ped[3]],outline=(0,255,0),width=5)
        boxes=[ped]
        vb=phase_boxes(vj)
        if phase in vb:
            v=vb[phase][1]; dr.rectangle([v[0],v[1],v[0]+v[2],v[1]+v[3]],outline=(0,128,255),width=5); boxes.append(v)
        xs=[b[0] for b in boxes]+[b[0]+b[2] for b in boxes]; ys=[b[1] for b in boxes]+[b[1]+b[3] for b in boxes]
        cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2; w=(max(xs)-min(xs))*1.5; h=(max(ys)-min(ys))*1.5
        x0=max(0,min(W-2,int(cx-w/2))); y0=max(0,min(H-2,int(cy-h/2)))
        x1=max(x0+1,min(W,int(cx+w/2))); y1=max(y0+1,min(H,int(cy+h/2)))
        return fr,fr.crop((x0,y0,x1,y1))
    except Exception as ex:
        print(f"  render-fail {stem}: {ex}",flush=True); return None

def gen(model,proc,imgs,prompt,maxpix):
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
    ap.add_argument("--vqa-file",default="/ptmp/anujs/savana/aicity-data/wts_real_test/WTS_VQA_PUBLIC_TEST.json")
    ap.add_argument("--model-path",default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter-path",default="/ptmp/anujs/savana/aicity-outputs/lora_grounded/v0-20260614-012307/checkpoint-540")
    ap.add_argument("--fallback-vqa",default="/ptmp/anujs/savana/aicity-outputs/realtest_v1_vqa/merged/submission_vqa.json")
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--shard",default="0/1")
    ap.add_argument("--max-pixels",type=int,default=602112)
    args=ap.parse_args()
    i,n=map(int,args.shard.split("/")); out=Path(args.out_dir)/f"shard_{i}_of_{n}"; out.mkdir(parents=True,exist_ok=True)

    fb={r["id"]:r["correct"] for r in json.loads(Path(args.fallback_vqa).read_text())}
    by_scen,vids=build_indices()
    scens=sorted(by_scen.keys(),key=len,reverse=True)
    print(f"scenarios_with_both_boxes={len(by_scen)} videos={len(vids)} fallback_answers={len(fb)}",flush=True)

    proc=AutoProcessor.from_pretrained(args.model_path)
    model=AutoModelForImageTextToText.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to("cuda").eval()
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,args.adapter_path).merge_and_unload()
    print("model+adapter loaded",flush=True)

    data=json.loads(Path(args.vqa_file).read_text()); data=data[i::n]
    sub=[]; ng=nfb=0; cache={}
    for ei,e in enumerate(data):
        vname=e["videos"][0] if e.get("videos") else ""
        stem=vname[:-4] if vname.endswith(".mp4") else vname
        scen=next((s for s in scens if stem.startswith(s)),None)
        for ph in e.get("event_phase",[]):
            phase=to_num(ph["labels"][0])
            key=(scen,phase); imgs=None
            if scen is not None:
                if key not in cache:
                    bv=best_view(by_scen[scen],phase)
                    cache[key]=render(bv[1],bv[2],bv[3],phase,vids) if bv else None
                imgs=cache[key]
            for q in ph.get("conversations",[]):
                qid=q["id"]
                if imgs is None:
                    sub.append({"id":qid,"correct":fb.get(qid,"a")}); nfb+=1; continue
                o={k:q.get(k,"") for k in ("a","b","c","d")}
                prompt=ROLE_PROMPT+f"{q['question']}\n(a) {o['a']}\n(b) {o['b']}\n(c) {o['c']}\n(d) {o['d']}\nAnswer with a single letter only."
                try: a=gen(model,proc,list(imgs),prompt,args.max_pixels)
                except Exception: a=None
                if a is None: sub.append({"id":qid,"correct":fb.get(qid,"a")}); nfb+=1
                else: sub.append({"id":qid,"correct":a}); ng+=1
        if (ei+1)%25==0: print(f"  {ei+1}/{len(data)} entries (grounded={ng} fallback={nfb})",flush=True)
    (out/"submission_vqa.json").write_text(json.dumps(sub,indent=2))
    print(f"shard {i}/{n}: {len(sub)} answers (grounded={ng} fallback={nfb}) -> {out}",flush=True)

if __name__=="__main__":
    main()
