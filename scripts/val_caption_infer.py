"""Generate captions on SynWTS val to score locally with the official metrics_all.py.

Faithful baseline: imports the VERBATIM CAP_PROMPT + generate() + parse_caption_pair +
norm_phase from infer_realtest (no reimplementation drift). For WTS we generate from the
vehicle_view video and score against the overhead GT (the validated 54.67 config). Writes
a per-shard predictions dict {scenario: [{labels, caption_pedestrian, caption_vehicle}]}
plus a same-scenario diff list (our vs GT) for qualitative analysis.

--prompt-py PATH lets an experiment override CAP_PROMPT (a .py that assigns CAP_PROMPT=...)
so we can A/B phrasing/template prompts without editing this file.
"""
from __future__ import annotations
import sys, json, glob, argparse
from pathlib import Path
import torch
sys.path.insert(0, "/work/anujs/savana/aicity-track2/scripts")
from infer_realtest import CAP_PROMPT, generate, parse_caption_pair, norm_phase
from transformers import AutoProcessor, AutoModelForImageTextToText

CAP_VAL = "/ptmp/anujs/savana/aicity-data/synwts/data/annotations/caption/val"
VID_VAL = "/ptmp/anujs/savana/aicity-data/synwts/data/videos/val"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter-path", default="/ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756")
    ap.add_argument("--out-dir", default="/ptmp/anujs/savana/aicity-outputs/capval_v1")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=360*640)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-py", default=None)
    ap.add_argument("--view", choices=["vehicle","overhead"], default="vehicle")
    args=ap.parse_args()
    i,n=map(int,args.shard.split("/"))
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    cap_prompt=CAP_PROMPT
    if args.prompt_py:
        ns={}; exec(Path(args.prompt_py).read_text(), ns); cap_prompt=ns["CAP_PROMPT"]
        print("USING OVERRIDE PROMPT:\n"+cap_prompt[:400], flush=True)

    proc=AutoProcessor.from_pretrained(args.model_path)
    model=AutoModelForImageTextToText.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to("cuda").eval()
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,args.adapter_path).merge_and_unload()
    print(f"model+adapter loaded ({args.adapter_path})", flush=True)

    jsons=sorted(p for p in glob.glob(f"{CAP_VAL}/*/overhead_view/*_caption.json") if "._" not in p)
    if args.limit: jsons=jsons[:args.limit]
    jsons=jsons[i::n]
    pred={}; diff=[]; shown=0
    for ji,jf in enumerate(jsons):
        key=Path(jf).name.replace("_caption.json","")
        scen=Path(jf).parts[-3]
        if args.view=="overhead":
            c=sorted(x for x in glob.glob(f"{VID_VAL}/{scen}/overhead_view/*.mp4") if "._" not in x)
            vp=c[0] if c else None
        else:
            vp=f"{VID_VAL}/{scen}/vehicle_view/{key}_vehicle_view.mp4"
            if not Path(vp).exists():
                c=[x for x in glob.glob(f"{VID_VAL}/{scen}/vehicle_view/*.mp4") if "._" not in x]
                vp=c[0] if c else None
        gt=json.loads(Path(jf).read_text())
        segs=[]
        for ep in gt.get("event_phase",[]):
            num,name=norm_phase(ep["labels"][0])
            ped=veh=""
            if vp:
                try:
                    raw=generate(model,proc,vp,cap_prompt.format(phase_name=name),args.max_new_tokens,args.fps,args.max_pixels)
                    ped,veh=parse_caption_pair(raw)
                except Exception as ex:
                    print(f"  gen-fail {key} p{num}: {ex}", flush=True)
            segs.append({"labels":[num],"caption_pedestrian":ped,"caption_vehicle":veh})
            diff.append({"key":key,"phase":num,"our_ped":ped,"gt_ped":ep.get("caption_pedestrian",""),
                         "our_veh":veh,"gt_veh":ep.get("caption_vehicle","")})
            if shown<2 and ped:
                shown+=1; print(f"[sample {key} p{num}]\n OUR_PED: {ped[:300]}\n GT_PED : {ep.get('caption_pedestrian','')[:300]}", flush=True)
        pred[key]=segs
        if (ji+1)%5==0: print(f"  {ji+1}/{len(jsons)} scenarios", flush=True)
    (out/f"val_pred_shard{i}of{n}.json").write_text(json.dumps(pred,indent=2))
    (out/f"val_diff_shard{i}of{n}.json").write_text(json.dumps(diff,indent=2,ensure_ascii=False))
    print(f"shard {i}/{n}: {len(pred)} scenarios -> {out}", flush=True)

if __name__=="__main__": main()
