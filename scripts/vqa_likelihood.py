"""Discriminative VQA scoring (VL-JEPA principle retrofitted onto our Qwen LoRA).

Instead of generating a letter, score each candidate option:
  scheme T: mean token log-prob of the OPTION TEXT as the assistant completion
  scheme L: first-token logits restricted to {a,b,c,d}  (sanity baseline ~= greedy)
Evaluates on SynWTS val (val.jsonl has GT letters) so we get FREE accuracy numbers
vs the 80.01 letter-generation baseline. No submission needed.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

def parse_example(d):
    u=d["messages"][0]["content"]; gt=d["messages"][1]["content"].strip().lower()
    q=u.replace("<video>","")
    opts=dict(re.findall(r"\(([abcd])\)\s*(.*)", q))
    if len(opts)<4 or gt not in "abcd": return None
    stem=q.split("(a)")[0]
    return stem, opts, gt, d["videos"][0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--val-jsonl",default="/ptmp/anujs/savana/aicity-data/sft/val.jsonl")
    ap.add_argument("--model-path",default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter-path",default="/ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756")
    ap.add_argument("--shard",default="0/1")
    ap.add_argument("--out-dir",default="/ptmp/anujs/savana/aicity-outputs/vqa_likelihood_val")
    ap.add_argument("--fps",type=float,default=1.0)
    ap.add_argument("--max-pixels",type=int,default=100352)
    ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args()
    i,n=map(int,args.shard.split("/"))
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    rows=[]
    for line in open(args.val_jsonl):
        d=json.loads(line)
        if d["messages"][1]["content"].strip().lower() in ("a","b","c","d"):
            p=parse_example(d)
            if p: rows.append(p)
    rows=rows[i::n]
    if args.limit: rows=rows[:args.limit]
    print(f"val VQA examples in shard: {len(rows)}",flush=True)

    proc=AutoProcessor.from_pretrained(args.model_path)
    model=AutoModelForImageTextToText.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to("cuda").eval()
    from peft import PeftModel
    model=PeftModel.from_pretrained(model,args.adapter_path).merge_and_unload()
    tok=proc.tokenizer
    letter_ids={L:tok.encode(L,add_special_tokens=False)[0] for L in "abcd"}
    print("model+adapter loaded",flush=True)

    from qwen_vl_utils import process_vision_info
    accT=accL=tot=0; results=[]
    for ei,(stem,opts,gt,video) in enumerate(rows):
        prompt=stem+"\n".join(f"({k}) {v}" for k,v in sorted(opts.items()))+"\nAnswer with a single letter only."
        content=[{"type":"video","video":f"file://{video}","fps":args.fps,"max_pixels":args.max_pixels},{"type":"text","text":prompt}]
        msgs=[{"role":"user","content":content}]
        text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        _,vids=process_vision_info(msgs)
        base=proc(text=[text],videos=vids,return_tensors="pt").to(model.device)
        plen=base.input_ids.shape[1]
        with torch.inference_mode():
            o=model(**base)
            logits=o.logits[0,-1]
            predL=max("abcd",key=lambda L:logits[letter_ids[L]].item())
            pv=None
            if hasattr(o,"past_key_values"): pv=o.past_key_values
            scores={}
            for L in "abcd":
                cand=tok(" "+opts[L].strip(),return_tensors="pt",add_special_tokens=False).input_ids.to(model.device)
                if pv is not None:
                    co=model(input_ids=cand,past_key_values=pv,use_cache=False)
                    lg=torch.cat([logits.unsqueeze(0),co.logits[0,:-1]],0)
                else:
                    full=torch.cat([base.input_ids,cand],1)
                    co=model(input_ids=full); lg=co.logits[0,plen-1:-1]
                lp=torch.log_softmax(lg.float(),-1)
                tokp=lp[torch.arange(cand.shape[1]),cand[0]]
                scores[L]=tokp.mean().item()
            predT=max(scores,key=scores.get)
        accL+=predL==gt; accT+=predT==gt; tot+=1
        results.append({"gt":gt,"predL":predL,"predT":predT})
        if (ei+1)%100==0:
            print(f"  {ei+1}/{len(rows)}  letter-logit={accL/tot:.4f}  option-text={accT/tot:.4f}",flush=True)
    (out/f"shard_{i}_of_{n}.json").write_text(json.dumps({"n":tot,"accL":accL/tot,"accT":accT/tot,"rows":results}))
    print(f"FINAL shard {i}/{n}: n={tot}  letter-logit acc={accL/tot:.4f}  option-text acc={accT/tot:.4f}",flush=True)

if __name__=="__main__":
    main()
