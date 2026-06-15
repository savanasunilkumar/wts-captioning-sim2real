"""Test-time VQA tuning on SynWTS val (has GT) -- sweep frames (fps), resolution
(max_pixels), and self-consistency voting. Reports VQA accuracy head-to-head vs
the v1 baseline config. Free: no submission, uses synthetic-val ground truth.

Frame count is governed by fps with the FPS_MAX_FRAMES env ceiling (set in sbatch).
Voting: when --votes > 1, sample with temperature and take the majority letter.
"""
from __future__ import annotations
import argparse, collections, re, sys, time
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, "/work/anujs/savana/aicity-track2/scripts")
from wts_dataset import WTSDataset

VQA_PROMPT_PHASED = """You are a traffic safety analyst. Watch this video ({phase_name} phase).

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Answer with a single letter only."""
VQA_PROMPT_ENV = """You are a traffic safety analyst. Watch this pedestrian-vehicle traffic video.

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Answer with a single letter only."""

def parse_letter(text: str) -> str:
    m = re.search(r"\b([abcd])\b", text.lower())
    return m.group(1) if m else "a"

def pick_video(view, vids):
    if view in vids and vids[view]:
        return vids[view][0]
    for v in vids.values():
        if v:
            return v[0]
    return None

def gen(model, proc, video, prompt, fps, max_pixels, votes, max_new):
    messages = [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video}", "fps": fps, "max_pixels": max_pixels},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        _, vid = process_vision_info(messages)
        inputs = proc(text=[text], videos=vid, return_tensors="pt", padding=True).to(model.device)
    except Exception:
        inputs = proc(text=[text], videos=[str(video)], return_tensors="pt", padding=True).to(model.device)
    cut = inputs.input_ids.shape[1]
    if votes <= 1:
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return parse_letter(proc.batch_decode(out[:, cut:], skip_special_tokens=True)[0])
    cnt = collections.Counter()
    with torch.inference_mode():
        for _ in range(votes):
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9)
            cnt[parse_letter(proc.batch_decode(out[:, cut:], skip_special_tokens=True)[0])] += 1
    return cnt.most_common(1)[0][0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ptmp/anujs/savana/aicity-data/synwts")
    ap.add_argument("--model-path", default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter-path", default="/ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756")
    ap.add_argument("--max-scenarios", type=int, default=8)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=360 * 640)
    ap.add_argument("--votes", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--tag", default="cfg")
    args = ap.parse_args()

    print(f"loading model + adapter ({args.tag}) ...", flush=True)
    proc = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()

    ds = WTSDataset(args.data_root, "val")
    sids = ds.scenarios()[:args.max_scenarios]
    i, n = map(int, args.shard.split("/"))
    sids = sids[i::n]

    correct = tot = 0
    t0 = time.time()
    for si, sid in enumerate(sids):
        vids = ds.get_video_paths(sid)
        for q in ds.load_vqa(sid):
            if q.correct is None:
                continue
            vfor = "vehicle_view" if q.view == "vehicle_view" else "overhead_view"
            video = pick_video(vfor, vids)
            if not video:
                continue
            opts = {k: q.options.get(k, "") for k in ("a", "b", "c", "d")}
            tmpl = VQA_PROMPT_ENV if q.view == "environment" else VQA_PROMPT_PHASED
            prompt = tmpl.format(phase_name=q.phase_name or "", question=q.question, **opts)
            pred = gen(model, proc, video, prompt, args.fps, args.max_pixels, args.votes, args.max_new)
            tot += 1
            correct += int(pred == q.correct)
        print(f"  [{si+1}/{len(sids)}] {sid}: running acc {100*correct/max(tot,1):.1f}% ({tot} q, {time.time()-t0:.0f}s)", flush=True)

    acc = 100 * correct / tot if tot else 0
    print(f"[RESULT {args.tag}] fps={args.fps} max_pixels={args.max_pixels} votes={args.votes} "
          f"-> VQA {correct}/{tot} = {acc:.2f}%  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
