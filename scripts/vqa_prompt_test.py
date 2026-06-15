"""Targeted prompt A/B on SynWTS val (GT) — baseline VQA prompt vs a reference-frame
-clarified prompt, broken down per question type. Tests whether clarifying the spatial
reference frame fixes the diagnosed weak categories (pedestrian orientation,
vehicle-relative-position). Runs BOTH prompts per question (model loaded once)."""
from __future__ import annotations
import argparse, re, sys, time
from collections import defaultdict
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
sys.path.insert(0, "/work/anujs/savana/aicity-track2/scripts")
from wts_dataset import WTSDataset

BASE_PHASED = """You are a traffic safety analyst. Watch this video ({phase_name} phase).

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Answer with a single letter only."""

REFRAME_PHASED = """You are a traffic safety analyst. Watch this video ({phase_name} phase).

Before answering, fix the spatial reference frame: locate the pedestrian and the vehicle, note which way each is facing and moving, and identify whose viewpoint the question is asked from (pedestrian's or vehicle's).

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Answer with a single letter only (the option correct from the exact reference frame the question specifies)."""

BASE_ENV = BASE_PHASED.replace(" ({phase_name} phase)", "")
REFRAME_ENV = REFRAME_PHASED.replace(" ({phase_name} phase)", "")

def parse_letter(t):
    m = re.search(r"\b([abcd])\b", t.lower()); return m.group(1) if m else "a"

def pick_video(view, vids):
    if view in vids and vids[view]: return vids[view][0]
    for v in vids.values():
        if v: return v[0]
    return None

def gen(model, proc, video, prompt, fps, maxpix):
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video}", "fps": fps, "max_pixels": maxpix},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        _, vi = process_vision_info(msgs)
        inp = proc(text=[text], videos=vi, return_tensors="pt", padding=True).to(model.device)
    except Exception:
        inp = proc(text=[text], videos=[str(video)], return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=4, do_sample=False)
    return parse_letter(proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ptmp/anujs/savana/aicity-data/synwts")
    ap.add_argument("--model-path", default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter-path", default="/ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756")
    ap.add_argument("--max-scenarios", type=int, default=12)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=360 * 640)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()

    ds = WTSDataset(args.data_root, "val")
    sids = ds.scenarios()[:args.max_scenarios]
    res = defaultdict(lambda: {"n": 0, "base_c": 0, "ref_c": 0})
    t0 = time.time()
    for si, sid in enumerate(sids):
        vids = ds.get_video_paths(sid)
        for q in ds.load_vqa(sid):
            if q.correct is None: continue
            vfor = "vehicle_view" if q.view == "vehicle_view" else "overhead_view"
            video = pick_video(vfor, vids)
            if not video: continue
            opts = {k: q.options.get(k, "") for k in ("a", "b", "c", "d")}
            typ = q.question.strip().lower()[:55]
            if q.view == "environment":
                pb = BASE_ENV.format(question=q.question, **opts)
                pr = REFRAME_ENV.format(question=q.question, **opts)
            else:
                pb = BASE_PHASED.format(phase_name=q.phase_name or "", question=q.question, **opts)
                pr = REFRAME_PHASED.format(phase_name=q.phase_name or "", question=q.question, **opts)
            ab = gen(model, proc, video, pb, args.fps, args.max_pixels)
            ar = gen(model, proc, video, pr, args.fps, args.max_pixels)
            r = res[typ]
            r["n"] += 1; r["base_c"] += int(ab == q.correct); r["ref_c"] += int(ar == q.correct)
        print(f"[{si+1}/{len(sids)}] {sid} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'N':>4} {'base':>6} {'reframe':>7} {'delta':>6}  question-type")
    print("-" * 90)
    tn = tb = tr = 0
    for typ, r in sorted(res.items(), key=lambda kv: -kv[1]["n"]):
        b = 100 * r["base_c"] / r["n"]; rf = 100 * r["ref_c"] / r["n"]
        star = "  <==" if (rf - b) >= 5 else ("  !!" if (rf - b) <= -5 else "")
        print(f"{r['n']:>4} {b:>5.1f}% {rf:>6.1f}% {rf-b:>+5.1f}  {typ}{star}")
        tn += r["n"]; tb += r["base_c"]; tr += r["ref_c"]
    print("-" * 90)
    print(f"OVERALL  base={100*tb/tn:.2f}%   reframe={100*tr/tn:.2f}%   delta={100*(tr-tb)/tn:+.2f} pts  (N={tn})")

if __name__ == "__main__":
    main()
