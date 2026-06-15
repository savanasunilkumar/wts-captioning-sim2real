"""Inference on the REAL WTS public test set -> AI City Track 2 submission JSONs.

Reuses Qwen3-VL-8B + LoRA (v1) with the validated prompts. Handles the real
public-test layout (WTS multi-view + external BDD_PC_5K) and the flat VQA file
with UUID question ids.

Per-shard outputs (merged later by merge_realtest.py):
  submission_captions.json : {scenario_or_video: [{labels, caption_pedestrian, caption_vehicle, start_time, end_time}]}
  submission_vqa.json      : [{id, correct}]

Format verified against woven-visionai/wts-dataset evaluation/metrics_all.py:
  - WTS key = scenario id (prefix 2023...), BDD key = videoNNN (no .mp4 ext).
  - GT segments come from overhead_view (vehicle_view is skipped) -> enumerate phases from overhead.
  - labels is the phase NUMBER string "0".."4"; segments matched by labels.
  - VQA "correct" is a single lowercase letter a/b/c/d, matched by the UUID id.
"""
from __future__ import annotations
import argparse, json, re, time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---------- phase mapping ----------
PHASE_NUM_TO_NAME = {"0": "pre-recognition", "1": "recognition", "2": "judgment", "3": "action", "4": "avoidance"}
PHASE_NAME_TO_NUM = {"pre-recognition": "0", "prerecognition": "0", "pre_recognition": "0", "recognition": "1",
                     "judgment": "2", "judgement": "2", "action": "3", "avoidance": "4"}

def norm_phase(label) -> tuple[str, str]:
    label = str(label)
    if label in PHASE_NUM_TO_NAME:
        return label, PHASE_NUM_TO_NAME[label]
    if label in PHASE_NAME_TO_NUM:
        return PHASE_NAME_TO_NUM[label], label
    return label, label

# ---------- prompts (verbatim from the validated 54.67 eval) ----------
CAP_PROMPT = """You are a traffic safety analyst. Watch this video segment depicting the {phase_name} phase of a pedestrian-vehicle traffic event.

Provide TWO captions in this exact format:
PEDESTRIAN: <pedestrian's position relative to vehicle, attention/line of sight, body action, appearance, and environment>
VEHICLE: <vehicle's position relative to pedestrian, field of view, action and speed, and environment>

Output only those two labeled lines."""

VQA_PROMPT_PHASED = """You are a traffic safety analyst. Watch this video ({phase_name} phase).

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Answer with a single letter only."""

VQA_PROMPT_COT = """You are a traffic safety analyst. Watch this video segment from the {phase_name} phase of a pedestrian-vehicle traffic event.

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Think step by step about what you observe in the video (pedestrian position/orientation, vehicle action, environment), briefly evaluate each option, then end with one final line in this exact format: "Answer: X" where X is one of a, b, c, or d."""

# ---------- parsing ----------
def parse_caption_pair(text: str) -> tuple[str, str]:
    text = text.strip()
    # Split on the "VEHICLE:" marker -- REQUIRE the colon, so the *word* "vehicle"
    # appearing inside the pedestrian caption cannot trigger a false split.
    m = re.search(r"(?i)\bVEHICLE\s*:", text)
    if m:
        ped_part, veh = text[:m.start()], text[m.end():].strip()
    else:
        ped_part, veh = text, ""
    ped = re.sub(r"(?i)^\s*PEDESTRIAN\s*:?\s*", "", ped_part.strip()).strip()
    return ped, veh

def parse_letter(text: str) -> str:
    m = re.search(r"\b([abcd])\b", text.lower())
    return m.group(1) if m else "a"

def parse_letter_cot(text: str) -> str:
    """Extract a/b/c/d from a chain-of-thought response. Prefer an explicit
    'Answer: X' / 'final answer: X' marker (CoT prompt asks for this); else
    fall back to the LAST single-letter token in the response."""
    t = text.lower()
    for pat in (r"final\s+answer\s*[:=]\s*\(?([abcd])\)?",
                r"answer\s*[:=]\s*\(?([abcd])\)?",
                r"answer\s+is\s+\(?([abcd])\)?",
                r"the\s+correct\s+option\s+is\s+\(?([abcd])\)?"):
        m = re.search(pat, t)
        if m:
            return m.group(1)
    matches = re.findall(r"\b([abcd])\b", t)
    if matches:
        return matches[-1]
    return "a"

# ---------- generation (verbatim pattern from validated eval) ----------
def generate(model, processor, video_path, prompt, max_new_tokens, fps, max_pixels) -> str:
    messages = [{"role": "user", "content": [
        {"type": "video", "video": f"file://{video_path}", "fps": fps, "max_pixels": max_pixels},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        from qwen_vl_utils import process_vision_info
        _, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], videos=video_inputs, return_tensors="pt", padding=True).to(model.device)
    except Exception:
        inputs = processor(text=[text], videos=[str(video_path)], return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

# ---------- data ----------
def build_video_index(test_root: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in test_root.rglob("*.mp4"):
        idx.setdefault(p.name, p)
    return idx

def caption_units(test_root: Path) -> list[dict]:
    """Mirror metrics_all.read_gt EXACTLY so our keys match the ground truth 1:1:
    recursively find every *_caption.json, skip vehicle_view (its captions duplicate
    overhead), and derive the key as filename.strip('_caption.json')."""
    units = []
    for jf in sorted(test_root.rglob("*_caption.json")):
        if "vehicle_view" in str(jf):
            continue  # scorer skips vehicle_view; overhead is the GT source
        key = jf.name.strip("_caption.json")  # replicate read_gt key derivation verbatim
        data = json.loads(jf.read_text())
        segs = []
        for ep in data.get("event_phase", []):
            num, name = norm_phase(ep["labels"][0])
            segs.append((num, name, ep.get("start_time", ""), ep.get("end_time", "")))
        if "BDD_PC_5K" in str(jf):
            video = data.get("video_name") or f"{key}.mp4"
            kind = "bdd"
        else:
            video = f"{key}_vehicle_view.mp4"   # generate from vehicle view (validated config)
            kind = "wts"
        units.append({"key": key, "video": video, "segs": segs, "kind": kind})
    return units

def resolve_video(vindex: dict[str, Path], basename: str, key: str | None = None) -> Path | None:
    vp = vindex.get(basename)
    if vp is not None:
        return vp
    if key:  # fallback: any video whose name starts with the scenario id
        cand = sorted(p for nm, p in vindex.items() if nm.startswith(key))
        if cand:
            return cand[0]
    return None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", default="/ptmp/anujs/savana/aicity-data/wts_real_test/unpacked/WTS_DATASET_PUBLIC_TEST")
    ap.add_argument("--vqa-file", default="/ptmp/anujs/savana/aicity-data/wts_real_test/WTS_VQA_PUBLIC_TEST.json")
    ap.add_argument("--model-path", default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter-path", default="/ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", choices=["captions", "vqa"], required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: take every nth work item starting at i")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=360 * 640)
    ap.add_argument("--vqa-mode", choices=["simple", "cot"], default="simple",
                    help="simple = 4-token letter (v1 validated); cot = chain-of-thought, 256-token (v3)")
    args = ap.parse_args()

    i, n = map(int, args.shard.split("/"))
    out_dir = Path(args.out_dir) / f"shard_{i}_of_{n}"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_root = Path(args.test_root)

    print(f"Loading {args.model_path} ...", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
        print(f"merged adapter: {args.adapter_path}", flush=True)
    print(f"loaded in {time.time()-t0:.1f}s VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    vindex = build_video_index(test_root)
    print(f"video index: {len(vindex)} mp4 files", flush=True)

    if args.task == "captions":
        units = caption_units(test_root)
        total = len(units)
        units = units[i::n]
        print(f"[captions] shard {i}/{n}: {len(units)}/{total} units", flush=True)
        cap = defaultdict(list)
        for ui, u in enumerate(units):
            vp = resolve_video(vindex, u["video"], u["key"])
            t = time.time()
            for (num, name, st, et) in u["segs"]:
                if vp is None:
                    ped, veh = "", ""
                else:
                    resp = generate(model, processor, vp, CAP_PROMPT.format(phase_name=name),
                                    max_new_tokens=320, fps=args.fps, max_pixels=args.max_pixels)
                    ped, veh = parse_caption_pair(resp)
                cap[u["key"]].append({"labels": [num], "caption_pedestrian": ped, "caption_vehicle": veh,
                                       "start_time": st, "end_time": et})
            flag = "" if vp is not None else " !!NO VIDEO!!"
            print(f"  [{ui+1}/{len(units)}] {u['key']} ({u['kind']}) {len(u['segs'])}seg {time.time()-t:.1f}s{flag}", flush=True)
        (out_dir / "submission_captions.json").write_text(json.dumps(dict(cap), indent=2))
        print(f"saved {len(cap)} caption keys -> {out_dir}/submission_captions.json", flush=True)

    else:  # vqa
        data = json.loads(Path(args.vqa_file).read_text())
        total = len(data)
        data = data[i::n]
        if args.vqa_mode == "cot":
            vqa_template, vqa_max_new, vqa_parser = VQA_PROMPT_COT, 256, parse_letter_cot
        else:
            vqa_template, vqa_max_new, vqa_parser = VQA_PROMPT_PHASED, 4, parse_letter
        print(f"[vqa] shard {i}/{n}: {len(data)}/{total} entries | mode={args.vqa_mode} max_new={vqa_max_new}", flush=True)
        sub = []
        nmiss = 0
        for ei, entry in enumerate(data):
            vbase = entry["videos"][0] if entry.get("videos") else None
            vp = resolve_video(vindex, vbase) if vbase else None
            if vp is None:
                nmiss += 1
            for ph in entry.get("event_phase", []):
                num, name = norm_phase(ph["labels"][0])
                for q in ph.get("conversations", []):
                    qid = q["id"]
                    if vp is None:
                        sub.append({"id": qid, "correct": "a"})
                        continue
                    opts = {k: q.get(k, "") for k in ("a", "b", "c", "d")}
                    prompt = vqa_template.format(phase_name=name, question=q["question"], **opts)
                    resp = generate(model, processor, vp, prompt, max_new_tokens=vqa_max_new,
                                    fps=args.fps, max_pixels=args.max_pixels)
                    sub.append({"id": qid, "correct": vqa_parser(resp)})
            if (ei + 1) % 10 == 0:
                print(f"  {ei+1}/{len(data)} entries, {len(sub)} answers", flush=True)
        (out_dir / "submission_vqa.json").write_text(json.dumps(sub, indent=2))
        print(f"saved {len(sub)} vqa answers ({nmiss} entries had no video) -> {out_dir}/submission_vqa.json", flush=True)

if __name__ == "__main__":
    main()
