"""Zero-shot eval of Qwen3-VL on SynWTS val (or real WTS test).

Outputs two JSONs in AI City Track 2 submission format:
  - submission_captions.json
  - submission_vqa.json
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path
from collections import defaultdict

# project-root imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from scripts.wts_dataset import WTSDataset

# ---------- prompts ----------
CAP_PROMPT = """You are a traffic safety analyst. Watch this video segment depicting the {phase_name} phase of a pedestrian-vehicle traffic event.

Provide TWO captions in this exact format:
PEDESTRIAN: <pedestrian's position relative to vehicle, attention/line of sight, body action, appearance, and environment>
VEHICLE: <vehicle's position relative to pedestrian, field of view, action and speed, and environment>

Output only those two labeled lines."""

CAP_PROMPT_FEWSHOT = """You are a traffic safety analyst. Below is an example of a high-quality caption pair for the {phase_name} phase of a pedestrian-vehicle traffic event. Match this exact style, level of detail, and length (~150-200 words per caption).

EXAMPLE (style reference only — different scenario):
PEDESTRIAN: {ex_ped}
VEHICLE: {ex_veh}

Now watch THIS video segment (also a {phase_name} phase) and produce TWO captions describing what you actually observe in the video, in the same detailed style.

Output exactly in this format:
PEDESTRIAN: <detailed description>
VEHICLE: <detailed description>"""

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

VQA_PROMPT_COT_PHASED = """You are a traffic safety analyst. Watch this video segment from the {phase_name} phase of a pedestrian-vehicle traffic event.

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Think step by step about what you observe in the video (pedestrian position/orientation, vehicle action, environment), briefly evaluate each option, then end with one final line in this exact format: "Answer: X" where X is one of a, b, c, or d."""

VQA_PROMPT_COT_ENV = """You are a traffic safety analyst. Watch this pedestrian-vehicle traffic video.

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Think step by step about what you observe in the video, briefly evaluate each option, then end with one final line in this exact format: "Answer: X" where X is one of a, b, c, or d."""

# ---------- helpers ----------
def parse_caption_pair(text: str) -> tuple[str, str]:
    text = text.strip()
    pm = re.search(r"PEDESTRIAN[:\s]+(.+?)(?=\n\s*VEHICLE[:\s]|$)", text, re.DOTALL | re.IGNORECASE)
    vm = re.search(r"VEHICLE[:\s]+(.+?)$", text, re.DOTALL | re.IGNORECASE)
    ped = pm.group(1).strip() if pm else ""
    veh = vm.group(1).strip() if vm else ""
    return ped, veh

def parse_letter(text: str) -> str:
    m = re.search(r"\b([abcd])\b", text.lower())
    return m.group(1) if m else "a"

def parse_letter_cot(text: str) -> str:
    t = text.lower()
    for pat in (r"final\s+answer\s*[:=]\s*\(?([abcd])\)?",
                r"answer\s*[:=]\s*\(?([abcd])\)?",
                r"answer\s+is\s+\(?([abcd])\)?",
                r"the\s+correct\s+option\s+is\s+\(?([abcd])\)?"):
        m = re.search(pat, t)
        if m:
            return m.group(1)
    matches = re.findall(r"\b([abcd])\b", t)
    return matches[-1] if matches else "a"

def pick_video(view: str, vids: dict) -> Path | None:
    if view in vids and vids[view]:
        return vids[view][0]
    for v in vids.values():
        if v: return v[0]
    return None

def generate(model, processor, video_path: Path, prompt: str, max_new_tokens: int, fps: float, max_pixels: int) -> str:
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

# ---------- main ----------
def load_caption_exemplars(ds, num_per_phase: int = 1, seed: int = 42):
    """Load N caption exemplars per phase from a dataset split."""
    from collections import defaultdict
    by_phase = defaultdict(list)
    for sid in ds.scenarios():
        for cv in ds.load_captions(sid):
            if cv.view != "vehicle_view":
                continue
            for seg in cv.segments:
                by_phase[seg.phase_num].append((seg.pedestrian, seg.vehicle))
    rng = random.Random(seed)
    return {p: rng.sample(by_phase[p], min(num_per_phase, len(by_phase[p]))) for p in by_phase}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/ptmp/anujs/savana/aicity-data/synwts")
    p.add_argument("--split", default="val")
    p.add_argument("--model-path", default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    p.add_argument("--out-dir", default="/work/anujs/savana/aicity-track2/outputs/zeroshot")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--adapter-path", type=str, default=None, help="LoRA adapter checkpoint to merge")
    p.add_argument("--num-fewshot", type=int, default=0, help="N caption exemplars per phase from train split")
    p.add_argument("--scenario-slice", type=str, default=None, help="i/n: take every nth scenario starting at i")
    p.add_argument("--max-pixels", type=int, default=360*640)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--skip-captions", action="store_true")
    p.add_argument("--skip-vqa", action="store_true")
    p.add_argument("--vqa-mode", choices=["simple", "cot"], default="simple",
                   help="simple = 4-token letter; cot = chain-of-thought, 256-token")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_path}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        print(f"Merged LoRA adapter: {args.adapter_path}")
    print(f"Loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")

    exemplars = {}
    if args.num_fewshot > 0:
        train_ds = WTSDataset(args.data_root, split="train")
        exemplars = load_caption_exemplars(train_ds, num_per_phase=args.num_fewshot)
        n_ex = sum(len(v) for v in exemplars.values())
        print(f"Loaded {n_ex} caption exemplars across {len(exemplars)} phases (from train)")

    ds = WTSDataset(args.data_root, split=args.split)
    scenarios = ds.scenarios()
    if args.max_scenarios: scenarios = scenarios[:args.max_scenarios]
    if args.scenario_slice:
        i, n = map(int, args.scenario_slice.split("/"))
        scenarios = scenarios[i::n]
        out_dir = out_dir / f"shard_{i}_of_{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Slice {i}/{n}: processing {len(scenarios)} scenarios -> {out_dir}")
    else:
        print(f"Processing {len(scenarios)} scenarios from {args.data_root}/{args.split}")

    cap_sub = defaultdict(list)
    vqa_sub, vqa_gt = [], []

    for si, sid in enumerate(scenarios):
        print(f"\n[{si+1}/{len(scenarios)}] {sid}")
        caps = ds.load_captions(sid)
        vqas = ds.load_vqa(sid)
        vids = ds.get_video_paths(sid)

        # captions (use vehicle_view as canonical)
        if not args.skip_captions:
            cv = next((c for c in caps if c.view == "vehicle_view"), caps[0] if caps else None)
            video = pick_video("vehicle_view", vids)
            if cv and video:
                for seg in cv.segments:
                    t = time.time()
                    if exemplars and seg.phase_num in exemplars:
                        ex_ped, ex_veh = exemplars[seg.phase_num][0]
                        cap_prompt = CAP_PROMPT_FEWSHOT.format(
                            phase_name=seg.phase_name, ex_ped=ex_ped, ex_veh=ex_veh)
                        max_tok = 600
                    else:
                        cap_prompt = CAP_PROMPT.format(phase_name=seg.phase_name)
                        max_tok = 320
                    resp = generate(model, processor, video, cap_prompt,
                                    max_new_tokens=max_tok, fps=args.fps, max_pixels=args.max_pixels)
                    ped, veh = parse_caption_pair(resp)
                    cap_sub[sid].append({
                        "labels": [seg.phase_num],
                        "caption_pedestrian": ped,
                        "caption_vehicle": veh,
                        "start_time": seg.start_time,
                        "end_time": seg.end_time,
                    })
                    print(f"  cap[{seg.phase_num}] {time.time()-t:.1f}s ped:{len(ped)} veh:{len(veh)}")

        # VQA
        if not args.skip_vqa:
            cot = args.vqa_mode == "cot"
            vqa_max_new = 256 if cot else 4
            vqa_parser = parse_letter_cot if cot else parse_letter
            for q in vqas:
                vfor = "vehicle_view" if q.view == "vehicle_view" else "overhead_view"
                video = pick_video(vfor, vids)
                if not video: continue
                opts = {k: q.options.get(k, "") for k in ("a", "b", "c", "d")}
                if q.view == "environment":
                    tmpl = VQA_PROMPT_COT_ENV if cot else VQA_PROMPT_ENV
                else:
                    tmpl = VQA_PROMPT_COT_PHASED if cot else VQA_PROMPT_PHASED
                prompt = tmpl.format(phase_name=q.phase_name or "", question=q.question, **opts)
                resp = generate(model, processor, video, prompt, max_new_tokens=vqa_max_new, fps=args.fps, max_pixels=args.max_pixels)
                letter = vqa_parser(resp)
                qid = f"{sid}_{q.view}_{q.file_id}_{q.question_idx}"
                vqa_sub.append({"id": qid, "correct": letter})
                if q.correct is not None: vqa_gt.append({"id": qid, "correct": q.correct})
            print(f"  vqa done: {len(vqas)} questions")

    # save
    (out_dir / "submission_captions.json").write_text(json.dumps(dict(cap_sub), indent=2))
    (out_dir / "submission_vqa.json").write_text(json.dumps(vqa_sub, indent=2))
    if vqa_gt:
        (out_dir / "vqa_gt.json").write_text(json.dumps(vqa_gt, indent=2))
        n_correct = sum(1 for pred, gt in zip(vqa_sub, vqa_gt) if pred["correct"] == gt["correct"])
        print(f"\nVQA acc (local): {n_correct}/{len(vqa_gt)} = {n_correct/len(vqa_gt)*100:.1f}%")
    print(f"\nSaved: {out_dir}")

if __name__ == "__main__":
    main()
