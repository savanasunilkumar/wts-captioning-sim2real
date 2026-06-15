"""Caption-fusion VQA on the REAL WTS test (rank-2's 'qwen_caption_fusion' method).

For each question: look up OUR precomputed scene caption for that (scenario, phase) from
the banked v1 caption submission (already top-3 quality on the leaderboard), then answer
with [video + that caption + the question]. The model reasons over its own structured
scene description instead of answering cold. Reuses infer_realtest's generate/parse/phase
helpers + v1 LoRA. Falls back to a plain VQA prompt if no caption is found, 'a' if no video.
Output: <out>/shard_i_of_n/submission_vqa.json
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import torch
sys.path.insert(0, "/work/anujs/savana/aicity-track2/scripts")
from infer_realtest import generate, parse_letter, norm_phase, build_video_index, resolve_video
from transformers import AutoProcessor, AutoModelForImageTextToText

TEST = "/ptmp/anujs/savana/aicity-data/wts_real_test/unpacked/WTS_DATASET_PUBLIC_TEST"
CAPS_DEFAULT = "/ptmp/anujs/savana/aicity-outputs/realtest_v1_cap/merged/submission_captions.json"

FUSION_PROMPT = ("You are a traffic safety analyst. Watch this video ({phase_name} phase). "
    "A scene description is provided to help you reason; rely on the video for anything not covered.\n"
    "Scene description:\n- Pedestrian: {ped}\n- Vehicle: {veh}\n"
    "Question: {question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\nAnswer with a single letter only.")
PLAIN_PROMPT = ("You are a traffic safety analyst. Watch this video ({phase_name} phase).\n"
    "Question: {question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\nAnswer with a single letter only.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa-file", default="/ptmp/anujs/savana/aicity-data/wts_real_test/WTS_VQA_PUBLIC_TEST.json")
    ap.add_argument("--model-path", default="/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct")
    ap.add_argument("--adapter-path", default="/ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756")
    ap.add_argument("--captions", default=CAPS_DEFAULT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=360*640)
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/")); out = Path(args.out_dir)/f"shard_{i}_of_{n}"; out.mkdir(parents=True, exist_ok=True)

    # caption index keyed by (caption_unit, phase_num) -> (ped, veh)
    caps = json.loads(Path(args.captions).read_text())
    cap_idx = {}
    for key, segs in caps.items():
        for s in segs:
            cap_idx[(key, norm_phase(s["labels"][0])[0])] = (s.get("caption_pedestrian",""), s.get("caption_vehicle",""))
    cap_keys = sorted(caps.keys(), key=len, reverse=True)  # longest-prefix wins
    def cap_key_for(vname):
        stem = vname[:-4] if vname.endswith(".mp4") else vname
        for k in cap_keys:
            if stem == k or stem.startswith(k):
                return k
        return None

    vid_idx = build_video_index(Path(TEST))
    print(f"caption_units={len(caps)} cap_segments={len(cap_idx)} videos={len(vid_idx)}", flush=True)

    proc = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to("cuda").eval()
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    print("model+adapter loaded", flush=True)

    data = json.loads(Path(args.vqa_file).read_text()); data = data[i::n]
    sub = []; nfuse = nplain = nnovid = 0
    for ei, e in enumerate(data):
        vname = e["videos"][0] if e.get("videos") else None
        ck = cap_key_for(vname) if vname else None
        vpath = resolve_video(vid_idx, vname, ck) if vname else None
        for ph in e.get("event_phase", []):
            num, name = norm_phase(ph["labels"][0])
            cap = cap_idx.get((ck, num)) if ck else None
            for q in ph.get("conversations", []):
                qid = q["id"]; o = {k: q.get(k, "") for k in ("a","b","c","d")}
                if vpath is None:
                    sub.append({"id": qid, "correct": "a"}); nnovid += 1; continue
                if cap and (cap[0] or cap[1]):
                    pr = FUSION_PROMPT.format(phase_name=name, ped=cap[0], veh=cap[1], question=q["question"], **o); nfuse += 1
                else:
                    pr = PLAIN_PROMPT.format(phase_name=name, question=q["question"], **o); nplain += 1
                try:
                    ans = parse_letter(generate(model, proc, vpath, pr, 8, args.fps, args.max_pixels))
                except Exception as ex:
                    ans = "a"
                    if (nfuse+nplain) % 500 == 1: print(f"  gen-fail qid={qid}: {ex}", flush=True)
                sub.append({"id": qid, "correct": ans})
        if (ei+1) % 25 == 0: print(f"  {ei+1}/{len(data)} entries, {len(sub)} ans (fused={nfuse} plain={nplain} novid={nnovid})", flush=True)
    (out/"submission_vqa.json").write_text(json.dumps(sub, indent=2))
    print(f"shard {i}/{n}: {len(sub)} answers (fused={nfuse} plain={nplain} no-video={nnovid}) -> {out}", flush=True)

if __name__ == "__main__":
    main()
