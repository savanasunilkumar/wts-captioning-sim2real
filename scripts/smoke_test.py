"""Smoke test: load Qwen3-VL-8B-Instruct on GPU explicitly, run one inference."""
import json, time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "/ptmp/anujs/savana/aicity-data/weights/Qwen3-VL-8B-Instruct"
OUT_PATH = Path("/work/anujs/savana/aicity-track2/outputs/smoke_test.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  CUDA available: {torch.cuda.is_available()}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("[1/4] Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("[2/4] Loading model (bf16) → moving to GPU...")
t = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
).to(device).eval()
print(f"      Loaded in {time.time()-t:.1f}s")
print(f"      model.device = {model.device}")
print(f"      param devices: {set(str(p.device) for p in model.parameters())}")
print(f"      VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")

print("[3/4] Building prompt...")
img = Image.new("RGB", (224, 224), color=(64, 192, 64))
messages = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": "What is the dominant color of this image? Answer in one word."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)

print("[4/4] Generating...")
t = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
elapsed = time.time() - t
resp = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

print(f"\n=== RESPONSE ===\n{resp}\n================")
print(f"Gen: {elapsed:.2f}s  peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({
    "model_path": MODEL_PATH,
    "device": str(model.device),
    "response": resp,
    "gen_time_s": elapsed,
    "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
}, indent=2))
print(f"Saved → {OUT_PATH}")
