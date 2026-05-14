# Zero-Shot Baseline — Qwen3-VL-8B-Instruct on SynWTS val

**Date:** 2026-05-14
**Setup:** 8-way parallel SLURM array, A100 80GB × 8, ~30 min wallclock
**Model:** Qwen/Qwen3-VL-8B-Instruct (no fine-tune, no few-shot, no SAM regions)

## Final Scores

| Subtask | Score |
|---|---|
| Captioning (mean of BLEU-4/METEOR/ROUGE-L/CIDEr × 100) | **6.93** |
| VQA accuracy | **48.63%** |
| **Combined (mean of both)** | **27.78** |

## Captioning detail

| Metric | Pedestrian | Vehicle |
|---|---|---|
| BLEU-4 | 0.008 | 0.009 |
| METEOR | 0.104 | 0.110 |
| ROUGE-L | 0.140 | 0.182 |
| CIDEr | 0.003 | 0.008 |

## Stats
- 48 scenarios (full val split)
- 240 caption segments (5 per scenario × 48)
- 3,531 VQA questions

## Headroom
- Captions are ~30 words (GT ~200 words) → fixable with few-shot prompting + LoRA FT.
- VQA at 2× random already, but room to grow with per-phase video clipping + FT.
