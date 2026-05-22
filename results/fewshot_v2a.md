# Few-Shot v2a — Qwen3-VL-8B-Instruct + 1 caption exemplar per phase

**Date:** 2026-05-14
**Change from baseline:** caption prompt now includes 1 GT exemplar (from train split) per phase as a style reference. No training.

## Scores

| Subtask | Zero-shot (v1) | Few-shot (v2a) | Δ |
|---|---|---|---|
| Captioning | 6.93 | **18.85** | +11.92 |
| VQA accuracy | 48.63% | 48.63% | 0 (prompts unchanged) |
| **Combined** | 27.78 | **33.74** | **+5.96** |

## Captioning detail

| Metric | ZS ped | FS ped | ZS veh | FS veh |
|---|---|---|---|---|
| BLEU-4 | 0.008 | 0.095 | 0.009 | 0.106 |
| METEOR | 0.104 | 0.311 | 0.110 | 0.339 |
| ROUGE-L | 0.140 | 0.278 | 0.182 | 0.318 |
| CIDEr | 0.003 | 0.295 | 0.008 | 0.306 |

## Notes
- Caption length jumped ~5-6× (170 → 910 chars ped, 156 → 982 veh), matching WTS structured style.
- Exemplars drawn from train split (seed=42), phase-matched to the target segment.
- Next levers: more exemplars (v2a+), per-phase video clipping for VQA (v2b), LoRA fine-tune (v3).
