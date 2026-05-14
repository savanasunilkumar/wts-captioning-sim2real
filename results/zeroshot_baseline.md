# Zero-Shot Baseline — Qwen3-VL-8B-Instruct on SynWTS val

**Date:** 2026-05-14
**Setup:** 8-way parallel SLURM array, A100 80GB × 8, ~30 min wallclock

## Numbers

| Metric | Value |
|---|---|
| Scenarios | 48 (full val) |
| Caption segments | 240 |
| VQA questions | 3,531 |
| VQA accuracy | **48.63%** |
| Random baseline (4-choice) | 25% |

## Notes
- No fine-tuning, no few-shot, no SAM regions yet.
- Captions are too short (~30 words vs ~200-word GT). Expect low BLEU/CIDEr.
- VQA much stronger relative to random because the model has real scene understanding.
