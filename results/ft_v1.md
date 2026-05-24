# Fine-Tuned v1 — LoRA Qwen3-VL-8B on SynWTS

**Date:** 2026-05-23
**Adapter:** /ptmp/anujs/savana/aicity-outputs/lora_v1/v0-20260521-203741/checkpoint-756
**Config:** LoRA r=64 a=128, all-linear, freeze_vit, 3 epochs, lr 1e-4, 8039 examples

## Scores (SynWTS val — synthetic)

| Subtask | Zero-shot | Few-shot | Fine-tuned v1 |
|---|---|---|---|
| Captioning | 6.93 | 18.85 | **29.32** |
| VQA accuracy | 48.63 | 48.63 | **80.01** |
| Combined | 27.78 | 33.74 | **54.67** |

## Caption detail (ped / veh)
- BLEU: 0.218 / 0.257
- METEOR: 0.406 / 0.527
- ROUGE-L: 0.370 / 0.464
- CIDEr: 0.723 / 0.304

## Caveat
Measured on synthetic val (training distribution). Real-test will show a sim2real gap. Next: measure on real WTS + Cosmos augmentation.
