# WTS Captioning Sim2Real — AI City Challenge 2026 Track 2

Transportation Safety Understanding and Captioning (Sim2Real).

- **Backbone:** Qwen3-VL-8B-Instruct (LoRA fine-tuned)
- **Preprocessor:** SAM 3.1 (frozen, region grounding)
- **Augmentation:** Cosmos Transfer 2.5 (frozen, offline sim→real)
- **Inference ensemble:** Cosmos-Reason2-8B (frozen, optional)
- **Cluster:** ISU Nova HPC (A100 80GB)

## Layout

| Path | Purpose |
|---|---|
| `scripts/` | training, eval, preprocessing |
| `configs/` | LoRA configs, training hyperparams |
| `notebooks/` | exploration |
| `slurm/` | SLURM batch templates |
| `data/` → `/ptmp/anujs/savana/aicity-data/` | dataset (scratch) |
| `outputs/` → `/ptmp/anujs/savana/aicity-outputs/` | checkpoints (scratch) |
