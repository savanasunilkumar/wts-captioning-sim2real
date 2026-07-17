# Track 2 (Sim2Real) — Technical Report Source Document
### AI City Challenge 2026 · Team "Cyclone Intelligence" (ID 5) · repo: `savanasunilkumar/wts-captioning-sim2real`

This is the complete factual record for writing the workshop report. Every number here comes from the eval-server leaderboard, a training run's `args.json`/logs, or a script in the repo.

---

## 1. Task, data, and metric

**Challenge.** AI City Challenge 2026, Track 2: *Transportation Safety Understanding and Captioning (Sim2Real)*. Models may be fine-tuned **only on synthetic data** (SynWTS, HF `mlcglab/synwts` — a digital twin of the real WTS dataset with matched camera geometry). Evaluation is on **real** video: the WTS public test set plus external BDD_PC_5K clips. Open-weight base models permitted.

**Data.**
| Split | Content |
|---|---|
| SynWTS train | 96 scenarios (multi-view: several `overhead_view` cameras + one `vehicle_view` dashcam per scenario), 5 phases each |
| SynWTS val | 48 scenarios, same structure |
| Real test | 1,040 VQA entries / **12,316 questions**; 459 caption units (84 WTS scenarios + 375 BDD clips) / 2,289 caption segments |

Each scenario is segmented into 5 phases: `pre-recognition(0), recognition(1), judgment(2), action(3), avoidance(4)`. Captions are a pedestrian paragraph + vehicle paragraph per phase. VQA is 4-way multiple choice; the question set uses **14 unique templates** (identical between SynWTS and the real test — verified by exact string match, 12,316/12,316).

**Test composition (own analysis; a paper figure).** By referenced video: **BDD ego-dashcam 8,761 (71%)**, WTS overhead 2,601 (21%), WTS vehicle_view 954 (8%). Vehicle-centric templates dominate: "what is vehicle's field of view?" (2,154), "what is the action taken by vehicle?" (2,149), "what is the pedestrian's awareness regarding vehicle?" (1,465).

**Metric.** `S2 = ½·(caption_combined + VQA_accuracy)`, with `caption_combined = ¼·(100·BLEU-4 + 100·METEOR + 100·ROUGE-L + 10·CIDEr)`, averaged over pedestrian/vehicle captions and internal/external splits. Leaderboard scores a fixed **50% subset** of the test set (final ranking on the full set).

## 2. Final system

**Model.** Qwen3-VL-Instruct + LoRA. Two configurations were submitted; the 32B is the best system.

| | v1 (8B) | v8 (32B, best) |
|---|---|---|
| Base | Qwen3-VL-8B-Instruct | Qwen3-VL-32B-Instruct (dense) |
| LoRA | r=64, α=128, all-linear, ViT frozen | same |
| Data | SynWTS train: 8,039 ex. (940 caption + 7,099 VQA) | train+val: 12,045 ex. (1,415 + 10,630) |
| Epochs / lr | 3 / 1e-4 (warmup 0.03) | 2 / 1e-4 (warmup 0.03) |
| Batch | 1×32 grad-accum (global 32), bf16, grad ckpt | 1×8×4 GPUs (global 32), DeepSpeed ZeRO-3 |
| Hardware / time | 1×A100-80GB, ~3.5 h | 4×A100-80GB, 5 h 41 m (754 steps, 27 s/step) |
| Video input | 1 fps, ≤16 frames, `VIDEO_MAX_PIXELS=100352` | same |

**Training format.** One SFT example per (scenario-phase, target): user message = `<video>` + prompt; assistant = caption pair or the answer letter. Caption examples use the scenario's `vehicle_view` video; VQA examples predominantly overhead views.

**Inference (identical for all submissions).** *Stateless*: every question is an independent greedy `generate()` (max 4 new tokens), letter parsed by regex. Captions: one generation per phase, `PEDESTRIAN:`/`VEHICLE:` split on the labeled colon. Sharded 4×A100; 32B ≈ 2 s/question (~1 h 50 m for full VQA).

**Exact prompts (verbatim):**
```
CAPTION:
You are a traffic safety analyst. Watch this video segment depicting the {phase_name} phase of a pedestrian-vehicle traffic event.

Provide TWO captions in this exact format:
PEDESTRIAN: <pedestrian's position relative to vehicle, attention/line of sight, body action, appearance, and environment>
VEHICLE: <vehicle's position relative to pedestrian, field of view, action and speed, and environment>

Output only those two labeled lines.

VQA:
You are a traffic safety analyst. Watch this video ({phase_name} phase).

Question: {question}
(a) {a}
(b) {b}
(c) {c}
(d) {d}

Answer with a single letter only.
```

## 3. Main results

**Synthetic validation (SynWTS val, 48 scenarios, official scorer run locally):**
| | Captions | VQA | Combined |
|---|---|---|---|
| Zero-shot 8B | 6.93 | 48.63 | 27.78 |
| Few-shot 8B | 18.85 | 48.63 | 33.74 |
| v1 LoRA 8B | **29.32** | **80.01** | **54.67** |

**Real test (leaderboard, 50% subset):**
| System | S2 | VQA Acc | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|
| v1 (8B) | 44.4433 | 60.3329 | 0.2393 | 0.4003 | 0.4378 | 0.6474 |
| **v8 (32B)** | **44.8110** | **61.1855** | 0.2393 | 0.4066 | 0.4396 | 0.5191 |

**The headline transfer asymmetry:** captions 29.32 → 28.55 (**−0.8**); VQA 80.01 → 60.33 (**−19.7**). Captions cross the sim2real gap essentially intact; VQA loses ~20 points.

**Leaderboard at close (General board):**
| Rank | Model | S2 | Acc |
|---|---|---|---|
| 1 | VL-JEPA-based | 60.43 | 87.37 |
| 2–3 | Qwen-based (two identical entries) | 57.45 | 84.73 |
| 4 | TrafficInternVL (official baseline) | 47.16 | 66.71 |
| 5 | **ours (v8)** | **44.81** | **61.19** |

Our caption metrics are at top-3 level (ROUGE-L 0.4378 vs rank-3's 0.4284; CIDEr 0.6474 vs 0.6590); the entire deficit is VQA. Context to state neutrally: published 2025 systems trained on **real** WTS data report VQA in the low-to-mid 80s (e.g., 81.5% for a Qwen2.5-VL-7B LoRA); the General board is not rule-reviewed.

## 4. Ablation ledger (all leaderboard-scored on the real test)

| # | Intervention | VQA Acc | Δ vs reference | Conclusion |
|---|---|---|---|---|
| 1 | Base model, no SFT | 29.76 | −30.6 vs v1 | SFT is essential |
| 2 | v1: LoRA SFT (8B) | 60.33 | reference | the 5-week baseline |
| 3 | Drawn-box grounding (green ped / blue veh + union crop, grounded retrain) | 57.21 | **−3.12** | visual prompts hurt |
| 4 | Grounding, corrected (exact-phase, best-view, v1 fallback; 452 answer changes) | 59.44 | **−0.89** | hurts even when clean |
| 5 | Caption-fusion (own caption injected into VQA prompt) | 58.26 | **−2.07** | caption errors become priors |
| 6 | Scale: 32B dense, +50% data (v8) | 61.19 | **+0.85** | capacity is not the bottleneck |
| 7 | Photometric domain randomization (Real-ESRGAN-style video degradation, 80/20 mix, 32B retrain) | 60.50 | **−0.69** | gap is not pixel statistics |
| 8 | Cross-model majority vote (v1+v8+v9) | 60.62 | −0.57 | weaker models outvote the stronger |
| 9 | LoRA checkpoint soup (avg of steps 600/750/754) | 60.90 | −0.28 | checkpoints too converged (110 answers changed) |
| 10 | Two-stage SFT (caption→merge→VQA) **+ ego-view-aligned data** (79% dashcam, mirroring test) | 60.90 | ≈0 | view alignment alone doesn't recover the gap |

Measured off-leaderboard: chain-of-thought (worse than direct on val A/B); high-res / more frames / self-consistency voting (no gain); discriminative option-text likelihood scoring under the letter-SFT model (**13–16% — below chance**; letter-logit variant ≈ greedy, 76–78 vs 80.01 val); Qwen3-VL-30B-A3B sparse (no val gain over 8B). Caption-side test-time study with a local official-metric scorer: perception-focused prompt / overhead view / 2.6×-resolution move vehicle captions 39→41 but pedestrian captions are frozen at ~27 (BLEU/ROUGE ~0.22/0.36) under every variant.

Alternative base models: GLM-4.1V-9B and InternVL3.5-38B could not be trained in the pinned stack (ms-swift 4.2.1 + transformers 5.8.1 video-template failures); Molmo2-8B required three remote-code compatibility patches and was abandoned at the deadline. Useful as an infrastructure-friction observation only.

## 5. Analyses (the paper's discussion material)

**5.1 Error taxonomy (synth val, per question type).** Failures concentrate in *pedestrian body orientation* (~40% accuracy) and *vehicle-relative position* (~36%); most other categories score 75–100%. The weak axis is spatial-relational reasoning; it is unchanged by scale.

**5.2 Why captions transfer but VQA doesn't (proposed explanation).** WTS reference captions are strongly templated — demographics, clothing, weather, road-surface phrasing recur nearly verbatim — so n-gram metrics reward surface structure the model learns from synthetic data and reproduces on real video. VQA cannot be answered from templates: it requires grounded perception of real event dynamics, where models trained on scripted synthetic behavior fail. Consistent with this, interventions on pixels (degradation), viewpoint (ego-alignment), capacity (32B), and inference scaffolding all failed: the residual gap lies in **behavioral realism** of the synthetic data.

**5.3 Scaffolds systematically hurt.** Every added inference structure — drawn boxes, injected captions, chain-of-thought, voting, likelihood scoring — *reduced* real-test accuracy relative to plain stateless prompting (rows 3–5, 8; plus off-board CoT/voting). Interpretation: narrow-format letter-SFT sharpens a direct video→answer mapping; off-distribution prompt or visual additions cost more than their signal adds. Independently corroborated by STER-VLM's published ablation (visual prompts hurt their WTS captioner: 30.66 vs 31.85 text-only) and their stateless-vs-chained result (83.1 vs 66.5).

**5.4 Test composition.** 79% of test questions reference first-person video where the "vehicle" is the invisible camera-car, while ~half of all questions are vehicle-centric; naive synthetic training is dominated by third-person overhead views. Correcting this mismatch in training (row 10) did *not* close the gap — a notable negative result that strengthens the behavioral-gap conclusion.

**5.5 Limitations.** Leaderboard scores are on a fixed 50% subset (granularity ~0.016% Acc per question; small deltas near ±0.3 are at the edge of resolution). No test ground truth → no per-category error analysis on real data. Most experiments share one base-model family (Qwen3-VL).

## 6. Suggested paper skeleton

*Title direction:* "When Only Captions Cross: A Systematic Study of Synthetic-to-Real Transfer for Traffic-Safety Video VLMs."

1. **Intro** — sim2real for safety-critical video understanding; the SynWTS→WTS setting.
2. **System** — §2 content (model, data, prompts, stateless inference).
3. **Results** — §3 tables.
4. **Ablations** — §4 table + one paragraph per row group (grounding, fusion, scale, domain randomization, ensembling, view alignment).
5. **Analysis** — §5.1–5.4.
6. **Discussion** — what would cross the gap: real-data fine-tuning (rule-dependent; matches the 80s-Acc cluster), embedding-predictive architectures (VL-JEPA-based rank 1 at 87.4), and *behaviorally* realistic synthetic generation rather than photorealistic rendering.

**Figures/tables:** T1 main results (§3); T2 ablation ledger (§4); F1 transfer asymmetry bars (val vs real, captions vs VQA); F2 test view-composition; F3 per-question-type accuracy; optional F4 S2 across submissions (timeline).

## 7. Reproducibility appendix

- **Environment:** ms-swift 4.2.1, transformers 5.8.1, torch 2.12.0+cu130 (A100-only build), DeepSpeed, decord, qwen_vl_utils. ISU Nova HPC, SLURM, 4×A100-SXM4-80GB.
- **Wall-times:** 8B train ~3.5 h (1 GPU); 32B train 5 h 41 m (4 GPU ZeRO-3); full VQA inference 1 h 50 m (4 shards); full caption inference ~3 h; official caption metric runs locally in seconds.
- **Code map (`scripts/`):** `build_sft_data.py` (SynWTS→JSONL), `infer_realtest.py` (submission inference; contains the exact prompts), `validate_submission.py` (format checker mirroring the official reader — 0 rejected submissions in 15), `analyze_vqa.py` (error taxonomy), `build_grounded_data.py`/`infer_grounded_hybrid.py` (grounding), `infer_fusion.py`, `vqa_likelihood.py` (discriminative scoring), `degrade_videos.py` (video degradation: γ/brightness/contrast/saturation → Gaussian blur σ0.4–2.2 → down/up-scale 0.45–0.85 → Gaussian noise σ2–14 → JPEG q30–70 → H.264 CRF 28–34), `val_caption_infer.py` (caption lab). `slurm/` holds every job file including the ZeRO-3 32B config.
- **Checkpoints:** v1 8B `lora_v1/v0-20260521-203741/checkpoint-756` (backup: `/work/anujs/savana/model_backup/`); v8 32B `lora_32b/v0-20260706-125809/checkpoint-754`; variants under `/ptmp/anujs/savana/aicity-outputs/`. **`/ptmp` auto-purges after 60 idle days — archive to `/work` before writing.**
- **Submission artifacts:** each scored pair under `aicity-outputs/*/merged/`; best set in `aicity-outputs/FINAL/`. Official caption scorer: `wts-dataset/evaluation/eval-metrics-AIC-Track2/metrics_all.py` (requires a GT root containing `annotations/`; `--internal` for WTS-only).

*Compliance: all fine-tuning data is synthetic (SynWTS train+val). No real frames or labels used in training; no test-set self-training. Repo history is single-author.*
