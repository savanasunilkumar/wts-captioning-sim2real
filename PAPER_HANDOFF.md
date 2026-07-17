# Sim2Real Traffic-Safety VLM — Complete Technical Report Source
### AI City Challenge 2026, Track 2 · Team "Cyclone Intelligence" (ID 5)
**Repo:** `github.com/savanasunilkumar/wts-captioning-sim2real` · **Competition window:** May 21 – July 11, 2026
**Purpose of this document:** everything a co-author needs to write the workshop report without access to the experimenter. Every number is traceable to the eval server, a run's `args.json`/logs, or a repo script.

---

# PART I — BACKGROUND

## 1. The challenge and datasets

**AI City Challenge** is an annual CVPR-workshop competition (10th edition in 2026; the 9th-edition summary is arXiv:2508.13564). **Track 2 (2026): "Transportation Safety Understanding and Captioning (Sim2Real)"** — a new sim2real variant of the traffic-safety track run in 2024–2025.

- **WTS (Woven Traffic Safety) dataset** — real, pedestrian-centric traffic-incident videos; multi-view (several fixed *overhead* CCTV cameras + one *vehicle_view* dashcam per scenario); staged pedestrian–vehicle interactions; dense per-phase captions and multiple-choice VQA. GitHub: `woven-visionai/wts-dataset` (dataset paper: "WTS: A Pedestrian-Centric Traffic Video Dataset…", ECCV 2024 — verify arXiv ID when citing).
- **BDD_PC_5K** — pedestrian-centric clips from the BDD100K corpus (single ego-dashcam, US streets), annotated in WTS format; used as the "external" evaluation split.
- **SynWTS** (new for 2026; HF `mlcglab/synwts`) — a **synthetic digital twin** of WTS: rendered replicas of the same locations/camera geometries with scripted pedestrian–vehicle events. 2026 rule: **models may be fine-tuned on SynWTS only** (both its train and val splits are synthetic and legal). Open-weight pretrained backbones allowed; API models disallowed.

**Event structure.** Every scenario is annotated in 5 phases, each with its own captions and questions:
`0 pre-recognition → 1 recognition → 2 judgment → 3 action → 4 avoidance`.

**Evaluation metric.**
```
caption_combined = ¼ (100·BLEU-4 + 100·METEOR + 100·ROUGE-L + 10·CIDEr)   [averaged ped/veh × internal/external]
S2 = ½ (caption_combined + VQA_accuracy)
```
The public leaderboard scores a fixed **50% subset** of the test set; final ranking uses the full set. Two files per submission: captions JSON (per-scenario segments, `caption_pedestrian` + `caption_vehicle` required per phase) and VQA JSON `[{id, correct}]`. Missing answers score zero.

## 2. Related work the paper should cite

| Work | Relevance | Ref |
|---|---|---|
| 9th AI City Challenge summary | track history, 2025 results | arXiv:2508.13564 |
| STER-VLM (2025 Track-2 team) | two-stage training; **their ablation shows drawn visual prompts hurt on WTS** (31.85 text-only vs 30.66 with boxes); stateless vs chained VQA 83.08 vs 66.50; ~81% VQA with real-data LoRA at 7B | arXiv:2508.13470 |
| TrafficInternVL | official 2026 baseline (S2 47.16 / Acc 66.71); keyframe global+local views, role-aware prompts | 2025 AI City workshop |
| Dual-model task-specific framework | caption/VQA task separation (2025, S2 45.76) | arXiv:2510.11907 |
| VL-JEPA | rank-1's architecture class: predicts continuous text embeddings; discriminative VQA by nearest-candidate-embedding; no public weights (2B-sample pretraining) | arXiv:2512.10942 |
| Real-ESRGAN | the second-order degradation recipe we adapted for video domain randomization | arXiv:2107.10833 |
| FDA (Fourier Domain Adaptation) | input-level sim2real alternative; requires real target frames (rule-relevant discussion) | arXiv:2004.05498 |
| Qwen3-VL | base models (8B/32B dense, 30B-A3B sparse) | `QwenLM/Qwen3-VL` |
| ms-swift | training framework | `modelscope/ms-swift` |

# PART II — TASK DATA IN DETAIL

## 3. The complete VQA question inventory (real test, 12,316 questions)

Exactly **14 templates**, all present verbatim in SynWTS training data (100% template overlap — phrasing shift is ruled out as an error source):

| # questions | Template |
|---|---|
| 2,154 | what is vehicle's field of view? |
| 2,149 | what is the action taken by vehicle? |
| 1,465 | what is the pedestrian's awareness regarding vehicle? |
| 815 | what is the position of the pedestrian relative to the vehicle? |
| 813 | what is the orientation of the pedestrian's body? |
| 798 | what is the position of the vehicle relative to the pedestrian? |
| 786 | what is the pedestrian's line of sight? |
| 739 | what is the pedestrian's visual status? |
| 618 | what is the pedestrian's action? |
| 613 | what is the pedestrian's direction of travel? |
| 512 | what is pedestrian's speed? |
| 294 | what is relative distance of pedestrian from vehicle? |
| 282 | what is the fine-grained action taken by the pedestrian? |
| 278 | what is relative distance of vehicle from pedestrian? |

**Worked VQA example (verbatim from the test set):**
> video `20231013_101845_normal_192.168.0.13_4_event_2.mp4`, phase *avoidance*
> **Q:** What is the orientation of the pedestrian's body?
> (a) Opposite direction to the vehicle · (b) Perpendicular to the vehicle and to the right · (c) Diagonally to the left, in the opposite direction to the vehicle · (d) Perpendicular to the vehicle and to the left

Note the option style: fine-grained **vehicle-relative spatial relations** — the category our error analysis identifies as hardest.

## 4. Test composition (own analysis; paper figure)

By the video each question references: **BDD ego-dashcam 8,761 (71.1%) · WTS overhead 2,601 (21.1%) · WTS vehicle_view 954 (7.7%)** → **79% of the test is first-person video in which "the vehicle" is the invisible camera-car**, and the two most frequent templates (35% of all questions) ask about that vehicle's field-of-view and action. Naive SynWTS training data is ~74% overhead (third-person). This mismatch motivated experiment E10.

## 5. Training data built from SynWTS

| File | Composition | Used by |
|---|---|---|
| `train.jsonl` | 8,039 examples = 940 caption + 7,099 VQA (96 train scenarios) | v1 |
| `train_plus_val.jsonl` | 12,045 = 1,415 caption + 10,630 VQA (train+val, both synthetic) | v8, v9 |
| ego-remapped variant | same 12,045 with VQA videos remapped to 79% vehicle_view / 21% overhead | E10 (v12) |
| `train_grounded.jsonl` | 5,741 image-VQA examples with rendered boxes (below, E3) | grounded model |

Answer-letter prior in training VQA: a 2,963 / b 3,007 / c 2,681 / d 1,979 — mildly imbalanced, mirrored by model output distributions (e.g., v8 output: a 3,997 / b 3,677 / c 2,681 / d 1,961).

**SFT example format** (ms-swift "messages" JSONL):
```json
{"messages":[{"role":"user","content":"<video>You are a traffic safety analyst. ...prompt..."},
             {"role":"assistant","content":"b"}],
 "videos":["/path/scenario/overhead_view/....mp4"]}
```

# PART III — SYSTEM

## 6. Final system configuration (verbatim from `args.json`)

| | v1 (8B) | v8 (32B — best submission) |
|---|---|---|
| Base model | Qwen3-VL-8B-Instruct | Qwen3-VL-32B-Instruct (dense) |
| Tuning | LoRA r=64 α=128, target `all-linear`, **ViT frozen**, bf16 | same |
| Dataset | train.jsonl (8,039) | train_plus_val.jsonl (12,045) |
| Epochs | 3 | 2 |
| LR / warmup | 1e-4 / 0.03 | 1e-4 / 0.03 |
| Batch | per-device 1 × grad-accum 32 (global 32), 1 GPU | per-device 1 × accum 8 × 4 GPUs (global 32), DeepSpeed ZeRO-3 |
| Grad checkpointing | on | on |
| Video input | fps 1.0, ≤16 frames, `VIDEO_MAX_PIXELS=100352` (~100k px/frame) | same |
| Steps / wall-time | 756 steps ≈ 3.5 h (1×A100-80GB) | 754 steps, 5 h 41 m (4×A100-80GB, 27 s/step) |
| Final train loss | ~0.9 (plateau) | 0.35 (train avg 0.49, token-acc 0.88) |

**Inference protocol (all submissions).** *Stateless*: each question is one independent greedy `generate()` with `max_new_tokens=4`; answer letter extracted by regex `\b([abcd])\b` (fallback "a"; parse failures ≈ 0). Captions: one 512-token generation per phase; split on `VEHICLE:` (colon required to avoid mid-caption false splits). Sharded across 4×A100 (one full-precision bf16 model per GPU; the 63 GB 32B fits one 80 GB card at this pixel budget). Throughput: 32B ≈ 2 s/question → full VQA in 1 h 50 m; captions ≈ 3 h.

**Exact prompts** — see §2 of the repo's `scripts/infer_realtest.py`; verbatim:
```
CAPTION:  "You are a traffic safety analyst. Watch this video segment depicting the {phase_name}
phase of a pedestrian-vehicle traffic event.\n\nProvide TWO captions in this exact format:\n
PEDESTRIAN: <pedestrian's position relative to vehicle, attention/line of sight, body action,
appearance, and environment>\nVEHICLE: <vehicle's position relative to pedestrian, field of view,
action and speed, and environment>\n\nOutput only those two labeled lines."

VQA:  "You are a traffic safety analyst. Watch this video ({phase_name} phase).\n\n
Question: {question}\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\nAnswer with a single letter only."
```

## 7. Worked caption example (model vs. reference, same scene & phase)

SynWTS-val scenario `20230707_18_CY23_T1`, phase 4 (avoidance):

> **Ours (v1):** "The pedestrian, a male in his 30s, stood perpendicular to the vehicle and to the left. He was positioned directly in front of the vehicle, at a close distance. His line of sight was focused on the vehicle, and he closely watched it. The pedestrian's direction of travel was in front, and he was slowly moving forward. He was aware of the vehicle and noticed its presence…"
> **Reference:** "The pedestrian is a male in his 30s, standing at approximately 170 cm tall. He is wearing a white jacket and blue slacks, with a white hat on his head. The weather is cloudy and the brightness of the surroundings is bright. The road surface is dry and level, made of asphalt. The traffic volume is light on this residential road… The pedestrian is directly in front of a ve…"

> **Ours (veh):** "The vehicle was positioned on the left side of the pedestrian and was close in distance… The vehicle took emergency action and avoided the pedestrian by applying the brakes."
> **Reference (veh):** "The vehicle is situated in front of the pedestrian, at a close distance… The vehicle takes emergency braking action, leading to a speed of 0 km/h…"

This example is representative of the central caption phenomenon: **the templated attribute clauses (demographics, weather, road) and event clauses (emergency braking) match closely; the variable specifics (clothing colors, exact orientation) frequently do not** — yet n-gram metrics remain high because the shared template dominates.

# PART IV — RESULTS

## 8. Main results

**Synthetic validation (SynWTS val, official scorer run locally):**
| | Captions | VQA | Combined |
|---|---|---|---|
| Zero-shot 8B | 6.93 | 48.63 | 27.78 |
| Few-shot 8B (2–3 exemplars) | 18.85 | 48.63 | 33.74 |
| **v1 LoRA 8B** | **29.32** | **80.01** | **54.67** |

v1 caption detail (ped/veh): BLEU .218/.257 · METEOR .406/.527 · ROUGE-L .370/.464 · CIDEr .723/.304.

**Real test (leaderboard, 50% subset) — every scored submission:**
| Date | System | S2 | VQA Acc | Notes |
|---|---|---|---|---|
| 6/02 | v1: 8B LoRA | 44.4433 | 60.3329 | baseline; capt .2393/.4003/.4378/.6474 |
| 6/03 | base 8B, no SFT | 29.1570 | 29.7605 | SFT is essential |
| 6/05 | early caption variant | 42.8790 | 59.8457 | weaker captions |
| 6/06 | v1 rerun | 44.4027 | 60.2517 | reproducibility ±0.1 |
| 6/14 | E3 drawn-box grounding | 42.8801 | 57.2067 | −3.12 Acc |
| 6/15 | E5 caption-fusion | 43.4079 | 58.2623 | −2.07 |
| 7/06 | E4 grounding corrected (hybrid) | 43.9967 | 59.4397 | −0.89 |
| 7/07 | **E6 v8: 32B dense** | **44.8110** | **61.1855** | **best; +0.85** |
| 7/07 | E7 v9: degradation aug | 44.5245 | 60.4953 | −0.69 |
| 7/07 | E8 v10: majority vote | 44.5854 | 60.6171 | −0.57 |
| 7/07 | E9 v11: checkpoint soup | 44.7275 | 60.9013 | −0.28 |
| 7/10 | E10 v12: two-stage + ego-view | 44.7275 | 60.9013 | ≈0 (see §12 note) |

**Transfer asymmetry (the headline):** captions 29.32→28.55 (**−0.8**) vs VQA 80.01→60.33 (**−19.7**).

**Leaderboard at close (General/public board):** VL-JEPA 60.43 S2 / 87.37 Acc (rank 1); two byte-identical Qwen entries 57.45 / 84.73 (ranks 2–3); official baseline TrafficInternVL 47.16 / 66.71; ours 44.81 / 61.19 (rank 5). Our caption metrics equal the top-3 cluster (ROUGE-L 0.4378 vs 0.4284; CIDEr 0.6474 vs 0.6590; BLEU 0.2393 vs 0.2442); the deficit is 100% VQA. Context to state neutrally: 84.7 Acc matches published *real-data-trained* systems (STER-VLM reports 81.5 at 7B on real WTS), and the General board is not rule-reviewed; the prize track is.

# PART V — THE TEN EXPERIMENTS (full detail)

Each: motivation → exact setup → result → interpretation. All "Acc" numbers are leaderboard-scored on the real test unless marked (val).

**E1. Supervised fine-tuning (v1).** Base 8B answers at 29.8 real / 48.6 (val); LoRA SFT lifts to 60.33 real / 80.01 (val). Establishes both the system and the ~19.7-point sim2real VQA drop. *(Setup in §6.)*

**E2. Prompting variants (val A/Bs, no submissions).** (a) Chain-of-thought ("think step by step… end with Answer: X", 256 tokens): below direct answering. (b) Few-shot exemplars: no VQA change. (c) Self-consistency voting, higher resolution (2.6×), more frames (2 fps): no gains. (d) Stateless vs chained context: our loop is stateless by construction; STER-VLM independently measured 83.08 stateless vs 66.50 chained on a LoRA InternVL3-38B. **Direct, stateless, single-letter prompting is optimal.**

**E3. Drawn-box spatial grounding.** Motivation: 2025 winners' recipe (TrafficInternVL-style); our error analysis shows spatial categories weakest. Setup: for each (scenario, phase), select best view (max pedestrian-box area among views with both ped+veh boxes → overhead), draw GREEN pedestrian / BLUE vehicle rectangles (width 5) on the annotated keyframe + a union crop (×1.5 context); train a grounded LoRA on 5,741 two-image examples with a role-aware prompt ("the pedestrian is outlined in a GREEN box…"); infer with same rendering on test bboxes. Result: **57.21 (−3.12)**. Follow-up E4 isolates why.

**E4. Grounding, corrected.** The v6 inference had two faults (nearest-phase box fallback on 54% of questions; arbitrary camera instead of training's best-view). Corrected run: exact-phase boxes only, training-faithful best-view, and v1's answer wherever grounding wasn't cleanly available (80% of questions; only 452 answers differed from v1). Result: **59.44** — the grounded model *loses the disagreements* even under clean conditions. Independently corroborated: STER-VLM's own ablation (visual prompts 30.66 vs text-only 31.85 on WTS captions). **Synthetic-box grounding does not transfer; visual prompt injection hurts.**

**E5. Caption-fusion.** Motivation: rank-2's method name ("qwen_caption_fusion"); our captions are top-3-grade. Setup: inject our own scene caption (pedestrian+vehicle paragraphs) into the VQA prompt as "Scene description", then ask the question with video. 100% of questions fused. Result: **58.26 (−2.07)**. Interpretation: captions are accurate on templated attributes but weak on exact spatial attributes — precisely the attributes hard questions ask — so fusion injects wrong priors.

**E6. Model scale (v8).** 8B→32B dense (4× parameters; the 30B-A3B *sparse* model with 3B active params had shown no val gain and is not a real scale test), +50% data (val split added). Training: §6. Synthetic fit improves dramatically (loss 0.9→0.35; token-acc 88%); real test: **61.19 (+0.85)**. **Capacity is not the bottleneck; the gap survives scale.**

**E7. Photometric domain randomization (v9).** Motivation: classic sim2real closer; the one train-time lever untried; recipe verified from Real-ESRGAN (arXiv:2107.10833). Setup: per-video sampled degradation chain applied to all 283 unique training videos — brightness .85–1.15, contrast .8–1.2, saturation .7–1.15, gamma .75–1.35 → Gaussian blur σ 0.4–2.2 → downscale ×0.45–0.85 & upscale (mixed interpolation) → Gaussian noise σ 2–14 → per-frame JPEG q 30–70 → H.264 re-encode CRF 28–34; 80% degraded / 20% clean mix; full 32B retrain (identical hyperparameters; final loss 0.35 — fits degraded data equally well). Result: **60.50 (−0.69)**. **The gap is not pixel statistics** — consistent with SynWTS being a geometry-matched digital twin and the vision tower being frozen (real-image-pretrained).

**E8. Cross-model ensembling (v10).** Majority vote over three differently-trained models (v1 8B, v8 32B, v9 32B-degraded); ties → v8. 242 answers changed vs v8. Result: **60.62 (−0.57)** — two weaker models outvote the stronger one more often than they correct it.

**E9. LoRA checkpoint souping (v11).** Element-wise average of v8's adapter checkpoints (steps 600/750/754). Only 110/12,316 answers changed (end-of-run checkpoints nearly converged). Result: **60.90 (−0.28)**.

**E10. Two-stage training + ego-view alignment (v12).** Motivation: (a) 2025 teams report caption-first→VQA-second staging; (b) §4's test-composition finding (79% ego-view test vs 74% overhead training). Setup: stage-1 caption-only SFT (1,415 ex., 2 ep) → merge into base → stage-2 VQA SFT on **ego-remapped** data (79% vehicle_view / 21% overhead, mirroring the test; 10,630 ex., 2 ep, ZeRO-3). Inference unchanged (each question already uses its referenced video). 1,110 answers changed vs v8, 63% of changes on BDD questions (the mechanism engaged where aimed). Result: **60.90 (≈0)**. **Even view-aligned synthetic training does not recover the gap — strengthening the conclusion that the residual mismatch is behavioral (real pedestrian/vehicle dynamics vs scripted synthetic events), not viewpoint.**

**Also attempted (infrastructure outcomes, worth one line each):** GLM-4.1V-9B and InternVL3.5-38B untrainable in the pinned stack (ms-swift 4.2.1 / transformers 5.8.1 video-template failures; InternVL additionally hit transformers-5.x remote-code API drift). Molmo2-8B: swift has native model+video template; three remote-code incompatibilities patched live (CLI arg, `ProcessorMixin` kwargs strictness, `ROPE_INIT_FUNCTIONS['default']` removed in transformers 5.x); abandoned at the deadline before training. Discriminative option-likelihood scoring (VL-JEPA-inspired): mean token-log-prob of each option text under the letter-SFT model scores **13–16% (below chance)** on val — letter-format SFT makes full-text likelihoods anti-correlated; the mechanism requires an embedding-prediction architecture.

# PART VI — ANALYSES

## 9. Error taxonomy (synth val, v1, `scripts/analyze_vqa.py`)
Failure concentrates in two spatial-relational categories: **pedestrian body orientation ≈ 40% accuracy** and **vehicle-relative position ≈ 36%**, vs 75–100% on most other categories (awareness, visual status, action, speed, field-of-view). Unchanged by scale (E6). Combined with §3's worked example, this yields a crisp paper claim: *fine-grained vehicle-relative spatial relations are the sim2real failure mode.*

## 10. The caption study (local official scorer; zero submissions burned)
We reproduced the official caption metric offline (pycocoevalcap + `metrics_all.py` against a GT-root mirror; identical formula), enabling free iteration:
| Config (v1 adapter) | Ped combined | Veh combined | Mean |
|---|---|---|---|
| baseline (vehicle-view video, standard prompt) | 27.0 | 39.1 | 33.03 |
| + perception-focused prompt (explicit attribute checklist) | 26.4 | 40.7 | 33.55 |
| + overhead view | 27.2 | 40.3 | 33.73 |
| + 2.6× resolution, 2 fps | 26.6 | 40.9 | 33.77 |

**Pedestrian captions are frozen (~27; BLEU/ROUGE ≈ .22/.36) under every test-time change** — the ceiling is the trained model, and since leaderboard caption metrics are clustered (all top teams BLEU ≈ .24–.28), captions are effectively saturated as a competitive axis. Vehicle captions respond mildly (+1.8).

## 11. Why captions transfer but VQA doesn't (discussion section material)
The references are strongly templated: fixed attribute clauses (demographics, clothing, weather, road surface) recur nearly verbatim between synthetic and real annotations, so n-gram metrics reward a surface structure the model reproduces regardless of domain (§7 example). VQA has no such shortcut: 71% of it is *ego-view real driving footage* asking fine-grained relational questions, which requires perception of real event dynamics. Four independent interventions each falsified an alternative explanation: pixels (E7), viewpoint (E10), capacity (E6), and inference scaffolding (E3–E5, E8). What remains is **behavioral realism** of the synthetic events — scripted trajectories, timing, and gaze patterns that don't match real ones. Actionable implication for the community: invest in *behaviorally* realistic synthetic generation (trajectory/dynamics fidelity), not photorealism; or permit limited real-data adaptation; or move to discriminative architectures (VL-JEPA's 87.4 shows the ceiling is reachable).

## 12. Limitations & data notes
- Leaderboard = fixed 50% subset (~6,158 questions): 1 question ≈ 0.016 Acc; deltas within ±0.3 are near resolution limits. Three late submissions (v11, a duplicate upload, v12) displayed identical metrics; v12's file was verified different (1,137 answers, distinct MD5) — treat v12 as "no measurable change" and note subset granularity.
- No test-set ground truth → per-category error analysis is only available on synthetic val.
- One base-model family (Qwen3-VL) carries most experiments; family-swap attempts were blocked by tooling, not science.
- Caption references' template redundancy inflates n-gram metrics' insensitivity to attribute errors — worth a sentence when interpreting caption scores.

# PART VII — PAPER PLAN, ASSETS, REPRODUCIBILITY

## 13. Suggested structure
*Title direction:* **"When Only Captions Cross: A Systematic Study of Synthetic-to-Real Transfer for Traffic-Safety Video VLMs."**
1. Intro (sim2real for safety-critical video; SynWTS→WTS+BDD setting)
2. Related work (§2 table)
3. Data & task (§1, §3–5; test-composition figure)
4. System (§6, prompts, stateless protocol)
5. Experiments E1–E10 (§8 tables + Part V)
6. Analysis (§9–11)
7. Discussion & limitations (§11–12)

**Figures/tables:** T1 main results (§8); T2 experiment ledger with Δ; F1 transfer-asymmetry bars (captions −0.8 vs VQA −19.7); F2 test view-composition (§4); F3 per-question-type accuracy (§9); F4 caption study (§10); optional worked-example box (§7).

## 14. Assets index
- **Repo** (all code, this document): `github.com/savanasunilkumar/wts-captioning-sim2real`. Key scripts: `build_sft_data.py`, `infer_realtest.py` (prompts + submission inference), `validate_submission.py` (0 rejected submissions in 15), `analyze_vqa.py`, `build_grounded_data.py`/`render_grounded.py`/`infer_grounded_hybrid.py`, `infer_fusion.py`, `vqa_likelihood.py`, `degrade_videos.py`, `val_caption_infer.py`; `slurm/` has every job including the ZeRO-3 32B config and the caption-lab harness.
- **Durable archive** `/work/anujs/savana/paper_archive` (1.1 GB): every scored submission pair (`*/merged/`), the FINAL best set, v8 adapter weights, all SLURM logs. v1 adapter backup: `/work/anujs/savana/model_backup/lora_v1_ckpt756`.
- **Checkpoints on scratch** (`/ptmp/anujs/savana/aicity-outputs/…` — auto-purges after 60 idle days): `lora_v1/…/checkpoint-756`, `lora_32b/…/checkpoint-754`, degraded / two-stage / grounded variants.
- **Environment:** ms-swift 4.2.1 · transformers 5.8.1 · torch 2.12.0+cu130 (A100-only build; V100 → `cudaErrorNoKernelImageForDevice`) · DeepSpeed · decord · qwen_vl_utils. Hardware: ISU Nova, 4×A100-SXM4-80GB (NVLINK), SLURM.
- **Wall-time budget for the paper's compute paragraph:** 8B train 3.5 h (1 GPU) · 32B train 5 h 41 m (4 GPU) · full VQA inference 1 h 50 m (4×1 GPU) · captions ≈ 3 h · video degradation 283 clips ≈ 30 min (16 CPU shards) · official caption metric runs locally in seconds.

*Compliance: all fine-tuning data synthetic (SynWTS train+val); no real frames/labels in training; no test-set self-training; open-weight base models only; repo history single-author.*
