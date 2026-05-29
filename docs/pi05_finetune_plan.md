# π0.5 Fine-Tuning Plan — L4 Counterfactual Critique

Full pivot from SimLingo. Goal: adapt π0.5 into a driving policy for Cart FSD using AR1 as a learning signal via counterfactual critique (L4).

---

## Access Status (validated 2026-05-27)

| Component | Status | Details |
|---|---|---|
| **π0.5 weights** | Available | Auto-download from `gs://openpi-assets/checkpoints/pi05_base/params`. Apache 2.0. |
| **AR1-10B weights** | Available | Ungated on HuggingFace `nvidia/Alpamayo-R1-10B`. 22 GB safetensors (BF16). Non-commercial. |
| **PhysicalAI-AV data** | Available | 306K clips, 133 TB total. Streaming via `physical_ai_av` SDK. |
| **openpi repo** | Cloned | `/Users/mmusic/Developer/Projects/cart/openpi` — JAX/Flax training pipeline. |
| **alpamayo repo** | Cloned | `/Users/mmusic/Developer/Projects/cart/alpamayo` — PyTorch inference + SFT fine-tuning. |

---

## Critical Technical Findings

### 1. AR1 has NO built-in critique mode

AR1 is trained to generate trajectories + Chain of Causation reasoning. It does NOT accept a trajectory as input for evaluation. The model's forward pass is:

```
images + ego_history + text_prompt → CoC reasoning text → trajectory (via diffusion)
```

**Implication:** L4 (counterfactual critique) requires fine-tuning AR1 into a critic. This is not optional — zero-shot prompting may partially work via the VLM backbone (`model.vlm.generate()`) but AR1 was never trained for this task.

The validation script (`pi05/modal_validate.py`) tests zero-shot critique as the first gating experiment.

### 2. Different frameworks: JAX vs PyTorch

| Component | Framework | Training infra |
|---|---|---|
| π0.5 (openpi) | JAX/Flax | JAX sharding, FSDP, LoRA built-in |
| AR1 (alpamayo) | PyTorch | DeepSpeed, HF Trainer, torchrun |

GRPO stage must orchestrate cross-framework: JAX for π0.5 sampling/gradients, PyTorch for AR1 critique inference. Options:
- **Option A:** Run both in same process (JAX + PyTorch can coexist on CUDA)
- **Option B:** Microservice architecture — AR1 critique server + π0.5 training client
- **Option C:** Convert one model (likely not worth it)

Recommendation: **Option B** for clean separation. AR1 critique runs as a Modal function, π0.5 training calls it.

### 3. Action space alignment is natural

Both models use the same representation:
- AR1: `UnicycleAccelCurvatureActionSpace` — (acceleration, curvature) × 64 waypoints at 10Hz
- π0.5: configurable `action_dim=2`, `action_horizon=64`
- Unicycle dynamics: acceleration bounds [-9.8, 9.8] m/s², curvature bounds [-0.2, 0.2] rad/m
- Integration: velocity → heading → position via trapezoidal rule

π0.5 outputs raw (accel, curvature) which can be integrated to XYZ waypoints using AR1's existing `UnicycleAccelCurvatureActionSpace.forward()`.

### 4. π0.5 architecture summary

- **Vision encoder:** SigLIP (So400m/14) → image token embeddings
- **Language model:** PaliGemma with Gemma-2B backbone
- **Action expert:** Gemma-300M with flow-matching decoder
- **π0.5 specific:** AdaRMS conditioning (time embedding via adaptive layer norm), discrete state tokenization
- **LoRA:** Built-in, rank=16 (Gemma-2B) + rank=32 (Gemma-300M), applies to attention + FFN
- **LoRA VRAM:** >22.5 GB, full fine-tune >70 GB
- **Training data format:** LeRobot v2 datasets

### 5. AR1 input requirements

- 4 cameras × 4 frames (0.3s history at 10Hz), resolution downsampled to 320×576
- 16-step ego history (1.6s at 10Hz) with XYZ + rotation matrices
- Text prompt (system + user + assistant start token)
- Output: CoC reasoning text + 64 waypoints (XYZ + rotation) via diffusion

### 6. AR1 fine-tuning for critic (Path 2)

AR1's SFT pipeline exists at `alpamayo/finetune/sft/`:
- Stage 1: Fine-tune VLM for discrete trajectory tokens (8× H100, DeepSpeed ZeRO-2)
- Stage 2: Freeze VLM, train action expert

For critic fine-tuning, we'd do a **modified Stage 1 only**: fine-tune the VLM backbone to output structured critique text instead of CoC + trajectory tokens. This avoids touching the diffusion head entirely.

---

## Execution Plan

### Phase 0: Tier 1 Validation (this week)

**Script:** `pi05/modal_validate.py`

```bash
# Run AR1 validation
modal run pi05/modal_validate.py::validate_ar1

# Run π0.5 validation  
modal run pi05/modal_validate.py::validate_pi05

# Run both in parallel
modal run pi05/modal_validate.py::validate_all
```

**What it tests:**

| Experiment | Question | Pass criterion |
|---|---|---|
| AR1 inference | Can we get trajectories + CoC reasoning? | minADE < 2.0m, CoC is parseable |
| AR1 zero-shot critique | Can AR1's VLM backbone evaluate a trajectory? | Responses are grounded in the scene (not generic) |
| AR1 latency | Per-inference cost on H100? | Needed for budget modeling |
| π0.5 action config | Does action_dim=2, action_horizon=64 work? | Forward pass produces (B, 64, 2) |
| π0.5 LoRA | Does freeze filter work? | LoRA params are trainable, base frozen |
| π0.5 sampling | Does action sampling produce correct shape? | sample_actions → (B, 64, 2) |

**Decision gate after Phase 0:**
- AR1 critique works zero-shot → proceed to L4 directly (skip critic fine-tuning)
- AR1 critique works partially → proceed with critic fine-tuning (Phase 1.5)
- AR1 critique fails completely → fall back to L3 (LLM-as-judge) or L2 (meta-action matching)

### Phase 1: Stage 1 BC Cold-Start

**Goal:** π0.5 produces non-degenerate driving trajectories.

1. **Data pipeline:** PhysicalAI-AV → AR1 pseudo-labels → LeRobot format
   - Run AR1 inference on N clips (target: 5-10K clips for cold-start)
   - Extract: front-wide camera frames + AR1 trajectory output
   - Convert to LeRobot dataset with (image, state=[speed, heading], actions=[accel, curvature])
   - Push to HuggingFace as `markmusic/pi05-driving-sft`

2. **Training config:** Add driving config to openpi
   - `action_dim=2, action_horizon=64, pi05=True`
   - LoRA: `gemma_2b_lora` + `gemma_300m_lora`
   - Weight loader: `pi05_base` checkpoint
   - Custom DrivingInputs/DrivingOutputs transforms

3. **Train:** `scripts/train.py pi05_driving --exp-name bc-coldstart`
   - Target: 10-30K steps, batch_size=32-128
   - Estimated cost: $800-1,200

4. **Eval:** Open-loop minADE on held-out PhysicalAI-AV clips

### Phase 1.5: Build the Critic (if needed)

**Goal:** AR1 reliably critiques arbitrary trajectories.

1. **Synthetic critique data:**
   - Take 2-5K good trajectories from PhysicalAI-AV
   - Apply perturbations: late brake (±0.5-1.0s), magnitude scaling (0.5-2.0×), lateral jitter, wrong direction, delayed response
   - Generate template critiques from perturbation parameters
   - Format: (scene_images, ego_history, perturbed_trajectory) → critique_text

2. **Fine-tune AR1 VLM for critique:**
   - Modified Stage 1 SFT: input includes trajectory, output is structured critique
   - LoRA on Qwen3-VL-8B backbone (AR1's VLM)
   - Target: 2-5K steps, ~$300-500

3. **Validate (Protocols 1 + 3):**
   - Protocol 1: perturbation recovery — ≥75% recall on 200 held-out trajectories
   - Protocol 3: critic-action consistency — corrected trajectories score higher

### Phase 2: L4 GRPO Post-Training

**Goal:** π0.5 outperforms BC baseline on long-tail driving via counterfactual critique.

```
for iter in 1..N:
    states = sample_from_dataset()
    for each state:
        for k in 1..K:
            tau_k = pi05.sample_trajectory(state)     # JAX
            critique_k = ar1_critic(state, tau_k)      # PyTorch (remote call)
            reward_k = parse_critique(critique_k) + safety_reward(tau_k)
    advantages = rewards - mean(rewards)  # within-group
    grpo_update(pi05, advantages, kl_to_bc_reference)
```

**Architecture:** AR1 critique server (Modal PyTorch function) + π0.5 training loop (Modal JAX function)

**Hyperparameters (starting point):**
- GRPO steps: 3-5K
- Batch size: 16
- Group size K: 2-4 (start K=2 to save budget)
- KL coefficient: 0.1
- Learning rate: 1e-6 to 1e-5
- Reward: α·critique_score + β·collision_penalty + γ·jerk_penalty

**Estimated cost:** $3,000-4,000

### Phase 3: Evaluation & Ablation

**Open-loop:** minADE₆ on PhysicalAI-AV held-out
- BC-only (Phase 1 checkpoint)
- BC + L4 (Phase 2 checkpoint)
- AR1-10B (upper bound)

**Closed-loop (if AlpaSim available):**
- Close encounter rate, off-road rate, progress
- Long-tail scenario subset

**Ablation (budget permitting):**
- L4 vs L2 (trajectory matching only, cheap — ~$500)

---

## Budget

| Phase | Estimated cost | Cumulative |
|---|---|---|
| Phase 0: Validation | $50-100 | $100 |
| Phase 1: BC cold-start | $800-1,200 | $1,300 |
| Phase 1.5: Critic fine-tune | $300-500 | $1,800 |
| Phase 2: L4 GRPO | $3,000-4,000 | $5,800 |
| Debug/smoke runs | $500-1,000 | $6,800 |
| Ablation (L2) | $500-800 | $7,600 |
| **Total** | | **$5,800-7,600** |

Fits within $7.7K budget. Margin is thin — no room for a second L4 run at different hyperparameters.

---

## File Structure

```
pi-drive/
├── pi05/
│   ├── modal_validate.py          # Tier 1 validation (built)
│   ├── modal_pseudolabel.py       # AR1 pseudo-label generation (Phase 1)
│   ├── modal_train_bc.py          # π0.5 BC training (Phase 1)
│   ├── modal_train_critic.py      # AR1 critic fine-tuning (Phase 1.5)
│   ├── modal_train_grpo.py        # L4 GRPO training (Phase 2)
│   ├── perturbations.py           # Trajectory perturbation toolkit
│   ├── critique_parser.py         # Critique text → scalar reward
│   ├── unicycle.py                # Accel+curvature ↔ XYZ conversion
│   └── eval/
│       ├── open_loop.py           # minADE evaluation
│       └── critic_validation.py   # Protocols 1 + 3
│
├── openpi/                        # Cloned openpi repo (π0.5 training)
│   └── (add driving config here)
│
└── alpamayo/                      # Cloned alpamayo repo (AR1 inference)
    └── (add critic fine-tune config here)
```

---

## Fallback Hierarchy

1. **L4 with fine-tuned critic** (primary) — counterfactual critique as reward
2. **L4 with zero-shot critic + filtering** (degraded) — if critic works partially
3. **L3 with LLM-as-judge** (alternative) — Claude/GPT-4 judges trajectory-reasoning consistency
4. **L2 with trajectory matching** (safety net) — reward = -||τ_π0.5 - τ_AR1||²

Decide fallback level after Phase 0 validation results.

---

## Next Action

Run the validation script:
```bash
cd /Users/mmusic/Developer/Projects/cart/pi-drive
modal run pi05/modal_validate.py::validate_all
```

This takes ~30 minutes and costs ~$10-20. Results determine whether L4 is viable and which path to take.
