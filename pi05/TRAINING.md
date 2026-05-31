# pi0.5 BC Training Pipeline

Stage 1 behavior cloning: fine-tune pi0.5 on PhysicalAI-AV ground truth driving data so it produces non-degenerate trajectories before Stage 2 (L4 counterfactual critique RL with AR1).

## Architecture

**Model:** pi0.5 (3.35B params)
- VLM: PaliGemma (SigLIP vision + Gemma-2B language) -- LoRA rank=32, alpha=64
- Action expert: Gemma-300M -- full fine-tune (manipulation actions don't transfer to driving)
- Action space: 128-dim flat vector = 64 timesteps x 2 (acceleration, curvature) at 10Hz = 6.4s horizon
- Uses alpamayo's `UnicycleAccelCurvatureActionSpace` -- kinematically feasible by construction

**Base checkpoint:** `gs://openpi-assets/checkpoints/pi05_base/params` (loaded by openpi's `CheckpointWeightLoader`). Shape-mismatched layers (action projections: 32-dim robot -> 128-dim driving) are randomly initialized; everything else loads from the base.

## Data Pipeline

```
PhysicalAI-AV (306K clips on HuggingFace)
    |
    | preload_dataset: snapshot_download to Modal volume
    | (git-lfs batch API, no per-file rate limits)
    v
Local Modal volume (/cache/physical_ai_av/)
    |
    | extract_parallel: 25 workers read local zips
    | traj_to_action() -> (accel, curvature) labels
    | trajectory_to_nav_prompt() -> navigation text
    v
Extracted data (/cache/extracted/xlarge/)
    samples.parquet + images/*.jpg
    |
    | _build_lerobot_from_extracted (inside train_bc)
    | Builds train + eval LeRobot datasets locally
    v
LeRobot datasets on volume (no HF push)
    /cache/hf/lerobot/markmusic/pi05-physical-av-bc       (train)
    /cache/hf/lerobot/markmusic/pi05-physical-av-bc-eval   (eval)
    |
    | openpi training loop (JAX/Flax, 8x H100)
    v
Checkpoints -> /cache/checkpoints/ + auto-push to HuggingFace
```

### Data Filters
- Right-hand traffic countries only (US, Germany, France, etc.)
- Daytime only (hour_of_day 8-18)
- Must have egomotion + camera_front_wide_120fov
- Speed > 1 m/s at t0 (skip parked)
- Trajectory length > 10m (skip very slow segments)
- Result: ~163K clips pass filters, we sample 50K

### Train/Eval Split
- 15% held out at clip level (not sample level)
- ~42,500 train clips x 5 t0s x ~60% pass rate = ~127K train samples
- ~7,500 eval clips -> ~22K eval samples
- Eval uses a SEPARATE LeRobot dataset with same normalization stats

## Navigation Labels + Prompt Dropout

**Labels:** Trajectory-derived discrete prompts from future egomotion geometry:
- `"drive forward"` -- |lateral| < 1m
- `"bear left/right"` -- 1-3m lateral displacement
- `"turn left/right"` -- >3m lateral displacement
- `"stop"` -- forward < 2m

**30% prompt dropout:** During training, 30% of samples have their navigation label replaced with a flat `"drive"` prompt. This is classifier-free guidance training:
- Forces the model to learn from visual context alone (30% of the time)
- When the label IS available, helps disambiguate at intersections (70%)
- At inference on the cart, the route planner always provides the real navigation intent

Implementation: `DrivingInputs.__call__` in the openpi patch (modal_train_bc.py).

## Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| GPUs | 8x H100 80GB | Data parallel (fsdp_devices=1) |
| Batch size | 96 (12/GPU) | SteerVLA-aligned |
| Steps | 15,000 (~11 epochs over 127K samples) | |
| LR | 3e-5 peak, cosine decay to 3e-6 | SteerVLA Table 2 |
| Warmup | 750 steps (5%) | |
| AdamW betas | (0.9, 0.999) | |
| LoRA | rank=32, alpha=64 on VLM Gemma-2B | SteerVLA-aligned |
| Action expert | Full fine-tune (Gemma-300M) | Domain shift too large for LoRA |
| EMA decay | 0.99 | openpi default |
| Precision | bfloat16 | |
| Save interval | 500 steps (~every 40 min) | |
| Eval | 20 batches from held-out eval dataset | |

## Running

### Prerequisites
- Modal account with secrets: `huggingface`, `wandb`, `google-ai` (optional)
- HuggingFace token with write access
- Checkpoint repo: `markmusic/pi05-physical-av-bc-checkpoint`

### Commands

```bash
# 1. Pre-flight: validate HF push
modal run pi05/modal_train_bc.py::validate_hf

# 2. Bulk download dataset to Modal volume (~1-2 hours, one-time)
modal run --detach pi05/modal_extract_data.py::preload_dataset --n-clips 50000

# 3. Extract from local volume (~30-60 min)
modal run --detach pi05/modal_extract_data.py::extract_parallel --scale xlarge

# 4. Train (auto-builds LeRobot datasets + norm stats, ~20 hours)
modal run --detach pi05/modal_train_bc.py::train_bc --scale xlarge

# During training: save checkpoint + push to HuggingFace
modal run pi05/modal_train_bc.py::trigger_push_hf

# After training: upload specific checkpoint
modal run pi05/modal_train_bc.py::upload_checkpoint --step 5000
```

### Resilience
- **Preload:** `snapshot_download` resumes automatically on re-run (checks existing files)
- **Extraction:** Checkpoints every 5 batches to parquet; resumes from last checkpoint
- **Training:** orbax checkpoints every 500 steps; `--resume` flag to continue
- **All functions:** `--detach` mode survives laptop disconnect

### Monitoring
- **W&B:** Project `pi05-driving`, logs loss/grad_norm/param_norm/learning_rate/eval_loss
- **Eval loss:** Computed from held-out eval dataset at every save_interval
- **Key signals:** eval_loss diverging from train_loss = overfitting

## What the Model Sees

**Input:**
- Front camera image (480x640 -> resized to 224x224 by SigLIP)
- State vector: [speed (m/s), heading_rate (rad/s)]
- Text prompt: navigation label OR "drive" (30% dropout)
- Two dummy wrist camera slots (zeros, masked out)

**Output:**
- 128-dim action vector (64 timesteps x 2: acceleration + curvature)
- Decoded to XYZ trajectory via `action_to_traj()` at inference

## Costs (estimated)

| Step | Resource | Time | Cost |
|---|---|---|---|
| Preload | CPU | ~1-2 hr | ~$5 |
| Extraction | CPU x25 workers | ~30-60 min | ~$20 |
| Training | H100:8 | ~20 hr | ~$500 |
| **Total** | | | **~$525** |
