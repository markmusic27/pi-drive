# Stage 1: Behavior Cloning Cold-Start — Implementation Plan

## Context

π0.5 is a generalist manipulation VLA that has never seen driving data. Before we can do L4 counterfactual critique RL (Stage 2), we need π0.5 to produce non-degenerate driving trajectories. This Stage 1 BC cold-start fine-tunes π0.5 directly on **PhysicalAI-AV ground truth egomotion** — real human driving trajectories from the dataset, converted to (acceleration, curvature) action labels via alpamayo's `traj_to_action()`.

**Key correction:** We do NOT use AR1 pseudo-labels for BC. AR1 is reserved for Stage 2 (L4 counterfactual critique). BC trains on ground truth.

**Validated so far (2026-05-27):**
- π0.5 loads and runs with `action_dim=2, action_horizon=64` on H100 (3.35B params, LoRA works)
- AR1-10B loads and runs inference on PhysicalAI-AV (1.07s/call on H100, 23.2 GB VRAM)
- Both models accessible without gating
- `traj_to_action()` can convert egomotion XYZ+rotation → (accel, curvature) directly, no AR1 needed

**Budget for Stage 1:** ~$830-1,530 of $7.7K total (8× H100 parallelization trades cost for speed)

---

## Navigation Conditioning

### The problem
PhysicalAI-AV has **NO navigation/route data**. Without navigation conditioning, π0.5 can't know whether to go straight or turn at an intersection — it learns an average over all possible maneuvers, leading to mode averaging.

Evidence this matters: Alpamayo paper Table 6 shows "Route ✓" configs substantially outperform "Route ×" on planning metrics.

### What Alpamayo does (paper, not open-sourced)
The paper shows route conditioning as text input: "In 400 feet, turn right". This was part of training but **not released** — the open-source model and dataset have no navigation data. The codebase has token definitions for route (route_start, route_pad, route_end) but the feature is disabled.

### Our approach: synthesize navigation from future trajectory

Since π0.5 takes a text prompt, we derive navigation intent from the future egomotion trajectory geometry and use it as the prompt:

```python
def trajectory_to_nav_prompt(ego_future_xyz: np.ndarray) -> str:
    """Derive navigation text from future trajectory shape.
    
    ego_future_xyz: (64, 3) in ego-local frame, 10Hz, 6.4s horizon
    """
    # Total lateral displacement at trajectory endpoint
    final_y = ego_future_xyz[-1, 1]   # positive = left in ego frame
    final_x = ego_future_xyz[-1, 0]   # forward distance
    
    # Compute heading change from trajectory arc
    mid_y = ego_future_xyz[31, 1]     # midpoint lateral
    
    # Classify maneuver
    if abs(final_y) < 1.0:
        return "drive forward"
    elif final_y > 3.0:
        return "turn left"
    elif final_y < -3.0:
        return "turn right"
    elif final_y > 1.0:
        return "bear left"
    else:
        return "bear right"
```

**Thresholds** (tuned on data): The exact thresholds for lateral displacement will be calibrated on the first data batch. Golf cart speeds (~5-10 m/s) over 6.4s means ~32-64m forward travel; a 3m lateral offset corresponds to a gentle curve, >3m is a definite turn.

**At inference time on the cart:** The navigation prompt comes from the route planner/GPS — "turn left ahead", "go straight", etc. This is the same information a human driver gets from navigation.

### Prompt vocabulary
Keep it small and discrete for clean conditioning:
- `"drive forward"` — straight/slight curves
- `"bear left"` / `"bear right"` — gentle lane changes or curves
- `"turn left"` / `"turn right"` — intersection turns
- `"stop"` — if speed approaches zero (rare in PhysicalAI-AV)

Each sample in the LeRobot dataset gets its prompt set to the derived navigation command. The `prompt_from_task` mechanism in openpi extracts this per-sample.

---

## Pipeline Overview

```
PhysicalAI-AV clips (306K available)
    │
    ▼
┌─────────────────────────────────┐
│  Step 1: Extract GT Egomotion   │  Modal CPU/GPU, ~$30-60
│  load_physical_aiavdataset()    │
│  traj_to_action() → (accel,    │
│  curvature) + nav prompt        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Step 2: LeRobot Dataset Build  │  Modal CPU, ~$10-20
│  Convert to LeRobot v2 format   │
│  Push to HuggingFace            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Step 3: Compute Norm Stats     │  Modal CPU, ~$5
│  Mean/std/quantiles for         │
│  state and actions              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Step 4: π0.5 Fine-Tune         │  Modal H100, ~$500-800
│  LoRA(VLM) + full(action exp.)  │
│  Flow-matching loss on GT       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Step 5: Eval (Open-Loop)       │  Modal H100, ~$50
│  minADE on held-out clips       │
└─────────────────────────────────┘
```

---

## Step 1: Extract Ground Truth Egomotion + Navigation

### What it does
Load PhysicalAI-AV clips, extract ground truth egomotion trajectories, convert to (acceleration, curvature) action labels using alpamayo's `traj_to_action()`, and synthesize navigation prompts from trajectory geometry. **No AR1 inference needed.**

### Conversion pipeline
```python
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

action_space = UnicycleAccelCurvatureActionSpace()

# For each (clip_id, t0):
data = load_physical_aiavdataset(clip_id, t0_us=t0)
# Returns: ego_history_xyz (1,1,16,3), ego_history_rot (1,1,16,3,3)
#          ego_future_xyz (1,1,64,3), ego_future_rot (1,1,64,3,3)

# Convert trajectory geometry → (accel, curvature) action labels
actions = action_space.traj_to_action(
    traj_history_xyz=data['ego_history_xyz'],
    traj_history_rot=data['ego_history_rot'],
    traj_future_xyz=data['ego_future_xyz'],
    traj_future_rot=data['ego_future_rot'],
)
# actions shape: (1, 1, 64, 2) — normalized [accel, curvature]

# Derive navigation prompt from future trajectory
nav_prompt = trajectory_to_nav_prompt(data['ego_future_xyz'][0, 0].numpy())
```

### Scale
| Scale | Clips | Samples (5 per clip) | Time | Cost |
|---|---|---|---|---|
| Tiny (smoke) | 50 | 250 | ~10 min | $5 |
| Small | 500 | 2,500 | ~30 min | $15 |
| **Medium (target)** | **5,000** | **25,000** | **~2 hr** | **$40** |
| Large | 20,000 | 100,000 | ~8 hr | $120 |

Without AR1 inference bottleneck, we can afford **more data** at lower cost. Target 5K clips / 25K samples (2× the previous plan). Each clip yields ~5 training samples at different timestamps (t0 = 3s, 5s, 7s, 9s, 11s into the 20s clip).

### Data filtering

Metadata available via `data_collection.parquet` (joined on `clip_id`) and `feature_presence.parquet`. Existing filtering pattern at `other_models/simlingo/scripts/extract_nvidia.py:79-94`.

```python
# Filter from data_collection.parquet + feature_presence.parquet
RIGHT_HAND_TRAFFIC = [
    'United States', 'Germany', 'France', 'Italy', 'Spain',
    'Netherlands', 'China', 'Canada', 'Mexico', 'Brazil',
]
# Excludes: UK, Japan, Australia, India, Singapore, etc. (left-hand traffic)

filtered = metadata[
    (metadata['country'].isin(RIGHT_HAND_TRAFFIC)) &
    (metadata['time_of_day'] == 'daytime')
]
filtered = filtered.merge(features, on='clip_id')
filtered = filtered[
    (filtered['has_egomotion'] == True) &
    (filtered['has_camera_front_wide'] == True)
]
```

- **Camera:** Front-wide 120° FOV only (matches cart's single front camera)
- **Driving side:** Right-hand traffic countries only (cart drives on right side of road)
- **Time:** Daytime only (`time_of_day == 'daytime'`)
- **Sensors:** Must have egomotion + front-wide camera
- **Speed filter:** ego velocity > 1 m/s at t0 (skip parked/stationary)
- **Trajectory quality:** future trajectory length > 10m (skip very slow segments)

### Per-sample output
For each (clip_id, t0) pair:
```python
{
    "image": np.ndarray,           # (H, W, 3) uint8 — front-wide camera at t0
    "state": np.ndarray,           # (2,) float32 — [speed, heading_rate] from ego history
    "actions": np.ndarray,         # (64, 2) float32 — [accel, curvature] from GT egomotion
    "nav_prompt": str,             # "drive forward", "turn left", etc.
    "clip_id": str,
    "t0_us": int,
}
```

### Key files
- Data loading: `alpamayo/src/alpamayo_r1/load_physical_aiavdataset.py`
- Action conversion: `alpamayo/src/alpamayo_r1/action_space/unicycle_accel_curvature.py:227-299` (`traj_to_action`)
- Clip enumeration: `physical_ai_av.PhysicalAIAVDatasetInterface` → `avdi.clip_index` DataFrame
- SimLingo route derivation (reference): `other_models/simlingo/scripts/nvidia_loader.py:115` (`egomotion_to_route()`)

### Implementation: `pi05/modal_extract_data.py`
```
Modal function: extract_driving_data(scale="medium", ...)
  1. Initialize PhysicalAIAVDatasetInterface (streaming mode)
  2. Filter clip_index for US/daytime/valid/has_egomotion
  3. Sample N clip_ids
  4. Initialize UnicycleAccelCurvatureActionSpace
  5. For each clip_id:
       For each t0 in [3s, 5s, 7s, 9s, 11s]:
         - load_physical_aiavdataset(clip_id, t0_us)
         - Extract front-wide camera image at t0
         - traj_to_action() → (64, 2) accel+curvature
         - trajectory_to_nav_prompt() → navigation text
         - Compute state: [speed, heading_rate] from ego history
         - Append to output
  6. Save to Modal volume as parquet + images
```

No GPU needed for this step (traj_to_action is CPU-only). But we need GPU memory for the alpamayo package install which includes torch. A single A10G or even CPU-only instance works.

---

## Step 2: LeRobot Dataset Build

### What it does
Convert the extracted egomotion data (parquet + images) into a LeRobot v2 dataset and push to HuggingFace. Each sample's task is set to its navigation prompt so `prompt_from_task=True` works.

### LeRobot v2 format
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.create(
    repo_id="markmusic/pi05-driving-bc",
    robot_type="cart_fsd",
    fps=10,  # 10 Hz action frequency
    features={
        "observation.images.front": {
            "dtype": "image",
            "shape": (480, 640, 3),  # downsampled from 1080×1920
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["speed", "heading_rate"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (2,),  # per-step action dim
            "names": ["acceleration", "curvature"],
        },
    },
    image_writer_threads=10,
)
```

### Episode structure
Each (clip_id, t0) pair = one episode of length 1 frame (single observation → 64-step action chunk). The `delta_timestamps` mechanism in openpi handles action chunking. This is standard for VLA training.

### Navigation as task
Each episode's task string is the derived navigation prompt: `"drive forward"`, `"turn left"`, etc. With `prompt_from_task=True` in the data config, openpi's `PromptFromLeRobotTask` transform extracts this as the text prompt per sample.

```python
# During dataset build:
for sample in extracted_data:
    dataset.add_frame({
        "observation.images.front": sample["image"],
        "observation.state": sample["state"],
        "actions": sample["actions"][t],  # per-timestep for action_horizon
    })
    dataset.save_episode(task=sample["nav_prompt"])  # e.g. "turn left"
```

### Image resolution
PhysicalAI-AV cameras are 1080×1920. Store at 480×640 (2.25× downsample) to balance quality and storage. The openpi `ResizeImages` transform handles the final 224×224 resize for SigLIP.

### Implementation: `pi05/modal_extract_data.py` (second function)
```
Modal function: build_lerobot_dataset(...)
  1. Load parquet + images from Modal volume
  2. Create LeRobotDataset with schema above
  3. For each sample:
       - dataset.add_frame({image, state, actions})
       - dataset.save_episode(task=nav_prompt)
  4. dataset.consolidate()
  5. dataset.push_to_hub(repo_id="markmusic/pi05-driving-bc", private=True)
```

---

## Step 3: Compute Normalization Stats

### What it does
Compute mean/std/q01/q99 for `state` and `actions` fields. π0.5 uses quantile normalization (not z-score).

### How
```bash
# Run from within openpi repo
uv run scripts/compute_norm_stats.py --config-name pi05_driving
```

This iterates the LeRobot dataset, computes running statistics, and saves to `assets/markmusic/pi05-driving-bc/norm_stats.json`.

### Expected ranges
- **acceleration:** roughly [-3, 3] m/s² for normal driving (AR1 bounds: [-9.8, 9.8])
- **curvature:** roughly [-0.05, 0.05] rad/m for normal driving (AR1 bounds: [-0.2, 0.2])
- **speed:** 0-15 m/s (~0-33 mph, golf cart range)
- **heading_rate:** roughly [-0.5, 0.5] rad/s

---

## Step 4: π0.5 Hybrid Fine-Tune (8× H100)

### Fine-tuning strategy: LoRA(VLM) + Full(action expert)

Following SteerVLA's approach (LoRA on language model, full fine-tune on remaining parameters):

- **VLM (PaliGemma Gemma-2B):** LoRA rank=32, alpha=64 — preserves vision-language knowledge about roads, traffic, driving. SteerVLA-aligned rank (32 vs openpi default 16).
- **Action expert (Gemma-300M):** Full fine-tune — manipulation actions (7-DOF robot arms) don't transfer to driving (2-DOF accel+curvature via unicycle model).

### Action space: Unicycle kinematic model

We use alpamayo's `UnicycleAccelCurvatureActionSpace` for the action representation:
- **Output:** (acceleration, curvature) × 64 waypoints at 10Hz = 6.4s horizon
- **Conversion:** `traj_to_action()` inverts unicycle dynamics: XYZ + rotation → heading → velocity → accel + curvature
- **Inference:** `action_to_traj()` integrates accel + curvature back to XYZ waypoints for vehicle control
- **Benefit over raw waypoints (SteerVLA):** outputs are kinematically feasible by construction — no post-hoc controller needed

### Multi-GPU: 8× H100 data parallelism

openpi supports multi-GPU via JAX FSDP mesh. With 8 H100s:
- `fsdp_devices=1` → pure data parallelism (model fits on single H100 at ~40-50 GB)
- `batch_size=96` → 12 samples per GPU
- JAX mesh shape: `(8, 1)` = 8 data-parallel replicas × 1 FSDP shard

### Training config

Add to `openpi/src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi05_driving",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=2,           # (acceleration, curvature) — unicycle model
        action_horizon=64,      # 6.4s at 10Hz
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",  # FULL fine-tune, not LoRA
    ),
    data=LeRobotDrivingDataConfig(
        repo_id="markmusic/pi05-driving-bc",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
    ).get_freeze_filter(),
    # --- SteerVLA-aligned hyperparameters ---
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=390,       # 5% of total steps
        peak_lr=3e-5,           # SteerVLA: 3×10⁻⁵
        decay_steps=7_800,      # = num_train_steps
        decay_lr=3e-6,          # 10× decay from peak
    ),
    optimizer=_optimizer.AdamW(
        b1=0.9,                 # SteerVLA beta1
        b2=0.999,               # SteerVLA beta2 (openpi default: 0.95)
        clip_gradient_norm=1.0,
    ),
    num_train_steps=7_800,      # ~25K samples / 96 batch × 30 epochs
    batch_size=96,              # SteerVLA: 96-120, splits 12/GPU across 8 H100s
    fsdp_devices=1,             # pure data parallelism
    save_interval=250,
    log_interval=50,
    checkpoint_base_dir="/cache/checkpoints",
)
```

### LoRA config customization

openpi's default Gemma-2B LoRA uses rank=16, alpha=16. To match SteerVLA (rank=32, alpha=64), we add a new variant `gemma_2b_lora_driving` in `openpi/src/openpi/models/gemma.py`:

```python
if variant == "gemma_2b_lora_driving":
    return Config(
        width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=64.0), "ffn": lora.LoRAConfig(rank=32, alpha=64.0)},
    )
```

**LoRA dropout:** SteerVLA uses 0.1 dropout, but openpi's LoRA implementation does NOT support dropout. This is a minor gap — the effect is regularization, partially compensated by EMA and weight decay.

### Custom data transforms

Create `openpi/src/openpi/policies/driving_policy.py`:

```python
@dataclasses.dataclass(frozen=True)
class DrivingInputs(transforms.DataTransformFn):
    model_type: ModelType = ModelType.PI05

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        return {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "actions": data.get("actions"),
            "prompt": data.get("prompt"),
        }

@dataclasses.dataclass(frozen=True)
class DrivingOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :2])}
```

### Weight loading behavior
The `CheckpointWeightLoader` merges base checkpoint weights with the new model. Since `action_dim` changes from 32 → 2:
- **Loaded from checkpoint:** SigLIP vision encoder, PaliGemma Gemma-2B (all layers), Gemma-300M action expert (all layers except action projection)
- **Randomly initialized:** `action_in_proj` (2 → width) and `action_out_proj` (width → 2)
- **LoRA weights:** Initialized fresh on Gemma-2B only

### Hyperparameters (SteerVLA-aligned)

| Param | Value | Source |
|---|---|---|
| VLM fine-tune | LoRA rank=32, alpha=64 | SteerVLA Table 2 |
| Action expert fine-tune | Full (all params) | SteerVLA A.1 + domain shift rationale |
| Learning rate | 3×10⁻⁵ peak, cosine decay | SteerVLA Table 2 |
| Warmup | 5% of total steps (~390 steps) | SteerVLA Table 2 |
| AdamW betas | (0.9, 0.999) | SteerVLA Table 2 |
| Batch size | 96 (12/GPU × 8 GPUs) | SteerVLA high-level policy |
| Epochs | ~30 | SteerVLA low-level policy |
| Steps | ~7,800 (25K / 96 × 30) | Derived |
| EMA decay | 0.99 | openpi default |
| Precision | bfloat16 | standard |
| GPUs | 8× H100 80GB | User requirement |
| Gradient accumulation | N/A | Not supported in openpi JAX |
| LoRA dropout | N/A | Not supported in openpi LoRA |

### Training command
```bash
# On Modal 8× H100
cd /opt/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_driving \
    --exp-name bc-coldstart \
    --overwrite
```

### Expected training time
- 25K samples, batch 96 → ~260 steps/epoch, 7,800 steps ≈ 30 epochs
- With 8× H100 data parallelism: ~5-10s/step at steady state
- Estimate: **11-22 hours** (vs 28-55 hours on 1× H100)
- Cost: ~$700-1,400 (8× H100 hourly rate is higher, but wall time is ~4× shorter)

### Monitoring & Checkpoints

#### Weights & Biases
openpi has W&B built in (`wandb_enabled=True` is the default). The training loop logs every `log_interval` steps:
- `loss` (flow-matching MSE)
- `grad_norm`
- `param_norm`
- Camera view images (step 0, sanity check)

Add `WANDB_API_KEY` as a Modal secret. W&B project name defaults to `"openpi"` — override via `project_name` in TrainConfig.

```python
secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")],
```

#### Checkpoint saving to HuggingFace
openpi saves checkpoints to local disk via orbax at every `save_interval` steps. To persist across Modal container restarts and push to HuggingFace:

1. **During training:** Checkpoints save to a Modal volume (`/cache/checkpoints/`)
   - `checkpoint_base_dir="/cache/checkpoints"` in TrainConfig
   - `save_interval=250` (every ~1 epoch, ~20 min with 8 GPUs) — frequent enough for manual inspection
   - `keep_period=1000` keeps permanent snapshots at steps 1000, 2000, etc.

2. **After training:** Upload best checkpoint to HuggingFace Hub
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(
       folder_path="/cache/checkpoints/pi05_driving/bc-coldstart/<best_step>/params",
       repo_id="markmusic/pi05-driving-bc-checkpoint",
       repo_type="model",
   )
   ```

3. **Periodic HF uploads during training:** Add a callback after each checkpoint save that uploads to HF if a trigger file exists or at every `keep_period` step.

#### Manual checkpoint triggering
To save a checkpoint on demand (when you see a good loss on W&B):

**Approach: file-based trigger.** The training loop checks for a trigger file on the Modal volume before each step. When you want to save:

```python
# In the training wrapper, patch the training loop:
import signal

def handle_save_signal(signum, frame):
    """Write trigger file when SIGUSR1 received."""
    pathlib.Path("/cache/save_trigger").touch()

signal.signal(signal.SIGUSR1, handle_save_signal)

# In the patched training loop, after each step:
trigger = pathlib.Path("/cache/save_trigger")
if trigger.exists():
    trigger.unlink()
    _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    logging.info(f"Manual checkpoint saved at step {step}")
```

**How to trigger from outside:** Use Modal's `Function.call()` or a separate Modal function that writes the trigger file to the shared volume. Or simply set `save_interval=250` so checkpoints are frequent enough (~every 20 min) that manual triggering is rarely needed.

### Modal GPU config
```python
@app.function(
    image=pi05_image,
    gpu="H100:8",              # 8× H100 80GB
    timeout=60 * 60 * 24,     # 24 hour timeout
    volumes={
        "/cache": modal.Volume.from_name("pi05-cache", create_if_missing=True),
    },
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    memory=128 * 1024,         # 128 GB system RAM
)
def train_bc(...):
    ...
```

---

## Step 5: Open-Loop Evaluation

### Metrics
- **minADE₆**: minimum Average Displacement Error across 6 trajectory samples
- **ADE@t**: ADE at specific time horizons (1s, 3s, 5s, 6.4s)
- **FDE**: Final Displacement Error at t=6.4s

### Protocol
1. Hold out 10% of clips (500 clips) as eval set during data extraction
2. Run π0.5 inference on eval set: `model.sample_actions(obs, num_steps=10)` → (64, 2)
3. Convert (accel, curvature) → XYZ via `action_to_traj()` for comparison with ground truth
4. Compute minADE against PhysicalAI-AV ground truth egomotion
5. Test with each navigation prompt to verify conditioning works

### Baselines
| Model | Expected minADE | Notes |
|---|---|---|
| Ground truth | 0 m | Upper bound (training signal) |
| AR1-10B | ~0.6-1.0 m | Reference (not used for training) |
| π0.5 BC cold-start | ~1.5-3.0 m | This stage's output |
| π0.5 random | >>10 m | Before any training |

### Navigation conditioning test
- Compare minADE when providing correct vs incorrect nav prompts
- e.g., on a left-turn clip, compare "turn left" vs "turn right" vs "drive forward"
- If conditioning works, correct prompt should have lowest ADE

---

## File Structure

```
pi-drive/pi05/
├── modal_validate.py              # Tier 1 validation (done)
├── modal_extract_data.py          # Step 1+2: GT egomotion extraction + LeRobot build
├── modal_train_bc.py              # Step 4: π0.5 hybrid fine-tune on Modal
├── nav_prompt.py                  # trajectory_to_nav_prompt() utility
└── eval/
    └── open_loop.py               # Step 5: minADE evaluation

openpi/ (modifications to cloned repo)
├── src/openpi/training/config.py  # Add pi05_driving config
└── src/openpi/policies/
    └── driving_policy.py          # DrivingInputs, DrivingOutputs transforms
```

---

## Execution Sequence

| Step | Script | GPU | Time | Cost | Depends on |
|---|---|---|---|---|---|
| 1a. Smoke extract (50 clips) | `modal_extract_data.py::extract --scale tiny` | CPU | 10 min | $3 | — |
| 1b. Smoke LeRobot build | `modal_extract_data.py::build_dataset` | CPU | 5 min | $1 | 1a |
| 1c. Smoke train (100 steps) | `modal_train_bc.py::train --steps 100` | H100:8 | 15 min | $50 | 1b |
| **Gate: loss decreases? If yes →** |
| 2a. Medium extract (5K clips) | `modal_extract_data.py::extract --scale medium` | CPU | 2 hr | $40 | — |
| 2b. Medium LeRobot build | `modal_extract_data.py::build_dataset` | CPU | 30 min | $5 | 2a |
| 2c. Compute norm stats | `compute_norm_stats.py` | CPU | 10 min | $2 | 2b |
| 2d. Production train (7.8K steps) | `modal_train_bc.py::train` | H100:8 | 11-22 hr | $700-1,400 | 2c |
| 2e. Open-loop eval | `eval/open_loop.py` | H100 | 30 min | $25 | 2d |
| **Total** | | | | **$830-1,530** | |

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| GT egomotion has noise/GPS jitter | `traj_to_action()` applies 2nd-order smoothing (Tikhonov regularization) |
| Nav prompt thresholds miscalibrated | Calibrate on first data batch; check class balance |
| π0.5 ignores nav prompt (mode collapse) | Test with swapped prompts in eval; if ignored, increase prompt diversity |
| Action expert full fine-tune overfits | EMA decay=0.99, dense checkpoints, early stopping on val loss |
| 25K frames too few for full action expert | Action expert is only 300M params; comparable to LIBERO scale |
| Training diverges | Dense checkpoints (every 1K steps), cosine LR with warmup |
| LeRobot format issues | Smoke test end-to-end before committing to medium scale |
| Single-camera limits driving ability | Sufficient for cold-start; multi-camera can be added in Stage 2 |

---

## Success Criteria

- [ ] Loss decreases monotonically during training (no divergence)
- [ ] Final train loss < 0.5 (flow-matching loss on normalized actions)
- [ ] Eval minADE < 3.0 m (π0.5 produces directionally correct trajectories)
- [ ] Sampled trajectories are smooth (no jitter, no degenerate collapse to zero)
- [ ] π0.5 responds differently to different scenes (not mode-collapsed)
- [ ] Navigation conditioning works: correct prompt gives lower ADE than incorrect prompt

---

## Verification Plan

1. **Smoke test (Step 1c):** Extract 50 clips → build LeRobot → train 100 steps → loss decreases
2. **Data quality check:** After medium extraction, inspect 20 random samples:
   - Are images reasonable? (daytime, US roads, front camera)
   - Do action labels make sense? (accel near 0 for cruising, positive curvature for left turns)
   - Are nav prompts correctly classified? (manual spot-check)
3. **Training convergence:** Monitor loss curve — should decrease smoothly after initial JIT warmup
4. **Navigation test:** On 50 eval clips, compare ADE with correct vs swapped prompts
5. **Trajectory visualization:** Plot 10 sampled trajectories vs ground truth on held-out clips
