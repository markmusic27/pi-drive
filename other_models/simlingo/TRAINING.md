# SimLingo NVIDIA Fine-Tuning — Operator's Guide

Single-source reference for the production training run on NVIDIA PhysicalAI-AV
real-world driving data. Captures the training stack, launch procedure, monitoring,
and every gotcha we've hit so we don't re-discover them.

---

## 1. Goal

Fine-tune SimLingo VLA (InternVL2-1B vision + Qwen2.5-0.5B LM) on real-world
NVIDIA driving clips, warm-starting from the public `RenzKa/simlingo` checkpoint.
Output: a model that produces drivable waypoints from a single forward-camera
image + speed + target point, intended for golf-cart deployment.

---

## 2. Training stack at a glance

| Component | Choice | Source / Config |
|---|---|---|
| Base model | `RenzKa/simlingo` (LoRA warm start) | `config/nvidia_finetune.yaml:11` |
| Vision encoder | InternVL2-1B (frozen + LoRA q/k/v/o_proj) | same |
| Language model | Qwen2.5-0.5B (LoRA q/k/v/o_proj) | same |
| Fine-tuning | LoRA rank=32, alpha=64, dropout=0.1 | same |
| Precision | bf16 | same |
| Batch | 48 per rank × 4 grad-accum × 4 GPUs = 768 effective | same |
| LR | 3e-5, cosine, 5% warmup | same |
| Loss | Smooth L1 (waypoints) + 0.5 × Smooth L1 (route) | same |
| Distributed | DDP (NCCL), find_unused_parameters=False prod | same |
| GPU | **B200:8** on Modal (sm_100, CUDA 12.4 / PyTorch 2.4.1) | `modal_training.py:411` |
| Path | **vec** (Tier A batched) — ~13% faster than unvec at steady state | `config/nvidia_finetune.yaml:50` |
| Early stop | val/loss patience=3, min_delta=1e-4 | same |
| Checkpoints | every 250 steps + best-val + end-of-epoch | same |
| HF Hub | auto-create + upload `lora_final.pt` + `lora_best.pt` | `nvidia_trainer.py:push_to_hub` |
| Data | NVIDIA front-wide 120° camera, daytime US | `scripts/extract_nvidia.py` |
| Waypoints | 11 points at 0.25s spacing (2.75s horizon) | dataset/config |

---

## 3. Launch commands

### Production training (after medium extract finishes)
```bash
cd /Users/mmusic/Developer/Projects/cart/pi-drive/simlingo
# Repo is auto-created on first upload (private=true by default).
modal run --detach modal_training.py::train_multigpu \
  --config-path /app/config/nvidia_finetune.yaml \
  --wandb-project simlingo-nvidia-finetune \
  --hf-repo markmusic/simlingo-nvidia
```

### Resuming after early stop / manual stop
```bash
modal run --detach modal_training.py::train_multigpu \
  --config-path /app/config/nvidia_finetune.yaml \
  --wandb-project simlingo-nvidia-finetune \
  --hf-repo markmusic/simlingo-nvidia \
  --resume-from /checkpoints/checkpoint_epoch_5.pt
```

### Manually stopping (preserves checkpoints on volume)
```bash
modal app list                # find your app id
modal app stop <app-id> -y    # safe to stop anytime; resume with --resume-from
```

### Smoke test (1 epoch, tiny data, sanity check)
```bash
modal run --detach modal_training.py::train_multigpu \
  --config-path /app/config/nvidia_smoke.yaml \
  --wandb-project simlingo-nvidia-smoke
```

### Data extraction (medium, 2.5K clips)
```bash
modal run --detach modal_training.py::prepare_nvidia_data --scale medium
```

### Monitoring a launched app
```bash
modal app list                          # find your app ID
modal app logs <app-id> 2>&1 | tail -50 # latest logs
```

W&B URL convention: `https://wandb.ai/markmusic/<project>`

---

## 4. Configs

| Config | Purpose | Notes |
|---|---|---|
| `nvidia_finetune.yaml` | **Production** — 20 epochs full data | `vectorized` flag set per latest bench |
| `nvidia_smoke.yaml` | Quick sanity check (1 epoch tiny) | Used for DDP smoke validation |
| `nvidia_smoke_vec.yaml` | Vec-path smoke | Validated 0.25% loss delta vs unvec |
| `nvidia_bench_unvec.yaml` | 5-epoch bench, no eval/save | Steady-state step time comparison |
| `nvidia_bench_vec.yaml` | Same as above, vectorized=true | Pair with above |

---

## 5. Validated decisions

**Numerical equivalence of vec vs unvec paths** (validated 2026-05-18 on tiny smoke):
- per-sample path: val/loss = 5.2004 (`olive-hill-5`)
- vectorized path: val/loss = 5.2131 (`fiery-sponge-1`)
- Δ = 0.25% — within run-to-run variance

**Vec IS faster at steady state** (corrected 2026-05-18 after full profile):

Per-step section timings (avg of steps 10/15/20, H100:4, bs=48):

| Section      | VEC (ms) | UNVEC (ms) | Δ              |
|--------------|---------:|-----------:|---------------:|
| image_prep   |   1,673  |    1,353   | +320 (vec slower) |
| lang_build   |     146  |      305   | −159 (vec faster) |
| forward      |  17,203  |   16,710   | +493 (vec ~3% slower) |
| **backward** |  **3,088** | **6,909** | **−3,821 (vec 55% faster)** |
| **total**    | **~22.1s** | **~25.3s** | **vec 13% faster** |

The earlier "vec slower" call was based on warmup step 5 data (vec bwd 11.6s →
3.1s by step 10 as the cached graph stabilizes). At steady state vec wins by
doing one batched backward instead of N per-sample backwards. Source: W&B
runs `breezy-pond-2` (vec) and `warm-smoke-1` (unvec) in `simlingo-nvidia-prof`.

**DDP smoke succeeded on H100:4** — single set of checkpoints saved (proves
rank-0 gating works), no `module.` prefix issues, full epoch + final checkpoint
completed. Commit `6cee269`.

---

## 6. Gotchas we have hit (read this before debugging)

### 6.1 B200 image: PyTorch 2.4.1 cu124 was NOT enough (RESOLVED 2026-05-18)
First B200 rebuild attempt used CUDA 12.4.1 + PyTorch 2.4.1 cu124 wheels and
**failed at first forward pass** with:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
PyTorch did not include prebuilt sm_100 kernels in 2.4.x. The fix is **CUDA
12.8.1 base + PyTorch 2.7.0 cu128 wheels** (`modal_training.py:66-90`). 2.7
is the first release with sm_100 binaries baked in.

If you ever see "no kernel image is available" again, it's almost certainly
because someone downgraded torch to <2.6 or switched the wheel index back to
cu124. Run a smoke first.

### 6.2 Multi-rank wandb.init creates duplicate runs
The trainer object is built AFTER `wandb.init()` runs at entrypoint, so we
can't gate via `trainer.is_main`. Use the RANK env var instead. This is now
implemented at the bottom of `nvidia_trainer.py`:
```python
_entrypoint_rank = int(os.environ.get("RANK", "0"))
if _entrypoint_rank == 0:
    wandb.init(...)
else:
    os.environ["WANDB_MODE"] = "disabled"
    wandb.init(mode="disabled")
```

### 6.3 subprocess output buffering — RESOLVED (2026-05-18)
`train_multigpu` now streams torchrun stdout/stderr via `subprocess.Popen`
line-by-line, so `modal app logs <app-id>` shows live training output.
W&B remains the most reliable live-metrics surface.

For W&B-only monitoring (e.g. checking from a different machine):
```python
import wandb
api = wandb.Api()
r = api.run(f'markmusic/{project}/{run_id}')
print(list(r.scan_history(keys=['_step', 'train/loss'])))
```

### 6.4 DDP state_dict has `module.` prefix
DistributedDataParallel wraps the model, so `model.state_dict()` keys start
with `module.`. We store `self._raw_model = self.model.module if distributed
else self.model` and use `_raw_model.state_dict()` for checkpoint saves.
Load path is symmetric.

### 6.5 GitHub push fails on >100MB files
Visualization output videos (`simlingo/special/*.mp4`, `waypoint_viz/*.mp4`)
exceed GitHub's 100MB limit. Gitignore patterns now cover `*.mp4` and
`simlingo/special/`. If you ever see a large file rejection on push:
```bash
git rm --cached <big-file>
git reset --mixed origin/main   # restructure local commits if needed
```

### 6.6 NVIDIA egomotion timestamp range mismatches
Some clips have `egomotion.timestamps.min() ≈ 33ms` instead of 0. The
extractor passes `timestamps.min()=0` for waypoint sampling and skips these
clips with a logged error. Not fatal — typical loss is ~1% of clips.

### 6.7 `max_steps` in config is NOT implemented
The trainer respects `epochs` only. If you set `max_steps: 50` it will be
ignored. To limit step count: pick `epochs` such that `epochs × steps_per_epoch`
matches your target. Approximate `steps_per_epoch = ceil(num_samples /
(batch_size × world_size × grad_accum))`.

**Update (2026-05-18):** Early stopping IS now wired (`training.early_stopping:
true`, `early_stopping_patience: 3`, `early_stopping_min_delta: 1e-4`). It
breaks out of the epoch loop after N consecutive end-of-epoch evals show no
val/loss improvement. DDP-aware: rank 0 decides, broadcasts to peers so no
NCCL hang.

### 6.8 `vectorized: true` forces `max_num=1` in dynamic_preprocess
This guarantees all images yield P=1 patches so they can be safely stacked.
If you ever launch vec with `max_num >= 2` it will crash at the
`torch.cat(pixel_values_list, dim=0)` step inside `train_step_batched`.

### 6.9 LoRA + `find_unused_parameters` interaction
LoRA-only fine-tuning leaves many params with zero gradients. DDP requires
either:
- `find_unused_parameters=True` (scans autograd graph every backward, slight overhead)
- Disable grad for all non-LoRA params at setup (cleaner, what production uses)

Smoke configs use `True` to surface any unused-param issues; production
uses `False`.

### 6.10 SimLingo's DrivingInput is naturally batch-1 in the unvec path
Reading the SimLingo source: the model's `forward()` does support batched
input (verified Task #6), but the per-sample loop in `train_step` was the
original pattern. Vec path was added to use the upstream batching.

### 6.11 Camera intrinsics: hardcoded 110° FOV, NVIDIA is 120°
`get_camera_intrinsics(W, H, 110)` in the trainer assumes 110° FOV. NVIDIA
front-wide camera is 120°. This is a known mismatch — the actual K matrix
from clip calibration should be used instead. See plan doc Phase 6 / Camera
Configuration section. **Pre-production TODO.**

### 6.12 Background tasks in Claude Code can fire stale notifications
The Claude task system polls long-running background processes and reports
their completion. These show up as "task-notification" messages but contain
no new info if the task was just a status-check poller. Safe to ignore unless
they reference an actual new event.

---

## 7. Cost & time estimates (recalibrated 2026-05-18)

Profile data showed forward dominates at ~16s/step in BOTH vec and unvec; the
prior $450/12h estimate was 10x optimistic. Real numbers below.

**B200:8 production stack** (~$72/hr Modal; 1.5-1.8x H100, exact TBD):

| Dataset | Frames | Epochs | Steps/optim | Wall-clock (B200:8 @1.5x) | Cost |
|---|---|---|---|---|---|
| Tiny smoke | 2.5K | 1 | ~6 | ~5 min | $6 |
| Small | 25K | 10 | ~3K | ~7 hr | $500 |
| **Medium (target, 20ep, vec)** | **125K** | **20** | **~13K** | **~40 hr** | **~$2,800** |
| Medium (early-stop @ ep 5-6, vec) | 125K | 5-6 | ~3-4K | ~10-12 hr | ~$800 |

**Recommendation:** launch with `epochs: 20`, but plan for early stop at epoch
5-6 based on val/loss plateau. With `early_stopping_patience: 3` and dense
checkpoints (every 250 steps), the realistic cost is ~$1,000-1,500.

Budget headroom (~$8.8K credits): 5-8 production runs at conservative
projections, more if early stop fires aggressively.

---

## 8. What to expect from the production run

Healthy training signals to verify in the first hour:
- `train/loss` decreases monotonically through warmup (from ~3.5 baseline)
- `val/loss` tracks `train/loss` within 10-20% (no big gap = not overfitting)
- `train/wp_loss` should fall toward ~0.5 by end of epoch 1
- `train/lr` follows cosine schedule (rises during warmup, then drops)
- Visualization panel renders BEV + projected waypoints every eval_steps

Red flags:
- Loss diverges or NaNs → almost always a precision issue (check bf16 ops)
- val/loss flat while train/loss drops → overfitting, reduce LR or add dropout
- val/loss > 2x train/loss → distribution mismatch (check train/val split)
- Only one rank logging to W&B → rank-0 gating working correctly (not a bug)

---

## 9. Useful queries

### Get latest W&B run for a project
```python
import wandb
api = wandb.Api()
runs = sorted(api.runs('markmusic/simlingo-nvidia-finetune'),
              key=lambda r: r.created_at, reverse=True)
print(runs[0].name, runs[0].id, runs[0].state)
```

### Per-step time from W&B history
```python
r = api.run(f'markmusic/{project}/{run_id}')
hist = list(r.scan_history(keys=['_runtime', '_step', 'train/loss']))
for h in hist:
    print(f"step={h['_step']} t={h['_runtime']:.1f}s loss={h['train/loss']:.4f}")
```

### Check Modal app status
```bash
modal app list | head -10
modal app logs <app-id> 2>&1 | tail -30
```

---

## 10. Phase checklist (where we are right now)

- [x] Data pipeline (egomotion → waypoints, projection viz validated)
- [x] Tiny extraction (50 clips)
- [x] DDP infrastructure (rank-0 gating, sampler, _raw_model)
- [x] Smoke test on H100:4 (DDP smoke validated)
- [x] Vectorization opt-in flag (numerically equivalent, NOT faster — disabled in prod)
- [x] Profile vec vs unvec per-section timing (forward is the bottleneck)
- [x] Camera intrinsics fix (110° → 120° via `config.data.fov_deg`)
- [x] B200 image rebuild (CUDA 12.4.1 / PyTorch 2.4.1)
- [x] Early stopping + HF Hub auto-push wired
- [ ] Medium extraction (in flight, ~48% complete at time of writing)
- [ ] B200:8 1-epoch smoke (validate new image before multi-day run)
- [ ] Production training run on medium dataset (B200:8, unvec, early-stop)
- [ ] Open-loop eval (ADE/FDE on held-out)

---

_Last updated: 2026-05-18 (during bench investigation)_
