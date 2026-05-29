"""Tier 1 validation: confirm AR1 and π0.5 load and run.

Usage:
    modal run pi05/modal_validate.py::validate_ar1
    modal run pi05/modal_validate.py::validate_pi05
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-tier1-validation"
CACHE_DIR = "/cache"
ALPAMAYO_DIR = "/opt/alpamayo"
OPENPI_DIR = "/opt/openpi"

# ---------------------------------------------------------------------------
# AR1 Image — install alpamayo from repo, skip cosmos-rl and vllm
# ---------------------------------------------------------------------------

ar1_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs", "build-essential", "ninja-build", "ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "torchvision>=0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("wheel", "setuptools", "packaging")
    .pip_install("flash-attn>=2.8.3", extra_options="--no-build-isolation")
    .run_commands(
        f"git clone --depth 1 https://github.com/NVlabs/alpamayo.git {ALPAMAYO_DIR}",
        # Remove cosmos-rl and vllm from deps (not needed for inference)
        f"sed -i '/cosmos-rl/d; /vllm/d' {ALPAMAYO_DIR}/pyproject.toml",
        f"pip install -e {ALPAMAYO_DIR}",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/hf",
        }
    )
)

# ---------------------------------------------------------------------------
# π0.5 Image — JAX + openpi
# ---------------------------------------------------------------------------

pi05_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "build-essential", "clang")
    .pip_install("uv")
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git {OPENPI_DIR}",
        f"cd {OPENPI_DIR} && uv sync",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/hf",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        }
    )
)

# ---------------------------------------------------------------------------
# Volumes & App
# ---------------------------------------------------------------------------

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# AR1: load weights, run inference on one clip, print CoC + trajectory shape
# ---------------------------------------------------------------------------


@app.function(
    image=ar1_image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=64 * 1024,
)
def validate_ar1():
    import torch
    import numpy as np
    import time

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    # Load data
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    print(f"Loading PhysicalAI-AV clip {clip_id}...")
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    print(f"  image_frames: {data['image_frames'].shape}")
    print(f"  ego_history_xyz: {data['ego_history_xyz'].shape}")
    print(f"  ego_future_xyz: {data['ego_future_xyz'].shape}")

    # Load model
    print("\nLoading AR1-10B...")
    t0 = time.time()
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16
    ).to("cuda")
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Prepare inputs
    processor = helper.get_processor(model.tokenizer)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )

    # Run inference
    print("\nRunning inference...")
    torch.cuda.manual_seed_all(42)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    infer_time = time.time() - t0

    # Results
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    min_ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1).min()

    print(f"\n--- AR1 RESULTS ---")
    print(f"  pred_xyz shape: {pred_xyz.shape}")
    print(f"  pred_rot shape: {pred_rot.shape}")
    print(f"  minADE: {min_ade:.4f} m")
    print(f"  inference time: {infer_time:.2f}s")
    print(f"  CoC trace:\n{extra['cot'][0][0]}")

    # Quick latency check (3 more runs — re-prepare inputs each time since .pop() mutates)
    import copy
    latencies = []
    for _ in range(3):
        fresh_inputs = processor.apply_chat_template(
            helper.create_message(data["image_frames"].flatten(0, 1)),
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        fresh_model_inputs = helper.to_device(
            {
                "tokenized_data": fresh_inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
            "cuda",
        )
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=fresh_model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=False,
            )
        torch.cuda.synchronize()
        latencies.append(time.time() - t0)
    print(f"  avg latency (3 runs): {np.mean(latencies):.2f}s")
    print(f"  VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("\nAR1 VALIDATION PASSED")


# ---------------------------------------------------------------------------
# π0.5: load weights, forward pass + sample with action_dim=2, horizon=64
# ---------------------------------------------------------------------------


@app.function(
    image=pi05_image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=60 * 30,
    memory=64 * 1024,
)
def validate_pi05():
    import subprocess
    import os

    script = r'''
import sys, time
sys.path.insert(0, "/opt/openpi/src")

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from openpi.models.pi0_config import Pi0Config
from openpi.models import model as _model

print("JAX devices:", jax.devices())

# --- Test 1: forward pass with driving config ---
print("\n--- TEST 1: forward pass (action_dim=2, action_horizon=64) ---")
config = Pi0Config(
    pi05=True,
    action_dim=2,
    action_horizon=64,
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
)

print(f"  config: pi05={config.pi05}, action_dim={config.action_dim}, "
      f"action_horizon={config.action_horizon}, max_token_len={config.max_token_len}")

rng = jax.random.PRNGKey(42)
t0 = time.time()
model = config.create(rng)
print(f"  model created in {time.time() - t0:.1f}s")

param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model)))
print(f"  parameters: {param_count:,}")

B = 1
dummy_obs = _model.Observation(
    images={
        "base_0_rgb": jnp.zeros([B, 224, 224, 3]),
        "left_wrist_0_rgb": jnp.zeros([B, 224, 224, 3]),
        "right_wrist_0_rgb": jnp.zeros([B, 224, 224, 3]),
    },
    image_masks={
        "base_0_rgb": jnp.ones([B], dtype=jnp.bool_),
        "left_wrist_0_rgb": jnp.ones([B], dtype=jnp.bool_),
        "right_wrist_0_rgb": jnp.zeros([B], dtype=jnp.bool_),
    },
    state=jnp.zeros([B, 2]),
    tokenized_prompt=jnp.zeros([B, config.max_token_len], dtype=jnp.int32),
    tokenized_prompt_mask=jnp.ones([B, config.max_token_len], dtype=jnp.bool_),
)
dummy_actions = jnp.zeros([B, 64, 2])

t0 = time.time()
loss = model.compute_loss(rng, dummy_obs, dummy_actions, train=False)
print(f"  loss shape: {loss.shape}")
print(f"  loss value: {float(loss.mean()):.4f}")
print(f"  forward time: {time.time() - t0:.1f}s")
assert loss.shape == (B, 64), f"expected (1,64), got {loss.shape}"
print("  PASS")

# --- Test 2: action sampling ---
print("\n--- TEST 2: action sampling ---")
t0 = time.time()
actions = model.sample_actions(rng, dummy_obs, num_steps=10)
print(f"  sampled shape: {actions.shape}")
print(f"  range: [{float(actions.min()):.4f}, {float(actions.max()):.4f}]")
print(f"  sample time: {time.time() - t0:.1f}s")
assert actions.shape == (B, 64, 2), f"expected (1,64,2), got {actions.shape}"
print("  PASS")

# --- Test 3: LoRA config ---
print("\n--- TEST 3: LoRA freeze filter ---")
lora_config = Pi0Config(
    pi05=True,
    action_dim=2,
    action_horizon=64,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
lora_model = lora_config.create(jax.random.PRNGKey(0))
lora_params = sum(p.size for p in jax.tree.leaves(nnx.state(lora_model)))
freeze_filter = lora_config.get_freeze_filter()
print(f"  LoRA params: {lora_params:,}")
print(f"  freeze filter: {type(freeze_filter).__name__}")
print("  PASS")

print("\nPI05 VALIDATION PASSED")
'''

    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True,
        text=True,
        cwd=OPENPI_DIR,
        env={
            **os.environ,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        },
    )

    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
    return result.returncode
