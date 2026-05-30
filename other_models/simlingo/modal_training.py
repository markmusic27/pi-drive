"""Modal training app for SimLingo fine-tuning on NVIDIA real-world data.

This trains SimLingo VLA on NVIDIA PhysicalAI-AV data, warm-starting from
the pretrained SimLingo checkpoint with LoRA fine-tuning.

Setup:
    1. Configure Modal secrets:
       modal secret create wandb WANDB_API_KEY=<your-key>
       modal secret create huggingface HF_TOKEN=<your-token>

    2. Prepare data (run extraction first):
       modal run modal_training.py::prepare_nvidia_data --scale tiny

    3. Run training:
       modal run modal_training.py::train --config config/nvidia_finetune.yaml

Usage:
    # Prepare tiny dataset for validation
    modal run modal_training.py::prepare_nvidia_data --scale tiny

    # Run training
    modal run modal_training.py::train

    # Run training with W&B and HF checkpointing
    modal run modal_training.py::train --wandb-project simlingo-nvidia --hf-repo your-name/simlingo-nvidia

    # Evaluate on held-out data
    modal run modal_training.py::evaluate
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "simlingo-nvidia-training"

# Upstream SimLingo
SIMLINGO_REPO_URL = "https://github.com/RenzKa/simlingo.git"
SIMLINGO_REPO_DIR = "/opt/simlingo"

# HF model artifacts
HF_MODEL_REPO = "RenzKa/simlingo"
HF_CKPT_FILE = "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
HF_HYDRA_CONFIG_FILE = "simlingo/.hydra/config.yaml"

# Volume mountpoints
CACHE_DIR = "/cache"
DATA_DIR = "/data"
NVIDIA_DATA_DIR = "/nvidia_data"
OUTPUTS_DIR = "/outputs"
CHECKPOINTS_DIR = "/checkpoints"
CKPT_DIR = f"{DATA_DIR}/checkpoint"

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

training_image = (
    modal.Image.from_registry(
        # CUDA 12.8 base — required for B200 sm_100 kernels in PyTorch.
        # PyTorch 2.4.1+cu124 did NOT include sm_100 in its prebuilt kernels
        # (smoke failed with "no kernel image is available for execution on
        # the device"). PyTorch 2.7 with cu128 wheels is the first official
        # release with sm_100 binaries baked in.
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",  # Python 3.11 for physical_ai_av
    )
    .apt_install(
        "git",
        "git-lfs",
        "build-essential",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
        "ffmpeg",  # For video encoding
    )
    # PyTorch 2.7.0 + CUDA 12.8 wheels (sm_100 / B200 / Blackwell support).
    .pip_install(
        "torch==2.7.0",
        "torchvision==0.22.0",
        "torchaudio==2.7.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # NVIDIA PhysicalAI-AV SDK first (has strict huggingface-hub requirement)
    .pip_install(
        "physical_ai_av",
    )
    # Training dependencies (versions relaxed to be compatible with physical_ai_av)
    .pip_install(
        "transformers==4.44.2",  # Pinned for InternVL2 compatibility
        "tokenizers>=0.19.0,<0.20",  # Match transformers 4.44 requirement
        "sentencepiece",
        "peft>=0.13.0",
        "accelerate>=1.0.0",
        "pytorch-lightning>=2.4.0",
        "lightning>=2.3.0",
        "hydra-core>=1.3.0",
        "hydra-zen>=0.12.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "timm>=0.9.16",
        "scipy>=1.10.0",
        "scikit-image>=0.21.0",
        "imgaug>=0.4.0",
        "Pillow>=10.2.0",
        "filterpy>=1.4.5",
        "ujson>=5.9.0",
        "matplotlib>=3.7.0",
        "tqdm",
        "numpy<2",
        "opencv-python-headless>=4.10.0",
        "line_profiler",
        # W&B for experiment tracking
        "wandb",
    )
    # Skip flash-attn for now (takes 20+ min to build, not needed for testing)
    # For training, uncomment this:
    # .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("deepspeed>=0.16.0")
    # Clone SimLingo repo
    .run_commands(
        f"git clone --depth 1 {SIMLINGO_REPO_URL} {SIMLINGO_REPO_DIR}",
    )
    .env(
        {
            "PYTHONPATH": SIMLINGO_REPO_DIR,
            "HF_HOME": f"{CACHE_DIR}/hf",
            "HUGGINGFACE_HUB_CACHE": f"{CACHE_DIR}/hf/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_DIR}/hf/transformers",
            "TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("scripts")
    .add_local_dir("config", remote_path="/app/config")
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

cache_volume = modal.Volume.from_name("simlingo-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("simlingo-data", create_if_missing=True)
nvidia_data_volume = modal.Volume.from_name("simlingo-nvidia-data", create_if_missing=True)
output_volume = modal.Volume.from_name("simlingo-outputs", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("simlingo-checkpoints", create_if_missing=True)

VOLUMES = {
    CACHE_DIR: cache_volume,
    DATA_DIR: data_volume,
    NVIDIA_DATA_DIR: nvidia_data_volume,
    OUTPUTS_DIR: output_volume,
    CHECKPOINTS_DIR: checkpoint_volume,
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 6,  # 6 hours for large datasets
    cpu=8,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def prepare_nvidia_data(
    scale: str = "tiny",
    output_subdir: str = "extracted",
    create_split: bool = True,
    val_ratio: float = 0.1,
) -> dict:
    """Download and extract NVIDIA data for training.

    Args:
        scale: Dataset scale (tiny/small/medium/large).
        output_subdir: Subdirectory within NVIDIA_DATA_DIR.
        create_split: Whether to create train/val split.
        val_ratio: Validation set ratio.

    Returns:
        Summary of extraction.
    """
    sys.path.insert(0, os.path.dirname(__file__))

    import physical_ai_av

    from scripts.extract_nvidia import (
        SCALE_CONFIGS,
        create_train_val_split,
        extract_dataset,
        sample_clip_ids_from_api,
    )

    output_dir = Path(NVIDIA_DATA_DIR) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get HF token from secret
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment")

    print(f"Initializing NVIDIA PhysicalAI-AV with scale={scale}")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(token=token)

    # Get clip IDs
    config = SCALE_CONFIGS[scale]
    print(f"Scale: {scale} - {config['description']}")

    clip_ids = sample_clip_ids_from_api(avdi, config["num_clips"])
    if not clip_ids:
        raise ValueError("No clips available from API")

    # Extract data
    summary = extract_dataset(
        avdi=avdi,
        clip_ids=clip_ids,
        output_dir=output_dir,
        frames_per_clip=config["frames_per_clip"],
    )

    # Create train/val split
    if create_split:
        create_train_val_split(output_dir, val_ratio=val_ratio)

    nvidia_data_volume.commit()

    return summary


@app.function(
    image=training_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 3,
    cpu=4,
    secrets=[modal.Secret.from_name("huggingface")],
)
def prepare_base_model(force: bool = False) -> dict:
    """Download SimLingo base model and InternVL2 weights."""
    from huggingface_hub import hf_hub_download, snapshot_download

    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    # Download SimLingo checkpoint
    for filename in (HF_CKPT_FILE, HF_HYDRA_CONFIG_FILE):
        dst = Path(CKPT_DIR) / filename
        if dst.exists() and not force:
            print(f"  cached: {dst}")
            continue
        print(f"  downloading: {filename}")
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            local_dir=CKPT_DIR,
            repo_type="model",
        )

    # Download InternVL2-1B
    print("  downloading: OpenGVLab/InternVL2-1B")
    snapshot_download(
        repo_id="OpenGVLab/InternVL2-1B",
        local_dir=f"{CACHE_DIR}/hf/snapshots/InternVL2-1B",
        ignore_patterns=["*.bin"],
    )

    cache_volume.commit()
    data_volume.commit()

    return {"ckpt_dir": CKPT_DIR, "status": "ready"}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes=VOLUMES,
    gpu="H100",  # Use H100 for fastest training
    timeout=60 * 60 * 24,  # 24 hours max
    cpu=8,
    memory=64 * 1024,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def train(
    config_path: str = "/app/config/nvidia_finetune.yaml",
    data_dir: str | None = None,
    wandb_project: str = "simlingo-nvidia-finetune",
    wandb_entity: str | None = None,
    hf_repo: str | None = None,
    resume_from: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    lora_rank: int | None = None,
) -> dict:
    """Train SimLingo on NVIDIA data with LoRA fine-tuning.

    Args:
        config_path: Path to training config YAML.
        data_dir: Override data directory.
        wandb_project: W&B project name.
        wandb_entity: W&B entity (team/user).
        hf_repo: HuggingFace repo for checkpoint uploads (e.g., "user/simlingo-nvidia").
        resume_from: Checkpoint path to resume from.
        epochs: Override number of epochs.
        batch_size: Override batch size.
        learning_rate: Override learning rate.
        lora_rank: Override LoRA rank.

    Returns:
        Training summary with metrics.
    """
    import torch
    import wandb
    import yaml
    from omegaconf import OmegaConf

    sys.path.insert(0, SIMLINGO_REPO_DIR)
    sys.path.insert(0, os.path.dirname(__file__))

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if data_dir:
        config["data"]["train_dir"] = data_dir
    if epochs:
        config["training"]["epochs"] = epochs
    if batch_size:
        config["training"]["batch_size"] = batch_size
    if learning_rate:
        config["training"]["learning_rate"] = learning_rate
    if lora_rank:
        config["lora"]["rank"] = lora_rank

    # Set default data directory if not specified
    if not config["data"].get("train_dir"):
        config["data"]["train_dir"] = f"{NVIDIA_DATA_DIR}/extracted"

    print("Training config:")
    print(yaml.dump(config, default_flow_style=False))

    # Initialize W&B
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        config=config,
        name=f"nvidia-finetune-lr{config['training']['learning_rate']}-lora{config['lora']['rank']}",
    )

    # Import training utilities
    from scripts.nvidia_trainer import NVIDIASimLingoTrainer

    # Initialize trainer
    trainer = NVIDIASimLingoTrainer(
        config=config,
        ckpt_dir=CKPT_DIR,
        output_dir=CHECKPOINTS_DIR,
        hf_repo=hf_repo,
    )

    # Resume if specified
    if resume_from:
        print(f"Resuming from: {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Train
    print("Starting training...")
    metrics = trainer.train()

    # Save final checkpoint
    final_ckpt = trainer.save_checkpoint("final")
    print(f"Saved final checkpoint: {final_ckpt}")

    # Push to HuggingFace if configured
    if hf_repo:
        print(f"Pushing checkpoint to HuggingFace: {hf_repo}")
        trainer.push_to_hub(hf_repo)

    wandb.finish()
    checkpoint_volume.commit()

    return {
        "metrics": metrics,
        "checkpoint": str(final_ckpt),
        "config": config,
    }


@app.function(
    image=training_image,
    volumes=VOLUMES,
    # H100:8 (sm_90) — 2x throughput vs H100:4. We pair this with the config's
    # batch_size=24 (override from 48) so the effective batch stays 24*8*4=768,
    # preserving the same convergence-per-step as the previous H100:4 setup
    # while halving the wallclock per epoch. B200 NCCL is still blocked on
    # Modal (4+ min silent stall at comm init with PyTorch 2.7+cu128); revisit
    # B200:8 once that's fixed.
    gpu="H100:8",
    timeout=86400,  # 24h (Modal's hard maximum per function call). User
                    # requested 36h but Modal caps at 86400s. To go longer,
                    # re-invoke with --resume-from <latest ckpt> after the
                    # job hits the cap; combine with the SAVE_NOW sentinel
                    # (request_checkpoint below) for proactive saves.
    cpu=64,        # 8 GPUs * dataloader workers
    memory=512 * 1024,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def train_multigpu(
    config_path: str = "/app/config/nvidia_finetune.yaml",
    wandb_project: str = "simlingo-nvidia-finetune",
    hf_repo: str | None = None,
    resume_from: str | None = None,
    auto_resume: bool = False,
) -> dict:
    """Multi-GPU training with DDP on 8x H100.

    Streams torchrun stdout/stderr live to Modal logs (no more buffering blindspot).
    Pass `--hf-repo <user>/<repo>` to override the config's hub.repo_id.
    Pass `--resume-from <ckpt>` to resume from a saved checkpoint (e.g. on early
    stop + restart).
    Pass `--auto-resume` to auto-pick the most recent checkpoint on the
    `simlingo-checkpoints` volume — useful when bridging a 24h timeout into a
    follow-on run.
    """
    import subprocess
    import sys as _sys

    # Auto-resume: find the most recently modified checkpoint on the volume
    # if no explicit --resume-from was provided. Preference order:
    # checkpoint_best.pt > checkpoint_epoch_*.pt (highest) > checkpoint_step_*.pt
    # > any other checkpoint_*.pt by mtime.
    if auto_resume and not resume_from:
        ckpt_dir = Path(CHECKPOINTS_DIR)
        candidates = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            resume_from = str(candidates[0])
            print(f"[auto-resume] picked latest checkpoint: {resume_from}", flush=True)
        else:
            print("[auto-resume] no checkpoints found; starting fresh", flush=True)

    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=29500",
        "-m", "scripts.nvidia_trainer",
        "--config", config_path,
        "--wandb-project", wandb_project,
        "--output-dir", CHECKPOINTS_DIR,
    ]
    if hf_repo:
        cmd.extend(["--hf-repo", hf_repo])
    if resume_from:
        cmd.extend(["--resume-from", resume_from])

    print(f"Running: {' '.join(cmd)}", flush=True)

    # NCCL config for B200 (sm_100) on Modal single-node multi-GPU:
    #   - NCCL_DEBUG=INFO + NCCL_DEBUG_SUBSYS=ALL surfaces every init step so we
    #     can see exactly where bootstrap stalls.
    #   - NCCL_IB_DISABLE=1: Modal containers don't expose InfiniBand; let
    #     NCCL skip the IB probe (which can stall ~30s otherwise).
    #   - NCCL_SOCKET_IFNAME=lo: all 4 ranks are in the SAME container, so
    #     bootstrap goes over loopback. Without this NCCL may pick a stale
    #     virtual interface and hang.
    #   - TORCH_NCCL_BLOCKING_WAIT=1 / TORCH_NCCL_ASYNC_ERROR_HANDLING=1 turn
    #     indefinite hangs into a timeout + stack trace (PyTorch 2.7 names).
    #   - PYTHONUNBUFFERED=1 ensures child process prints stream to logs.
    env = os.environ.copy()
    env.update({
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "ALL",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "lo",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTHONUNBUFFERED": "1",
    })

    # Stream output instead of capturing — gives live train/loss visibility in
    # `modal app logs <app-id>` and avoids the OOM risk of unbounded buffering
    # on a multi-day run.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        for line in proc.stdout:
            _sys.stdout.write(line)
            _sys.stdout.flush()
    finally:
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"torchrun exited with code {rc}")

    checkpoint_volume.commit()
    return {"status": "completed", "returncode": rc}


# ---------------------------------------------------------------------------
# Out-of-band checkpoint controls
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes=VOLUMES,
    timeout=60,
    cpu=1,
    memory=512,
)
def request_checkpoint(tag: str = "manual") -> dict:
    """Ask the running trainer to save a checkpoint right now.

    Writes a sentinel file `SAVE_NOW.<tag>` into the simlingo-checkpoints
    volume and commits it. The trainer's rank-0 process polls the volume
    every 10 optimizer steps; on detection it saves
    `checkpoint_<tag>.pt`, `lora_<tag>.pt`, `training_state_<tag>.pt`,
    then deletes the sentinel.

    Usage:
        modal run simlingo/modal_training.py::request_checkpoint --tag liked
    """
    from pathlib import Path

    # Sanitize: filename-safe chars only, fall back to "manual".
    safe_tag = "".join(c for c in tag if c.isalnum() or c in "_-").strip("_-")
    safe_tag = safe_tag or "manual"

    sentinel = Path(CHECKPOINTS_DIR) / f"SAVE_NOW.{safe_tag}"
    sentinel.touch()
    checkpoint_volume.commit()
    print(f"Queued checkpoint save with tag='{safe_tag}' -> {sentinel}")
    return {"status": "queued", "tag": safe_tag, "sentinel": str(sentinel)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=60 * 60,
    cpu=8,
    memory=64 * 1024,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def evaluate_loss(
    checkpoint_name: str = "epoch_1",
    config_path: str = "/app/config/nvidia_finetune.yaml",
) -> dict:
    """Load a saved checkpoint and run trainer.validate() to report val loss.

    `checkpoint_name` is the suffix from `checkpoint_<name>.pt` on the
    `simlingo-checkpoints` volume (e.g. `epoch_1`, `best`, `step_500`,
    `manual_liked`).

    Returns the validate() metrics dict (val_loss, val_wp_loss, val_route_loss).

    Usage:
        modal run simlingo/modal_training.py::evaluate_loss --checkpoint-name epoch_1
    """
    import os
    import yaml
    import wandb

    sys.path.insert(0, SIMLINGO_REPO_DIR)
    sys.path.insert(0, "/app")

    from scripts.nvidia_trainer import NVIDIASimLingoTrainer

    # Disable W&B for the eval-only run.
    os.environ["WANDB_MODE"] = "disabled"
    wandb.init(mode="disabled")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = NVIDIASimLingoTrainer(
        config=config,
        ckpt_dir=os.environ.get("CKPT_DIR", "/data/checkpoint"),
        output_dir=CHECKPOINTS_DIR,
        hf_repo=None,
    )

    ckpt_path = f"{CHECKPOINTS_DIR}/checkpoint_{checkpoint_name}.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)

    metrics = trainer.validate()
    print(f"[evaluate_loss] checkpoint={checkpoint_name} metrics={metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Evaluation (legacy inference path)
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes=VOLUMES,
    gpu="L4",
    timeout=60 * 60 * 2,
    cpu=4,
    memory=24 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def evaluate(
    checkpoint_path: str | None = None,
    data_dir: str | None = None,
    num_samples: int = 100,
    save_overlays: bool = True,
) -> dict:
    """Evaluate fine-tuned model on held-out data.

    Args:
        checkpoint_path: Path to fine-tuned checkpoint. If None, uses base SimLingo.
        data_dir: Path to evaluation data.
        num_samples: Number of samples to evaluate.
        save_overlays: Whether to save visualization overlays.

    Returns:
        Evaluation metrics.
    """
    sys.path.insert(0, SIMLINGO_REPO_DIR)
    sys.path.insert(0, os.path.dirname(__file__))

    from scripts import inference
    from scripts.nvidia_loader import ExternalSample

    # Set paths
    if checkpoint_path is None:
        ckpt_path = str(Path(CKPT_DIR) / HF_CKPT_FILE)
    else:
        ckpt_path = checkpoint_path

    hydra_cfg_path = str(Path(CKPT_DIR) / HF_HYDRA_CONFIG_FILE)

    if data_dir is None:
        data_dir = f"{NVIDIA_DATA_DIR}/extracted"

    out_dir = Path(OUTPUTS_DIR) / "nvidia_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation samples from extracted data
    samples = load_extracted_samples(data_dir, num_samples, split="val")

    print(f"Evaluating on {len(samples)} samples...")

    summary = inference.run_external_samples(
        ckpt_path=ckpt_path,
        hydra_cfg_path=hydra_cfg_path,
        out_dir=str(out_dir),
        samples=samples,
        use_cot=True,
        save_overlays=save_overlays,
        label="nvidia_eval",
    )

    output_volume.commit()

    print("Evaluation metrics:", summary["metrics"])
    return summary


def load_extracted_samples(
    data_dir: str,
    num_samples: int,
    split: str = "val",
) -> list:
    """Load ExternalSample objects from extracted data on disk.

    Args:
        data_dir: Path to extracted data directory.
        num_samples: Number of samples to load.
        split: Which split to use (train/val).

    Returns:
        List of ExternalSample objects.
    """
    import gzip
    import json
    import random

    from scripts.nuscenes_loader import ExternalSample

    data_dir = Path(data_dir)

    # Load split file if exists
    split_file = data_dir / "train_val_split.json"
    if split_file.exists():
        with open(split_file) as f:
            split_info = json.load(f)
        clip_ids = split_info.get(split, [])
    else:
        # Use all clips
        clip_ids = [d.name for d in data_dir.iterdir() if d.is_dir() and (d / "rgb").exists()]

    random.shuffle(clip_ids)
    samples = []

    for clip_id in clip_ids:
        if len(samples) >= num_samples:
            break

        clip_dir = data_dir / clip_id
        rgb_dir = clip_dir / "rgb"
        meas_dir = clip_dir / "measurements"

        if not rgb_dir.exists():
            continue

        frame_files = sorted(rgb_dir.glob("*.jpg"))

        for frame_file in frame_files:
            if len(samples) >= num_samples:
                break

            frame_idx = int(frame_file.stem)
            meas_file = meas_dir / f"{frame_idx:04d}.json.gz"

            if not meas_file.exists():
                continue

            with gzip.open(meas_file, "rt") as f:
                meas = json.load(f)

            samples.append(ExternalSample(
                rgb_path=frame_file,
                speed_mps=meas["speed"],
                target_points=np.array(
                    [meas["target_point"], meas["target_point_next"]],
                    dtype=np.float32,
                ),
                intrinsics=np.array(meas["intrinsics"], dtype=np.float32) if "intrinsics" in meas else None,
                fov_deg=meas.get("fov_deg", 120.0),
                cam_translation_xyz=(0.0, 1.5, 2.0),  # Default for NVIDIA
                crop_bottom=False,
                gt_wps=np.array(meas["waypoints"], dtype=np.float32),
                gt_route=np.array(meas["route"], dtype=np.float32),
                gt_commentary=None,
                meta={
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                },
            ))

    return samples


# Import numpy for load_extracted_samples
import numpy as np


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes=VOLUMES,
    gpu="L4",
    timeout=60 * 60,
    cpu=4,
    memory=24 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def visualize_nvidia_clip(
    clip_id: str,
    num_frames: int = 20,
    make_video: bool = True,
    fps: int = 10,
) -> dict:
    """Generate visualization of a single NVIDIA clip with waypoints.

    This is useful for validating the data pipeline before training.

    Args:
        clip_id: NVIDIA clip ID to visualize.
        num_frames: Number of frames to visualize.
        make_video: Whether to create MP4 from frames.
        fps: Video frame rate.

    Returns:
        Path to output files.
    """
    sys.path.insert(0, os.path.dirname(__file__))

    import physical_ai_av

    from scripts.nvidia_viz import (
        create_visualization_video,
        visualize_clip_waypoints,
    )

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(token=token)

    output_dir = Path(OUTPUTS_DIR) / "nvidia_viz" / clip_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing clip: {clip_id}")
    visualize_clip_waypoints(
        avdi=avdi,
        clip_id=clip_id,
        output_dir=output_dir,
        num_frames=num_frames,
    )

    if make_video:
        video_path = output_dir / "visualization.mp4"
        create_visualization_video(output_dir, video_path, fps=fps)
        print(f"Created video: {video_path}")

    output_volume.commit()

    return {
        "output_dir": str(output_dir),
        "num_frames": num_frames,
        "clip_id": clip_id,
    }


# ---------------------------------------------------------------------------
# Inference on a raw mp4 + ego.jsonl (the 'special' folder) with a LoRA ckpt
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes=VOLUMES,
    gpu="H100",
    timeout=60 * 60 * 2,
    cpu=8,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def visualize_special_predictions(
    lora_name: str = "lora_best.pt",
    video_rel: str = "special/front.mp4",
    ego_rel: str = "special/ego.jsonl",
    out_subdir: str = "special_predictions",
    fov_deg: float = 70.0,
    cam_height_m: float = 1.2,
    frame_stride: int = 1,
    max_frames: int | None = None,
    output_fps: int = 30,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
) -> dict:
    """Run SimLingo + a saved LoRA on a raw mp4 + ego.jsonl pair and produce a
    camera+BEV side-by-side viz video.

    Inputs (relative to /outputs):
      - video_rel : the mp4
      - ego_rel   : matching ego.jsonl

    Output (under /outputs/<out_subdir>):
      - predictions.mp4
      - frames/frame_*.jpg
      - summary.json
    """
    sys.path.insert(0, os.path.dirname(__file__))

    from scripts.special_inference import run_special_inference

    out_dir = Path(OUTPUTS_DIR) / out_subdir
    video_path = Path(OUTPUTS_DIR) / video_rel
    ego_path = Path(OUTPUTS_DIR) / ego_rel
    if not video_path.exists():
        raise FileNotFoundError(f"video not found at {video_path}")
    if not ego_path.exists():
        raise FileNotFoundError(f"ego.jsonl not found at {ego_path}")

    ckpt_path = Path(CKPT_DIR) / HF_CKPT_FILE
    hydra_cfg_path = Path(CKPT_DIR) / HF_HYDRA_CONFIG_FILE
    lora_path = Path(CHECKPOINTS_DIR) / lora_name
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found at {lora_path}")

    lora_cfg = {
        "rank": lora_rank,
        "alpha": lora_alpha,
        "target_modules": [m.strip() for m in lora_target_modules.split(",") if m.strip()],
    }

    summary = run_special_inference(
        video_path=video_path,
        ego_path=ego_path,
        ckpt_path=ckpt_path,
        hydra_cfg_path=hydra_cfg_path,
        lora_path=lora_path,
        lora_cfg=lora_cfg,
        out_dir=out_dir,
        fov_deg=fov_deg,
        cam_height_m=cam_height_m,
        frame_stride=frame_stride,
        max_frames=max_frames,
        output_fps=output_fps,
    )
    output_volume.commit()
    print("[done]", summary)
    return summary


# ---------------------------------------------------------------------------
# Test / Debug Functions
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes=VOLUMES,
    timeout=60 * 30,  # 30 minutes
    cpu=4,
    memory=16 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_nvidia_connection() -> dict:
    """Test connection to NVIDIA PhysicalAI-AV API and list available clips.

    This is a quick sanity check to verify HF token and API access.

    Returns:
        dict with connection status and sample clip IDs.
    """
    import physical_ai_av

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found")
        return {"status": "error", "message": "HF_TOKEN not found"}

    print("Initializing NVIDIA PhysicalAI-AV interface...")
    print(f"physical_ai_av version: {physical_ai_av.__version__}")

    try:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(token=token)
        print("Connection successful!")

        # Print available attributes/methods on avdi
        print(f"AVDI type: {type(avdi)}")
        print(f"Available features: {dir(avdi.features)}")

        # Try to list clips using different methods
        print("Attempting to list clips...")
        clip_count = "unknown"
        sample_clips = []

        # Method 1: list_clips()
        if hasattr(avdi, 'list_clips'):
            try:
                clips = avdi.list_clips()
                clip_count = len(clips) if clips else 0
                sample_clips = list(clips[:5]) if clips else []
                print(f"Method list_clips() returned {clip_count} clips")
            except Exception as e:
                print(f"list_clips() failed: {e}")

        # Method 2: clips property/method
        if not sample_clips and hasattr(avdi, 'clips'):
            try:
                clips = list(avdi.clips())[:100] if callable(avdi.clips) else list(avdi.clips)[:100]
                clip_count = len(clips)
                sample_clips = clips[:5]
                print(f"Method clips() returned {clip_count} clips")
            except Exception as e:
                print(f"clips() failed: {e}")

        # Method 3: get_clip_ids()
        if not sample_clips and hasattr(avdi, 'get_clip_ids'):
            try:
                clips = avdi.get_clip_ids()
                clip_count = len(clips) if clips else 0
                sample_clips = list(clips[:5]) if clips else []
                print(f"Method get_clip_ids() returned {clip_count} clips")
            except Exception as e:
                print(f"get_clip_ids() failed: {e}")

        # Method 4: Check metadata
        if not sample_clips and hasattr(avdi, 'metadata'):
            try:
                print(f"Metadata available: {type(avdi.metadata)}")
            except Exception as e:
                print(f"metadata access failed: {e}")

        # Method 5: Check features_df
        if not sample_clips and hasattr(avdi.features, 'features_df'):
            try:
                df = avdi.features.features_df
                print(f"features_df type: {type(df)}")
                print(f"features_df columns: {list(df.columns) if hasattr(df, 'columns') else 'N/A'}")
                if hasattr(df, 'columns') and 'clip_id' in df.columns:
                    clips = df['clip_id'].unique().tolist()
                    clip_count = len(clips)
                    sample_clips = clips[:5]
                    print(f"Found {clip_count} unique clip_ids in features_df")
            except Exception as e:
                print(f"features_df access failed: {e}")

        # Method 6: Use clip_index
        if not sample_clips and hasattr(avdi, 'clip_index'):
            try:
                clip_index = avdi.clip_index
                print(f"clip_index type: {type(clip_index)}")
                if hasattr(clip_index, 'columns'):
                    print(f"clip_index columns: {list(clip_index.columns)}")
                    print(f"clip_index shape: {clip_index.shape}")
                    # Get clip IDs
                    if 'clip_id' in clip_index.columns:
                        clips = clip_index['clip_id'].tolist()
                    else:
                        # Try index
                        clips = clip_index.index.tolist()
                    clip_count = len(clips)
                    sample_clips = clips[:10]
                    print(f"Found {clip_count} clips via clip_index")
                    print(f"Sample clip IDs: {sample_clips}")
            except Exception as e:
                import traceback
                print(f"clip_index access failed: {e}")
                print(traceback.format_exc())

        # Method 7: Try data_collection
        if not sample_clips and hasattr(avdi, 'data_collection'):
            try:
                dc = avdi.data_collection
                print(f"data_collection type: {type(dc)}")
                if hasattr(dc, 'columns'):
                    print(f"data_collection columns: {list(dc.columns)}")
                    print(f"First few rows: {dc.head()}")
            except Exception as e:
                print(f"data_collection access failed: {e}")

        print(f"\nResult: {clip_count} clips available")
        if sample_clips:
            print(f"Sample clips: {sample_clips}")

        result = {
            "status": "success",
            "clip_count": clip_count,
            "sample_clips": sample_clips,
            "message": "API connection successful",
        }
        print(f"\nReturning: {result}")
        return result

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


@app.function(
    image=training_image,
    volumes=VOLUMES,
    timeout=60 * 60,  # 1 hour for downloading
    cpu=4,
    memory=32 * 1024,  # More memory for video
    secrets=[modal.Secret.from_name("huggingface")],
)
def visualize_single_clip(
    clip_id: str,
    num_frames: int = 50,
    fps: int = 5,
) -> dict:
    """Visualize a single clip with many frames for detailed review.

    Args:
        clip_id: NVIDIA clip ID to visualize.
        num_frames: Number of frames to extract (max ~50 for 20s clip).
        fps: Video playback FPS.

    Returns:
        dict with output paths.
    """
    import physical_ai_av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import subprocess

    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "error", "message": "HF_TOKEN not found"}

    print(f"Visualizing clip: {clip_id}")
    print("Initializing NVIDIA PhysicalAI-AV interface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(token=token)

    output_dir = Path(OUTPUTS_DIR) / "single_clip_viz" / clip_id[:20]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load clip data
        camera_feature = avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
        print("Loading video...")
        video = avdi.get_clip_feature(clip_id, camera_feature, maybe_stream=True)
        print("Loading egomotion...")
        egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)

        # Use egomotion as interpolator
        if isinstance(egomotion, physical_ai_av.utils.interpolation.Interpolator):
            interpolator = egomotion
        else:
            interpolator = physical_ai_av.utils.interpolation.Interpolator([egomotion])

        # Default intrinsics for 120° FOV
        K = np.array([
            [554, 0, 960],
            [0, 554, 540],
            [0, 0, 1]
        ], dtype=np.float32)

        # Sample timestamps across full clip (leave 3s at end for waypoint horizon)
        max_ts_us = 17_000_000  # 17 seconds
        timestamps_us = np.linspace(0, max_ts_us, num_frames).astype(int)

        print(f"Decoding {num_frames} frames across {max_ts_us/1e6:.1f}s...")
        frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)

        saved_frames = []
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        cam_height = 1.5

        for i, (frame, ts_us) in enumerate(zip(frames, actual_ts)):
            current_state = interpolator(ts_us)
            current_pose = current_state.pose

            # Get rotation matrix
            if hasattr(current_pose, 'rotation'):
                try:
                    if hasattr(current_pose.rotation, 'as_matrix'):
                        R_current = current_pose.rotation.as_matrix()
                    elif hasattr(current_pose.rotation, 'R'):
                        R_current = current_pose.rotation.R
                    else:
                        R_current = np.array(current_pose.rotation).reshape(3, 3)
                except Exception:
                    R_current = np.eye(3)
            else:
                R_current = np.eye(3)

            current_trans = np.array(current_pose.translation)

            # Compute waypoints
            waypoints = []
            for j in range(1, 12):
                future_us = ts_us + int(j * 0.25 * 1_000_000)
                try:
                    future_state = interpolator(future_us)
                    future_pose = future_state.pose
                    future_trans = np.array(future_pose.translation)
                    diff = future_trans - current_trans
                    relative_pos = R_current.T @ diff
                    waypoints.append([float(relative_pos[0]), float(relative_pos[1])])
                except Exception:
                    if waypoints:
                        waypoints.append(waypoints[-1])
                    else:
                        waypoints.append([0.0, 0.0])

            waypoints = np.array(waypoints, dtype=np.float32)

            # Get speed
            try:
                speed_mps = float(np.linalg.norm(current_state.velocity[:2]))
            except Exception:
                speed_mps = 0.0

            # Create visualization
            h, w = frame.shape[:2]
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)

            # Project waypoints
            pixels = []
            for x_fwd, y_left in waypoints:
                if x_fwd > 0.5:
                    cam_x = -y_left
                    cam_y = cam_height
                    cam_z = x_fwd
                    u = fx * cam_x / cam_z + cx
                    v = fy * cam_y / cam_z + cy
                    if 0 <= u < w and 0 <= v < h:
                        pixels.append((int(u), int(v)))

            # Draw waypoints
            for j, (u, v) in enumerate(pixels):
                progress = j / max(len(waypoints) - 1, 1)
                r = int(255 * progress)
                g = int(255 * (1 - progress))
                radius = 8
                draw.ellipse(
                    (u - radius, v - radius, u + radius, v + radius),
                    fill=(r, g, 0), outline=(255, 255, 255), width=2,
                )
                if j > 0:
                    prev_u, prev_v = pixels[j - 1]
                    draw.line([(prev_u, prev_v), (u, v)], fill=(r, g, 0), width=3)

            # Add info overlay
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except Exception:
                font = ImageFont.load_default()

            pil_img = pil_img.convert("RGBA")
            overlay = Image.new("RGBA", (w, 80), (0, 0, 0, 200))
            pil_img.paste(overlay, (0, 0), overlay)
            pil_img = pil_img.convert("RGB")
            draw = ImageDraw.Draw(pil_img)

            info_text = (
                f"Clip: {clip_id[:30]}...\n"
                f"Frame {i+1}/{num_frames} | Time: {ts_us/1e6:.2f}s | "
                f"Speed: {speed_mps:.1f} m/s ({speed_mps*2.237:.1f} mph) | "
                f"Waypoints: {len(pixels)}/11"
            )
            draw.text((10, 8), info_text, fill=(255, 255, 255), font=font)

            # Save frame
            frame_path = output_dir / f"frame_{i:04d}.png"
            pil_img.save(frame_path)
            saved_frames.append(str(frame_path))

            if i % 10 == 0:
                print(f"  Frame {i+1}/{num_frames}: speed={speed_mps:.1f} m/s, waypoints={len(pixels)}/11")

        # Create video with ffmpeg
        video_path = output_dir / "full_clip.mp4"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(output_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(video_path),
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True)

        print(f"\nVisualization complete!")
        print(f"Output: {output_dir}")
        print(f"Video: {video_path} ({num_frames} frames @ {fps} fps = {num_frames/fps:.1f}s)")

        output_volume.commit()

        return {
            "status": "success",
            "clip_id": clip_id,
            "output_dir": str(output_dir),
            "video_path": str(video_path),
            "num_frames": num_frames,
            "duration_s": num_frames / fps,
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "clip_id": clip_id,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


@app.function(
    image=training_image,
    volumes=VOLUMES,
    timeout=60 * 60,  # 1 hour for downloading
    cpu=4,
    memory=32 * 1024,  # More memory for video
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_waypoint_visualization(
    clip_id: str | None = None,
    num_frames: int = 20,
) -> dict:
    """Quick test: extract a few frames from one clip and visualize waypoints.

    If no clip_id is provided, will try to get the first available clip.

    Args:
        clip_id: Specific clip ID to test, or None to auto-select.
        num_frames: Number of frames to visualize.

    Returns:
        dict with output paths and status.
    """
    import physical_ai_av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "error", "message": "HF_TOKEN not found"}

    print("Initializing NVIDIA PhysicalAI-AV interface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(token=token)

    # Get a clip ID if not provided
    if clip_id is None:
        print("No clip_id provided, fetching first valid clip...")
        try:
            # Use clip_index to get valid clips
            clip_index = avdi.clip_index
            valid_clips = clip_index[clip_index['clip_is_valid'] == True].index.tolist()
            if valid_clips:
                clip_id = valid_clips[0]
                print(f"Selected clip: {clip_id}")
            else:
                clip_id = clip_index.index[0]
                print(f"No valid clips found, using first clip: {clip_id}")
        except Exception as e:
            import traceback
            print(f"Could not get clip: {e}")
            print(traceback.format_exc())
            return {"status": "error", "message": f"Could not list clips: {e}"}

    if clip_id is None:
        return {"status": "error", "message": "No clips available"}

    print(f"Using clip: {clip_id}")

    # Create output directory
    output_dir = Path(OUTPUTS_DIR) / "waypoint_test" / clip_id[:20]
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load clip data
        print("Loading video data...")
        print(f"Available camera features: {dir(avdi.features.CAMERA)}")
        try:
            camera_feature = avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
            print(f"Camera feature: {camera_feature}")
        except AttributeError as e:
            print(f"CAMERA_FRONT_WIDE_120FOV not found, trying alternatives...")
            print(f"Camera options: {[x for x in dir(avdi.features.CAMERA) if not x.startswith('_')]}")
            # Try to get first available camera
            camera_attrs = [x for x in dir(avdi.features.CAMERA) if not x.startswith('_') and 'FRONT' in x]
            if camera_attrs:
                camera_feature = getattr(avdi.features.CAMERA, camera_attrs[0])
                print(f"Using camera: {camera_attrs[0]}")
            else:
                raise ValueError("No front camera found")

        print(f"Calling get_clip_feature for video...")
        print("This may take a minute to stream from HuggingFace...")
        import sys
        sys.stdout.flush()
        try:
            # Use maybe_stream=True to stream from HF Hub without full download
            video = avdi.get_clip_feature(clip_id, camera_feature, maybe_stream=True)
            print(f"Video loaded: {type(video)}")
        except Exception as e:
            import traceback
            print(f"ERROR loading video: {e}")
            print(traceback.format_exc())
            return {"status": "error", "message": f"Failed to load video: {e}"}

        print("Loading egomotion data...")
        print(f"Available label features: {[x for x in dir(avdi.features.LABELS) if not x.startswith('_')]}")
        try:
            egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
            print(f"Egomotion loaded: {type(egomotion)}")
        except Exception as e:
            import traceback
            print(f"ERROR loading egomotion: {e}")
            print(traceback.format_exc())
            return {"status": "error", "message": f"Failed to load egomotion: {e}"}

        print("Loading calibration...")
        try:
            intrinsics_data = avdi.get_clip_feature(
                clip_id, avdi.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True
            )
            print(f"Intrinsics data type: {type(intrinsics_data)}")
            print(f"Intrinsics attributes: {[x for x in dir(intrinsics_data) if not x.startswith('_')]}")

            # Explore camera_models attribute
            if hasattr(intrinsics_data, 'camera_models'):
                camera_models = intrinsics_data.camera_models
                print(f"camera_models type: {type(camera_models)}")
                print(f"camera_models: {camera_models}")
                # Try to extract intrinsics from camera_models
                if isinstance(camera_models, dict):
                    for cam_name, model in camera_models.items():
                        print(f"  {cam_name}: {type(model)}, attrs: {[x for x in dir(model) if not x.startswith('_')][:10]}")
                        if 'front_wide' in cam_name.lower():
                            if hasattr(model, 'fx'):
                                K = np.array([
                                    [model.fx, 0, model.cx],
                                    [0, model.fy, model.cy],
                                    [0, 0, 1]
                                ], dtype=np.float32)
                                print(f"Found intrinsics from {cam_name}: fx={model.fx}, fy={model.fy}")
                                break
                            elif hasattr(model, 'K'):
                                K = np.array(model.K, dtype=np.float32)
                                print(f"Found K matrix from {cam_name}")
                                break

            # If we still don't have K, use defaults for 120° FOV
            if 'K' not in dir() or K is None:
                # 120° FOV on 1920x1080: fx = 1920 / (2 * tan(60°)) ≈ 554
                K = np.array([
                    [554, 0, 960],
                    [0, 554, 540],
                    [0, 0, 1]
                ], dtype=np.float32)
                print(f"Using default 120° FOV intrinsics")

            print(f"Final K: {K}")
        except Exception as e:
            import traceback
            print(f"ERROR loading calibration: {e}")
            print(traceback.format_exc())
            K = np.array([
                [554, 0, 960],
                [0, 554, 540],
                [0, 0, 1]
            ], dtype=np.float32)
            print(f"Using default intrinsics: {K}")

        # egomotion is already an Interpolator in newer API versions
        if isinstance(egomotion, physical_ai_av.utils.interpolation.Interpolator):
            interpolator = egomotion
            print("Egomotion is already an Interpolator")
        else:
            interpolator = physical_ai_av.utils.interpolation.Interpolator([egomotion])
            print("Created Interpolator from egomotion data")

        # Sample timestamps - leave 3s at end for waypoint horizon
        # Clips are ~20s, so sample from 0-14s to ensure full waypoint coverage
        max_ts_us = 14_000_000  # 14 seconds (leaves 6s for 2.75s waypoint horizon + buffer)
        timestamps_us = np.linspace(0, max_ts_us, num_frames).astype(int)
        print(f"Sampling {num_frames} timestamps from 0 to {max_ts_us/1e6:.1f}s")

        print(f"Decoding {num_frames} frames from video...")
        sys.stdout.flush()
        frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)

        print("Generating visualizations with waypoints...")
        saved_frames = []

        for i, (frame, ts_us) in enumerate(zip(frames, actual_ts)):
            # Get current pose
            current_state = interpolator(ts_us)
            current_pose = current_state.pose

            # Compute waypoints (11 points, 0.25s spacing)
            waypoints = []

            # Debug: print interpolator info on first frame
            if i == 0:
                print(f"Interpolator type: {type(interpolator)}")
                print(f"Current state: {type(current_state)}")
                print(f"Current pose: {type(current_pose)}")
                print(f"Current pose attrs: {[x for x in dir(current_pose) if not x.startswith('_')][:10]}")
                if hasattr(current_pose, 'translation'):
                    print(f"Current translation: {current_pose.translation}")
                if hasattr(current_pose, 'rotation'):
                    print(f"Current rotation: {current_pose.rotation}")

            for j in range(1, 12):
                future_us = ts_us + int(j * 0.25 * 1_000_000)
                try:
                    future_state = interpolator(future_us)
                    future_pose = future_state.pose

                    if i == 0 and j == 1:
                        print(f"Future pose (j=1): translation={future_pose.translation}")

                    # Get relative position: transform future to current frame
                    if hasattr(current_pose, 'inverse'):
                        relative_pos = current_pose.inverse().apply(future_pose.translation)
                    else:
                        # Manual calculation if inverse() not available
                        diff = future_pose.translation - current_pose.translation
                        # Rotate by inverse of current rotation
                        # For now, just use the diff directly (assuming small rotation)
                        relative_pos = diff

                    if i == 0 and j <= 3:
                        print(f"  wp{j}: relative_pos={relative_pos}")

                    waypoints.append([float(relative_pos[0]), float(relative_pos[1])])
                except Exception as e:
                    if i == 0:
                        print(f"  wp{j} error: {e}")
                    if waypoints:
                        waypoints.append(waypoints[-1])
                    else:
                        waypoints.append([0.0, 0.0])

            waypoints = np.array(waypoints, dtype=np.float32)

            # Get speed
            try:
                velocity = current_state.velocity
                speed_mps = float(np.linalg.norm(velocity[:2]))
            except Exception:
                speed_mps = 0.0

            # Create visualization
            h, w = frame.shape[:2]
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)

            # Project waypoints to image
            # Simple pinhole projection
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            cam_height = 1.5  # Approximate camera height above ground

            # Debug: print first frame's waypoints
            if i == 0:
                print(f"Frame 0 waypoints (x_fwd, y_left):")
                for j, (x_fwd, y_left) in enumerate(waypoints):
                    print(f"  wp{j}: x_fwd={x_fwd:.2f}m, y_left={y_left:.2f}m")
                print(f"Camera: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
                print(f"Image size: {w}x{h}")

            pixels = []
            for x_fwd, y_left in waypoints:
                if x_fwd > 0.5:  # In front of camera (at least 0.5m)
                    # Project ground point to image
                    # Camera frame: X_right, Y_down, Z_forward
                    # Ego frame: X_forward, Y_left, Z_up
                    # Ground point in ego: (x_fwd, y_left, 0)
                    # Camera is at height cam_height looking forward
                    # In camera frame: X = -y_left, Y = cam_height, Z = x_fwd
                    cam_x = -y_left
                    cam_y = cam_height  # camera height above ground
                    cam_z = x_fwd

                    # Project: u = fx * cam_x / cam_z + cx
                    #          v = fy * cam_y / cam_z + cy
                    u = fx * cam_x / cam_z + cx
                    v = fy * cam_y / cam_z + cy

                    if i == 0 and len(pixels) < 3:
                        print(f"  Proj: ({x_fwd:.1f}, {y_left:.1f}) -> ({u:.0f}, {v:.0f})")

                    if 0 <= u < w and 0 <= v < h:
                        pixels.append((int(u), int(v)))

            # Draw waypoints with color gradient
            for j, (u, v) in enumerate(pixels):
                progress = j / max(len(waypoints) - 1, 1)
                r = int(255 * progress)
                g = int(255 * (1 - progress))
                radius = 8
                draw.ellipse(
                    (u - radius, v - radius, u + radius, v + radius),
                    fill=(r, g, 0),
                    outline=(255, 255, 255),
                    width=2,
                )
                # Connect with lines
                if j > 0:
                    prev_u, prev_v = pixels[j - 1]
                    draw.line([(prev_u, prev_v), (u, v)], fill=(r, g, 0), width=3)

            # Add info overlay
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20
                )
            except Exception:
                font = ImageFont.load_default()

            # Semi-transparent background - convert to RGBA first
            pil_img = pil_img.convert("RGBA")
            overlay = Image.new("RGBA", (w, 100), (0, 0, 0, 200))
            pil_img.paste(overlay, (0, 0), overlay)
            # Convert back to RGB for saving
            pil_img = pil_img.convert("RGB")

            info_text = (
                f"Clip: {clip_id[:30]}...\n"
                f"Frame {i+1}/{num_frames} | Time: {ts_us/1e6:.2f}s | "
                f"Speed: {speed_mps:.1f} m/s ({speed_mps*2.237:.1f} mph)\n"
                f"Waypoints: {len(pixels)}/11 visible | "
                f"Green=0.25s ahead, Red=2.75s ahead"
            )
            draw.text((10, 10), info_text, fill=(255, 255, 255), font=font)

            # Save frame
            frame_path = output_dir / f"frame_{i:04d}.png"
            pil_img.save(frame_path)
            saved_frames.append(str(frame_path))

            # Debug: print more info if few waypoints visible
            if len(pixels) < 5:
                print(f"  Frame {i+1}: speed={speed_mps:.1f} m/s, "
                      f"waypoints visible={len(pixels)}/11, "
                      f"t={ts_us/1e6:.2f}s")
                if len(waypoints) > 0:
                    print(f"    First wp: x={waypoints[0][0]:.1f}m, last wp: x={waypoints[-1][0]:.1f}m")
            else:
                print(f"  Saved frame {i+1}: speed={speed_mps:.1f} m/s, "
                      f"waypoints visible={len(pixels)}/11")

        # Create video using ffmpeg for better compatibility
        video_path = output_dir / "waypoint_test.mp4"
        import subprocess

        # Use ffmpeg for reliable video encoding
        ffmpeg_cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-framerate", "2",  # 2 fps for slow viewing
            "-i", str(output_dir / "frame_%04d.png"),
            "-c:v", "libx264",  # H.264 codec for compatibility
            "-pix_fmt", "yuv420p",  # Required for most players
            "-crf", "18",  # High quality
            str(video_path),
        ]
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ffmpeg warning: {result.stderr}")
                # Fallback to OpenCV
                import cv2
                first = cv2.imread(saved_frames[0])
                h, w = first.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, 2.0, (w, h))
                for fp in saved_frames:
                    writer.write(cv2.imread(fp))
                writer.release()
        except FileNotFoundError:
            # ffmpeg not available, use OpenCV
            import cv2
            first = cv2.imread(saved_frames[0])
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 2.0, (w, h))
            for fp in saved_frames:
                writer.write(cv2.imread(fp))
            writer.release()

        print(f"\nVisualization complete!")
        print(f"Output directory: {output_dir}")
        print(f"Video: {video_path}")

        output_volume.commit()

        return {
            "status": "success",
            "clip_id": clip_id,
            "output_dir": str(output_dir),
            "video_path": str(video_path),
            "num_frames": len(saved_frames),
            "message": "Waypoint visualization created successfully",
        }

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "clip_id": clip_id,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


@app.function(
    image=training_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 2,  # 2 hours for multiple clips
    cpu=4,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_multiple_clips(
    num_clips: int = 3,
    num_frames_per_clip: int = 10,
    daytime_only: bool = True,
    min_speed_mps: float = 3.0,
) -> dict:
    """Test waypoint visualization on multiple random clips.

    Args:
        num_clips: Number of random clips to test.
        num_frames_per_clip: Frames per clip.
        daytime_only: If True, only use daytime clips.
        min_speed_mps: Minimum average speed to filter clips (avoids stopped cars).

    Returns:
        Summary with paths to all outputs.
    """
    import physical_ai_av
    import numpy as np
    import random
    from PIL import Image, ImageDraw, ImageFont
    import subprocess

    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "error", "message": "HF_TOKEN not found"}

    print("Initializing NVIDIA PhysicalAI-AV interface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(token=token)

    # Get filtered clip IDs
    try:
        clip_index = avdi.clip_index
        print(f"Total clips in index: {len(clip_index)}")
        print(f"Clip index columns: {list(clip_index.columns)}")

        # Start with valid clips
        valid_mask = clip_index['clip_is_valid'] == True
        print(f"Valid clips: {valid_mask.sum()}")

        # Filter for daytime if requested
        if daytime_only and 'time_of_day' in clip_index.columns:
            daytime_mask = clip_index['time_of_day'] == 'day'
            valid_mask = valid_mask & daytime_mask
            print(f"Daytime clips: {valid_mask.sum()}")
        elif daytime_only:
            print("'time_of_day' not in clip_index, checking data_collection metadata...")
            try:
                if hasattr(avdi, 'data_collection'):
                    dc = avdi.data_collection
                    print(f"data_collection columns: {list(dc.columns)}")

                    # Use hour_of_day to filter for daytime (6 AM - 6 PM)
                    if 'hour_of_day' in dc.columns:
                        print(f"hour_of_day range: {dc['hour_of_day'].min()} - {dc['hour_of_day'].max()}")
                        # Daytime = hours 6-18 (6 AM to 6 PM)
                        daytime_mask_dc = (dc['hour_of_day'] >= 6) & (dc['hour_of_day'] <= 18)
                        daytime_clips = set(dc[daytime_mask_dc].index.tolist())
                        print(f"Found {len(daytime_clips)} daytime clips (hours 6-18)")

                        # Intersect with valid clips
                        valid_clip_set = set(clip_index[valid_mask].index.tolist())
                        filtered_clips = list(valid_clip_set & daytime_clips)
                        print(f"After intersection with valid: {len(filtered_clips)} clips")

                        if filtered_clips:
                            valid_mask = clip_index.index.isin(filtered_clips)
                    else:
                        print("No hour_of_day column found, using all clips")
            except Exception as e:
                print(f"Could not filter by daytime: {e}")
                import traceback
                traceback.print_exc()

        # Get filtered clips
        filtered_clips = clip_index[valid_mask].index.tolist()

        random.seed(42)
        random.shuffle(filtered_clips)
        selected_clips = filtered_clips[:num_clips]
        print(f"Selected {len(selected_clips)} clips: {selected_clips}")
    except Exception as e:
        return {"status": "error", "message": f"Could not get clips: {e}"}

    results = []
    base_output_dir = Path(OUTPUTS_DIR) / "multi_clip_test"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for clip_idx, clip_id in enumerate(selected_clips):
        print(f"\n{'='*60}")
        print(f"Processing clip {clip_idx+1}/{num_clips}: {clip_id}")
        print(f"{'='*60}")

        clip_output_dir = base_output_dir / f"clip_{clip_idx}_{clip_id[:15]}"
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load clip data
            camera_feature = avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
            video = avdi.get_clip_feature(clip_id, camera_feature, maybe_stream=True)
            egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)

            # Use egomotion as interpolator
            if isinstance(egomotion, physical_ai_av.utils.interpolation.Interpolator):
                interpolator = egomotion
            else:
                interpolator = physical_ai_av.utils.interpolation.Interpolator([egomotion])

            # Default intrinsics for 120° FOV
            K = np.array([
                [554, 0, 960],
                [0, 554, 540],
                [0, 0, 1]
            ], dtype=np.float32)

            # Sample timestamps
            max_ts_us = 14_000_000
            timestamps_us = np.linspace(0, max_ts_us, num_frames_per_clip).astype(int)

            print(f"Decoding {num_frames_per_clip} frames...")
            frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)

            saved_frames = []
            frame_stats = []

            for i, (frame, ts_us) in enumerate(zip(frames, actual_ts)):
                current_state = interpolator(ts_us)
                current_pose = current_state.pose

                # Compute waypoints using rotation matrix for proper transform
                waypoints = []

                # Get current rotation as matrix
                if hasattr(current_pose, 'rotation'):
                    try:
                        if hasattr(current_pose.rotation, 'as_matrix'):
                            R_current = current_pose.rotation.as_matrix()
                        elif hasattr(current_pose.rotation, 'R'):
                            R_current = current_pose.rotation.R
                        else:
                            R_current = np.array(current_pose.rotation).reshape(3, 3)
                    except Exception:
                        R_current = np.eye(3)
                else:
                    R_current = np.eye(3)

                current_trans = np.array(current_pose.translation)

                # Debug first frame
                if i == 0:
                    print(f"  Frame 0: current_trans={current_trans}")
                    print(f"  R_current type: {type(R_current)}")

                for j in range(1, 12):
                    future_us = ts_us + int(j * 0.25 * 1_000_000)
                    try:
                        future_state = interpolator(future_us)
                        future_pose = future_state.pose
                        future_trans = np.array(future_pose.translation)

                        # Compute relative position: rotate diff into current frame
                        diff = future_trans - current_trans
                        # R_current^T @ diff gives diff in current frame
                        relative_pos = R_current.T @ diff

                        waypoints.append([float(relative_pos[0]), float(relative_pos[1])])

                        # Debug first frame
                        if i == 0 and j <= 3:
                            print(f"    wp{j}: x_fwd={relative_pos[0]:.2f}m, y_left={relative_pos[1]:.2f}m, diff={diff}")
                    except Exception as e:
                        if i == 0 and j == 1:
                            print(f"    wp{j} error: {e}")
                            import traceback
                            traceback.print_exc()
                        if waypoints:
                            waypoints.append(waypoints[-1])
                        else:
                            waypoints.append([0.0, 0.0])

                waypoints = np.array(waypoints, dtype=np.float32)

                # Get speed
                try:
                    speed_mps = float(np.linalg.norm(current_state.velocity[:2]))
                except Exception:
                    speed_mps = 0.0

                # Create visualization
                h, w = frame.shape[:2]
                pil_img = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_img)

                # Project waypoints
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                cam_height = 1.5

                pixels = []
                for wp_idx, (x_fwd, y_left) in enumerate(waypoints):
                    if x_fwd > 0.5:  # In front of camera
                        cam_x = -y_left  # Camera X = -ego Y_left (right is positive in camera)
                        cam_y = cam_height  # Camera Y = height above ground (down is positive)
                        cam_z = x_fwd  # Camera Z = ego X_forward
                        u = fx * cam_x / cam_z + cx
                        v = fy * cam_y / cam_z + cy
                        if 0 <= u < w and 0 <= v < h:
                            pixels.append((int(u), int(v)))
                        elif i == 0 and wp_idx < 3:
                            print(f"    wp{wp_idx} out of bounds: u={u:.0f}, v={v:.0f}")
                    elif i == 0 and wp_idx < 3:
                        print(f"    wp{wp_idx} behind camera: x_fwd={x_fwd:.2f}m")

                # Draw waypoints
                for j, (u, v) in enumerate(pixels):
                    progress = j / max(len(waypoints) - 1, 1)
                    r = int(255 * progress)
                    g = int(255 * (1 - progress))
                    radius = 8
                    draw.ellipse(
                        (u - radius, v - radius, u + radius, v + radius),
                        fill=(r, g, 0), outline=(255, 255, 255), width=2,
                    )
                    if j > 0:
                        prev_u, prev_v = pixels[j - 1]
                        draw.line([(prev_u, prev_v), (u, v)], fill=(r, g, 0), width=3)

                # Add warning if speed is very low
                if speed_mps < 1.0:
                    draw.text((w//2 - 150, h//2), "LOW SPEED - LIMITED WAYPOINTS",
                              fill=(255, 100, 100), font=font)

                # Add info overlay
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
                except Exception:
                    font = ImageFont.load_default()

                pil_img = pil_img.convert("RGBA")
                overlay = Image.new("RGBA", (w, 90), (0, 0, 0, 200))
                pil_img.paste(overlay, (0, 0), overlay)
                pil_img = pil_img.convert("RGB")
                draw = ImageDraw.Draw(pil_img)

                info_text = (
                    f"Clip {clip_idx+1}/{num_clips}: {clip_id[:25]}...\n"
                    f"Frame {i+1}/{num_frames_per_clip} | Time: {ts_us/1e6:.2f}s | "
                    f"Speed: {speed_mps:.1f} m/s ({speed_mps*2.237:.1f} mph)\n"
                    f"Waypoints: {len(pixels)}/11 visible"
                )
                draw.text((10, 8), info_text, fill=(255, 255, 255), font=font)

                # Save frame
                frame_path = clip_output_dir / f"frame_{i:04d}.png"
                pil_img.save(frame_path)
                saved_frames.append(str(frame_path))
                frame_stats.append({
                    "frame": i,
                    "time_s": ts_us/1e6,
                    "speed_mps": speed_mps,
                    "waypoints_visible": len(pixels),
                    "first_wp_dist": float(waypoints[0][0]) if len(waypoints) > 0 else 0,
                    "last_wp_dist": float(waypoints[-1][0]) if len(waypoints) > 0 else 0,
                })

                print(f"  Frame {i+1}: speed={speed_mps:.1f} m/s, "
                      f"waypoints={len(pixels)}/11, t={ts_us/1e6:.2f}s")

            # Create video with ffmpeg
            video_path = clip_output_dir / "waypoints.mp4"
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", "3",
                "-i", str(clip_output_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                str(video_path),
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True)

            results.append({
                "clip_id": clip_id,
                "status": "success",
                "output_dir": str(clip_output_dir),
                "video_path": str(video_path),
                "num_frames": len(saved_frames),
                "frame_stats": frame_stats,
            })

            print(f"✓ Clip {clip_idx+1} complete: {clip_output_dir}")

        except Exception as e:
            import traceback
            print(f"✗ Clip {clip_idx+1} failed: {e}")
            results.append({
                "clip_id": clip_id,
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            })

    output_volume.commit()

    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{num_clips} clips processed successfully")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*60}")

    return {
        "status": "success" if success_count == num_clips else "partial",
        "output_dir": str(base_output_dir),
        "clips_processed": success_count,
        "clips_total": num_clips,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    action: str = "prepare",
    scale: str = "tiny",
    config: str = "/app/config/nvidia_finetune.yaml",
    wandb_project: str = "simlingo-nvidia-finetune",
):
    """Main entrypoint for training pipeline.

    Actions:
        prepare: Download base model and prepare data
        train: Run training
        evaluate: Evaluate model

    Examples:
        modal run modal_training.py --action prepare --scale tiny
        modal run modal_training.py --action train
        modal run modal_training.py --action evaluate
    """
    if action == "prepare":
        print("Preparing base model...")
        prepare_base_model.remote()
        print("Preparing NVIDIA data...")
        prepare_nvidia_data.remote(scale=scale)
    elif action == "train":
        train.remote(config_path=config, wandb_project=wandb_project)
    elif action == "evaluate":
        evaluate.remote()
    else:
        print(f"Unknown action: {action}")
