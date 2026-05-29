"""SimLingo LoRA fine-tuning trainer for NVIDIA PhysicalAI-AV data.

This module implements the training loop for fine-tuning SimLingo on real-world
driving data using LoRA (Low-Rank Adaptation) to efficiently update model weights.

Architecture:
    - Base model: SimLingo (InternVL2-1B backbone)
    - Fine-tuning: LoRA on attention layers
    - Output: Waypoints + Route predictions

Training:
    - Waypoint regression loss (Smooth L1)
    - Route regression loss (Smooth L1)
    - Optional language loss (cross-entropy for meta-actions)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# PROF_APPLIED_v1
def _prof_mark(name, state):
    """Profile helper: cuda-sync + record elapsed since last mark."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    state[name] = now - state["_last"]
    state["_last"] = now


def _prof_log(path_prefix, step, is_main, prof):
    """Print + wandb.log profile dict if rank-0. No-op otherwise."""
    if not is_main:
        return
    if step % 5 != 0:
        return
    prof = {k: v for k, v in prof.items() if k != "_last"}
    msg = " ".join(f"{k}={v*1000:.0f}ms" for k, v in prof.items())
    print(f"[PROF {path_prefix} step={step}] {msg}", flush=True)
    try:
        import wandb as _wandb
        _wandb.log({f"prof_{path_prefix}/{k}_ms": v * 1000 for k, v in prof.items()},
                   step=step)
    except Exception:
        pass

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Paths
SIMLINGO_REPO_DIR = "/opt/simlingo"
CACHE_DIR = "/cache"
HF_CKPT_FILE = "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
HF_HYDRA_CONFIG_FILE = "simlingo/.hydra/config.yaml"


# ---------------------------------------------------------------------------
# Rule-based meta-action labeling (Plan B: SteerVLA-style prompt conditioning)
# ---------------------------------------------------------------------------
#
# SteerVLA disambiguates multimodal futures by conditioning the low-level
# policy on a textual meta-action produced by a high-level VLM. Rather than
# training a high-level model from scratch, we derive a meta-action label
# directly from the GT future trajectory at train time. This gives the policy
# a free disambiguating signal in its prompt at zero labeling cost and lets
# Smooth L1 regression behave well (the conditional distribution is closer to
# unimodal once the action category is fixed).
#
# Waypoints are in ego frame: x_forward, y_left (matches NVIDIA + SimLingo).

def waypoints_to_meta_action(
    waypoints: np.ndarray,
    current_speed_mps: float = 0.0,
) -> str:
    """Classify driving behavior from future ego-frame waypoints.

    Uses the terminal waypoint's heading angle relative to the forward axis
    as the primary discriminant. Heading change is more robust than raw
    lateral offset because it is approximately speed-invariant.

    Args:
        waypoints: [N, 2] in ego frame (x_forward, y_left). N=11 typical.
        current_speed_mps: Current speed in m/s, used for stop detection.

    Returns:
        Short imperative description suitable for prompt injection.
    """
    if waypoints is None or len(waypoints) == 0:
        return "continuing straight"

    end = waypoints[-1]
    forward = float(end[0])
    lateral = float(end[1])

    # Stop / hold detection — minimal forward progress over the horizon.
    # Horizon is ~2.75s, so forward < 1m means the vehicle is essentially halting.
    if forward < 1.0:
        if current_speed_mps < 1.0:
            return "remaining stopped"
        return "coming to a stop"

    # Heading change at the end of the horizon (degrees).
    # Positive = left turn, negative = right turn (because y is y_left).
    heading_deg = float(np.degrees(np.arctan2(lateral, forward)))

    if heading_deg > 20.0:
        return "turning left"
    if heading_deg < -20.0:
        return "turning right"
    if heading_deg > 5.0:
        return "drifting left"
    if heading_deg < -5.0:
        return "drifting right"
    return "continuing straight"


# ---------------------------------------------------------------------------
# Mixture Density Network (MDN) Head for Multimodal Waypoint Prediction
# ---------------------------------------------------------------------------

class MDNHead(nn.Module):
    """Mixture Density Network head for multimodal waypoint prediction.

    Instead of predicting a single (x, y) coordinate per waypoint, this predicts
    a mixture of Gaussians, allowing the model to express uncertainty and
    multimodal distributions (e.g., turn left OR turn right at intersection).

    For each waypoint, outputs:
        - pi: mixture weights (n_mixtures,) - which mode is most likely
        - mu: means (n_mixtures, 2) - center of each Gaussian
        - sigma: std devs (n_mixtures, 2) - spread of each Gaussian
    """

    def __init__(
        self,
        input_dim: int,
        n_waypoints: int = 11,
        n_mixtures: int = 5,
        min_sigma: float = 0.01,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.n_mixtures = n_mixtures
        self.min_sigma = min_sigma

        # Output dimensions per waypoint:
        # - n_mixtures weights (pi)
        # - n_mixtures * 2 means (mu_x, mu_y for each mixture)
        # - n_mixtures * 2 sigmas (sigma_x, sigma_y for each mixture)
        output_per_wp = n_mixtures + n_mixtures * 2 + n_mixtures * 2

        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_waypoints * output_per_wp),
        )

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: Model features [batch, hidden_dim]

        Returns:
            dict with:
                pi: [batch, n_waypoints, n_mixtures] - mixture weights (softmax)
                mu: [batch, n_waypoints, n_mixtures, 2] - means
                sigma: [batch, n_waypoints, n_mixtures, 2] - std devs (positive)
        """
        batch_size = features.shape[0]
        out = self.head(features)

        # Reshape to [batch, n_waypoints, params_per_wp]
        out = out.view(batch_size, self.n_waypoints, -1)

        # Split into pi, mu, sigma
        n_mix = self.n_mixtures
        pi_logits = out[..., :n_mix]  # [B, n_wp, n_mix]
        mu = out[..., n_mix:n_mix + n_mix * 2]  # [B, n_wp, n_mix * 2]
        sigma_raw = out[..., n_mix + n_mix * 2:]  # [B, n_wp, n_mix * 2]

        # Apply activations
        pi = F.softmax(pi_logits, dim=-1)  # weights sum to 1
        mu = mu.view(batch_size, self.n_waypoints, n_mix, 2)
        sigma = F.softplus(sigma_raw).view(batch_size, self.n_waypoints, n_mix, 2)
        sigma = sigma + self.min_sigma  # ensure positive

        return {"pi": pi, "mu": mu, "sigma": sigma}

    def sample(self, mdn_output: dict, temperature: float = 1.0) -> torch.Tensor:
        """Sample waypoints from the mixture distribution.

        Args:
            mdn_output: Output from forward()
            temperature: Sampling temperature (higher = more random)

        Returns:
            waypoints: [batch, n_waypoints, 2]
        """
        pi = mdn_output["pi"]  # [B, n_wp, n_mix]
        mu = mdn_output["mu"]  # [B, n_wp, n_mix, 2]
        sigma = mdn_output["sigma"]  # [B, n_wp, n_mix, 2]

        batch_size, n_wp, n_mix = pi.shape

        # Sample mixture component for each waypoint
        if temperature != 1.0:
            pi = F.softmax(torch.log(pi + 1e-8) / temperature, dim=-1)

        # Sample component indices
        component_idx = torch.multinomial(
            pi.view(-1, n_mix), 1
        ).view(batch_size, n_wp, 1, 1).expand(-1, -1, -1, 2)

        # Gather means and sigmas for selected components
        selected_mu = torch.gather(mu, 2, component_idx).squeeze(2)  # [B, n_wp, 2]
        selected_sigma = torch.gather(sigma, 2, component_idx).squeeze(2)

        # Sample from Gaussian
        noise = torch.randn_like(selected_mu)
        waypoints = selected_mu + selected_sigma * noise * temperature

        return waypoints

    def get_mode(self, mdn_output: dict) -> torch.Tensor:
        """Get the most likely waypoint (mode of highest-weight mixture).

        Args:
            mdn_output: Output from forward()

        Returns:
            waypoints: [batch, n_waypoints, 2]
        """
        pi = mdn_output["pi"]  # [B, n_wp, n_mix]
        mu = mdn_output["mu"]  # [B, n_wp, n_mix, 2]

        # Get index of highest weight mixture for each waypoint
        max_idx = pi.argmax(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, 2)

        # Gather the means
        waypoints = torch.gather(mu, 2, max_idx).squeeze(2)  # [B, n_wp, 2]

        return waypoints


def mdn_loss(
    mdn_output: dict,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Negative log-likelihood loss for MDN.

    Args:
        mdn_output: dict with pi, mu, sigma from MDNHead
        target: Ground truth waypoints [batch, n_waypoints, 2]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: NLL loss
    """
    pi = mdn_output["pi"]  # [B, n_wp, n_mix]
    mu = mdn_output["mu"]  # [B, n_wp, n_mix, 2]
    sigma = mdn_output["sigma"]  # [B, n_wp, n_mix, 2]

    # Expand target for broadcasting: [B, n_wp, 1, 2]
    target = target.unsqueeze(2)

    # Compute log probability for each mixture component
    # log N(x | mu, sigma) = -0.5 * ((x - mu) / sigma)^2 - log(sigma) - 0.5 * log(2*pi)
    diff = (target - mu) / sigma  # [B, n_wp, n_mix, 2]
    log_prob_per_dim = -0.5 * diff ** 2 - torch.log(sigma) - 0.5 * np.log(2 * np.pi)
    log_prob_per_mix = log_prob_per_dim.sum(dim=-1)  # [B, n_wp, n_mix]

    # Weight by mixture probabilities and sum (log-sum-exp trick for numerical stability)
    log_pi = torch.log(pi + 1e-8)
    log_prob_weighted = log_pi + log_prob_per_mix  # [B, n_wp, n_mix]
    log_prob = torch.logsumexp(log_prob_weighted, dim=-1)  # [B, n_wp]

    # Negative log likelihood
    nll = -log_prob

    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class TrainingSample:
    """A single training sample from extracted NVIDIA data."""
    rgb_path: Path
    speed_mps: float
    target_points: np.ndarray  # [2, 2]
    waypoints: np.ndarray      # [11, 2]
    route: np.ndarray          # [20, 2]
    meta: dict


class NVIDIATrainingDataset(Dataset):
    """Dataset for training on extracted NVIDIA data."""

    def __init__(
        self,
        data_dir: str | Path,
        clip_ids: list[str] | None = None,
        transform=None,
        fov_deg: float = 120.0,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.fov_deg = fov_deg

        # Find all samples
        self.samples: list[TrainingSample] = []

        if clip_ids is None:
            # Use all clips
            clip_dirs = [d for d in self.data_dir.iterdir()
                        if d.is_dir() and (d / "rgb").exists()]
        else:
            clip_dirs = [self.data_dir / c for c in clip_ids
                        if (self.data_dir / c / "rgb").exists()]

        for clip_dir in clip_dirs:
            rgb_dir = clip_dir / "rgb"
            meas_dir = clip_dir / "measurements"

            for rgb_file in sorted(rgb_dir.glob("*.jpg")):
                frame_idx = int(rgb_file.stem)
                meas_file = meas_dir / f"{frame_idx:04d}.json.gz"

                if meas_file.exists():
                    self.samples.append(TrainingSample(
                        rgb_path=rgb_file,
                        speed_mps=0.0,  # Loaded lazily
                        target_points=np.zeros((2, 2)),
                        waypoints=np.zeros((11, 2)),
                        route=np.zeros((20, 2)),
                        meta={"meas_file": str(meas_file)},
                    ))

        print(f"Loaded {len(self.samples)} samples from {len(clip_dirs)} clips")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load measurement
        with gzip.open(sample.meta["meas_file"], "rt") as f:
            meas = json.load(f)

        # Load and process image
        image = Image.open(sample.rgb_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            # Default: resize to 448x448 and normalize
            image = image.resize((448, 448), Image.BILINEAR)
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            "image": image,
            "speed_mps": torch.tensor(meas["speed"], dtype=torch.float32),
            "target_points": torch.tensor(
                [meas["target_point"], meas["target_point_next"]],
                dtype=torch.float32,
            ),
            "waypoints": torch.tensor(meas["waypoints"], dtype=torch.float32),
            "route": torch.tensor(meas["route"], dtype=torch.float32),
            "rgb_path": str(sample.rgb_path),
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    Args:
        data_dir: Path to extracted data.
        batch_size: Batch size.
        val_split: Validation split ratio.
        num_workers: Number of data loading workers.
        seed: Random seed.

    Returns:
        (train_loader, val_loader)
    """
    data_dir = Path(data_dir)

    # Check for existing split file
    split_file = data_dir / "train_val_split.json"
    if split_file.exists():
        with open(split_file) as f:
            split = json.load(f)
        train_clips = split["train"]
        val_clips = split["val"]
    else:
        # Create split
        all_clips = [d.name for d in data_dir.iterdir()
                    if d.is_dir() and (d / "rgb").exists()]
        random.seed(seed)
        random.shuffle(all_clips)
        val_count = int(len(all_clips) * val_split)
        val_clips = all_clips[:val_count]
        train_clips = all_clips[val_count:]

    print(f"Train clips: {len(train_clips)}, Val clips: {len(val_clips)}")

    train_dataset = NVIDIATrainingDataset(data_dir, clip_ids=train_clips)
    val_dataset = NVIDIATrainingDataset(data_dir, clip_ids=val_clips)

    # Multi-worker only flags (PyTorch requires num_workers > 0)
    extra = {}
    if num_workers > 0:
        extra["prefetch_factor"] = prefetch_factor
        extra["persistent_workers"] = persistent_workers

    # Distributed sampler: shards the dataset across DDP ranks
    train_sampler = None
    val_sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed,
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        **extra,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        **extra,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class NVIDIASimLingoTrainer:
    """Trainer for SimLingo fine-tuning on NVIDIA data."""

    def __init__(
        self,
        config: dict,
        ckpt_dir: str,
        output_dir: str,
        hf_repo: str | None = None,
    ):
        self.config = config
        # FOV_APPLIED_v1
        self.fov_deg = float(config.get("data", {}).get("fov_deg", 120.0))
        self.ckpt_dir = Path(ckpt_dir)
        self.output_dir = Path(output_dir)
        self.hf_repo = hf_repo

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Distributed setup -------------------------------------------------
        # If torchrun was used, LOCAL_RANK / WORLD_SIZE are set in the env.
        # Otherwise we fall back to single-GPU mode.
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_distributed = self.world_size > 1
        self.is_main = self.rank == 0

        if self.is_distributed:
            import torch.distributed as dist
            # CRITICAL ORDERING: set_device(local_rank) MUST come before
            # init_process_group(backend="nccl"). NCCL otherwise grabs cuda:0 on
            # every rank during init and locks the context — subsequent
            # set_device(>=1) then fails with "CUDA-capable device(s) is/are
            # busy or unavailable" on PyTorch 2.7+ (stricter than 2.4).
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            if self.is_main:
                print(f"[DDP] world_size={self.world_size}, rank={self.rank}, "
                      f"local_rank={self.local_rank}, device={self.device}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {self.device}")

        # Load model
        self._setup_model()

        # Setup LoRA if enabled
        if config["lora"]["enabled"]:
            self._setup_lora()

        # Wrap model in DDP after LoRA injection so adapters are visible to DDP
        if self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            print(f"[rank{self.rank}] Wrapping model in DDP...", flush=True)
            # find_unused_parameters=True because only LoRA adapters have grads;
            # frozen base params would otherwise trigger DDP's "unused parameter" check.
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
            print(f"[rank{self.rank}] DDP wrap complete.", flush=True)
            # Keep an unwrapped handle for attribute access (.forward, custom methods)
            self._raw_model = self.model.module
        else:
            self._raw_model = self.model

        # Setup optimizer
        print(f"[rank{self.rank}] Setting up optimizer...", flush=True)
        self._setup_optimizer()
        print(f"[rank{self.rank}] Optimizer ready.", flush=True)

        # Setup dataloaders
        print(f"[rank{self.rank}] Setting up dataloaders...", flush=True)
        self._setup_data()
        print(f"[rank{self.rank}] Dataloaders ready (train batches={len(self.train_loader)}, val batches={len(self.val_loader)}).", flush=True)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        # Early-stop bookkeeping (val/loss patience counter).
        self._es_no_improve_count = 0
        self._es_should_stop = False

    def _setup_model(self):
        """Load and prepare SimLingo model."""
        sys.path.insert(0, SIMLINGO_REPO_DIR)

        import hydra
        from omegaconf import OmegaConf
        from transformers import AutoProcessor

        # Load Hydra config
        hydra_cfg_path = self.ckpt_dir / HF_HYDRA_CONFIG_FILE
        cfg = OmegaConf.load(hydra_cfg_path)
        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            cfg.model.vision_model.variant,
            trust_remote_code=True,
        )
        # Get tokenizer - may be part of processor or need separate loading
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            self.tokenizer = self.processor.tokenizer
        elif "tokenizer" in self.processor.__dict__:
            self.tokenizer = self.processor.tokenizer
        else:
            # Load tokenizer separately for InternVL2
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.vision_model.variant,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                self.tokenizer = None

        # Add special tokens if tokenizer is available AND they don't already exist
        # The SimLingo checkpoint has these tokens at specific IDs - don't re-add them
        if self.tokenizer is not None and hasattr(self.tokenizer, 'add_special_tokens'):
            special_tokens = [
                "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>",
                "<ORG_WAYPOINTS>", "<WAYPOINT_LAST>", "<ROUTE>",
                "<ROUTE_DIFF>", "<TARGET_POINT>",
            ]
            # Check if TARGET_POINT already exists
            existing_id = self.tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
            unk_id = self.tokenizer.unk_token_id
            print(f"[DEBUG] Before adding tokens: TARGET_POINT ID = {existing_id}, unk_id = {unk_id}")

            if existing_id == unk_id:
                # Token doesn't exist, need to add it
                print("[DEBUG] Adding special tokens to tokenizer...")
                self.tokenizer.add_special_tokens({
                    "additional_special_tokens": special_tokens
                })
                new_id = self.tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
                print(f"[DEBUG] After adding: TARGET_POINT ID = {new_id}")
            else:
                print(f"[DEBUG] Special tokens already exist (TARGET_POINT = {existing_id})")

            self.tokenizer.padding_side = "left"

        # Ensure InternVL2-1B pretrained symlink exists
        workdir = Path("/tmp/simlingo_work")
        workdir.mkdir(parents=True, exist_ok=True)
        pretrained = workdir / "pretrained"
        pretrained.mkdir(exist_ok=True)
        link = pretrained / "InternVL2-1B"
        src = Path(f"{CACHE_DIR}/hf/snapshots/InternVL2-1B")
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src)
        os.chdir(workdir)

        # Load model
        cache_dir = f"pretrained/{cfg.model.vision_model.variant.split('/')[1]}"

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            self.model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=self.processor,
                cache_dir=cache_dir,
                _recursive_=False,
            ).to(self.device)
        finally:
            torch.set_default_dtype(default_dtype)

        # Load pretrained weights
        ckpt_path = self.ckpt_dir / HF_CKPT_FILE
        print(f"Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        self.cfg = cfg
        self.use_global_img = bool(cfg.data_module.use_global_img)

        # Import utilities needed for building model inputs
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
        from simlingo_training.utils.internvl2_utils import (
            build_transform, dynamic_preprocess, get_custom_chat_template
        )
        from simlingo_training.utils.projection import (
            get_camera_extrinsics, get_camera_intrinsics
        )

        self.DrivingInput = DrivingInput
        self.LanguageLabel = LanguageLabel
        self.build_transform = build_transform
        self.dynamic_preprocess = dynamic_preprocess
        self.get_custom_chat_template = get_custom_chat_template
        self.get_camera_extrinsics = get_camera_extrinsics
        self.get_camera_intrinsics = get_camera_intrinsics
        self.image_transform = build_transform(input_size=448)

        # Get placeholder token ID - must match what appears in phrase_ids
        # The tokenizer's ID after add_special_tokens is what gets tokenized,
        # so we MUST use that same ID in placeholder_values
        if self.tokenizer is not None:
            self.placeholder_token_id = self.tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
            print(f"[DEBUG] Using TARGET_POINT token ID from tokenizer: {self.placeholder_token_id}")
        else:
            # Fallback - this shouldn't happen
            self.placeholder_token_id = 151648
            print(f"[WARNING] No tokenizer, using fallback ID: {self.placeholder_token_id}")

        # Setup MDN head if enabled
        self.mdn_head = None
        self.use_mdn = self.config.get("mdn", {}).get("enabled", False)
        if self.use_mdn:
            mdn_config = self.config["mdn"]
            print(f"[MDN] Initializing MDN head with {mdn_config['n_mixtures']} mixtures")
            self.mdn_head = MDNHead(
                input_dim=mdn_config.get("input_dim", 896),
                n_waypoints=self.config["model"].get("num_waypoints", 11),
                n_mixtures=mdn_config.get("n_mixtures", 5),
                min_sigma=mdn_config.get("min_sigma", 0.01),
            ).to(self.device)
            print(f"[MDN] MDN head parameters: {sum(p.numel() for p in self.mdn_head.parameters()):,}")

            # Register hook to capture features before waypoint head
            # SimLingo's waypoint prediction happens in the language model head
            # We need to capture the last hidden state
            self._setup_feature_extraction_hook()

    def _setup_feature_extraction_hook(self):
        """Register a forward hook that captures the LLM's final hidden state.

        Strategy:
          1. Locate the final `LayerNorm`/`RMSNorm` of the LLM by trying the
             well-known InternVL2 / Qwen / InternLM2 attribute paths in order.
          2. If none match, fall back to taking the LAST `nn.LayerNorm` /
             `RMSNorm` module reachable under any submodule named `llm` or
             `language_model`.
          3. The captured tensor (shape [B, seq, hidden]) is stored on the
             trainer as `self._last_features`. Reduction to [B, hidden] is
             deferred to `train_step` so the caller controls token selection.

        SimLingo's own waypoint heads read from `features[:, -len_driving:]`
        — the last `len_driving` positions of the LLM output correspond to
        the driving-input adaptor tokens. We mean-pool over the last
        `num_waypoints` positions at use time as a stable proxy for this
        (we don't have direct access to SimLingo's internal `len_driving`,
        but the driving adaptor emits one position per waypoint output).

        Note: the hook stores on the *trainer* (`self`), and `train_step`
        must read from `self._last_features` — NOT `self.model._last_features`.
        """
        self._last_features = None

        # Candidate attribute paths in priority order. These are the most
        # common locations for the final norm in InternVL2-based models.
        candidate_paths = [
            ("language_model", "model", "norm"),
            ("language_model", "model", "final_layernorm"),
            ("llm", "model", "norm"),
            ("llm", "model", "final_layernorm"),
            ("llm", "norm"),
            ("language_model", "norm"),
        ]

        llm_module = None
        chosen_path = None
        for path in candidate_paths:
            obj = self.model
            ok = True
            for attr in path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(obj, nn.Module):
                llm_module = obj
                chosen_path = ".".join(path)
                break

        # Fallback: walk named_modules and pick the deepest LayerNorm/RMSNorm
        # under a parent named 'llm' or 'language_model'.
        if llm_module is None:
            norm_candidates = []
            for name, module in self.model.named_modules():
                lname = name.lower()
                if ("llm" in lname or "language_model" in lname) and (
                    isinstance(module, nn.LayerNorm)
                    or type(module).__name__ in ("RMSNorm", "InternLM2RMSNorm", "Qwen2RMSNorm")
                ):
                    norm_candidates.append((name, module))
            if norm_candidates:
                # Use the last (deepest in the forward sequence) candidate
                chosen_path, llm_module = norm_candidates[-1]

        if llm_module is None:
            print("[MDN] WARNING: Could not locate LLM final norm for feature hook")
            print("[MDN] Available top-level model attributes:",
                  [n for n, _ in self.model.named_children()])
            return

        # Use a weak reference closure that writes to the trainer (self).
        trainer = self

        def capture_hook(_module, _input, output):
            feat = output[0] if isinstance(output, tuple) else output
            # Store the raw [B, seq, hidden] tensor; reduce at use time.
            trainer._last_features = feat

        llm_module.register_forward_hook(capture_hook)
        self._mdn_hook_path = chosen_path
        print(f"[MDN] Registered feature hook on '{chosen_path}' "
              f"({type(llm_module).__name__})")

    def _setup_lora(self):
        """Apply LoRA to model by injecting into target modules directly.

        We can't use get_peft_model because it wraps forward() in a way that
        doesn't work with DrivingModel's custom forward signature.
        """
        import re
        import torch.nn as nn

        target_modules = self.config["lora"]["target_modules"]
        r = self.config["lora"]["rank"]
        alpha = self.config["lora"]["alpha"]
        dropout = self.config["lora"]["dropout"]

        # First, print model structure to find modules
        print("\n[DEBUG] Looking for LoRA target modules...")
        print(f"  Target patterns: {target_modules}")

        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Find and wrap target modules with LoRA
        # Match any module whose name contains any of the target strings
        lora_layers = []
        linear_modules = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_modules.append(name)
                # Check if any target module pattern is in the name
                for target in target_modules:
                    if target in name:
                        # Create LoRA layer
                        in_features = module.in_features
                        out_features = module.out_features

                        # Add LoRA A and B matrices
                        lora_A = nn.Linear(in_features, r, bias=False)
                        lora_B = nn.Linear(r, out_features, bias=False)

                        # Initialize: A with Kaiming, B with zeros
                        nn.init.kaiming_uniform_(lora_A.weight, a=5 ** 0.5)
                        nn.init.zeros_(lora_B.weight)

                        # Move to device and set dtype
                        lora_A = lora_A.to(self.device).to(torch.bfloat16)
                        lora_B = lora_B.to(self.device).to(torch.bfloat16)

                        # Enable gradients for LoRA params
                        lora_A.weight.requires_grad = True
                        lora_B.weight.requires_grad = True

                        # Store reference
                        lora_layers.append((name, module, lora_A, lora_B))
                        break  # Only match once per module

        if len(lora_layers) == 0:
            print(f"  [WARNING] No LoRA target modules found!")
            print(f"  Available Linear modules (first 20): {linear_modules[:20]}")
            # Fall back to matching any attention-related Linear layers
            print("  Falling back to matching 'attn' and 'proj' in module names...")
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    if 'attn' in name.lower() and 'proj' in name.lower():
                        in_features = module.in_features
                        out_features = module.out_features

                        lora_A = nn.Linear(in_features, r, bias=False)
                        lora_B = nn.Linear(r, out_features, bias=False)
                        nn.init.kaiming_uniform_(lora_A.weight, a=5 ** 0.5)
                        nn.init.zeros_(lora_B.weight)
                        lora_A = lora_A.to(self.device).to(torch.bfloat16)
                        lora_B = lora_B.to(self.device).to(torch.bfloat16)
                        lora_A.weight.requires_grad = True
                        lora_B.weight.requires_grad = True
                        lora_layers.append((name, module, lora_A, lora_B))

        print(f"  Found {len(lora_layers)} modules to apply LoRA to")
        if lora_layers:
            print(f"  First few: {[l[0] for l in lora_layers[:5]]}")

        # Instead of wrapping, we'll use hooks to add LoRA outputs
        self._lora_modules = {}
        scaling = alpha / r

        for name, original_module, lora_A, lora_B in lora_layers:
            # Store LoRA layers as model parameters so they're saved/optimized
            safe_name = name.replace(".", "_")
            setattr(self.model, f"lora_A_{safe_name}", lora_A)
            setattr(self.model, f"lora_B_{safe_name}", lora_B)

            # Register forward hook
            def make_hook(lora_A, lora_B, scaling):
                def hook(module, input, output):
                    x = input[0]
                    # Handle different input types
                    if isinstance(x, tuple):
                        x = x[0]
                    lora_out = lora_B(lora_A(x.to(lora_A.weight.dtype)))
                    return output + scaling * lora_out.to(output.dtype)
                return hook

            original_module.register_forward_hook(make_hook(lora_A, lora_B, scaling))
            self._lora_modules[name] = (lora_A, lora_B)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"LoRA applied to {len(lora_layers)} modules")
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100*trainable_params/total_params:.4f}")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Get trainable parameters from model
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Add MDN head parameters if enabled
        if self.mdn_head is not None:
            mdn_params = list(self.mdn_head.parameters())
            trainable_params.extend(mdn_params)
            print(f"[MDN] Adding {sum(p.numel() for p in mdn_params):,} MDN parameters to optimizer")

        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            betas=tuple(self.config["training"]["betas"]),
        )

        # Scheduler will be set up after dataloader creation

    def _setup_data(self):
        """Setup dataloaders."""
        data_dir = self.config["data"]["train_dir"]
        if data_dir is None:
            raise ValueError("data.train_dir must be set in config")

        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=self.config["training"]["batch_size"],
            val_split=self.config["data"]["val_split"],
            num_workers=self.config["data"]["num_workers"],
            prefetch_factor=self.config["data"].get("prefetch_factor", 4),
            persistent_workers=self.config["data"].get("persistent_workers", True),
            world_size=self.world_size,
            rank=self.rank,
        )

        # Setup scheduler
        num_training_steps = (
            len(self.train_loader) * self.config["training"]["epochs"]
            // self.config["training"]["gradient_accumulation_steps"]
        )
        num_warmup_steps = int(num_training_steps * self.config["training"]["warmup_ratio"])

        from transformers import get_cosine_schedule_with_warmup

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _crop_bottom(self, rgb: np.ndarray, frac: float = 0.3) -> np.ndarray:
        """Crop bottom of image (to hide vehicle hood)."""
        H, W = rgb.shape[:2]
        crop_h = int(H * (1 - frac))
        return rgb[:crop_h]

    def _process_single_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Process a single image tensor for SimLingo.

        Args:
            image: [3, H, W] tensor, normalized [0, 1]

        Returns:
            pixel_values: [1, 1, num_patches, 3, 448, 448] tensor
            (H, W): cropped image dimensions
        """
        # Convert to numpy RGB for processing
        rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Crop bottom (for NVIDIA data, use less aggressive crop since roof-mounted)
        crop_frac = self.config["data"].get("crop_bottom_frac", 0.15)
        rgb = self._crop_bottom(rgb, frac=crop_frac)
        H, W, _ = rgb.shape

        # Apply InternVL2 dynamic preprocessing.
        # When vectorized training is enabled, force max_num=1 so all images
        # yield P=1 patches and can be safely stacked across the batch.
        _vec_on = self.config.get('training', {}).get('vectorized', False)
        pil_img = Image.fromarray(rgb)
        patches = self.dynamic_preprocess(
            pil_img,
            image_size=448,
            use_thumbnail=self.use_global_img,
            max_num=1 if _vec_on else 2,
        )
        pixel_values = torch.stack([self.image_transform(p) for p in patches])  # [P, 3, 448, 448]
        pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)  # [1, 1, P, 3, 448, 448]
        return pixel_values, (H, W)

    def _build_batch_language_labels(
        self,
        speed_batch: torch.Tensor,        # [B]
        target_points_batch: torch.Tensor,  # [B, 2, 2]
        num_image_tokens: int,
        meta_actions: list[str] | None = None,  # [B] strings, optional
    ):
        """Build language labels for a batch.

        Args:
            speed_batch: Per-sample current speed.
            target_points_batch: Per-sample target points [B, 2, 2].
            num_image_tokens: Total image tokens emitted by the vision tower.
            meta_actions: Optional per-sample meta-action string. When
                provided, injected into the prompt to disambiguate multimodal
                futures (Plan B / SteerVLA-style conditioning). When None,
                falls back to the legacy prompt with no meta-action.

        Returns:
            prompt_lls, prompt_inf_lls: Lists of LanguageLabel for each sample
        """
        batch_size = speed_batch.shape[0]
        prompt_lls = []
        prompt_inf_lls = []

        for i in range(batch_size):
            speed = float(speed_batch[i].item())
            target_points_np = target_points_batch[i].cpu().numpy()  # [2, 2]

            # Build prompt text. The meta-action sentence is injected before
            # the prediction instruction so the model attends to the action
            # category while decoding waypoints.
            speed_rounded = round(speed, 1)
            if meta_actions is not None and meta_actions[i]:
                action_clause = f"The vehicle is {meta_actions[i]}. "
            else:
                action_clause = ""
            prompt_text = (
                f"Current speed: {speed_rounded} m/s. "
                f"Target waypoint: <TARGET_POINT><TARGET_POINT>. "
                f"{action_clause}"
                f"Predict the waypoints."
            )

            # Build conversation for tokenization
            conversation_all = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Waypoints:"},
                        ],
                    },
                ]
            ]

            conv_dict, question_dict = self.get_custom_chat_template(
                conversation_all,
                self.tokenizer,
                encoder_variant=self.cfg.model.vision_model.variant,
                num_image_tokens_total=num_image_tokens,
            )

            # Provide values for all possible special token IDs
            # The model iterates through special tokens in phrase_ids and looks up values
            # We need to cover all potential IDs:
            # - Our tokenizer's ID (151662)
            # - Model's training ID (151648)
            # - Any other special tokens that might appear
            placeholder_dict = {
                self.placeholder_token_id: target_points_np,
                151648: target_points_np,  # Model's expected TARGET_POINT ID
            }
            # Also provide empty/dummy values for chat template special tokens
            # to avoid KeyError when they appear in phrase_ids
            for special_id in [151644, 151645, 151646, 151647, 151649, 151650, 151651, 151652, 151653, 151654, 151655, 151656, 151657, 151658, 151659, 151660, 151661, 151662]:
                if special_id not in placeholder_dict:
                    placeholder_dict[special_id] = target_points_np  # Use same values as fallback
            placeholder_batch_list = [placeholder_dict]

            # Debug: check what token IDs appear in phrase_ids (one-shot, not per sample)
            if not getattr(self, "_phrase_id_debug_printed", False):
                phrase_ids = conv_dict["phrase_ids"]
                unique_ids = torch.unique(phrase_ids).tolist()
                print(f"[DEBUG] Phrase IDs unique values (first 20): {unique_ids[:20]}")
                print(f"[DEBUG] Using placeholder_token_id: {self.placeholder_token_id}")
                self._phrase_id_debug_printed = True

            def _to_ll(d):
                return self.LanguageLabel(
                    phrase_ids=d["phrase_ids"].to(self.device),
                    phrase_valid=d["phrase_valid"].to(self.device),
                    phrase_mask=d["phrase_mask"].to(self.device),
                    placeholder_values=placeholder_batch_list,
                    language_string=d["language_string"],
                    loss_masking=d.get("loss_masking"),
                )

            prompt_lls.append(_to_ll(conv_dict))
            prompt_inf_lls.append(_to_ll(question_dict))

        return prompt_lls, prompt_inf_lls

    def _build_driving_input_single(
        self,
        pixel_values: torch.Tensor,
        speed: float,
        target_points_np: np.ndarray,
        prompt_ll,
        prompt_inf_ll,
        HW: tuple[int, int],
    ):
        """Build DrivingInput for a single sample."""
        H, W = HW
        intrinsics = self.get_camera_intrinsics(W, H, self.fov_deg).unsqueeze(0).to(self.device).float()
        extrinsics = self.get_camera_extrinsics().unsqueeze(0).to(self.device).float()

        return self.DrivingInput(
            camera_images=pixel_values.to(self.device).bfloat16(),
            image_sizes=None,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            vehicle_speed=torch.tensor([[speed]], dtype=torch.float32, device=self.device),
            target_point=torch.tensor(target_points_np[0:1], dtype=torch.float32, device=self.device),
            prompt=prompt_ll,
            prompt_inference=prompt_inf_ll,
        )

    def _build_batched_language_label(self, prompt_lls: list):
        """Stack a list of per-sample LanguageLabel objects into one batched
        LanguageLabel with right-padding on phrase_ids/phrase_valid/phrase_mask.

        Each input LanguageLabel has phrase_ids shape [1, tokens_i] (varying tokens_i).
        Output has shape [B, max_tokens], padded with pad_id=0 on the right.
        placeholder_values lists are concatenated (one dict per sample).
        """
        import torch.nn.functional as F

        max_len = max(ll.phrase_ids.shape[1] for ll in prompt_lls)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0

        phrase_ids_list = []
        phrase_valid_list = []
        phrase_mask_list = []
        loss_masking_list = []
        placeholder_values_combined = []
        language_strings = []
        any_loss_masking = False

        for ll in prompt_lls:
            cur_len = ll.phrase_ids.shape[1]
            pad_len = max_len - cur_len

            if pad_len > 0:
                pids = F.pad(ll.phrase_ids, (0, pad_len), value=pad_id)
                pvalid = F.pad(ll.phrase_valid.to(torch.bool), (0, pad_len), value=False)
                pmask = F.pad(ll.phrase_mask.to(torch.bool), (0, pad_len), value=False)
            else:
                pids = ll.phrase_ids
                pvalid = ll.phrase_valid.to(torch.bool)
                pmask = ll.phrase_mask.to(torch.bool)

            phrase_ids_list.append(pids)
            phrase_valid_list.append(pvalid)
            phrase_mask_list.append(pmask)

            if ll.loss_masking is not None:
                any_loss_masking = True
                lm = ll.loss_masking
                if pad_len > 0:
                    lm = F.pad(lm.to(torch.bool), (0, pad_len), value=False)
                loss_masking_list.append(lm)
            else:
                loss_masking_list.append(None)

            # placeholder_values is a list of dicts (one per sample). Each
            # per-sample LL was built with a single-element list, so we
            # extend rather than append.
            placeholder_values_combined.extend(ll.placeholder_values)
            language_strings.append(ll.language_string)

        phrase_ids = torch.cat(phrase_ids_list, dim=0)
        phrase_valid = torch.cat(phrase_valid_list, dim=0)
        phrase_mask = torch.cat(phrase_mask_list, dim=0)

        if any_loss_masking:
            # Replace any None entries with all-False masks of max_len
            for i, lm in enumerate(loss_masking_list):
                if lm is None:
                    loss_masking_list[i] = torch.zeros(
                        (1, max_len), dtype=torch.bool, device=phrase_ids.device
                    )
            loss_masking = torch.cat(loss_masking_list, dim=0)
        else:
            loss_masking = None

        return self.LanguageLabel(
            phrase_ids=phrase_ids,
            phrase_valid=phrase_valid,
            phrase_mask=phrase_mask,
            placeholder_values=placeholder_values_combined,
            language_string=language_strings,
            loss_masking=loss_masking,
        )

    def _build_driving_input_batched(
        self,
        pixel_values_batch: torch.Tensor,  # [B, 1, P, 3, 448, 448]
        speeds: torch.Tensor,              # [B]
        target_points_batch: torch.Tensor, # [B, 2, 2]
        prompt_ll_batched,                  # batched LanguageLabel
        prompt_inf_ll_batched,              # batched LanguageLabel
        HW: tuple[int, int],
    ):
        """Build a single DrivingInput with [B, ...] tensors."""
        B = pixel_values_batch.shape[0]
        H, W = HW
        K = self.get_camera_intrinsics(W, H, self.fov_deg).to(self.device).float()        # [3, 3]
        E = self.get_camera_extrinsics().to(self.device).float()                  # [4, 4]
        intrinsics = K.unsqueeze(0).expand(B, -1, -1).contiguous()                # [B, 3, 3]
        extrinsics = E.unsqueeze(0).expand(B, -1, -1).contiguous()                # [B, 4, 4]

        vehicle_speed = speeds.to(self.device).float().reshape(B, 1)              # [B, 1]
        # Take the first target point per sample (matches single-sample build)
        target_point = target_points_batch[:, 0, :].to(self.device).float()       # [B, 2]

        return self.DrivingInput(
            camera_images=pixel_values_batch.to(self.device).bfloat16(),
            image_sizes=None,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            vehicle_speed=vehicle_speed,
            target_point=target_point,
            prompt=prompt_ll_batched,
            prompt_inference=prompt_inf_ll_batched,
        )

    def train_step_batched(self, batch: dict, accumulation_step: int = 0, grad_accum: int = 1) -> dict:
        """Fully batched training step: one forward + one backward over the whole batch.

        Replaces the per-sample loop in train_step. Use only when
        `training.vectorized: true` is set in the config AND the dataset is
        configured to produce a stable image patch count (max_num=1 path).
        """
        self.model.train()
        B = batch["image"].shape[0]
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available - cannot build language labels")

        _prof = {"_last": time.perf_counter()}
        # ----- Preprocess all images -----
        pixel_values_list = []
        HW_ref = None
        for i in range(B):
            pv, hw = self._process_single_image(batch["image"][i])
            if HW_ref is None:
                HW_ref = hw
            pixel_values_list.append(pv)  # each: [1, 1, P, 3, 448, 448]

        # All should have same P (vectorized forces max_num=1 -> P=1)
        try:
            pixel_values_batch = torch.cat(pixel_values_list, dim=0)  # [B, 1, P, 3, 448, 448]
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to stack pixel_values across batch. Ensure training.vectorized is "
                "set so max_num=1 is forced. Underlying error: " + str(e)
            )

        _prof_mark("image_prep", _prof)
        num_patches = pixel_values_batch.shape[2]
        num_image_tokens = num_patches * 256

        # ----- Build per-sample LanguageLabels -----
        meta_actions = [
            waypoints_to_meta_action(
                batch["waypoints"][i].cpu().numpy(),
                current_speed_mps=float(batch["speed_mps"][i].item()),
            )
            for i in range(B)
        ]
        _prof_mark("meta_actions", _prof)
        prompt_lls, prompt_inf_lls = self._build_batch_language_labels(
            batch["speed_mps"],
            batch["target_points"],
            num_image_tokens,
            meta_actions=meta_actions,
        )
        _prof_mark("lang_label_build", _prof)

        # ----- Stack LanguageLabels into batched LanguageLabel -----
        prompt_ll_batched = self._build_batched_language_label(prompt_lls)
        prompt_inf_ll_batched = self._build_batched_language_label(prompt_inf_lls)
        _prof_mark("lang_label_pad", _prof)

        # ----- Build batched DrivingInput -----
        driving_input = self._build_driving_input_batched(
            pixel_values_batch,
            batch["speed_mps"],
            batch["target_points"],
            prompt_ll_batched,
            prompt_inf_ll_batched,
            HW_ref,
        )
        _prof_mark("driving_input", _prof)

        # ----- One forward pass for the whole batch -----
        speed_wps, route_wps, language = self.model(driving_input)
        _prof_mark("forward", _prof)

        if self.use_mdn and self.mdn_head is not None:
            captured = self._last_features
            if captured is None:
                raise RuntimeError("MDN mode enabled but no features captured by hook")
            if captured.dim() == 3:
                n_drive = min(self.config["model"].get("num_waypoints", 11), captured.shape[1])
                feats = captured[:, -n_drive:, :].mean(dim=1)
            else:
                feats = captured
            mdn_output = self.mdn_head(feats.float())
            pred_wps = mdn_output  # dict with pi, mu, sigma
            self._last_features = None
        else:
            pred_wps = speed_wps

        if pred_wps is None:
            raise RuntimeError("Model returned None for waypoints in batched train_step")

        gt_wps = batch["waypoints"].to(self.device)  # [B, 11, 2]
        gt_route = batch["route"].to(self.device)    # [B, 20, 2]

        # ----- Shape alignment -----
        if isinstance(pred_wps, dict):
            gt_wps_aligned = gt_wps
        else:
            pred_wps = pred_wps.float()
            if pred_wps.shape[1] != gt_wps.shape[1]:
                min_len = min(pred_wps.shape[1], gt_wps.shape[1])
                pred_wps = pred_wps[:, :min_len, :]
                gt_wps_aligned = gt_wps[:, :min_len, :]
            else:
                gt_wps_aligned = gt_wps

        # ----- Single loss + backward for the whole batch -----
        loss, loss_dict = self.compute_loss(
            pred_wps,
            route_wps.float() if route_wps is not None else None,
            gt_wps_aligned,
            gt_route,
        )
        _prof_mark("loss_compute", _prof)

        # compute_loss already does mean reduction across batch, so we only
        # divide by grad_accum (not also by batch_size).
        loss_scaled = loss / grad_accum
        loss_scaled.backward()
        _prof_mark("backward", _prof)
        _prof_log("vec",
                  getattr(self, "global_step", 0),
                  getattr(self, "is_main", True),
                  _prof)

        return {
            "total_loss": loss_dict["total_loss"],
            "wp_loss": loss_dict["wp_loss"],
            "route_loss": loss_dict["route_loss"],
        }

    def compute_loss(
        self,
        pred_wps: torch.Tensor | dict,
        pred_route: torch.Tensor | None,
        gt_wps: torch.Tensor,
        gt_route: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute training loss.

        Args:
            pred_wps: Predicted waypoints [B, 11, 2] OR MDN output dict if using MDN
            pred_route: Predicted route [B, ?, 2] or None
            gt_wps: Ground truth waypoints [B, 11, 2]
            gt_route: Ground truth route [B, 20, 2]

        Returns:
            total_loss, loss_dict
        """
        # Waypoint loss - use MDN loss if pred_wps is a dict (MDN output)
        if isinstance(pred_wps, dict) and "pi" in pred_wps:
            # MDN mode - compute negative log-likelihood loss
            wp_loss = mdn_loss(pred_wps, gt_wps, reduction="mean")
        else:
            # Standard regression mode
            wp_loss = F.smooth_l1_loss(pred_wps, gt_wps)

        # Route loss (if available)
        if pred_route is not None:
            # Interpolate pred_route to match gt_route size if needed
            if pred_route.shape[1] != gt_route.shape[1]:
                # Simple interpolation to match sizes
                pred_route = F.interpolate(
                    pred_route.permute(0, 2, 1),
                    size=gt_route.shape[1],
                    mode='linear',
                    align_corners=True,
                ).permute(0, 2, 1)
            route_loss = F.smooth_l1_loss(pred_route, gt_route)
        else:
            route_loss = torch.tensor(0.0, device=self.device)

        # Weighted sum (with defaults if not in config)
        loss_config = self.config.get("loss", {})
        wp_weight = loss_config.get("waypoint_weight", 1.0)
        route_weight = loss_config.get("route_weight", 0.5)

        total_loss = wp_weight * wp_loss + route_weight * route_loss

        loss_dict = {
            "wp_loss": wp_loss.item(),
            "route_loss": route_loss.item() if isinstance(route_loss, torch.Tensor) else 0.0,
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict

    def train_step(self, batch: dict, accumulation_step: int = 0, grad_accum: int = 1) -> dict:
        """Single training step with actual forward pass and loss computation.

        Args:
            batch: Input batch
            accumulation_step: Current step within gradient accumulation (0 to grad_accum-1)
            grad_accum: Total gradient accumulation steps

        Note: optimizer.zero_grad() and optimizer.step() are called by the training loop,
        not here. This allows proper gradient accumulation.
        """
        self.model.train()

        batch_size = batch["image"].shape[0]

        # Check that we have a tokenizer
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available - cannot build language labels")

        # Process each sample in the batch
        # Note: SimLingo's DrivingInput is designed for batch_size=1 internally,
        # so we process samples one at a time and accumulate gradients
        total_wp_loss = 0.0
        total_route_loss = 0.0
        valid_samples = 0

        _prof_totals = {"image_prep": 0.0, "lang_build": 0.0, "forward": 0.0,
                        "loss_compute": 0.0, "backward": 0.0}
        for i in range(batch_size):
            _prof_iter = {"_last": time.perf_counter()}
            try:
                # Process single image
                image = batch["image"][i]  # [3, H, W]
                speed = float(batch["speed_mps"][i].item())
                target_points = batch["target_points"][i].cpu().numpy()  # [2, 2]
                gt_wps = batch["waypoints"][i:i+1].to(self.device)  # [1, 11, 2]
                gt_route = batch["route"][i:i+1].to(self.device)  # [1, 20, 2]

                # Debug: print sample info on first batch
                if self.global_step == 0 and i == 0:
                    print(f"[DEBUG] First sample info:")
                    print(f"  Image shape: {image.shape}")
                    print(f"  Speed: {speed:.2f} m/s")
                    print(f"  Target points: {target_points}")
                    print(f"  GT waypoints shape: {gt_wps.shape}, first: {gt_wps[0, 0]}")
                    print(f"  GT route shape: {gt_route.shape}")

                # Process image for model
                pixel_values, HW = self._process_single_image(image)
                _prof_mark("image_prep", _prof_iter)

                if self.global_step == 0 and i == 0:
                    print(f"  Processed image shape: {pixel_values.shape}, HW: {HW}")

                # Calculate number of image tokens
                num_patches = pixel_values.shape[2]  # [1, 1, P, 3, 448, 448]
                num_image_tokens = num_patches * 256  # 256 tokens per patch for InternVL2-1B

                # Build language labels for this sample.
                # Plan B: derive a meta-action label from the GT future
                # waypoints at train time and inject it into the prompt. This
                # gives Smooth L1 regression a disambiguated, near-unimodal
                # target distribution per (image, meta-action) pair without
                # any external labeling cost.
                speed_tensor = batch["speed_mps"][i:i+1]
                target_points_tensor = batch["target_points"][i:i+1]
                meta_action = waypoints_to_meta_action(
                    gt_wps[0].detach().cpu().numpy(),
                    current_speed_mps=speed,
                )
                if self.global_step == 0 and i == 0:
                    print(f"  Meta-action (rule-based): {meta_action!r}")
                prompt_lls, prompt_inf_lls = self._build_batch_language_labels(
                    speed_tensor,
                    target_points_tensor,
                    num_image_tokens,
                    meta_actions=[meta_action],
                )

                # Build DrivingInput
                driving_input = self._build_driving_input_single(
                    pixel_values, speed, target_points, prompt_lls[0], prompt_inf_lls[0], HW
                )
                _prof_mark("lang_build", _prof_iter)

                # --- Forward pass --------------------------------------------------
                # Always run the full model forward (single pass). In MDN mode the
                # hook installed on the LLM's final norm captures `_last_features`
                # on the trainer; we then pass that through the MDN head and IGNORE
                # the model's built-in waypoint head output.
                speed_wps, route_wps, language = self.model(driving_input)

                if self.use_mdn and self.mdn_head is not None:
                    captured = self._last_features  # set by capture_hook
                    if captured is None:
                        raise RuntimeError(
                            "MDN mode is enabled but the feature hook did not "
                            "capture any features. Check _setup_feature_extraction_hook."
                        )
                    # captured is [B, seq, hidden]. SimLingo's own waypoint heads
                    # read `features[:, -len_driving:]` (the driving-adaptor token
                    # positions at the tail of the sequence). We mean-pool over
                    # the last `num_waypoints` positions as a stable proxy for
                    # those adaptor tokens, then feed [B, hidden] into the MDN.
                    if captured.dim() == 3:
                        n_drive = min(
                            self.config["model"].get("num_waypoints", 11),
                            captured.shape[1],
                        )
                        feats = captured[:, -n_drive:, :].mean(dim=1)
                    else:
                        feats = captured
                    mdn_output = self.mdn_head(feats.float())
                    pred_wps = mdn_output  # dict with pi, mu, sigma
                    # Clear captured features so a stale tensor isn't reused.
                    self._last_features = None
                else:
                    pred_wps = speed_wps

                # --- Debug logging (first step only) -------------------------------
                if self.global_step == 0 and i == 0:
                    if isinstance(pred_wps, dict):
                        print(f"  [MDN] pi={pred_wps['pi'].shape}, "
                              f"mu={pred_wps['mu'].shape}, "
                              f"sigma={pred_wps['sigma'].shape}")
                    else:
                        sw = pred_wps.shape if pred_wps is not None else None
                        rw = route_wps.shape if route_wps is not None else None
                        print(f"  Model output: speed_wps={sw} route_wps={rw}")
                        if pred_wps is not None:
                            print(f"  First pred waypoint: {pred_wps[0, 0]}")

                # --- Skip if no prediction -----------------------------------------
                if pred_wps is None:
                    print(f"Warning: Sample {i} - Model returned None for waypoints")
                    continue
                _prof_mark("forward", _prof_iter)

                # --- Shape alignment for the non-MDN path --------------------------
                if isinstance(pred_wps, dict):
                    gt_wps_aligned = gt_wps
                else:
                    pred_wps = pred_wps.float()
                    if pred_wps.shape[1] != gt_wps.shape[1]:
                        min_len = min(pred_wps.shape[1], gt_wps.shape[1])
                        pred_wps = pred_wps[:, :min_len, :]
                        gt_wps_aligned = gt_wps[:, :min_len, :]
                    else:
                        gt_wps_aligned = gt_wps

                # --- Loss ----------------------------------------------------------
                loss, loss_dict = self.compute_loss(
                    pred_wps,
                    route_wps.float() if route_wps is not None else None,
                    gt_wps_aligned,
                    gt_route,
                )

                if self.global_step == 0 and i == 0:
                    print(f"  Loss: wp={loss_dict['wp_loss']:.4f}, route={loss_dict['route_loss']:.4f}")

                # Backward pass for this sample (accumulate gradients)
                # Scale by both batch_size and grad_accum for proper averaging
                _prof_mark("loss_compute", _prof_iter)
                loss_scaled = loss / (batch_size * grad_accum)
                loss_scaled.backward()
                _prof_mark("backward", _prof_iter)

                for _k in ("image_prep", "lang_build", "forward", "loss_compute", "backward"):
                    _prof_totals[_k] += _prof_iter.get(_k, 0.0)

                total_wp_loss += loss_dict["wp_loss"]
                total_route_loss += loss_dict["route_loss"]
                valid_samples += 1

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                if self.global_step == 0:
                    traceback.print_exc()
                continue

        _prof_totals["_last"] = 0.0  # satisfy _prof_log contract
        _prof_log("unvec",
                  getattr(self, "global_step", 0),
                  getattr(self, "is_main", True),
                  _prof_totals)

        # If no valid samples, return zero loss
        if valid_samples == 0:
            print("Warning: No valid samples in batch!")
            return {"total_loss": 0.0, "wp_loss": 0.0, "route_loss": 0.0}

        # Average losses (for logging only - gradients are already accumulated)
        avg_loss_dict = {
            "wp_loss": total_wp_loss / valid_samples,
            "route_loss": total_route_loss / valid_samples,
            "total_loss": (total_wp_loss + total_route_loss) / valid_samples,
        }

        if self.global_step == 0:
            print(f"[DEBUG] First step avg loss: {avg_loss_dict}")

        return avg_loss_dict

    def validate(self) -> dict:
        """Run validation loop. Computes val loss and renders a viz panel
        of the first sample for qualitative monitoring."""
        self.model.eval()

        total_wp_loss = 0.0
        total_route_loss = 0.0
        total_loss_sum = 0.0
        num_samples = 0
        viz_panel = None
        viz_caption = ""

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                batch_size = batch["image"].shape[0]

                for i in range(batch_size):
                    try:
                        image = batch["image"][i]
                        speed = float(batch["speed_mps"][i].item())
                        target_points = batch["target_points"][i].cpu().numpy()
                        gt_wps = batch["waypoints"][i:i+1].to(self.device)
                        gt_route = batch["route"][i:i+1].to(self.device)

                        pixel_values, HW = self._process_single_image(image)
                        num_patches = pixel_values.shape[2]
                        num_image_tokens = num_patches * 256

                        meta_action = waypoints_to_meta_action(
                            gt_wps[0].detach().cpu().numpy(),
                            current_speed_mps=speed,
                        )
                        speed_tensor = batch["speed_mps"][i:i+1]
                        target_points_tensor = batch["target_points"][i:i+1]
                        prompt_lls, prompt_inf_lls = self._build_batch_language_labels(
                            speed_tensor,
                            target_points_tensor,
                            num_image_tokens,
                            meta_actions=[meta_action],
                        )

                        driving_input = self._build_driving_input_single(
                            pixel_values, speed, target_points,
                            prompt_lls[0], prompt_inf_lls[0], HW,
                        )

                        speed_wps, route_wps, language = self.model(driving_input)

                        if speed_wps is None:
                            continue

                        pred_wps = speed_wps.float()
                        if pred_wps.shape[1] != gt_wps.shape[1]:
                            min_len = min(pred_wps.shape[1], gt_wps.shape[1])
                            pred_wps_aligned = pred_wps[:, :min_len, :]
                            gt_wps_aligned = gt_wps[:, :min_len, :]
                        else:
                            pred_wps_aligned = pred_wps
                            gt_wps_aligned = gt_wps

                        loss, loss_dict = self.compute_loss(
                            pred_wps_aligned,
                            route_wps.float() if route_wps is not None else None,
                            gt_wps_aligned,
                            gt_route,
                        )

                        total_wp_loss += loss_dict["wp_loss"]
                        total_route_loss += loss_dict["route_loss"]
                        total_loss_sum += loss_dict["total_loss"]
                        num_samples += 1

                        # Render viz on first sample only
                        if viz_panel is None and batch_idx == 0 and i == 0:
                            try:
                                from PIL import Image as PILImage
                                from scripts.viz_training import (
                                    render_prediction_panel,
                                )

                                rgb_path = batch.get("rgb_path", [None])[i]
                                if rgb_path:
                                    rgb_img = np.array(
                                        PILImage.open(rgb_path).convert("RGB")
                                    )
                                else:
                                    # Fallback: denormalize the tensor
                                    img = image.detach().cpu().numpy()
                                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                                    rgb_img = np.transpose(img, (1, 2, 0))

                                H, W = rgb_img.shape[:2]
                                K = self.get_camera_intrinsics(W, H, self.fov_deg).cpu().numpy()
                                gt_np = gt_wps_aligned[0].cpu().numpy()
                                pred_np = pred_wps_aligned[0].cpu().numpy()

                                viz_panel = render_prediction_panel(
                                    rgb_img, gt_np, pred_np, K,
                                    speed_mps=speed,
                                    meta_action=meta_action,
                                )
                                viz_caption = (
                                    f"step {self.global_step} | speed {speed:.1f}m/s | "
                                    f"action: {meta_action} | "
                                    f"wp_l1 {loss_dict['wp_loss']:.3f}"
                                )
                            except Exception as e:
                                print(f"[viz] Failed to render panel: {e}")
                    except Exception as e:
                        print(f"[validate] Sample {i} error: {e}")
                        continue

        avg_wp = total_wp_loss / max(num_samples, 1)
        avg_route = total_route_loss / max(num_samples, 1)
        avg_total = total_loss_sum / max(num_samples, 1)

        # all_reduce val metrics across ranks so every rank sees the global mean
        if getattr(self, 'is_distributed', False):
            import torch.distributed as dist
            t = torch.tensor([avg_total, avg_wp, avg_route], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= self.world_size
            avg_total, avg_wp, avg_route = t.tolist()

        # Log viz to W&B (rank 0 only)
        if viz_panel is not None and getattr(self, 'is_main', True):
            try:
                import wandb
                from scripts.viz_training import panel_to_wandb_image
                wb_img = panel_to_wandb_image(viz_panel, caption=viz_caption)
                if wb_img is not None:
                    wandb.log(
                        {"val/prediction_panel": wb_img},
                        step=self.global_step,
                    )
            except Exception as e:
                print(f"[viz] Failed to log panel to wandb: {e}")

        self.model.train()
        return {
            "val_loss": avg_total,
            "val_wp_loss": avg_wp,
            "val_route_loss": avg_route,
        }

    def train(self) -> dict:
        """Main training loop."""
        import wandb

        epochs = self.config["training"]["epochs"]
        grad_accum = self.config["training"]["gradient_accumulation_steps"]
        logging_steps = self.config["training"]["logging_steps"]
        eval_steps = self.config["training"]["eval_steps"]
        save_steps = self.config["training"]["save_steps"]
        es_enabled = bool(self.config["training"].get("early_stopping", False))
        es_patience = int(self.config["training"].get("early_stopping_patience", 3))
        es_min_delta = float(self.config["training"].get("early_stopping_min_delta", 1e-4))

        print(f"Starting training for {epochs} epochs")
        print(f"  Total steps: {len(self.train_loader) * epochs // grad_accum}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {self.config['training']['batch_size'] * grad_accum}")

        all_metrics = []

        # Exponential moving average for smoother loss logging
        ema_loss = None
        ema_alpha = 0.1  # Smoothing factor

        # Honor resume: skip epochs already finished in the loaded checkpoint.
        # `_resume_start_epoch` is set by load_checkpoint(); defaults to 0.
        start_epoch = getattr(self, "_resume_start_epoch", 0)
        if start_epoch > 0 and getattr(self, "is_main", True):
            print(f"[resume] skipping {start_epoch} completed epoch(s); "
                  f"starting at epoch {start_epoch + 1}/{epochs}")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            self.model.train()

            # Rotate DistributedSampler shuffle per epoch
            if getattr(self, 'is_distributed', False):
                from torch.utils.data.distributed import DistributedSampler
                if isinstance(self.train_loader.sampler, DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)

            epoch_losses = []
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=not getattr(self, 'is_main', True),
            )

            # Accumulated losses for logging
            accum_losses = {"total_loss": 0.0, "wp_loss": 0.0, "route_loss": 0.0}

            for step, batch in enumerate(pbar):
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}

                # Zero gradients at start of accumulation cycle
                if step % grad_accum == 0:
                    self.optimizer.zero_grad()

                # Training step (accumulates gradients)
                # Dispatch to batched implementation when opted in.
                _step_fn = self.train_step_batched if self.config.get('training', {}).get('vectorized', False) else self.train_step
                loss_dict = _step_fn(batch, step % grad_accum, grad_accum)

                # Accumulate losses for logging
                for k in accum_losses:
                    accum_losses[k] += loss_dict[k] / grad_accum

                # Optimizer step at end of accumulation cycle
                if (step + 1) % grad_accum == 0 or step == len(self.train_loader) - 1:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Optimizer and scheduler step
                    self.optimizer.step()
                    self.scheduler.step()

                    # Record accumulated loss
                    epoch_losses.append(accum_losses["total_loss"])

                    # Update EMA
                    if ema_loss is None:
                        ema_loss = accum_losses["total_loss"]
                    else:
                        ema_loss = ema_alpha * accum_losses["total_loss"] + (1 - ema_alpha) * ema_loss

                    # Update progress bar with smoothed loss
                    pbar.set_postfix(loss=round(ema_loss, 3))

                    self.global_step += 1

                    # Logging (rank 0 only)
                    if self.global_step % logging_steps == 0 and getattr(self, 'is_main', True):
                        wandb.log({
                            "train/loss": accum_losses["total_loss"],
                            "train/loss_ema": ema_loss,
                            "train/wp_loss": accum_losses["wp_loss"],
                            "train/route_loss": accum_losses["route_loss"],
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch + step / len(self.train_loader),
                        }, step=self.global_step)

                    # Reset accumulated losses
                    accum_losses = {"total_loss": 0.0, "wp_loss": 0.0, "route_loss": 0.0}

                    # Evaluation (all ranks participate; only rank 0 logs/saves)
                    if self.global_step % eval_steps == 0:
                        val_metrics = self.validate()
                        if getattr(self, 'is_main', True):
                            wandb.log({
                                "val/loss": val_metrics["val_loss"],
                                "val/wp_loss": val_metrics.get("val_wp_loss", 0.0),
                                "val/route_loss": val_metrics.get("val_route_loss", 0.0),
                            }, step=self.global_step)

                            if val_metrics["val_loss"] < self.best_val_loss:
                                self.best_val_loss = val_metrics["val_loss"]
                                self.save_checkpoint("best")

                    # Save checkpoint (rank 0 only)
                    if self.global_step % save_steps == 0 and getattr(self, 'is_main', True):
                        self.save_checkpoint(f"step_{self.global_step}")

                    # Proactive (out-of-band) checkpoint request: poll the
                    # checkpoint volume for a SAVE_NOW sentinel every 10 steps.
                    # Sentinels are dropped by `request_checkpoint` (see
                    # modal_training.py). Rank 0 only — save_checkpoint already
                    # uses the unwrapped model.
                    if (
                        getattr(self, 'is_main', True)
                        and self.global_step % 10 == 0
                    ):
                        self._check_save_request()

            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            if getattr(self, 'is_main', True):
                print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
            all_metrics.append({"epoch": epoch + 1, "loss": avg_epoch_loss})

            # End-of-epoch validation + viz (all ranks participate)
            if getattr(self, 'is_main', True):
                print(f"Running end-of-epoch validation...")
            val_metrics = self.validate()
            if getattr(self, 'is_main', True):
                wandb.log({
                    "val/loss": val_metrics["val_loss"],
                    "val/wp_loss": val_metrics.get("val_wp_loss", 0.0),
                    "val/route_loss": val_metrics.get("val_route_loss", 0.0),
                    "val/best_loss": self.best_val_loss,
                    "val/es_patience_remaining": max(0, es_patience - self._es_no_improve_count),
                    "epoch": epoch + 1,
                }, step=self.global_step)
                print(f"Epoch {epoch + 1} val loss: {val_metrics['val_loss']:.4f}")

                # Save best checkpoint + early-stop bookkeeping
                improved = val_metrics["val_loss"] < (self.best_val_loss - es_min_delta)
                if improved:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best")
                    self._es_no_improve_count = 0
                else:
                    self._es_no_improve_count += 1
                    if es_enabled:
                        print(
                            f"  [early-stop] no improvement for "
                            f"{self._es_no_improve_count}/{es_patience} epochs "
                            f"(best={self.best_val_loss:.4f}, cur={val_metrics['val_loss']:.4f})"
                        )

                # Always save end-of-epoch checkpoint
                self.save_checkpoint(f"epoch_{epoch + 1}")

                # Decide early stop on rank 0; broadcast to other ranks below.
                if es_enabled and self._es_no_improve_count >= es_patience:
                    self._es_should_stop = True
                    print(
                        f"  [early-stop] triggered after epoch {epoch + 1} "
                        f"(best val/loss={self.best_val_loss:.4f})"
                    )

            # Broadcast the stop decision across DDP ranks so all peers exit
            # the epoch loop together (otherwise NCCL hangs on next collective).
            if getattr(self, 'is_distributed', False):
                import torch.distributed as dist
                flag = torch.tensor(
                    [1 if self._es_should_stop else 0],
                    dtype=torch.int32,
                    device=self.device,
                )
                dist.broadcast(flag, src=0)
                self._es_should_stop = bool(flag.item())

            if self._es_should_stop:
                break

        return {
            "final_loss": all_metrics[-1]["loss"] if all_metrics else 0.0,
            "best_val_loss": self.best_val_loss,
            "early_stopped": self._es_should_stop,
            "epochs_completed": self.current_epoch + 1,
        }

    def _check_save_request(self) -> None:
        """Look for SAVE_NOW sentinel files in output_dir and trigger an
        out-of-band checkpoint save when found.

        Sentinels are dropped by the `request_checkpoint` Modal function in
        a separate container. To make those writes visible inside the
        long-running training container we `reload()` the Modal volume
        first. After saving we delete the sentinel and `commit()` so the
        new checkpoint is visible to downstream readers.

        Safe to call from rank 0 only — silently no-ops outside Modal.
        """
        try:
            try:
                import modal
                _vol = modal.Volume.from_name("simlingo-checkpoints")
                _vol.reload()
            except Exception:
                _vol = None

            for sentinel in sorted(self.output_dir.glob("SAVE_NOW*")):
                tag = sentinel.name[len("SAVE_NOW"):].lstrip(".") or (
                    f"manual_step_{self.global_step}"
                )
                print(f"[manual-ckpt] sentinel '{sentinel.name}' -> saving as '{tag}'")
                self.save_checkpoint(tag)
                try:
                    sentinel.unlink()
                except FileNotFoundError:
                    pass
                if _vol is not None:
                    try:
                        _vol.commit()
                    except Exception as e:
                        print(f"[manual-ckpt] commit warning: {e}")
        except Exception as e:
            # Polling failure should never break training.
            print(f"[manual-ckpt] poll error (non-fatal): {e}")

    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint with LoRA weights and MDN head."""
        ckpt_path = self.output_dir / f"checkpoint_{name}.pt"

        # Save LoRA weights separately (use _raw_model to avoid DDP `module.` prefix)
        _src_model = getattr(self, '_raw_model', self.model)
        lora_state = {}
        for key, value in _src_model.state_dict().items():
            if "lora_A_" in key or "lora_B_" in key:
                lora_state[key] = value

        lora_path = self.output_dir / f"lora_{name}.pt"
        torch.save(lora_state, lora_path)
        print(f"Saved LoRA weights ({len(lora_state)} tensors) to {lora_path}")

        # Save MDN head if enabled
        if self.mdn_head is not None:
            mdn_path = self.output_dir / f"mdn_head_{name}.pt"
            torch.save(self.mdn_head.state_dict(), mdn_path)
            print(f"Saved MDN head to {mdn_path}")

        # Save full model state if needed (use _raw_model to avoid DDP `module.` prefix)
        torch.save(getattr(self, '_raw_model', self.model).state_dict(), ckpt_path)

        # Save training state
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "use_mdn": self.use_mdn,
        }
        torch.save(state, self.output_dir / f"training_state_{name}.pt")

        print(f"Saved checkpoint: {name}")
        return ckpt_path

    def load_checkpoint(self, path: str):
        """Load checkpoint to resume training."""
        path = Path(path)

        # Check if it's a LoRA-only checkpoint
        _dst_model = getattr(self, '_raw_model', self.model)
        if path.name.startswith("lora_") and path.suffix == ".pt":
            lora_state = torch.load(path, map_location=self.device)
            current_state = _dst_model.state_dict()
            for key, value in lora_state.items():
                if key in current_state:
                    current_state[key] = value
            _dst_model.load_state_dict(current_state)
        elif path.suffix == ".pt":
            state_dict = torch.load(path, map_location=self.device)
            _dst_model.load_state_dict(state_dict, strict=False)

        # Load MDN head if it exists
        name = path.stem.replace("checkpoint_", "").replace("lora_", "")
        mdn_path = path.parent / f"mdn_head_{name}.pt"
        if mdn_path.exists() and self.mdn_head is not None:
            mdn_state = torch.load(mdn_path, map_location=self.device)
            self.mdn_head.load_state_dict(mdn_state)
            print(f"Loaded MDN head from {mdn_path}")

        # Load training state
        state_path = path.parent / f"training_state_{name}.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state["global_step"]
            self.current_epoch = state["current_epoch"]
            self.best_val_loss = state["best_val_loss"]
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            # Skip epochs already completed. `current_epoch` is the index of
            # the epoch that was active when the checkpoint was saved, and
            # save_checkpoint(f"epoch_{epoch+1}") is called at end-of-epoch,
            # so on resume we start from current_epoch + 1.
            self._resume_start_epoch = self.current_epoch + 1
        else:
            self._resume_start_epoch = 0

        print(f"Loaded checkpoint from {path}")

    def push_to_hub(self, repo_id: str):
        """Push final + best LoRA weights to HuggingFace Hub.

        Creates the repo if it doesn't exist (honoring `hub.private` from config).
        Uploads `lora_final.pt` and (if present) `lora_best.pt`. Optimizer state
        is intentionally NOT pushed (too large for HF, store on Modal volume).
        """
        from huggingface_hub import HfApi

        hub_cfg = self.config.get("hub", {}) if isinstance(self.config, dict) else {}
        private = bool(hub_cfg.get("private", True))
        push_best = bool(hub_cfg.get("push_best_checkpoint", True))

        api = HfApi()

        # Ensure the repo exists (idempotent).
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            print(f"create_repo({repo_id}) raised: {e} — attempting upload anyway")

        # (Re-)write lora_final.pt from current weights so we always push the
        # freshest LoRA state at end-of-training.
        final_path = self.output_dir / "lora_final.pt"
        _src_model = getattr(self, "_raw_model", self.model)
        final_state = {
            k: v for k, v in _src_model.state_dict().items()
            if ("lora_A_" in k or "lora_B_" in k)
        }
        torch.save(final_state, final_path)

        targets = [(final_path, "lora_final.pt")]
        if push_best:
            best_path = self.output_dir / "lora_best.pt"
            if best_path.exists():
                targets.append((best_path, "lora_best.pt"))
            else:
                print("  [hub] lora_best.pt not found; skipping best-ckpt upload")

        for src, dst in targets:
            try:
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=dst,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print(f"  [hub] uploaded {src.name} -> {repo_id}:{dst}")
            except Exception as e:
                print(f"  [hub] failed to upload {src.name}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train SimLingo on NVIDIA data")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output-dir", default="/checkpoints", help="Output directory")
    parser.add_argument("--wandb-project", default="simlingo-nvidia-finetune")
    parser.add_argument("--hf-repo", help="HuggingFace repo for checkpoint upload (overrides hub.repo_id in config)")
    parser.add_argument("--resume-from", help="Path to a checkpoint to resume training from")

    args = parser.parse_args()

    import wandb
    import yaml

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Detect rank for rank-0 wandb gating (torchrun sets RANK).
    _entrypoint_rank = int(os.environ.get("RANK", "0"))
    _is_main_entry = _entrypoint_rank == 0

    # Initialize W&B on rank 0 only; non-main ranks disable wandb entirely.
    if _is_main_entry:
        _wandb_cfg = config.get("wandb", {}) if isinstance(config, dict) else {}
        _world_size = int(os.environ.get("WORLD_SIZE", "1"))
        _tags = [
            f"world_size={_world_size}",
            f"vectorized={config.get('training', {}).get('vectorized', False)}",
            f"epochs={config.get('training', {}).get('epochs', 'na')}",
            f"bs={config.get('training', {}).get('batch_size', 'na')}",
            f"lr={config.get('training', {}).get('learning_rate', 'na')}",
        ]
        wandb.init(
            project=args.wandb_project,
            entity=_wandb_cfg.get("entity"),
            config=config,
            tags=_tags,
            save_code=True,  # Snapshot the trainer.py + config in W&B
        )
        # Define metric step axis so val/* and train/* align on the same _step.
        try:
            wandb.define_metric("train/*", step_metric="_step")
            wandb.define_metric("val/*", step_metric="_step", summary="min")
            wandb.define_metric("val/loss", summary="min")
        except Exception:
            pass
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(mode="disabled")

    # Get checkpoint directory from environment or default
    ckpt_dir = os.environ.get("CKPT_DIR", "/data/checkpoint")

    # Resolve HF repo: CLI flag overrides config.hub.repo_id.
    _hub_cfg = config.get("hub", {}) if isinstance(config, dict) else {}
    _resolved_hf_repo = args.hf_repo or (
        _hub_cfg.get("repo_id") if _hub_cfg.get("push_to_hub") else None
    )

    # Create trainer
    trainer = NVIDIASimLingoTrainer(
        config=config,
        ckpt_dir=ckpt_dir,
        output_dir=args.output_dir,
        hf_repo=_resolved_hf_repo,
    )

    # Optional: resume from a prior checkpoint.
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        if _is_main_entry:
            print(f"Resumed from {args.resume_from} at step {trainer.global_step}")

    # Train (may early-stop based on config.training.early_stopping).
    metrics = trainer.train()
    if _is_main_entry:
        print(f"Training finished: {metrics}")

    # Save final
    if _is_main_entry:
        trainer.save_checkpoint("final")

    # Push to hub if configured (rank 0 only). CLI takes precedence over config.
    if _resolved_hf_repo and _is_main_entry:
        trainer.push_to_hub(_resolved_hf_repo)

    if _is_main_entry:
        wandb.finish()


if __name__ == "__main__":
    main()
