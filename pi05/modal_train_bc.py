"""π0.5 behavior cloning training on Modal (8× H100).

Trains π0.5 on PhysicalAI-AV ground truth driving data using the
pi05_driving config in openpi. LoRA on VLM + full fine-tune on action expert.

Usage:
    modal run pi05/modal_train_bc.py::train_bc
    modal run pi05/modal_train_bc.py::train_bc --num-steps 100  # smoke test
    modal run pi05/modal_train_bc.py::upload_checkpoint --step 5000
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-train-bc"
CACHE_DIR = "/cache"
OPENPI_DIR = "/opt/openpi"

# ---------------------------------------------------------------------------
# Image: openpi (JAX/Flax) + our driving config patches
# ---------------------------------------------------------------------------

train_image = (
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
    .pip_install("huggingface_hub", "wandb")
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
# Training function
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="H100:8",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    memory=128 * 1024,
)
def train_bc(
    num_steps: int | None = None,
    exp_name: str = "bc-coldstart",
    resume: bool = False,
    batch_size: int = 96,
):
    import os
    import pathlib
    import shutil
    import subprocess
    import sys

    # Patch openpi with our driving config
    _patch_openpi()

    checkpoint_dir = f"{CACHE_DIR}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    cmd = [
        "uv", "run", "python", "-m", "scripts.train",
        "pi05_driving",
        "--exp-name", exp_name,
    ]

    if num_steps is not None:
        cmd.extend(["--num-train-steps", str(num_steps)])

    if batch_size != 96:
        cmd.extend(["--batch-size", str(batch_size)])

    if resume:
        cmd.append("--resume")
    else:
        cmd.append("--overwrite")

    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        "WANDB_PROJECT": "pi05-driving",
    }

    print(f"=== Training π0.5 driving BC ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    result = subprocess.run(
        cmd,
        cwd=OPENPI_DIR,
        env=env,
        text=True,
    )

    cache_volume.commit()

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return result.returncode

    print("Training complete!")
    return 0


# ---------------------------------------------------------------------------
# Upload checkpoint to HuggingFace
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    volumes=VOLUMES,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=16 * 1024,
)
def upload_checkpoint(
    step: int | None = None,
    exp_name: str = "bc-coldstart",
    repo_id: str = "markmusic/pi05-driving-bc-checkpoint",
):
    import os

    from huggingface_hub import HfApi

    ckpt_base = f"{CACHE_DIR}/checkpoints/pi05_driving/{exp_name}"

    if not os.path.exists(ckpt_base):
        print(f"No checkpoint dir at {ckpt_base}")
        # List what exists
        for root, dirs, files in os.walk(f"{CACHE_DIR}/checkpoints"):
            for d in dirs:
                print(f"  {os.path.join(root, d)}")
        return

    # Find the checkpoint step
    if step is None:
        steps = sorted(
            [int(d) for d in os.listdir(ckpt_base) if d.isdigit()],
            reverse=True,
        )
        if not steps:
            print("No checkpoint steps found")
            return
        step = steps[0]
        print(f"Using latest checkpoint at step {step}")

    params_dir = os.path.join(ckpt_base, str(step), "params")
    if not os.path.exists(params_dir):
        print(f"No params dir at {params_dir}")
        return

    print(f"Uploading checkpoint step {step} to {repo_id}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=params_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Checkpoint at step {step}",
    )
    print(f"Uploaded to https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Patch openpi with driving config
# ---------------------------------------------------------------------------


def _patch_openpi():
    """Copy our driving config patches into the openpi repo."""
    import shutil

    # 1. Copy driving_policy.py
    driving_policy_src = "/opt/driving_policy.py"
    driving_policy_dst = f"{OPENPI_DIR}/src/openpi/policies/driving_policy.py"

    # Write driving_policy.py inline since we can't mount from the host
    with open(driving_policy_dst, "w") as f:
        f.write('''"""Data transforms for π0.5 driving policy (Cart FSD)."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DrivingInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
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
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DrivingOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :2])}
''')

    # 2. Patch gemma.py to add gemma_2b_lora_driving variant
    gemma_path = f"{OPENPI_DIR}/src/openpi/models/gemma.py"
    with open(gemma_path, "r") as f:
        content = f.read()

    if "gemma_2b_lora_driving" not in content:
        # Add variant to Literal type
        content = content.replace(
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]',
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_2b_lora_driving"]',
        )
        # Add config block before gemma_300m_lora
        content = content.replace(
            '    if variant == "gemma_300m_lora":',
            '''    if variant == "gemma_2b_lora_driving":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=64.0), "ffn": lora.LoRAConfig(rank=32, alpha=64.0)},
        )
    if variant == "gemma_300m_lora":''',
        )
        with open(gemma_path, "w") as f:
            f.write(content)

    # 3. Patch config.py to add driving config
    config_path = f"{OPENPI_DIR}/src/openpi/training/config.py"
    with open(config_path, "r") as f:
        content = f.read()

    if "pi05_driving" not in content:
        # Add import
        content = content.replace(
            "import openpi.policies.droid_policy as droid_policy",
            "import openpi.policies.driving_policy as driving_policy\nimport openpi.policies.droid_policy as droid_policy",
        )

        # Add LeRobotDrivingDataConfig class before TrainConfig
        driving_data_config = '''
@dataclasses.dataclass(frozen=True)
class LeRobotDrivingDataConfig(DataConfigFactory):
    """Data config for Cart FSD driving with pi0.5."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.images.front": "observation/image",
                        "observation.state": "observation/state",
                        "action": "actions",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[driving_policy.DrivingInputs(model_type=model_config.model_type)],
            outputs=[driving_policy.DrivingOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

'''
        content = content.replace(
            "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
            driving_data_config + "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
        )

        # Add training config entry before closing bracket
        driving_train_config = '''
    #
    # Cart FSD driving config.
    #
    TrainConfig(
        name="pi05_driving",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=2,
            action_horizon=64,
            paligemma_variant="gemma_2b_lora_driving",
            action_expert_variant="gemma_300m",
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
            paligemma_variant="gemma_2b_lora_driving",
            action_expert_variant="gemma_300m",
        ).get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=390,
            peak_lr=3e-5,
            decay_steps=7_800,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.999,
            clip_gradient_norm=1.0,
        ),
        num_train_steps=7_800,
        batch_size=96,
        fsdp_devices=1,
        save_interval=250,
        log_interval=50,
        checkpoint_base_dir="/cache/checkpoints",
    ),
'''
        # Insert before the closing bracket of _CONFIGS
        import re
        content = content.replace(
            "    *polaris_config.get_polaris_configs(),\n]",
            "    *polaris_config.get_polaris_configs()," + driving_train_config + "]",
        )

        with open(config_path, "w") as f:
            f.write(content)

    print("openpi patched with driving config")
