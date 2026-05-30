"""Build LeRobot v2.1 dataset from extracted driving data and push to HuggingFace.

Uses the EXACT same openpi environment (including its pinned lerobot + datasets
versions) that training will use. This guarantees parquet metadata compatibility —
the same `datasets` library version writes and reads the files.

Usage:
    modal run pi05/modal_build_dataset.py::build_lerobot_dataset --scale large
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-build-dataset"
CACHE_DIR = "/cache"
OPENPI_DIR = "/opt/openpi"

build_image = (
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
    .env({"HF_HOME": f"{CACHE_DIR}/hf"})
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}

app = modal.App(APP_NAME)


@app.function(
    image=build_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 12,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def build_lerobot_dataset(
    scale: str = "tiny",
    repo_id: str = "markmusic/pi05-physical-av-bc",
    push: bool = True,
    checkpoint_interval: int = 1000,
    max_samples: int = 0,
):
    import json
    import os
    import shutil
    import subprocess
    import tempfile

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    samples_path = f"{output_dir}/samples.parquet"

    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"No extracted data at {samples_path}. Run extract_driving_data first."
        )

    build_script = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    )
    build_script.write(f'''
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

output_dir = "{output_dir}"
repo_id = "{repo_id}"
push = {push}
cache_dir = "{CACHE_DIR}"
checkpoint_interval = {checkpoint_interval}
max_samples = {max_samples}

samples_path = f"{{output_dir}}/samples.parquet"
df = pd.read_parquet(samples_path)
print(f"Loaded {{len(df)}} samples from {{samples_path}}")

clip_ids = df["clip_id"].unique()
n_eval = max(1, int(len(clip_ids) * 0.1))
eval_clips = set(clip_ids[:n_eval])
df["split"] = df["clip_id"].apply(lambda c: "eval" if c in eval_clips else "train")
train_df = df[df["split"] == "train"].reset_index(drop=True)
if max_samples > 0 and len(train_df) > max_samples:
    train_df = train_df.iloc[:max_samples]
    print(f"Capped to {{max_samples}} train samples")
print(f"Train samples: {{len(train_df)}}, Eval: {{(df['split'] == 'eval').sum()}}")

import datasets
print(f"datasets version: {{datasets.__version__}}")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# --- Resume support ---
progress_file = f"{{output_dir}}/build_progress.json"
start_idx = 0

local_path = f"{{cache_dir}}/hf/lerobot/{{repo_id}}"

if os.path.exists(progress_file):
    with open(progress_file) as f:
        progress = json.load(f)
    start_idx = progress.get("next_idx", 0)
    if start_idx > 0 and os.path.exists(local_path):
        print(f"Resuming from sample {{start_idx}}/{{len(train_df)}}")
        dataset = LeRobotDataset(repo_id)
        print(f"Loaded existing dataset with {{len(dataset)}} frames")
    else:
        start_idx = 0

if start_idx == 0:
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        print(f"Cleaned up stale dataset at {{local_path}}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="cart_fsd",
        fps=10,
        features={{
            "observation.images.front": {{
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            }},
            "observation.state": {{
                "dtype": "float32",
                "shape": (2,),
                "names": ["speed", "heading_rate"],
            }},
            "action": {{
                "dtype": "float32",
                "shape": (128,),
                "names": None,
            }},
        }},
        image_writer_threads=20,
    )

t_start = time.time()

for idx in tqdm(range(start_idx, len(train_df)), desc="Building", initial=start_idx, total=len(train_df)):
    row = train_df.iloc[idx]
    img_path = f"{{output_dir}}/{{row['image_path']}}"
    img = np.array(Image.open(img_path))

    actions = np.stack([np.array(a, dtype=np.float32) for a in row["actions"]])
    actions_flat = actions.flatten()
    state = np.array([row["speed"], row["heading_rate"]], dtype=np.float32)

    task = row["nav_prompt"]
    dataset.add_frame({{
        "observation.images.front": img,
        "observation.state": state,
        "action": actions_flat,
        "task": task,
    }})
    dataset.save_episode()

    # Checkpoint: save progress + commit volume
    if (idx + 1) % checkpoint_interval == 0:
        # Flush pending image writes
        if hasattr(dataset, 'image_writer') and dataset.image_writer is not None:
            dataset.image_writer.wait_until_done()

        with open(progress_file, "w") as f:
            json.dump({{"next_idx": idx + 1, "total": len(train_df)}}, f)

        import modal
        vol = modal.Volume.from_name("pi05-cache")
        vol.commit()

        elapsed = time.time() - t_start
        rate = (idx + 1 - start_idx) / elapsed
        remaining = (len(train_df) - idx - 1) / rate if rate > 0 else 0
        print(f"  Checkpoint: {{idx + 1}}/{{len(train_df)}} samples, "
              f"{{rate:.1f}} samples/s, ~{{remaining/60:.0f}} min remaining")

print(f"Dataset built: {{len(dataset)}} frames (1 per sample)")
elapsed = time.time() - t_start
print(f"Build time: {{elapsed:.0f}}s ({{elapsed/60:.1f}} min)")

if push:
    print("Pushing to HuggingFace...")
    dataset.push_to_hub(tags=["driving", "pi05", "cart-fsd"])
    print(f"Pushed to {{repo_id}}")

# Clean up progress file
if os.path.exists(progress_file):
    os.remove(progress_file)

print("BUILD COMPLETE")
''')
    build_script.flush()
    script_path = build_script.name
    build_script.close()

    venv_python = f"{OPENPI_DIR}/.venv/bin/python"
    print(f"Running build script with {venv_python}")

    result = subprocess.run(
        [venv_python, script_path],
        text=True,
        env={**os.environ, "HF_HOME": f"{CACHE_DIR}/hf"},
    )

    cache_volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Build script failed with return code {result.returncode}")

    return "Dataset built successfully"
