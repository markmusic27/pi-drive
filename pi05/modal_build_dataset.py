"""Build LeRobot v2.1 dataset from extracted driving data and push to HuggingFace.

Uses the EXACT same openpi environment (including its pinned lerobot + datasets
versions) that training will use. This guarantees parquet metadata compatibility —
the same `datasets` library version writes and reads the files.

Usage:
    modal run pi05/modal_build_dataset.py::build_lerobot_dataset --scale tiny
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-build-dataset"
CACHE_DIR = "/cache"
OPENPI_DIR = "/opt/openpi"

# Use the same image as training — clone openpi, uv sync — so lerobot and
# datasets versions match exactly.
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
    timeout=60 * 60 * 2,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def build_lerobot_dataset(
    scale: str = "tiny",
    repo_id: str = "markmusic/pi05-physical-av-bc",
    push: bool = True,
):
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    samples_path = f"{output_dir}/samples.parquet"

    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"No extracted data at {samples_path}. Run extract_driving_data first."
        )

    # Write a build script that runs inside openpi's venv
    build_script = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    )
    build_script.write(f'''
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

output_dir = "{output_dir}"
repo_id = "{repo_id}"
push = {push}
cache_dir = "{CACHE_DIR}"

samples_path = f"{{output_dir}}/samples.parquet"
df = pd.read_parquet(samples_path)
print(f"Loaded {{len(df)}} samples from {{samples_path}}")

train_df = df[df["split"] == "train"].reset_index(drop=True)
print(f"Train samples: {{len(train_df)}}")

# Check datasets version for debugging
import datasets
print(f"datasets version: {{datasets.__version__}}")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

local_path = f"{{cache_dir}}/hf/lerobot/{{repo_id}}"
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
            "shape": (2,),
            "names": ["acceleration", "curvature"],
        }},
    }},
    image_writer_threads=10,
)

for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building"):
    img_path = f"{{output_dir}}/{{row['image_path']}}"
    img = Image.open(img_path)

    actions = np.stack([np.array(a, dtype=np.float32) for a in row["actions"]])
    state = np.array([row["speed"], row["heading_rate"]], dtype=np.float32)

    task = row["nav_prompt"]
    for t in range(len(actions)):
        dataset.add_frame({{
            "observation.images.front": np.array(img),
            "observation.state": state,
            "action": actions[t],
            "task": task,
        }})

    dataset.save_episode()

print(f"Dataset built: {{len(dataset)}} frames")

if push:
    dataset.push_to_hub(tags=["driving", "pi05", "cart-fsd"])
    print(f"Pushed to {{repo_id}}")

print("BUILD COMPLETE")
''')
    build_script.flush()
    script_path = build_script.name
    build_script.close()

    # Run the build script inside openpi's venv
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
