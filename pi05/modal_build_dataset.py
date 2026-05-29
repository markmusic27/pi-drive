"""Build LeRobot v2 dataset from extracted driving data and push to HuggingFace.

Usage:
    modal run pi05/modal_build_dataset.py::build_lerobot_dataset --scale tiny
    modal run pi05/modal_build_dataset.py::build_lerobot_dataset --scale medium
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-build-dataset"
CACHE_DIR = "/cache"

lerobot_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs", "build-essential", "linux-libc-dev")
    .pip_install(
        "torch==2.8.0",
        "torchvision>=0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "pip install 'lerobot[dataset] @ git+https://github.com/huggingface/lerobot.git' pandas pyarrow Pillow tqdm huggingface_hub",
    )
    .env({"HF_HOME": f"{CACHE_DIR}/hf"})
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}

app = modal.App(APP_NAME)


@app.function(
    image=lerobot_image,
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

    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    samples_path = f"{output_dir}/samples.parquet"

    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"No extracted data at {samples_path}. Run extract_driving_data first."
        )

    df = pd.read_parquet(samples_path)
    print(f"Loaded {len(df)} samples from {samples_path}")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    print(f"Train samples: {len(train_df)}")

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

    import shutil
    local_path = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        print(f"Cleaned up stale dataset at {local_path}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="cart_fsd",
        fps=10,
        features={
            "observation.images.front": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["speed", "heading_rate"],
            },
            "action": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["acceleration", "curvature"],
            },
        },
        image_writer_threads=10,
    )

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building"):
        img_path = f"{output_dir}/{row['image_path']}"
        img = Image.open(img_path)

        actions = np.stack([np.array(a, dtype=np.float32) for a in row["actions"]])
        state = np.array([row["speed"], row["heading_rate"]], dtype=np.float32)

        task = row["nav_prompt"]
        for t in range(len(actions)):
            dataset.add_frame({
                "observation.images.front": np.array(img),
                "observation.state": state,
                "action": actions[t],
                "task": task,
            })

        dataset.save_episode()

    print(f"Dataset built: {len(dataset)} frames")

    if push:
        dataset.push_to_hub(tags=["driving", "pi05", "cart-fsd"])
        print(f"Pushed to {repo_id}")

    cache_volume.commit()
    return len(train_df)
