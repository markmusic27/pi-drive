"""Extract PhysicalAI-AV ground truth egomotion for π0.5 BC training.

Step 1: Extract GT egomotion → (accel, curvature) action labels.
Step 1b: Label navigation intent with Gemini Flash (VLM-based).
Step 2: Build LeRobot v2 dataset and push to HuggingFace.

Usage:
    modal run pi05/modal_extract_data.py::extract_driving_data --scale tiny
    modal run pi05/modal_extract_data.py::label_nav_prompts --scale tiny
    modal run pi05/modal_extract_data.py::build_lerobot_dataset
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-extract-data"
CACHE_DIR = "/cache"
ALPAMAYO_DIR = "/opt/alpamayo"

# ---------------------------------------------------------------------------
# Image: alpamayo (for traj_to_action) + physical_ai_av (for data loading)
# ---------------------------------------------------------------------------

extract_image = (
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
    .run_commands(
        f"git clone --depth 1 https://github.com/NVlabs/alpamayo.git {ALPAMAYO_DIR}",
        f"sed -i '/cosmos-rl/d; /vllm/d; /flash-attn/d; /deepspeed/d; /liger-kernel/d; /torchao/d' {ALPAMAYO_DIR}/pyproject.toml",
        f"pip install -e {ALPAMAYO_DIR}",
    )
    .pip_install("pandas", "pyarrow", "Pillow", "tqdm")
    .env({"HF_HOME": f"{CACHE_DIR}/hf"})
)

label_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "google-genai",
        "pandas",
        "pyarrow",
        "Pillow",
        "tqdm",
    )
)

# ---------------------------------------------------------------------------
# Volumes & App
# ---------------------------------------------------------------------------

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Data filtering constants
# ---------------------------------------------------------------------------

RIGHT_HAND_TRAFFIC = [
    "United States",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Netherlands",
    "China",
    "Canada",
    "Mexico",
    "Brazil",
    "Poland",
    "Sweden",
    "Norway",
    "Denmark",
    "Finland",
    "Austria",
    "Switzerland",
    "Belgium",
    "Czech Republic",
    "Portugal",
]

SCALES = {
    "tiny": 50,
    "small": 500,
    "medium": 5_000,
    "large": 20_000,
}

T0_OFFSETS_US = [
    3_000_000,   # 3s into clip
    5_000_000,   # 5s
    7_000_000,   # 7s
    9_000_000,   # 9s
    11_000_000,  # 11s
]


# ---------------------------------------------------------------------------
# Step 1: Extract GT egomotion + images + nav prompts
# ---------------------------------------------------------------------------


@app.function(
    image=extract_image,
    volumes=VOLUMES,
    gpu="T4",
    timeout=60 * 60 * 8,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def extract_driving_data(scale: str = "tiny", seed: int = 42, eval_frac: float = 0.1):
    import json
    import os
    import time

    import numpy as np
    import pandas as pd
    import physical_ai_av
    import torch
    from PIL import Image
    from tqdm import tqdm

    from alpamayo_r1.action_space.unicycle_accel_curvature import (
        UnicycleAccelCurvatureActionSpace,
    )
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    import sys
    sys.path.insert(0, "/opt/alpamayo/..")
    # nav_prompt is in our repo, but we inline it here for Modal isolation
    def trajectory_to_nav_prompt(ego_future_xyz):
        forward = ego_future_xyz[-1, 0]
        lateral = ego_future_xyz[-1, 1]
        if forward < 2.0:
            return "stop"
        if abs(lateral) < 1.0:
            return "drive forward"
        if lateral > 3.0:
            return "turn left"
        if lateral < -3.0:
            return "turn right"
        if lateral > 0:
            return "bear left"
        return "bear right"

    max_clips = SCALES[scale]
    print(f"=== Extracting {scale} scale: {max_clips} clips ===")

    # Initialize dataset interface
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    print(f"Dataset initialized, {len(avdi.clip_index)} total clips")

    # --- Filter clips ---
    clip_index = avdi.clip_index.copy()

    # Try loading data_collection metadata for country/time filtering
    try:
        metadata_dir = os.path.dirname(avdi._clip_index_path)
        dc_path = os.path.join(metadata_dir, "data_collection.parquet")
        fp_path = os.path.join(metadata_dir, "metadata", "feature_presence.parquet")

        if os.path.exists(dc_path):
            metadata = pd.read_parquet(dc_path)
            filtered = metadata[
                (metadata["country"].isin(RIGHT_HAND_TRAFFIC))
                & (metadata["time_of_day"] == "daytime")
            ]
            valid_clips = set(filtered["clip_id"].tolist())
            print(f"  Country+daylight filter: {len(valid_clips)} clips")
        else:
            valid_clips = set(clip_index.index.tolist())
            print("  No data_collection.parquet, skipping country/daylight filter")

        if os.path.exists(fp_path):
            features = pd.read_parquet(fp_path)
            has_sensors = features[
                (features.get("has_egomotion", True) == True)
                & (features.get("has_camera_front_wide", True) == True)
            ]
            valid_clips &= set(has_sensors["clip_id"].tolist())
            print(f"  Sensor filter: {len(valid_clips)} clips")
    except Exception as e:
        print(f"  Warning: metadata filtering failed ({e}), using all clips")
        valid_clips = set(clip_index.index.tolist())

    available_clips = [c for c in clip_index.index if c in valid_clips]
    print(f"  Available after filtering: {len(available_clips)} clips")

    # Sample clips
    rng = np.random.default_rng(seed)
    if len(available_clips) > max_clips:
        selected_clips = rng.choice(available_clips, size=max_clips, replace=False).tolist()
    else:
        selected_clips = available_clips[:max_clips]

    # Split into train/eval
    n_eval = max(1, int(len(selected_clips) * eval_frac))
    rng.shuffle(selected_clips)
    eval_clips = set(selected_clips[:n_eval])
    train_clips = set(selected_clips[n_eval:])
    print(f"  Selected {len(selected_clips)} clips: {len(train_clips)} train, {len(eval_clips)} eval")

    # Initialize action space converter
    action_space = UnicycleAccelCurvatureActionSpace()

    # Extract samples
    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    samples = []
    errors = []
    t_start = time.time()

    for clip_idx, clip_id in enumerate(tqdm(selected_clips, desc="Extracting")):
        for t0_us in T0_OFFSETS_US:
            sample_id = f"{clip_id}_{t0_us}"
            try:
                # Load egomotion + front camera only
                data = load_physical_aiavdataset(
                    clip_id,
                    t0_us=t0_us,
                    avdi=avdi,
                    maybe_stream=True,
                    camera_features=[avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV],
                    num_frames=1,  # single frame at t0
                )

                # Speed filter: skip near-stationary
                ego_hist = data["ego_history_xyz"][0, 0].numpy()  # (16, 3)
                dt = 0.1
                velocities = np.linalg.norm(np.diff(ego_hist, axis=0), axis=1) / dt
                speed_at_t0 = velocities[-1]
                if speed_at_t0 < 1.0:
                    continue

                # Trajectory length filter
                ego_fut = data["ego_future_xyz"][0, 0].numpy()  # (64, 3)
                traj_length = np.sum(np.linalg.norm(np.diff(ego_fut, axis=0), axis=1))
                if traj_length < 10.0:
                    continue

                # Convert trajectory → (accel, curvature)
                actions = action_space.traj_to_action(
                    traj_history_xyz=data["ego_history_xyz"],
                    traj_history_rot=data["ego_history_rot"],
                    traj_future_xyz=data["ego_future_xyz"],
                    traj_future_rot=data["ego_future_rot"],
                )  # (1, 1, 64, 2)
                actions_np = actions[0, 0].numpy()  # (64, 2)

                # Trajectory geometry for VLM labeling context
                lateral_disp = float(ego_fut[-1, 1])  # positive = left
                forward_disp = float(ego_fut[-1, 0])
                from alpamayo_r1.geometry.rotation import so3_to_yaw_torch
                fut_yaws = so3_to_yaw_torch(data["ego_future_rot"][0, 0]).numpy()
                heading_change_deg = float(np.degrees(fut_yaws[-1] - fut_yaws[0]))

                # Placeholder nav prompt (will be replaced by VLM in label_nav_prompts)
                nav_prompt = trajectory_to_nav_prompt(ego_fut)

                # Compute state: [speed, heading_rate]
                heading_rate = velocities[-1] - velocities[-2] if len(velocities) > 1 else 0.0
                # Better heading rate: use rotation
                ego_hist_rot = data["ego_history_rot"][0, 0].numpy()  # (16, 3, 3)
                from alpamayo_r1.geometry.rotation import so3_to_yaw_torch
                yaws = so3_to_yaw_torch(data["ego_history_rot"][0, 0]).numpy()
                heading_rate = (yaws[-1] - yaws[-2]) / dt if len(yaws) > 1 else 0.0
                state = np.array([speed_at_t0, heading_rate], dtype=np.float32)

                # Save front camera image
                # image_frames: (1, 1, 3, H, W) → (H, W, 3) uint8
                img_tensor = data["image_frames"][0, 0]  # (3, H, W)
                img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                # Downsample to 480×640
                img_pil = Image.fromarray(img_np)
                img_pil = img_pil.resize((640, 480), Image.LANCZOS)
                img_path = f"{output_dir}/images/{sample_id}.jpg"
                img_pil.save(img_path, quality=90)

                samples.append({
                    "sample_id": sample_id,
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                    "split": "eval" if clip_id in eval_clips else "train",
                    "nav_prompt": nav_prompt,
                    "speed": float(state[0]),
                    "heading_rate": float(state[1]),
                    "lateral_disp": lateral_disp,
                    "forward_disp": forward_disp,
                    "heading_change_deg": heading_change_deg,
                    "actions": actions_np.tolist(),
                    "image_path": f"images/{sample_id}.jpg",
                })

            except Exception as e:
                errors.append({"clip_id": clip_id, "t0_us": t0_us, "error": str(e)})
                if len(errors) <= 5:
                    print(f"  Error on {clip_id} t0={t0_us}: {e}")

        if (clip_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (clip_idx + 1) / elapsed
            print(f"  {clip_idx+1}/{len(selected_clips)} clips, "
                  f"{len(samples)} samples, {len(errors)} errors, "
                  f"{rate:.1f} clips/s")

    # Save metadata
    df = pd.DataFrame(samples)
    df.to_parquet(f"{output_dir}/samples.parquet", index=False)

    # Save error log
    if errors:
        pd.DataFrame(errors).to_parquet(f"{output_dir}/errors.parquet", index=False)

    # Nav prompt distribution
    if len(df) > 0:
        print(f"\n=== Results ===")
        print(f"Total samples: {len(df)}")
        print(f"Train: {(df['split'] == 'train').sum()}, Eval: {(df['split'] == 'eval').sum()}")
        print(f"Errors: {len(errors)}")
        print(f"\nNav prompt distribution:")
        print(df["nav_prompt"].value_counts().to_string())
        print(f"\nSpeed stats: mean={df['speed'].mean():.2f}, "
              f"std={df['speed'].std():.2f}, "
              f"min={df['speed'].min():.2f}, max={df['speed'].max():.2f}")

    cache_volume.commit()
    print(f"\nSaved to {output_dir}")
    print(f"Total time: {time.time() - t_start:.1f}s")
    return len(df)


# ---------------------------------------------------------------------------
# Step 1b: Label navigation intent with Gemini Flash
# ---------------------------------------------------------------------------

NAV_CATEGORIES = [
    "continue straight",
    "turn left",
    "turn right",
    "change lanes left",
    "change lanes right",
    "stop",
    "u-turn",
]

NAV_PROMPT_TEMPLATE = """Look at this driving scene from a front-facing camera. The vehicle is traveling at {speed:.1f} m/s ({speed_mph:.0f} mph).

Over the next 6.4 seconds, the vehicle:
- Moved {forward:.1f}m forward
- Drifted {lateral_abs:.1f}m to the {lateral_dir}
- Changed heading by {heading:.1f}° to the {heading_dir}

Based on the VISIBLE ROAD GEOMETRY in the image combined with the trajectory info, classify the navigation intent as exactly one of:
- "continue straight" — following the road, even if it curves or winds
- "turn left" — turning left at an intersection, junction, or decision point
- "turn right" — turning right at an intersection, junction, or decision point
- "change lanes left" — changing to a lane on the left
- "change lanes right" — changing to a lane on the right
- "stop" — vehicle is stopping
- "u-turn" — making a U-turn

IMPORTANT: If the road curves but there is NO intersection or fork visible, the intent is "continue straight". Only use turn labels when there is an actual decision point visible in the image.

Respond with ONLY the category name, nothing else."""


@app.function(
    image=label_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 4,
    memory=8 * 1024,
    secrets=[modal.Secret.from_name("google-ai")],
)
def label_nav_prompts(scale: str = "tiny", batch_size: int = 16):
    import base64
    import os
    import time

    import pandas as pd
    from google import genai
    from PIL import Image
    from tqdm import tqdm

    client = genai.Client()

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    samples_path = f"{output_dir}/samples.parquet"

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"No extracted data at {samples_path}")

    df = pd.read_parquet(samples_path)
    print(f"Loaded {len(df)} samples for VLM labeling")

    labeled = []
    errors = []
    t_start = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        img_path = f"{output_dir}/{row['image_path']}"
        if not os.path.exists(img_path):
            errors.append({"idx": idx, "error": "image not found"})
            labeled.append(row["nav_prompt"])
            continue

        lateral = row.get("lateral_disp", 0.0)
        heading = row.get("heading_change_deg", 0.0)
        forward = row.get("forward_disp", 0.0)

        prompt = NAV_PROMPT_TEMPLATE.format(
            speed=row["speed"],
            speed_mph=row["speed"] * 2.237,
            forward=forward,
            lateral_abs=abs(lateral),
            lateral_dir="left" if lateral > 0 else "right",
            heading=abs(heading),
            heading_dir="left" if heading > 0 else "right",
        )

        try:
            img = Image.open(img_path)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[img, prompt],
            )

            raw = response.text.strip().strip('"').lower()
            if raw in NAV_CATEGORIES:
                labeled.append(raw)
            else:
                for cat in NAV_CATEGORIES:
                    if cat in raw:
                        labeled.append(cat)
                        break
                else:
                    print(f"  Warning: unexpected VLM output '{raw}' for {row['sample_id']}, keeping trajectory-based label")
                    labeled.append(row["nav_prompt"])

        except Exception as e:
            errors.append({"idx": idx, "error": str(e)})
            labeled.append(row["nav_prompt"])
            if len(errors) <= 5:
                print(f"  Error on {row['sample_id']}: {e}")
            time.sleep(1)

    df["nav_prompt_traj"] = df["nav_prompt"]
    df["nav_prompt"] = labeled

    df.to_parquet(samples_path, index=False)

    print(f"\n=== VLM Labeling Results ===")
    print(f"Total: {len(df)}, Errors: {len(errors)}")
    print(f"\nVLM nav prompt distribution:")
    print(pd.Series(labeled).value_counts().to_string())
    print(f"\nOriginal trajectory-based distribution:")
    print(df["nav_prompt_traj"].value_counts().to_string())
    print(f"\nAgreement rate: {sum(a == b for a, b in zip(labeled, df['nav_prompt_traj'])) / len(df):.1%}")
    print(f"Total time: {time.time() - t_start:.1f}s")

    cache_volume.commit()
    return len(df) - len(errors)


# Step 2 (LeRobot dataset build) is in modal_build_dataset.py
