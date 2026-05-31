"""Extract PhysicalAI-AV ground truth egomotion for π0.5 BC training.

Two-phase pipeline:
  1. Preload: bulk download dataset to Modal volume (snapshot_download, no rate limits)
  2. Extract: parallel extraction from local volume (fast, no HF API calls)

Usage:
    # Phase 1: Bulk download to Modal volume (run once, ~1-2 hours)
    modal run --detach pi05/modal_extract_data.py::preload_dataset --n-clips 50000

    # Phase 2: Extract from local volume (fast, ~30-60 min with 25 workers)
    modal run --detach pi05/modal_extract_data.py::extract_parallel --scale xlarge

    # Legacy: direct extraction with HF streaming (slower, rate-limited)
    modal run pi05/modal_extract_data.py::extract_driving_data --scale tiny
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

DATASET_DIR = f"{CACHE_DIR}/physical_ai_av"
DATASET_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"

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
    "xlarge": 50_000,
    "full": 163_000,
}

T0_OFFSETS_US = [
    3_000_000,   # 3s into clip
    5_000_000,   # 5s
    7_000_000,   # 7s
    9_000_000,   # 9s
    11_000_000,  # 11s
]


# ---------------------------------------------------------------------------
# Local dataset interface (for preloaded data)
# ---------------------------------------------------------------------------


def _create_local_avdi(local_dir=None):
    """Create local PAI interface from pre-downloaded dataset on volume."""
    import os
    import pandas as pd
    from alpamayo_r1.data.pai_utils import PhysicalAIAVDatasetLocalInterface

    if local_dir is None:
        local_dir = DATASET_DIR
    avdi = PhysicalAIAVDatasetLocalInterface(local_dir=local_dir)
    avdi.feature_presence = avdi.sensor_presence
    dc_path = os.path.join(local_dir, "metadata", "data_collection.parquet")
    if os.path.exists(dc_path):
        avdi.data_collection = pd.read_parquet(dc_path)
    return avdi


# ---------------------------------------------------------------------------
# Preload: bulk download PhysicalAI-AV to Modal volume
# ---------------------------------------------------------------------------


@app.function(
    image=extract_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 4,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def _download_chunk_batch(chunk_ids_list: list[int], batch_idx: int) -> dict:
    """Download a batch of chunks to the shared volume. Called in parallel by preload_dataset."""
    import os
    import time

    from huggingface_hub import snapshot_download

    t0 = time.time()
    patterns = []
    for cid in chunk_ids_list:
        cs = f"chunk_{cid:04d}"
        patterns.append(f"camera/camera_front_wide_120fov/camera_front_wide_120fov.{cs}.*")
        patterns.append(f"labels/egomotion/egomotion.{cs}.*")

    print(f"[Worker {batch_idx}] Downloading {len(chunk_ids_list)} chunks ({len(patterns)} files)...")
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=DATASET_DIR,
        local_dir_use_symlinks=False,
    )
    cache_volume.commit()

    elapsed = time.time() - t0
    print(f"[Worker {batch_idx}] Done — {len(chunk_ids_list)} chunks in {elapsed/60:.1f} min")
    return {"batch_idx": batch_idx, "n_chunks": len(chunk_ids_list), "elapsed": elapsed}


@app.function(
    image=extract_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 6,
    memory=64 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def preload_dataset(n_clips: int = 50_000, seed: int = 42, eval_frac: float = 0.15,
                    n_workers: int = 10, chunks_per_worker: int = 280):
    """Bulk download PhysicalAI-AV camera + egomotion to Modal volume.

    Parallelizes across n_workers Modal containers, each downloading a batch of chunks.
    Each worker commits to the volume independently so progress is never lost.

    Usage:
        modal run --detach pi05/modal_extract_data.py::preload_dataset --n-clips 50000
    """
    import os
    import time

    import numpy as np
    import pandas as pd
    from huggingface_hub import snapshot_download

    t_start = time.time()

    # Step 1: Download metadata (small, fast)
    print("=== Step 1: Downloading metadata ===")
    snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["features.csv", "clip_index.parquet", "metadata/**"],
        local_dir=DATASET_DIR,
        local_dir_use_symlinks=False,
    )
    cache_volume.commit()
    print(f"  Metadata downloaded in {time.time() - t_start:.0f}s")

    # Step 2: Filter clips and identify needed chunks
    print("=== Step 2: Filtering clips ===")
    avdi = _create_local_avdi()
    filtered = _filter_clips(avdi)
    print(f"  {len(filtered)} clips pass filters")

    rng = np.random.default_rng(seed)
    if len(filtered) > n_clips:
        selected = rng.choice(filtered, size=n_clips, replace=False).tolist()
    else:
        selected = list(filtered)
    rng.shuffle(selected)

    # Train/eval split at clip level
    n_eval = max(1, int(len(selected) * eval_frac))
    eval_clips = selected[:n_eval]
    train_clips = selected[n_eval:]
    print(f"  Selected {len(selected)} clips: {len(train_clips)} train, {n_eval} eval")

    # Find needed chunks
    chunk_ids = set()
    for clip_id in selected:
        try:
            chunk_val = avdi.clip_index.at[clip_id, "chunk"]
            if isinstance(chunk_val, str):
                chunk_ids.add(int(chunk_val.replace("chunk_", "").replace("chunk", "")))
            else:
                chunk_ids.add(int(chunk_val))
        except (KeyError, ValueError):
            pass
    print(f"  Clips span {len(chunk_ids)} chunks")

    # Save clip lists for extraction
    preload_dir = f"{CACHE_DIR}/preload"
    os.makedirs(preload_dir, exist_ok=True)
    pd.DataFrame({"clip_id": selected}).to_parquet(f"{preload_dir}/selected_clips.parquet")
    pd.DataFrame({"clip_id": eval_clips}).to_parquet(f"{preload_dir}/eval_clips.parquet")
    cache_volume.commit()

    # Step 3: Parallel download across workers
    chunk_list = sorted(chunk_ids)
    batches = [chunk_list[i:i + chunks_per_worker]
               for i in range(0, len(chunk_list), chunks_per_worker)]
    print(f"=== Step 3: Downloading {len(chunk_ids)} chunks across {len(batches)} workers ===")

    results = list(_download_chunk_batch.map(
        batches,
        list(range(len(batches))),
    ))

    total_chunks = sum(r["n_chunks"] for r in results)
    print(f"  All workers done — {total_chunks} chunks downloaded")

    # Verify download
    cache_volume.reload()
    cam_dir = os.path.join(DATASET_DIR, "camera", "camera_front_wide_120fov")
    ego_dir = os.path.join(DATASET_DIR, "labels", "egomotion")
    n_cam = len(os.listdir(cam_dir)) if os.path.isdir(cam_dir) else 0
    n_ego = len(os.listdir(ego_dir)) if os.path.isdir(ego_dir) else 0
    print(f"  Verified: {n_cam} camera files, {n_ego} egomotion files")

    with open(f"{preload_dir}/done", "w") as f:
        f.write(f"{len(selected)} clips, {len(chunk_ids)} chunks\n")
        f.write(f"train: {len(train_clips)}, eval: {n_eval}\n")

    cache_volume.commit()
    elapsed = time.time() - t_start
    print(f"\n=== Preload complete in {elapsed/60:.1f} min ===")
    print(f"  {len(selected)} clips across {len(chunk_ids)} chunks")
    print(f"  Run extraction next: modal run --detach pi05/modal_extract_data.py::extract_parallel --scale xlarge")
    return len(selected)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _trajectory_to_nav_prompt(ego_future_xyz):
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


def _filter_clips(avdi):
    """Filter clip_index for right-hand-traffic, daytime, valid sensors."""
    import pandas as pd

    clip_index = avdi.clip_index.copy()
    all_clip_ids = set(clip_index.index.tolist())
    valid_clips = all_clip_ids

    # Use avdi's built-in data_collection DataFrame
    try:
        dc = avdi.data_collection
        print(f"  data_collection: {len(dc)} rows, columns: {dc.columns.tolist()}")

        # Find the country column (might be 'country' or 'Country' etc.)
        country_col = None
        tod_col = None
        for col in dc.columns:
            if col.lower() == "country":
                country_col = col
            if col.lower() in ("time_of_day", "timeofday", "tod"):
                tod_col = col

        filters = pd.Series(True, index=dc.index)

        if country_col:
            filters &= dc[country_col].isin(RIGHT_HAND_TRAFFIC)
            print(f"  Country filter ({country_col}): {filters.sum()} clips")

        if tod_col:
            filters &= dc[tod_col] == "daytime"
            print(f"  + Daylight filter ({tod_col}): {filters.sum()} clips")

        # hour_of_day: filter for daytime hours (8 AM - 6 PM)
        hour_col = None
        for col in dc.columns:
            if col.lower() in ("hour_of_day", "hour"):
                hour_col = col
        if hour_col and not tod_col:
            filters &= dc[hour_col].between(8, 18)
            print(f"  + Daylight filter ({hour_col} 8-18): {filters.sum()} clips")

        # clip_id may be a column or the index
        if "clip_id" in dc.columns:
            filtered_ids = set(dc.loc[filters, "clip_id"].tolist())
        else:
            filtered_ids = set(dc.index[filters].tolist())
        valid_clips = filtered_ids & all_clip_ids
        print(f"  Country+daylight filter: {len(valid_clips)} clips")
    except Exception as e:
        print(f"  Warning: data_collection filter failed ({e})")

    # Use avdi's feature_presence for sensor filtering
    try:
        fp = avdi.feature_presence
        print(f"  feature_presence: {len(fp)} rows, columns: {fp.columns.tolist()[:20]}")

        ego_col = None
        cam_col = None
        for col in fp.columns:
            if col.lower() in ("egomotion", "egomotion.offline"):
                ego_col = ego_col or col
            if "camera_front_wide" in col.lower() or "front_wide" in col.lower():
                cam_col = cam_col or col

        if ego_col and cam_col:
            has_data = fp[(fp[ego_col] == True) & (fp[cam_col] == True)]
            if "clip_id" in fp.columns:
                sensor_ids = set(has_data["clip_id"].tolist())
            else:
                sensor_ids = set(has_data.index.tolist())
            valid_clips &= sensor_ids
            print(f"  Sensor filter ({ego_col}, {cam_col}): {len(valid_clips)} clips")
        elif ego_col:
            has_data = fp[fp[ego_col] == True]
            if "clip_id" in fp.columns:
                sensor_ids = set(has_data["clip_id"].tolist())
            else:
                sensor_ids = set(has_data.index.tolist())
            valid_clips &= sensor_ids
            print(f"  Sensor filter ({ego_col} only): {len(valid_clips)} clips")
    except Exception as e:
        print(f"  Warning: feature_presence filter failed ({e})")

    result = [c for c in clip_index.index if c in valid_clips]
    print(f"  Final: {len(result)} clips pass all filters")
    return result


def _load_with_retry(clip_id, t0_us, avdi, max_retries=3):
    """Load clip data with exponential backoff on 429/timeout errors."""
    import glob
    import os
    import time
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    for attempt in range(max_retries):
        try:
            return load_physical_aiavdataset(
                clip_id,
                t0_us=t0_us,
                avdi=avdi,
                maybe_stream=True,
                camera_features=[avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV],
                num_frames=1,
            )
        except Exception as e:
            msg = str(e).lower()
            retryable = "429" in msg or "too many requests" in msg or "timeout" in msg or "file is not a zip" in msg
            if retryable and attempt < max_retries - 1:
                # If bad zip, purge any locally cached zip files for this clip
                if "zip" in msg:
                    hf_home = os.environ.get("HF_HOME", "")
                    if hf_home:
                        for zf in glob.glob(f"{hf_home}/hub/**/*.zip", recursive=True):
                            if clip_id in zf:
                                os.remove(zf)
                wait = 2 ** (attempt + 1) + (attempt * 2)
                time.sleep(wait)
                continue
            raise


def _process_clips(clip_ids, eval_clip_set, scale, avdi=None):
    """Process a list of clips, return sample dicts. Core extraction logic."""
    import os
    import numpy as np
    from PIL import Image

    if avdi is None:
        import physical_ai_av
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    from alpamayo_r1.action_space.unicycle_accel_curvature import (
        UnicycleAccelCurvatureActionSpace,
    )

    action_space = UnicycleAccelCurvatureActionSpace()
    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    samples = []
    errors = []
    dt = 0.1

    for clip_id in clip_ids:
        for t0_us in T0_OFFSETS_US:
            sample_id = f"{clip_id}_{t0_us}"
            try:
                data = _load_with_retry(clip_id, t0_us, avdi)

                ego_hist = data["ego_history_xyz"][0, 0].numpy()
                velocities = np.linalg.norm(np.diff(ego_hist, axis=0), axis=1) / dt
                speed_at_t0 = velocities[-1]
                if speed_at_t0 < 1.0:
                    continue

                ego_fut = data["ego_future_xyz"][0, 0].numpy()
                traj_length = np.sum(np.linalg.norm(np.diff(ego_fut, axis=0), axis=1))
                if traj_length < 10.0:
                    continue

                actions = action_space.traj_to_action(
                    traj_history_xyz=data["ego_history_xyz"],
                    traj_history_rot=data["ego_history_rot"],
                    traj_future_xyz=data["ego_future_xyz"],
                    traj_future_rot=data["ego_future_rot"],
                )
                actions_np = actions[0, 0].numpy()

                lateral_disp = float(ego_fut[-1, 1])
                forward_disp = float(ego_fut[-1, 0])
                from alpamayo_r1.geometry.rotation import so3_to_yaw_torch
                fut_yaws = so3_to_yaw_torch(data["ego_future_rot"][0, 0]).numpy()
                heading_change_deg = float(np.degrees(fut_yaws[-1] - fut_yaws[0]))

                nav_prompt = _trajectory_to_nav_prompt(ego_fut)

                yaws = so3_to_yaw_torch(data["ego_history_rot"][0, 0]).numpy()
                heading_rate = (yaws[-1] - yaws[-2]) / dt if len(yaws) > 1 else 0.0
                state = np.array([speed_at_t0, heading_rate], dtype=np.float32)

                img_tensor = data["image_frames"][0, 0]
                img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_pil = img_pil.resize((640, 480), Image.LANCZOS)
                img_pil.save(f"{output_dir}/images/{sample_id}.jpg", quality=90)

                samples.append({
                    "sample_id": sample_id,
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                    "split": "eval" if clip_id in eval_clip_set else "train",
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

    return samples, errors


# ---------------------------------------------------------------------------
# Parallel extraction (recommended for large scale)
# ---------------------------------------------------------------------------


@app.function(
    image=extract_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 2,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
    max_containers=25,
)
def _extract_clip_batch(
    clip_ids: list[str],
    eval_clip_ids: list[str],
    scale: str,
    batch_idx: int,
    preloaded: bool = False,
) -> dict:
    """Process a batch of clips. Called in parallel by extract_parallel.

    If preloaded=True, reads from local volume (fast, no HF API calls).
    Otherwise uses a local HF cache to avoid 429 corruption.
    """
    import os
    import shutil
    import tempfile
    import time
    from collections import Counter

    if preloaded:
        # Data pre-downloaded to volume — read locally, no HF API calls
        avdi = _create_local_avdi()
        print(f"Batch {batch_idx}: using preloaded local data ({len(clip_ids)} clips)")
    else:
        # Original: local HF cache to avoid 429 corruption
        local_hf = tempfile.mkdtemp(prefix="hf_local_")
        shared_hf = f"{CACHE_DIR}/hf"
        if os.path.isdir(shared_hf):
            shared_hub = os.path.join(shared_hf, "hub")
            if os.path.isdir(shared_hub):
                n_copied = 0
                for dirpath, dirnames, filenames in os.walk(shared_hub):
                    for fname in filenames:
                        if fname.endswith(".zip"):
                            continue
                        src = os.path.join(dirpath, fname)
                        rel = os.path.relpath(src, shared_hf)
                        dst = os.path.join(local_hf, rel)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                        n_copied += 1
                print(f"Batch {batch_idx}: copied {n_copied} metadata files to local HF dir")
        os.environ["HF_HOME"] = local_hf
        avdi = None

    t0 = time.time()
    eval_set = set(eval_clip_ids)
    samples, errors = _process_clips(clip_ids, eval_set, scale, avdi=avdi)

    cache_volume.commit()

    if not preloaded:
        shutil.rmtree(local_hf, ignore_errors=True)

    elapsed = time.time() - t0
    rate = len(clip_ids) / elapsed if elapsed > 0 else 0

    if errors:
        error_types = Counter(e["error"][:80] for e in errors)
        top_errors = error_types.most_common(3)
        error_summary = "; ".join(f"{msg}({n})" for msg, n in top_errors)
    else:
        error_summary = "none"

    print(
        f"Batch {batch_idx}: {len(clip_ids)} clips → {len(samples)} samples, "
        f"{len(errors)} errors, {elapsed:.0f}s ({rate:.1f} clips/s) "
        f"[{error_summary}]"
    )
    return {"samples": samples, "errors": errors}


@app.function(
    image=extract_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 12,
    memory=32 * 1024,
    secrets=[modal.Secret.from_name("huggingface")],
)
def extract_parallel(
    scale: str = "large",
    seed: int = 42,
    eval_frac: float = 0.15,
    batch_size: int = 80,
):
    """Parallel extraction across many Modal containers.

    If preload_dataset was run first, reads from local volume (fast).
    Otherwise falls back to HF streaming (slower, rate-limited).
    """
    import os
    import time

    import numpy as np
    import pandas as pd

    max_clips = SCALES[scale]
    preload_dir = f"{CACHE_DIR}/preload"
    preloaded = os.path.exists(f"{preload_dir}/done")

    if preloaded:
        print(f"=== Parallel extraction: {scale} scale, PRELOADED data ===")
        avdi = _create_local_avdi()
        clip_df = pd.read_parquet(f"{preload_dir}/selected_clips.parquet")
        selected_clips = clip_df["clip_id"].tolist()
        eval_df = pd.read_parquet(f"{preload_dir}/eval_clips.parquet")
        eval_clips = eval_df["clip_id"].tolist()
        print(f"Using {len(selected_clips)} preloaded clips ({len(eval_clips)} eval)")
    else:
        print(f"=== Parallel extraction: {scale} scale, {max_clips} clips (HF streaming) ===")
        import physical_ai_av
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
        print(f"Dataset: {len(avdi.clip_index)} total clips")
        available_clips = _filter_clips(avdi)
        print(f"Available after filtering: {len(available_clips)} clips")
        rng = np.random.default_rng(seed)
        if len(available_clips) > max_clips:
            selected_clips = rng.choice(available_clips, size=max_clips, replace=False).tolist()
        else:
            selected_clips = available_clips[:max_clips]
        n_eval = max(1, int(len(selected_clips) * eval_frac))
        rng.shuffle(selected_clips)
        eval_clips = selected_clips[:n_eval]

    print(f"Selected {len(selected_clips)} clips: {len(selected_clips) - len(eval_clips)} train, {len(eval_clips)} eval")

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    os.makedirs(output_dir, exist_ok=True)

    # Resume: skip clips already extracted in a previous run
    checkpoint_path = f"{output_dir}/samples_checkpoint.parquet"
    already_extracted_clips = set()
    prev_samples = []
    if os.path.exists(checkpoint_path):
        prev_df = pd.read_parquet(checkpoint_path)
        prev_samples = prev_df.to_dict("records")
        already_extracted_clips = set(prev_df["clip_id"].unique())
        print(f"Resuming: {len(prev_samples)} samples from {len(already_extracted_clips)} clips already extracted")

    selected_clips = [c for c in selected_clips if c not in already_extracted_clips]
    print(f"Remaining after resume: {len(selected_clips)} clips")

    batches = [
        selected_clips[i : i + batch_size]
        for i in range(0, len(selected_clips), batch_size)
    ]
    n_batches = len(batches)
    # Purge ALL zip files from the shared HF cache.
    # Workers use local HF caches now, so they don't need shared zips.
    # Previous runs left corrupted zips (429 error pages) that bloat copytree.
    import glob
    purged = 0
    for zf in glob.glob(f"{CACHE_DIR}/hf/hub/**/*.zip", recursive=True):
        os.remove(zf)
        purged += 1
    if purged:
        print(f"Purged {purged} zip files from shared cache (workers use local caches)")

    # Orchestrator already initialized avdi, which cached metadata to HF_HOME.
    # Commit so workers can read from the shared cache (minus corrupted files).
    cache_volume.commit()
    print("Committed metadata cache for workers")

    print(f"Launching {n_batches} parallel workers ({batch_size} clips each)...")
    t_start = time.time()

    all_samples = list(prev_samples)
    all_errors = []
    completed = 0

    for r in _extract_clip_batch.map(
        batches,
        [eval_clips] * n_batches,
        [scale] * n_batches,
        list(range(n_batches)),
        [preloaded] * n_batches,
    ):
        all_samples.extend(r["samples"])
        all_errors.extend(r["errors"])
        completed += 1

        # Checkpoint every 5 batches (more frequent for resilience)
        if completed % 5 == 0:
            ckpt = pd.DataFrame(all_samples)
            ckpt.to_parquet(f"{output_dir}/samples_checkpoint.parquet", index=False)
            cache_volume.commit()
            print(f"  Checkpoint: {completed}/{n_batches} batches, {len(all_samples)} samples (total)")

    df = pd.DataFrame(all_samples)
    df.to_parquet(f"{output_dir}/samples.parquet", index=False)

    if all_errors:
        pd.DataFrame(all_errors).to_parquet(f"{output_dir}/errors.parquet", index=False)

    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    elapsed = time.time() - t_start
    print(f"\n=== Results ===")
    print(f"Total samples: {len(df)}")
    if len(df) > 0:
        print(f"Train: {(df['split'] == 'train').sum()}, Eval: {(df['split'] == 'eval').sum()}")
        print(f"Errors: {len(all_errors)}")
        print(f"\nNav prompt distribution:")
        print(df["nav_prompt"].value_counts().to_string())
        print(f"\nSpeed stats: mean={df['speed'].mean():.2f}, "
              f"std={df['speed'].std():.2f}, "
              f"min={df['speed'].min():.2f}, max={df['speed'].max():.2f}")

    cache_volume.commit()
    print(f"\nWall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Effective rate: {len(selected_clips)/elapsed:.1f} clips/s")
    return len(df)


# ---------------------------------------------------------------------------
# Step 1: Extract GT egomotion + images + nav prompts (sequential, for small scales)
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
    """Sequential extraction — use extract_parallel for large scale."""
    import os
    import time

    import numpy as np
    import pandas as pd
    import physical_ai_av
    from tqdm import tqdm

    max_clips = SCALES[scale]
    print(f"=== Extracting {scale} scale: {max_clips} clips ===")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    print(f"Dataset: {len(avdi.clip_index)} total clips")

    available_clips = _filter_clips(avdi)
    print(f"Available after filtering: {len(available_clips)} clips")

    rng = np.random.default_rng(seed)
    if len(available_clips) > max_clips:
        selected_clips = rng.choice(available_clips, size=max_clips, replace=False).tolist()
    else:
        selected_clips = available_clips[:max_clips]

    n_eval = max(1, int(len(selected_clips) * eval_frac))
    rng.shuffle(selected_clips)
    eval_clips = set(selected_clips[:n_eval])
    print(f"Selected {len(selected_clips)} clips: {len(selected_clips) - n_eval} train, {n_eval} eval")

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    os.makedirs(output_dir, exist_ok=True)

    # Resume from checkpoint
    checkpoint_path = f"{output_dir}/samples_checkpoint.parquet"
    processed_ids = set()
    prev_samples = []
    if os.path.exists(checkpoint_path):
        existing = pd.read_parquet(checkpoint_path)
        prev_samples = existing.to_dict("records")
        processed_ids = set(existing["sample_id"].tolist())
        print(f"Resuming from checkpoint: {len(prev_samples)} samples already extracted")

    remaining_clips = [c for c in selected_clips if not {f"{c}_{t0}" for t0 in T0_OFFSETS_US}.issubset(processed_ids)]
    print(f"Remaining: {len(remaining_clips)} clips")

    t_start = time.time()
    checkpoint_interval = 50
    all_samples = list(prev_samples)
    all_errors = []

    for batch_start in range(0, len(remaining_clips), checkpoint_interval):
        batch = remaining_clips[batch_start : batch_start + checkpoint_interval]
        samples, errors = _process_clips(batch, eval_clips, scale, avdi=avdi)
        all_samples.extend(samples)
        all_errors.extend(errors)

        ckpt_df = pd.DataFrame(all_samples)
        ckpt_df.to_parquet(checkpoint_path, index=False)
        cache_volume.commit()

        done = batch_start + len(batch)
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        print(f"Checkpoint: {done}/{len(remaining_clips)} clips, "
              f"{len(all_samples)} samples, {len(all_errors)} errors, "
              f"{rate:.1f} clips/s")

    df = pd.DataFrame(all_samples)
    df.to_parquet(f"{output_dir}/samples.parquet", index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if all_errors:
        pd.DataFrame(all_errors).to_parquet(f"{output_dir}/errors.parquet", index=False)

    if len(df) > 0:
        print(f"\n=== Results ===")
        print(f"Total samples: {len(df)}")
        print(f"Train: {(df['split'] == 'train').sum()}, Eval: {(df['split'] == 'eval').sum()}")
        print(f"Errors: {len(all_errors)}")
        print(f"\nNav prompt distribution:")
        print(df["nav_prompt"].value_counts().to_string())

    cache_volume.commit()
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
def label_nav_prompts(scale: str = "tiny", checkpoint_interval: int = 50):
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

    # Resume: if nav_prompt_traj column exists, skip already-labeled rows
    already_labeled = "nav_prompt_traj" in df.columns
    if already_labeled:
        unlabeled_mask = df["nav_prompt"] == df["nav_prompt_traj"]
        n_done = (~unlabeled_mask).sum()
        print(f"  Resuming: {n_done} already labeled, {unlabeled_mask.sum()} remaining")
    else:
        unlabeled_mask = pd.Series([True] * len(df))
        df["nav_prompt_traj"] = df["nav_prompt"].copy()

    labeled = df["nav_prompt"].tolist()
    errors = []
    t_start = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        if not unlabeled_mask.iloc[idx]:
            continue

        img_path = f"{output_dir}/{row['image_path']}"
        if not os.path.exists(img_path):
            errors.append({"idx": idx, "error": "image not found"})
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
                labeled[idx] = raw
            else:
                for cat in NAV_CATEGORIES:
                    if cat in raw:
                        labeled[idx] = cat
                        break
                else:
                    print(f"  Warning: unexpected VLM output '{raw}' for {row['sample_id']}, keeping trajectory-based label")

        except Exception as e:
            errors.append({"idx": idx, "error": str(e)})
            if len(errors) <= 5:
                print(f"  Error on {row['sample_id']}: {e}")
            time.sleep(2)

        # Checkpoint every N samples
        if (idx + 1) % checkpoint_interval == 0:
            df["nav_prompt"] = labeled
            df.to_parquet(samples_path, index=False)
            cache_volume.commit()
            n_labeled = sum(a != b for a, b in zip(labeled, df["nav_prompt_traj"]))
            print(f"  Checkpoint: {n_labeled}/{len(df)} labeled, {len(errors)} errors")

    # Final save
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
