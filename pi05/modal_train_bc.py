"""π0.5 behavior cloning training on Modal (8× H100).

Trains π0.5 on PhysicalAI-AV ground truth driving data using the
pi05_driving config in openpi. LoRA on VLM + full fine-tune on action expert.

Usage:
    # Pre-flight: validate HF push works
    modal run pi05/modal_train_bc.py::validate_hf

    # Fresh download dataset from HF (needed once, gets all parquets)
    modal run --detach pi05/modal_train_bc.py::fresh_download

    # Consolidate 10K parquets into 10 chunk files (needed once)
    modal run --detach pi05/modal_train_bc.py::consolidate_dataset

    # Train (detached so laptop can close)
    modal run --detach pi05/modal_train_bc.py::train_bc --num-steps 5000

    # During training — on-demand checkpoint save:
    modal run pi05/modal_train_bc.py::trigger_save

    # During training — save checkpoint + push to HuggingFace:
    modal run pi05/modal_train_bc.py::trigger_push_hf

    # After training — upload a specific checkpoint:
    modal run pi05/modal_train_bc.py::upload_checkpoint --step 5000

    # Run diagnostic to test dataset loading:
    modal run --detach pi05/modal_train_bc.py::diagnose_dataset

Dataset issues solved (2026-05-30):
    1. HF API rate limits (429): LeRobot's snapshot_download makes per-file API calls
       to check ETags. With 10K files, hits 1000 req/5min rate limit. Fixed by patching
       download_episodes to skip when local data exists on the volume.

    2. No images/ directory: Images are embedded directly in parquet files (not stored
       as separate PNGs). The LeRobotDataset __init__ assertion checks for separate
       image files — patched to skip this check.

    3. 10K individual parquets too slow: Each episode stored as 1 parquet file (~300KB).
       Loading 10K files on Modal FUSE is very slow. Fixed by consolidating into 10
       chunk-level files (~300MB each). Must commit volume after each chunk or progress
       is lost on timeout.

    4. HF-generated file-*.parquet conflicts: Git-cloning from HF includes auto-generated
       file-000.parquet files that are incompatible with our episode format. Cleaned up
       during consolidation.

    5. Dataset not on volume: Original dataset was built on a different container and
       only pushed to HF. Used git clone (bypasses per-file rate limits) to download
       fresh copy to the volume.
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
    .pip_install("huggingface_hub", "wandb", "pyarrow", "pandas")
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

TRIGGER_DIR = f"{CACHE_DIR}/triggers"
HF_CHECKPOINT_REPO = "markmusic/pi05-physical-av-bc-checkpoint"


# ---------------------------------------------------------------------------
# Pre-flight: validate HF push works before spending GPU money
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    timeout=300,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
)
def validate_hf():
    """Test HF token + repo access. Run this before training."""
    import os
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()
    print(f"HF token valid: logged in as {user['name']}")

    api.create_repo(
        HF_CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True
    )
    print(f"Repo {HF_CHECKPOINT_REPO} exists and is writable")

    test_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    )
    test_file.write("preflight check")
    test_file.close()
    api.upload_file(
        path_or_fileobj=test_file.name,
        path_in_repo=".preflight_test",
        repo_id=HF_CHECKPOINT_REPO,
        repo_type="model",
        commit_message="preflight validation",
    )
    os.unlink(test_file.name)
    api.delete_file(
        ".preflight_test",
        repo_id=HF_CHECKPOINT_REPO,
        repo_type="model",
        commit_message="remove preflight test",
    )
    print("HF push validated — write + delete succeeded")
    return True


# ---------------------------------------------------------------------------
# Trigger functions: on-demand checkpoint save / HF push
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    timeout=60,
    volumes=VOLUMES,
)
def trigger_save():
    """Write a trigger file that tells the training loop to save a checkpoint NOW."""
    import os
    import pathlib

    pathlib.Path(TRIGGER_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{TRIGGER_DIR}/save_now").touch()
    cache_volume.commit()
    print("Trigger written: training will save a checkpoint at the next step check")
    return "save_now trigger queued"


@app.function(
    image=train_image,
    timeout=60,
    volumes=VOLUMES,
)
def trigger_push_hf():
    """Write a trigger that saves a checkpoint AND pushes it to HuggingFace."""
    import os
    import pathlib

    pathlib.Path(TRIGGER_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{TRIGGER_DIR}/save_now").touch()
    pathlib.Path(f"{TRIGGER_DIR}/push_hf").touch()
    cache_volume.commit()
    print("Trigger written: training will save + push checkpoint to HF")
    return "push_hf trigger queued"


# ---------------------------------------------------------------------------
# Consolidate dataset: merge 10K tiny parquets → 10 larger ones
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    timeout=60 * 60,
    volumes=VOLUMES,
    memory=32 * 1024,
)
def consolidate_dataset(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Merge per-episode parquet files into chunk-level files for fast loading.

    Commits after each chunk so progress survives timeouts.
    """
    import glob
    import os
    import shutil
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
    data_dir = os.path.join(dataset_dir, "data")
    marker = os.path.join(dataset_dir, ".consolidated")

    if os.path.exists(marker):
        print("Dataset already consolidated")
        return

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No data directory at {data_dir}")

    chunk_dirs = sorted(glob.glob(os.path.join(data_dir, "chunk-*")))
    print(f"Consolidating {len(chunk_dirs)} chunks...")

    for chunk_dir in chunk_dirs:
        episode_files = sorted(glob.glob(os.path.join(chunk_dir, "episode_*.parquet")))
        if len(episode_files) <= 1:
            print(f"  {os.path.basename(chunk_dir)}: already consolidated")
            continue

        print(f"  {os.path.basename(chunk_dir)}: merging {len(episode_files)} files...", end=" ", flush=True)

        # Read and merge on local /tmp (fast), then copy back to volume
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = []
            for f in episode_files:
                try:
                    tables.append(pq.read_table(f))
                except Exception as e:
                    print(f"\n    WARNING: skipping corrupt file {os.path.basename(f)}: {e}")

            merged = pa.concat_tables(tables)
            tmp_out = os.path.join(tmpdir, "episode_000000.parquet")
            pq.write_table(merged, tmp_out)

            # Delete originals, copy merged file
            for f in episode_files:
                os.remove(f)
            shutil.copy2(tmp_out, os.path.join(chunk_dir, "episode_000000.parquet"))

        print(f"{len(merged)} rows", flush=True)
        cache_volume.commit()

    # Remove HuggingFace auto-generated file-*.parquet files that conflict
    for chunk_dir in chunk_dirs:
        for extra in glob.glob(os.path.join(chunk_dir, "file-*.parquet")):
            print(f"  Removing HF-generated {os.path.basename(chunk_dir)}/{os.path.basename(extra)}")
            os.remove(extra)

    with open(marker, "w") as f:
        f.write("done")
    cache_volume.commit()
    print("Consolidation complete")


# ---------------------------------------------------------------------------
# Fresh download: git clone dataset from HF to get all files incl. images
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    timeout=60 * 60 * 2,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=32 * 1024,
)
def fresh_download(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Git-clone dataset from HuggingFace to get all files including images.

    Uses git-lfs batch API which avoids per-file rate limits.
    Downloads to /cache/hf/lerobot/<repo_id>-fresh, then replaces the original.
    """
    import os
    import shutil
    import subprocess

    fresh_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}-fresh"
    final_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"

    # Get HF token for private repo
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    if not hf_token:
        raise RuntimeError("No HF token found — set HUGGING_FACE_HUB_TOKEN secret")

    clone_url = f"https://user:{hf_token}@huggingface.co/datasets/{repo_id}"

    # Remove stale fresh dir from any previous attempt
    if os.path.exists(fresh_dir):
        print(f"Removing previous fresh download at {fresh_dir}")
        shutil.rmtree(fresh_dir)

    print(f"Git-cloning {repo_id} to {fresh_dir}...")
    print("This downloads all files including 10K images via git-lfs batch API")
    result = subprocess.run(
        ["git", "lfs", "install"],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    print(f"git lfs install: {result.stdout.strip()}")

    result = subprocess.run(
        ["git", "clone", "--depth=1", clone_url, fresh_dir],
        text=True,
        timeout=60 * 90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git clone failed (rc={result.returncode})")

    # Verify we got what we need
    for subdir in ["data", "meta"]:
        path = os.path.join(fresh_dir, subdir)
        if os.path.exists(path):
            n_files = sum(1 for _ in os.scandir(path))
            print(f"  {subdir}/: {n_files} entries")
        else:
            print(f"  WARNING: no {subdir}/ directory!")

    images_dir = os.path.join(fresh_dir, "images")
    if os.path.exists(images_dir):
        n_imgs = 0
        for root, dirs, files in os.walk(images_dir):
            n_imgs += len(files)
        print(f"  images/: {n_imgs} files")
    else:
        print("  WARNING: no images/ directory!")

    # Check data chunk structure
    data_dir = os.path.join(fresh_dir, "data")
    if os.path.exists(data_dir):
        for chunk in sorted(os.listdir(data_dir)):
            chunk_path = os.path.join(data_dir, chunk)
            if os.path.isdir(chunk_path):
                files = os.listdir(chunk_path)
                print(f"  data/{chunk}: {len(files)} files")

    # Rename old dir (preserve it, don't delete per user instructions)
    backup_dir = f"{final_dir}-old"
    if os.path.exists(final_dir):
        if os.path.exists(backup_dir):
            print(f"Removing previous backup at {backup_dir}")
            shutil.rmtree(backup_dir)
        print(f"Moving old dataset to {backup_dir}")
        os.rename(final_dir, backup_dir)

    # Move fresh download into place
    os.rename(fresh_dir, final_dir)
    print(f"Fresh dataset installed at {final_dir}")

    # Remove git metadata to save space (not needed for training)
    git_dir = os.path.join(final_dir, ".git")
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir)
        print("Removed .git directory")

    cache_volume.commit()
    print("Done! Dataset ready with all images.")
    return 0


# ---------------------------------------------------------------------------
# Diagnostic: quickly test dataset loading to find hangs
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="H100",
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=32 * 1024,
)
def diagnose_dataset(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Test dataset loading step-by-step to find where it hangs."""
    import subprocess
    import time

    _patch_openpi()
    _consolidate_local_dataset(repo_id)
    _link_dataset_to_hf_cache(repo_id)

    script = f'''
import time, sys, os, glob
sys.stdout.reconfigure(line_buffering=True)

print("[1/8] Importing modules...")
t0 = time.time()
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
print(f"  Imports done in {{time.time()-t0:.1f}}s")

print("[2/8] Checking local dataset on volume...")
local_path = "/cache/hf/lerobot/markmusic/pi05-physical-av-bc"
if os.path.exists(local_path):
    dirs = os.listdir(local_path)
    print(f"  Local dataset found at {{local_path}}")
    print(f"  Contents: {{dirs[:20]}}")
    meta_path = os.path.join(local_path, "meta")
    if os.path.exists(meta_path):
        meta_files = os.listdir(meta_path)
        print(f"  meta/: {{meta_files}}")
        info_path = os.path.join(meta_path, "info.json")
        if os.path.exists(info_path):
            import json
            with open(info_path) as f:
                info = json.load(f)
            print(f"  info.json: version={{info.get('codebase_version')}}, total_episodes={{info.get('total_episodes')}}")
    data_path = os.path.join(local_path, "data")
    if os.path.exists(data_path):
        chunks = os.listdir(data_path)
        print(f"  data/: {{chunks[:10]}}")
        for chunk in chunks[:2]:
            chunk_path = os.path.join(data_path, chunk)
            if os.path.isdir(chunk_path):
                files = os.listdir(chunk_path)
                print(f"    {{chunk}}/: {{len(files)}} files, first={{files[:3]}}")
    img_path = os.path.join(local_path, "images")
    if os.path.exists(img_path):
        n_imgs = sum(1 for _ in glob.iglob(os.path.join(img_path, "**/*.png"), recursive=True))
        print(f"  images/: {{n_imgs}} PNG files")
else:
    print(f"  No local dataset at {{local_path}}")

print("[3/8] Checking HF hub cache symlink...")
hub_dir = "/cache/hf/hub/datasets--markmusic--pi05-physical-av-bc"
snapshot_dir = os.path.join(hub_dir, "snapshots/local")
refs_file = os.path.join(hub_dir, "refs/main")
if os.path.exists(snapshot_dir):
    is_link = os.path.islink(snapshot_dir)
    target = os.readlink(snapshot_dir) if is_link else "NOT A SYMLINK"
    print(f"  Snapshot dir exists: symlink={{is_link}}, target={{target}}")
    if os.path.exists(refs_file):
        with open(refs_file) as f:
            print(f"  refs/main: {{f.read().strip()}}")
    contents = os.listdir(snapshot_dir) if os.path.isdir(snapshot_dir) else []
    print(f"  Contents: {{contents[:15]}}")
else:
    print(f"  NO symlink at {{snapshot_dir}} — dataset will try to download from HF!")

print("[4/8] Loading config...")
t0 = time.time()
config = _config.get_config("pi05_driving")
data_config = config.data.create(config.assets_dirs, config.model)
print(f"  Config loaded in {{time.time()-t0:.1f}}s")
print(f"  repo_id: {{data_config.repo_id}}")
print(f"  action_sequence_keys: {{data_config.action_sequence_keys}}")
print(f"  action_horizon: {{config.model.action_horizon}}")

print("[5/8] Loading dataset metadata (should use cached symlink)...")
t0 = time.time()
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
print(f"  Metadata loaded in {{time.time()-t0:.1f}}s")
print(f"  fps: {{dataset_meta.fps}}")
print(f"  tasks: {{dataset_meta.tasks}}")

print("[6/8] Creating LeRobotDataset (should use cached symlink)...")
delta_timestamps = {{
    key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
    for key in data_config.action_sequence_keys
}}
print(f"  delta_timestamps: {{delta_timestamps}}")
t0 = time.time()
dataset = lerobot_dataset.LeRobotDataset(
    data_config.repo_id,
    delta_timestamps=delta_timestamps,
)
print(f"  Dataset created in {{time.time()-t0:.1f}}s")
print(f"  Length: {{len(dataset)}}")

print("[7/8] Loading first sample...")
t0 = time.time()
sample = dataset[0]
print(f"  Sample loaded in {{time.time()-t0:.1f}}s")
for k, v in sample.items():
    import numpy as np
    arr = np.asarray(v) if not isinstance(v, str) else v
    if isinstance(arr, np.ndarray):
        print(f"    {{k}}: shape={{arr.shape}}, dtype={{arr.dtype}}")
    else:
        print(f"    {{k}}: {{repr(arr)[:80]}}")

print("[8/8] Testing DataLoader iteration...")
t0 = time.time()
from openpi.transforms import PromptFromLeRobotTask
dl = _data_loader.TorchDataLoader(
    _data_loader.TransformedDataset(dataset, [PromptFromLeRobotTask(dataset_meta.tasks)]),
    local_batch_size=4,
    num_workers=0,
    shuffle=False,
    num_batches=2,
)
batch = next(iter(dl))
print(f"  First batch loaded in {{time.time()-t0:.1f}}s")
for k, v in batch.items():
    import numpy as np
    arr = np.asarray(v) if not isinstance(v, str) else v
    if isinstance(arr, np.ndarray):
        print(f"    {{k}}: shape={{arr.shape}}")
    else:
        print(f"    {{k}}: type={{type(arr).__name__}}")

print("DIAGNOSTIC COMPLETE — dataset loads successfully")
'''
    import tempfile
    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    )
    script_file.write(script)
    script_file.close()

    result = subprocess.run(
        [f"{OPENPI_DIR}/.venv/bin/python", "-u", script_file.name],
        cwd=OPENPI_DIR,
        text=True,
        env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
    )
    return result.returncode


# ---------------------------------------------------------------------------
# Compute normalization stats (must run before training)
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="H100",
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=32 * 1024,
)
def compute_norm_stats():
    import os
    import subprocess

    _patch_openpi()
    _consolidate_local_dataset()
    _link_dataset_to_hf_cache()

    cmd = [
        f"{OPENPI_DIR}/.venv/bin/python", "-m", "scripts.compute_norm_stats",
        "--config-name=pi05_driving",
    ]

    print(f"=== Computing norm stats ===")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=OPENPI_DIR, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Norm stats computation failed (rc={result.returncode})")

    # Copy norm stats to cache volume so training can find them
    assets_dir = f"{OPENPI_DIR}/assets/pi05_driving/markmusic/pi05-physical-av-bc"
    print(f"Norm stats saved to {assets_dir}")
    for root, dirs, files in os.walk(f"{OPENPI_DIR}/assets"):
        for f in files:
            print(f"  {os.path.join(root, f)}")

    # Persist to cache volume for reuse
    cache_assets = f"{CACHE_DIR}/norm_stats/pi05_driving/markmusic/pi05-physical-av-bc"
    os.makedirs(cache_assets, exist_ok=True)
    subprocess.run(["cp", "-r", f"{assets_dir}/.", cache_assets], check=True)
    cache_volume.commit()

    print("Norm stats computed and cached")
    return 0


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
    exp_name: str = "bc-coldstart-v2",
    resume: bool = False,
    batch_size: int = 96,
    skip_hf_validation: bool = False,
    scale: str = "xlarge",
):
    import os
    import pathlib
    import shutil
    import subprocess
    import sys
    import threading
    import time

    TRAIN_REPO = "markmusic/pi05-physical-av-bc"
    EVAL_REPO = "markmusic/pi05-physical-av-bc-eval"

    # --- Pre-flight: validate HF push works ---
    if not skip_hf_validation:
        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()
        print(f"Pre-flight: HF token valid (user={user['name']})")
        api.create_repo(
            HF_CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True
        )
        print(f"Pre-flight: HF repo {HF_CHECKPOINT_REPO} accessible")

    # Patch openpi with our driving config
    _patch_openpi()

    # Build LeRobot datasets from extracted data (train + eval, no HF push)
    extracted_path = f"{CACHE_DIR}/extracted/{scale}/samples.parquet"
    if os.path.exists(extracted_path):
        _build_lerobot_from_extracted(scale, train_repo=TRAIN_REPO, eval_repo=EVAL_REPO)
    else:
        print(f"No extracted data at {extracted_path}, using existing LeRobot dataset")

    # Consolidate dataset parquets if needed (many files → few files)
    _consolidate_local_dataset()

    # Symlink local datasets into HF hub cache so lerobot skips downloads
    _link_dataset_to_hf_cache(repo_id=TRAIN_REPO)
    _link_dataset_to_hf_cache(repo_id=EVAL_REPO)

    # Restore cached norm stats if available
    cache_assets = f"{CACHE_DIR}/norm_stats_v2/pi05_driving/{TRAIN_REPO}"
    assets_dir = f"{OPENPI_DIR}/assets/pi05_driving/{TRAIN_REPO}"
    if os.path.exists(cache_assets) and not os.path.exists(assets_dir):
        os.makedirs(os.path.dirname(assets_dir), exist_ok=True)
        shutil.copytree(cache_assets, assets_dir)
        print(f"Restored norm stats from cache")

    # Compute norm stats if still missing
    if not os.path.exists(assets_dir):
        print("Norm stats missing — computing now...")
        stats_cmd = [
            f"{OPENPI_DIR}/.venv/bin/python", "-u", "-m", "scripts.compute_norm_stats",
            "--config-name=pi05_driving",
        ]
        stats_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        print(f"Running: {' '.join(stats_cmd)}")
        stats_result = subprocess.run(
            stats_cmd, cwd=OPENPI_DIR, text=True, env=stats_env,
        )
        if stats_result.returncode != 0:
            raise RuntimeError(f"Failed to compute norm stats (rc={stats_result.returncode})")
        os.makedirs(os.path.dirname(cache_assets), exist_ok=True)
        if os.path.exists(cache_assets):
            shutil.rmtree(cache_assets)
        shutil.copytree(assets_dir, cache_assets)
        cache_volume.commit()
        print("Norm stats computed and cached")

    # Symlink train norm stats for eval dataset (same normalization)
    eval_assets = f"{OPENPI_DIR}/assets/pi05_driving/{EVAL_REPO}"
    if os.path.exists(assets_dir) and not os.path.exists(eval_assets):
        os.makedirs(os.path.dirname(eval_assets), exist_ok=True)
        os.symlink(assets_dir, eval_assets)
        print(f"Symlinked eval norm stats → train norm stats")

    checkpoint_dir = f"{CACHE_DIR}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    cmd = [
        f"{OPENPI_DIR}/.venv/bin/python", "-u", "-m", "scripts.train",
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
        "PYTHONUNBUFFERED": "1",
    }

    print(f"=== Training π0.5 driving BC ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # --- Background thread: watches for push_hf trigger and uploads ---
    _stop_watcher = threading.Event()

    def _hf_upload_watcher():
        from huggingface_hub import HfApi

        hf_api = HfApi()
        ckpt_base = f"{checkpoint_dir}/pi05_driving/{exp_name}"
        while not _stop_watcher.is_set():
            _stop_watcher.wait(30)
            if _stop_watcher.is_set():
                break
            try:
                cache_volume.reload()
                trigger = pathlib.Path(f"{TRIGGER_DIR}/push_hf")
                if trigger.exists():
                    trigger.unlink(missing_ok=True)
                    cache_volume.commit()
                    if not os.path.exists(ckpt_base):
                        print("[hf-watcher] No checkpoints yet, skipping upload")
                        continue
                    steps = sorted(
                        [int(d) for d in os.listdir(ckpt_base) if d.isdigit()],
                        reverse=True,
                    )
                    if not steps:
                        print("[hf-watcher] No checkpoint steps found")
                        continue
                    latest = steps[0]
                    params_dir = os.path.join(ckpt_base, str(latest), "params")
                    if not os.path.exists(params_dir):
                        params_dir = os.path.join(ckpt_base, str(latest))
                    print(f"[hf-watcher] Uploading step {latest} to {HF_CHECKPOINT_REPO}")
                    hf_api.create_repo(
                        HF_CHECKPOINT_REPO,
                        repo_type="model",
                        private=True,
                        exist_ok=True,
                    )
                    hf_api.upload_folder(
                        folder_path=params_dir,
                        repo_id=HF_CHECKPOINT_REPO,
                        repo_type="model",
                        commit_message=f"Checkpoint at step {latest}",
                    )
                    print(f"[hf-watcher] Uploaded step {latest} to HF")
            except Exception as e:
                print(f"[hf-watcher] Error: {e}")

    watcher_thread = threading.Thread(target=_hf_upload_watcher, daemon=True)
    watcher_thread.start()
    print("Background HF upload watcher started (polls every 30s)")

    result = subprocess.run(
        cmd,
        cwd=OPENPI_DIR,
        env=env,
        text=True,
    )

    _stop_watcher.set()
    cache_volume.commit()

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return result.returncode

    # Auto-upload final checkpoint to HF
    ckpt_base = f"{checkpoint_dir}/pi05_driving/{exp_name}"
    if os.path.exists(ckpt_base):
        from huggingface_hub import HfApi

        hf_api = HfApi()
        steps = sorted(
            [int(d) for d in os.listdir(ckpt_base) if d.isdigit()], reverse=True
        )
        if steps:
            latest = steps[0]
            params_dir = os.path.join(ckpt_base, str(latest), "params")
            if not os.path.exists(params_dir):
                params_dir = os.path.join(ckpt_base, str(latest))
            print(f"Uploading final checkpoint (step {latest}) to {HF_CHECKPOINT_REPO}")
            hf_api.create_repo(
                HF_CHECKPOINT_REPO,
                repo_type="model",
                private=True,
                exist_ok=True,
            )
            hf_api.upload_folder(
                folder_path=params_dir,
                repo_id=HF_CHECKPOINT_REPO,
                repo_type="model",
                commit_message=f"Final checkpoint at step {latest}",
            )
            print(f"Final checkpoint uploaded to {HF_CHECKPOINT_REPO}")

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
    repo_id: str = "markmusic/pi05-physical-av-bc-checkpoint",
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
# Convert LeRobot v3.0 dataset to v2.1 for openpi compatibility
# ---------------------------------------------------------------------------


def _convert_dataset_v3_to_v21(repo_id: str):
    """Download dataset from HF and convert v3.0 format to v2.1.

    openpi bundles lerobot 0.1.0 (Python 3.11) which expects v2.0/v2.1 format.
    Our dataset was built with lerobot 0.5.x which produces v3.0.
    Key differences: v2.1 uses tasks.jsonl, v3.0 uses tasks.parquet.
    """
    import json
    import os
    import subprocess

    dataset_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"

    # Download from HF if not already cached
    if not os.path.exists(dataset_dir):
        print(f"Downloading dataset {repo_id} from HuggingFace...")
        subprocess.run(
            ["huggingface-cli", "download", repo_id,
             "--repo-type", "dataset",
             "--local-dir", dataset_dir],
            check=True,
        )

    meta_dir = os.path.join(dataset_dir, "meta")
    info_path = os.path.join(meta_dir, "info.json")
    tasks_parquet = os.path.join(meta_dir, "tasks.parquet")
    tasks_jsonl = os.path.join(meta_dir, "tasks.jsonl")

    if not os.path.exists(info_path):
        print(f"No info.json found at {info_path}, skipping conversion")
        return

    with open(info_path) as f:
        info = json.load(f)

    if info.get("codebase_version") != "v3.0":
        print(f"Dataset already in {info.get('codebase_version')} format")
        return

    print(f"Converting dataset from v3.0 to v2.1...")

    # Convert tasks.parquet → tasks.jsonl
    if os.path.exists(tasks_parquet) and not os.path.exists(tasks_jsonl):
        import pyarrow.parquet as pq
        table = pq.read_table(tasks_parquet)
        with open(tasks_jsonl, "w") as f:
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                f.write(json.dumps(row) + "\n")
        print(f"  Created tasks.jsonl with {table.num_rows} tasks")

    # Convert episodes parquet → episodes.jsonl
    episodes_parquet = os.path.join(meta_dir, "episodes", "chunk-000", "file-000.parquet")
    episodes_jsonl = os.path.join(meta_dir, "episodes.jsonl")
    if os.path.exists(episodes_parquet) and not os.path.exists(episodes_jsonl):
        import pyarrow.parquet as pq
        table = pq.read_table(episodes_parquet)
        with open(episodes_jsonl, "w") as f:
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                f.write(json.dumps(row) + "\n")
        print(f"  Created episodes.jsonl with {table.num_rows} episodes")

    # Update info.json to v2.1
    info["codebase_version"] = "v2.1"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Updated info.json codebase_version to v2.1")

    cache_volume.commit()
    print("Dataset conversion complete")


# ---------------------------------------------------------------------------
# Build LeRobot datasets from extracted data (no HF push)
# ---------------------------------------------------------------------------


def _build_lerobot_from_extracted(
    scale: str,
    train_repo: str = "markmusic/pi05-physical-av-bc",
    eval_repo: str = "markmusic/pi05-physical-av-bc-eval",
):
    """Build train + eval LeRobot datasets from extracted parquet + images.

    Runs inside openpi's venv (needs lerobot). No HF push — data stays on volume.
    """
    import os
    import subprocess
    import tempfile

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    samples_path = f"{output_dir}/samples.parquet"
    train_path = f"{CACHE_DIR}/hf/lerobot/{train_repo}"
    eval_path = f"{CACHE_DIR}/hf/lerobot/{eval_repo}"

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"No extracted data at {samples_path}. Run extraction first.")

    if os.path.exists(f"{train_path}/.built") and os.path.exists(f"{eval_path}/.built"):
        print(f"LeRobot datasets already built (train={train_path}, eval={eval_path})")
        return

    build_script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp")
    build_script.write(f'''
import json, os, shutil, time, sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

output_dir = "{output_dir}"
cache_dir = "{CACHE_DIR}"
splits = {{
    "train": "{train_repo}",
    "eval": "{eval_repo}",
}}

df = pd.read_parquet(f"{{output_dir}}/samples.parquet")
print(f"Loaded {{len(df)}} total samples")
print(f"Split distribution: {{df['split'].value_counts().to_dict()}}")

for split_name, repo_id in splits.items():
    local_path = f"{{cache_dir}}/hf/lerobot/{{repo_id}}"
    marker = f"{{local_path}}/.built"

    if os.path.exists(marker):
        print(f"{{split_name}} already built at {{local_path}}")
        continue

    split_df = df[df["split"] == split_name].reset_index(drop=True)
    print(f"\\nBuilding {{split_name}}: {{len(split_df)}} samples → {{repo_id}}")

    if os.path.exists(local_path):
        shutil.rmtree(local_path)

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

    t0 = time.time()
    for idx in tqdm(range(len(split_df)), desc=f"Building {{split_name}}"):
        row = split_df.iloc[idx]
        img = np.array(Image.open(f"{{output_dir}}/{{row['image_path']}}"))
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

        if (idx + 1) % 5000 == 0:
            if hasattr(dataset, "image_writer") and dataset.image_writer is not None:
                dataset.image_writer.wait_until_done()
            import modal
            vol = modal.Volume.from_name("pi05-cache")
            vol.commit()
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"  Checkpoint: {{idx+1}}/{{len(split_df)}}, {{rate:.1f}} samples/s")

    with open(marker, "w") as f:
        f.write(f"{{len(split_df)}} samples")
    import modal
    vol = modal.Volume.from_name("pi05-cache")
    vol.commit()
    print(f"Built {{split_name}}: {{len(split_df)}} samples in {{time.time()-t0:.0f}}s")

print("\\nBUILD COMPLETE")
''')
    build_script.flush()
    script_path = build_script.name
    build_script.close()

    venv_python = f"{OPENPI_DIR}/.venv/bin/python"
    print(f"Building LeRobot datasets with {venv_python}")
    result = subprocess.run(
        [venv_python, "-u", script_path],
        text=True,
        env={**__import__("os").environ, "HF_HOME": f"{CACHE_DIR}/hf", "PYTHONUNBUFFERED": "1"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"LeRobot build failed (rc={result.returncode})")


# ---------------------------------------------------------------------------
# Patch openpi with driving config
# ---------------------------------------------------------------------------


def _link_dataset_to_hf_cache(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Symlink our local dataset into the HF hub cache so lerobot skips downloads."""
    import os

    local_dataset = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
    if not os.path.exists(local_dataset):
        print(f"No local dataset at {local_dataset}, will download from HF")
        return

    hub_dir = f"{CACHE_DIR}/hf/hub/datasets--{repo_id.replace('/', '--')}"
    snapshot_dir = f"{hub_dir}/snapshots/local"

    if os.path.exists(snapshot_dir):
        print("HF hub cache already linked to local dataset")
        return

    os.makedirs(f"{hub_dir}/refs", exist_ok=True)
    os.makedirs(os.path.dirname(snapshot_dir), exist_ok=True)

    with open(f"{hub_dir}/refs/main", "w") as f:
        f.write("local")

    os.symlink(local_dataset, snapshot_dir)

    cache_volume.commit()
    print(f"Linked local dataset → HF hub cache (skips HF downloads)")


def _consolidate_local_dataset(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Merge per-episode parquet files into chunk-level files for fast loading."""
    import glob
    import os
    import shutil
    import tempfile

    dataset_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
    data_dir = os.path.join(dataset_dir, "data")
    marker = os.path.join(dataset_dir, ".consolidated")

    if not os.path.exists(data_dir):
        print(f"No local dataset at {data_dir}, skipping consolidation")
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    chunk_dirs = sorted(glob.glob(os.path.join(data_dir, "chunk-*")))

    if os.path.exists(marker):
        # Already consolidated — just clean up any leftover HF files
        needs_cleanup = False
        for chunk_dir in chunk_dirs:
            if glob.glob(os.path.join(chunk_dir, "file-*.parquet")):
                needs_cleanup = True
                break
        if not needs_cleanup:
            print("Dataset already consolidated")
            return
        print("Cleaning up leftover HF-generated files...")
    else:
        print(f"Consolidating {len(chunk_dirs)} chunks (one-time operation)...")

    for chunk_dir in chunk_dirs:
        episode_files = sorted(glob.glob(os.path.join(chunk_dir, "episode_*.parquet")))
        if len(episode_files) <= 1:
            print(f"  {os.path.basename(chunk_dir)}: already consolidated")
            continue

        print(f"  {os.path.basename(chunk_dir)}: merging {len(episode_files)} files...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tables = []
            for f in episode_files:
                try:
                    tables.append(pq.read_table(f))
                except Exception as e:
                    print(f"\n    WARNING: skipping corrupt file {os.path.basename(f)}: {e}")

            merged = pa.concat_tables(tables)
            tmp_out = os.path.join(tmpdir, "episode_000000.parquet")
            pq.write_table(merged, tmp_out)

            for f in episode_files:
                os.remove(f)
            shutil.copy2(tmp_out, os.path.join(chunk_dir, "episode_000000.parquet"))

        print(f"{len(merged)} rows", flush=True)
        cache_volume.commit()

    # Remove HuggingFace auto-generated file-*.parquet files that conflict
    for chunk_dir in chunk_dirs:
        for extra in glob.glob(os.path.join(chunk_dir, "file-*.parquet")):
            print(f"  Removing HF-generated {os.path.basename(chunk_dir)}/{os.path.basename(extra)}")
            os.remove(extra)

    with open(marker, "w") as f:
        f.write("done")
    cache_volume.commit()
    print("Consolidation complete")


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
            import random
            if random.random() < 0.3:
                inputs["prompt"] = "drive"
            else:
                inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DrivingOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]  # (128,) -> (1, 128)
        return {"actions": actions}
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
                        "observation/image": "observation.images.front",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
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
            action_sequence_keys=("action",),
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
            action_dim=128,
            action_horizon=1,
            paligemma_variant="gemma_2b_lora_driving",
            action_expert_variant="gemma_300m",
        ),
        data=LeRobotDrivingDataConfig(
            repo_id="markmusic/pi05-physical-av-bc",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=128,
            action_horizon=1,
            paligemma_variant="gemma_2b_lora_driving",
            action_expert_variant="gemma_300m",
        ).get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=750,
            peak_lr=3e-5,
            decay_steps=15_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.999,
            clip_gradient_norm=1.0,
        ),
        num_train_steps=15_000,
        batch_size=96,
        fsdp_devices=1,
        save_interval=500,
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

    # 4. Patch lerobot to accept our dataset format
    import glob
    import os
    for lerobot_utils in glob.glob(
        f"{OPENPI_DIR}/.venv/lib/python*/site-packages/lerobot/common/datasets/utils.py"
    ):
        with open(lerobot_utils, "r") as f:
            content = f.read()
        if "PATCHED" not in content:
            content = content.replace(
                "raise ForwardCompatibilityError(repo_id, min(upper_versions))",
                "pass  # PATCHED: accept our dataset version",
            )
            with open(lerobot_utils, "w") as f:
                f.write(content)
            print(f"  Patched version check in {lerobot_utils}")

    # 4b. Patch lerobot to skip downloading data when it already exists locally
    #     AND skip the file-path assertion (images are embedded in parquet, no separate files).
    for lerobot_ds in glob.glob(
        f"{OPENPI_DIR}/.venv/lib/python*/site-packages/lerobot/common/datasets/lerobot_dataset.py"
    ):
        with open(lerobot_ds, "r") as f:
            ds_content = f.read()
        if "DOWNLOAD_PATCHED" not in ds_content:
            # Patch 1: skip download when local data exists
            ds_content = ds_content.replace(
                "def download_episodes(self",
                """def download_episodes(self, download_videos=True):  # DOWNLOAD_PATCHED
        import os as _os
        _data_dir = _os.path.join(str(self.root), "data")
        _has_data = _os.path.exists(_data_dir) and len(_os.listdir(_data_dir)) > 0
        if _has_data:
            print(f"  DOWNLOAD_PATCHED: local data at {self.root}, skipping HF download")
            return
        return self._original_download_episodes(download_videos)

    def _original_download_episodes(self""",
            )
            # Patch 2: skip file-path assertion (images embedded in parquet, no separate files)
            ds_content = ds_content.replace(
                "assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())",
                "pass  # DOWNLOAD_PATCHED: images embedded in parquet, no separate files to check",
            )
            with open(lerobot_ds, "w") as f:
                f.write(ds_content)
            print(f"  Patched lerobot_dataset.py to skip download when local data exists")

    # 5. Patch scripts/train.py to handle shape mismatches when loading
    #    base checkpoint with different action_dim (32→2).
    train_script = f"{OPENPI_DIR}/scripts/train.py"
    with open(train_script, "r") as f:
        content = f.read()
    if "SHAPE_PATCHED" not in content:
        old_validate = "at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)"
        new_validate = """# SHAPE_PATCHED: filter shape-mismatched params before validation
    import logging as _log
    def _filter_shapes(expected, loaded):
        import jax
        e_flat, e_struct = jax.tree.flatten(expected)
        l_flat, _ = jax.tree.flatten(loaded)
        fixed = []
        for e, l in zip(e_flat, l_flat):
            if hasattr(e, 'shape') and hasattr(l, 'shape') and e.shape != l.shape:
                _log.warning(f"Shape mismatch: expected {e.shape}, got {l.shape} — using init weights")
                fixed.append(e)
            else:
                fixed.append(l)
        return jax.tree.unflatten(e_struct, fixed)
    loaded_params = _filter_shapes(params_shape, loaded_params)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)"""
        if old_validate in content:
            content = content.replace(old_validate, new_validate)
            with open(train_script, "w") as f:
                f.write(content)
            print("  Patched scripts/train.py for shape mismatch handling")
        else:
            print("  WARNING: Could not find validation call in train.py")
            for i, line in enumerate(content.split('\n')):
                if 'check_pytree_equality' in line:
                    print(f"    Line {i+1}: {line.strip()}")

    # 6. Add LR logging + eval loss to scripts/train.py
    with open(train_script, "r") as f:
        content = f.read()
    if "LR_EVAL_PATCHED" not in content:
        # Patch A: Add eval_step_fn before main()
        content = content.replace(
            "def main(config: _config.TrainConfig):",
            '''def eval_step_fn(
    config: _config.TrainConfig,
    rng,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    return jnp.mean(model.compute_loss(rng, observation, actions, train=False))


def main(config: _config.TrainConfig):  # LR_EVAL_PATCHED''',
        )

        # Patch B: Add LR schedule, peval_step, and proper eval from held-out dataset
        content = content.replace(
            "    start_step = int(train_state.step)",
            '''    lr_schedule_fn = config.lr_schedule.create()

    peval_step = jax.jit(
        functools.partial(eval_step_fn, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    # PROPER EVAL: load held-out eval dataset (not training data)
    import dataclasses as _dc
    _eval_repo = data_config.repo_id + "-eval"
    _eval_batches = []
    try:
        _eval_dc = _dc.replace(data_config, repo_id=_eval_repo)
        _eval_loader = _data_loader.get_data_loader(config, _eval_dc, mesh)
        _eval_iter = iter(_eval_loader)
        for _ in range(20):
            try:
                _eval_batches.append(next(_eval_iter))
            except StopIteration:
                break
        logging.info(f"Loaded {len(_eval_batches)} HELD-OUT eval batches from {_eval_repo}")
    except Exception as _e:
        logging.warning(f"Eval dataset not found ({_e}), falling back to train data")
        for _ in range(5):
            _eval_batches.append(next(data_iter))
        logging.info(f"Cached {len(_eval_batches)} train batches as eval (NOT held-out)")

    start_step = int(train_state.step)''',
        )

        # Patch C: Add LR to logged metrics
        content = content.replace(
            "            wandb.log(reduced_info, step=step)",
            '''            reduced_info["learning_rate"] = float(lr_schedule_fn(step))
            wandb.log(reduced_info, step=step)''',
        )

        # Patch D: Add eval loss at checkpoint intervals
        content = content.replace(
            '''        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)''',
            '''        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            if _eval_batches:
                _el = []
                for _eb in _eval_batches:
                    with sharding.set_mesh(mesh):
                        _el.append(peval_step(train_rng, train_state, _eb))
                _eval_loss = float(np.mean([float(jax.device_get(x)) for x in _el]))
                pbar.write(f"Step {step}: eval_loss={_eval_loss:.4f}")
                wandb.log({"eval_loss": _eval_loss}, step=step)
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)''',
        )

        with open(train_script, "w") as f:
            f.write(content)
        print("  Patched scripts/train.py for LR + eval loss logging")

    # 7. Add trigger-file checkpoint save (on-demand save via /cache/triggers/save_now)
    with open(train_script, "r") as f:
        content = f.read()
    if "TRIGGER_PATCHED" not in content:
        content = content.replace(
            "    infos = []",
            '''    import pathlib as _pathlib  # TRIGGER_PATCHED
    _trigger_dir = _pathlib.Path("/cache/triggers")

    infos = []''',
            1,  # only replace FIRST occurrence (the 4-space one before the for-loop)
        )

        content = content.replace(
            "        batch = next(data_iter)",
            '''        # Check for on-demand save trigger every 10 steps
        if step % 10 == 0 and _trigger_dir.exists():
            _save_trigger = _trigger_dir / "save_now"
            if _save_trigger.exists():
                try:
                    _save_trigger.unlink(missing_ok=True)
                except OSError:
                    pass
                pbar.write(f"[trigger] Manual checkpoint save at step {step}")
                if _eval_batches:
                    _el = []
                    for _eb in _eval_batches:
                        with sharding.set_mesh(mesh):
                            _el.append(peval_step(train_rng, train_state, _eb))
                    _eval_loss = float(np.mean([float(jax.device_get(x)) for x in _el]))
                    pbar.write(f"[trigger] eval_loss={_eval_loss:.4f}")
                    wandb.log({"eval_loss": _eval_loss}, step=step)
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
                pbar.write(f"[trigger] Checkpoint saved at step {step}")

        batch = next(data_iter)''',
        )

        with open(train_script, "w") as f:
            f.write(content)
        print("  Patched scripts/train.py for trigger-file checkpoint saves")

    # Compile-check the patched train.py to catch syntax/indentation errors early
    with open(train_script, "r") as f:
        source = f.read()
    try:
        compile(source, train_script, "exec")
        print("  Compile-check passed for scripts/train.py")
    except SyntaxError as e:
        lines = source.split("\n")
        ctx_start = max(0, e.lineno - 5)
        ctx_end = min(len(lines), e.lineno + 5)
        ctx = "\n".join(f"  {i+1:4d}: {lines[i]}" for i in range(ctx_start, ctx_end))
        raise SyntaxError(f"Patched train.py has syntax error at line {e.lineno}:\n{ctx}") from e

    print("openpi patched with driving config")
