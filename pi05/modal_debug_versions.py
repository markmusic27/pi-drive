"""Debug: check exact package versions in openpi's venv."""

from __future__ import annotations

import modal

APP_NAME = "pi05-debug"
CACHE_DIR = "/cache"
OPENPI_DIR = "/opt/openpi"

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
)

app = modal.App(APP_NAME)


@app.function(image=train_image, timeout=60 * 5)
def check_versions():
    import subprocess
    result = subprocess.run(
        [f"{OPENPI_DIR}/.venv/bin/python", "-m", "pip", "list", "--format=columns"],
        capture_output=True, text=True,
    )
    for line in result.stdout.split("\n"):
        if any(pkg in line.lower() for pkg in ["datasets", "lerobot", "huggingface"]):
            print(line)

    # Check if List is in feature types
    result2 = subprocess.run(
        [f"{OPENPI_DIR}/.venv/bin/python", "-c",
         "from datasets.features import features; print('List' in features._FEATURE_TYPES, list(features._FEATURE_TYPES.keys()))"],
        capture_output=True, text=True,
    )
    print(f"\nList in feature types: {result2.stdout.strip()}")
    if result2.stderr:
        print(f"stderr: {result2.stderr.strip()}")
