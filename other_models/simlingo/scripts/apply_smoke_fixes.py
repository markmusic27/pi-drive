"""Fix the two bugs uncovered by the first DDP smoke launch.

Run from repo root:
    python simlingo/scripts/apply_smoke_fixes.py

Fixes:
  1. nvidia_trainer.py entrypoint: gate wandb.init/wandb.finish to rank 0.
     The previous run spawned 4 wandb runs in parallel (one per torchrun proc).
     The trainer object hasn't been built yet at that point, so we check
     the RANK env var directly (set by torchrun).
  2. modal_training.py train_multigpu: revert gpu="B200:4" -> "H100:4" because
     the current image is CUDA 12.1.1 / PyTorch built for sm_50..sm_90 only,
     while B200 is sm_100 (Blackwell). H100 works in the existing image.
     Also revert nproc_per_node back to 4 (unchanged) and cpu/memory to
     H100-appropriate values.

Idempotent — re-running on already-patched files is a no-op.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "simlingo" / "scripts" / "nvidia_trainer.py"
MODAL = REPO_ROOT / "simlingo" / "modal_training.py"


def patch(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"  [skip] {label}: already patched")
        return text
    if old not in text:
        raise SystemExit(
            f"  [FAIL] {label}: could not locate anchor text. Manual edit needed."
        )
    print(f"  [ok]   {label}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# nvidia_trainer.py — gate wandb.init/wandb.finish via RANK env var
# ---------------------------------------------------------------------------
print(f"Patching {TRAINER}...")
src = TRAINER.read_text()

src = patch(
    src,
    old=(
        "    import wandb\n"
        "    import yaml\n"
        "\n"
        "    # Load config\n"
        '    with open(args.config, "r") as f:\n'
        "        config = yaml.safe_load(f)\n"
        "\n"
        "    # Initialize W&B\n"
        "    wandb.init(project=args.wandb_project, config=config)\n"
    ),
    new=(
        "    import wandb\n"
        "    import yaml\n"
        "\n"
        "    # Load config\n"
        '    with open(args.config, "r") as f:\n'
        "        config = yaml.safe_load(f)\n"
        "\n"
        "    # Detect rank for rank-0 wandb gating (torchrun sets RANK).\n"
        '    _entrypoint_rank = int(os.environ.get("RANK", "0"))\n'
        "    _is_main_entry = _entrypoint_rank == 0\n"
        "\n"
        "    # Initialize W&B on rank 0 only; non-main ranks disable wandb entirely.\n"
        "    if _is_main_entry:\n"
        "        wandb.init(project=args.wandb_project, config=config)\n"
        "    else:\n"
        '        os.environ["WANDB_MODE"] = "disabled"\n'
        "        wandb.init(mode=\"disabled\")\n"
    ),
    label="trainer entrypoint: rank-0 wandb.init",
)

src = patch(
    src,
    old=(
        "    # Push to hub if configured\n"
        "    if args.hf_repo:\n"
        "        trainer.push_to_hub(args.hf_repo)\n"
        "\n"
        "    wandb.finish()\n"
    ),
    new=(
        "    # Push to hub if configured (rank 0 only)\n"
        "    if args.hf_repo and _is_main_entry:\n"
        "        trainer.push_to_hub(args.hf_repo)\n"
        "\n"
        "    if _is_main_entry:\n"
        "        wandb.finish()\n"
    ),
    label="trainer entrypoint: rank-0 wandb.finish + hf push",
)

TRAINER.write_text(src)
print(f"  wrote {TRAINER}\n")


# ---------------------------------------------------------------------------
# modal_training.py — revert train_multigpu to H100:4 (B200 needs newer image)
# ---------------------------------------------------------------------------
print(f"Patching {MODAL}...")
mod = MODAL.read_text()

mod = patch(
    mod,
    old=(
        '    gpu="B200:4",  # 4x B200 for fastest training\n'
        "    timeout=60 * 60 * 24,  # 24 hours max (Modal limit)\n"
        "    cpu=32,\n"
        "    memory=256 * 1024,\n"
    ),
    new=(
        '    gpu="H100:4",  # 4x H100 (B200 requires CUDA 12.4+ / PyTorch 2.4+ with sm_100; current image is 12.1)\n'
        "    timeout=60 * 60 * 24,  # 24 hours max (Modal limit)\n"
        "    cpu=32,\n"
        "    memory=256 * 1024,\n"
    ),
    label="modal train_multigpu: B200:4 -> H100:4",
)

mod = patch(
    mod,
    old='    Same as `train` but uses 4x B200 GPUs for fastest training on larger datasets.\n',
    new='    Same as `train` but uses 4x H100 GPUs for fastest training on larger datasets.\n',
    label="modal docstring: B200 -> H100",
)

MODAL.write_text(mod)
print(f"  wrote {MODAL}\n")

print("Smoke-fix patches applied.")
print("Re-launch with:")
print("  cd simlingo && modal run --detach modal_training.py::train_multigpu \\")
print("    --config-path /app/config/nvidia_smoke.yaml \\")
print("    --wandb-project simlingo-nvidia-smoke")
