"""Idempotent applier: replace hardcoded 110-degree FOV in nvidia_trainer.py
with the config-driven `data.fov_deg` value (defaults to 120 for NVIDIA
front-wide camera).

Three call sites currently hardcode 110:
  - _process_single_image            (per-sample DrivingInput)
  - _build_driving_input_batched     (vec-path DrivingInput)
  - viz code path                    (visualization)

This is the quick-fix variant: use a single `self.fov_deg` from
`config.data.fov_deg`. A future improvement would be to load per-clip
intrinsics from the measurement JSONs (already saved by extract_nvidia.py)
and pass them through the batch.

Usage:
    python scripts/apply_fov_fix.py             # apply
    python scripts/apply_fov_fix.py --revert    # remove
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

TRAINER = Path(__file__).parent / "nvidia_trainer.py"
SENTINEL = "# FOV_APPLIED_v1"

# Patch 1: store fov_deg on trainer at init
INIT_ANCHOR = """        self.config = config
        self.ckpt_dir = Path(ckpt_dir)
"""
INIT_REPLACE = """        self.config = config
        """ + SENTINEL + """
        self.fov_deg = float(config.get("data", {}).get("fov_deg", 120.0))
        self.ckpt_dir = Path(ckpt_dir)
"""

# Patches 2-4: replace hardcoded 110 with self.fov_deg
SITE_REPLACEMENTS = [
    (
        '        intrinsics = self.get_camera_intrinsics(W, H, 110).unsqueeze(0).to(self.device).float()',
        '        intrinsics = self.get_camera_intrinsics(W, H, self.fov_deg).unsqueeze(0).to(self.device).float()',
    ),
    (
        '        K = self.get_camera_intrinsics(W, H, 110).to(self.device).float()        # [3, 3]',
        '        K = self.get_camera_intrinsics(W, H, self.fov_deg).to(self.device).float()        # [3, 3]',
    ),
    (
        '                                K = self.get_camera_intrinsics(W, H, 110).cpu().numpy()',
        '                                K = self.get_camera_intrinsics(W, H, self.fov_deg).cpu().numpy()',
    ),
]

PATCHES = [("init", INIT_ANCHOR, INIT_REPLACE)] + [
    (f"site_{i}", a, b) for i, (a, b) in enumerate(SITE_REPLACEMENTS)
]


def apply(content: str) -> str:
    if SENTINEL in content:
        print(f"Sentinel '{SENTINEL}' already present. No-op.", file=sys.stderr)
        return content
    new = content
    for name, anchor, replacement in PATCHES:
        count = new.count(anchor)
        if count != 1:
            raise RuntimeError(
                f"Anchor '{name}' matched {count} times (expected 1). Refusing."
            )
        new = new.replace(anchor, replacement, 1)
        print(f"  applied: {name}", file=sys.stderr)
    return new


def revert(content: str) -> str:
    if SENTINEL not in content:
        print(f"Sentinel '{SENTINEL}' not present. No-op.", file=sys.stderr)
        return content
    new = content
    for name, anchor, replacement in reversed(PATCHES):
        count = new.count(replacement)
        if count != 1:
            raise RuntimeError(f"Reverse anchor '{name}' matched {count} times.")
        new = new.replace(replacement, anchor, 1)
        print(f"  reverted: {name}", file=sys.stderr)
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revert", action="store_true")
    args = parser.parse_args()
    content = TRAINER.read_text()
    new = revert(content) if args.revert else apply(content)
    TRAINER.write_text(new)
    print(f"OK: wrote {TRAINER}", file=sys.stderr)


if __name__ == "__main__":
    main()
