"""Batch extraction of NVIDIA PhysicalAI-AV data for SimLingo training.

This script extracts frames and waypoints from the NVIDIA dataset into
SimLingo's training format, similar to how the CARLA data is structured.

Output structure:
    output_dir/
      <clip_id>/
        rgb/
          0000.jpg
          0001.jpg
          ...
        measurements/
          0000.json.gz
          0001.json.gz
          ...

Usage:
    # Extract tiny validation set (50 clips)
    python extract_nvidia.py --scale tiny --output-dir /data/nvidia_extracted

    # Extract small training set (500 clips)
    python extract_nvidia.py --scale small --output-dir /data/nvidia_extracted

    # Extract with custom clip list
    python extract_nvidia.py --clip-ids clip1,clip2,clip3 --output-dir /data/nvidia_extracted
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from tqdm import tqdm

if TYPE_CHECKING:
    import physical_ai_av


# Data scale presets
SCALE_CONFIGS = {
    "tiny": {"num_clips": 50, "frames_per_clip": 50, "description": "Validation (50 clips, 2.5K frames)"},
    "small": {"num_clips": 500, "frames_per_clip": 50, "description": "Experiment (500 clips, 25K frames)"},
    "medium": {"num_clips": 2500, "frames_per_clip": 50, "description": "Medium (2.5K clips, 125K frames)"},
    "large": {"num_clips": 15000, "frames_per_clip": 50, "description": "Full scale (15K clips, 750K frames)"},
}


def filter_clips_for_stanford_conditions(
    metadata_path: Path | None = None,
    max_clips: int | None = None,
) -> list[str]:
    """Filter NVIDIA clips for Stanford-like conditions.

    Filters for:
    - United States location
    - Daytime lighting
    - Clear weather (if available)
    - Has required features (egomotion, front camera)

    Args:
        metadata_path: Path to metadata parquet files (optional).
        max_clips: Maximum clips to return.

    Returns:
        List of filtered clip IDs.
    """
    try:
        import pandas as pd

        # Load metadata if available
        if metadata_path and (metadata_path / "data_collection.parquet").exists():
            metadata = pd.read_parquet(metadata_path / "data_collection.parquet")
            features = pd.read_parquet(metadata_path / "feature_presence.parquet")

            # Apply filters
            filtered = metadata[
                (metadata['country'] == 'United States') &
                (metadata['time_of_day'] == 'daytime')
            ].copy()

            # Join with feature presence
            filtered = filtered.merge(features, on='clip_id')
            filtered = filtered[
                (filtered['has_egomotion'] == True) &
                (filtered['has_camera_front_wide'] == True)
            ]

            clip_ids = filtered['clip_id'].tolist()
            print(f"Filtered to {len(clip_ids)} clips matching Stanford conditions")

            if max_clips:
                clip_ids = clip_ids[:max_clips]

            return clip_ids

    except Exception as e:
        print(f"Warning: Could not load metadata ({e}), will sample clips dynamically")

    return []


def extract_clip(
    avdi: "physical_ai_av.PhysicalAIAVDatasetInterface",
    clip_id: str,
    output_dir: Path,
    frames_per_clip: int = 50,
    camera: str = "camera_front_wide_120fov",
    waypoint_horizon_s: float = 2.75,
    waypoint_spacing_s: float = 0.25,
    num_waypoints: int = 11,
) -> dict:
    """Extract a single clip to disk.

    Args:
        avdi: PhysicalAI-AV dataset interface.
        clip_id: Clip to extract.
        output_dir: Base output directory.
        frames_per_clip: Number of frames to extract.
        camera: Camera to use.
        waypoint_horizon_s: Waypoint prediction horizon.
        waypoint_spacing_s: Time between waypoints.
        num_waypoints: Number of waypoints per frame.

    Returns:
        dict with extraction statistics.
    """
    import physical_ai_av

    from .nvidia_loader import (
        egomotion_to_route,
        egomotion_to_waypoints,
        get_camera_config,
        get_speed_from_egomotion,
        get_target_points,
    )

    clip_dir = output_dir / clip_id
    rgb_dir = clip_dir / "rgb"
    meas_dir = clip_dir / "measurements"

    # Skip if already extracted
    if (rgb_dir / f"{frames_per_clip - 1:04d}.jpg").exists():
        return {"clip_id": clip_id, "status": "skipped", "frames": 0}

    rgb_dir.mkdir(parents=True, exist_ok=True)
    meas_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load clip data (with streaming enabled)
        video = avdi.get_clip_feature(
            clip_id,
            getattr(avdi.features.CAMERA, camera.upper()),
            maybe_stream=True,
        )
        egomotion = avdi.get_clip_feature(
            clip_id,
            avdi.features.LABELS.EGOMOTION,
            maybe_stream=True,
        )
        cam_config = get_camera_config(avdi, clip_id, camera)

        # Create or use interpolator (egomotion may already be Interpolator when streaming)
        if isinstance(egomotion, physical_ai_av.utils.interpolation.Interpolator):
            interpolator = egomotion
        else:
            interpolator = physical_ai_av.utils.interpolation.Interpolator([egomotion])

        # Sample timestamps (avoid last 3s for waypoint horizon)
        max_ts_us = 17_000_000
        timestamps_us = np.linspace(0, max_ts_us, frames_per_clip).astype(int)

        # Decode all frames at once
        frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)

        extracted = 0
        for i, (frame, ts_us) in enumerate(zip(frames, actual_ts)):
            try:
                # Save image
                img_path = rgb_dir / f"{i:04d}.jpg"
                Image.fromarray(frame).save(img_path, quality=95)

                # Compute waypoints
                waypoints = egomotion_to_waypoints(
                    interpolator, ts_us,
                    horizon_s=waypoint_horizon_s,
                    spacing_s=waypoint_spacing_s,
                    num_waypoints=num_waypoints,
                )

                # Compute route
                route = egomotion_to_route(interpolator, ts_us)

                # Get speed and target points
                speed = get_speed_from_egomotion(interpolator, ts_us)
                target_points = get_target_points(interpolator, ts_us)

                # Build measurement dict (matching SimLingo CARLA format)
                measurement = {
                    "speed": float(speed),
                    "waypoints": waypoints.tolist(),
                    "route": route.tolist(),
                    "target_point": target_points[0].tolist(),
                    "target_point_next": target_points[1].tolist(),
                    "timestamp_us": int(ts_us),
                    "clip_id": clip_id,
                    "frame_idx": i,
                    # Camera info for visualization
                    "camera": camera,
                    "fov_deg": cam_config["fov_deg"],
                    "intrinsics": cam_config["intrinsics"].tolist(),
                }

                # Save measurement
                meas_path = meas_dir / f"{i:04d}.json.gz"
                with gzip.open(meas_path, "wt") as f:
                    json.dump(measurement, f)

                extracted += 1

            except Exception as e:
                print(f"  Frame {i} error: {e}")
                continue

        return {
            "clip_id": clip_id,
            "status": "success",
            "frames": extracted,
        }

    except Exception as e:
        return {
            "clip_id": clip_id,
            "status": "error",
            "error": str(e),
            "frames": 0,
        }


def extract_dataset(
    avdi: "physical_ai_av.PhysicalAIAVDatasetInterface",
    clip_ids: list[str],
    output_dir: Path,
    frames_per_clip: int = 50,
    num_workers: int = 4,
    camera: str = "camera_front_wide_120fov",
) -> dict:
    """Extract multiple clips in parallel.

    Args:
        avdi: PhysicalAI-AV dataset interface.
        clip_ids: List of clips to extract.
        output_dir: Base output directory.
        frames_per_clip: Frames per clip.
        num_workers: Parallel workers.
        camera: Camera to use.

    Returns:
        Summary statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    success = 0
    errors = 0
    skipped = 0
    total_frames = 0

    # Sequential extraction (safer for API rate limits)
    # Use ThreadPoolExecutor for I/O-bound video decoding
    print(f"Extracting {len(clip_ids)} clips to {output_dir}...")

    for clip_id in tqdm(clip_ids, desc="Extracting clips"):
        result = extract_clip(
            avdi=avdi,
            clip_id=clip_id,
            output_dir=output_dir,
            frames_per_clip=frames_per_clip,
            camera=camera,
        )
        results.append(result)

        if result["status"] == "success":
            success += 1
            total_frames += result["frames"]
        elif result["status"] == "skipped":
            skipped += 1
        else:
            errors += 1
            print(f"  Error on {clip_id}: {result.get('error', 'unknown')}")

    summary = {
        "total_clips": len(clip_ids),
        "success": success,
        "errors": errors,
        "skipped": skipped,
        "total_frames": total_frames,
        "output_dir": str(output_dir),
    }

    # Save extraction log
    log_path = output_dir / "extraction_log.json"
    with open(log_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nExtraction complete:")
    print(f"  Success: {success}, Errors: {errors}, Skipped: {skipped}")
    print(f"  Total frames: {total_frames}")
    print(f"  Log saved to: {log_path}")

    return summary


def create_train_val_split(
    data_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Create train/val split from extracted data.

    Args:
        data_dir: Directory containing extracted clips.
        val_ratio: Fraction for validation.
        seed: Random seed.

    Returns:
        dict with train/val clip lists.
    """
    import random

    data_dir = Path(data_dir)
    clip_dirs = [d for d in data_dir.iterdir() if d.is_dir() and (d / "rgb").exists()]

    random.seed(seed)
    random.shuffle(clip_dirs)

    val_count = int(len(clip_dirs) * val_ratio)
    val_clips = [d.name for d in clip_dirs[:val_count]]
    train_clips = [d.name for d in clip_dirs[val_count:]]

    split = {
        "train": train_clips,
        "val": val_clips,
        "train_count": len(train_clips),
        "val_count": len(val_clips),
    }

    # Save split
    split_path = data_dir / "train_val_split.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Created split: {len(train_clips)} train, {len(val_clips)} val")
    print(f"Saved to: {split_path}")

    return split


def sample_clip_ids_from_api(
    avdi: "physical_ai_av.PhysicalAIAVDatasetInterface",
    num_clips: int,
    seed: int = 42,
    daytime_only: bool = True,
) -> list[str]:
    """Sample clip IDs directly from the API.

    Uses clip_index to get valid clips, optionally filtered for daytime.

    Args:
        avdi: PhysicalAI-AV dataset interface.
        num_clips: Number of clips to sample.
        seed: Random seed.
        daytime_only: Filter for daytime clips (hours 6-18).

    Returns:
        List of clip IDs.
    """
    import random

    try:
        # Use clip_index to get valid clips
        clip_index = avdi.clip_index
        valid_mask = clip_index['clip_is_valid'] == True

        # Filter for daytime if requested
        if daytime_only and hasattr(avdi, 'data_collection'):
            try:
                dc = avdi.data_collection
                if 'hour_of_day' in dc.columns:
                    daytime_mask = (dc['hour_of_day'] >= 6) & (dc['hour_of_day'] <= 18)
                    daytime_clips = set(dc[daytime_mask].index.tolist())
                    valid_clips = [c for c in clip_index[valid_mask].index.tolist() if c in daytime_clips]
                    print(f"Filtered to {len(valid_clips)} daytime clips")
                else:
                    valid_clips = clip_index[valid_mask].index.tolist()
            except Exception as e:
                print(f"Daytime filter failed: {e}, using all valid clips")
                valid_clips = clip_index[valid_mask].index.tolist()
        else:
            valid_clips = clip_index[valid_mask].index.tolist()

        print(f"Found {len(valid_clips)} valid clips")

        random.seed(seed)
        random.shuffle(valid_clips)
        return valid_clips[:num_clips]

    except Exception as e:
        print(f"Warning: Could not get clips from API: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract NVIDIA data for SimLingo training")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--scale", choices=list(SCALE_CONFIGS.keys()),
                        help="Preset scale (tiny/small/medium/large)")
    parser.add_argument("--num-clips", type=int, help="Custom number of clips")
    parser.add_argument("--frames-per-clip", type=int, default=50,
                        help="Frames to extract per clip")
    parser.add_argument("--clip-ids", help="Comma-separated clip IDs")
    parser.add_argument("--hf-token", help="HuggingFace token")
    parser.add_argument("--metadata-path", help="Path to metadata parquet files")
    parser.add_argument("--create-split", action="store_true",
                        help="Create train/val split after extraction")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    import physical_ai_av

    # Get HF token
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required (--hf-token or HF_TOKEN env)")

    # Initialize dataset interface
    print("Initializing NVIDIA PhysicalAI-AV interface...")
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(token=token)

    # Determine clip list
    if args.clip_ids:
        clip_ids = [c.strip() for c in args.clip_ids.split(",")]
    elif args.scale:
        config = SCALE_CONFIGS[args.scale]
        print(f"Scale: {args.scale} - {config['description']}")
        num_clips = config["num_clips"]
        frames_per_clip = config["frames_per_clip"]

        # Try metadata filtering first
        metadata_path = Path(args.metadata_path) if args.metadata_path else None
        clip_ids = filter_clips_for_stanford_conditions(metadata_path, max_clips=num_clips)

        if not clip_ids:
            # Fall back to API sampling
            clip_ids = sample_clip_ids_from_api(avdi, num_clips, seed=args.seed)
    elif args.num_clips:
        clip_ids = sample_clip_ids_from_api(avdi, args.num_clips, seed=args.seed)
    else:
        raise ValueError("Must specify --scale, --num-clips, or --clip-ids")

    if not clip_ids:
        raise ValueError("No clips to extract")

    print(f"Will extract {len(clip_ids)} clips")

    # Extract
    output_dir = Path(args.output_dir)
    summary = extract_dataset(
        avdi=avdi,
        clip_ids=clip_ids,
        output_dir=output_dir,
        frames_per_clip=args.frames_per_clip,
    )

    # Create train/val split if requested
    if args.create_split:
        create_train_val_split(output_dir, val_ratio=args.val_ratio, seed=args.seed)

    print("\nDone!")
    return summary


if __name__ == "__main__":
    main()
