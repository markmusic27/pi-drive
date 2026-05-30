"""Visualize NVIDIA data with projected waypoints to validate conversion.

This script generates overlay visualizations showing:
  - Camera frame from NVIDIA dataset
  - Ground-truth waypoints projected to image (green gradient: near→far)
  - Speed and timestamp info

Use this to verify that the egomotion→waypoints conversion is correct
before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import physical_ai_av


def _safe_font(size: int = 14) -> ImageFont.ImageFont:
    """Get a font that works across systems."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except Exception:
            return ImageFont.load_default()


def project_waypoints_to_image(
    waypoints: np.ndarray,  # [N, 2] ego frame (x_fwd, y_left)
    K: np.ndarray,          # [3, 3] camera intrinsics
    cam_extrinsics,         # RigidTransform: camera pose in ego frame
    image_size: tuple[int, int] = (1920, 1080),
) -> list[tuple[int, int]]:
    """Project ego-frame waypoints to image pixels.

    Args:
        waypoints: Shape [N, 2] with (x_forward, y_left) in ego frame.
        K: Camera intrinsics matrix [3, 3].
        cam_extrinsics: RigidTransform from ego to camera frame.
        image_size: (width, height) for bounds checking.

    Returns:
        List of (u, v) pixel coordinates. Points behind camera are filtered.
    """
    w, h = image_size
    pixels = []

    # Get inverse transform: ego → camera
    cam_inv = cam_extrinsics.inverse()

    for x_fwd, y_left in waypoints:
        # Convert ego point to 3D (z=0, on ground plane)
        ego_point = np.array([x_fwd, y_left, 0.0])

        # Transform to camera frame
        cam_point = cam_inv.apply(ego_point)

        # Camera frame convention (typically): x_right, y_down, z_forward
        # The exact convention depends on NVIDIA's setup, but typically:
        # cam_x = -ego_y (right)
        # cam_y = -ego_z (down)
        # cam_z = ego_x (forward)

        # Check if point is in front of camera
        if cam_point[2] <= 0:
            continue

        # Project to image plane
        u = K[0, 0] * cam_point[0] / cam_point[2] + K[0, 2]
        v = K[1, 1] * cam_point[1] / cam_point[2] + K[1, 2]

        # Bounds check
        if 0 <= u < w and 0 <= v < h:
            pixels.append((int(u), int(v)))

    return pixels


def project_waypoints_simple(
    waypoints: np.ndarray,  # [N, 2] ego frame (x_fwd, y_left)
    K: np.ndarray,          # [3, 3] camera intrinsics
    cam_translation_xyz: tuple[float, float, float] = (0.0, 1.5, 2.0),
    image_size: tuple[int, int] = (1920, 1080),
) -> list[tuple[int, int]]:
    """Project waypoints using simple pinhole model (no rotation).

    This is a fallback when full extrinsics aren't available.
    Uses the same convention as viz.py for consistency.

    Args:
        waypoints: Shape [N, 2] with (x_forward, y_left) in ego frame.
        K: Camera intrinsics [3, 3].
        cam_translation_xyz: Camera position (X_right, Y_up, Z_forward).
        image_size: (width, height).

    Returns:
        List of (u, v) pixel coordinates.
    """
    w, h = image_size
    cam_x, cam_y, cam_z = cam_translation_xyz
    pixels = []

    for x_fwd, y_left in waypoints:
        # Convert ego frame to camera frame
        # Ego: (x_fwd, y_left, 0) on ground plane
        # Camera: X_right, Y_down, Z_forward
        cam_X = -y_left - cam_x  # X_right = -y_left
        cam_Y = cam_y            # Y_down from camera height
        cam_Z = x_fwd + cam_z    # Z_forward

        if cam_Z <= 0.1:  # Behind camera
            continue

        # Project
        u = K[0, 0] * cam_X / cam_Z + K[0, 2]
        v = K[1, 1] * cam_Y / cam_Z + K[1, 2]

        if 0 <= u < w and 0 <= v < h:
            pixels.append((int(u), int(v)))

    return pixels


def draw_waypoints_on_image(
    image: np.ndarray,
    waypoints: np.ndarray,
    K: np.ndarray,
    cam_translation_xyz: tuple[float, float, float] = (0.0, 1.5, 2.0),
    speed_mps: float = 0.0,
    timestamp_s: float = 0.0,
    frame_idx: int = 0,
    clip_id: str = "",
) -> Image.Image:
    """Draw waypoints and info on image.

    Args:
        image: RGB image array.
        waypoints: Shape [N, 2] waypoints in ego frame.
        K: Camera intrinsics.
        cam_translation_xyz: Camera mount position.
        speed_mps: Vehicle speed for display.
        timestamp_s: Timestamp in clip.
        frame_idx: Frame index.
        clip_id: Clip identifier.

    Returns:
        PIL Image with overlay.
    """
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    # Project waypoints
    pixels = project_waypoints_simple(
        waypoints, K, cam_translation_xyz, image_size=(w, h)
    )

    # Draw waypoints with color gradient (green=near → red=far)
    num_wps = len(waypoints)
    for j, (u, v) in enumerate(pixels):
        # Color gradient
        progress = j / max(num_wps - 1, 1)
        r = int(255 * progress)
        g = int(255 * (1 - progress))
        b = 0

        radius = 6
        draw.ellipse(
            (u - radius, v - radius, u + radius, v + radius),
            fill=(r, g, b),
            outline=(255, 255, 255),
            width=1,
        )

        # Draw connecting lines
        if j > 0 and len(pixels) > j:
            prev_u, prev_v = pixels[j - 1]
            draw.line([(prev_u, prev_v), (u, v)], fill=(r, g, b, 128), width=2)

    # Add info overlay
    font = _safe_font(16)
    font_large = _safe_font(20)

    # Semi-transparent background for text
    overlay_height = 80
    overlay = Image.new("RGBA", (w, overlay_height), (0, 0, 0, 180))
    pil_img.paste(overlay, (0, 0), overlay)

    # Draw text
    y = 8
    draw.text(
        (10, y),
        f"Clip: {clip_id[:20]}..." if len(clip_id) > 20 else f"Clip: {clip_id}",
        fill=(255, 255, 255),
        font=font_large,
    )
    y += 24

    draw.text(
        (10, y),
        f"Frame: {frame_idx}  |  Time: {timestamp_s:.2f}s  |  Speed: {speed_mps:.1f} m/s ({speed_mps * 2.237:.1f} mph)",
        fill=(200, 200, 200),
        font=font,
    )
    y += 20

    draw.text(
        (10, y),
        f"Waypoints: {len(pixels)}/{len(waypoints)} visible  |  Horizon: 2.75s  |  Spacing: 0.25s",
        fill=(200, 200, 200),
        font=font,
    )

    # Legend
    legend_x = w - 200
    draw.text((legend_x, 8), "Near (0.25s)", fill=(0, 255, 0), font=font)
    draw.text((legend_x, 28), "Far (2.75s)", fill=(255, 0, 0), font=font)

    return pil_img


def visualize_clip_waypoints(
    avdi: "physical_ai_av.PhysicalAIAVDatasetInterface",
    clip_id: str,
    output_dir: Path,
    num_frames: int = 10,
    camera: str = "camera_front_wide_120fov",
) -> None:
    """Generate visualization frames for a single clip.

    Args:
        avdi: PhysicalAI-AV dataset interface.
        clip_id: Clip to visualize.
        output_dir: Directory for output PNGs.
        num_frames: Number of frames to visualize.
        camera: Camera to use.
    """
    import physical_ai_av

    from .nvidia_loader import (
        egomotion_to_waypoints,
        get_camera_config,
        get_speed_from_egomotion,
    )

    print(f"Loading clip: {clip_id}", flush=True)

    # Load clip data
    video = avdi.get_clip_feature(
        clip_id,
        getattr(avdi.features.CAMERA, camera.upper()),
    )
    egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION)
    cam_config = get_camera_config(avdi, clip_id, camera)

    # Create interpolator
    interpolator = physical_ai_av.utils.interpolation.Interpolator([egomotion])

    # Sample timestamps (avoid last 3s for waypoint horizon)
    max_ts_us = 17_000_000
    timestamps_us = np.linspace(0, max_ts_us, num_frames).astype(int)

    # Decode frames
    print(f"Decoding {num_frames} frames...", flush=True)
    frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating visualizations...", flush=True)
    for i, (frame, ts_us) in enumerate(zip(frames, actual_ts)):
        # Compute waypoints
        waypoints = egomotion_to_waypoints(interpolator, ts_us)
        speed = get_speed_from_egomotion(interpolator, ts_us)

        # Draw overlay
        overlay_img = draw_waypoints_on_image(
            image=frame,
            waypoints=waypoints,
            K=cam_config["intrinsics"],
            cam_translation_xyz=cam_config["cam_translation_xyz"],
            speed_mps=speed,
            timestamp_s=ts_us / 1_000_000,
            frame_idx=i,
            clip_id=clip_id,
        )

        # Save
        out_path = output_dir / f"frame_{i:04d}.png"
        overlay_img.save(out_path)
        print(f"  Saved: {out_path.name} (speed={speed:.1f} m/s)", flush=True)

    print(f"\nDone! Saved {num_frames} frames to {output_dir}", flush=True)


def create_visualization_video(
    image_dir: Path,
    output_path: Path,
    fps: int = 10,
) -> None:
    """Stitch visualization frames into a video.

    Args:
        image_dir: Directory containing frame_NNNN.png files.
        output_path: Output MP4 path.
        fps: Frames per second.
    """
    image_paths = sorted(image_dir.glob("frame_*.png"))
    if not image_paths:
        raise ValueError(f"No frames found in {image_dir}")

    first = cv2.imread(str(image_paths[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))

    try:
        for path in image_paths:
            frame = cv2.imread(str(path))
            writer.write(frame)
    finally:
        writer.release()

    print(f"Wrote video: {output_path} ({len(image_paths)} frames @ {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Visualize NVIDIA data with waypoints")
    parser.add_argument("--clip-id", required=True, help="NVIDIA clip ID to visualize")
    parser.add_argument("--output-dir", default="viz_output", help="Output directory")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames")
    parser.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env)")
    parser.add_argument("--make-video", action="store_true", help="Create MP4 from frames")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")

    args = parser.parse_args()

    import os

    import physical_ai_av

    # Get HF token
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required (--hf-token or HF_TOKEN env)")

    # Initialize dataset interface
    print("Initializing NVIDIA PhysicalAI-AV interface...", flush=True)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(token=token)

    output_dir = Path(args.output_dir)

    # Generate frames
    visualize_clip_waypoints(
        avdi=avdi,
        clip_id=args.clip_id,
        output_dir=output_dir,
        num_frames=args.num_frames,
    )

    # Optionally create video
    if args.make_video:
        video_path = output_dir / "visualization.mp4"
        create_visualization_video(output_dir, video_path, fps=args.fps)


if __name__ == "__main__":
    main()
