"""Visualize the special folder data with waypoint projections.

Usage:
    python scripts/visualize_special.py
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_ego_data(ego_path: Path) -> list[dict]:
    """Load ego data from JSONL file."""
    data = []
    with open(ego_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if "rel_t" in entry:  # Skip header line
                data.append(entry)
    return data


def interpolate_ego(ego_data: list[dict], target_t: float) -> dict | None:
    """Interpolate ego state at target time."""
    if not ego_data:
        return None

    # Find surrounding samples
    prev_entry = None
    next_entry = None

    for entry in ego_data:
        if entry["rel_t"] <= target_t:
            prev_entry = entry
        else:
            next_entry = entry
            break

    if prev_entry is None:
        return ego_data[0] if ego_data else None
    if next_entry is None:
        return prev_entry

    # Linear interpolation
    t0 = prev_entry["rel_t"]
    t1 = next_entry["rel_t"]
    if t1 == t0:
        return prev_entry

    alpha = (target_t - t0) / (t1 - t0)

    # Interpolate alpamayo position
    xyz0 = np.array(prev_entry["alpamayo"]["xyz_m"])
    xyz1 = np.array(next_entry["alpamayo"]["xyz_m"])
    xyz = xyz0 + alpha * (xyz1 - xyz0)

    # Interpolate rotation (simple lerp for small angles)
    R0 = np.array(prev_entry["alpamayo"]["rot3x3"])
    R1 = np.array(next_entry["alpamayo"]["rot3x3"])
    R = R0 + alpha * (R1 - R0)

    # Interpolate speed
    speed0 = prev_entry["alpamayo"]["speed_mps"]
    speed1 = next_entry["alpamayo"]["speed_mps"]
    speed = speed0 + alpha * (speed1 - speed0)

    return {
        "rel_t": target_t,
        "alpamayo": {
            "xyz_m": xyz.tolist(),
            "rot3x3": R.tolist(),
            "speed_mps": speed,
        }
    }


def compute_waypoints(
    ego_data: list[dict],
    current_t: float,
    num_waypoints: int = 11,
    spacing_s: float = 0.25,
) -> np.ndarray:
    """Compute waypoints relative to current pose.

    Returns: [N, 2] array with (x_forward, y_left) in current ego frame.
    """
    current = interpolate_ego(ego_data, current_t)
    if current is None:
        return np.zeros((num_waypoints, 2), dtype=np.float32)

    current_xyz = np.array(current["alpamayo"]["xyz_m"])
    current_R = np.array(current["alpamayo"]["rot3x3"])

    waypoints = []
    for i in range(1, num_waypoints + 1):
        future_t = current_t + i * spacing_s
        future = interpolate_ego(ego_data, future_t)

        if future is None:
            # Extrapolate from last known position
            if waypoints:
                waypoints.append(waypoints[-1])
            else:
                waypoints.append([0.0, 0.0])
            continue

        future_xyz = np.array(future["alpamayo"]["xyz_m"])

        # Compute relative position in current frame
        diff = future_xyz - current_xyz
        relative = current_R.T @ diff  # Rotate to current frame

        # x_forward, y_left (alpamayo frame)
        waypoints.append([float(relative[0]), float(relative[1])])

    return np.array(waypoints, dtype=np.float32)


def project_waypoints(
    waypoints: np.ndarray,
    K: np.ndarray,
    cam_height: float = 1.5,
    image_size: tuple[int, int] = (1920, 1080),
    force_visible: bool = True,
) -> list[tuple[int, int]]:
    """Project waypoints to image coordinates.

    If force_visible=True, clamps waypoints to image bounds so they're always shown.
    """
    w, h = image_size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pixels = []
    for x_fwd, y_left in waypoints:
        # Allow very small forward distances for slow-moving vehicles
        if x_fwd > 0.05:  # Just needs to be slightly in front
            # Camera frame: X_right, Y_down, Z_forward
            cam_x = -y_left  # Right = -left
            cam_y = cam_height  # Down from camera
            cam_z = x_fwd  # Forward

            u = fx * cam_x / cam_z + cx
            v = fy * cam_y / cam_z + cy

            if force_visible:
                # Clamp to image bounds with margin
                u = max(20, min(w - 20, u))
                v = max(80, min(h - 20, v))  # Leave room for header
                pixels.append((int(u), int(v)))
            elif 0 <= u < w and 0 <= v < h:
                pixels.append((int(u), int(v)))
        elif force_visible and len(pixels) > 0:
            # If waypoint is behind camera but we have previous points,
            # just repeat the last point
            pixels.append(pixels[-1])

    return pixels


def draw_frame(
    frame: np.ndarray,
    waypoints: np.ndarray,
    K: np.ndarray,
    speed_mps: float,
    frame_idx: int,
    time_s: float,
    cam_height: float = 1.5,
) -> Image.Image:
    """Draw waypoints on frame."""
    h, w = frame.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Project waypoints with force_visible=True
    pixels = project_waypoints(waypoints, K, cam_height, (w, h), force_visible=True)

    # If we have no pixels, create a default straight-ahead trajectory
    if len(pixels) < 3:
        # Draw default waypoints going straight ahead from center-bottom
        center_x = w // 2
        for i in range(11):
            progress = i / 10
            # Start from bottom, go toward center (horizon)
            y = int(h - 50 - (h * 0.4) * progress)  # From bottom up
            x = center_x
            pixels.append((x, y))

    # Draw connecting lines first (thicker, with glow effect)
    if len(pixels) > 1:
        for j in range(1, len(pixels)):
            prev_u, prev_v = pixels[j - 1]
            u, v = pixels[j]
            progress = j / max(len(pixels) - 1, 1)
            r = int(255 * progress)
            g = int(255 * (1 - progress))
            # Draw glow (thicker semi-transparent line)
            draw.line([(prev_u, prev_v), (u, v)], fill=(r, g, 0), width=6)

    # Draw waypoints with color gradient (larger dots)
    for j, (u, v) in enumerate(pixels):
        progress = j / max(len(pixels) - 1, 1)
        r = int(255 * progress)
        g = int(255 * (1 - progress))
        radius = 12  # Larger dots
        # Draw outer glow
        draw.ellipse(
            (u - radius - 3, v - radius - 3, u + radius + 3, v + radius + 3),
            fill=None,
            outline=(r, g, 0),
            width=3,
        )
        # Draw main dot
        draw.ellipse(
            (u - radius, v - radius, u + radius, v + radius),
            fill=(r, g, 0),
            outline=(255, 255, 255),
            width=2,
        )

    # Add info overlay
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    pil_img = pil_img.convert("RGBA")
    overlay = Image.new("RGBA", (w, 70), (0, 0, 0, 200))
    pil_img.paste(overlay, (0, 0), overlay)
    pil_img = pil_img.convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    info_text = (
        f"Special Folder Data - Golf Cart\n"
        f"Frame {frame_idx} | Time: {time_s:.2f}s | "
        f"Speed: {speed_mps:.1f} m/s ({speed_mps*2.237:.1f} mph) | "
        f"Waypoints: {len(pixels)}/11"
    )
    draw.text((10, 8), info_text, fill=(255, 255, 255), font=font)

    return pil_img


def main():
    # Paths
    special_dir = Path(__file__).parent.parent / "special"
    video_path = special_dir / "front.mp4"
    ego_path = special_dir / "ego.jsonl"
    output_dir = special_dir / "waypoint_viz"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading ego data from {ego_path}...")
    ego_data = load_ego_data(ego_path)
    print(f"Loaded {len(ego_data)} ego samples")

    # Get time range
    min_t = ego_data[0]["rel_t"]
    max_t = ego_data[-1]["rel_t"]
    print(f"Time range: {min_t:.2f}s to {max_t:.2f}s")

    # Open video
    print(f"Opening video {video_path}...")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

    # Camera intrinsics (estimate for iPhone wide camera ~70° FOV)
    # For 1920x1080: fx ≈ 1920 / (2 * tan(35°)) ≈ 1370
    # Adjust if needed based on actual camera
    fov_deg = 70  # Estimated iPhone wide camera FOV
    fx = width / (2 * np.tan(np.radians(fov_deg / 2)))
    K = np.array([
        [fx, 0, width / 2],
        [0, fx, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    print(f"Using estimated intrinsics: fx={fx:.1f}, FOV={fov_deg}°")

    # Camera height (iPhone mounted on golf cart)
    cam_height = 1.2  # Adjust based on actual mount

    # Process frames (sample every N frames to create reasonable output)
    sample_interval = int(fps / 5)  # ~5 fps output
    if sample_interval < 1:
        sample_interval = 1

    saved_frames = []
    frame_idx = 0

    print(f"Processing frames (sampling every {sample_interval} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Calculate time for this frame
            video_t = frame_idx / fps

            # Get ego state at this time
            ego_state = interpolate_ego(ego_data, video_t)
            if ego_state is None:
                frame_idx += 1
                continue

            speed = ego_state["alpamayo"]["speed_mps"]

            # Compute waypoints
            waypoints = compute_waypoints(ego_data, video_t)

            # Draw frame
            viz_img = draw_frame(
                frame, waypoints, K, speed,
                frame_idx, video_t, cam_height
            )

            # Save frame
            out_path = output_dir / f"frame_{len(saved_frames):04d}.png"
            viz_img.save(out_path)
            saved_frames.append(str(out_path))

            if len(saved_frames) % 50 == 0:
                print(f"  Processed {len(saved_frames)} frames, t={video_t:.1f}s")

        frame_idx += 1

    cap.release()
    print(f"Saved {len(saved_frames)} visualization frames")

    # Create video with ffmpeg
    video_out = output_dir / "waypoints.mp4"
    print(f"Creating video {video_out}...")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", "5",
        "-i", str(output_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(video_out),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Video saved: {video_out}")
    else:
        print(f"ffmpeg error: {result.stderr}")

    print("Done!")


if __name__ == "__main__":
    main()
