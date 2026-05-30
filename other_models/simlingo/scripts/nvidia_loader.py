"""NVIDIA PhysicalAI-AV dataset adapter for SimLingo inference and training.

This module adapts NVIDIA's real-world driving data to SimLingo's expected input format.
Key tasks:
  - Load clips via physical_ai_av SDK
  - Convert egomotion to SimLingo-format waypoints (ego-frame, x_forward, y_left)
  - Build ExternalSample objects for inference pipeline

Coordinate convention (matches SimLingo/nuScenes):
  x = forward (along heading direction)
  y = left
Units: meters.

NVIDIA ego frame is +x_forward, +y_left, +z_up — same as SimLingo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    import physical_ai_av

# Re-export ExternalSample for convenience (defined in nuscenes_loader.py).
from .nuscenes_loader import ExternalSample, _route_resample_1m


# ---------------------------------------------------------------------------
# Egomotion → Waypoints conversion
# ---------------------------------------------------------------------------


def _get_rotation_matrix(pose) -> np.ndarray:
    """Extract 3x3 rotation matrix from pose."""
    if hasattr(pose, 'rotation'):
        rot = pose.rotation
        if hasattr(rot, 'as_matrix'):
            return rot.as_matrix()
        elif hasattr(rot, 'R'):
            return rot.R
        else:
            try:
                return np.array(rot).reshape(3, 3)
            except Exception:
                return np.eye(3)
    return np.eye(3)


def egomotion_to_waypoints(
    interpolator,
    frame_timestamp_us: int,
    horizon_s: float = 2.75,
    spacing_s: float = 0.25,
    num_waypoints: int = 11,
) -> np.ndarray:
    """Convert NVIDIA egomotion to SimLingo-format waypoints.

    NVIDIA egomotion is ego-centric with origin at clip start (t=0). For each
    frame, we compute waypoints relative to *that frame's* pose.

    Args:
        interpolator: physical_ai_av Interpolator with egomotion loaded.
        frame_timestamp_us: Current frame timestamp in microseconds.
        horizon_s: Total prediction horizon in seconds.
        spacing_s: Time spacing between waypoints.
        num_waypoints: Number of waypoints to generate.

    Returns:
        np.ndarray: Shape [num_waypoints, 2] in ego frame (x_forward, y_left).
                    Note: SimLingo uses y_left (same as NVIDIA), NOT y_right.
    """
    # Get current ego pose at frame time
    current_state = interpolator(frame_timestamp_us)
    current_pose = current_state.pose  # RigidTransform

    # Get rotation matrix for transforming to current frame
    R_current = _get_rotation_matrix(current_pose)
    current_trans = np.array(current_pose.translation)

    waypoints = []
    for i in range(1, num_waypoints + 1):
        future_us = frame_timestamp_us + int(i * spacing_s * 1_000_000)
        try:
            future_state = interpolator(future_us)
        except Exception:
            # Beyond available data — extrapolate from last known position
            if waypoints:
                waypoints.append(waypoints[-1].copy())
            else:
                waypoints.append(np.array([0.0, 0.0], dtype=np.float32))
            continue

        future_pose = future_state.pose
        future_trans = np.array(future_pose.translation)

        # Transform future pose to current ego frame
        # Compute relative position: R_current^T @ (future - current)
        diff = future_trans - current_trans
        relative_pos = R_current.T @ diff

        # Extract x_forward, y_left (ignore z_up)
        # NVIDIA uses: x_forward, y_left, z_up (same as SimLingo!)
        x_fwd = float(relative_pos[0])
        y_left = float(relative_pos[1])

        waypoints.append(np.array([x_fwd, y_left], dtype=np.float32))

    return np.array(waypoints, dtype=np.float32)


def egomotion_to_route(
    interpolator,
    frame_timestamp_us: int,
    horizon_s: float = 2.75,
    num_points: int = 20,
) -> np.ndarray:
    """Compute 1m-spaced route waypoints from egomotion trajectory.

    Args:
        interpolator: physical_ai_av Interpolator with egomotion loaded.
        frame_timestamp_us: Current frame timestamp in microseconds.
        horizon_s: Total horizon in seconds.
        num_points: Number of route points (1m spacing).

    Returns:
        np.ndarray: Shape [num_points, 2] in ego frame (x_forward, y_left).
    """
    # Sample trajectory at high resolution
    sample_spacing_us = 50_000  # 50ms = 20 Hz
    num_samples = int(horizon_s * 1_000_000 / sample_spacing_us) + 1

    current_state = interpolator(frame_timestamp_us)
    current_pose = current_state.pose
    R_current = _get_rotation_matrix(current_pose)
    current_trans = np.array(current_pose.translation)

    trajectory = []
    for i in range(num_samples):
        future_us = frame_timestamp_us + i * sample_spacing_us
        try:
            future_state = interpolator(future_us)
            future_pose = future_state.pose
            future_trans = np.array(future_pose.translation)
            diff = future_trans - current_trans
            relative_pos = R_current.T @ diff
            trajectory.append([float(relative_pos[0]), float(relative_pos[1])])
        except Exception:
            break

    if len(trajectory) < 2:
        # Not enough data, return straight line
        return np.array([[i, 0.0] for i in range(num_points)], dtype=np.float32)

    trajectory = np.array(trajectory, dtype=np.float32)
    return _route_resample_1m(trajectory, num=num_points)


def get_speed_from_egomotion(
    interpolator,
    frame_timestamp_us: int,
    window_s: float = 0.5,
) -> float:
    """Compute vehicle speed from egomotion velocity.

    Args:
        interpolator: physical_ai_av Interpolator with egomotion loaded.
        frame_timestamp_us: Current frame timestamp in microseconds.
        window_s: Time window for velocity averaging.

    Returns:
        Speed in m/s (2D magnitude).
    """
    try:
        state = interpolator(frame_timestamp_us)
        # Velocity is in ego frame: [vx_forward, vy_left, vz_up]
        velocity = state.velocity
        speed = np.linalg.norm(velocity[:2])  # 2D speed
        return float(speed)
    except Exception:
        return 0.0


def get_target_points(
    interpolator,
    frame_timestamp_us: int,
    horizon_1_s: float = 2.0,
    horizon_2_s: float = 4.0,
) -> np.ndarray:
    """Get target points for SimLingo prompt (2s and 4s ahead).

    Args:
        interpolator: physical_ai_av Interpolator with egomotion loaded.
        frame_timestamp_us: Current frame timestamp in microseconds.
        horizon_1_s: First target point horizon.
        horizon_2_s: Second target point horizon.

    Returns:
        np.ndarray: Shape [2, 2] with two target points in ego frame.
    """
    current_state = interpolator(frame_timestamp_us)
    current_pose = current_state.pose
    R_current = _get_rotation_matrix(current_pose)
    current_trans = np.array(current_pose.translation)

    target_points = []
    for horizon_s in [horizon_1_s, horizon_2_s]:
        future_us = frame_timestamp_us + int(horizon_s * 1_000_000)
        try:
            future_state = interpolator(future_us)
            future_trans = np.array(future_state.pose.translation)
            diff = future_trans - current_trans
            relative_pos = R_current.T @ diff
            target_points.append([float(relative_pos[0]), float(relative_pos[1])])
        except Exception:
            # Extrapolate based on current speed
            speed = get_speed_from_egomotion(interpolator, frame_timestamp_us)
            target_points.append([speed * horizon_s, 0.0])

    return np.array(target_points, dtype=np.float32)


# ---------------------------------------------------------------------------
# Camera intrinsics helpers
# ---------------------------------------------------------------------------


def get_camera_config(
    avdi: "physical_ai_av.PhysicalAIAVDatasetInterface",
    clip_id: str,
    camera_name: str = "camera_front_wide_120fov",
) -> dict:
    """Get camera configuration from NVIDIA calibration data.

    Args:
        avdi: PhysicalAI-AV dataset interface.
        clip_id: Clip identifier.
        camera_name: Camera sensor name.

    Returns:
        dict with intrinsics matrix, FOV, image size, and extrinsics.
    """
    # Default values for 120° FOV wide camera
    w = 1920  # NVIDIA front camera width
    h = 1080  # NVIDIA front camera height
    default_fov = 120.0
    default_fx = w / (2 * np.tan(np.radians(default_fov / 2)))  # ~554

    try:
        intrinsics = avdi.get_clip_feature(
            clip_id, avdi.features.CALIBRATION.CAMERA_INTRINSICS, maybe_stream=True
        )
        # Check if we have a method to get the camera matrix
        if hasattr(intrinsics, 'get_camera_matrix'):
            K = intrinsics.get_camera_matrix(camera_name)
        elif hasattr(intrinsics, 'camera_models'):
            # Fallback: try to access camera_models dict
            cam_model = intrinsics.camera_models.get(camera_name)
            if cam_model and hasattr(cam_model, 'K'):
                K = cam_model.K
            else:
                raise ValueError("No camera matrix found")
        else:
            raise ValueError("Unknown intrinsics format")

        fx = K[0, 0]
        fov_deg = float(2 * np.degrees(np.arctan(w / (2 * fx))))

    except Exception as e:
        # Use default intrinsics for 120° FOV
        K = np.array([
            [default_fx, 0, w / 2],
            [0, default_fx, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        fov_deg = default_fov

    # Default camera mount position (roof-mounted, looking forward)
    cam_translation_xyz = (0.0, 1.5, 2.0)  # (X_right, Y_up, Z_forward)

    try:
        extrinsics = avdi.get_clip_feature(
            clip_id, avdi.features.CALIBRATION.SENSOR_EXTRINSICS, maybe_stream=True
        )
        if hasattr(extrinsics, 'get_transform'):
            cam_ext = extrinsics.get_transform(camera_name)
            cam_t = cam_ext.translation
            cam_translation_xyz = (
                -float(cam_t[1]),  # X_right = -y_left
                float(cam_t[2]),   # Y_up = z_up
                float(cam_t[0]),   # Z_forward = x_forward
            )
    except Exception:
        pass  # Use default

    return {
        "intrinsics": np.array(K, dtype=np.float32),
        "fov_deg": fov_deg,
        "image_size": [w, h],
        "cam_translation_xyz": cam_translation_xyz,
    }


# ---------------------------------------------------------------------------
# Sample iteration
# ---------------------------------------------------------------------------


def iter_nvidia_samples(
    *,
    avdi: "physical_ai_av.PhysicalAIAVDatasetInterface",
    clip_ids: list[str],
    camera: str = "camera_front_wide_120fov",
    horizon_sec_target: float = 2.0,
    horizon_sec_next_target: float = 4.0,
    horizon_sec_gt_wps: float = 2.75,
    num_gt_wps: int = 11,
    frames_per_clip: int = 50,
    max_samples: int | None = None,
) -> Iterable[ExternalSample]:
    """Yield ExternalSamples from NVIDIA PhysicalAI-AV clips.

    Args:
        avdi: PhysicalAI-AV dataset interface (authenticated).
        clip_ids: List of clip IDs to process.
        camera: Camera feature to use.
        horizon_sec_target / horizon_sec_next_target: Target point horizons.
        horizon_sec_gt_wps: Total horizon for GT waypoints.
        num_gt_wps: Number of GT waypoints.
        frames_per_clip: Number of frames to sample per clip.
        max_samples: Optional cap on total samples.

    Yields:
        ExternalSample objects ready for SimLingo inference.
    """
    import physical_ai_av

    yielded = 0

    for clip_id in clip_ids:
        if max_samples is not None and yielded >= max_samples:
            return

        try:
            # Load clip data
            video = avdi.get_clip_feature(
                clip_id,
                getattr(avdi.features.CAMERA, camera.upper()),
            )
            egomotion = avdi.get_clip_feature(
                clip_id, avdi.features.LABELS.EGOMOTION
            )

            # Get camera config
            cam_config = get_camera_config(avdi, clip_id, camera)

            # Create interpolator for egomotion
            interpolator = physical_ai_av.utils.interpolation.Interpolator([egomotion])

            # Clip is ~20s, leave room for waypoint horizon
            max_ts_us = 17_000_000  # 17s
            timestamps_us = np.linspace(0, max_ts_us, frames_per_clip).astype(int)

            # Decode frames at requested timestamps
            frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)

        except Exception as e:
            print(f"  WARN: failed to load clip {clip_id}: {e}", flush=True)
            continue

        for frame_idx, (frame, ts_us) in enumerate(zip(frames, actual_ts)):
            if max_samples is not None and yielded >= max_samples:
                return

            try:
                # Compute waypoints
                gt_wps = egomotion_to_waypoints(
                    interpolator, ts_us,
                    horizon_s=horizon_sec_gt_wps,
                    num_waypoints=num_gt_wps,
                )

                # Compute route
                gt_route = egomotion_to_route(
                    interpolator, ts_us,
                    horizon_s=horizon_sec_gt_wps,
                )

                # Get speed
                speed_mps = get_speed_from_egomotion(interpolator, ts_us)

                # Get target points
                target_points = get_target_points(
                    interpolator, ts_us,
                    horizon_1_s=horizon_sec_target,
                    horizon_2_s=horizon_sec_next_target,
                )

                # Save frame temporarily for inference
                # Note: In extraction mode, we save to disk; here we use temp path
                import tempfile
                from PIL import Image

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    Image.fromarray(frame).save(f.name, quality=95)
                    rgb_path = Path(f.name)

                yield ExternalSample(
                    rgb_path=rgb_path,
                    speed_mps=speed_mps,
                    target_points=target_points,
                    intrinsics=cam_config["intrinsics"],
                    fov_deg=cam_config["fov_deg"],
                    cam_translation_xyz=cam_config["cam_translation_xyz"],
                    crop_bottom=False,  # NVIDIA cameras are roof-mounted
                    gt_wps=gt_wps,
                    gt_route=gt_route,
                    gt_commentary=None,  # No commentary available yet
                    meta={
                        "clip_id": clip_id,
                        "frame_idx": frame_idx,
                        "timestamp_us": int(ts_us),
                        "camera": camera,
                    },
                )
                yielded += 1

            except Exception as e:
                print(f"  WARN: failed to process frame {frame_idx} of {clip_id}: {e}",
                      flush=True)
                continue


def iter_nvidia_video_frames(
    *,
    avdi: "physical_ai_av.PhysicalAIAVDatasetInterface",
    clip_id: str,
    camera: str = "camera_front_wide_120fov",
    max_frames: int = 200,
    fps: int = 10,
    horizon_sec_gt_wps: float = 2.75,
    num_gt_wps: int = 11,
) -> Iterable[ExternalSample]:
    """Iterate through a single clip at video rate for visualization.

    Args:
        avdi: PhysicalAI-AV dataset interface.
        clip_id: Single clip ID to process.
        camera: Camera to use.
        max_frames: Maximum frames to extract.
        fps: Target frame rate.
        horizon_sec_gt_wps: Waypoint horizon.
        num_gt_wps: Number of waypoints.

    Yields:
        ExternalSample objects for video visualization.
    """
    import physical_ai_av

    try:
        video = avdi.get_clip_feature(
            clip_id,
            getattr(avdi.features.CAMERA, camera.upper()),
        )
        egomotion = avdi.get_clip_feature(
            clip_id, avdi.features.LABELS.EGOMOTION
        )
        cam_config = get_camera_config(avdi, clip_id, camera)
        interpolator = physical_ai_av.utils.interpolation.Interpolator([egomotion])
    except Exception as e:
        print(f"  ERROR: failed to load clip {clip_id}: {e}", flush=True)
        return

    # Sample at target FPS, leaving room for waypoint horizon
    frame_spacing_us = int(1_000_000 / fps)
    max_ts_us = 17_000_000  # Leave 3s for waypoint horizon
    timestamps_us = np.arange(0, max_ts_us, frame_spacing_us)[:max_frames]

    frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)

    for frame_idx, (frame, ts_us) in enumerate(zip(frames, actual_ts)):
        try:
            gt_wps = egomotion_to_waypoints(
                interpolator, ts_us,
                horizon_s=horizon_sec_gt_wps,
                num_waypoints=num_gt_wps,
            )
            gt_route = egomotion_to_route(interpolator, ts_us)
            speed_mps = get_speed_from_egomotion(interpolator, ts_us)
            target_points = get_target_points(interpolator, ts_us)

            import tempfile
            from PIL import Image

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                Image.fromarray(frame).save(f.name, quality=95)
                rgb_path = Path(f.name)

            yield ExternalSample(
                rgb_path=rgb_path,
                speed_mps=speed_mps,
                target_points=target_points,
                intrinsics=cam_config["intrinsics"],
                fov_deg=cam_config["fov_deg"],
                cam_translation_xyz=cam_config["cam_translation_xyz"],
                crop_bottom=False,
                gt_wps=gt_wps,
                gt_route=gt_route,
                gt_commentary=None,
                meta={
                    "clip_id": clip_id,
                    "frame_idx": frame_idx,
                    "timestamp_us": int(ts_us),
                    "t_in_clip_s": float(ts_us / 1_000_000),
                },
            )

        except Exception as e:
            print(f"  WARN: frame {frame_idx}: {e}", flush=True)
            continue
