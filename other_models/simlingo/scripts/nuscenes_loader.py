"""nuScenes adapter: turn nuScenes keyframes into the same per-sample input
shape SimLingo expects from the CARLA loader.

Per keyframe we need to synthesise the four things the model conditions on
that don't come for free from the front-camera image:

  - vehicle_speed (m/s) — from consecutive ego_pose deltas
  - target_point + next_target_point — future ego positions transformed into
    the *current* ego frame
  - camera_intrinsics / extrinsics — only for the overlay visualiser
  - ground-truth waypoints + path — for ADE/FDE metrics

Coordinate convention (matches SimLingo training):
  x = forward (along the heading direction)
  y = left
Units: meters.

NB: nuScenes ego frame is +x_forward, +y_left, +z_up. That matches SimLingo's
convention exactly, so we just need to rotate world->ego by the inverse of the
ego pose's heading quaternion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers (avoids pulling pyquaternion just for one rotation).
# ---------------------------------------------------------------------------


def _quat_to_rotmat(q_wxyz: list[float]) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    n = w * w + x * x + y * y + z * z
    s = 0.0 if n < 1e-12 else 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _world_to_ego_xy(world_xy: np.ndarray, ego_pose: dict) -> np.ndarray:
    """Project a Nx2 array of world (x, y) into the ego frame at `ego_pose`."""
    R = _quat_to_rotmat(ego_pose["rotation"])  # world<-ego
    t = np.asarray(ego_pose["translation"], dtype=np.float64)
    # Build the 2D top-down ego frame: rotate by -yaw, translate by -t.
    rel = world_xy - t[:2][None, :]
    R_inv = R.T
    # The 2D part of R_inv (top-left 2x2) maps world XY into ego XY.
    R_inv_2d = R_inv[:2, :2]
    return (rel @ R_inv_2d.T).astype(np.float32)


# ---------------------------------------------------------------------------
# Sample type — the shape consumed by inference.run_external_samples
# ---------------------------------------------------------------------------


@dataclass
class ExternalSample:
    """A single frame from a non-CARLA dataset.

    The fields with `gt_` prefix are optional: if present we'll compute
    ADE/FDE; if not, we still run inference and just skip metrics.
    """

    rgb_path: Path
    speed_mps: float
    target_points: np.ndarray  # [2, 2] in ego frame [x_forward, y_left]
    intrinsics: np.ndarray | None = None  # [3, 3]; falls back to viz default
    fov_deg: float = 110.0
    cam_translation_xyz: tuple[float, float, float] = (0.0, 2.0, 1.5)
    crop_bottom: bool = True  # CARLA's bonnet hack; OFF for roof-mount cams
    gt_wps: np.ndarray | None = None
    gt_route: np.ndarray | None = None
    gt_commentary: str | None = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# nuScenes traversal
# ---------------------------------------------------------------------------


def _interp_trajectory(
    t_query: np.ndarray,
    t_known: np.ndarray,
    xy_known: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a 2D trajectory onto a new time grid.

    Args:
        t_query: shape [N] target timestamps in seconds since t0.
        t_known: shape [K] anchor timestamps in seconds since t0.
        xy_known: shape [K, 2] anchor positions in ego frame.

    Returns positions of shape [N, 2]; queries outside the known range are
    clamped to the nearest endpoint.
    """
    xs = np.interp(t_query, t_known, xy_known[:, 0])
    ys = np.interp(t_query, t_known, xy_known[:, 1])
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def _route_resample_1m(points: np.ndarray, num: int = 20) -> np.ndarray:
    """Same 1m-spaced resampling SimLingo uses for the `route` head."""
    pts = np.concatenate((np.zeros_like(points[:1]), points), axis=0)
    shift = np.roll(pts, 1, axis=0)
    shift[0] = shift[1]
    dists = np.linalg.norm(pts - shift, axis=1)
    dists = np.cumsum(dists)
    dists += np.arange(len(dists)) * 1e-4
    x = np.arange(0, num, 1)
    return np.stack(
        [np.interp(x, dists, pts[:, 0]), np.interp(x, dists, pts[:, 1])],
        axis=-1,
    ).astype(np.float32)


def _build_K_from_nuscenes(intrinsic_3x3: list[list[float]]) -> np.ndarray:
    return np.asarray(intrinsic_3x3, dtype=np.float32)


def iter_nuscenes_samples(
    *,
    dataroot: str,
    version: str = "v1.0-mini",
    sensor: str = "CAM_FRONT",
    horizon_sec_target: float = 2.0,
    horizon_sec_next_target: float = 4.0,
    horizon_sec_gt_wps: float = 2.75,
    num_gt_wps: int = 11,
    max_samples: int | None = None,
    frame_stride: int = 1,
    scene_filter: list[str] | None = None,
) -> Iterable[ExternalSample]:
    """Yield ExternalSamples from a downloaded nuScenes set.

    Args:
        dataroot: directory containing the nuScenes layout
            (`<root>/v1.0-mini/`, `<root>/samples/CAM_FRONT/...`, etc).
        sensor: which camera. Must be a roof-mounted one — bonnet crop is
            forced off here.
        horizon_sec_target / horizon_sec_next_target: how far ahead to look
            for the two target points the model gets in its prompt.
        horizon_sec_gt_wps: total horizon for GT waypoints used in ADE/FDE.
            SimLingo predicts 11 wps at 0.25s spacing == 2.75s, hence the
            default.
        num_gt_wps: how many GT waypoints to produce; matches the model's
            output length.
        max_samples: optional cap.
        frame_stride: skip every Nth keyframe within a scene to spread
            coverage when sampling few frames.
        scene_filter: optional list of scene tokens to include (otherwise all).
    """
    # Late import so nuscenes-devkit only matters when this function actually
    # runs (i.e. inside the Modal container).
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    yielded = 0

    for scene in nusc.scene:
        if scene_filter is not None and scene["token"] not in scene_filter:
            continue

        # 1) Build chronological keyframe list for this scene.
        sample_tokens: list[str] = []
        token = scene["first_sample_token"]
        while token:
            sample_tokens.append(token)
            token = nusc.get("sample", token)["next"]

        if not sample_tokens:
            continue

        # 2) Pre-compute (t_sec, ego_xyz, ego_pose) for the whole scene; we'll
        # interpolate future positions from these arrays.
        ts_us, world_xy, full_poses = [], [], []
        for tok in sample_tokens:
            sample = nusc.get("sample", tok)
            sd = nusc.get("sample_data", sample["data"][sensor])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            ts_us.append(ep["timestamp"])
            world_xy.append(ep["translation"][:2])
            full_poses.append(ep)
        ts_us = np.asarray(ts_us, dtype=np.float64)
        ts_sec = (ts_us - ts_us[0]) * 1e-6
        world_xy = np.asarray(world_xy, dtype=np.float64)

        # 3) Walk forward, build a sample whenever we have enough future
        # horizon to compute both target points.
        max_future = ts_sec[-1]
        needed_horizon = max(horizon_sec_next_target, horizon_sec_gt_wps)

        for i in range(0, len(sample_tokens), frame_stride):
            t_now = ts_sec[i]
            if t_now + needed_horizon > max_future:
                # Not enough future data left in this scene.
                break

            sample = nusc.get("sample", sample_tokens[i])
            sd = nusc.get("sample_data", sample["data"][sensor])
            ep_now = full_poses[i]
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

            # --- Future poses, in *current* ego frame -------------------
            future_xy_ego = _world_to_ego_xy(world_xy[i:], ep_now)
            future_t = ts_sec[i:] - t_now  # 0, 0.5, 1.0, ... seconds

            # --- Target points (the prompt inputs) ----------------------
            tp1 = _interp_trajectory(
                np.array([horizon_sec_target]), future_t, future_xy_ego
            )[0]
            tp2 = _interp_trajectory(
                np.array([horizon_sec_next_target]), future_t, future_xy_ego
            )[0]
            target_points = np.stack([tp1, tp2], axis=0)

            # --- GT waypoints at 0.25s spacing --------------------------
            wp_grid = np.linspace(
                horizon_sec_gt_wps / num_gt_wps,
                horizon_sec_gt_wps,
                num_gt_wps,
            )
            gt_wps = _interp_trajectory(wp_grid, future_t, future_xy_ego)

            # --- GT path (1m-spaced route) ------------------------------
            # Use everything from t=0 to t=horizon_sec_gt_wps as the route.
            mask = future_t <= horizon_sec_gt_wps
            route_anchor = future_xy_ego[mask]
            if len(route_anchor) >= 2:
                gt_route = _route_resample_1m(route_anchor)
            else:
                gt_route = None

            # --- Speed: ||pos(0.5s) - pos(0)|| / 0.5 ---------------------
            speed_mps = (
                float(np.linalg.norm(future_xy_ego[1] - future_xy_ego[0]))
                / max(future_t[1] - future_t[0], 1e-3)
                if len(future_xy_ego) >= 2
                else 0.0
            )

            # --- Camera intrinsics + extrinsics from calibrated_sensor ---
            K = _build_K_from_nuscenes(cs["camera_intrinsic"])
            # nuScenes camera extrinsic: translation relative to ego (in ego
            # frame). Camera is roof-mounted, ~1.7m up, ~1.7m forward.
            cam_t = np.asarray(cs["translation"], dtype=np.float32)  # [x_fwd, y_left, z_up]
            # Project to the OpenCV-frame convention used by `viz._project_points_to_image`:
            #   (X_right=cam_y_left * -1 ≈ 0, Y_down=cam_z_up * -1, Z_forward=cam_x_fwd)
            # The viz uses only the Z (forward) translation meaningfully and
            # ignores small Y_left offsets. Pass [0, cam_z_up, cam_x_fwd]:
            cam_translation_xyz = (
                0.0,
                float(cam_t[2]),
                float(cam_t[0]),
            )

            rgb_path = Path(dataroot) / sd["filename"]
            if not rgb_path.exists():
                continue

            yield ExternalSample(
                rgb_path=rgb_path,
                speed_mps=speed_mps,
                target_points=target_points.astype(np.float32),
                intrinsics=K,
                cam_translation_xyz=cam_translation_xyz,
                crop_bottom=False,  # nuScenes camera is roof-mounted
                gt_wps=gt_wps,
                gt_route=gt_route,
                gt_commentary=None,
                meta={
                    "scene": scene["name"],
                    "sample_token": sample["token"],
                    "timestamp_us": int(ts_us[i]),
                    "sensor": sensor,
                    "image_hw": (sd["height"], sd["width"]),
                },
            )
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                return


def iter_nuscenes_video_frames(
    *,
    dataroot: str,
    version: str = "v1.0-mini",
    sensor: str = "CAM_FRONT",
    scene_names: list[str] | None = None,
    max_frames: int | None = None,
    horizon_sec_target: float = 2.0,
    horizon_sec_next_target: float = 4.0,
    horizon_sec_gt_wps: float = 2.75,
    num_gt_wps: int = 11,
) -> Iterable[ExternalSample]:
    """Like `iter_nuscenes_samples` but walks the *full* ~12 Hz CAM_FRONT
    sample_data chain (not just 2 Hz keyframes), so the resulting overlays can
    be stitched into a smooth video.

    Args:
        scene_names: scenes to include in the order given. Default = the first
            scene by `scene` table order.
        max_frames: hard cap on total frames across all chosen scenes.

    Target points + speed + GT waypoints are interpolated from the sequence of
    sample_data ego_poses (every sample_data has its own ego_pose, sampled at
    sensor rate, so we get richer signal than from keyframes alone).
    """
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

    # Build {scene_name -> token} once so callers can pass human-readable names.
    by_name = {s["name"]: s["token"] for s in nusc.scene}
    if scene_names is None:
        scene_names = [nusc.scene[0]["name"]]
    if max_frames is None:
        max_frames = 10_000

    yielded = 0
    for sname in scene_names:
        if sname not in by_name:
            print(f"  WARN: scene {sname!r} not in dataset, skipping", flush=True)
            continue
        scene = nusc.get("scene", by_name[sname])

        # 1) Find the first sample_data for this sensor inside the scene.
        first_sample = nusc.get("sample", scene["first_sample_token"])
        sd_token = first_sample["data"][sensor]
        # nuScenes sample_data is a doubly-linked list; rewind to the very first
        # entry so we capture pre-keyframe frames too.
        while True:
            sd = nusc.get("sample_data", sd_token)
            if sd["prev"] == "":
                break
            sd_token = sd["prev"]

        # 2) Walk the chain, collecting (ts_us, world_xy, ego_pose, sd) tuples.
        records: list[tuple[int, np.ndarray, dict, dict]] = []
        while sd_token != "":
            sd = nusc.get("sample_data", sd_token)
            # Stop once we leave the scene (sample_data links can cross scene
            # boundaries through sweeps but the `sample_token` lookup respects
            # scene grouping).
            try:
                sample = nusc.get("sample", sd["sample_token"])
            except KeyError:
                break
            if sample["scene_token"] != scene["token"]:
                break
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            records.append(
                (
                    ep["timestamp"],
                    np.asarray(ep["translation"][:2], dtype=np.float64),
                    ep,
                    sd,
                )
            )
            sd_token = sd["next"]

        if not records:
            continue

        # 3) Vectorize timestamps + positions for fast interpolation.
        ts_us = np.asarray([r[0] for r in records], dtype=np.float64)
        ts_sec = (ts_us - ts_us[0]) * 1e-6
        world_xy = np.stack([r[1] for r in records], axis=0)

        # Use the calibrated sensor of the first sample_data for this sensor.
        cs_token = records[0][3]["calibrated_sensor_token"]
        cs = nusc.get("calibrated_sensor", cs_token)
        K = _build_K_from_nuscenes(cs["camera_intrinsic"])
        cam_t = np.asarray(cs["translation"], dtype=np.float32)
        cam_translation_xyz = (
            0.0,
            float(cam_t[2]),
            float(cam_t[0]),
        )

        max_future = ts_sec[-1]
        needed_horizon = max(horizon_sec_next_target, horizon_sec_gt_wps)

        for i, (_, _, ep_now, sd) in enumerate(records):
            t_now = ts_sec[i]
            if t_now + needed_horizon > max_future:
                break
            future_xy_ego = _world_to_ego_xy(world_xy[i:], ep_now)
            future_t = ts_sec[i:] - t_now

            tp1 = _interp_trajectory(
                np.array([horizon_sec_target]), future_t, future_xy_ego
            )[0]
            tp2 = _interp_trajectory(
                np.array([horizon_sec_next_target]), future_t, future_xy_ego
            )[0]
            target_points = np.stack([tp1, tp2], axis=0)

            wp_grid = np.linspace(
                horizon_sec_gt_wps / num_gt_wps,
                horizon_sec_gt_wps,
                num_gt_wps,
            )
            gt_wps = _interp_trajectory(wp_grid, future_t, future_xy_ego)

            mask = future_t <= horizon_sec_gt_wps
            route_anchor = future_xy_ego[mask]
            gt_route = (
                _route_resample_1m(route_anchor) if len(route_anchor) >= 2 else None
            )

            # Use a short forward window for the speed estimate so it tracks the
            # ~12 Hz signal rather than the 2 Hz keyframe spacing.
            speed_window_s = 0.5
            window_mask = future_t <= speed_window_s
            if window_mask.sum() >= 2:
                end_idx = int(window_mask.sum()) - 1
                dt = max(future_t[end_idx] - future_t[0], 1e-3)
                speed_mps = float(np.linalg.norm(future_xy_ego[end_idx] - future_xy_ego[0])) / dt
            else:
                speed_mps = 0.0

            rgb_path = Path(dataroot) / sd["filename"]
            if not rgb_path.exists():
                continue

            yield ExternalSample(
                rgb_path=rgb_path,
                speed_mps=speed_mps,
                target_points=target_points.astype(np.float32),
                intrinsics=K,
                cam_translation_xyz=cam_translation_xyz,
                crop_bottom=False,
                gt_wps=gt_wps,
                gt_route=gt_route,
                gt_commentary=None,
                meta={
                    "scene": scene["name"],
                    "sample_data_token": sd["token"],
                    "is_keyframe": sd["is_key_frame"],
                    "timestamp_us": int(ts_us[i]),
                    "sensor": sensor,
                    "t_in_scene_s": float(t_now),
                },
            )
            yielded += 1
            if yielded >= max_frames:
                return
