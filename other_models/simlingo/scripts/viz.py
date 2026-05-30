"""Side-by-side overlays of GT vs predicted waypoints on the front-cam image.

By default we use the CARLA-trained camera params (110° FOV, mount ~2m up and
1.5m back). For nuScenes (and the cart later) we accept caller-supplied
intrinsics + extrinsics so the projection matches the real camera that took
the picture.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _project_points_to_image(
    points: np.ndarray,
    K: np.ndarray,
    cam_translation_xyz: tuple[float, float, float] = (0.0, 2.0, 1.5),
    cam_rotation_rvec: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Project (x_forward, y_right) ego-frame points to image pixels.

    Default (`cam_translation_xyz=(0, 2, 1.5)`, no rotation) matches the CARLA
    training-time projection used in `team_code/simlingo_utils.py`:
        world->cam is just a fixed translation `[0, 2, 1.5]`.
    Callers (e.g. nuScenes loader) can pass their own camera mount in the same
    OpenCV (X-right, Y-down, Z-forward) frame.
    """
    tvec = np.asarray([cam_translation_xyz], dtype=np.float32)
    rvec = (
        cam_rotation_rvec.astype(np.float32)
        if cam_rotation_rvec is not None
        else np.zeros((3, 1), dtype=np.float32)
    )
    dist = np.zeros((5, 1), dtype=np.float32)
    pixels: list[tuple[int, int]] = []
    for x_fwd, y_right in points:
        # Convert ego (X_forward, Y_right, Z_up) into OpenCV camera-axes input:
        # X_cam_right = y_right, Y_cam_down = 0, Z_cam_forward = x_fwd + cam_z
        pos_3d = np.array([y_right, 0.0, x_fwd + tvec[0][2]], dtype=np.float32)
        p2d, _ = cv2.projectPoints(pos_3d, rvec=rvec, tvec=tvec, cameraMatrix=K, distCoeffs=dist)
        pixels.append((int(p2d[0][0][0]), int(p2d[0][0][1])))
    return pixels


def _camera_intrinsics(w: int, h: int, fov_deg: float = 110.0) -> np.ndarray:
    focal = w / (2.0 * np.tan(fov_deg * np.pi / 360.0))
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    pixels: list[tuple[int, int]],
    color: tuple[int, int, int],
    radius: int = 3,
) -> None:
    for x, y in pixels:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _safe_font(size: int = 14) -> ImageFont.ImageFont:
    # The base image we'll Pillow-draw on inside Modal won't have arial; fall
    # back to the default bitmap font which is fine for a sanity check.
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def write_overlay(
    *,
    rgb_path: Path,
    pred_wps: np.ndarray | None,
    pred_route: np.ndarray | None,
    gt_wps: np.ndarray | None,
    gt_route: np.ndarray | None,
    pred_text: str,
    gt_text: str | None,
    out_path: Path,
    crop_bottom: bool = True,
    intrinsics: np.ndarray | None = None,
    fov_deg: float = 110.0,
    cam_translation_xyz: tuple[float, float, float] = (0.0, 2.0, 1.5),
) -> None:
    """Render a single overlay PNG.

    Args:
        crop_bottom: drop ~30% bottom of image (used for CARLA's bonnet view).
            Set False for roof-mounted cameras (nuScenes, our cart).
        intrinsics: optional 3x3 K matrix. If None, a pinhole K is synthesised
            from `fov_deg`.
        cam_translation_xyz: OpenCV-frame translation of the camera relative to
            the rear axle / ego origin (X_right, Y_down, Z_forward). The CARLA
            default (0, 2, 1.5) means the camera is 2m above and 1.5m forward
            of the ego origin.
    """
    bgr = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if crop_bottom:
        h = rgb.shape[0]
        crop_h = int(h - (h * 4.8) // 16)
        rgb = rgb[:crop_h, :, :]

    H, W, _ = rgb.shape
    if intrinsics is None:
        K = _camera_intrinsics(W, H, fov_deg)
    else:
        K = intrinsics.astype(np.float32)

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    def _proj(pts):
        return _project_points_to_image(pts, K, cam_translation_xyz=cam_translation_xyz)

    if gt_route is not None:
        _draw_polyline(draw, _proj(gt_route), (0, 180, 0), radius=2)
    if gt_wps is not None:
        _draw_polyline(draw, _proj(gt_wps), (0, 255, 0), radius=3)
    if pred_route is not None:
        _draw_polyline(draw, _proj(pred_route), (180, 0, 0), radius=2)
    if pred_wps is not None:
        _draw_polyline(draw, _proj(pred_wps), (255, 64, 64), radius=3)

    # Caption: predicted commentary on top of a black strip for readability.
    caption_height = 120
    canvas = Image.new("RGB", (W, H + caption_height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    caption_draw = ImageDraw.Draw(canvas)
    font = _safe_font(14)
    y = H + 6
    for label, text, color in [
        ("PRED", pred_text or "<no language>", (255, 200, 200)),
        ("GT  ", gt_text or "<no commentary annotation>", (200, 255, 200)),
    ]:
        wrapped = textwrap.wrap(f"{label}: {text}", width=max(1, W // 8))
        for line in wrapped[:3]:
            caption_draw.text((6, y), line, fill=color, font=font)
            y += 16
        y += 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
