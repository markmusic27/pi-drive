"""Training-time waypoint visualization.

Renders a side-by-side panel with:
  - Left: camera image with GT (green) and predicted (blue) waypoints projected
  - Right: BEV (top-down) plot with GT and predicted waypoints in ego frame

Logged to W&B every val pass to qualitatively monitor model behavior.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .nvidia_viz import project_waypoints_simple, _safe_font
except ImportError:
    from nvidia_viz import project_waypoints_simple, _safe_font


# ---------------------------------------------------------------------------
# Camera projection panel
# ---------------------------------------------------------------------------


def _draw_waypoint_track(
    draw: ImageDraw.ImageDraw,
    pixels: list[tuple[int, int]],
    color: tuple[int, int, int],
    radius: int = 5,
    line_width: int = 2,
) -> None:
    """Draw a sequence of waypoints as a connected track on an image."""
    for j, (u, v) in enumerate(pixels):
        if j > 0:
            prev_u, prev_v = pixels[j - 1]
            draw.line([(prev_u, prev_v), (u, v)], fill=color, width=line_width)
        draw.ellipse(
            (u - radius, v - radius, u + radius, v + radius),
            fill=color,
            outline=(255, 255, 255),
            width=1,
        )


def render_camera_panel(
    image: np.ndarray,           # [H, W, 3] uint8 RGB
    gt_wps: np.ndarray,          # [N, 2] ego (x_fwd, y_left), meters
    pred_wps: np.ndarray,        # [M, 2] ego (x_fwd, y_left), meters
    K: np.ndarray,               # [3, 3] camera intrinsics
    cam_translation_xyz: tuple[float, float, float] = (0.0, 1.5, 2.0),
    speed_mps: float = 0.0,
    meta_action: str | None = None,
) -> Image.Image:
    """Render camera image with GT (green) and predicted (blue) waypoints.

    Returns a PIL Image with both tracks projected, plus a header overlay.
    """
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    gt_pixels = project_waypoints_simple(gt_wps, K, cam_translation_xyz, (w, h))
    pred_pixels = project_waypoints_simple(pred_wps, K, cam_translation_xyz, (w, h))

    _draw_waypoint_track(draw, gt_pixels, color=(40, 230, 80), radius=5)        # green = GT
    _draw_waypoint_track(draw, pred_pixels, color=(70, 140, 255), radius=5)     # blue = pred

    # Header overlay
    overlay = Image.new("RGBA", (w, 56), (0, 0, 0, 180))
    pil_img.paste(overlay, (0, 0), overlay)
    font = _safe_font(15)
    font_big = _safe_font(18)
    draw.text((10, 6), f"Speed: {speed_mps:.1f} m/s", fill=(255, 255, 255), font=font_big)
    if meta_action:
        draw.text((10, 30), f"Action: {meta_action}", fill=(220, 220, 220), font=font)
    # Legend
    legend_x = w - 200
    draw.text((legend_x, 6), "● GT", fill=(40, 230, 80), font=font_big)
    draw.text((legend_x, 30), "● Pred", fill=(70, 140, 255), font=font)

    return pil_img


# ---------------------------------------------------------------------------
# BEV (top-down) panel
# ---------------------------------------------------------------------------


def render_bev_panel(
    gt_wps: np.ndarray,          # [N, 2] ego (x_fwd, y_left), meters
    pred_wps: np.ndarray,        # [M, 2] ego (x_fwd, y_left), meters
    size_px: int = 512,
    meters_forward: float = 30.0,
    meters_lateral: float = 15.0,
) -> Image.Image:
    """Render a top-down BEV view of GT vs predicted waypoints.

    Ego is at the bottom-center, +x forward (up on image), +y left (left on image).
    """
    W = H = size_px
    img = Image.new("RGB", (W, H), (18, 18, 22))
    draw = ImageDraw.Draw(img)

    # Ego origin position on canvas (bottom-center)
    ego_u, ego_v = W // 2, int(H * 0.92)

    # Scale: meters → pixels
    fwd_scale = (H - 40) / meters_forward                   # x_fwd → -v (up)
    lat_scale = (W - 40) / (2 * meters_lateral)             # y_left → -u (left)

    def ego_to_pix(x_fwd: float, y_left: float) -> tuple[int, int]:
        u = ego_u - int(y_left * lat_scale)
        v = ego_v - int(x_fwd * fwd_scale)
        return u, v

    # Grid (every 5m forward, every 5m lateral)
    grid_color = (50, 50, 60)
    for m in range(5, int(meters_forward) + 1, 5):
        u_l = ego_u - int(meters_lateral * lat_scale)
        u_r = ego_u + int(meters_lateral * lat_scale)
        v = ego_v - int(m * fwd_scale)
        draw.line([(u_l, v), (u_r, v)], fill=grid_color, width=1)
    for m in range(-int(meters_lateral), int(meters_lateral) + 1, 5):
        u = ego_u - int(m * lat_scale)
        v_t = ego_v - int(meters_forward * fwd_scale)
        v_b = ego_v
        draw.line([(u, v_t), (u, v_b)], fill=grid_color, width=1)

    # Ego marker (white triangle pointing up)
    draw.polygon(
        [(ego_u, ego_v - 12), (ego_u - 8, ego_v + 6), (ego_u + 8, ego_v + 6)],
        fill=(240, 240, 240),
    )

    # GT track (green)
    gt_pix = [ego_to_pix(float(x), float(y)) for x, y in gt_wps]
    _draw_waypoint_track(draw, gt_pix, color=(40, 230, 80), radius=4, line_width=2)

    # Pred track (blue)
    pred_pix = [ego_to_pix(float(x), float(y)) for x, y in pred_wps]
    _draw_waypoint_track(draw, pred_pix, color=(70, 140, 255), radius=4, line_width=2)

    # Header
    font = _safe_font(14)
    draw.text((8, 6), f"BEV  |  fwd {meters_forward:.0f}m × lat ±{meters_lateral:.0f}m",
              fill=(220, 220, 220), font=font)
    draw.text((8, H - 22), "● GT", fill=(40, 230, 80), font=font)
    draw.text((60, H - 22), "● Pred", fill=(70, 140, 255), font=font)

    return img


# ---------------------------------------------------------------------------
# Combined panel
# ---------------------------------------------------------------------------


def render_prediction_panel(
    image: np.ndarray,
    gt_wps: np.ndarray,
    pred_wps: np.ndarray,
    K: np.ndarray,
    cam_translation_xyz: tuple[float, float, float] = (0.0, 1.5, 2.0),
    speed_mps: float = 0.0,
    meta_action: str | None = None,
    bev_size: int = 512,
) -> Image.Image:
    """Render a combined camera+BEV panel side-by-side.

    Returns a single PIL Image, width = camera_w + bev_size, height = max(camera_h, bev_size).
    """
    cam_panel = render_camera_panel(
        image, gt_wps, pred_wps, K, cam_translation_xyz, speed_mps, meta_action
    )
    bev_panel = render_bev_panel(gt_wps, pred_wps, size_px=bev_size)

    cam_w, cam_h = cam_panel.size
    bev_w, bev_h = bev_panel.size

    total_w = cam_w + bev_w
    total_h = max(cam_h, bev_h)

    combined = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    combined.paste(cam_panel, (0, (total_h - cam_h) // 2))
    combined.paste(bev_panel, (cam_w, (total_h - bev_h) // 2))

    return combined


def panel_to_wandb_image(panel: Image.Image, caption: str = ""):
    """Convert a PIL panel to a wandb.Image (returns None if wandb not available)."""
    try:
        import wandb
        buf = io.BytesIO()
        panel.save(buf, format="PNG")
        buf.seek(0)
        return wandb.Image(Image.open(buf), caption=caption)
    except ImportError:
        return None
