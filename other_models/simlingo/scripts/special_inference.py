"""Inference + camera/BEV viz for raw mp4 + egomotion (the 'special' folder).

Loads SimLingo with an arbitrary LoRA checkpoint and runs per-frame inference
on a video with paired ego.jsonl, rendering a side-by-side camera+BEV panel
per frame (GT from egomotion in green, model prediction in blue) and stitching
the result into an mp4.

Runs inside the Modal training image (see modal_training.py); not intended for
local use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_ego_data(ego_path: Path) -> list[dict]:
    """Load ego data from JSONL, skipping the schema header row."""
    rows = []
    with open(ego_path) as fh:
        for line in fh:
            entry = json.loads(line)
            if "rel_t" in entry:
                rows.append(entry)
    return rows


def _slerp_R(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    """Cheap rotation interpolation: lerp + re-orthogonalize via SVD."""
    R = (1 - alpha) * R0 + alpha * R1
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def interpolate_ego(ego_data: list[dict], target_t: float) -> dict | None:
    """Return interpolated ego state (xyz_m, rot3x3, speed_mps) at target_t."""
    if not ego_data:
        return None

    prev_entry = None
    next_entry = None
    for entry in ego_data:
        if entry["rel_t"] <= target_t:
            prev_entry = entry
        else:
            next_entry = entry
            break

    if prev_entry is None:
        return ego_data[0]
    if next_entry is None:
        return prev_entry

    t0 = prev_entry["rel_t"]
    t1 = next_entry["rel_t"]
    if t1 == t0:
        return prev_entry
    alpha = (target_t - t0) / (t1 - t0)

    a0 = prev_entry["alpamayo"]
    a1 = next_entry["alpamayo"]
    xyz = np.asarray(a0["xyz_m"]) + alpha * (np.asarray(a1["xyz_m"]) - np.asarray(a0["xyz_m"]))
    R = _slerp_R(np.asarray(a0["rot3x3"]), np.asarray(a1["rot3x3"]), alpha)
    speed = a0["speed_mps"] + alpha * (a1["speed_mps"] - a0["speed_mps"])
    return {
        "rel_t": target_t,
        "alpamayo": {"xyz_m": xyz.tolist(), "rot3x3": R.tolist(), "speed_mps": float(speed)},
    }


def compute_future_points(
    ego_data: list[dict],
    current_t: float,
    offsets_s: list[float],
) -> np.ndarray:
    """Compute future ego positions in the current ego frame.

    Returns an [N, 2] array of (x_forward, y_left) in current ego frame, one
    entry per offset. Used for both ground-truth waypoints and target points.
    """
    current = interpolate_ego(ego_data, current_t)
    if current is None:
        return np.zeros((len(offsets_s), 2), dtype=np.float32)

    current_xyz = np.asarray(current["alpamayo"]["xyz_m"])
    current_R = np.asarray(current["alpamayo"]["rot3x3"])

    pts = []
    for dt in offsets_s:
        future = interpolate_ego(ego_data, current_t + dt)
        if future is None:
            # Hold last known point if we've run out of future
            pts.append(pts[-1] if pts else [0.0, 0.0])
            continue
        diff = np.asarray(future["alpamayo"]["xyz_m"]) - current_xyz
        rel = current_R.T @ diff  # rotate into current ego frame
        pts.append([float(rel[0]), float(rel[1])])
    return np.asarray(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# LoRA application (mirrors nvidia_trainer._setup_lora, sans optimizer/DDP)
# ---------------------------------------------------------------------------


def apply_lora_and_load_weights(
    model,
    lora_path: Path,
    target_modules: list[str],
    rank: int,
    alpha: int,
    device,
) -> int:
    """Inject LoRA adapters into matching nn.Linear modules and load weights.

    Matches the training-time injection pattern in
    `nvidia_trainer.NvidiaTrainer._setup_lora`: it creates `lora_A_<safe>` and
    `lora_B_<safe>` attributes on the model and registers forward hooks that
    add `scaling * B(A(x))` to the original Linear output. We then load the
    saved LoRA tensors back into those attributes.
    """
    import torch
    import torch.nn as nn

    scaling = alpha / rank
    lora_pairs: list[tuple[str, nn.Module, nn.Linear, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue
        in_features = module.in_features
        out_features = module.out_features
        lora_A = nn.Linear(in_features, rank, bias=False).to(device).to(torch.bfloat16)
        lora_B = nn.Linear(rank, out_features, bias=False).to(device).to(torch.bfloat16)
        # init doesn't matter — we're about to overwrite from the checkpoint
        lora_pairs.append((name, module, lora_A, lora_B))

    if not lora_pairs:
        raise RuntimeError(
            f"no LoRA target modules matched in model (patterns={target_modules})"
        )

    for name, original_module, lora_A, lora_B in lora_pairs:
        safe_name = name.replace(".", "_")
        setattr(model, f"lora_A_{safe_name}", lora_A)
        setattr(model, f"lora_B_{safe_name}", lora_B)

        def make_hook(_A, _B, _s):
            def hook(_m, inputs, output):
                x = inputs[0]
                if isinstance(x, tuple):
                    x = x[0]
                lora_out = _B(_A(x.to(_A.weight.dtype)))
                return output + _s * lora_out.to(output.dtype)
            return hook

        original_module.register_forward_hook(make_hook(lora_A, lora_B, scaling))

    # Load LoRA tensors
    lora_state = torch.load(lora_path, map_location=device)
    current_state = model.state_dict()
    loaded = 0
    for key, value in lora_state.items():
        if key in current_state:
            current_state[key] = value
            loaded += 1
    model.load_state_dict(current_state)
    print(
        f"[lora] applied to {len(lora_pairs)} modules, loaded {loaded}/{len(lora_state)} tensors from {lora_path}",
        flush=True,
    )
    return loaded


# ---------------------------------------------------------------------------
# Per-frame inference + render loop
# ---------------------------------------------------------------------------


def run_special_inference(
    *,
    video_path: Path,
    ego_path: Path,
    ckpt_path: Path,
    hydra_cfg_path: Path,
    lora_path: Path | None,
    lora_cfg: dict[str, Any] | None,
    out_dir: Path,
    fov_deg: float = 70.0,
    cam_height_m: float = 1.2,
    cam_forward_offset_m: float = 0.5,
    crop_bottom_frac: float = 0.0,
    frame_stride: int = 1,
    max_frames: int | None = None,
    output_fps: int = 30,
    bev_size: int = 480,
) -> dict[str, Any]:
    """Run SimLingo on a raw mp4 + ego.jsonl pair, render camera+BEV per frame, stitch mp4.

    Parameters mirror what `_predict_one` and `render_prediction_panel` expect.
    """
    import sys
    import time

    import cv2
    import torch
    from PIL import Image

    sys.path.insert(0, "/opt/simlingo")  # for simlingo_training imports

    from scripts.inference import _predict_one, _setup_model_and_tokens
    from scripts.viz_training import render_prediction_panel

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    tmp_rgb_path = out_dir / "_tmp_frame.jpg"

    print(">> loading ego data", flush=True)
    ego_data = load_ego_data(ego_path)
    print(f"   {len(ego_data)} ego samples, t in [{ego_data[0]['rel_t']:.2f}, {ego_data[-1]['rel_t']:.2f}]", flush=True)

    print(">> opening video", flush=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   {W}x{H} @ {fps:.1f}fps, {total_frames} frames", flush=True)

    fx = W / (2 * np.tan(np.radians(fov_deg / 2)))
    K = np.asarray([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=np.float32)
    cam_translation_xyz = (0.0, cam_height_m, cam_forward_offset_m)

    print(">> loading model", flush=True)
    t0 = time.time()
    model, tokenizer, cfg, num_image_tokens_total, use_global_img, device = (
        _setup_model_and_tokens(str(ckpt_path), str(hydra_cfg_path))
    )
    print(f"   base model loaded in {time.time() - t0:.1f}s", flush=True)

    if lora_path is not None:
        if lora_cfg is None:
            raise ValueError("lora_path given but lora_cfg is None")
        print(f">> applying LoRA from {lora_path}", flush=True)
        apply_lora_and_load_weights(
            model,
            lora_path=Path(lora_path),
            target_modules=lora_cfg["target_modules"],
            rank=int(lora_cfg["rank"]),
            alpha=int(lora_cfg["alpha"]),
            device=device,
        )
        model.eval()

    print(">> running inference", flush=True)
    panel_paths: list[Path] = []
    n_predicted = 0
    n_skipped = 0
    t_pred_total = 0.0
    t_loop_start = time.time()
    frame_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if max_frames is not None and n_predicted >= max_frames:
            break

        video_t = frame_idx / fps
        ego_state = interpolate_ego(ego_data, video_t)
        if ego_state is None:
            n_skipped += 1
            frame_idx += 1
            continue

        speed = float(ego_state["alpamayo"]["speed_mps"])

        # GT waypoints: 11 points at 0.25s spacing (matches training format)
        gt_wps = compute_future_points(ego_data, video_t, [0.25 * (i + 1) for i in range(11)])

        # Target points: future ego positions at 2s and 4s (proxy for nav goals)
        target_points = compute_future_points(ego_data, video_t, [2.0, 4.0])

        # _predict_one expects a file path
        cv2.imwrite(str(tmp_rgb_path), bgr)

        t1 = time.time()
        try:
            pred_wps, pred_route, pred_text, _ = _predict_one(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
                use_global_img=use_global_img,
                num_image_tokens_total=num_image_tokens_total,
                rgb_path=tmp_rgb_path,
                speed_mps=speed,
                target_points=target_points,
                use_cot=False,
            )
        except Exception as exc:
            print(f"   [frame {frame_idx}] predict error: {exc!r}", flush=True)
            n_skipped += 1
            frame_idx += 1
            continue
        t_pred_total += time.time() - t1

        if pred_wps is None:
            n_skipped += 1
            frame_idx += 1
            continue

        # Render camera+BEV panel
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        panel = render_prediction_panel(
            image=rgb,
            gt_wps=gt_wps,
            pred_wps=pred_wps,
            K=K,
            cam_translation_xyz=cam_translation_xyz,
            speed_mps=speed,
            meta_action=(pred_text[:40] + "…") if pred_text and len(pred_text) > 40 else pred_text,
            bev_size=bev_size,
        )

        panel_path = frames_dir / f"frame_{n_predicted:05d}.jpg"
        panel.save(panel_path, quality=88)
        panel_paths.append(panel_path)
        n_predicted += 1

        if n_predicted % 25 == 0:
            avg_ms = (t_pred_total / max(1, n_predicted)) * 1000
            wall = time.time() - t_loop_start
            print(
                f"   [{n_predicted}/{total_frames // frame_stride}] frame_idx={frame_idx} t={video_t:.1f}s speed={speed:.1f} "
                f"pred_avg={avg_ms:.0f}ms wall={wall:.0f}s",
                flush=True,
            )

        frame_idx += 1

    cap.release()
    if tmp_rgb_path.exists():
        tmp_rgb_path.unlink()

    print(
        f">> done predicting: {n_predicted} panels, {n_skipped} skipped, "
        f"avg {1000 * t_pred_total / max(1, n_predicted):.0f}ms/frame inference",
        flush=True,
    )

    # Stitch with ffmpeg for quality + size
    import subprocess

    out_video = out_dir / "predictions.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(output_fps),
        "-i", str(frames_dir / "frame_%05d.jpg"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_video),
    ]
    print(f">> stitching: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:\n" + result.stderr[-2000:], flush=True)
        raise RuntimeError("ffmpeg failed")
    print(f"   wrote {out_video} ({out_video.stat().st_size // (1024 * 1024)} MiB)", flush=True)

    summary = {
        "out_video": str(out_video),
        "frames_dir": str(frames_dir),
        "num_panels": n_predicted,
        "num_skipped": n_skipped,
        "avg_pred_ms": 1000 * t_pred_total / max(1, n_predicted),
        "wall_s": time.time() - t_loop_start,
        "lora_path": str(lora_path) if lora_path else None,
        "video": str(video_path),
        "fov_deg": fov_deg,
        "cam_height_m": cam_height_m,
        "frame_stride": frame_stride,
        "output_fps": output_fps,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
