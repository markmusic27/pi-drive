"""Offline SimLingo inference + metrics.

This runs *inside* the Modal container (see modal_app.py). The job:

1. Load the released SimLingo checkpoint using the upstream Hydra config.
2. Walk a few CARLA validation routes (extracted from the HF dataset tarballs).
3. For each sampled frame, reconstruct the same `DrivingInput` the live CARLA
   agent would assemble (image patches, ego-frame target points, prompt with
   `<TARGET_POINT>` placeholders) and run the model.
4. Compute waypoint ADE/FDE against ground-truth waypoints derived from each
   frame's `ego_matrix`, plus a couple of cheap commentary checks.

Most of the heavy lifting (chat template assembly, image patching, placeholder
substitution) is delegated to the upstream `simlingo_training.utils.*` modules
so behaviour stays in lock-step with the trained model.
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path / env setup helpers
# ---------------------------------------------------------------------------


def _ensure_internvl_pretrained_symlink(cache_root: str = "/cache/hf/snapshots") -> Path:
    """Upstream `get_custom_chat_template` expects to find
    `pretrained/InternVL2-1B/conversation.py` relative to cwd.

    We pre-snapshot the model into the HF cache during `prepare_assets`, so all
    we need to do here is point a `pretrained/` dir at it.
    """
    workdir = Path("/tmp/simlingo_work")
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)

    pretrained = workdir / "pretrained"
    pretrained.mkdir(exist_ok=True)
    link = pretrained / "InternVL2-1B"
    src = Path(cache_root) / "InternVL2-1B"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(src)
    return workdir


# ---------------------------------------------------------------------------
# Dataset walking (raw, without the upstream `BaseDataset` bucket machinery)
# ---------------------------------------------------------------------------


@dataclass
class FrameSample:
    route_dir: Path  # ".../Town12/route_xx"
    frame_idx: int  # 0-indexed; matches the filename digits (e.g. 0042 -> 42)
    rgb_path: Path
    measurements_dir: Path
    commentary_path: Path | None  # may not exist


def _find_route_dirs(data_root: str) -> list[Path]:
    # Routes are nested as ".../Town<N>/<route_id>/". Look one level above an
    # `rgb/` subdir so we don't depend on the parent prefix structure.
    candidates = glob.glob(os.path.join(data_root, "**", "rgb"), recursive=True)
    return sorted({Path(c).parent for c in candidates})


def _collect_video_samples(
    data_root: str,
    num_frames: int,
    pred_len: int = 11,
    seed: int = 0,
) -> list[FrameSample]:
    """Pick `num_frames` consecutive usable frames from a single route.

    We need contiguous frames so the resulting video plays back as real-time
    driving footage. Returns an empty list if no route is long enough.
    """
    routes = _find_route_dirs(data_root)
    print(f"  found {len(routes)} route dirs under {data_root}", flush=True)
    rng = random.Random(seed)
    rng.shuffle(routes)

    for route_dir in routes:
        rgb_dir = route_dir / "rgb"
        meas_dir = route_dir / "measurements"
        if not (rgb_dir.exists() and meas_dir.exists()):
            continue
        frame_files = sorted(rgb_dir.glob("[0-9][0-9][0-9][0-9].jpg"))
        usable = [
            int(p.stem)
            for p in frame_files
            if (meas_dir / f"{int(p.stem) + pred_len:04d}.json.gz").exists()
        ]
        if len(usable) < num_frames:
            continue
        # Find the first run of `num_frames` strictly-consecutive indices.
        run_start = 0
        for k in range(1, len(usable)):
            if usable[k] != usable[k - 1] + 1:
                run_start = k
            if k - run_start + 1 >= num_frames:
                chosen = usable[run_start : run_start + num_frames]
                print(
                    f"  using route {route_dir.name} frames "
                    f"{chosen[0]}..{chosen[-1]} ({num_frames} consecutive)",
                    flush=True,
                )
                return [
                    FrameSample(
                        route_dir=route_dir,
                        frame_idx=idx,
                        rgb_path=rgb_dir / f"{idx:04d}.jpg",
                        measurements_dir=meas_dir,
                        commentary_path=_commentary_path_for(route_dir, idx),
                    )
                    for idx in chosen
                ]
    return []


def _collect_samples(
    data_root: str,
    num_samples: int,
    frame_stride: int,
    pred_len: int = 11,
    seed: int = 0,
) -> list[FrameSample]:
    routes = _find_route_dirs(data_root)
    print(f"  found {len(routes)} route dirs under {data_root}", flush=True)
    rng = random.Random(seed)
    rng.shuffle(routes)

    samples: list[FrameSample] = []
    for route_dir in routes:
        rgb_dir = route_dir / "rgb"
        meas_dir = route_dir / "measurements"
        if not (rgb_dir.exists() and meas_dir.exists()):
            continue
        # Match the file naming convention used by the data agent: NNNN.jpg.
        frame_files = sorted(rgb_dir.glob("[0-9][0-9][0-9][0-9].jpg"))
        # We need future measurements to be available for the GT waypoints.
        usable = [
            int(p.stem)
            for p in frame_files
            if (meas_dir / f"{int(p.stem) + pred_len:04d}.json.gz").exists()
        ]
        if not usable:
            continue
        # Stride to spread samples across the route.
        for i in range(0, len(usable), frame_stride):
            idx = usable[i]
            # Optional commentary annotation lives in a parallel `commentary/`
            # tree the dataset publishes alongside `data/`.
            comm_path = _commentary_path_for(route_dir, idx)
            samples.append(
                FrameSample(
                    route_dir=route_dir,
                    frame_idx=idx,
                    rgb_path=rgb_dir / f"{idx:04d}.jpg",
                    measurements_dir=meas_dir,
                    commentary_path=comm_path,
                )
            )
            if len(samples) >= num_samples:
                return samples
    return samples


def _commentary_path_for(route_dir: Path, frame_idx: int) -> Path | None:
    """The upstream dataset stores commentary at the same path with `data/` ->
    `commentary/` and `measurements/` -> `commentary/`."""
    parts = list(route_dir.parts)
    try:
        # rewrite the first `data` segment, like the upstream loader does.
        idx_data = parts.index("data")
        parts[idx_data] = "commentary"
    except ValueError:
        return None
    candidate = Path(*parts) / "commentary" / f"{frame_idx:04d}.json.gz"
    return candidate if candidate.exists() else None


def _load_json_gz(path: Path) -> dict:
    import ujson

    with gzip.open(path, "rt") as fh:
        return ujson.load(fh)


# ---------------------------------------------------------------------------
# Building a DrivingInput from raw files
# ---------------------------------------------------------------------------


def _crop_bottom(img: np.ndarray) -> np.ndarray:
    """Drop the bottom ~30% of the image to hide the bonnet. The exact ratio
    comes from upstream data collection (`(h * 4.8) // 16`)."""
    h = img.shape[0]
    new_h = int(h - (h * 4.8) // 16)
    return img[:new_h, :, :]


def _compute_waypoints_from_ego_matrix(
    measurements: list[dict],
) -> np.ndarray:
    """Replicates `BaseDataset.get_waypoints`: future positions in the current
    ego frame, dropping the height dimension. Returns shape [F, 2]."""
    origin = np.asarray(measurements[0]["ego_matrix"])[:3]
    origin_t = origin[:, 3:4]
    origin_R = origin[:, :3]
    wps = []
    for m in measurements[1:]:
        p = np.asarray(m["ego_matrix"])[:3, 3:4]
        ego = origin_R.T @ (p - origin_t)
        wps.append(ego[:2, 0])
    return np.asarray(wps, dtype=np.float32)


def _equal_spacing_route(points: np.ndarray, num: int = 20) -> np.ndarray:
    """Mirror of `BaseDataset.equal_spacing_route`: 1m-spaced interpolation
    starting at the origin, padded out to `num` points."""
    pts = np.concatenate((np.zeros_like(points[:1]), points))
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


def _build_prompt(speed_mps: float, use_cot: bool) -> str:
    speed_rounded = round(speed_mps, 1)
    # The placeholder appears twice because the model embeds *both* target
    # points (current and next) by replacing each placeholder slot. See
    # `team_code/agent_simlingo.py::tick`.
    target_segment = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
    if use_cot:
        return f"Current speed: {speed_rounded} m/s. {target_segment} What should the ego do next?"
    return f"Current speed: {speed_rounded} m/s. {target_segment} Predict the waypoints."


def _process_image(
    rgb_path: Path,
    use_global_img: bool,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Run the upstream InternVL2 dynamic patching on a single front-cam frame.

    Returns:
        pixel_values: shape [1, T=1, num_patches, 3, 448, 448], float32.
        (H, W): the size of the cropped image fed to the patcher.
    """
    import cv2
    from PIL import Image

    from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess

    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(rgb_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Live agent does an extra JPEG roundtrip; on already-jpeg dataset frames
    # this is essentially a no-op so we skip it for speed.
    rgb = _crop_bottom(rgb)
    H, W, _ = rgb.shape

    transform = build_transform(input_size=448)
    pil = Image.fromarray(rgb)
    patches = dynamic_preprocess(
        pil,
        image_size=448,
        use_thumbnail=use_global_img,
        max_num=2,
    )
    pixel_values = torch.stack([transform(p) for p in patches])  # [P, 3, 448, 448]
    pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)  # [1, T=1, P, 3, 448, 448]
    return pixel_values, (H, W)


def _build_language_label(
    prompt_text: str,
    target_points_np: np.ndarray,
    tokenizer,
    encoder_variant: str,
    num_image_tokens_total: int,
    device: torch.device,
):
    """Tokenize the prompt, slot in image tokens, and build two LanguageLabel
    objects (the upstream forward expects both `prompt` and `prompt_inference`
    to be present)."""
    from simlingo_training.utils.custom_types import LanguageLabel
    from simlingo_training.utils.internvl2_utils import get_custom_chat_template

    # The upstream code expects each conversation as a list-of-dict messages
    # with a list-of-content blocks. We mimic the agent's two-turn shape so the
    # InternLM2 chat template renders identically to training.
    conversation_all = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Waypoints:"},
                ],
            },
        ]
    ]

    conv_dict, question_dict = get_custom_chat_template(
        conversation_all,
        tokenizer,
        encoder_variant=encoder_variant,
        num_image_tokens_total=num_image_tokens_total,
    )

    placeholder_token_id = tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
    placeholder_batch_list = [{placeholder_token_id: target_points_np}]

    def _to_ll(d):
        return LanguageLabel(
            phrase_ids=d["phrase_ids"].to(device),
            phrase_valid=d["phrase_valid"].to(device),
            phrase_mask=d["phrase_mask"].to(device),
            placeholder_values=placeholder_batch_list,
            language_string=d["language_string"],
            loss_masking=d.get("loss_masking"),
        )

    return _to_ll(conv_dict), _to_ll(question_dict)


def _build_driving_input(
    pixel_values: torch.Tensor,
    speed_mps: float,
    target_points_np: np.ndarray,
    prompt_ll,
    prompt_inference_ll,
    HW: tuple[int, int],
    device: torch.device,
):
    """Assemble the upstream `DrivingInput` namedtuple."""
    from simlingo_training.utils.custom_types import DrivingInput
    from simlingo_training.utils.projection import (
        get_camera_extrinsics,
        get_camera_intrinsics,
    )

    H, W = HW
    intrinsics = get_camera_intrinsics(W, H, 110).unsqueeze(0).to(device).float()
    extrinsics = get_camera_extrinsics().unsqueeze(0).to(device).float()
    return DrivingInput(
        camera_images=pixel_values.to(device).bfloat16(),
        image_sizes=None,
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        vehicle_speed=torch.tensor([[speed_mps]], dtype=torch.float32, device=device),
        target_point=torch.tensor(
            target_points_np[0:1], dtype=torch.float32, device=device
        ),
        prompt=prompt_ll,
        prompt_inference=prompt_inference_ll,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(ckpt_path: str, hydra_cfg_path: str, device: torch.device):
    """Instantiate the released SimLingo checkpoint exactly the way Hydra would."""
    import hydra
    from omegaconf import OmegaConf
    from transformers import AutoProcessor

    cfg = OmegaConf.load(hydra_cfg_path)
    # The agent's live setup forwards `use_global_img` from the data module to
    # the vision model; replicate that to keep grid sizes consistent.
    cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

    print(f"  load processor: {cfg.model.vision_model.variant}", flush=True)
    processor = AutoProcessor.from_pretrained(
        cfg.model.vision_model.variant,
        trust_remote_code=True,
    )
    if "tokenizer" in processor.__dict__:
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    # These match `team_code/agent_simlingo.py` exactly. Without them the
    # placeholder substitution in the model won't find <TARGET_POINT>.
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<WAYPOINTS>",
                "<WAYPOINTS_DIFF>",
                "<ORG_WAYPOINTS_DIFF>",
                "<ORG_WAYPOINTS>",
                "<WAYPOINT_LAST>",
                "<ROUTE>",
                "<ROUTE_DIFF>",
                "<TARGET_POINT>",
            ]
        }
    )
    tokenizer.padding_side = "left"

    cache_dir = f"pretrained/{cfg.model.vision_model.variant.split('/')[1]}"

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = hydra.utils.instantiate(
            cfg.model,
            cfg_data_module=cfg.data_module,
            processor=processor,
            cache_dir=cache_dir,
            _recursive_=False,
        ).to(device)
    finally:
        torch.set_default_dtype(default_dtype)

    print(f"  load state dict: {ckpt_path}", flush=True)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARN: missing keys in state_dict (n={len(missing)})", flush=True)
        for k in missing[:5]:
            print(f"        {k}", flush=True)
    if unexpected:
        print(f"  WARN: unexpected keys in state_dict (n={len(unexpected)})", flush=True)
        for k in unexpected[:5]:
            print(f"        {k}", flush=True)

    model.eval()
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _ade_fde(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """L2 displacement averaged over the trajectory (ADE) and final-step
    displacement (FDE), in meters."""
    n = min(len(pred), len(gt))
    if n == 0:
        return float("nan"), float("nan")
    d = np.linalg.norm(pred[:n] - gt[:n], axis=-1)
    return float(d.mean()), float(d[-1])


def _implied_speed(wps: np.ndarray, wp_freq: int = 5, carla_fps: int = 20) -> float:
    """Same as upstream PID: distance between wps half a second apart, *2."""
    one_sec = int(carla_fps // wp_freq)
    half_sec = one_sec // 2
    if len(wps) <= one_sec:
        return float("nan")
    return float(np.linalg.norm(wps[half_sec - 2] - wps[one_sec - 2]) * 2.0)


# ---------------------------------------------------------------------------
# Public entrypoint (called from modal_app.py::run)
# ---------------------------------------------------------------------------


def _setup_model_and_tokens(
    ckpt_path: str,
    hydra_cfg_path: str,
):
    """Common bootstrap: patch sys.path, load model + tokenizer + cfg, and
    compute the number of image tokens the prompt expects.

    Returns (model, tokenizer, cfg, num_image_tokens_total, use_global_img,
    device).
    """
    sys.path.insert(0, "/opt/simlingo")
    _ensure_internvl_pretrained_symlink()

    from simlingo_training.utils.internvl2_utils import get_num_image_tokens_per_patch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}", flush=True)
    print(">> loading model", flush=True)
    t0 = time.time()
    model, tokenizer, cfg = _load_model(ckpt_path, hydra_cfg_path, device)
    print(f"   loaded in {time.time() - t0:.1f}s", flush=True)

    use_global_img = bool(cfg.data_module.use_global_img)
    NUM_IMAGE_PATCHES = 2
    num_image_tokens_total = (
        get_num_image_tokens_per_patch(cfg.model.vision_model.variant) * NUM_IMAGE_PATCHES
    )
    return model, tokenizer, cfg, num_image_tokens_total, use_global_img, device


def _predict_one(
    *,
    model,
    tokenizer,
    cfg,
    device: torch.device,
    use_global_img: bool,
    num_image_tokens_total: int,
    rgb_path: Path,
    speed_mps: float,
    target_points: np.ndarray,
    use_cot: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, str, tuple[int, int]]:
    """Run the model on a single (image, speed, target_points) tuple.

    Returns (pred_wps, pred_route_1m, pred_text, (H, W)) where the image
    dimensions are after the bonnet crop done by `_process_image`.
    """
    pixel_values, HW = _process_image(rgb_path, use_global_img=use_global_img)

    prompt_text = _build_prompt(speed_mps, use_cot=use_cot)
    prompt_ll, prompt_inf_ll = _build_language_label(
        prompt_text,
        target_points_np=target_points,
        tokenizer=tokenizer,
        encoder_variant=cfg.model.vision_model.variant,
        num_image_tokens_total=num_image_tokens_total,
        device=device,
    )

    driving_input = _build_driving_input(
        pixel_values,
        speed_mps=speed_mps,
        target_points_np=target_points,
        prompt_ll=prompt_ll,
        prompt_inference_ll=prompt_inf_ll,
        HW=HW,
        device=device,
    )

    with torch.inference_mode():
        speed_wps, route_wps, language = model(driving_input)

    pred_wps = speed_wps[0].float().cpu().numpy() if speed_wps is not None else None
    pred_route = (
        _equal_spacing_route(route_wps[0].float().cpu().numpy(), num=20)
        if route_wps is not None
        else None
    )
    pred_text = language[0] if language else ""
    return pred_wps, pred_route, pred_text, HW


def run_inference(
    *,
    ckpt_path: str,
    hydra_cfg_path: str,
    data_root: str,
    out_dir: str,
    num_samples: int,
    frame_stride: int,
    use_cot: bool,
    save_overlays: bool,
    seed: int,
) -> dict[str, Any]:
    from . import viz  # local

    model, tokenizer, cfg, num_image_tokens_total, use_global_img, device = (
        _setup_model_and_tokens(ckpt_path, hydra_cfg_path)
    )

    print(">> collecting samples", flush=True)
    samples = _collect_samples(
        data_root,
        num_samples=num_samples,
        frame_stride=frame_stride,
        seed=seed,
    )
    print(f"   collected {len(samples)} samples", flush=True)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir_p / "overlays"
    if save_overlays:
        overlays_dir.mkdir(exist_ok=True)

    per_sample: list[dict] = []
    n_written = 0
    for i, s in enumerate(samples):
        try:
            measurements_now_to_future = []
            for fi in range(s.frame_idx, s.frame_idx + 12):
                measurements_now_to_future.append(
                    _load_json_gz(s.measurements_dir / f"{fi:04d}.json.gz")
                )
            current = measurements_now_to_future[0]

            speed_mps = float(current["speed"])
            target_point = np.asarray(current["target_point"], dtype=np.float32)
            next_target_point = np.asarray(current["target_point_next"], dtype=np.float32)
            target_points = np.stack([target_point, next_target_point], axis=0)

            gt_wps = _compute_waypoints_from_ego_matrix(measurements_now_to_future)
            gt_path = _equal_spacing_route(
                np.asarray(current["route"], dtype=np.float32)
            )

            pred_wps, pred_route, pred_text, _ = _predict_one(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
                use_global_img=use_global_img,
                num_image_tokens_total=num_image_tokens_total,
                rgb_path=s.rgb_path,
                speed_mps=speed_mps,
                target_points=target_points,
                use_cot=use_cot,
            )

            ade_wp, fde_wp = (
                _ade_fde(pred_wps, gt_wps)
                if pred_wps is not None
                else (float("nan"), float("nan"))
            )
            ade_path, fde_path = (
                _ade_fde(pred_route, gt_path)
                if pred_route is not None
                else (float("nan"), float("nan"))
            )

            gt_commentary = None
            if s.commentary_path is not None:
                try:
                    gt_commentary = _load_json_gz(s.commentary_path).get("commentary")
                except Exception:
                    gt_commentary = None

            entry = {
                "route": str(s.route_dir),
                "frame": s.frame_idx,
                "speed_mps": speed_mps,
                "target_point": target_point.tolist(),
                "next_target_point": next_target_point.tolist(),
                "ade_wp": ade_wp,
                "fde_wp": fde_wp,
                "ade_path": ade_path,
                "fde_path": fde_path,
                "implied_speed_pred": _implied_speed(pred_wps) if pred_wps is not None else None,
                "implied_speed_gt": _implied_speed(gt_wps),
                "pred_commentary": pred_text,
                "gt_commentary": gt_commentary,
                "prompt": _build_prompt(speed_mps, use_cot=use_cot),
            }
            per_sample.append(entry)

            if save_overlays:
                overlay_path = overlays_dir / f"{i:04d}.png"
                viz.write_overlay(
                    rgb_path=s.rgb_path,
                    pred_wps=pred_wps,
                    pred_route=pred_route,
                    gt_wps=gt_wps,
                    gt_route=gt_path,
                    pred_text=pred_text,
                    gt_text=gt_commentary,
                    out_path=overlay_path,
                )
                n_written += 1

            if (i + 1) % 5 == 0 or i == len(samples) - 1:
                print(
                    f"  [{i + 1}/{len(samples)}] ADE_wp={ade_wp:.2f} FDE_wp={fde_wp:.2f}"
                    f"  ADE_path={ade_path:.2f} pred='{pred_text[:60]}'",
                    flush=True,
                )

        except Exception as exc:
            print(f"  [{i}] ERROR: {exc!r}", flush=True)
            per_sample.append({"error": repr(exc), "route": str(s.route_dir), "frame": s.frame_idx})

    metrics = _aggregate(per_sample)
    summary = {
        "metrics": metrics,
        "num_samples": len(samples),
        "num_written": n_written,
        "out_dir": str(out_dir_p),
        "ckpt": ckpt_path,
    }
    with open(out_dir_p / "predictions.json", "w") as fh:
        json.dump({"summary": summary, "per_sample": per_sample}, fh, indent=2, default=str)
    return summary


# ---------------------------------------------------------------------------
# Generic per-sample driver (CARLA-free) — used for nuScenes and later the cart
# ---------------------------------------------------------------------------


def run_external_samples(
    *,
    ckpt_path: str,
    hydra_cfg_path: str,
    out_dir: str,
    samples: Iterable[Any],
    use_cot: bool,
    save_overlays: bool,
    label: str = "external",
) -> dict[str, Any]:
    """Run the model on an arbitrary stream of `ExternalSample` objects.

    `samples` must yield objects with at least:
        rgb_path, speed_mps, target_points; optionally gt_wps, gt_route,
        gt_commentary, intrinsics, fov_deg, cam_translation_xyz, crop_bottom,
        meta.
    Defined in scripts/nuscenes_loader.py::ExternalSample.
    """
    from . import viz  # local

    model, tokenizer, cfg, num_image_tokens_total, use_global_img, device = (
        _setup_model_and_tokens(ckpt_path, hydra_cfg_path)
    )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir_p / "overlays"
    if save_overlays:
        overlays_dir.mkdir(exist_ok=True)

    per_sample: list[dict] = []
    n_written = 0
    print(f">> running {label} inference", flush=True)
    for i, s in enumerate(samples):
        try:
            pred_wps, pred_route, pred_text, _ = _predict_one(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                device=device,
                use_global_img=use_global_img,
                num_image_tokens_total=num_image_tokens_total,
                rgb_path=s.rgb_path,
                speed_mps=float(s.speed_mps),
                target_points=np.asarray(s.target_points, dtype=np.float32),
                use_cot=use_cot,
            )

            gt_wps = getattr(s, "gt_wps", None)
            gt_route = getattr(s, "gt_route", None)
            ade_wp, fde_wp = (
                _ade_fde(pred_wps, np.asarray(gt_wps))
                if (pred_wps is not None and gt_wps is not None)
                else (float("nan"), float("nan"))
            )
            ade_path, fde_path = (
                _ade_fde(pred_route, np.asarray(gt_route))
                if (pred_route is not None and gt_route is not None)
                else (float("nan"), float("nan"))
            )

            entry = {
                "meta": getattr(s, "meta", {}),
                "rgb_path": str(s.rgb_path),
                "speed_mps": float(s.speed_mps),
                "target_points": np.asarray(s.target_points).tolist(),
                "ade_wp": ade_wp,
                "fde_wp": fde_wp,
                "ade_path": ade_path,
                "fde_path": fde_path,
                "implied_speed_pred": _implied_speed(pred_wps) if pred_wps is not None else None,
                "implied_speed_gt": _implied_speed(np.asarray(gt_wps)) if gt_wps is not None else None,
                "pred_commentary": pred_text,
                "gt_commentary": getattr(s, "gt_commentary", None),
            }
            per_sample.append(entry)

            if save_overlays:
                overlay_path = overlays_dir / f"{i:04d}.png"
                viz.write_overlay(
                    rgb_path=s.rgb_path,
                    pred_wps=pred_wps,
                    pred_route=pred_route,
                    gt_wps=gt_wps,
                    gt_route=gt_route,
                    pred_text=pred_text,
                    gt_text=getattr(s, "gt_commentary", None),
                    out_path=overlay_path,
                    crop_bottom=getattr(s, "crop_bottom", True),
                    intrinsics=getattr(s, "intrinsics", None),
                    fov_deg=getattr(s, "fov_deg", 110.0),
                    cam_translation_xyz=getattr(
                        s, "cam_translation_xyz", (0.0, 2.0, 1.5)
                    ),
                )
                n_written += 1

            if (i + 1) % 5 == 0:
                print(
                    f"  [{i + 1}] ADE_wp={ade_wp:.2f} FDE_wp={fde_wp:.2f}"
                    f"  ADE_path={ade_path:.2f} pred='{pred_text[:60]}'",
                    flush=True,
                )

        except Exception as exc:
            print(f"  [{i}] ERROR: {exc!r}", flush=True)
            per_sample.append({"error": repr(exc), "meta": getattr(s, "meta", {})})

    metrics = _aggregate(per_sample)
    summary = {
        "metrics": metrics,
        "num_samples": sum(1 for _ in per_sample),
        "num_written": n_written,
        "out_dir": str(out_dir_p),
        "ckpt": ckpt_path,
        "label": label,
    }
    with open(out_dir_p / "predictions.json", "w") as fh:
        json.dump({"summary": summary, "per_sample": per_sample}, fh, indent=2, default=str)
    print(f">> {label} done — wrote {n_written} overlays", flush=True)
    return summary


def _prepare_frame_context(
    sample: FrameSample,
    *,
    tokenizer,
    encoder_variant: str,
    num_image_tokens_total: int,
    use_cot: bool,
    use_global_img: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Load measurements + image and assemble the `DrivingInput` for one frame."""
    measurements_now_to_future = [
        _load_json_gz(sample.measurements_dir / f"{fi:04d}.json.gz")
        for fi in range(sample.frame_idx, sample.frame_idx + 12)
    ]
    current = measurements_now_to_future[0]

    speed_mps = float(current["speed"])
    target_point = np.asarray(current["target_point"], dtype=np.float32)
    next_target_point = np.asarray(current["target_point_next"], dtype=np.float32)
    target_points = np.stack([target_point, next_target_point], axis=0)

    gt_wps = _compute_waypoints_from_ego_matrix(measurements_now_to_future)
    gt_path = _equal_spacing_route(np.asarray(current["route"], dtype=np.float32))

    pixel_values, HW = _process_image(sample.rgb_path, use_global_img=use_global_img)
    prompt_text = _build_prompt(speed_mps, use_cot=use_cot)
    prompt_ll, prompt_inf_ll = _build_language_label(
        prompt_text,
        target_points_np=target_points,
        tokenizer=tokenizer,
        encoder_variant=encoder_variant,
        num_image_tokens_total=num_image_tokens_total,
        device=device,
    )
    driving_input = _build_driving_input(
        pixel_values,
        speed_mps=speed_mps,
        target_points_np=target_points,
        prompt_ll=prompt_ll,
        prompt_inference_ll=prompt_inf_ll,
        HW=HW,
        device=device,
    )

    gt_commentary = None
    if sample.commentary_path is not None:
        try:
            gt_commentary = _load_json_gz(sample.commentary_path).get("commentary")
        except Exception:
            gt_commentary = None

    return {
        "driving_input": driving_input,
        "gt_wps": gt_wps,
        "gt_path": gt_path,
        "gt_commentary": gt_commentary,
        "prompt_text": prompt_text,
        "speed_mps": speed_mps,
    }


def _stitch_video(image_paths: list[Path], out_path: Path, fps: int) -> None:
    """Encode a list of PNGs (assumed identical size) into an mp4."""
    import cv2

    if not image_paths:
        raise ValueError("no frames to stitch")
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"cv2 could not read {image_paths[0]}")
    h, w, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    try:
        for p in image_paths:
            frame = cv2.imread(str(p))
            if frame is None:
                print(f"  skip unreadable frame: {p}", flush=True)
                continue
            writer.write(frame)
    finally:
        writer.release()


def run_video_inference(
    *,
    ckpt_path: str,
    hydra_cfg_path: str,
    data_root: str,
    out_dir: str,
    num_frames: int = 200,
    fps: int = 20,
    use_cot: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Run inference on a contiguous span of validation frames, write per-frame
    overlays + an mp4, and report wall-clock Hz for the model forward pass."""
    sys.path.insert(0, "/opt/simlingo")
    _ensure_internvl_pretrained_symlink()

    from simlingo_training.utils.internvl2_utils import get_num_image_tokens_per_patch

    from . import viz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> device: {device}", flush=True)
    print(">> loading model", flush=True)
    t0 = time.time()
    model, tokenizer, cfg = _load_model(ckpt_path, hydra_cfg_path, device)
    print(f"   loaded in {time.time() - t0:.1f}s", flush=True)

    use_global_img = bool(cfg.data_module.use_global_img)
    NUM_IMAGE_PATCHES = 2
    num_image_tokens_total = (
        get_num_image_tokens_per_patch(cfg.model.vision_model.variant) * NUM_IMAGE_PATCHES
    )

    print(">> collecting video samples", flush=True)
    samples = _collect_video_samples(data_root, num_frames=num_frames, seed=seed)
    if not samples:
        raise RuntimeError(
            f"could not find a route with {num_frames} consecutive usable frames"
        )

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir_p / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    # Warmup: lazy CUDA kernels (and language-head generation graph) only
    # materialize on the first call, which makes that call wildly slower than
    # steady-state. Run once and discard.
    print(">> warmup", flush=True)
    warm_ctx = _prepare_frame_context(
        samples[0],
        tokenizer=tokenizer,
        encoder_variant=cfg.model.vision_model.variant,
        num_image_tokens_total=num_image_tokens_total,
        use_cot=use_cot,
        use_global_img=use_global_img,
        device=device,
    )
    with torch.inference_mode():
        _ = model(warm_ctx["driving_input"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    overlay_paths: list[Path] = []
    per_frame: list[dict[str, float]] = []
    wall_t0 = time.perf_counter()

    for i, s in enumerate(samples):
        try:
            t_total0 = time.perf_counter()

            t_pre0 = time.perf_counter()
            ctx = _prepare_frame_context(
                s,
                tokenizer=tokenizer,
                encoder_variant=cfg.model.vision_model.variant,
                num_image_tokens_total=num_image_tokens_total,
                use_cot=use_cot,
                use_global_img=use_global_img,
                device=device,
            )
            t_pre = time.perf_counter() - t_pre0

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_inf0 = time.perf_counter()
            with torch.inference_mode():
                speed_wps, route_wps, language = model(ctx["driving_input"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_inf = time.perf_counter() - t_inf0

            pred_wps = (
                speed_wps[0].float().cpu().numpy() if speed_wps is not None else None
            )
            pred_route = (
                _equal_spacing_route(route_wps[0].float().cpu().numpy(), num=20)
                if route_wps is not None
                else None
            )
            pred_text = language[0] if language else ""

            overlay_path = overlays_dir / f"{i:04d}.png"
            viz.write_overlay(
                rgb_path=s.rgb_path,
                pred_wps=pred_wps,
                pred_route=pred_route,
                gt_wps=ctx["gt_wps"],
                gt_route=ctx["gt_path"],
                pred_text=pred_text,
                gt_text=ctx["gt_commentary"],
                out_path=overlay_path,
            )
            overlay_paths.append(overlay_path)

            t_total = time.perf_counter() - t_total0
            per_frame.append(
                {
                    "preprocess_s": t_pre,
                    "inference_s": t_inf,
                    "total_s": t_total,
                }
            )

            if (i + 1) % 10 == 0 or i == len(samples) - 1:
                running_hz = (i + 1) / (time.perf_counter() - wall_t0)
                print(
                    f"  [{i + 1:>3}/{len(samples)}] "
                    f"pre={t_pre * 1000:5.0f}ms inf={t_inf * 1000:5.0f}ms "
                    f"total={t_total * 1000:5.0f}ms  running_hz={running_hz:.2f}",
                    flush=True,
                )
        except Exception as exc:
            print(f"  [{i}] ERROR: {exc!r}", flush=True)

    wall_total = time.perf_counter() - wall_t0
    inf_times = np.asarray([row["inference_s"] for row in per_frame], dtype=np.float64)
    total_times = np.asarray([row["total_s"] for row in per_frame], dtype=np.float64)

    def _stats(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean_ms": float(arr.mean() * 1000),
            "p50_ms": float(np.median(arr) * 1000),
            "p90_ms": float(np.percentile(arr, 90) * 1000),
            "mean_hz": float(1.0 / arr.mean()) if arr.mean() > 0 else float("nan"),
            "p50_hz": float(1.0 / np.median(arr)) if np.median(arr) > 0 else float("nan"),
        }

    timing = {
        "num_frames": len(per_frame),
        "wall_total_s": wall_total,
        "wall_hz": (len(per_frame) / wall_total) if wall_total > 0 else float("nan"),
        "model_forward": _stats(inf_times) if len(inf_times) else {},
        "end_to_end_per_frame": _stats(total_times) if len(total_times) else {},
    }

    print(
        f">> timing: wall_hz={timing['wall_hz']:.2f}  "
        f"model_forward_mean_hz={timing['model_forward'].get('mean_hz', float('nan')):.2f}  "
        f"(mean={timing['model_forward'].get('mean_ms', float('nan')):.0f}ms, "
        f"p50={timing['model_forward'].get('p50_ms', float('nan')):.0f}ms, "
        f"p90={timing['model_forward'].get('p90_ms', float('nan')):.0f}ms)",
        flush=True,
    )

    print(">> stitching video", flush=True)
    video_path = out_dir_p / "predictions.mp4"
    _stitch_video(overlay_paths, video_path, fps=fps)
    print(f"   wrote {video_path}", flush=True)

    summary = {
        "video_path": str(video_path),
        "fps": fps,
        "timing": timing,
        "out_dir": str(out_dir_p),
        "ckpt": ckpt_path,
    }
    with open(out_dir_p / "video_summary.json", "w") as fh:
        json.dump(
            {"summary": summary, "per_frame": per_frame},
            fh,
            indent=2,
            default=str,
        )
    return summary


def _aggregate(rows: Iterable[dict]) -> dict[str, float]:
    keys = ["ade_wp", "fde_wp", "ade_path", "fde_path"]
    agg: dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), float) and not math.isnan(r[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_p50"] = float(np.median(vals)) if vals else float("nan")
        agg[f"{k}_p90"] = float(np.percentile(vals, 90)) if vals else float("nan")
    agg["num_evaluated"] = float(sum(1 for r in rows if "ade_wp" in r))
    return agg
