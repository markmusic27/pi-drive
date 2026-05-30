"""Optimized SimLingo inference pipeline.

This module provides the main optimized inference class that wraps the SimLingo
model with various optimizations:

- torch.compile with max-autotune mode
- GPU-accelerated preprocessing
- Static KV cache pre-allocation
- Quantization support
- Streaming inference for multi-frame processing

Usage:
    from simlingo_optimized import OptimizedSimLingo, OptimizationConfig

    config = OptimizationConfig(
        use_torch_compile=True,
        use_gpu_preprocessing=True,
    )

    model = OptimizedSimLingo(
        ckpt_path="/path/to/checkpoint.pt",
        config=config,
    )

    waypoints, route, commentary = model.predict(
        image=image_bytes,
        speed_mps=5.0,
        target_points=target_points,
    )
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .config import OptimizationConfig
from .preprocessing import create_preprocessor


def _install_waypoints_only_forward(model) -> None:
    """Replace the upstream DrivingModel.forward with a fixed waypoints-only path.

    The upstream `simlingo_training.models.driving.DrivingModel.forward` has a
    bug in its `predict_language=False` branch: `forward_model` returns the
    tuple `(adaptor_features, adaptor_logits)`, but the code passes that tuple
    directly to `split_outputs_by_adaptor`, which then attempts fancy indexing
    on a tuple and raises TypeError. The training-time code path consumes the
    pair correctly via `compute_loss`; we replicate that consumption here for
    inference by splitting BOTH features and logits per-adaptor and passing
    both into `get_predictions`. The result is the canonical training-time
    regression prediction with no autoregressive language step.

    The wrapper dispatches on `self.predict_language`:
      - False  -> our fixed waypoints-only single-prefill path
      - True   -> defer to the original upstream forward (autoregressive
                  greedy_sample loop). This lets callers A/B compare both
                  modes on the same model instance.
    """
    import types

    # Save the bound original forward exactly once.
    if not hasattr(model, "_orig_forward"):
        model._orig_forward = model.forward

    def _dispatch_forward(self, example, return_language=None, prompt_ids=None):
        if getattr(self, "predict_language", False):
            return self._orig_forward(
                example, return_language=return_language, prompt_ids=prompt_ids
            )

        # waypoints-only fast path
        self.speed_wps = None
        self.route = None
        self.language = []
        try:
            driving_input = example.driving_input
        except AttributeError:
            driving_input = example

        adaptor_dict = self.adaptors(example, inference=True)
        adaptor_dict = self.vision_model.image_encoder.replace_placeholder_tokens(
            adaptor_dict=adaptor_dict,
            pixel_values=driving_input.camera_images,
            placeholder_values=driving_input.prompt_inference.placeholder_values,
            wp_encoder=self.wp_encoder,
        )

        adaptor_features, adaptor_logits = self.forward_model(driving_input, adaptor_dict)
        features_by_adaptor = self.adaptors.split_outputs_by_adaptor(
            adaptor_dict, adaptor_features
        )
        logits_by_adaptor = self.adaptors.split_outputs_by_adaptor(
            adaptor_dict, adaptor_logits
        )
        predictions = self.adaptors.driving.get_predictions(
            features_by_adaptor["driving"],
            logits_by_adaptor["driving"],
        )

        for k, v in predictions.items():
            if v is not None:
                setattr(self, k, v)

        return self.speed_wps, self.route, self.language

    model.forward = types.MethodType(_dispatch_forward, model)


class OptimizedSimLingo(nn.Module):
    """Optimized SimLingo model wrapper for inference.

    This class wraps the original SimLingo model and applies various
    optimizations based on the provided configuration.
    """

    def __init__(
        self,
        ckpt_path: str,
        config: Optional[OptimizationConfig] = None,
        hydra_cfg_path: Optional[str] = None,
        simlingo_repo_path: str = "/opt/simlingo",
        cache_dir: str = "/cache/hf/snapshots",
    ):
        """Initialize optimized SimLingo model.

        Args:
            ckpt_path: Path to model checkpoint (.pt or .ckpt file)
            config: Optimization configuration. If None, uses default config.
            hydra_cfg_path: Path to Hydra config YAML. If None, uses default.
            simlingo_repo_path: Path to simlingo_training repo.
            cache_dir: Path to HuggingFace cache directory.
        """
        super().__init__()
        self.ckpt_path = ckpt_path
        self.config = config or OptimizationConfig()
        self.hydra_cfg_path = hydra_cfg_path
        self.simlingo_repo_path = simlingo_repo_path
        self.cache_dir = cache_dir

        # Device and dtype
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.dtype = self.config.get_torch_dtype()

        # State
        self._model = None
        self._tokenizer = None
        self._cfg = None
        self._preprocessor = None
        self._num_image_tokens_total = None
        self._compiled = False
        self._warmed_up = False

        # Streaming state
        self._kv_cache = None
        self._frame_count = 0
        self._streaming_manager = None
        self._vision_hook_handle = None
        self._cached_vision_output = None
        self._use_cached_vision = False
        self._current_image_np = None

        # Timing stats
        self._timing_stats = {
            "preprocess_ms": [],
            "inference_ms": [],
            "total_ms": [],
        }

        # Apply global optimizations
        self._apply_global_optimizations()

    def _apply_global_optimizations(self):
        """Apply global CUDA optimizations."""
        if not torch.cuda.is_available():
            return

        if self.config.tf32_mode:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    def _ensure_simlingo_path(self):
        """Ensure simlingo_training is in sys.path."""
        if self.simlingo_repo_path not in sys.path:
            sys.path.insert(0, self.simlingo_repo_path)

    def _ensure_pretrained_symlink(self):
        """Set up pretrained model symlink for InternVL2."""
        workdir = Path("/tmp/simlingo_optimized_work")
        workdir.mkdir(parents=True, exist_ok=True)
        os.chdir(workdir)

        pretrained = workdir / "pretrained"
        pretrained.mkdir(exist_ok=True)
        link = pretrained / "InternVL2-1B"
        src = Path(self.cache_dir) / "InternVL2-1B"

        if src.exists() and (not link.exists() or link.is_symlink()):
            if link.is_symlink():
                link.unlink()
            if src.exists():
                link.symlink_to(src)

        return workdir

    def _load_model(self):
        """Load the base SimLingo model."""
        if self._model is not None:
            return

        self._ensure_simlingo_path()
        self._ensure_pretrained_symlink()

        import hydra
        from omegaconf import OmegaConf
        from transformers import AutoProcessor

        # Load config
        if self.hydra_cfg_path:
            cfg = OmegaConf.load(self.hydra_cfg_path)
        else:
            # Use default SimLingo config
            cfg = self._get_default_config()

        cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img

        if self.config.verbose:
            print(f"Loading processor: {cfg.model.vision_model.variant}", flush=True)

        # Load processor/tokenizer
        processor = AutoProcessor.from_pretrained(
            cfg.model.vision_model.variant,
            trust_remote_code=True,
        )

        # Get tokenizer - may be nested in processor or need separate loading
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "add_special_tokens"):
            tokenizer = processor.tokenizer
        else:
            # Fallback: load tokenizer separately
            from transformers import AutoTokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.vision_model.variant,
                    trust_remote_code=True,
                )
            except ValueError:
                # Try with slow tokenizer if fast tokenizer fails
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg.model.vision_model.variant,
                    trust_remote_code=True,
                    use_fast=False,
                )

        # Add special tokens
        tokenizer.add_special_tokens({
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
        })
        tokenizer.padding_side = "left"

        cache_dir = f"pretrained/{cfg.model.vision_model.variant.split('/')[1]}"

        # Load model with bfloat16
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=cache_dir,
                _recursive_=False,
            ).to(self.device)
        finally:
            torch.set_default_dtype(default_dtype)

        # Load checkpoint
        if self.config.verbose:
            print(f"Loading checkpoint: {self.ckpt_path}", flush=True)

        state_dict = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing and self.config.verbose:
            print(f"Missing keys: {len(missing)}", flush=True)
        if unexpected and self.config.verbose:
            print(f"Unexpected keys: {len(unexpected)}", flush=True)

        model.eval()

        # Highest-ROI optimization: skip autoregressive commentary generation.
        # The upstream `simlingo_training.models.driving.DrivingModel` hardcodes
        # `self.predict_language = True` in __init__, which causes forward() to
        # run a per-batch-item greedy_sample loop (up to 100 LLM forward passes
        # per sample) plus an additional concat-and-forward to obtain driving
        # features. We want the False branch: single vision+LLM forward pass
        # that hands the hidden states straight to the regression adaptors.
        #
        # The upstream False branch is buggy though: `forward_model` returns
        # `(adaptor_features, adaptor_logits)` but the code passes that tuple
        # to `split_outputs_by_adaptor`, which then tries to do fancy indexing
        # on a tuple and raises TypeError. We monkey-patch a fixed forward()
        # that properly unpacks the tuple and feeds both features and logits
        # to `get_predictions`, matching exactly how the upstream training
        # path consumes them (so outputs match training-time predictions).
        if self.config.waypoints_only and hasattr(model, "predict_language"):
            model.predict_language = False
            _install_waypoints_only_forward(model)
            if self.config.verbose:
                print(
                    "waypoints_only=True: disabled autoregressive language head "
                    "+ installed fixed forward()",
                    flush=True,
                )

        # Store references
        self._model = model
        self._tokenizer = tokenizer
        self._cfg = cfg

        # Compute image tokens
        from simlingo_training.utils.internvl2_utils import (
            get_num_image_tokens_per_patch,
        )

        NUM_IMAGE_PATCHES = self.config.max_image_patches
        self._num_image_tokens_total = (
            get_num_image_tokens_per_patch(cfg.model.vision_model.variant)
            * NUM_IMAGE_PATCHES
        )

        # Apply optimizations
        self._apply_optimizations()

        if self.config.verbose:
            print(f"Model loaded on {self.device}", flush=True)

    def _get_default_config(self):
        """Get default SimLingo config."""
        from omegaconf import OmegaConf

        # Minimal config matching the released SimLingo model
        return OmegaConf.create({
            "model": {
                "_target_": "simlingo_training.model.SimLingoModel",
                "vision_model": {
                    "_target_": "simlingo_training.model.vision.InternVL2VisionModel",
                    "variant": "OpenGVLab/InternVL2-1B",
                    "use_global_img": True,
                },
                "num_waypoints": 11,
                "waypoint_spacing_s": 0.25,
                "route_points": 20,
            },
            "data_module": {
                "use_global_img": True,
            },
        })

    def _apply_optimizations(self):
        """Apply configured optimizations to the model."""
        if self._model is None:
            return

        # Streaming (set up FIRST, before torch.compile)
        # This wraps the vision encoder for caching
        if self.config.enable_streaming:
            self._setup_streaming()

        # Quantization (Phase 2)
        if self.config.quantization != "none":
            self._apply_quantization()

        # torch.compile (Phase 1) - applied AFTER streaming wrapper
        if self.config.use_torch_compile and not self._compiled:
            self._apply_torch_compile()

        # Fused kernels (Phase 3)
        if self.config.use_fused_kernels:
            self._apply_fused_kernels()

    def _apply_torch_compile(self):
        """Apply torch.compile to the vision encoder only.

        We only compile the vision encoder because:
        1. The language model uses DynamicCache which doesn't work well with torch.compile
        2. The vision encoder is compute-bound and benefits most from compilation
        3. The LLM's autoregressive generation has dynamic shapes that cause recompilations
        """
        if self._compiled:
            return

        if self.config.verbose:
            print(
                f"Applying torch.compile (mode={self.config.compile_mode})",
                flush=True,
            )

        try:
            # Only compile the vision encoder - it's compute-bound and has static shapes
            # The language model uses DynamicCache which causes issues with torch.compile
            vision_compiled = False

            # Try different attribute names for the vision encoder
            for attr_name in ["vision_model", "visual", "vit", "image_encoder"]:
                if hasattr(self._model, attr_name):
                    vision = getattr(self._model, attr_name)
                    compiled_vision = torch.compile(
                        vision,
                        mode=self.config.compile_mode,
                        dynamic=False,  # Vision has static shapes
                        fullgraph=False,
                    )
                    setattr(self._model, attr_name, compiled_vision)
                    vision_compiled = True
                    if self.config.verbose:
                        print(f"  Compiled {attr_name}", flush=True)
                    break

            # Also try to find vision encoder nested in the model
            if not vision_compiled and hasattr(self._model, "model"):
                inner = self._model.model
                for attr_name in ["vision_model", "visual", "vit"]:
                    if hasattr(inner, attr_name):
                        vision = getattr(inner, attr_name)
                        compiled_vision = torch.compile(
                            vision,
                            mode=self.config.compile_mode,
                            dynamic=False,
                            fullgraph=False,
                        )
                        setattr(inner, attr_name, compiled_vision)
                        vision_compiled = True
                        if self.config.verbose:
                            print(f"  Compiled model.{attr_name}", flush=True)
                        break

            if not vision_compiled and self.config.verbose:
                print("  Warning: Could not find vision encoder to compile", flush=True)

            # Skip language model compilation - DynamicCache causes issues
            if self.config.verbose:
                print("  Skipping language_model (DynamicCache incompatible)", flush=True)

            self._compiled = True
        except Exception as e:
            if self.config.verbose:
                print(f"torch.compile failed: {e}", flush=True)

    def _apply_quantization(self):
        """Apply quantization to the model."""
        if self.config.quantization == "none":
            return

        if self.config.verbose:
            print(f"Applying quantization: {self.config.quantization}", flush=True)

        # Import quantization module
        from .quantization import quantize_model

        self._model = quantize_model(
            self._model,
            quantization_type=self.config.quantization,
            quantize_vision=self.config.quantize_vision_encoder,
            quantize_llm=self.config.quantize_language_head,
            keep_embedding_fp16=self.config.keep_embedding_fp16,
        )

        # Disable gradients after quantization - required for bitsandbytes layers
        # The quantized layers modify weights internally during forward pass,
        # which conflicts with gradient tracking from PEFT/LoRA
        self._model.requires_grad_(False)
        self._model.eval()

    def _apply_fused_kernels(self):
        """Apply custom fused kernels."""
        if not self.config.use_fused_kernels:
            return

        if self.config.verbose:
            print("Applying fused kernels", flush=True)

        try:
            from .kernels import apply_fused_kernels

            self._model = apply_fused_kernels(self._model)
        except ImportError as e:
            if self.config.verbose:
                print(f"Fused kernels not available: {e}", flush=True)

    def _ensure_preprocessor(self):
        """Initialize preprocessor if needed."""
        if self._preprocessor is not None:
            return

        self._preprocessor = create_preprocessor(
            use_gpu=self.config.use_gpu_preprocessing,
            image_size=self.config.image_size,
            crop_bottom=self.config.crop_bottom,
            device=str(self.device),
            dtype=self.dtype,
        )

    def _build_prompt(self, speed_mps: float, use_cot: bool = True) -> str:
        """Build the prompt string."""
        speed_rounded = round(speed_mps, 1)
        target_segment = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        if use_cot:
            return f"Current speed: {speed_rounded} m/s. {target_segment} What should the ego do next?"
        return f"Current speed: {speed_rounded} m/s. {target_segment} Predict the waypoints."

    def _build_language_label(
        self,
        prompt_text: str,
        target_points_np: np.ndarray,
    ):
        """Build language labels for the model."""
        from simlingo_training.utils.custom_types import LanguageLabel
        from simlingo_training.utils.internvl2_utils import get_custom_chat_template

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
            self._tokenizer,
            encoder_variant=self._cfg.model.vision_model.variant,
            num_image_tokens_total=self._num_image_tokens_total,
        )

        # Build placeholder_values keyed by token id. The upstream
        # `replace_placeholder_tokens` scans the input ids for any token id
        # >= the smallest additional-special-token id and looks each one up in
        # this dict. Some of those ids belong to base-tokenizer special tokens
        # (e.g. image context, <|im_end|>) that aren't in
        # `additional_special_tokens_ids`, so a plain dict misses them and
        # raises KeyError. Use a defaultdict so any unregistered special id
        # silently resolves to a deterministic zero array (the upstream
        # wp_encoder then encodes those to a fixed embedding, which matches
        # training-time scaffolding where these placeholder slots existed but
        # were filled with zero/null at inference). <TARGET_POINT> still gets
        # the real target_points_np.
        from collections import defaultdict

        target_token_id = self._tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
        zero_placeholder = np.zeros_like(target_points_np)
        placeholder_dict: dict = defaultdict(lambda: zero_placeholder)
        if target_token_id is not None:
            placeholder_dict[target_token_id] = target_points_np
        placeholder_batch_list = [placeholder_dict]

        def _to_ll(d):
            return LanguageLabel(
                phrase_ids=d["phrase_ids"].to(self.device),
                phrase_valid=d["phrase_valid"].to(self.device),
                phrase_mask=d["phrase_mask"].to(self.device),
                placeholder_values=placeholder_batch_list,
                language_string=d["language_string"],
                loss_masking=d.get("loss_masking"),
            )

        return _to_ll(conv_dict), _to_ll(question_dict)

    def _build_driving_input(
        self,
        pixel_values: torch.Tensor,
        speed_mps: float,
        target_points_np: np.ndarray,
        prompt_ll,
        prompt_inference_ll,
        HW: Tuple[int, int],
    ):
        """Build DrivingInput for the model."""
        from simlingo_training.utils.custom_types import DrivingInput
        from simlingo_training.utils.projection import (
            get_camera_extrinsics,
            get_camera_intrinsics,
        )

        H, W = HW
        intrinsics = (
            get_camera_intrinsics(W, H, self.config.default_fov_deg)
            .unsqueeze(0)
            .to(self.device)
            .float()
        )
        extrinsics = get_camera_extrinsics().unsqueeze(0).to(self.device).float()

        return DrivingInput(
            camera_images=pixel_values.to(self.device).to(self.dtype),
            image_sizes=None,
            camera_intrinsics=intrinsics,
            camera_extrinsics=extrinsics,
            vehicle_speed=torch.tensor(
                [[speed_mps]], dtype=torch.float32, device=self.device
            ),
            target_point=torch.tensor(
                target_points_np[0:1], dtype=torch.float32, device=self.device
            ),
            prompt=prompt_ll,
            prompt_inference=prompt_inference_ll,
        )

    def _equal_spacing_route(
        self, points: np.ndarray, num: int = 20
    ) -> np.ndarray:
        """Resample route to equal 1m spacing."""
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

    def warmup(self, num_iterations: int = 3):
        """Warmup the model to trigger JIT compilation.

        The first few inference calls are slower due to:
        - torch.compile graph capture
        - CUDA kernel compilation
        - Memory allocation

        Args:
            num_iterations: Number of warmup iterations
        """
        if self._warmed_up:
            return

        self.load()

        if self.config.verbose:
            print(f"Warming up ({num_iterations} iterations)...", flush=True)

        # Create dummy input
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_speed = 5.0
        dummy_targets = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32)

        for i in range(num_iterations):
            try:
                _ = self.predict(
                    image=dummy_image,
                    speed_mps=dummy_speed,
                    target_points=dummy_targets,
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"Warmup iteration {i} failed: {e}", flush=True)

        self._warmed_up = True

        if self.config.verbose:
            print("Warmup complete", flush=True)

    def load(self):
        """Explicitly load the model.

        This is called automatically on first predict(), but can be called
        manually to control when model loading happens.
        """
        self._load_model()
        self._ensure_preprocessor()

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        speed_mps: float,
        target_points: np.ndarray,
        use_cot: bool = True,
        predict_language: Optional[bool] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Run inference on a single image.

        Args:
            image: Input image (file path, bytes, or numpy array)
            speed_mps: Current vehicle speed in m/s
            target_points: Target waypoints array of shape [2, 2] or [N, 2]
                          First row is current target, second is next target
            use_cot: Use chain-of-thought prompting
            predict_language: Optional per-call override for autoregressive
                language generation. When None (default), uses the value from
                OptimizationConfig.waypoints_only. Pass True to force commentary
                generation for this call, or False to force skip.

        Returns:
            Tuple of (waypoints, route, commentary):
                - waypoints: Predicted waypoints [11, 2] in ego frame, or None
                - route: Predicted route [20, 2] at 1m spacing, or None
                - commentary: Generated text commentary (empty when language
                              head is disabled)
        """
        # Ensure model is loaded
        if self._model is None:
            self.load()

        # Apply per-call language toggle if requested. We restore the previous
        # value in a finally block so concurrent benchmark configs don't leak.
        _prev_predict_language = None
        if predict_language is not None and hasattr(self._model, "predict_language"):
            _prev_predict_language = self._model.predict_language
            self._model.predict_language = bool(predict_language)

        try:
            t_total_start = time.perf_counter()

            # Preprocess image
            t_pre_start = time.perf_counter()
            pixel_values, HW = self._preprocessor(image, return_original_size=True)
            t_pre = (time.perf_counter() - t_pre_start) * 1000

            # Ensure target_points is correct shape
            target_points_np = np.asarray(target_points, dtype=np.float32)
            if target_points_np.ndim == 1:
                target_points_np = target_points_np.reshape(1, -1)
            if target_points_np.shape[0] == 1:
                # Duplicate if only one target provided
                target_points_np = np.vstack([target_points_np, target_points_np])

            # Build prompt and inputs
            prompt_text = self._build_prompt(speed_mps, use_cot=use_cot)
            prompt_ll, prompt_inf_ll = self._build_language_label(
                prompt_text, target_points_np
            )
            driving_input = self._build_driving_input(
                pixel_values,
                speed_mps=speed_mps,
                target_points_np=target_points_np,
                prompt_ll=prompt_ll,
                prompt_inference_ll=prompt_inf_ll,
                HW=HW,
            )

            # Run inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_inf_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=self.dtype
            ):
                speed_wps, route_wps, language = self._model(driving_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_inf = (time.perf_counter() - t_inf_start) * 1000
            t_total = (time.perf_counter() - t_total_start) * 1000

            # Record timing
            self._timing_stats["preprocess_ms"].append(t_pre)
            self._timing_stats["inference_ms"].append(t_inf)
            self._timing_stats["total_ms"].append(t_total)

            # Post-process outputs
            pred_wps = (
                speed_wps[0].float().cpu().numpy() if speed_wps is not None else None
            )
            pred_route = (
                self._equal_spacing_route(route_wps[0].float().cpu().numpy(), num=20)
                if route_wps is not None
                else None
            )
            pred_text = language[0] if language else ""

            return pred_wps, pred_route, pred_text
        finally:
            if _prev_predict_language is not None:
                self._model.predict_language = _prev_predict_language

    def _setup_streaming(self):
        """Set up streaming inference with vision caching.

        NOTE: This should be called BEFORE torch.compile to ensure the wrapper
        is properly integrated into the model graph.
        """
        if self._streaming_manager is not None:
            return

        from .streaming_kv import StreamingInferenceManager

        # Use lower similarity threshold for driving (0.6) since consecutive frames
        # change gradually but are still from same scene
        self._streaming_manager = StreamingInferenceManager(
            similarity_threshold=0.6,  # Lower threshold for driving footage
            max_cache_age=self.config.streaming_window_size,
            enable_caching=True,
        )

        # Set up vision caching wrapper
        self._setup_vision_hooks()

    def _setup_vision_hooks(self):
        """Set up hooks to measure and potentially cache vision encoder outputs.

        First, we measure how much time vision encoding takes to determine
        if caching is worthwhile.
        """
        if self._model is None or self._vision_hook_handle is not None:
            return

        parent = self
        self._vision_timing = []

        # Find the actual vision model by traversing the model
        vision_model = None
        vision_path = None

        # Print all top-level modules to understand structure
        if self.config.verbose:
            print(f"  Model structure: {type(self._model).__name__}", flush=True)
            for name, child in self._model.named_children():
                print(f"    - {name}: {type(child).__name__}", flush=True)
                # Also print nested structure for vision model
                if "vision" in name.lower():
                    for n2, c2 in child.named_children():
                        print(f"      - {name}.{n2}: {type(c2).__name__}", flush=True)

        # Check for deeply nested vision encoder
        # Structure: DrivingModel -> vision_model (VLMEncoderModel) -> image_encoder (LingoInternVLModel)
        # Hook at VLMEncoderModel level since that's what gets called
        if hasattr(self._model, "vision_model"):
            vision_model = self._model.vision_model
            vision_path = "vision_model"
        else:
            # Fallback: try direct attributes
            for attr_name in ["vision_model", "visual", "vit", "image_encoder", "img_encoder"]:
                if hasattr(self._model, attr_name):
                    vision_model = getattr(self._model, attr_name)
                    vision_path = attr_name
                    break

            # Try nested structure (e.g., model.model.vision_model)
            if vision_model is None:
                if hasattr(self._model, "model"):
                    nested = self._model.model
                    for attr_name in ["vision_model", "visual", "vit", "vision_tower"]:
                        if hasattr(nested, attr_name):
                            vision_model = getattr(nested, attr_name)
                            vision_path = f"model.{attr_name}"
                            break

        if vision_model is None:
            if self.config.verbose:
                print("  Could not find vision encoder for streaming", flush=True)
            return

        # Timing hooks
        self._vision_start_time = None

        def vision_pre_hook(module, input):
            """Record start time of vision encoding."""
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            parent._vision_start_time = time.perf_counter()
            return input

        def vision_post_hook(module, input, output):
            """Record end time and capture output."""
            # Always print to verify hook is being called
            print(f"    [Vision Hook] Called! Module type: {type(module).__name__}", flush=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if parent._vision_start_time is not None:
                elapsed = (time.perf_counter() - parent._vision_start_time) * 1000
                parent._vision_timing.append(elapsed)
                print(f"    [Vision] Encoding took {elapsed:.1f}ms, output shape: {output.shape}", flush=True)

            # Cache the output
            parent._cached_vision_output = output
            return output

        # Register hooks on both VLMEncoderModel AND its children
        handles = []

        # Hook the VLMEncoderModel
        handles.append(vision_model.register_forward_pre_hook(vision_pre_hook))
        handles.append(vision_model.register_forward_hook(vision_post_hook))

        # Also hook all immediate children to see what's being called
        for name, child in vision_model.named_children():
            def make_debug_hook(n):
                def hook(module, input, output):
                    print(f"    [Debug] {n} called, output shape: {getattr(output, 'shape', type(output))}", flush=True)
                    return output
                return hook
            handles.append(child.register_forward_hook(make_debug_hook(name)))

        self._vision_pre_hook = handles[0]
        self._vision_hook_handle = handles[1]

        if self.config.verbose:
            print(f"  Registered vision timing hooks on {vision_path} and {len(handles)-2} children", flush=True)

    def get_vision_timing(self) -> dict:
        """Get vision encoder timing statistics."""
        if not hasattr(self, '_vision_timing') or not self._vision_timing:
            return {"count": 0}
        import numpy as np
        arr = np.array(self._vision_timing)
        return {
            "count": len(arr),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "total_ms": float(arr.sum()),
        }

    def predict_stream(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        speed_mps: float,
        target_points: np.ndarray,
        use_cot: bool = True,
        reset_stream: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """Run streaming inference with vision embedding caching.

        In streaming mode, vision embeddings from previous frames are cached
        and reused when the scene is similar. This provides significant speedup
        for continuous driving scenarios.

        Args:
            image: Input image
            speed_mps: Current vehicle speed in m/s
            target_points: Target waypoints array
            use_cot: Use chain-of-thought prompting
            reset_stream: Reset the streaming state (new scene/route)

        Returns:
            Same as predict(): (waypoints, route, commentary)
        """
        if not self.config.enable_streaming:
            return self.predict(image, speed_mps, target_points, use_cot)

        # Ensure model is loaded and streaming is set up
        if self._model is None:
            self.load()
        self._setup_streaming()

        if reset_stream:
            self._streaming_manager.reset()
            self._cached_vision_output = None
            self._use_cached_vision = False

        t_total_start = time.perf_counter()

        # Convert image to numpy for hashing
        if isinstance(image, (str, Path)):
            import cv2
            img_np = cv2.imread(str(image))
            if img_np is not None:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image, bytes):
            import cv2
            img_np = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            if img_np is not None:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            img_np = None

        self._current_image_np = img_np

        # Check if we can use cached vision embeddings
        # Simple logic: use cache if it exists and we're not on frame 0
        cache_age = self._frame_count
        max_cache_age = self.config.streaming_window_size

        if self._cached_vision_output is not None and cache_age > 0 and cache_age < max_cache_age:
            # Use cached vision embeddings
            self._use_cached_vision = True
            if self.config.verbose and self._frame_count % 10 == 0:
                print(f"  [Stream] Frame {self._frame_count}: Using CACHED vision (age={cache_age})", flush=True)
        else:
            # Need to compute fresh vision embeddings
            self._use_cached_vision = False
            if cache_age >= max_cache_age:
                self._cached_vision_output = None  # Force refresh
            if self.config.verbose and self._frame_count % 10 == 0:
                reason = "no cache" if self._cached_vision_output is None else f"cache too old ({cache_age} >= {max_cache_age})"
                print(f"  [Stream] Frame {self._frame_count}: RECOMPUTING vision ({reason})", flush=True)

        # Preprocess image
        t_pre_start = time.perf_counter()
        pixel_values, HW = self._preprocessor(image, return_original_size=True)
        t_pre = (time.perf_counter() - t_pre_start) * 1000

        # Ensure target_points is correct shape
        target_points_np = np.asarray(target_points, dtype=np.float32)
        if target_points_np.ndim == 1:
            target_points_np = target_points_np.reshape(1, -1)
        if target_points_np.shape[0] == 1:
            target_points_np = np.vstack([target_points_np, target_points_np])

        # Build prompt and inputs
        prompt_text = self._build_prompt(speed_mps, use_cot=use_cot)
        prompt_ll, prompt_inf_ll = self._build_language_label(
            prompt_text, target_points_np
        )
        driving_input = self._build_driving_input(
            pixel_values,
            speed_mps=speed_mps,
            target_points_np=target_points_np,
            prompt_ll=prompt_ll,
            prompt_inference_ll=prompt_inf_ll,
            HW=HW,
        )

        # Run inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_inf_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self.dtype
        ):
            speed_wps, route_wps, language = self._model(driving_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_inf = (time.perf_counter() - t_inf_start) * 1000
        t_total = (time.perf_counter() - t_total_start) * 1000

        # Record timing
        self._timing_stats["preprocess_ms"].append(t_pre)
        self._timing_stats["inference_ms"].append(t_inf)
        self._timing_stats["total_ms"].append(t_total)

        # Advance frame counter
        self._streaming_manager.advance_frame()
        self._frame_count += 1

        # Clear state
        self._use_cached_vision = False
        self._current_image_np = None

        # Post-process outputs
        pred_wps = (
            speed_wps[0].float().cpu().numpy() if speed_wps is not None else None
        )
        pred_route = (
            self._equal_spacing_route(route_wps[0].float().cpu().numpy(), num=20)
            if route_wps is not None
            else None
        )
        pred_text = language[0] if language else ""

        return pred_wps, pred_route, pred_text

    def get_streaming_stats(self) -> dict:
        """Get streaming cache statistics.

        Returns:
            Dictionary with cache hit rate and other stats
        """
        if self._streaming_manager is None:
            return {"enabled": False}
        return {
            "enabled": True,
            **self._streaming_manager.get_stats(),
        }

    def get_timing_stats(self) -> dict:
        """Get timing statistics from recent predictions.

        Returns:
            Dictionary with timing statistics (mean, p50, p90 in ms)
        """
        stats = {}
        for key, values in self._timing_stats.items():
            if values:
                arr = np.array(values)
                stats[key] = {
                    "mean": float(arr.mean()),
                    "p50": float(np.median(arr)),
                    "p90": float(np.percentile(arr, 90)),
                    "count": len(values),
                }
        return stats

    def reset_timing_stats(self):
        """Reset timing statistics."""
        for key in self._timing_stats:
            self._timing_stats[key] = []

    def get_hz(self) -> float:
        """Get current inference rate in Hz.

        Returns:
            Inference rate based on mean total time
        """
        stats = self.get_timing_stats()
        if "total_ms" in stats and stats["total_ms"]["count"] > 0:
            return 1000.0 / stats["total_ms"]["mean"]
        return 0.0

    def forward(self, driving_input) -> Tuple[Any, Any, Any]:
        """Direct model forward pass.

        This bypasses preprocessing and is useful for benchmarking
        the model inference time separately.

        Args:
            driving_input: DrivingInput namedtuple

        Returns:
            Model outputs (waypoints, route, language)
        """
        if self._model is None:
            self.load()

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self.dtype
        ):
            return self._model(driving_input)

    def __repr__(self) -> str:
        return (
            f"OptimizedSimLingo(\n"
            f"  ckpt_path={self.ckpt_path},\n"
            f"  device={self.device},\n"
            f"  compiled={self._compiled},\n"
            f"  warmed_up={self._warmed_up},\n"
            f"  config={self.config.to_dict()}\n"
            f")"
        )
