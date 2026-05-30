"""Optimization configuration for SimLingo inference.

This module defines the OptimizationConfig dataclass which controls which
optimizations are applied during inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class QuantizationType(str, Enum):
    """Supported quantization modes."""

    NONE = "none"
    INT8 = "int8"  # INT8 weights and activations
    W4A8 = "w4a8"  # INT4 weights, INT8 activations (recommended)
    FP8 = "fp8"  # FP8 for Blackwell/Hopper GPUs


@dataclass
class OptimizationConfig:
    """Configuration for SimLingo inference optimizations.

    Attributes:
        use_torch_compile: Enable torch.compile with max-autotune mode.
            Provides 1.5-2x speedup after warmup. Requires PyTorch 2.0+.

        compile_mode: torch.compile mode. Options:
            - "default": Good balance of compile time and performance
            - "reduce-overhead": Faster compile, less optimization
            - "max-autotune": Slowest compile, maximum performance

        use_dynamic_shapes: Allow dynamic tensor shapes in compiled model.
            Required for variable batch sizes or image resolutions.

        use_gpu_preprocessing: Use GPU-accelerated image preprocessing.
            Replaces cv2/PIL with torchvision GPU transforms.
            Reduces preprocessing from 300-400ms to 30-50ms.

        use_static_kv_cache: Pre-allocate KV cache tensors to avoid
            runtime allocations. Reduces memory allocation overhead.

        kv_cache_max_length: Maximum sequence length for KV cache.
            Affects memory usage vs flexibility tradeoff.

        quantization: Quantization mode for model weights and activations.
            - "none": Full precision (bf16)
            - "int8": INT8 weights and activations
            - "w4a8": INT4 weights, INT8 activations (recommended)
            - "fp8": FP8 for Blackwell/Hopper GPUs

        quantize_vision_encoder: Whether to quantize the vision encoder.
            Vision encoders are compute-bound and benefit from INT8.

        quantize_language_head: Whether to quantize the language model head.
            LLM decoders are memory-bound and benefit from W4A8.

        keep_embedding_fp16: Keep embedding and output layers at full
            precision for better accuracy and fine-tuning compatibility.

        enable_streaming: Enable streaming inference mode for multi-frame
            processing. Reuses KV cache across frames.

        streaming_window_size: Number of frames to keep in streaming window.
            Each frame reuses (window_size-1)/window_size of computation.

        use_fused_kernels: Enable custom Triton kernels for fused operations.
            Includes fused Q/K/V projection and fused MLP gate/up.

        use_flash_attention: Use Flash Attention for attention computation.
            Requires flash-attn package. Enabled by default if available.

        enable_cuda_graphs: Use CUDA Graphs for autoregressive generation.
            Reduces kernel launch overhead for small batch sizes.

        tf32_mode: Enable TF32 precision for matmul operations.
            Provides 3x speedup on Ampere+ GPUs with minimal accuracy loss.

        cudnn_benchmark: Enable cuDNN benchmark mode for faster convolutions.
            Should be disabled for variable input sizes.

        device: Device to run inference on. "cuda" or "cpu".

        dtype: Default dtype for model weights and activations.
            "bfloat16" recommended for modern GPUs.

        verbose: Print optimization status and timing information.
    """

    # torch.compile settings
    use_torch_compile: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"
    use_dynamic_shapes: bool = True

    # Preprocessing
    use_gpu_preprocessing: bool = True

    # KV cache
    use_static_kv_cache: bool = True
    kv_cache_max_length: int = 2048

    # Quantization
    quantization: Literal["none", "int8", "w4a8", "fp8"] = "none"
    quantize_vision_encoder: bool = True
    quantize_language_head: bool = True
    keep_embedding_fp16: bool = True

    # Streaming
    enable_streaming: bool = False
    streaming_window_size: int = 4

    # Custom kernels
    use_fused_kernels: bool = False  # Requires triton
    use_flash_attention: bool = True

    # CUDA optimizations
    enable_cuda_graphs: bool = False
    tf32_mode: bool = True
    cudnn_benchmark: bool = True

    # Device settings
    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    # Debugging
    verbose: bool = False

    # Waypoint-only inference mode (highest ROI for CARLA driving)
    # When True (default), skip the autoregressive language/commentary head.
    # The upstream model exposes a `predict_language` attribute; when False the
    # forward pass takes the training code-path (single forward through the
    # vision+LLM stack producing waypoints via the regression adaptor) instead
    # of running the per-batch greedy_sample loop (up to 100 LLM steps per item
    # plus an extra concat-and-forward call to obtain driving features).
    # CARLA's PID controller only consumes speed_wps and route_wps, so the
    # commentary string is unused at deployment time.
    waypoints_only: bool = True

    # Model paths (can be overridden)
    hydra_config_name: str = "simlingo"  # Upstream hydra config name

    # InternVL2 specific
    image_size: int = 448
    max_image_patches: int = 2
    use_global_img: bool = True

    # Camera defaults (CARLA training)
    default_fov_deg: float = 110.0
    crop_bottom: bool = True  # Crop bonnet for CARLA images

    def __post_init__(self):
        """Validate configuration."""
        if self.quantization == "fp8" and self.device == "cpu":
            raise ValueError("FP8 quantization requires CUDA device")

        if self.enable_cuda_graphs and self.use_dynamic_shapes:
            # CUDA graphs require static shapes
            raise ValueError(
                "CUDA graphs require static shapes. "
                "Set use_dynamic_shapes=False or enable_cuda_graphs=False"
            )

    def get_torch_dtype(self):
        """Get torch dtype from string."""
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map[self.dtype]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "use_torch_compile": self.use_torch_compile,
            "compile_mode": self.compile_mode,
            "use_dynamic_shapes": self.use_dynamic_shapes,
            "use_gpu_preprocessing": self.use_gpu_preprocessing,
            "use_static_kv_cache": self.use_static_kv_cache,
            "kv_cache_max_length": self.kv_cache_max_length,
            "quantization": self.quantization,
            "quantize_vision_encoder": self.quantize_vision_encoder,
            "quantize_language_head": self.quantize_language_head,
            "keep_embedding_fp16": self.keep_embedding_fp16,
            "enable_streaming": self.enable_streaming,
            "streaming_window_size": self.streaming_window_size,
            "use_fused_kernels": self.use_fused_kernels,
            "use_flash_attention": self.use_flash_attention,
            "enable_cuda_graphs": self.enable_cuda_graphs,
            "tf32_mode": self.tf32_mode,
            "cudnn_benchmark": self.cudnn_benchmark,
            "device": self.device,
            "dtype": self.dtype,
            "waypoints_only": self.waypoints_only,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizationConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def fast_compile(cls) -> "OptimizationConfig":
        """Preset for fast compilation (Phase 1 optimizations)."""
        return cls(
            use_torch_compile=True,
            compile_mode="reduce-overhead",
            use_gpu_preprocessing=True,
            quantization="none",
            enable_streaming=False,
            use_fused_kernels=False,
        )

    @classmethod
    def quantized(cls) -> "OptimizationConfig":
        """Preset with W4A8 quantization (Phase 1+2 optimizations)."""
        return cls(
            use_torch_compile=True,
            compile_mode="max-autotune",
            use_gpu_preprocessing=True,
            quantization="w4a8",
            enable_streaming=False,
            use_fused_kernels=False,
        )

    @classmethod
    def full_optimized(cls) -> "OptimizationConfig":
        """Preset with all optimizations (Phase 1-4)."""
        return cls(
            use_torch_compile=True,
            compile_mode="max-autotune",
            use_gpu_preprocessing=True,
            quantization="w4a8",
            enable_streaming=True,
            streaming_window_size=4,
            use_fused_kernels=True,
            use_flash_attention=True,
            tf32_mode=True,
            cudnn_benchmark=True,
        )

    @classmethod
    def agx_thor(cls) -> "OptimizationConfig":
        """Preset optimized for NVIDIA AGX Thor (Blackwell GPU)."""
        return cls(
            use_torch_compile=True,
            compile_mode="max-autotune",
            use_gpu_preprocessing=True,
            quantization="fp8",  # FP8 for Blackwell
            enable_streaming=True,
            streaming_window_size=4,
            use_fused_kernels=True,
            use_flash_attention=True,
            enable_cuda_graphs=True,
            use_dynamic_shapes=False,  # Required for CUDA graphs
            tf32_mode=True,
            cudnn_benchmark=True,
        )
