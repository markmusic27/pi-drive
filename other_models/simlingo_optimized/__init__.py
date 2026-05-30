"""SimLingo Optimized Inference Module.

This module provides an optimized inference pipeline for SimLingo, targeting
5-8 Hz on NVIDIA AGX Thor (compared to ~0.55 Hz baseline on L4).

Key optimizations:
- torch.compile with max-autotune mode
- GPU-accelerated preprocessing (torchvision transforms)
- W4A8 quantization (INT4 weights, INT8 activations)
- Custom Triton kernels for fused attention and MLP
- Streaming KV cache for multi-frame inference
- TensorRT export for production deployment

Usage:
    from simlingo_optimized import OptimizedSimLingo, OptimizationConfig

    config = OptimizationConfig(
        use_torch_compile=True,
        use_gpu_preprocessing=True,
        quantization="w4a8",
        enable_streaming=True,
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

from .config import OptimizationConfig
from .inference import OptimizedSimLingo

__all__ = [
    "OptimizationConfig",
    "OptimizedSimLingo",
]
