"""Custom Triton kernels for fused operations.

This module provides custom Triton kernels that fuse multiple operations
to reduce memory bandwidth and kernel launch overhead:

- Fused Q/K/V projection: Single kernel for all attention projections
- Fused MLP gate/up: SiLU(X @ W_gate) * (X @ W_up) in one kernel

Expected speedup: 1.5-2x for attention-heavy models

Requirements:
- NVIDIA GPU with compute capability >= 7.0
- Triton library (pip install triton)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

# Check for Triton availability
_TRITON_AVAILABLE = False
try:
    import triton

    _TRITON_AVAILABLE = True
except ImportError:
    pass


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return _TRITON_AVAILABLE


def apply_fused_kernels(model: nn.Module) -> nn.Module:
    """Apply custom fused kernels to the model.

    This function walks the model and replaces standard PyTorch operations
    with fused Triton implementations where beneficial.

    Args:
        model: Model to optimize

    Returns:
        Model with fused kernels applied

    Raises:
        ImportError: If Triton is not available
    """
    if not _TRITON_AVAILABLE:
        raise ImportError(
            "Triton is required for fused kernels. "
            "Install with: pip install triton"
        )

    from .fused_attention import replace_attention_with_fused
    from .fused_mlp import replace_mlp_with_fused

    # Apply fused attention (Q/K/V projection)
    model = replace_attention_with_fused(model)

    # Apply fused MLP (gate/up projection)
    model = replace_mlp_with_fused(model)

    return model


if TYPE_CHECKING:
    from .fused_attention import FusedQKVProjection, fused_qkv_kernel
    from .fused_mlp import FusedGatedMLP, fused_gated_mlp_kernel

__all__ = [
    "is_triton_available",
    "apply_fused_kernels",
]
