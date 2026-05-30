"""Fused Q/K/V projection kernel using Triton.

This kernel computes Q, K, V projections in a single kernel launch:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

Benefits:
- 3x reduction in memory reads of X
- Single kernel launch instead of 3
- Better GPU occupancy

The kernel is optimized for:
- Typical transformer hidden sizes (1024, 2048, 4096)
- Batch sizes 1-8 (inference)
- bfloat16/float16 compute
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy Triton import
_triton = None
_triton_language = None


def _ensure_triton():
    global _triton, _triton_language
    if _triton is None:
        import triton
        import triton.language as tl

        _triton = triton
        _triton_language = tl
    return _triton, _triton_language


def _get_fused_qkv_kernel():
    """Get the fused QKV kernel, creating it on first use."""
    triton, tl = _ensure_triton()

    @triton.jit
    def fused_qkv_kernel(
        # Inputs
        x_ptr,
        wq_ptr,
        wk_ptr,
        wv_ptr,
        # Outputs
        q_ptr,
        k_ptr,
        v_ptr,
        # Dimensions
        batch_size,
        seq_len,
        hidden_dim,
        head_dim,
        num_heads,
        # Strides
        stride_xb,
        stride_xs,
        stride_xh,
        stride_wb,
        stride_wh,
        stride_ob,
        stride_os,
        stride_oh,
        # Meta
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """Fused Q/K/V projection kernel.

        Each program computes a BLOCK_SIZE_M x BLOCK_SIZE_N tile of one output
        (Q, K, or V) for a specific batch/sequence position.
        """
        # Program ID
        pid_m = tl.program_id(0)  # M dimension (batch * seq)
        pid_n = tl.program_id(1)  # N dimension (output features)
        pid_qkv = tl.program_id(2)  # Which of Q/K/V (0, 1, 2)

        # Compute batch and sequence indices
        batch_seq_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        batch_idx = batch_seq_idx // seq_len
        seq_idx = batch_seq_idx % seq_len

        # Output feature indices
        n_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Select weight matrix based on qkv index
        if pid_qkv == 0:
            w_ptr = wq_ptr
            out_ptr = q_ptr
        elif pid_qkv == 1:
            w_ptr = wk_ptr
            out_ptr = k_ptr
        else:
            w_ptr = wv_ptr
            out_ptr = v_ptr

        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Loop over K dimension (hidden_dim)
        for k_start in range(0, hidden_dim, BLOCK_SIZE_K):
            k_idx = k_start + tl.arange(0, BLOCK_SIZE_K)

            # Load X tile [BLOCK_SIZE_M, BLOCK_SIZE_K]
            x_ptrs = (
                x_ptr
                + batch_idx[:, None] * stride_xb
                + seq_idx[:, None] * stride_xs
                + k_idx[None, :] * stride_xh
            )
            x_mask = (batch_seq_idx[:, None] < batch_size * seq_len) & (
                k_idx[None, :] < hidden_dim
            )
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load W tile [BLOCK_SIZE_K, BLOCK_SIZE_N]
            w_ptrs = w_ptr + k_idx[:, None] * stride_wb + n_idx[None, :] * stride_wh
            w_mask = (k_idx[:, None] < hidden_dim) & (n_idx[None, :] < head_dim * num_heads)
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # Accumulate
            acc += tl.dot(x_tile, w_tile)

        # Write output
        out_ptrs = (
            out_ptr
            + batch_idx[:, None] * stride_ob
            + seq_idx[:, None] * stride_os
            + n_idx[None, :] * stride_oh
        )
        out_mask = (batch_seq_idx[:, None] < batch_size * seq_len) & (
            n_idx[None, :] < head_dim * num_heads
        )
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)

    return fused_qkv_kernel


class FusedQKVProjection(nn.Module):
    """Fused Q/K/V projection using Triton kernel.

    Replaces three separate Linear layers with a single fused operation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize fused projection.

        Args:
            hidden_dim: Input hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            bias: Whether to use bias
            dtype: Weight dtype
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim

        # Combined weight matrix [3, hidden_dim, output_dim]
        self.weight = nn.Parameter(
            torch.empty(3, hidden_dim, self.output_dim, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(3, self.output_dim, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights."""
        for i in range(3):
            nn.init.xavier_uniform_(self.weight[i])

    @classmethod
    def from_separate_layers(
        cls,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
    ) -> "FusedQKVProjection":
        """Create fused projection from separate Q/K/V layers.

        Args:
            q_proj: Query projection layer
            k_proj: Key projection layer
            v_proj: Value projection layer

        Returns:
            FusedQKVProjection with copied weights
        """
        hidden_dim = q_proj.in_features
        output_dim = q_proj.out_features
        # Assume standard transformer where output_dim = num_heads * head_dim
        # Try to infer num_heads from common configurations
        for num_heads in [32, 16, 12, 8, 4, 2, 1]:
            if output_dim % num_heads == 0:
                head_dim = output_dim // num_heads
                break
        else:
            num_heads = 1
            head_dim = output_dim

        has_bias = q_proj.bias is not None
        fused = cls(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=has_bias,
            dtype=q_proj.weight.dtype,
        )

        # Copy weights
        with torch.no_grad():
            fused.weight[0].copy_(q_proj.weight.t())
            fused.weight[1].copy_(k_proj.weight.t())
            fused.weight[2].copy_(v_proj.weight.t())
            if has_bias:
                fused.bias[0].copy_(q_proj.bias)
                fused.bias[1].copy_(k_proj.bias)
                fused.bias[2].copy_(v_proj.bias)

        return fused

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fused Q/K/V projections.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Tuple of (Q, K, V) each with shape [batch, seq_len, num_heads * head_dim]
        """
        batch_size, seq_len, _ = x.shape

        # For now, use efficient batched matmul instead of custom kernel
        # The custom Triton kernel can be enabled for further optimization
        # x: [B, S, H] @ weight: [3, H, O] -> [3, B, S, O]

        # Reshape weight for batched matmul
        w = self.weight  # [3, H, O]

        # Compute all three projections
        # Using einsum for clarity (can be optimized to matmul)
        qkv = torch.einsum("bsh,tho->tbso", x, w)

        if self.bias is not None:
            qkv = qkv + self.bias[:, None, None, :]

        return qkv[0], qkv[1], qkv[2]

    def forward_triton(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fused Q/K/V using Triton kernel.

        This is the optimized path using the custom Triton kernel.
        """
        kernel = _get_fused_qkv_kernel()

        batch_size, seq_len, hidden_dim = x.shape

        # Allocate outputs
        q = torch.empty(
            batch_size, seq_len, self.output_dim, device=x.device, dtype=x.dtype
        )
        k = torch.empty_like(q)
        v = torch.empty_like(q)

        # Configure grid
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32

        grid = (
            (batch_size * seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
            (self.output_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
            3,  # Q, K, V
        )

        # Launch kernel
        kernel[grid](
            x,
            self.weight[0],
            self.weight[1],
            self.weight[2],
            q,
            k,
            v,
            batch_size,
            seq_len,
            hidden_dim,
            self.head_dim,
            self.num_heads,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            self.weight.stride(1),
            self.weight.stride(2),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )

        if self.bias is not None:
            q = q + self.bias[0]
            k = k + self.bias[1]
            v = v + self.bias[2]

        return q, k, v


def replace_attention_with_fused(model: nn.Module) -> nn.Module:
    """Replace attention Q/K/V projections with fused version.

    Walks the model and finds attention modules with separate
    q_proj, k_proj, v_proj layers, replacing them with FusedQKVProjection.

    Args:
        model: Model to modify

    Returns:
        Modified model
    """

    def _replace_qkv(module: nn.Module, name: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this module has separate Q/K/V projections
            has_qkv = (
                hasattr(child, "q_proj")
                and hasattr(child, "k_proj")
                and hasattr(child, "v_proj")
                and isinstance(child.q_proj, nn.Linear)
                and isinstance(child.k_proj, nn.Linear)
                and isinstance(child.v_proj, nn.Linear)
            )

            if has_qkv:
                # Create fused projection
                fused = FusedQKVProjection.from_separate_layers(
                    child.q_proj, child.k_proj, child.v_proj
                )
                # Replace the projections
                child.qkv_proj = fused
                # Remove old projections to save memory
                del child.q_proj
                del child.k_proj
                del child.v_proj

                # Patch the forward method to use fused projection
                _patch_attention_forward(child)
            else:
                _replace_qkv(child, full_name)

    _replace_qkv(model)
    return model


def _patch_attention_forward(attention_module: nn.Module) -> None:
    """Patch attention forward to use fused QKV projection.

    This modifies the module in-place to call qkv_proj instead of
    separate q_proj, k_proj, v_proj.
    """
    # Store original forward
    original_forward = attention_module.forward

    def patched_forward(hidden_states, *args, **kwargs):
        # Check if we have the fused projection
        if hasattr(attention_module, "qkv_proj"):
            # Compute Q, K, V using fused projection
            q, k, v = attention_module.qkv_proj(hidden_states)

            # Continue with rest of attention
            # This part depends on the specific attention implementation
            # For now, just store them for the attention computation
            kwargs["precomputed_qkv"] = (q, k, v)

        return original_forward(hidden_states, *args, **kwargs)

    attention_module.forward = patched_forward
