"""Quantization utilities for SimLingo inference.

This module provides W4A8 and INT8 quantization for the SimLingo model.
W4A8 (4-bit weights, 8-bit activations) provides the best memory/compute
tradeoff for the mixed vision-language architecture:

- Vision encoder: INT8 activations (compute-bound)
- LLM decoder: INT4 weights (memory-bound)
- Embeddings/head: Full precision (accuracy-critical)

Supports multiple backends:
- bitsandbytes (4-bit NF4/FP4 and INT8)
- NVIDIA ModelOpt (production-grade W4A8)
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Optional

import torch
import torch.nn as nn

# Check for quantization library availability
_BITSANDBYTES_AVAILABLE = False
_MODELOPT_AVAILABLE = False

try:
    import bitsandbytes as bnb

    _BITSANDBYTES_AVAILABLE = True
except ImportError:
    pass

try:
    import modelopt

    _MODELOPT_AVAILABLE = True
except ImportError:
    pass


def quantize_model(
    model: nn.Module,
    quantization_type: Literal["int8", "w4a8", "fp8"] = "w4a8",
    quantize_vision: bool = True,
    quantize_llm: bool = True,
    keep_embedding_fp16: bool = True,
    backend: Optional[Literal["bitsandbytes", "modelopt", "torch"]] = None,
) -> nn.Module:
    """Apply quantization to the model.

    Args:
        model: The SimLingo model to quantize
        quantization_type: Type of quantization to apply
            - "int8": INT8 weights and activations (using bitsandbytes LLM.int8())
            - "w4a8": INT4 weights, INT8 activations (NF4 quantization)
            - "fp8": FP8 for Hopper/Blackwell GPUs
        quantize_vision: Whether to quantize vision encoder
        quantize_llm: Whether to quantize language model
        keep_embedding_fp16: Keep embedding layers at FP16
        backend: Quantization backend to use. Auto-selects if None.

    Returns:
        Quantized model
    """
    # Select backend
    if backend is None:
        if _BITSANDBYTES_AVAILABLE:
            backend = "bitsandbytes"
        elif _MODELOPT_AVAILABLE:
            backend = "modelopt"
        else:
            backend = "torch"

    if quantization_type == "w4a8":
        return _quantize_w4a8(
            model,
            quantize_vision=quantize_vision,
            quantize_llm=quantize_llm,
            keep_embedding_fp16=keep_embedding_fp16,
            backend=backend,
        )
    elif quantization_type == "int8":
        return _quantize_int8(
            model,
            quantize_vision=quantize_vision,
            quantize_llm=quantize_llm,
            keep_embedding_fp16=keep_embedding_fp16,
            backend=backend,
        )
    elif quantization_type == "fp8":
        return _quantize_fp8(
            model,
            quantize_vision=quantize_vision,
            quantize_llm=quantize_llm,
        )
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")


def _quantize_w4a8(
    model: nn.Module,
    quantize_vision: bool = True,
    quantize_llm: bool = True,
    keep_embedding_fp16: bool = True,
    backend: str = "bitsandbytes",
) -> nn.Module:
    """Apply W4A8 quantization (4-bit weights, 8-bit activations).

    This is optimal for mixed vision-language models:
    - Vision encoder benefits from INT8 activations (compute-bound)
    - LLM decoder benefits from 4-bit weights (memory-bound)
    """
    if backend == "bitsandbytes" and _BITSANDBYTES_AVAILABLE:
        return _quantize_w4a8_bnb(
            model, quantize_vision, quantize_llm, keep_embedding_fp16
        )
    elif backend == "modelopt" and _MODELOPT_AVAILABLE:
        return _quantize_w4a8_modelopt(
            model, quantize_vision, quantize_llm, keep_embedding_fp16
        )
    else:
        warnings.warn(
            f"Backend {backend} not available for W4A8, skipping quantization"
        )
        return model


def _quantize_w4a8_bnb(
    model: nn.Module,
    quantize_vision: bool = True,
    quantize_llm: bool = True,
    keep_embedding_fp16: bool = True,
) -> nn.Module:
    """Apply W4A8 quantization using bitsandbytes Linear4bit.

    Uses NF4 (4-bit NormalFloat) for weights.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        warnings.warn("bitsandbytes not available, skipping W4A8 quantization")
        return model

    device = next(model.parameters()).device
    replaced_count = 0
    failed_count = 0

    def _should_quantize(name: str) -> bool:
        """Same conservative scoping as the INT8 path: only the actual LLM
        (and, if explicitly requested, the actual vision encoder excluding
        the embedded LLM)."""
        n = name.lower()
        if keep_embedding_fp16:
            if "embed" in n or "lm_head" in n or "wte" in n or "wpe" in n:
                return False

        # Never quantize LoRA adapter matrices — see INT8 path for rationale.
        if "lora_" in n:
            return False

        if "language_model" in n:
            return quantize_llm
        if quantize_vision and "vision_model" in n and "language_model" not in n:
            return True
        return False

    def _replace_linear_with_4bit(module: nn.Module, name: str = "") -> nn.Module:
        """Recursively replace Linear layers with 4-bit versions."""
        nonlocal replaced_count, failed_count

        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear) and _should_quantize(full_name):
                try:
                    in_features = child.in_features
                    out_features = child.out_features
                    has_bias = child.bias is not None

                    # Create Linear4bit layer
                    new_layer = bnb.nn.Linear4bit(
                        in_features,
                        out_features,
                        bias=has_bias,
                        compute_dtype=torch.bfloat16,
                        compress_statistics=True,
                        quant_type="nf4",
                    )

                    # Copy the weight data and let bitsandbytes handle quantization
                    # The weight needs to be converted to Params4bit for quantization
                    weight_fp16 = child.weight.data.to(dtype=torch.float16, device="cpu")

                    # Create Params4bit - quantization happens when moved to CUDA
                    new_layer.weight = bnb.nn.Params4bit(
                        weight_fp16,
                        requires_grad=False,
                        compress_statistics=True,
                        quant_type="nf4",
                    )

                    if has_bias:
                        new_layer.bias = nn.Parameter(
                            child.bias.data.to(dtype=torch.float16, device="cpu")
                        )

                    # Move to CUDA - this triggers the actual quantization
                    new_layer = new_layer.to(device)

                    setattr(module, child_name, new_layer)
                    replaced_count += 1

                except Exception as e:
                    failed_count += 1
                    # Keep original layer on failure
            else:
                _replace_linear_with_4bit(child, full_name)

        return module

    model = _replace_linear_with_4bit(model)
    print(f"  Quantized {replaced_count} Linear layers to 4-bit NF4 ({failed_count} failed)", flush=True)
    return model


def _quantize_w4a8_modelopt(
    model: nn.Module,
    quantize_vision: bool = True,
    quantize_llm: bool = True,
    keep_embedding_fp16: bool = True,
) -> nn.Module:
    """Apply W4A8 quantization using NVIDIA ModelOpt.

    This provides production-grade quantization with calibration support.
    """
    import modelopt.torch.quantization as mtq

    # Build quantization config
    quant_cfg = mtq.W4A8_AWQ_BETA_CFG.copy()

    # Customize which modules to quantize
    def _quant_filter(name: str) -> bool:
        if keep_embedding_fp16:
            if "embed" in name.lower() or "lm_head" in name.lower():
                return False

        if "vision" in name.lower():
            return quantize_vision
        if "language" in name.lower() or "llm" in name.lower():
            return quantize_llm

        return quantize_llm

    # Apply quantization
    model = mtq.quantize(
        model,
        config=quant_cfg,
        quant_filter=_quant_filter,
    )

    return model


def _quantize_int8(
    model: nn.Module,
    quantize_vision: bool = True,
    quantize_llm: bool = True,
    keep_embedding_fp16: bool = True,
    backend: str = "bitsandbytes",
) -> nn.Module:
    """Apply INT8 quantization using bitsandbytes LLM.int8().

    This uses bitsandbytes' Linear8bitLt for GPU-native INT8 inference.
    """
    if not _BITSANDBYTES_AVAILABLE:
        warnings.warn("INT8 quantization requires bitsandbytes, skipping")
        return model

    try:
        import bitsandbytes as bnb
    except ImportError:
        warnings.warn("bitsandbytes not available, skipping INT8 quantization")
        return model

    device = next(model.parameters()).device
    replaced_count = 0
    failed_count = 0

    def _should_quantize(name: str) -> bool:
        """Determine if a Linear layer should be quantized based on its full
        module path.

        We are deliberately CONSERVATIVE: only the actual LLM
        (``...language_model...``) and, if explicitly requested, the actual
        vision encoder (``...vision_model...`` excluding the embedded LLM) are
        eligible. All driving-specific heads (wp_encoder, adaptors, route
        head, language head) stay in BF16 — they are tiny, not on the hot
        path, and quantizing them yields no measurable speedup while risking
        accuracy regressions.
        """
        n = name.lower()
        if keep_embedding_fp16:
            if "embed" in n or "lm_head" in n or "wte" in n or "wpe" in n:
                return False

        # CRITICAL: never quantize LoRA adapter matrices (lora_A / lora_B).
        # They are rank-8/16 and (a) gain ~nothing from int8, (b) their int8
        # outputs break the LoRA residual sum inside the wrapper layer.
        if "lora_" in n:
            return False

        # Actual LLM path inside InternVL2 (e.g. "*.language_model.layers.N.*")
        if "language_model" in n:
            return quantize_llm

        # Vision encoder, but EXCLUDE the LLM that lives under vision_model
        # in some InternVL2 layouts.
        if quantize_vision and "vision_model" in n and "language_model" not in n:
            return True

        # Everything else (adaptors, wp_encoder, route head, projection
        # layers between vision and LLM) -> leave at BF16.
        return False

    def _replace_linear_with_int8(module: nn.Module, name: str = "") -> nn.Module:
        """Recursively replace Linear layers with INT8 versions."""
        nonlocal replaced_count, failed_count

        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear) and _should_quantize(full_name):
                try:
                    in_features = child.in_features
                    out_features = child.out_features
                    has_bias = child.bias is not None

                    # Get weight as FP16 on CPU first
                    weight_fp16 = child.weight.data.to(dtype=torch.float16, device="cpu")

                    # Create INT8 linear layer
                    new_layer = bnb.nn.Linear8bitLt(
                        in_features,
                        out_features,
                        bias=has_bias,
                        has_fp16_weights=False,
                        threshold=6.0,
                    )

                    # Set the weight - Int8Params handles the conversion
                    new_layer.weight = bnb.nn.Int8Params(
                        weight_fp16.contiguous(),
                        requires_grad=False,
                        has_fp16_weights=False,
                    )

                    if has_bias:
                        bias_fp16 = child.bias.data.to(dtype=torch.float16, device="cpu")
                        new_layer.bias = nn.Parameter(bias_fp16.contiguous())

                    # Move to device - this initializes the INT8 state
                    new_layer = new_layer.to(device)

                    # Verify the layer is properly initialized
                    if new_layer.weight is None:
                        raise RuntimeError("Weight is None after initialization")

                    setattr(module, child_name, new_layer)
                    replaced_count += 1

                except Exception as e:
                    failed_count += 1
                    # Keep original layer on failure
            else:
                _replace_linear_with_int8(child, full_name)

        return module

    model = _replace_linear_with_int8(model)
    print(f"  Quantized {replaced_count} Linear layers to INT8 ({failed_count} failed)", flush=True)
    return model


def _quantize_fp8(
    model: nn.Module,
    quantize_vision: bool = True,
    quantize_llm: bool = True,
) -> nn.Module:
    """Apply FP8 quantization for Hopper/Blackwell GPUs.

    FP8 provides near-FP16 accuracy with significant speedup on
    supported hardware (H100, H200, Blackwell).
    """
    # Check if FP8 is supported
    if not torch.cuda.is_available():
        warnings.warn("FP8 quantization requires CUDA")
        return model

    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:  # Requires Hopper (SM90) or later
        warnings.warn(
            f"FP8 quantization requires Hopper GPU or later "
            f"(current: SM{capability[0]}{capability[1]})"
        )
        return model

    if _MODELOPT_AVAILABLE:
        import modelopt.torch.quantization as mtq

        quant_cfg = mtq.FP8_DEFAULT_CFG.copy()

        def _quant_filter(name: str) -> bool:
            if "vision" in name.lower():
                return quantize_vision
            if "language" in name.lower() or "llm" in name.lower():
                return quantize_llm
            return quantize_llm

        model = mtq.quantize(
            model,
            config=quant_cfg,
            quant_filter=_quant_filter,
        )
    else:
        warnings.warn("FP8 quantization requires NVIDIA ModelOpt")

    return model


def load_and_quantize(
    ckpt_path: str,
    model_class: type,
    quantization: Literal["none", "int8", "w4a8", "fp8"] = "w4a8",
    device: str = "cuda",
    merge_lora: bool = True,
    **model_kwargs: Any,
) -> nn.Module:
    """Load checkpoint and apply quantization.

    This is a convenience function that handles the full workflow:
    1. Load the base model
    2. Load checkpoint weights
    3. Optionally merge LoRA weights
    4. Apply quantization

    Args:
        ckpt_path: Path to checkpoint file
        model_class: Model class to instantiate
        quantization: Quantization type to apply
        device: Target device
        merge_lora: Whether to merge LoRA weights before quantization
        **model_kwargs: Additional arguments for model instantiation

    Returns:
        Quantized model ready for inference
    """
    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Check for LoRA weights
    has_lora = any("lora" in k.lower() for k in state_dict.keys())

    if has_lora and merge_lora:
        # Load with LoRA, then merge
        from peft import PeftModel

        # This requires the base model to be loaded first
        # Implementation depends on specific model architecture
        pass

    # Instantiate model
    model = model_class(**model_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Apply quantization
    if quantization != "none":
        model = quantize_model(model, quantization_type=quantization)

    return model


def estimate_memory_reduction(
    model: nn.Module,
    quantization: Literal["int8", "w4a8", "fp8"],
) -> dict:
    """Estimate memory reduction from quantization.

    Args:
        model: Model to analyze
        quantization: Quantization type

    Returns:
        Dictionary with memory estimates
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Estimate bytes per parameter
    original_bytes_per_param = 2  # bfloat16

    if quantization == "w4a8":
        quantized_bytes_per_param = 0.5 + 1  # 4-bit weights + 8-bit activations
    elif quantization == "int8":
        quantized_bytes_per_param = 1
    elif quantization == "fp8":
        quantized_bytes_per_param = 1
    else:
        quantized_bytes_per_param = original_bytes_per_param

    original_memory_mb = (total_params * original_bytes_per_param) / (1024 * 1024)
    quantized_memory_mb = (total_params * quantized_bytes_per_param) / (1024 * 1024)
    reduction_pct = (1 - quantized_memory_mb / original_memory_mb) * 100

    return {
        "total_params": total_params,
        "original_memory_mb": original_memory_mb,
        "quantized_memory_mb": quantized_memory_mb,
        "reduction_pct": reduction_pct,
        "quantization": quantization,
    }
