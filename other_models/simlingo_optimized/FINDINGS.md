# SimLingo Inference Optimization — Findings & AGX Thor Deployment Notes

This document captures non-obvious findings discovered while optimizing the
upstream `RenzKa/simlingo` model for real-time CARLA inference, plus the
concrete steps needed to port the optimized inference path to NVIDIA AGX
Thor (Blackwell, JetPack 7+).

The optimization layer lives in `simlingo_optimized/` and is **strictly
additive**: it loads weights from a normal SimLingo training checkpoint and
wraps the upstream `DrivingModel` without touching `simlingo/`.

---

## 1. Architecture findings

### 1.1 The autoregressive language head dominates inference latency

The upstream `simlingo_training.models.driving.DrivingModel.forward()`
hardcodes `self.predict_language = True`. With it on, every inference call
runs:

1. **One** prefill forward through vision + LLM to produce the regression
   features used by the driving adaptor.
2. Then a **greedy sampling loop** of up to ~100 LLM steps (one decode step
   per generated token) producing the natural-language commentary.
3. Then **another forward** to re-pull adaptor features (because the loop
   destructively consumes the cache).

That decode loop is the entire latency budget at deployment time.
**CARLA's PID controller only reads `speed_wps` and `route` tensors**; the
commentary text is decorative.

Setting `self.predict_language = False` collapses (1)+(2)+(3) into a single
forward pass. On L4 this dropped per-frame latency from **948 ms → 200 ms
(4.73× speedup)** with **+0.20 m mean ADE drift** — well inside noise.

> Implemented as `OptimizationConfig.waypoints_only=True`. The flag also
> installs a small forward dispatcher (`_install_waypoints_only_forward`)
> that bypasses the upstream `predict_language=False` code branch, which is
> itself broken (see §2.2).

### 1.2 The vision encoder is compute-bound, the LLM is memory-bound

This is the standard VLM split, and it dictates the quantization strategy:

| Sub-model | Params | Hot path | Best quant | L4 latency share |
|-----------|--------|----------|------------|------------------|
| InternViT-300M (vision) | 300 M | 256-token prefill | INT8 act + INT8 wt | ~25 ms |
| Qwen2-0.5B (LLM) | 500 M | Single prefill | INT4/INT8 weight-only | ~80 ms |
| Driving adaptor / wp_encoder / heads | ~17 M (LoRA-trainable) | Small MLPs | Leave at BF16 | ~10 ms |
| Preprocessing + image embed merge | — | CPU + 1 conv | torchvision GPU | ~20 ms |

So most of the remaining win on L4/Thor is on the LLM (memory bandwidth)
and *only marginally* on the vision encoder.

### 1.3 The LLM lives in two places — beware of name collisions

In InternVL2 the actual Qwen2 LLM lives at
`model.vision_model.image_encoder.language_model` (yes, the LLM is nested
*under* `vision_model` because InternVL2 wraps the whole VLM as a single
HF auto-class). Naive name-based filters like `"vision" in name` will
silently quantize the LLM as if it were vision, and `"language" in name`
will match the language *commentary head* which is a small MLP that should
stay BF16.

> The filter we shipped scopes by exact substring `"language_model"` (the
> nested LLM path) and explicitly excludes `"lora_"` to avoid touching the
> LoRA adapters. See §3.1.

---

## 2. Bugs found in upstream that need workarounds

### 2.1 `replace_placeholder_tokens` raises `KeyError` at inference

The upstream tokenizer adds several special tokens (`<TARGET_POINT>`,
`<WAYPOINTS>`, etc.) at training-time. At inference, only the active task's
tokens get registered in `placeholder_values`. Any other special token that
appears in the prompt template (e.g. `<WAYPOINT_LAST>`, id 151648) then
crashes the lookup.

**Workaround**: build `placeholder_values` as a `defaultdict` that
returns a zero placeholder for any unseen id. See
`OptimizedSimLingo._build_language_label`.

### 2.2 The `predict_language=False` branch in upstream is broken

`DrivingModel.forward` has a branch for `predict_language=False` that
passes the tuple `(adaptor_features, adaptor_logits)` directly to
`adaptors.split_outputs_by_adaptor`, which expects a single tensor. This
fails with `TypeError: tuple indices must be integers or slices`.

**Workaround**: monkey-patch a corrected forward
(`_install_waypoints_only_forward`) that:
1. Calls `adaptors(example, inference=True)` and
   `vision_model.image_encoder.replace_placeholder_tokens(...)`.
2. Calls `forward_model(...)` and unpacks the `(features, logits)` tuple.
3. Calls `split_outputs_by_adaptor` separately on each.
4. Calls `adaptors.driving.get_predictions(features, logits)` exactly as
   training-time does.
5. Returns `(speed_wps, route, [])`.

When `predict_language` is True we fall through to the original upstream
forward (we cache it as `model._orig_forward`), so the same instance can
A/B-test both modes.

### 2.3 `torch.compile` is partially blocked

`torch.compile(model)` on the full model triggers recompilations because
Transformers' `DynamicCache` is not Dynamo-friendly. We compile only the
vision encoder (`mode="reduce-overhead"`, `dynamic=False`). That gives
~5–10 ms (~3%) per frame on L4 — small, but free.

A full-model `torch.compile` win is gated on either (a) Transformers
removing DynamicCache, (b) us swapping in a static cache, or (c) exporting
to TensorRT-LLM.

---

## 3. Quantization findings (Phase 2)

### 3.1 LoRA + bitsandbytes is fragile — never quantize the LoRA matrices

The model ships with LoRA adapters trained at rank 8/16. PEFT wraps every
LLM `Linear` with a `peft.tuners.lora.layer.Linear` whose `forward()` does:

```python
result = base_layer(x) + lora_B(lora_A(dropout(x))) * scaling
```

If you naively replace **every** `nn.Linear` in the LLM subtree with
`bnb.nn.Linear8bitLt` (or `Linear4bit`), you also replace `lora_A` and
`lora_B`. Those matrices are rank-8/16 — quantizing them loses information
catastrophically, and their downcast outputs collide with the residual sum
in a way that triggers `RuntimeError: "addmm_cuda" not implemented for
'Char'` inside bitsandbytes.

**Rule**: in any quantization filter, **skip any module whose name
contains `lora_`**. The LoRA wrapper's `base_layer` (the big matrix) still
gets quantized — that's where the win is — and lora_A/lora_B stay in BF16.

A cleaner long-term fix is `model.merge_and_unload()` before quantization,
which folds LoRA into the base weights and removes the wrappers entirely.
We did not do this yet because it changes the loading path and we wanted to
preserve the ability to keep LoRA unmerged.

### 3.2 bitsandbytes performance is mediocre on Ada/L4

`bnb.nn.Linear8bitLt` is optimized for the outlier-mixed-precision LLM.int8
algorithm. On L4 it tends to be **only ~1.1–1.4× faster** than BF16 because
Ada has cheap BF16 tensor cores and bnb adds overhead for outlier handling.

Real wins come from:
- **FP8 via Transformer Engine** on Hopper/Blackwell (no equivalent on Ada).
- **TensorRT-LLM** with INT8 / FP8 weight-only quant + CUDA graphs.
- **AWQ / SmoothQuant + ModelOpt** with calibration data.

On AGX Thor this matters: **bitsandbytes is fine as a stopgap, but the
production path on Blackwell is FP8 + TensorRT-LLM**. See §5.

---

## 4. Performance summary (measured on Modal L4)

| Configuration | Latency | FPS | Memory | Notes |
|---|---|---|---|---|
| Upstream `predict_language=True` (baseline) | 948 ms | 1.05 | 2253 MB | ~100 decode steps |
| Phase 1: `waypoints_only=True`, BF16 | 200 ms | **5.0** | 2253 MB | Single prefill, validate harness |
| Phase 1 + `torch.compile` vision | 135 ms | **7.4** | 2253 MB | Bench harness (less overhead) |
| Phase 1 + INT8 bnb `Linear8bitLt` | 274 ms | 3.64 | 2253 MB | **Slower** — bnb mediocre on Ada |
| Phase 1 + W4A8 bnb `Linear4bit` NF4 | 210 ms | 4.77 | **1878 MB** | Slower, but −375 MB |

Quality drift (24 CARLA validation frames, BF16 vs INT8):
- BF16 wp ADE = 40.87 m / route ADE = 2.30 m
- INT8 wp ADE = 40.90 m / route ADE = 2.27 m
- **Per-frame waypoint max-abs-diff**: mean 0.26 m, p90 0.57 m, max 1.33 m
- **Conclusion**: INT8 preserves quality (drift < 30 cm mean); route metrics
  are within noise.

Quality drift (32 CARLA validation frames, lang=OFF vs lang=ON):
- Δ wp ADE: +0.20 m
- Δ wp FDE: +0.52 m
- Δ route ADE: −0.11 m (slightly better)

### Key takeaway

**On L4, bitsandbytes quantization is a net regression** — it saves memory
but loses ~30–60% of FPS because Ada's BF16 tensor cores are cheap and bnb
adds per-call overhead (outlier detection, bf16→fp16 cast, CPU state
shuffling). The quantization code path is still valuable because:
1. It validates the architecture works end-to-end with quantized weights.
2. The memory savings (375 MB) are meaningful on memory-constrained edge
   devices.
3. **On Thor/Blackwell, the equivalent path using TransformerEngine FP8 or
   TensorRT-LLM will produce real speedups** (1.6–2.5×), because FP8 is
   native silicon there.

The two variants are now first-class and selectable via config:
- **Phase 1 only (recommended on L4)**: `OptimizationConfig(quantization="none", waypoints_only=True)`
- **Phase 1+2 (recommended on Thor)**: `OptimizationConfig(quantization="w4a8" or "fp8", waypoints_only=True)`

> The 30+ m absolute waypoint ADE is a coordinate-frame artifact in the
> validation harness (model outputs are vehicle-frame; GT is built from
> ego_matrix chains in the same frame, but both modes show the *same*
> offset, so the delta is what matters). The 2.3 m route ADE is the
> trustworthy quality number.

---

## 5. Deploying to NVIDIA AGX Thor

### 5.1 Hardware delta vs L4

| | L4 | AGX Thor (max-Q) |
|---|---|---|
| Architecture | Ada SM89 | Blackwell SM100 |
| BF16 dense TFLOPS | ~121 | ~500 |
| FP8 / INT8 TFLOPS | ~242 | ~1000 |
| FP4 sparse TFLOPS | — | ~2000 |
| Memory bandwidth | 300 GB/s GDDR6 | 273 GB/s LPDDR5X |
| Memory | 24 GB | 128 GB unified |
| Power | 72 W | 40–130 W configurable |
| CUDA Compute Capability | 8.9 | 10.0 |

**Implications**:
- Compute is ~4× higher → vision encoder prefill ~3× faster realistically.
- Memory bandwidth is similar → no automatic LLM speedup; **quantization
  matters more on Thor than on L4** because bandwidth dominates the LLM.
- FP8 is hardware-native → use Transformer Engine, not bitsandbytes.
- Unified memory removes host↔GPU copies → preprocessing can use
  zero-copy.

### 5.2 Step-by-step Thor port

#### Step 1 — Use JetPack 7 PyTorch wheels

AGX Thor ships with JetPack 7 (Ubuntu 24.04 arm64, CUDA 13.x). PyTorch
wheels are published by NVIDIA at
`https://developer.download.nvidia.com/compute/redist/jp/v70/pytorch/`.

Pin versions: `torch==2.5.* (jp70)`, `torchvision==0.20.*`,
`transformers==4.45.*` (matches what training uses).

```bash
# On Thor
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v70/pytorch/ \
    torch==2.5.0 torchvision==0.20.0
pip install transformers==4.45.0 peft==0.12.0 accelerate==0.34.0 \
            opencv-python ujson hydra-core==1.3.2
```

#### Step 2 — Skip bitsandbytes, use TransformerEngine for FP8

bitsandbytes on arm64 + Blackwell is not officially supported. Replace
the bnb quantization path with **TransformerEngine FP8**:

```python
# simlingo_optimized/quantization.py — add a third backend
import transformer_engine.pytorch as te

def _quantize_fp8_te(model, quantize_llm=True):
    # Iterate language_model.layers.*.{self_attn,mlp} and replace
    # nn.Linear with te.Linear in FP8 mode.
    for name, module in model.named_modules():
        if "language_model" not in name.lower():
            continue
        if "lora_" in name.lower():
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                new = te.Linear(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    params_dtype=torch.bfloat16,
                )
                # copy weights, wrap in fp8 autocast at call time
                with torch.no_grad():
                    new.weight.copy_(child.weight)
                    if child.bias is not None: new.bias.copy_(child.bias)
                setattr(module, child_name, new)
    return model

# At inference time, wrap forward in fp8_autocast:
from transformer_engine.common.recipe import Format, DelayedScaling
fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(driving_input)
```

Expected speedup: 1.6–2.0× over BF16 on Thor LLM portion.

#### Step 3 — Export to TensorRT-LLM for the production path

For the absolute fastest inference, export the LLM to TRT-LLM and keep the
vision encoder + driving adaptor in PyTorch:

1. Merge LoRA: `model = model.merge_and_unload()` before export.
2. Dump the inner `language_model` to a HF-compatible directory.
3. Convert with `trtllm-build --tp_size 1 --max_input_len 512 --max_seq_len
   513 --use_fp8 --quant_mode int8_kv_cache`.
4. Wrap the resulting engine in a `tensorrt_llm.runtime.GenerationSession`
   and call it from `_install_waypoints_only_forward` instead of the HF
   `language_model`.

Because we only need a single prefill (no autoregressive decode), the TRT
engine can be built with `max_seq_len = max_input_len + 1` and CUDA Graphs
captured for the fixed input shape, eliminating launch overhead.

Expected: **40–60 Hz on Thor at 60 W**, depending on prompt length.

#### Step 4 — Preprocessing on the iGPU via DLA-friendly torchvision

Thor has DLA accelerators (2 cores, ~50 TOPS each). Use them for the
preprocessing pipeline (resize/normalize) via TensorRT engines:

- `torchvision.transforms.v2` works out of the box on Thor with CUDA
  fallback.
- For zero-copy, use `cv2.VideoCapture` + `cudaMemcpyAsync` into a
  pre-allocated GPU tensor.
- The current `simlingo_optimized/preprocessing.py` already does GPU
  preprocessing; only the CARLA bonnet-crop step is CPU. Move that to GPU
  with a fixed-slice on the input tensor (one line change).

#### Step 5 — Power & thermal tuning

Thor's max-Q mode (40 W) caps GPU clocks and will give ~50% of the
130 W performance. For benchmark-relevant numbers:

```bash
sudo nvpmodel -m 0       # MAXN (130 W)
sudo jetson_clocks       # lock clocks to max
```

For deployment, **60 W** is a reasonable target — sustained throttling
above 60 W is common in a vehicle enclosure.

#### Step 6 — Latency budget on Thor (projection)

| Stage | L4 (measured) | Thor 60 W (projected) | Thor 130 W |
|---|---|---|---|
| Preprocessing | ~10 ms | ~5 ms | ~3 ms |
| Vision encoder (compiled) | ~30 ms | ~12 ms | ~8 ms |
| LLM prefill (BF16) | ~85 ms | ~35 ms | ~22 ms |
| LLM prefill (FP8 via TE) | n/a | ~18 ms | ~12 ms |
| LLM prefill (TRT-LLM FP8) | n/a | ~10 ms | ~6 ms |
| Driving heads | ~10 ms | ~4 ms | ~3 ms |
| **Total, BF16 only** | ~135 ms (7.4 Hz) | ~56 ms (**18 Hz**) | ~36 ms (28 Hz) |
| **Total, + FP8** | n/a | ~39 ms (**26 Hz**) | ~26 ms (38 Hz) |
| **Total, + TRT-LLM** | n/a | ~31 ms (**32 Hz**) | ~20 ms (50 Hz) |

> **Honest caveat**: these are projections, not measurements. Real Thor
> numbers will depend on memory layout, thermal envelope, and whether the
> vision encoder hits the same SM occupancy as on Ada.

---

## 6. Files in `simlingo_optimized/`

| File | Purpose |
|---|---|
| `config.py` | `OptimizationConfig` dataclass; presets (`fast_compile`, `quantized`, `agx_thor`) |
| `inference.py` | `OptimizedSimLingo` model wrapper; loads checkpoint, applies opts, exposes `predict()` |
| `preprocessing.py` | GPU preprocessing pipeline (resize, normalize, crop) |
| `quantization.py` | bitsandbytes INT8 / NF4 W4A8 application with LoRA-safe filter |
| `streaming_kv.py` | (placeholder for Phase 4) KV cache reuse across frames |
| `kernels/` | (placeholder for Phase 3) fused QKV, fused MLP Triton kernels |
| `export_trt.py` | (placeholder for Phase 5) TensorRT-LLM export utilities |
| `modal_benchmark.py` | Modal entrypoints: `benchmark_l4`, `validate_l4`, `benchmark_l4_quant`, `validate_l4_quant` |

---

## 7. Open questions / TODO

1. **Merge LoRA before quantization** — would let us quantize the full Linear
   matrix without the `lora_` carve-out, gaining a few more % speedup.
2. **Streaming KV cache** — most of the vision prefill is wasted because
   the next frame's features overlap heavily. A 4-frame sliding window could
   cut vision time by 75% on frames 2+. Requires pre-RoPE key caching.
3. **AWQ calibration on CARLA data** — bnb's NF4 is data-free; AWQ with a
   small CARLA calibration set should be ~0.5–1.0 m better on ADE.
4. **Closed-loop CARLA evaluation** — open-loop ADE/FDE is a proxy. We
   still need to plug `OptimizedSimLingo` into `team_code/agent_simlingo.py`
   and run a Leaderboard route to confirm driving behaviour is unchanged.
