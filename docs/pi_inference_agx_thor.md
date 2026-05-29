# Inferencing π0.5 on NVIDIA AGX Thor

Goal: deploy the fine-tuned π0.5 driving policy on a Jetson AGX Thor at a real-time control rate (target **5–10 Hz**).

**Bottom line:** this is a published/solved path. The official NVIDIA Jetson AI Lab tutorial gets π0.5 to **~94 ms (≈10.6 Hz)** on a stock Thor dev kit. A community engine (FlashRT) pushes it to **44 ms (≈23 Hz)**. Both clear the target.

---

## Prior art

### 1. NVIDIA Jetson AI Lab — official tutorial (canonical recipe)

"OpenPi π₀.₅ on Jetson Thor" — https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/

Pipeline:

```
JAX ckpt ──► PyTorch (safetensors) ──► ONNX (FP8 + NVFP4) ──► TensorRT engine ──► inference
```

Stack:
- JetPack 7.x (L4T R38.x), CUDA 13.0, Docker 28.x+
- Base container `nvcr.io/nvidia/pytorch:25.09-py3` (PyTorch + CUDA + TensorRT + ModelOpt preinstalled)
- pip index `https://pypi.jetson-ai-lab.io/sbsa/cu130` (ONNX, ONNXRuntime, ONNX GraphSurgeon, ModelOpt, onnxslim)
- Pinned openpi commit `175f89c3` for reproducibility

Measured on a Thor dev kit (MAXN power), LIBERO config (2 views, `action_horizon=10`, 7-dim actions):

| Backend | Total latency | Hz | Speedup |
|---|---|---|---|
| PyTorch BF16 (baseline) | ~163 ms | ~6 | 1.0× |
| TensorRT FP8 | ~95 ms | ~10.5 | 1.71× |
| **TensorRT FP8 + NVFP4** | **~94 ms** | **~10.6** | **1.73×** |

Per-timestep cosine similarity stays >0.99 vs. the PyTorch reference, so quantization is not degrading action quality.

**Key gotchas:**
- **Pure FP16 is unsupported** — Gemma attention overflows FP16's 5-bit exponent range. Must use FP8/NVFP4.
- Two HF Transformers patches required for TRT export: a `GemmaRMSNorm.extra_repr()` guard, and replacing dynamic `-1` reshape dims in `GemmaAttention.forward()` with explicit `num_attention_heads * head_dim` for FP4 block quantization.
- `ACTION_HORIZON` is a build-time variable in the engine and **must match the model config**.
- Weights >6 GB → NVMe required. TRT build takes 10–30 min on Thor.

Representative commands (from the tutorial):

```bash
# Convert JAX → PyTorch
python examples/convert_jax_model_to_pytorch.py   # ~5-10 min, emits model.safetensors + config.json

# PyTorch → ONNX with quantization
python openpi_on_thor/pytorch_to_onnx.py \
  --checkpoint_dir <path> \
  --precision fp8 \
  --enable_llm_nvfp4 \
  --quantize_attention_matmul

# ONNX → TensorRT engine
ACTION_HORIZON=10 bash openpi_on_thor/build_engine.sh <onnx_path> <engine_path>

# Run TensorRT inference
python openpi_on_thor/pi05_inference.py \
  --config-name pi05_libero \
  --engine-path <engine_path> \
  --inference-mode tensorrt
```

### 2. FlashRT (custom-kernel engine — faster ceiling)

`LiangSu8899/FlashRT` — from-scratch real-time engine that skips the TRT compile step in favor of hand-written CUDA kernels + static CUDA-graph capture.

| Hardware | Model | Config | Latency | Hz |
|---|---|---|---|---|
| Jetson Thor (SM110) | Pi0.5 | 2-view FP8 | **44 ms** | **23** |
| RTX 5090 (SM120) | Pi0.5 | 2-view FP8 | 17.6 ms | 57 |
| Jetson Thor | GR00T N1.6 | T=50 | 45 ms | 22 |

Techniques: fused norm/activation/residual/RoPE kernels, FlashAttention-2, CUTLASS strided MHA, cuBLASLt GEMM, FP8 + NVFP4 with cached per-tensor calibration, **removing Q/DQ nodes from the pipeline** (cited as the big edge-device win), and full CUDA-graph replay (zero Python overhead per step). Runs in ~10 GB VRAM, single-stream — well suited to on-vehicle. Supports Pi0, Pi0.5, Pi0-FAST, GR00T N1.6.

### 3. openpi-flash (remote serving — not on-device)

`Hebbian-Robotics/openpi-flash` — a QUIC-first + Rust transport layer that cuts policy-server round-trip latency (4000 ms → 200 ms). Only relevant if inference is off-boarded to a remote GPU; not applicable if the policy runs on the Thor itself.

---

## Fast levers (ranked by impact)

LLM backbone + flow-matching denoising are 60–75% of latency pre-optimization (arXiv 2510.26742, "Running VLAs at Real-Time Speed").

1. **FP8 + NVFP4 quantization (TensorRT)** — single biggest lever (~1.7×); the difference between 6 Hz and 10.6 Hz. Mandatory (FP16 won't run).
2. **Action chunking + async re-planning** — decouples control rate from inference rate. Generate a trajectory chunk, execute open-loop, re-plan at 5–10 Hz. π0.5's real-time chunking (RTC) overlaps inference with execution of the prior chunk so the controller never stalls.
3. **Fewer flow-matching denoising steps** — π0.5 defaults to 10; each step is a full action-expert forward pass. ~4–5 often holds quality with near-linear latency savings on the expert.
4. **CUDA graph capture** (FlashRT approach) — eliminates per-step Python dispatch once the model is stable.
5. **Reduce vision token count** — fewer/lower-res camera views (see caveat below).

---

## ⚠️ Caveats specific to the driving model

Published 94 ms / 44 ms numbers are for **LIBERO manipulation**: 2 views, `action_horizon=10`, 7-dim. The driving fine-tune differs in ways that directly move latency:

- **Camera count / resolution drives prefill cost.** Prefill over image tokens is the dominant chunk. More views (AR1 used 4 cameras × 4 frames) or higher resolution inflate prefill token count and will push above 10 Hz. **This is the #1 risk vs. the target** — budget the camera setup for the latency goal.
- **Action horizon.** A longer trajectory output (e.g. AR1's 64-step) lengthens the action-expert sequence per denoising step vs. 10.

Neither is fixed — both are tunable — but "94 ms" is not a guarantee for our checkpoint until benchmarked on our actual config.

---

## Recommended path

1. Convert the π0.5 driving checkpoint JAX → PyTorch (`examples/convert_jax_model_to_pytorch.py`).
2. Run the Jetson AI Lab pipeline on a Thor with **our** camera/horizon config at FP8 + NVFP4. Measure real latency.
3. If over budget: cut denoising steps → reduce views/res → add chunking/RTC. Custom kernels (FlashRT) are the ceiling (~2× the TRT path) if still hungry after that.
4. **Hardware note:** the SM110 NVFP4 TensorRT engine must be built and benchmarked on the actual Thor — engines are hardware-locked. Convert/export can be dry-run elsewhere, but confirm Thor access before committing.

---

## Sources

- [OpenPi π₀.₅ on Jetson Thor — Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)
- [Real-Time Inference on Thor & RTX (Pi0.5/GR00T) — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788)
- [LiangSu8899/FlashRT — GitHub](https://github.com/LiangSu8899/FlashRT)
- [Running VLAs at Real-Time Speed — arXiv 2510.26742](https://arxiv.org/pdf/2510.26742)
- [Hebbian-Robotics/openpi-flash — GitHub](https://github.com/Hebbian-Robotics/openpi-flash)
- [openpi issue #826 — Thor support for Pi05](https://github.com/Physical-Intelligence/openpi/issues/826)
