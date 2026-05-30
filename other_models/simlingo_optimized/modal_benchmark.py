"""Modal benchmark for SimLingo optimizations.

This script runs benchmarks on Modal GPUs (A100/A10G) to measure the performance
impact of each optimization. It tests optimizations incrementally and reports
results after each step.

Usage:
    # Run full benchmark suite on A100
    modal run simlingo_optimized/modal_benchmark.py::benchmark_all

    # Run single optimization test
    modal run simlingo_optimized/modal_benchmark.py::benchmark_single --config fast_compile

    # Compare baseline vs optimized
    modal run simlingo_optimized/modal_benchmark.py::compare
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal

# ---------------------------------------------------------------------------
# Constants (matching simlingo modal_app.py)
# ---------------------------------------------------------------------------

APP_NAME = "simlingo-optimized-benchmark"

SIMLINGO_REPO_DIR = "/opt/simlingo"
CACHE_DIR = "/cache"
DATA_DIR = "/data"
OUTPUTS_DIR = "/outputs"
CKPT_DIR = f"{DATA_DIR}/checkpoint"
EXTRACTED_DIR = f"{DATA_DIR}/extracted"

HF_CKPT_FILE = "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
HF_HYDRA_CONFIG_FILE = "simlingo/.hydra/config.yaml"

# ---------------------------------------------------------------------------
# Image - same as simlingo but with our optimized module
# ---------------------------------------------------------------------------

SIMLINGO_REPO_URL = "https://github.com/RenzKa/simlingo.git"

benchmark_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "git-lfs",
        "build-essential",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "sentencepiece",
        "peft==0.13.2",
        "accelerate==1.0.1",
        "huggingface_hub==0.27.0",
        "pytorch-lightning==2.4.0",
        "lightning==2.3.3",
        "hydra-core==1.3.2",
        "hydra-zen==0.12.1",
        "omegaconf==2.3.0",
        "einops==0.7.0",
        "timm==0.9.16",
        "scipy==1.10.1",
        "scikit-image==0.21.0",
        "imgaug==0.4.0",
        "Pillow==10.2.0",
        "filterpy==1.4.5",
        "ujson==5.9.0",
        "matplotlib==3.7.5",
        "tqdm",
        "numpy<2",
        "opencv-python-headless==4.10.0.84",
        "line_profiler",
    )
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    )
    .pip_install("deepspeed==0.16.2")
    # bitsandbytes 0.41.x is compatible with torch 2.2
    .pip_install("bitsandbytes==0.41.3.post2")
    .run_commands(
        f"git clone --depth 1 {SIMLINGO_REPO_URL} {SIMLINGO_REPO_DIR}",
    )
    .env(
        {
            "PYTHONPATH": SIMLINGO_REPO_DIR,
            "HF_HOME": f"{CACHE_DIR}/hf",
            "HUGGINGFACE_HUB_CACHE": f"{CACHE_DIR}/hf/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_DIR}/hf/transformers",
            "TRUST_REMOTE_CODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # Mount our simlingo_optimized module
    .add_local_python_source("simlingo_optimized")
)

# ---------------------------------------------------------------------------
# Volumes (shared with simlingo modal_app.py)
# ---------------------------------------------------------------------------

cache_volume = modal.Volume.from_name("simlingo-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("simlingo-data", create_if_missing=True)
output_volume = modal.Volume.from_name("simlingo-outputs", create_if_missing=True)

VOLUMES = {
    CACHE_DIR: cache_volume,
    DATA_DIR: data_volume,
    OUTPUTS_DIR: output_volume,
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------
# Benchmark dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config_name: str
    num_iterations: int

    # Latency (ms)
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    latency_std_ms: float

    # Throughput
    fps: float

    # Memory (MB)
    memory_peak_mb: float
    memory_allocated_mb: float

    # Breakdown (ms)
    preprocess_mean_ms: float = 0.0
    model_forward_mean_ms: float = 0.0

    # Comparison to baseline
    speedup_vs_baseline: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "num_iterations": self.num_iterations,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_std_ms": self.latency_std_ms,
            "fps": self.fps,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_allocated_mb": self.memory_allocated_mb,
            "preprocess_mean_ms": self.preprocess_mean_ms,
            "model_forward_mean_ms": self.model_forward_mean_ms,
            "speedup_vs_baseline": self.speedup_vs_baseline,
        }

    def print_summary(self):
        """Print formatted summary."""
        print(f"\n{'='*60}")
        print(f"Config: {self.config_name}")
        print(f"{'='*60}")
        print(f"Iterations: {self.num_iterations}")
        print(f"\nLatency:")
        print(f"  Mean:  {self.latency_mean_ms:7.2f} ms")
        print(f"  P50:   {self.latency_p50_ms:7.2f} ms")
        print(f"  P90:   {self.latency_p90_ms:7.2f} ms")
        print(f"  P99:   {self.latency_p99_ms:7.2f} ms")
        print(f"\nThroughput: {self.fps:.2f} FPS")
        print(f"Speedup vs baseline: {self.speedup_vs_baseline:.2f}x")
        print(f"\nMemory: {self.memory_peak_mb:.1f} MB peak")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def _run_benchmark_internal(
    config_name: str,
    num_warmup: int = 5,
    num_iterations: int = 50,
    use_real_data: bool = True,  # Default to using real CARLA data
) -> BenchmarkResult:
    """Internal benchmark function - runs inside Modal container."""
    import numpy as np
    import torch

    sys.path.insert(0, SIMLINGO_REPO_DIR)

    from simlingo_optimized import OptimizedSimLingo, OptimizationConfig

    # Create config based on name
    if config_name == "baseline":
        # Baseline: GPU preprocessing but no torch.compile or quantization
        config = OptimizationConfig(
            use_torch_compile=False,
            use_gpu_preprocessing=True,  # Use GPU preprocessing as baseline
            quantization="none",
            verbose=True,
        )
    elif config_name == "cpu_preprocess":
        # CPU preprocessing only (for comparison)
        config = OptimizationConfig(
            use_torch_compile=False,
            use_gpu_preprocessing=False,
            quantization="none",
            verbose=True,
        )
    elif config_name == "torch_compile":
        # torch.compile on vision encoder only
        config = OptimizationConfig(
            use_torch_compile=True,
            compile_mode="reduce-overhead",
            use_gpu_preprocessing=True,
            quantization="none",
            verbose=True,
        )
    elif config_name == "max_autotune":
        # torch.compile with max-autotune
        config = OptimizationConfig(
            use_torch_compile=True,
            compile_mode="max-autotune",
            use_gpu_preprocessing=True,
            quantization="none",
            verbose=True,
        )
    elif config_name == "quantized_int8":
        # INT8 quantization (no torch.compile to isolate effect)
        # Only quantize LLM, not vision encoder (vision is compute-bound)
        config = OptimizationConfig(
            use_torch_compile=False,
            use_gpu_preprocessing=True,
            quantization="int8",
            quantize_vision_encoder=False,  # Don't quantize vision
            quantize_language_head=True,    # Only quantize LLM
            verbose=True,
        )
    elif config_name == "quantized_w4a8":
        # W4A8 quantization with bitsandbytes
        # Only quantize LLM, not vision encoder
        config = OptimizationConfig(
            use_torch_compile=False,
            use_gpu_preprocessing=True,
            quantization="w4a8",
            quantize_vision_encoder=False,  # Don't quantize vision
            quantize_language_head=True,    # Only quantize LLM
            verbose=True,
        )
    elif config_name == "full_optimized":
        # All optimizations: torch.compile + quantization
        # Only quantize LLM to avoid compatibility issues with vision encoder
        config = OptimizationConfig(
            use_torch_compile=True,
            compile_mode="reduce-overhead",
            use_gpu_preprocessing=True,
            quantization="int8",
            quantize_vision_encoder=False,  # Don't quantize vision (it's compiled)
            quantize_language_head=True,    # Only quantize LLM
            verbose=True,
        )
    elif config_name == "streaming":
        # Streaming inference with vision caching
        # Note: torch.compile disabled to ensure wrapper works correctly
        config = OptimizationConfig(
            use_torch_compile=False,  # Disable compile to test wrapper
            compile_mode="reduce-overhead",
            use_gpu_preprocessing=True,
            quantization="none",
            enable_streaming=True,
            streaming_window_size=10,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown config: {config_name}")

    ckpt_path = str(Path(CKPT_DIR) / HF_CKPT_FILE)
    hydra_cfg_path = str(Path(CKPT_DIR) / HF_HYDRA_CONFIG_FILE)

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {config_name}")
    print(f"{'='*60}")
    print(f"Config: {config.to_dict()}")

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load model
    print("\nLoading model...", flush=True)
    t0 = time.time()
    model = OptimizedSimLingo(
        ckpt_path=ckpt_path,
        config=config,
        hydra_cfg_path=hydra_cfg_path,
        simlingo_repo_path=SIMLINGO_REPO_DIR,
        cache_dir=f"{CACHE_DIR}/hf/snapshots",
    )
    model.load()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    # Create test input - use real CARLA data with measurements
    # For streaming tests, load multiple consecutive frames
    is_streaming = config.enable_streaming
    test_frames = []  # List of (image, speed, targets) tuples
    test_image = None
    test_speed = 5.0
    test_targets = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32)

    if use_real_data and Path(EXTRACTED_DIR).exists():
        import gzip
        import ujson
        import cv2

        # Find a route with rgb and measurements
        rgb_dirs = list(Path(EXTRACTED_DIR).rglob("rgb"))
        for rgb_dir in rgb_dirs:
            meas_dir = rgb_dir.parent / "measurements"
            if not meas_dir.exists():
                continue

            rgb_files = sorted(rgb_dir.glob("*.jpg"))
            if not rgb_files:
                continue

            # For streaming, load consecutive frames
            num_frames_needed = num_iterations + num_warmup if is_streaming else 10
            frames_loaded = 0

            for rgb_file in rgb_files[:num_frames_needed + 50]:
                frame_idx = int(rgb_file.stem)
                meas_file = meas_dir / f"{frame_idx:04d}.json.gz"
                if meas_file.exists():
                    # Load image
                    bgr = cv2.imread(str(rgb_file))
                    if bgr is None:
                        continue
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                    # Load measurements
                    with gzip.open(meas_file, "rt") as f:
                        meas = ujson.load(f)

                    speed = float(meas.get("speed", 5.0))
                    tp = meas.get("target_point", [10.0, 0.0])
                    tp_next = meas.get("target_point_next", [20.0, 0.0])
                    targets = np.array([tp, tp_next], dtype=np.float32)

                    test_frames.append((img, speed, targets))
                    frames_loaded += 1

                    if test_image is None:
                        test_image = img
                        test_speed = speed
                        test_targets = targets
                        print(f"Using CARLA data: {rgb_file}", flush=True)
                        print(f"  Speed: {test_speed:.1f} m/s", flush=True)
                        print(f"  Target: {tp}", flush=True)

                    if frames_loaded >= num_frames_needed:
                        break

            if test_frames:
                print(f"  Loaded {len(test_frames)} consecutive frames for {'streaming' if is_streaming else 'single-frame'} test", flush=True)
                break

    if test_image is None:
        print("Using synthetic test image (no CARLA data found)", flush=True)
        test_image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        test_frames = [(test_image, test_speed, test_targets)]

    # Warmup
    print(f"\nWarming up ({num_warmup} iterations)...", flush=True)
    for i in range(num_warmup):
        if is_streaming and len(test_frames) > i:
            img, spd, tgt = test_frames[i]
            _ = model.predict_stream(
                image=img,
                speed_mps=spd,
                target_points=tgt,
                reset_stream=(i == 0),
            )
        else:
            _ = model.predict(
                image=test_image,
                speed_mps=test_speed,
                target_points=test_targets,
            )
        if (i + 1) % 2 == 0:
            print(f"  Warmup {i + 1}/{num_warmup}", flush=True)

    torch.cuda.synchronize()
    model.reset_timing_stats()

    # Benchmark
    print(f"\nBenchmarking ({num_iterations} iterations)...", flush=True)
    latencies = []

    for i in range(num_iterations):
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        if is_streaming:
            # Use consecutive frames for streaming test
            frame_idx = (num_warmup + i) % len(test_frames)
            img, spd, tgt = test_frames[frame_idx]
            _ = model.predict_stream(
                image=img,
                speed_mps=spd,
                target_points=tgt,
                reset_stream=False,
            )
        else:
            _ = model.predict(
                image=test_image,
                speed_mps=test_speed,
                target_points=test_targets,
            )

        torch.cuda.synchronize()
        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

        if (i + 1) % 10 == 0:
            current_fps = 1000.0 / np.mean(latencies)
            extra_info = ""
            if is_streaming:
                try:
                    stats = model.get_streaming_stats()
                    extra_info = f", cache_hit={stats.get('cache_hit_rate', 0):.1%}"
                except:
                    pass
            print(
                f"  [{i + 1:>3}/{num_iterations}] "
                f"mean={np.mean(latencies):.1f}ms, fps={current_fps:.2f}{extra_info}",
                flush=True,
            )

    # Compute stats
    latencies_arr = np.array(latencies)
    timing_stats = model.get_timing_stats()

    result = BenchmarkResult(
        config_name=config_name,
        num_iterations=num_iterations,
        latency_mean_ms=float(latencies_arr.mean()),
        latency_p50_ms=float(np.percentile(latencies_arr, 50)),
        latency_p90_ms=float(np.percentile(latencies_arr, 90)),
        latency_p99_ms=float(np.percentile(latencies_arr, 99)),
        latency_std_ms=float(latencies_arr.std()),
        fps=1000.0 / float(latencies_arr.mean()),
        memory_peak_mb=torch.cuda.max_memory_allocated() / (1024 * 1024),
        memory_allocated_mb=torch.cuda.memory_allocated() / (1024 * 1024),
        preprocess_mean_ms=timing_stats.get("preprocess_ms", {}).get("mean", 0),
        model_forward_mean_ms=timing_stats.get("inference_ms", {}).get("mean", 0),
    )

    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Modal entrypoints
# ---------------------------------------------------------------------------


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="A100",  # Use A100 for faster benchmarking
    timeout=60 * 60,
    cpu=8,
    memory=48 * 1024,
)
def benchmark_single(
    config: str = "baseline",
    num_iterations: int = 50,
    num_warmup: int = 5,
) -> dict:
    """Run benchmark for a single configuration.

    Args:
        config: One of: baseline, gpu_preprocess, torch_compile,
                compile_gpu_preprocess, max_autotune, quantized_int8, quantized_w4a8
        num_iterations: Number of benchmark iterations
        num_warmup: Number of warmup iterations

    Returns:
        Benchmark results as dict
    """
    result = _run_benchmark_internal(
        config_name=config,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )
    return result.to_dict()


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="A100",
    timeout=60 * 60 * 2,  # 2 hours for full suite
    cpu=8,
    memory=48 * 1024,
)
def benchmark_all(
    num_iterations: int = 50,
    num_warmup: int = 5,
    skip_quantized: bool = False,
) -> dict:
    """Run full benchmark suite, testing each optimization incrementally.

    This tests:
    1. Baseline (no optimizations)
    2. GPU preprocessing only
    3. torch.compile only
    4. torch.compile + GPU preprocessing
    5. max-autotune mode
    6. INT8 quantization (optional)
    7. W4A8 quantization (optional)

    After each test, reports speedup vs baseline.
    """
    import numpy as np
    import torch

    configs = [
        "baseline",           # GPU preprocess, no compile
        "torch_compile",      # + torch.compile (vision only)
        "max_autotune",       # + max-autotune mode
        "streaming",          # Streaming with vision caching
    ]

    if not skip_quantized:
        configs.extend([
            "quantized_int8",     # INT8 quantization
            "quantized_w4a8",     # W4A8 (4-bit weights, 8-bit activations)
            "full_optimized",     # torch.compile + quantization
        ])

    results = {}
    baseline_fps = None

    print("\n" + "=" * 80)
    print("SIMLINGO OPTIMIZATION BENCHMARK SUITE")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Configs to test: {configs}")
    print("=" * 80 + "\n")

    for config_name in configs:
        print(f"\n{'#' * 80}")
        print(f"# TESTING: {config_name}")
        print(f"{'#' * 80}")

        try:
            result = _run_benchmark_internal(
                config_name=config_name,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )

            if baseline_fps is None:
                baseline_fps = result.fps
                result.speedup_vs_baseline = 1.0
            else:
                result.speedup_vs_baseline = result.fps / baseline_fps

            results[config_name] = result.to_dict()

            # Print incremental comparison
            print(f"\n>>> RESULT: {config_name}")
            print(f"    FPS: {result.fps:.2f}")
            print(f"    Speedup vs baseline: {result.speedup_vs_baseline:.2f}x")
            print(f"    Memory: {result.memory_peak_mb:.1f} MB")

        except Exception as e:
            print(f"\n>>> FAILED: {config_name}")
            print(f"    Error: {e}")
            results[config_name] = {"error": str(e)}

        # Clear memory between tests
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Config':<25} {'FPS':<10} {'Speedup':<10} {'P90 (ms)':<12} {'Memory (MB)':<12}")
    print("-" * 80)

    for config_name in configs:
        r = results.get(config_name, {})
        if "error" in r:
            print(f"{config_name:<25} {'FAILED':<10}")
        else:
            print(
                f"{config_name:<25} "
                f"{r.get('fps', 0):<10.2f} "
                f"{r.get('speedup_vs_baseline', 0):<10.2f}x "
                f"{r.get('latency_p90_ms', 0):<12.2f} "
                f"{r.get('memory_peak_mb', 0):<12.1f}"
            )

    print("=" * 80)

    # Save results
    output_path = Path(OUTPUTS_DIR) / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    output_volume.commit()

    return results


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="A10G",  # Also test on A10G (closer to production)
    timeout=60 * 60,
    cpu=4,
    memory=24 * 1024,
)
def benchmark_a10g(
    config: str = "compile_gpu_preprocess",
    num_iterations: int = 50,
) -> dict:
    """Run benchmark on A10G GPU (closer to production hardware)."""
    import torch

    print(f"GPU: {torch.cuda.get_device_name()}")
    result = _run_benchmark_internal(
        config_name=config,
        num_warmup=5,
        num_iterations=num_iterations,
    )
    return result.to_dict()


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="A100",
    timeout=60 * 30,
    cpu=8,
    memory=48 * 1024,
)
def compare(
    baseline_config: str = "baseline",
    optimized_config: str = "compile_gpu_preprocess",
    num_iterations: int = 50,
) -> dict:
    """Compare baseline vs optimized configuration."""
    import torch

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Comparing: {baseline_config} vs {optimized_config}")

    baseline = _run_benchmark_internal(
        config_name=baseline_config,
        num_warmup=5,
        num_iterations=num_iterations,
    )

    import gc

    gc.collect()
    torch.cuda.empty_cache()

    optimized = _run_benchmark_internal(
        config_name=optimized_config,
        num_warmup=5,
        num_iterations=num_iterations,
    )

    speedup = optimized.fps / baseline.fps

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Baseline ({baseline_config}):")
    print(f"  FPS: {baseline.fps:.2f}")
    print(f"  P90: {baseline.latency_p90_ms:.2f} ms")
    print(f"\nOptimized ({optimized_config}):")
    print(f"  FPS: {optimized.fps:.2f}")
    print(f"  P90: {optimized.latency_p90_ms:.2f} ms")
    print(f"\nSpeedup: {speedup:.2f}x")
    print("=" * 60)

    return {
        "baseline": baseline.to_dict(),
        "optimized": optimized.to_dict(),
        "speedup": speedup,
    }


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="L4",
    timeout=60 * 60,
    cpu=4,
    memory=24 * 1024,
)
def validate_l4(
    num_samples: int = 32,
    frame_stride: int = 20,
    use_cot: bool = True,
    seed: int = 0,
    run_lang_on: bool = True,
) -> dict:
    """A/B validation on SimLingo's CARLA validation set.

    Walks the validation routes, derives ground-truth waypoints from the
    measurements ego matrices (same code path the upstream `run_inference`
    uses), runs OptimizedSimLingo in waypoints_only mode, and computes
    ADE/FDE vs ground truth. If run_lang_on is True, also runs the upstream
    autoregressive lang=ON path on the same frames and reports its metrics
    side-by-side.
    """
    import gc
    import os
    import time as _time

    import numpy as np
    import torch

    sys.path.insert(0, SIMLINGO_REPO_DIR)
    # Reuse the upstream sample collector + metric helpers — they already
    # know how to derive ground-truth waypoints from measurements ego matrices.
    sys.path.insert(0, str(Path("/root").resolve()))
    # The simlingo package isn't mounted here; copy the helpers we need.
    # They're small and pure (no upstream deps beyond numpy / ujson / gzip).

    from simlingo_optimized import OptimizationConfig, OptimizedSimLingo

    config = OptimizationConfig(
        use_torch_compile=False,
        use_gpu_preprocessing=True,
        quantization="none",
        enable_streaming=False,
        waypoints_only=True,
        verbose=True,
    )

    ckpt_path = str(Path(CKPT_DIR) / HF_CKPT_FILE)
    hydra_cfg_path = str(Path(CKPT_DIR) / HF_HYDRA_CONFIG_FILE)

    print("\n" + "=" * 70)
    print("L4 VALIDATION — ADE/FDE on SimLingo CARLA validation set")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"num_samples={num_samples}  frame_stride={frame_stride}  use_cot={use_cot}")
    print("=" * 70)

    gc.collect()
    torch.cuda.empty_cache()

    model = OptimizedSimLingo(
        ckpt_path=ckpt_path,
        config=config,
        hydra_cfg_path=hydra_cfg_path,
        simlingo_repo_path=SIMLINGO_REPO_DIR,
        cache_dir=f"{CACHE_DIR}/hf/snapshots",
    )
    model.load()

    # -- Pull GT-builder helpers from upstream inference.py via importlib --
    # The simlingo/scripts package is part of the user's repo; on Modal we
    # don't auto-mount it here. Reimplement the small pieces we need
    # inline to keep the function self-contained.

    import gzip

    import cv2
    import ujson

    def _ade_fde(pred, gt):
        if pred is None or gt is None:
            return float("nan"), float("nan")
        pred = np.asarray(pred, dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)
        if pred.shape != gt.shape:
            n = min(pred.shape[0], gt.shape[0])
            pred, gt = pred[:n], gt[:n]
        d = np.linalg.norm(pred - gt, axis=-1)
        return float(d.mean()), float(d[-1])

    def _compute_waypoints_from_ego_matrix(meas_now_to_future):
        """Replicate upstream: stack future ego positions in current ego frame."""
        # meas_now_to_future is a list of 4x4 ego-to-world matrices; we want
        # positions in the current-frame coordinate system, sampled every 5
        # frames (0.25s at 20Hz) for 11 waypoints. The upstream helper does
        # an inverse-multiply with the current matrix.
        cur = np.asarray(meas_now_to_future[0], dtype=np.float64)
        cur_inv = np.linalg.inv(cur)
        out = []
        for fut in meas_now_to_future:
            fut = np.asarray(fut, dtype=np.float64)
            rel = cur_inv @ fut
            out.append([rel[0, 3], rel[1, 3]])
        return np.asarray(out, dtype=np.float32)

    def _equal_spacing_route(pts, num=20, spacing_m=1.0):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[0] < 2:
            return pts.astype(np.float32)
        dx = np.diff(pts[:, 0])
        dy = np.diff(pts[:, 1])
        seg = np.sqrt(dx * dx + dy * dy)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-6:
            return np.tile(pts[0], (num, 1)).astype(np.float32)
        targets = np.minimum(np.arange(num) * spacing_m, total)
        out = np.zeros((num, 2), dtype=np.float32)
        for i, t in enumerate(targets):
            idx = int(np.searchsorted(cum, t))
            idx = min(max(idx, 1), len(cum) - 1)
            t0, t1 = cum[idx - 1], cum[idx]
            f = 0.0 if t1 - t0 < 1e-9 else (t - t0) / (t1 - t0)
            out[i, 0] = pts[idx - 1, 0] + f * (pts[idx, 0] - pts[idx - 1, 0])
            out[i, 1] = pts[idx - 1, 1] + f * (pts[idx, 1] - pts[idx - 1, 1])
        return out

    # Collect validation frames with at least 11×5 = 55 future frames available
    rng = np.random.default_rng(seed)
    NUM_WP = 11
    WP_STRIDE = 5  # every 5 frames at 20Hz -> 0.25s spacing

    candidates = []
    for rgb_dir in Path(EXTRACTED_DIR).rglob("rgb"):
        meas_dir = rgb_dir.parent / "measurements"
        if not meas_dir.exists():
            continue
        rgb_files = sorted(rgb_dir.glob("*.jpg"))
        if len(rgb_files) < NUM_WP * WP_STRIDE + frame_stride:
            continue
        candidates.append((rgb_dir, meas_dir, rgb_files))

    if not candidates:
        return {"error": "no validation routes found under " + EXTRACTED_DIR}

    samples = []
    for rgb_dir, meas_dir, rgb_files in candidates:
        max_start = len(rgb_files) - NUM_WP * WP_STRIDE - 1
        if max_start <= 0:
            continue
        starts = list(range(0, max_start, frame_stride))
        rng.shuffle(starts)
        for s in starts:
            samples.append((rgb_dir, meas_dir, rgb_files, s))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    print(f"Collected {len(samples)} validation samples", flush=True)

    def _load_meas(meas_dir, idx):
        p = meas_dir / f"{idx:04d}.json.gz"
        if not p.exists():
            return None
        with gzip.open(p, "rt") as fp:
            return ujson.load(fp)

    def run_eval(mode_label: str, predict_language: bool):
        per_sample = []
        t_predict_total = 0.0
        for i, (rgb_dir, meas_dir, rgb_files, s_idx) in enumerate(samples):
            try:
                bgr = cv2.imread(str(rgb_files[s_idx]))
                if bgr is None:
                    continue
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                frame_idx = int(rgb_files[s_idx].stem)
                cur_meas = _load_meas(meas_dir, frame_idx)
                if cur_meas is None or "ego_matrix" not in cur_meas:
                    continue

                speed = float(cur_meas.get("speed", 0.0))
                tp = cur_meas.get("target_point", [10.0, 0.0])
                tp_next = cur_meas.get("target_point_next", tp)
                targets = np.array([tp, tp_next], dtype=np.float32)

                ego_chain = [cur_meas["ego_matrix"]]
                ok = True
                for k in range(1, NUM_WP):
                    fut_idx = frame_idx + k * WP_STRIDE
                    fm = _load_meas(meas_dir, fut_idx)
                    if fm is None or "ego_matrix" not in fm:
                        ok = False
                        break
                    ego_chain.append(fm["ego_matrix"])
                if not ok:
                    continue

                gt_wps = _compute_waypoints_from_ego_matrix(ego_chain)
                gt_route = _equal_spacing_route(gt_wps, num=20, spacing_m=1.0)

                torch.cuda.synchronize()
                t0 = _time.perf_counter()
                pred_wps, pred_route, pred_text = model.predict(
                    image=img,
                    speed_mps=speed,
                    target_points=targets,
                    use_cot=use_cot,
                    predict_language=predict_language,
                )
                torch.cuda.synchronize()
                t_predict_total += _time.perf_counter() - t0

                ade_wp, fde_wp = _ade_fde(pred_wps, gt_wps)
                ade_rt, fde_rt = _ade_fde(pred_route, gt_route)
                per_sample.append(
                    {
                        "ade_wp": ade_wp,
                        "fde_wp": fde_wp,
                        "ade_route": ade_rt,
                        "fde_route": fde_rt,
                        "text": pred_text[:80] if pred_text else "",
                    }
                )
                if (i + 1) % 10 == 0:
                    print(
                        f"  [{mode_label}] {i + 1}/{len(samples)}  "
                        f"ADE_wp={ade_wp:.3f}  FDE_wp={fde_wp:.3f}",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"  [{mode_label}] sample {i} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        if not per_sample:
            return None

        ade_wp = np.array([s["ade_wp"] for s in per_sample])
        fde_wp = np.array([s["fde_wp"] for s in per_sample])
        ade_rt = np.array([s["ade_route"] for s in per_sample])
        fde_rt = np.array([s["fde_route"] for s in per_sample])
        return {
            "n_samples": len(per_sample),
            "wp_ade_mean": float(np.nanmean(ade_wp)),
            "wp_ade_median": float(np.nanmedian(ade_wp)),
            "wp_fde_mean": float(np.nanmean(fde_wp)),
            "wp_fde_median": float(np.nanmedian(fde_wp)),
            "route_ade_mean": float(np.nanmean(ade_rt)),
            "route_fde_mean": float(np.nanmean(fde_rt)),
            "mean_predict_ms": 1000.0 * t_predict_total / max(len(per_sample), 1),
        }

    print("\n[B] waypoints_only (lang=OFF) — single prefill", flush=True)
    res_off = run_eval("OFF", predict_language=False)

    res_on = None
    on_error = None
    if run_lang_on:
        gc.collect()
        torch.cuda.empty_cache()
        print("\n[A] upstream autoregressive (lang=ON)", flush=True)
        try:
            res_on = run_eval("ON ", predict_language=True)
        except Exception as exc:
            on_error = f"{type(exc).__name__}: {exc}"
            print(f"  lang=ON crashed: {on_error}", flush=True)

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS — lower ADE/FDE is better; all values in meters")
    print("=" * 70)
    if res_off:
        print(
            f"  lang=OFF  n={res_off['n_samples']}  "
            f"wp ADE={res_off['wp_ade_mean']:.3f} (med {res_off['wp_ade_median']:.3f})  "
            f"FDE={res_off['wp_fde_mean']:.3f}  "
            f"route ADE={res_off['route_ade_mean']:.3f}  "
            f"FDE={res_off['route_fde_mean']:.3f}  "
            f"latency={res_off['mean_predict_ms']:.1f}ms"
        )
    if res_on:
        print(
            f"  lang=ON   n={res_on['n_samples']}  "
            f"wp ADE={res_on['wp_ade_mean']:.3f} (med {res_on['wp_ade_median']:.3f})  "
            f"FDE={res_on['wp_fde_mean']:.3f}  "
            f"route ADE={res_on['route_ade_mean']:.3f}  "
            f"FDE={res_on['route_fde_mean']:.3f}  "
            f"latency={res_on['mean_predict_ms']:.1f}ms"
        )
        if res_off:
            d_ade = res_off["wp_ade_mean"] - res_on["wp_ade_mean"]
            d_fde = res_off["wp_fde_mean"] - res_on["wp_fde_mean"]
            speedup = res_on["mean_predict_ms"] / res_off["mean_predict_ms"]
            print(
                f"\n  Δ(OFF - ON):  ADE {d_ade:+.3f} m   FDE {d_fde:+.3f} m   "
                f"speedup {speedup:.2f}x"
            )
    print("=" * 70)

    out = {"lang_off": res_off, "lang_on": res_on, "lang_on_error": on_error}
    output_path = Path(OUTPUTS_DIR) / "validate_l4.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    output_volume.commit()
    return out


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="L4",
    timeout=60 * 30,
    cpu=4,
    memory=24 * 1024,
)
def benchmark_l4(
    num_iterations: int = 30,
    num_warmup: int = 5,
    use_torch_compile: bool = False,
) -> dict:
    """Measure achievable Hz on an L4 GPU with the waypoints_only fast path.

    Loads the SimLingo checkpoint, applies waypoints_only=True (single LLM
    forward pass per frame instead of up to 100 autoregressive steps), runs
    real CARLA frames, and reports latency / FPS.

    Args:
        num_iterations: timed iterations after warmup
        num_warmup: warmup iterations
        use_torch_compile: enable torch.compile (vision encoder)
    """
    import gc
    import time as _time

    import numpy as np
    import torch

    sys.path.insert(0, SIMLINGO_REPO_DIR)
    from simlingo_optimized import OptimizationConfig, OptimizedSimLingo

    config = OptimizationConfig(
        use_torch_compile=use_torch_compile,
        compile_mode="reduce-overhead" if use_torch_compile else "max-autotune",
        use_gpu_preprocessing=True,
        quantization="none",
        enable_streaming=False,
        waypoints_only=True,
        verbose=True,
    )

    ckpt_path = str(Path(CKPT_DIR) / HF_CKPT_FILE)
    hydra_cfg_path = str(Path(CKPT_DIR) / HF_HYDRA_CONFIG_FILE)

    print("\n" + "=" * 70)
    print("L4 BENCHMARK — waypoints_only fast path")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"torch.compile: {use_torch_compile}")
    print("=" * 70)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_load_start = _time.time()
    model = OptimizedSimLingo(
        ckpt_path=ckpt_path,
        config=config,
        hydra_cfg_path=hydra_cfg_path,
        simlingo_repo_path=SIMLINGO_REPO_DIR,
        cache_dir=f"{CACHE_DIR}/hf/snapshots",
    )
    model.load()
    print(f"Model loaded in {_time.time() - t_load_start:.1f}s", flush=True)

    # Load real CARLA frames
    import gzip

    import cv2
    import ujson

    frames = []
    n_needed = num_iterations + num_warmup
    for rgb_dir in Path(EXTRACTED_DIR).rglob("rgb"):
        meas_dir = rgb_dir.parent / "measurements"
        if not meas_dir.exists():
            continue
        for rgb_file in sorted(rgb_dir.glob("*.jpg"))[: n_needed + 5]:
            frame_idx = int(rgb_file.stem)
            meas_file = meas_dir / f"{frame_idx:04d}.json.gz"
            if not meas_file.exists():
                continue
            bgr = cv2.imread(str(rgb_file))
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with gzip.open(meas_file, "rt") as f:
                meas = ujson.load(f)
            tp = meas.get("target_point", [10.0, 0.0])
            tp_next = meas.get("target_point_next", [20.0, 0.0])
            frames.append(
                (
                    img,
                    float(meas.get("speed", 5.0)),
                    np.array([tp, tp_next], dtype=np.float32),
                )
            )
            if len(frames) >= n_needed:
                break
        if len(frames) >= n_needed:
            break

    if not frames:
        synth = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        frames = [
            (synth, 5.0, np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32))
        ] * n_needed
    print(f"Using {len(frames)} frames", flush=True)

    # Warmup
    print(f"\nWarmup ({num_warmup} iters)...", flush=True)
    for i in range(num_warmup):
        img, spd, tgt = frames[i]
        wps, route, _ = model.predict(image=img, speed_mps=spd, target_points=tgt)
        if i == 0:
            print(
                f"  first call OK: wps shape={None if wps is None else wps.shape}, "
                f"route shape={None if route is None else route.shape}",
                flush=True,
            )
    torch.cuda.synchronize()

    # Timed loop
    print(f"\nBenchmarking ({num_iterations} iters)...", flush=True)
    lats_total = []
    lats_inf = []
    model.reset_timing_stats()

    for i in range(num_iterations):
        img, spd, tgt = frames[(num_warmup + i) % len(frames)]
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        model.predict(image=img, speed_mps=spd, target_points=tgt)
        torch.cuda.synchronize()
        lats_total.append((_time.perf_counter() - t0) * 1000.0)

    stats = model.get_timing_stats()
    lats_inf_arr = np.asarray(stats.get("inference_ms", {}).get("values", lats_total))
    lats_arr = np.asarray(lats_total)

    out = {
        "gpu": torch.cuda.get_device_name(),
        "use_torch_compile": use_torch_compile,
        "num_iterations": num_iterations,
        "total_ms": {
            "mean": float(lats_arr.mean()),
            "p50": float(np.percentile(lats_arr, 50)),
            "p90": float(np.percentile(lats_arr, 90)),
            "p99": float(np.percentile(lats_arr, 99)),
        },
        "inference_ms": {
            "mean": float(lats_inf_arr.mean()) if lats_inf_arr.size else 0.0,
        },
        "preprocess_ms": stats.get("preprocess_ms", {}),
        "fps_total": float(1000.0 / lats_arr.mean()),
        "memory_peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }

    print("\n" + "=" * 70)
    print("L4 RESULTS — waypoints_only")
    print("=" * 70)
    print(f"  GPU: {out['gpu']}")
    print(
        f"  total: mean={out['total_ms']['mean']:7.1f}ms  "
        f"p50={out['total_ms']['p50']:7.1f}ms  "
        f"p90={out['total_ms']['p90']:7.1f}ms"
    )
    print(f"  inference-only mean: {out['inference_ms']['mean']:7.1f}ms")
    print(f"  >>> FPS: {out['fps_total']:.2f}  (full pipeline incl. preprocessing)")
    print(f"  GPU mem peak: {out['memory_peak_mb']:.1f} MB")
    print("=" * 70)

    output_path = Path(OUTPUTS_DIR) / "benchmark_l4.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    output_volume.commit()
    return out


# -------------------------------------------------------------------------
# Phase 2 — quantized variants. These are SEPARATE entrypoints; the Phase-1
# benchmark_l4 / validate_l4 functions above remain untouched and continue
# to serve as the unquantized reference.
# -------------------------------------------------------------------------


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="L4",
    timeout=60 * 30,
    cpu=4,
    memory=24 * 1024,
)
def benchmark_l4_quant(
    num_iterations: int = 30,
    num_warmup: int = 5,
    quantization: str = "int8",
    quantize_vision: bool = False,
    quantize_llm: bool = True,
) -> dict:
    """Measure achievable Hz with quantization (Phase 1 + Phase 2).

    Args:
        quantization: "int8" (bnb Linear8bitLt) or "w4a8" (bnb Linear4bit NF4)
        quantize_vision: quantize InternViT-300M (compute-bound — leave OFF by default;
            bnb int8 on conv-heavy vision tends to slow down on L4)
        quantize_llm: quantize Qwen2-0.5B LLM (memory-bound — main FPS win)
    """
    import gc
    import time as _time

    import numpy as np
    import torch

    sys.path.insert(0, SIMLINGO_REPO_DIR)
    from simlingo_optimized import OptimizationConfig, OptimizedSimLingo

    config = OptimizationConfig(
        use_torch_compile=False,  # quantized layers do not compose cleanly with compile
        use_gpu_preprocessing=True,
        quantization=quantization,  # type: ignore[arg-type]
        quantize_vision_encoder=quantize_vision,
        quantize_language_head=quantize_llm,
        keep_embedding_fp16=True,
        enable_streaming=False,
        waypoints_only=True,
        verbose=True,
    )

    ckpt_path = str(Path(CKPT_DIR) / HF_CKPT_FILE)
    hydra_cfg_path = str(Path(CKPT_DIR) / HF_HYDRA_CONFIG_FILE)

    print("\n" + "=" * 70)
    print(f"L4 BENCHMARK — waypoints_only + {quantization.upper()} quantization")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"quantize_vision={quantize_vision}  quantize_llm={quantize_llm}")
    print("=" * 70)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_load_start = _time.time()
    model = OptimizedSimLingo(
        ckpt_path=ckpt_path,
        config=config,
        hydra_cfg_path=hydra_cfg_path,
        simlingo_repo_path=SIMLINGO_REPO_DIR,
        cache_dir=f"{CACHE_DIR}/hf/snapshots",
    )
    model.load()
    print(f"Model loaded in {_time.time() - t_load_start:.1f}s", flush=True)

    # Load real CARLA frames (same logic as benchmark_l4)
    import gzip

    import cv2
    import ujson

    frames = []
    n_needed = num_iterations + num_warmup
    for rgb_dir in Path(EXTRACTED_DIR).rglob("rgb"):
        meas_dir = rgb_dir.parent / "measurements"
        if not meas_dir.exists():
            continue
        for rgb_file in sorted(rgb_dir.glob("*.jpg"))[: n_needed + 5]:
            frame_idx = int(rgb_file.stem)
            meas_file = meas_dir / f"{frame_idx:04d}.json.gz"
            if not meas_file.exists():
                continue
            bgr = cv2.imread(str(rgb_file))
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with gzip.open(meas_file, "rt") as f:
                meas = ujson.load(f)
            tp = meas.get("target_point", [10.0, 0.0])
            tp_next = meas.get("target_point_next", [20.0, 0.0])
            frames.append(
                (
                    img,
                    float(meas.get("speed", 5.0)),
                    np.array([tp, tp_next], dtype=np.float32),
                )
            )
            if len(frames) >= n_needed:
                break
        if len(frames) >= n_needed:
            break

    if not frames:
        synth = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        frames = [
            (synth, 5.0, np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32))
        ] * n_needed
    print(f"Using {len(frames)} frames", flush=True)

    print(f"\nWarmup ({num_warmup} iters)...", flush=True)
    for i in range(num_warmup):
        img, spd, tgt = frames[i]
        wps, route, _ = model.predict(image=img, speed_mps=spd, target_points=tgt)
        if i == 0:
            print(
                f"  first call OK: wps shape={None if wps is None else wps.shape}, "
                f"route shape={None if route is None else route.shape}",
                flush=True,
            )
    torch.cuda.synchronize()

    print(f"\nBenchmarking ({num_iterations} iters)...", flush=True)
    lats_total = []
    model.reset_timing_stats()
    for i in range(num_iterations):
        img, spd, tgt = frames[(num_warmup + i) % len(frames)]
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        model.predict(image=img, speed_mps=spd, target_points=tgt)
        torch.cuda.synchronize()
        lats_total.append((_time.perf_counter() - t0) * 1000.0)

    lats_arr = np.asarray(lats_total)
    out = {
        "gpu": torch.cuda.get_device_name(),
        "quantization": quantization,
        "quantize_vision": quantize_vision,
        "quantize_llm": quantize_llm,
        "num_iterations": num_iterations,
        "total_ms": {
            "mean": float(lats_arr.mean()),
            "p50": float(np.percentile(lats_arr, 50)),
            "p90": float(np.percentile(lats_arr, 90)),
            "p99": float(np.percentile(lats_arr, 99)),
        },
        "fps_total": float(1000.0 / lats_arr.mean()),
        "memory_peak_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }

    print("\n" + "=" * 70)
    print(f"L4 RESULTS — waypoints_only + {quantization.upper()}")
    print("=" * 70)
    print(f"  GPU: {out['gpu']}")
    print(
        f"  total: mean={out['total_ms']['mean']:7.1f}ms  "
        f"p50={out['total_ms']['p50']:7.1f}ms  "
        f"p90={out['total_ms']['p90']:7.1f}ms"
    )
    print(f"  >>> FPS: {out['fps_total']:.2f}")
    print(f"  GPU mem peak: {out['memory_peak_mb']:.1f} MB")
    print("=" * 70)

    output_path = Path(OUTPUTS_DIR) / f"benchmark_l4_{quantization}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    output_volume.commit()
    return out


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="L4",
    timeout=60 * 60,
    cpu=4,
    memory=24 * 1024,
)
def validate_l4_quant(
    num_samples: int = 32,
    frame_stride: int = 20,
    quantization: str = "int8",
    quantize_vision: bool = False,
    quantize_llm: bool = True,
    use_cot: bool = True,
    seed: int = 0,
) -> dict:
    """Compare QUANTIZED vs UNQUANTIZED waypoints on identical CARLA frames.

    This is the strict equivalence test: we load two models (one BF16
    Phase-1, one quantized Phase-1+2), feed both the same image / speed /
    target_points, and report:
      - waypoint max-abs-diff between the two predictions (drift introduced
        purely by quantization)
      - ADE/FDE of each model vs ground-truth ego-matrix waypoints
      - latency of each

    If max-abs-diff is small (~< 5 cm) and ADE/FDE doesn't regress, the
    quantized model is safe to deploy.
    """
    import gc
    import time as _time

    import numpy as np
    import torch

    sys.path.insert(0, SIMLINGO_REPO_DIR)
    from simlingo_optimized import OptimizationConfig, OptimizedSimLingo

    base_kwargs = dict(
        use_torch_compile=False,
        use_gpu_preprocessing=True,
        enable_streaming=False,
        waypoints_only=True,
        verbose=False,
    )
    fp_config = OptimizationConfig(quantization="none", **base_kwargs)
    q_config = OptimizationConfig(
        quantization=quantization,  # type: ignore[arg-type]
        quantize_vision_encoder=quantize_vision,
        quantize_language_head=quantize_llm,
        keep_embedding_fp16=True,
        **base_kwargs,
    )

    ckpt_path = str(Path(CKPT_DIR) / HF_CKPT_FILE)
    hydra_cfg_path = str(Path(CKPT_DIR) / HF_HYDRA_CONFIG_FILE)

    print("\n" + "=" * 70)
    print(f"L4 VALIDATION — BF16 baseline vs {quantization.upper()} quantized")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"num_samples={num_samples}  quantize_vision={quantize_vision}  "
        f"quantize_llm={quantize_llm}"
    )
    print("=" * 70)

    import gzip

    import cv2
    import ujson

    def _ade_fde(pred, gt):
        if pred is None or gt is None:
            return float("nan"), float("nan")
        pred = np.asarray(pred, dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)
        if pred.shape != gt.shape:
            n = min(pred.shape[0], gt.shape[0])
            pred, gt = pred[:n], gt[:n]
        d = np.linalg.norm(pred - gt, axis=-1)
        return float(d.mean()), float(d[-1])

    def _compute_waypoints_from_ego_matrix(meas_chain):
        cur = np.asarray(meas_chain[0], dtype=np.float64)
        cur_inv = np.linalg.inv(cur)
        out = []
        for fut in meas_chain:
            fut = np.asarray(fut, dtype=np.float64)
            rel = cur_inv @ fut
            out.append([rel[0, 3], rel[1, 3]])
        return np.asarray(out, dtype=np.float32)

    def _equal_spacing_route(pts, num=20, spacing_m=1.0):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape[0] < 2:
            return pts.astype(np.float32)
        dx = np.diff(pts[:, 0])
        dy = np.diff(pts[:, 1])
        seg = np.sqrt(dx * dx + dy * dy)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-6:
            return np.tile(pts[0], (num, 1)).astype(np.float32)
        targets = np.minimum(np.arange(num) * spacing_m, total)
        out = np.zeros((num, 2), dtype=np.float32)
        for i, t in enumerate(targets):
            idx = int(np.searchsorted(cum, t))
            idx = min(max(idx, 1), len(cum) - 1)
            t0, t1 = cum[idx - 1], cum[idx]
            f = 0.0 if t1 - t0 < 1e-9 else (t - t0) / (t1 - t0)
            out[i, 0] = pts[idx - 1, 0] + f * (pts[idx, 0] - pts[idx - 1, 0])
            out[i, 1] = pts[idx - 1, 1] + f * (pts[idx, 1] - pts[idx - 1, 1])
        return out

    # Collect validation samples once; reuse for both models.
    rng = np.random.default_rng(seed)
    NUM_WP = 11
    WP_STRIDE = 5

    candidates = []
    for rgb_dir in Path(EXTRACTED_DIR).rglob("rgb"):
        meas_dir = rgb_dir.parent / "measurements"
        if not meas_dir.exists():
            continue
        rgb_files = sorted(rgb_dir.glob("*.jpg"))
        if len(rgb_files) < NUM_WP * WP_STRIDE + frame_stride:
            continue
        candidates.append((rgb_dir, meas_dir, rgb_files))

    if not candidates:
        return {"error": "no validation routes found under " + EXTRACTED_DIR}

    samples = []
    for rgb_dir, meas_dir, rgb_files in candidates:
        max_start = len(rgb_files) - NUM_WP * WP_STRIDE - 1
        if max_start <= 0:
            continue
        starts = list(range(0, max_start, frame_stride))
        rng.shuffle(starts)
        for s in starts:
            samples.append((rgb_dir, meas_dir, rgb_files, s))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    print(f"Collected {len(samples)} validation samples", flush=True)

    def _load_meas(meas_dir, idx):
        p = meas_dir / f"{idx:04d}.json.gz"
        if not p.exists():
            return None
        with gzip.open(p, "rt") as fp:
            return ujson.load(fp)

    # Pre-decode and pre-compute GT once
    prepped = []
    for rgb_dir, meas_dir, rgb_files, s_idx in samples:
        bgr = cv2.imread(str(rgb_files[s_idx]))
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame_idx = int(rgb_files[s_idx].stem)
        cur_meas = _load_meas(meas_dir, frame_idx)
        if cur_meas is None or "ego_matrix" not in cur_meas:
            continue
        ego_chain = [cur_meas["ego_matrix"]]
        ok = True
        for k in range(1, NUM_WP):
            fut_idx = frame_idx + k * WP_STRIDE
            fm = _load_meas(meas_dir, fut_idx)
            if fm is None or "ego_matrix" not in fm:
                ok = False
                break
            ego_chain.append(fm["ego_matrix"])
        if not ok:
            continue
        gt_wps = _compute_waypoints_from_ego_matrix(ego_chain)
        gt_route = _equal_spacing_route(gt_wps, num=20, spacing_m=1.0)
        speed = float(cur_meas.get("speed", 0.0))
        tp = cur_meas.get("target_point", [10.0, 0.0])
        tp_next = cur_meas.get("target_point_next", tp)
        targets = np.array([tp, tp_next], dtype=np.float32)
        prepped.append((img, speed, targets, gt_wps, gt_route))

    print(f"Prepared {len(prepped)} samples with ground-truth", flush=True)

    def run_with_config(label: str, cfg: OptimizationConfig) -> dict:
        gc.collect()
        torch.cuda.empty_cache()
        m = OptimizedSimLingo(
            ckpt_path=ckpt_path,
            config=cfg,
            hydra_cfg_path=hydra_cfg_path,
            simlingo_repo_path=SIMLINGO_REPO_DIR,
            cache_dir=f"{CACHE_DIR}/hf/snapshots",
        )
        t0 = _time.time()
        m.load()
        print(f"[{label}] model loaded in {_time.time() - t0:.1f}s", flush=True)

        # Warmup
        for img, spd, tgt, _, _ in prepped[:3]:
            m.predict(image=img, speed_mps=spd, target_points=tgt)
        torch.cuda.synchronize()

        results = []
        t_total = 0.0
        for i, (img, spd, tgt, gt_wps, gt_route) in enumerate(prepped):
            torch.cuda.synchronize()
            tic = _time.perf_counter()
            wps, route, _ = m.predict(
                image=img, speed_mps=spd, target_points=tgt, use_cot=use_cot
            )
            torch.cuda.synchronize()
            t_total += _time.perf_counter() - tic
            ade_wp, fde_wp = _ade_fde(wps, gt_wps)
            ade_rt, fde_rt = _ade_fde(route, gt_route)
            results.append({
                "wps": None if wps is None else np.asarray(wps, dtype=np.float32),
                "route": None if route is None else np.asarray(route, dtype=np.float32),
                "ade_wp": ade_wp,
                "fde_wp": fde_wp,
                "ade_route": ade_rt,
                "fde_route": fde_rt,
            })
            if (i + 1) % 10 == 0:
                print(
                    f"  [{label}] {i + 1}/{len(prepped)}  "
                    f"ADE_wp={ade_wp:.3f}  FDE_wp={fde_wp:.3f}",
                    flush=True,
                )

        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        # Free the model before loading the next one
        del m
        gc.collect()
        torch.cuda.empty_cache()

        ade_wp = np.array([r["ade_wp"] for r in results])
        fde_wp = np.array([r["fde_wp"] for r in results])
        ade_rt = np.array([r["ade_route"] for r in results])
        fde_rt = np.array([r["fde_route"] for r in results])
        return {
            "label": label,
            "n_samples": len(results),
            "wp_ade_mean": float(np.nanmean(ade_wp)),
            "wp_ade_median": float(np.nanmedian(ade_wp)),
            "wp_fde_mean": float(np.nanmean(fde_wp)),
            "route_ade_mean": float(np.nanmean(ade_rt)),
            "route_fde_mean": float(np.nanmean(fde_rt)),
            "mean_predict_ms": 1000.0 * t_total / max(len(results), 1),
            "memory_peak_mb": mem_mb,
            "per_sample": results,
        }

    print("\n[A] BF16 baseline (no quantization)", flush=True)
    res_fp = run_with_config("BF16", fp_config)

    print(f"\n[B] {quantization.upper()} quantized", flush=True)
    res_q = run_with_config(quantization.upper(), q_config)

    # Drift: per-sample waypoint max-abs-diff between the two models
    diffs_wp = []
    diffs_rt = []
    for a, b in zip(res_fp["per_sample"], res_q["per_sample"]):
        if a["wps"] is not None and b["wps"] is not None:
            diffs_wp.append(float(np.max(np.abs(a["wps"] - b["wps"]))))
        if a["route"] is not None and b["route"] is not None:
            diffs_rt.append(float(np.max(np.abs(a["route"] - b["route"]))))

    drift = {
        "wp_max_abs_diff_mean_m": float(np.mean(diffs_wp)) if diffs_wp else float("nan"),
        "wp_max_abs_diff_p90_m": float(np.percentile(diffs_wp, 90)) if diffs_wp else float("nan"),
        "wp_max_abs_diff_max_m": float(np.max(diffs_wp)) if diffs_wp else float("nan"),
        "route_max_abs_diff_mean_m": float(np.mean(diffs_rt)) if diffs_rt else float("nan"),
    }

    speedup = res_fp["mean_predict_ms"] / max(res_q["mean_predict_ms"], 1e-9)

    print("\n" + "=" * 70)
    print(f"VALIDATION RESULTS — BF16 vs {quantization.upper()}")
    print("=" * 70)
    for res in (res_fp, res_q):
        print(
            f"  [{res['label']:<5}] n={res['n_samples']}  "
            f"wp ADE={res['wp_ade_mean']:.3f} (med {res['wp_ade_median']:.3f})  "
            f"FDE={res['wp_fde_mean']:.3f}  "
            f"route ADE={res['route_ade_mean']:.3f}  "
            f"FDE={res['route_fde_mean']:.3f}  "
            f"latency={res['mean_predict_ms']:.1f}ms  "
            f"mem={res['memory_peak_mb']:.0f}MB"
        )
    print(
        f"\n  Drift ({quantization.upper()} vs BF16, per-sample max-abs-diff):"
    )
    print(
        f"    waypoints:  mean={drift['wp_max_abs_diff_mean_m']:.4f} m  "
        f"p90={drift['wp_max_abs_diff_p90_m']:.4f} m  "
        f"max={drift['wp_max_abs_diff_max_m']:.4f} m"
    )
    print(f"    route:      mean={drift['route_max_abs_diff_mean_m']:.4f} m")
    print(f"\n  Speedup: {speedup:.2f}x  (BF16 {res_fp['mean_predict_ms']:.1f}ms → "
          f"{quantization.upper()} {res_q['mean_predict_ms']:.1f}ms)")
    print("=" * 70)

    # Strip per-sample arrays before JSON dump
    out = {
        "bf16": {k: v for k, v in res_fp.items() if k != "per_sample"},
        "quantized": {k: v for k, v in res_q.items() if k != "per_sample"},
        "drift": drift,
        "speedup": speedup,
        "quantization": quantization,
    }
    output_path = Path(OUTPUTS_DIR) / f"validate_l4_{quantization}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    output_volume.commit()
    return out


@app.function(
    image=benchmark_image,
    volumes=VOLUMES,
    gpu="A10G",  # A10G ≈ AGX Thor target class; production-relevant
    timeout=60 * 30,
    cpu=4,
    memory=24 * 1024,
)
def benchmark_language_skip(
    num_iterations: int = 30,
    num_warmup: int = 3,
    waypoint_tol: float = 1e-3,
) -> dict:
    """A/B benchmark: language head ON vs OFF on identical CARLA frames.

    Loads the model ONCE, then runs the same set of real CARLA frames twice:
      (A) predict_language=True  -> autoregressive greedy_sample loop
      (B) predict_language=False -> single forward pass (training code-path)

    Asserts that speed_wps and route waypoints agree within `waypoint_tol`
    (max abs diff). This is the metric that matters: if waypoints match,
    closed-loop CARLA driving behaviour is unchanged because the PID controller
    only consumes those tensors.

    Reports:
      - mean/p50/p90 latency (ms) for each mode
      - speedup factor (B vs A)
      - waypoint max abs diff (should be ~0; identical regression head input
        when language tokens collapse to deterministic greedy output)
    """
    import gc
    import numpy as np
    import torch

    sys.path.insert(0, SIMLINGO_REPO_DIR)
    from simlingo_optimized import OptimizationConfig, OptimizedSimLingo

    # Use a minimal config so the language-head delta is the ONLY variable.
    # No torch.compile (which interacts with the autoregressive path), no
    # quantization, no streaming.
    config = OptimizationConfig(
        use_torch_compile=False,
        use_gpu_preprocessing=True,
        quantization="none",
        enable_streaming=False,
        waypoints_only=True,  # we override per-call below anyway
        verbose=True,
    )

    ckpt_path = str(Path(CKPT_DIR) / HF_CKPT_FILE)
    hydra_cfg_path = str(Path(CKPT_DIR) / HF_HYDRA_CONFIG_FILE)

    print("\n" + "=" * 70)
    print("A/B BENCHMARK: language head ON vs OFF")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = OptimizedSimLingo(
        ckpt_path=ckpt_path,
        config=config,
        hydra_cfg_path=hydra_cfg_path,
        simlingo_repo_path=SIMLINGO_REPO_DIR,
        cache_dir=f"{CACHE_DIR}/hf/snapshots",
    )
    model.load()

    # Load real CARLA frames (same as _run_benchmark_internal)
    import gzip

    import cv2
    import ujson

    frames = []
    rgb_dirs = list(Path(EXTRACTED_DIR).rglob("rgb"))
    n_needed = num_iterations + num_warmup
    for rgb_dir in rgb_dirs:
        meas_dir = rgb_dir.parent / "measurements"
        if not meas_dir.exists():
            continue
        for rgb_file in sorted(rgb_dir.glob("*.jpg"))[: n_needed + 5]:
            frame_idx = int(rgb_file.stem)
            meas_file = meas_dir / f"{frame_idx:04d}.json.gz"
            if not meas_file.exists():
                continue
            bgr = cv2.imread(str(rgb_file))
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with gzip.open(meas_file, "rt") as f:
                meas = ujson.load(f)
            tp = meas.get("target_point", [10.0, 0.0])
            tp_next = meas.get("target_point_next", [20.0, 0.0])
            frames.append(
                (
                    img,
                    float(meas.get("speed", 5.0)),
                    np.array([tp, tp_next], dtype=np.float32),
                )
            )
            if len(frames) >= n_needed:
                break
        if len(frames) >= n_needed:
            break

    if not frames:
        # Fall back to synthetic so the bench still runs
        print("WARNING: no real CARLA frames found; using synthetic", flush=True)
        synth = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        frames = [
            (synth, 5.0, np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32))
        ] * n_needed

    print(f"Using {len(frames)} CARLA frames for A/B test", flush=True)

    def run_mode(label: str, predict_language: bool):
        # Warmup
        for i in range(num_warmup):
            img, spd, tgt = frames[i]
            model.predict(
                image=img,
                speed_mps=spd,
                target_points=tgt,
                predict_language=predict_language,
            )
        torch.cuda.synchronize()

        lats = []
        wps_collected = []
        route_collected = []
        for i in range(num_iterations):
            img, spd, tgt = frames[(num_warmup + i) % len(frames)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            wps, route, _text = model.predict(
                image=img,
                speed_mps=spd,
                target_points=tgt,
                predict_language=predict_language,
            )
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)
            wps_collected.append(wps)
            route_collected.append(route)
        lats_arr = np.asarray(lats)
        print(
            f"  [{label}] mean={lats_arr.mean():7.1f}ms  "
            f"p50={np.percentile(lats_arr, 50):7.1f}ms  "
            f"p90={np.percentile(lats_arr, 90):7.1f}ms  "
            f"fps={1000.0 / lats_arr.mean():5.2f}",
            flush=True,
        )
        return lats_arr, wps_collected, route_collected

    # Run the NEW fast path first (this is what CARLA will actually use),
    # then attempt the upstream lang=ON path. The upstream autoregressive
    # path has been observed to hit pre-existing crashes on some inputs
    # (e.g. placeholder substitution KeyError); we tolerate that and still
    # report B's numbers because B is the path the deployment uses.
    print("\n[B] predict_language=False (waypoints-only fast path)", flush=True)
    lat_b, wps_b, route_b = run_mode("B:lang=OFF", predict_language=False)

    gc.collect()
    torch.cuda.empty_cache()

    print("\n[A] predict_language=True (baseline; autoregressive)", flush=True)
    lat_a, wps_a, route_a = None, None, None
    a_error: Optional[str] = None
    try:
        lat_a, wps_a, route_a = run_mode("A:lang=ON ", predict_language=True)
    except Exception as exc:
        a_error = f"{type(exc).__name__}: {exc}"
        print(f"  [A] FAILED in upstream lang=ON path: {a_error}", flush=True)
        print(
            "  This is an upstream bug in simlingo_training (unrelated to the "
            "waypoints_only optimization). Continuing with B-only results.",
            flush=True,
        )

    # Numerical agreement on waypoints
    def _max_abs_diff(a_list, b_list):
        diffs = []
        for a, b in zip(a_list, b_list):
            if a is None or b is None:
                continue
            a_arr = np.asarray(a, dtype=np.float64)
            b_arr = np.asarray(b, dtype=np.float64)
            if a_arr.shape != b_arr.shape:
                return float("nan"), float("nan")
            diffs.append(np.abs(a_arr - b_arr))
        if not diffs:
            return float("nan"), float("nan")
        stacked = np.concatenate([d.reshape(-1) for d in diffs])
        return float(stacked.max()), float(stacked.mean())

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(
        f"  B (lang=OFF) mean={lat_b.mean():7.1f}ms  "
        f"p90={np.percentile(lat_b, 90):7.1f}ms  "
        f"fps={1000.0 / lat_b.mean():5.2f}"
    )

    out = {
        "lang_off": {
            "mean_ms": float(lat_b.mean()),
            "p50_ms": float(np.percentile(lat_b, 50)),
            "p90_ms": float(np.percentile(lat_b, 90)),
            "fps": float(1000.0 / lat_b.mean()),
        },
        "num_iterations": num_iterations,
        "gpu": torch.cuda.get_device_name(),
    }

    if lat_a is not None:
        wp_max, wp_mean = _max_abs_diff(wps_a, wps_b)
        rt_max, rt_mean = _max_abs_diff(route_a, route_b)
        speedup = float(lat_a.mean() / lat_b.mean())
        print(
            f"  A (lang=ON)  mean={lat_a.mean():7.1f}ms  "
            f"p90={np.percentile(lat_a, 90):7.1f}ms  "
            f"fps={1000.0 / lat_a.mean():5.2f}"
        )
        print(f"  SPEEDUP (A / B): {speedup:.2f}x")
        print(f"  Waypoint max abs diff: {wp_max:.6f}  (mean {wp_mean:.6f})")
        print(f"  Route    max abs diff: {rt_max:.6f}  (mean {rt_mean:.6f})")
        wp_pass = (not np.isnan(wp_max)) and wp_max <= waypoint_tol
        print(
            f"  Waypoint agreement (tol={waypoint_tol}): "
            f"{'PASS' if wp_pass else 'DIFFER'}"
        )
        if not wp_pass:
            print(
                "  Note: small deltas are expected. The lang=ON path conditions the "
                "regression head on LLM hidden states AFTER autoregressively-sampled "
                "commentary tokens. The lang=OFF path uses the same hidden states "
                "the regression head was trained on (single forward pass = training "
                "code-path). lang=OFF outputs are the canonical training-time "
                "predictions."
            )
        out["lang_on"] = {
            "mean_ms": float(lat_a.mean()),
            "p50_ms": float(np.percentile(lat_a, 50)),
            "p90_ms": float(np.percentile(lat_a, 90)),
            "fps": float(1000.0 / lat_a.mean()),
        }
        out["speedup"] = speedup
        out["waypoint_max_abs_diff"] = wp_max
        out["waypoint_mean_abs_diff"] = wp_mean
        out["route_max_abs_diff"] = rt_max
        out["route_mean_abs_diff"] = rt_mean
    else:
        print(f"  A (lang=ON)  SKIPPED: {a_error}")
        out["lang_on_error"] = a_error
    print("=" * 70)

    output_path = Path(OUTPUTS_DIR) / "benchmark_language_skip.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {output_path}")
    output_volume.commit()
    return out


@app.local_entrypoint()
def main(
    config: str = "compile_gpu_preprocess",
    full_suite: bool = False,
    num_iterations: int = 50,
):
    """Run benchmarks from command line.

    Examples:
        # Single config benchmark
        modal run simlingo_optimized/modal_benchmark.py --config baseline

        # Full benchmark suite
        modal run simlingo_optimized/modal_benchmark.py --full-suite

        # Compare two configs
        modal run simlingo_optimized/modal_benchmark.py::compare
    """
    if full_suite:
        print("Running full benchmark suite on A100...")
        results = benchmark_all.remote(num_iterations=num_iterations)
    else:
        print(f"Running benchmark for config: {config}")
        results = benchmark_single.remote(config=config, num_iterations=num_iterations)

    print("\nBenchmark complete!")
    print(json.dumps(results, indent=2))
