"""Modal script to test TensorRT optimization for SimLingo inference.

This script:
1. Loads the SimLingo model
2. Compiles vision encoder with torch_tensorrt
3. Benchmarks TensorRT vs PyTorch inference

Usage:
    modal run simlingo_optimized/modal_trtllm.py
    modal run simlingo_optimized/modal_trtllm.py --check-only  # Check TRT support
"""

import modal

# Create Modal app
app = modal.App("simlingo-trtllm")

# Use PyTorch NGC container with TensorRT support
trt_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:24.08-py3",
    )
    .apt_install("git", "wget")
    # NGC container ships a broken cv2 (cv2.typing references missing DictValue).
    # Uninstall it first so our pinned opencv-python-headless takes over cleanly.
    .run_commands(
        "pip uninstall -y opencv opencv-python opencv-python-headless opencv-contrib-python || true",
        "rm -rf /usr/local/lib/python3.10/dist-packages/cv2",
    )
    .pip_install(
        # Pin to exact versions used by the working modal_benchmark.py setup
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "accelerate==1.0.1",
        "peft==0.13.2",
        "sentencepiece",
        "protobuf",
        "einops==0.7.0",
        "timm==0.9.16",
        "onnx",
        "numpy<2",
        "pillow==10.2.0",
        "opencv-python-headless==4.10.0.84",
        "hydra-core==1.3.2",
        "hydra-zen==0.12.1",
        "omegaconf==2.3.0",
        "lightning==2.3.3",
        "pytorch-lightning==2.4.0",
        "huggingface_hub==0.27.0",
        "ujson==5.9.0",
        "filterpy==1.4.5",
        # torch_tensorrt is pre-installed in the PyTorch NGC container
    )
    # Clone simlingo repo so simlingo_training is importable
    .run_commands(
        "git clone --depth 1 https://github.com/RenzKa/simlingo.git /opt/simlingo",
    )
    .env({"PYTHONPATH": "/opt/simlingo"})
    # Mount our simlingo_optimized module
    .add_local_python_source("simlingo_optimized")
)

# Volume for model storage (matching modal_benchmark.py)
vol = modal.Volume.from_name("simlingo-data", create_if_missing=True)

VOLUMES = {
    "/data": vol,
    "/cache": modal.Volume.from_name("simlingo-cache", create_if_missing=True),
}

CKPT_DIR = "/data/checkpoint"
CKPT_FILE = "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
HYDRA_CFG_FILE = "simlingo/.hydra/config.yaml"
SIMLINGO_REPO = "/opt/simlingo"
OUTPUT_DIR = "/data/trt_engines"


@app.function(
    image=trt_image,
    volumes=VOLUMES,
    gpu="A100",
    timeout=60 * 60,
    memory=48 * 1024,
)
def benchmark_trt():
    """Benchmark SimLingo with torch_tensorrt optimization."""
    import os
    import sys
    import time

    import numpy as np
    import torch

    # Check dependencies
    print("=" * 60)
    print("Checking dependencies...")
    print("=" * 60)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import tensorrt as trt
        print(f"TensorRT: {trt.__version__}")
    except ImportError:
        print("TensorRT: NOT AVAILABLE")

    try:
        import torch_tensorrt
        print(f"torch_tensorrt: {torch_tensorrt.__version__}")
    except ImportError:
        print("torch_tensorrt: NOT AVAILABLE")

    print()

    # Add simlingo to path
    if os.path.exists(SIMLINGO_REPO):
        sys.path.insert(0, SIMLINGO_REPO)
        print(f"Added {SIMLINGO_REPO} to path")

    # Load model
    print("=" * 60)
    print("Loading SimLingo model...")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check if checkpoint exists
    ckpt_path = f"{CKPT_DIR}/{CKPT_FILE}"
    hydra_cfg_path = f"{CKPT_DIR}/{HYDRA_CFG_FILE}"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return {"error": "Checkpoint not found"}
    if not os.path.exists(hydra_cfg_path):
        print(f"Hydra config not found: {hydra_cfg_path}")
        return {"error": "Hydra config not found"}

    # Import and load model
    try:
        from simlingo_optimized import OptimizedSimLingo, OptimizationConfig

        config = OptimizationConfig(
            use_torch_compile=False,  # We'll manually apply TRT
            use_gpu_preprocessing=True,
            verbose=True,
        )

        model = OptimizedSimLingo(
            ckpt_path=ckpt_path,
            config=config,
            hydra_cfg_path=hydra_cfg_path,
            simlingo_repo_path=SIMLINGO_REPO,
            cache_dir="/cache/hf/snapshots",
        )
        model.load()
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    # Test data
    test_image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
    test_speed = 5.0
    test_targets = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32)

    # Benchmark PyTorch baseline
    print()
    print("=" * 60)
    print("Benchmarking PyTorch baseline...")
    print("=" * 60)

    # Warmup
    for _ in range(5):
        _ = model.predict(test_image, test_speed, test_targets)

    torch.cuda.synchronize()

    # Benchmark
    latencies = []
    for i in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(test_image, test_speed, test_targets)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    pytorch_mean = np.mean(latencies)
    pytorch_p50 = np.percentile(latencies, 50)
    pytorch_p90 = np.percentile(latencies, 90)
    pytorch_fps = 1000 / pytorch_mean
    print(f"  Mean latency: {pytorch_mean:.1f} ms")
    print(f"  P50 latency:  {pytorch_p50:.1f} ms")
    print(f"  P90 latency:  {pytorch_p90:.1f} ms")
    print(f"  FPS: {pytorch_fps:.2f}")

    # Try torch.compile with TensorRT backend
    print()
    print("=" * 60)
    print("Trying torch.compile with tensorrt backend...")
    print("=" * 60)

    trt_results = None
    try:
        import torch_tensorrt

        # Get the vision encoder
        vision_encoder = model._model.vision_model.image_encoder

        # Try compiling with torch_tensorrt
        print("Compiling vision encoder with torch_tensorrt...")

        # Create sample input
        sample_input = torch.randn(1, 3, 448, 448, device="cuda", dtype=torch.float16)

        # Compile with TensorRT
        trt_vision = torch_tensorrt.compile(
            vision_encoder,
            inputs=[sample_input],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,  # 1GB
            truncate_long_and_double=True,
        )

        # Replace the vision encoder
        model._model.vision_model.image_encoder = trt_vision
        print("Vision encoder compiled with TensorRT!")

        # Warmup
        for _ in range(5):
            _ = model.predict(test_image, test_speed, test_targets)

        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for i in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.predict(test_image, test_speed, test_targets)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        trt_mean = np.mean(latencies)
        trt_p50 = np.percentile(latencies, 50)
        trt_p90 = np.percentile(latencies, 90)
        trt_fps = 1000 / trt_mean
        speedup = pytorch_mean / trt_mean

        print(f"  Mean latency: {trt_mean:.1f} ms")
        print(f"  P50 latency:  {trt_p50:.1f} ms")
        print(f"  P90 latency:  {trt_p90:.1f} ms")
        print(f"  FPS: {trt_fps:.2f}")
        print(f"  Speedup vs PyTorch: {speedup:.2f}x")

        trt_results = {
            "trt_latency_ms": trt_mean,
            "trt_p50_ms": trt_p50,
            "trt_p90_ms": trt_p90,
            "trt_fps": trt_fps,
            "speedup": speedup,
        }

    except Exception as e:
        print(f"torch_tensorrt compilation failed: {e}")
        import traceback
        traceback.print_exc()

    # Try torch.compile with inductor backend (dynamo)
    print()
    print("=" * 60)
    print("Trying torch.compile with inductor backend...")
    print("=" * 60)

    inductor_results = None
    try:
        # Reload model fresh
        model2 = OptimizedSimLingo(
            ckpt_path=ckpt_path,
            config=OptimizationConfig(
                use_torch_compile=True,  # Use torch.compile
                compile_mode="max-autotune",
                use_gpu_preprocessing=True,
                verbose=False,
            ),
            hydra_cfg_path=hydra_cfg_path,
            simlingo_repo_path=SIMLINGO_REPO,
            cache_dir="/cache/hf/snapshots",
        )
        model2.load()

        # Warmup (includes compilation)
        print("Warming up (includes compilation)...")
        for _ in range(5):
            _ = model2.predict(test_image, test_speed, test_targets)

        torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for i in range(20):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model2.predict(test_image, test_speed, test_targets)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        ind_mean = np.mean(latencies)
        ind_p50 = np.percentile(latencies, 50)
        ind_p90 = np.percentile(latencies, 90)
        ind_fps = 1000 / ind_mean
        speedup = pytorch_mean / ind_mean

        print(f"  Mean latency: {ind_mean:.1f} ms")
        print(f"  P50 latency:  {ind_p50:.1f} ms")
        print(f"  P90 latency:  {ind_p90:.1f} ms")
        print(f"  FPS: {ind_fps:.2f}")
        print(f"  Speedup vs PyTorch: {speedup:.2f}x")

        inductor_results = {
            "inductor_latency_ms": ind_mean,
            "inductor_p50_ms": ind_p50,
            "inductor_p90_ms": ind_p90,
            "inductor_fps": ind_fps,
            "speedup": speedup,
        }

    except Exception as e:
        print(f"torch.compile with inductor failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PyTorch baseline:  {pytorch_mean:.1f} ms ({pytorch_fps:.2f} FPS)")
    if trt_results:
        print(f"TensorRT vision:   {trt_results['trt_latency_ms']:.1f} ms ({trt_results['trt_fps']:.2f} FPS) - {trt_results['speedup']:.2f}x")
    if inductor_results:
        print(f"torch.compile:     {inductor_results['inductor_latency_ms']:.1f} ms ({inductor_results['inductor_fps']:.2f} FPS) - {inductor_results['speedup']:.2f}x")

    # Commit volume
    vol.commit()

    results = {
        "pytorch_latency_ms": pytorch_mean,
        "pytorch_p50_ms": pytorch_p50,
        "pytorch_p90_ms": pytorch_p90,
        "pytorch_fps": pytorch_fps,
    }
    if trt_results:
        results.update(trt_results)
    if inductor_results:
        results.update(inductor_results)

    return results


@app.function(
    image=trt_image,
    volumes=VOLUMES,
    gpu="A100",
    timeout=30 * 60,
)
def check_trt_support():
    """Check TensorRT and torch_tensorrt support."""
    import torch

    print("Checking TensorRT support...")
    print()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results = {}

    try:
        import tensorrt as trt
        print(f"TensorRT: {trt.__version__}")
        results["tensorrt"] = trt.__version__
    except ImportError as e:
        print(f"TensorRT: NOT AVAILABLE ({e})")
        results["tensorrt"] = None

    try:
        import torch_tensorrt
        print(f"torch_tensorrt: {torch_tensorrt.__version__}")
        results["torch_tensorrt"] = torch_tensorrt.__version__

        # Check supported backends
        print("\ntorch.compile backends available:")
        print("  - inductor (default)")
        print("  - tensorrt (via torch_tensorrt)")

    except ImportError as e:
        print(f"torch_tensorrt: NOT AVAILABLE ({e})")
        results["torch_tensorrt"] = None

    # Check if ONNX export works
    try:
        import onnx
        print(f"\nONNX: {onnx.__version__}")
        results["onnx"] = onnx.__version__
    except ImportError:
        print("\nONNX: NOT AVAILABLE")
        results["onnx"] = None

    return results


@app.local_entrypoint()
def main(check_only: bool = False):
    """Main entry point."""
    if check_only:
        print("Checking TensorRT support...")
        result = check_trt_support.remote()
    else:
        print("Benchmarking SimLingo with TensorRT optimizations...")
        result = benchmark_trt.remote()

    print("\nResult:")
    print(result)
