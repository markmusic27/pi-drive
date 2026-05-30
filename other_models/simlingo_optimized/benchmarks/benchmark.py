"""Performance benchmarking for SimLingo optimizations.

This module provides comprehensive benchmarking tools to measure:
- Inference latency (mean, p50, p90, p99)
- Throughput (frames per second)
- Memory usage
- Accuracy metrics (ADE/FDE compared to baseline)

Usage:
    python -m simlingo_optimized.benchmarks.benchmark \
        --ckpt /path/to/checkpoint.pt \
        --config fast_compile \
        --num-iterations 100
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""

    # Model settings
    ckpt_path: str = ""
    hydra_cfg_path: Optional[str] = None

    # Optimization preset
    optimization_preset: Literal[
        "baseline", "fast_compile", "quantized", "full_optimized"
    ] = "fast_compile"

    # Benchmark parameters
    num_warmup: int = 5
    num_iterations: int = 100
    batch_size: int = 1

    # Input settings
    image_size: tuple = (480, 640)  # H, W
    use_real_images: bool = False
    image_dir: Optional[str] = None

    # Output settings
    output_dir: str = "./benchmark_results"
    save_detailed: bool = True

    # Device settings
    device: str = "cuda"
    sync_cuda: bool = True

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    # Timing (milliseconds)
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_std_ms: float = 0.0

    # Throughput
    throughput_fps: float = 0.0

    # Memory (MB)
    memory_peak_mb: float = 0.0
    memory_allocated_mb: float = 0.0

    # Breakdown
    preprocess_mean_ms: float = 0.0
    model_forward_mean_ms: float = 0.0
    postprocess_mean_ms: float = 0.0

    # Configuration
    config_name: str = ""
    num_iterations: int = 0

    # Raw timing data
    latencies_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_std_ms": self.latency_std_ms,
            "throughput_fps": self.throughput_fps,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_allocated_mb": self.memory_allocated_mb,
            "preprocess_mean_ms": self.preprocess_mean_ms,
            "model_forward_mean_ms": self.model_forward_mean_ms,
            "postprocess_mean_ms": self.postprocess_mean_ms,
            "config_name": self.config_name,
            "num_iterations": self.num_iterations,
        }

    def print_summary(self):
        """Print formatted summary."""
        print(f"\n{'='*60}")
        print(f"Benchmark Results: {self.config_name}")
        print(f"{'='*60}")
        print(f"Iterations: {self.num_iterations}")
        print(f"\nLatency:")
        print(f"  Mean:  {self.latency_mean_ms:7.2f} ms")
        print(f"  P50:   {self.latency_p50_ms:7.2f} ms")
        print(f"  P90:   {self.latency_p90_ms:7.2f} ms")
        print(f"  P99:   {self.latency_p99_ms:7.2f} ms")
        print(f"  Std:   {self.latency_std_ms:7.2f} ms")
        print(f"\nThroughput: {self.throughput_fps:.2f} FPS")
        print(f"\nMemory:")
        print(f"  Peak:      {self.memory_peak_mb:.1f} MB")
        print(f"  Allocated: {self.memory_allocated_mb:.1f} MB")
        if self.preprocess_mean_ms > 0:
            print(f"\nBreakdown:")
            print(f"  Preprocess:  {self.preprocess_mean_ms:6.2f} ms")
            print(f"  Model:       {self.model_forward_mean_ms:6.2f} ms")
            print(f"  Postprocess: {self.postprocess_mean_ms:6.2f} ms")
        print(f"{'='*60}\n")


def run_benchmark(
    config: BenchmarkConfig,
    model: Optional[Any] = None,
) -> BenchmarkResult:
    """Run benchmark with given configuration.

    Args:
        config: Benchmark configuration
        model: Optional pre-loaded model. If None, creates new model.

    Returns:
        BenchmarkResult with timing and memory statistics
    """
    from simlingo_optimized import OptimizedSimLingo, OptimizationConfig

    # Create optimization config from preset
    if config.optimization_preset == "baseline":
        opt_config = OptimizationConfig(
            use_torch_compile=False,
            use_gpu_preprocessing=False,
            quantization="none",
        )
    elif config.optimization_preset == "fast_compile":
        opt_config = OptimizationConfig.fast_compile()
    elif config.optimization_preset == "quantized":
        opt_config = OptimizationConfig.quantized()
    elif config.optimization_preset == "full_optimized":
        opt_config = OptimizationConfig.full_optimized()
    else:
        raise ValueError(f"Unknown preset: {config.optimization_preset}")

    opt_config.verbose = False
    opt_config.device = config.device

    # Create or use provided model
    if model is None:
        if not config.ckpt_path:
            raise ValueError("Must provide ckpt_path or pre-loaded model")

        print(f"Loading model with config: {config.optimization_preset}")
        model = OptimizedSimLingo(
            ckpt_path=config.ckpt_path,
            config=opt_config,
            hydra_cfg_path=config.hydra_cfg_path,
        )
        model.load()

    # Clear GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Create dummy input
    H, W = config.image_size
    dummy_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    dummy_speed = 5.0
    dummy_targets = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32)

    # Warmup
    print(f"Warming up ({config.num_warmup} iterations)...")
    for _ in range(config.num_warmup):
        _ = model.predict(
            image=dummy_image,
            speed_mps=dummy_speed,
            target_points=dummy_targets,
        )

    if config.sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Reset timing stats
    model.reset_timing_stats()

    # Benchmark
    print(f"Benchmarking ({config.num_iterations} iterations)...")
    latencies = []

    for i in range(config.num_iterations):
        if config.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        _ = model.predict(
            image=dummy_image,
            speed_mps=dummy_speed,
            target_points=dummy_targets,
        )

        if config.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{config.num_iterations}")

    # Compute statistics
    latencies_arr = np.array(latencies)
    timing_stats = model.get_timing_stats()

    result = BenchmarkResult(
        latency_mean_ms=float(latencies_arr.mean()),
        latency_p50_ms=float(np.percentile(latencies_arr, 50)),
        latency_p90_ms=float(np.percentile(latencies_arr, 90)),
        latency_p99_ms=float(np.percentile(latencies_arr, 99)),
        latency_std_ms=float(latencies_arr.std()),
        throughput_fps=1000.0 / float(latencies_arr.mean()),
        config_name=config.optimization_preset,
        num_iterations=config.num_iterations,
        latencies_ms=latencies,
    )

    # Memory stats
    if torch.cuda.is_available():
        result.memory_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        result.memory_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    # Breakdown from model timing
    if "preprocess_ms" in timing_stats:
        result.preprocess_mean_ms = timing_stats["preprocess_ms"]["mean"]
    if "inference_ms" in timing_stats:
        result.model_forward_mean_ms = timing_stats["inference_ms"]["mean"]

    return result


def compare_configs(
    ckpt_path: str,
    configs: List[str] = ["baseline", "fast_compile", "quantized"],
    num_iterations: int = 50,
    output_dir: str = "./benchmark_results",
) -> Dict[str, BenchmarkResult]:
    """Compare multiple optimization configurations.

    Args:
        ckpt_path: Path to model checkpoint
        configs: List of optimization presets to compare
        num_iterations: Number of benchmark iterations per config
        output_dir: Directory for results

    Returns:
        Dictionary mapping config names to results
    """
    results = {}

    for config_name in configs:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config_name}")
        print(f"{'='*60}")

        # Clear memory between configs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        benchmark_config = BenchmarkConfig(
            ckpt_path=ckpt_path,
            optimization_preset=config_name,
            num_iterations=num_iterations,
            output_dir=output_dir,
        )

        try:
            result = run_benchmark(benchmark_config)
            result.print_summary()
            results[config_name] = result
        except Exception as e:
            print(f"Config {config_name} failed: {e}")
            results[config_name] = None

    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'Mean (ms)':<12} {'P90 (ms)':<12} {'FPS':<10} {'Memory (MB)':<12}")
    print("-" * 80)

    baseline_fps = None
    for name, result in results.items():
        if result is not None:
            if baseline_fps is None:
                baseline_fps = result.throughput_fps
            speedup = result.throughput_fps / baseline_fps if baseline_fps else 1.0
            print(
                f"{name:<20} {result.latency_mean_ms:<12.2f} {result.latency_p90_ms:<12.2f} "
                f"{result.throughput_fps:<10.2f} {result.memory_peak_mb:<12.1f} "
                f"({speedup:.2f}x)"
            )
        else:
            print(f"{name:<20} {'FAILED':<12}")

    print("=" * 80)

    # Save results
    output_path = Path(output_dir) / "comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {name: r.to_dict() if r else None for name, r in results.items()},
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")

    return results


def profile_model(
    model: Any,
    num_iterations: int = 10,
    use_nsys: bool = False,
) -> Dict[str, Any]:
    """Profile model execution with detailed breakdown.

    Args:
        model: OptimizedSimLingo model
        num_iterations: Number of profiling iterations
        use_nsys: Use NVIDIA Nsight Systems profiling

    Returns:
        Dictionary with profiling results
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Profiling requires CUDA")

    # Create dummy input
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_speed = 5.0
    dummy_targets = np.array([[10.0, 0.0], [20.0, 0.0]], dtype=np.float32)

    # Profile with PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        for _ in range(num_iterations):
            _ = model.predict(
                image=dummy_image,
                speed_mps=dummy_speed,
                target_points=dummy_targets,
            )
            profiler.step()

    # Extract key metrics
    key_averages = profiler.key_averages()

    results = {
        "top_operations": [],
        "cuda_time_total_ms": 0,
        "cpu_time_total_ms": 0,
    }

    for event in key_averages:
        if event.cuda_time_total > 0:
            results["top_operations"].append({
                "name": event.key,
                "cuda_time_ms": event.cuda_time_total / 1000 / num_iterations,
                "cpu_time_ms": event.cpu_time_total / 1000 / num_iterations,
                "count": event.count // num_iterations,
            })
            results["cuda_time_total_ms"] += event.cuda_time_total / 1000 / num_iterations
            results["cpu_time_total_ms"] += event.cpu_time_total / 1000 / num_iterations

    # Sort by CUDA time
    results["top_operations"].sort(
        key=lambda x: x["cuda_time_ms"], reverse=True
    )
    results["top_operations"] = results["top_operations"][:20]

    # Print profile table
    print(profiler.key_averages().table(
        sort_by="cuda_time_total", row_limit=20
    ))

    return results


def main():
    """Main entry point for benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark SimLingo optimization configurations"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--hydra-cfg", type=str, default=None, help="Path to Hydra config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="fast_compile",
        choices=["baseline", "fast_compile", "quantized", "full_optimized"],
        help="Optimization preset",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare all configurations"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./benchmark_results", help="Output directory"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run detailed profiling"
    )

    args = parser.parse_args()

    if args.compare:
        compare_configs(
            ckpt_path=args.ckpt,
            num_iterations=args.num_iterations,
            output_dir=args.output_dir,
        )
    else:
        config = BenchmarkConfig(
            ckpt_path=args.ckpt,
            hydra_cfg_path=args.hydra_cfg,
            optimization_preset=args.config,
            num_iterations=args.num_iterations,
            num_warmup=args.num_warmup,
            output_dir=args.output_dir,
        )

        result = run_benchmark(config)
        result.print_summary()

        if args.profile:
            from simlingo_optimized import OptimizedSimLingo, OptimizationConfig

            opt_config = OptimizationConfig.fast_compile()
            model = OptimizedSimLingo(
                ckpt_path=args.ckpt,
                config=opt_config,
            )
            model.load()
            profile_model(model, num_iterations=10)


if __name__ == "__main__":
    main()
