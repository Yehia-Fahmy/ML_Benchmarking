"""
Shared utilities for ML Benchmarking suite.
Provides common functionality for CUDA-only image generation benchmarks.
"""

from .config import (
    MODEL_CONFIGS,
    DEFAULT_PROMPTS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_SEED,
    OUTPUT_DIR,
    BENCHMARKS_DIR,
    get_model_config,
)
from .device import (
    detect_cuda_device,
    get_gpu_memory_usage,
    cleanup_memory,
    require_cuda,
)
from .benchmark import (
    BenchmarkResult,
    GenerationResult,
    run_benchmark,
    calculate_summary_stats,
)
from .validation import (
    validate_dimensions,
    validate_model_support,
    get_supported_optimizations,
    apply_optimizations,
)
from .export import (
    save_benchmark_results,
    save_single_result,
    print_summary,
    aggregate_results,
)

__all__ = [
    # Config
    "MODEL_CONFIGS",
    "DEFAULT_PROMPTS",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "DEFAULT_SEED",
    "OUTPUT_DIR",
    "BENCHMARKS_DIR",
    "get_model_config",
    # Device
    "detect_cuda_device",
    "get_gpu_memory_usage",
    "cleanup_memory",
    "require_cuda",
    # Benchmark
    "BenchmarkResult",
    "GenerationResult",
    "run_benchmark",
    "calculate_summary_stats",
    # Validation
    "validate_dimensions",
    "validate_model_support",
    "get_supported_optimizations",
    "apply_optimizations",
    # Export
    "save_benchmark_results",
    "save_single_result",
    "print_summary",
    "aggregate_results",
]

