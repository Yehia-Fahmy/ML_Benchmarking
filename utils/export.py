"""
Export utilities for saving benchmark results and updating documentation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from .benchmark import BenchmarkResult
from .config import DEFAULT_PROMPTS


def save_benchmark_results(
    results: List[BenchmarkResult],
    filepath: str,
    include_timestamp: bool = True
) -> str:
    """
    Save benchmark results to JSON file.
    
    Args:
        results: List of BenchmarkResult objects
        filepath: Path to save JSON file
        include_timestamp: Whether to include timestamp in output
    
    Returns:
        Path to saved file
    """
    output = {
        "benchmark_timestamp": datetime.now().isoformat() if include_timestamp else None,
        "models": [r.to_dict() for r in results],
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Benchmark results saved to: {filepath}")
    return filepath


def save_single_result(result: BenchmarkResult, filepath: str) -> str:
    """
    Save a single benchmark result to JSON file.
    
    Args:
        result: BenchmarkResult object
        filepath: Path to save JSON file
    
    Returns:
        Path to saved file
    """
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"✓ Benchmark result saved to: {filepath}")
    return filepath


def print_summary(result: BenchmarkResult):
    """Print formatted summary of benchmark results."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY - {result.model_name.upper()}")
    print("=" * 70)
    
    print(f"\nModel: {result.model_id}")
    print(f"Description: {result.description}")
    print(f"Device: {result.device}")
    print(f"Dtype: {result.dtype}")
    print(f"Resolution: {result.config['height']}x{result.config['width']}")
    print(f"Inference Steps: {result.config['num_inference_steps']}")
    print(f"Guidance Scale: {result.config['guidance_scale']}")
    print(f"Seed: {result.config['seed']}")
    
    summary = result.summary
    
    if summary.get("successful_images", 0) > 0:
        print(f"\nPerformance:")
        print(f"  Total Images: {summary['total_images']}")
        print(f"  Successful: {summary['successful_images']}")
        print(f"  Failed: {summary['failed_images']}")
        print(f"  Total Time: {summary['total_time_seconds']:.2f}s")
        print(f"  Mean Time: {summary['mean_time_seconds']:.2f}s ± {summary['std_time_seconds']:.2f}s")
        print(f"  Min Time: {summary['min_time_seconds']:.2f}s")
        print(f"  Max Time: {summary['max_time_seconds']:.2f}s")
        print(f"  Throughput: {summary['mean_images_per_second']:.3f} images/second")
        
        if "peak_gpu_memory_mb" in summary:
            print(f"\nGPU Memory:")
            print(f"  Peak Usage: {summary['peak_gpu_memory_mb']:.0f} MB ({summary['peak_gpu_memory_mb']/1024:.2f} GB)")
    else:
        print(f"\n⚠ All generations failed")
        if "error" in summary:
            print(f"  Error: {summary['error']}")
    
    print("\n" + "=" * 70)


def aggregate_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """
    Aggregate multiple benchmark results into a comparison summary.
    
    Args:
        results: List of BenchmarkResult objects
    
    Returns:
        Dictionary with aggregated comparison data
    """
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "total_models": len(results),
        "models": [],
    }
    
    for result in results:
        model_summary = {
            "name": result.model_name,
            "model_id": result.model_id,
            "description": result.description,
            "config": result.config,
            "summary": result.summary,
        }
        comparison["models"].append(model_summary)
    
    # Find fastest model
    successful_results = [r for r in results if r.summary.get("successful_images", 0) > 0]
    
    if successful_results:
        fastest = min(successful_results, key=lambda r: r.summary.get("mean_time_seconds", float('inf')))
        comparison["fastest_model"] = {
            "name": fastest.model_name,
            "mean_time_seconds": fastest.summary.get("mean_time_seconds", 0),
        }
        
        # Find most memory efficient
        models_with_memory = [r for r in successful_results if "peak_gpu_memory_mb" in r.summary]
        if models_with_memory:
            most_efficient = min(models_with_memory, key=lambda r: r.summary.get("peak_gpu_memory_mb", float('inf')))
            comparison["most_memory_efficient"] = {
                "name": most_efficient.model_name,
                "peak_gpu_memory_mb": most_efficient.summary.get("peak_gpu_memory_mb", 0),
            }
    
    return comparison


def print_comparison_summary(results: List[BenchmarkResult]):
    """Print a comparison table of multiple models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    # Header
    print(f"\n{'Model':<20} {'Steps':<8} {'CFG':<6} {'Mean Time':<15} {'Throughput':<12} {'Peak GPU':<12}")
    print("-" * 73)
    
    for result in results:
        summary = result.summary
        config = result.config
        
        if summary.get("successful_images", 0) > 0:
            mean_time = f"{summary['mean_time_seconds']:.2f}s"
            throughput = f"{summary['mean_images_per_second']:.3f} img/s"
            peak_mem = f"{summary.get('peak_gpu_memory_mb', 0)/1024:.2f} GB" if "peak_gpu_memory_mb" in summary else "N/A"
        else:
            mean_time = "FAILED"
            throughput = "-"
            peak_mem = "-"
        
        steps = str(config.get('num_inference_steps', '-'))
        cfg = str(config.get('guidance_scale', '-'))
        
        print(f"{result.model_name:<20} {steps:<8} {cfg:<6} {mean_time:<15} {throughput:<12} {peak_mem:<12}")
    
    print("-" * 73)
    
    # Find fastest and most efficient
    successful = [r for r in results if r.summary.get("successful_images", 0) > 0]
    
    if successful:
        fastest = min(successful, key=lambda r: r.summary.get("mean_time_seconds", float('inf')))
        print(f"\n✓ Fastest: {fastest.model_name} ({fastest.summary['mean_time_seconds']:.2f}s avg)")
        
        with_memory = [r for r in successful if "peak_gpu_memory_mb" in r.summary]
        if with_memory:
            efficient = min(with_memory, key=lambda r: r.summary.get("peak_gpu_memory_mb", float('inf')))
            print(f"✓ Most Memory Efficient: {efficient.model_name} ({efficient.summary['peak_gpu_memory_mb']/1024:.2f} GB peak)")
    
    print("\n" + "=" * 70)

