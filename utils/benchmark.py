"""
Benchmarking utilities for tracking inference performance.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import psutil
import torch

from .device import get_gpu_memory_usage, get_peak_gpu_memory, reset_peak_memory_stats, cleanup_memory
from .config import get_model_config


@dataclass
class GenerationResult:
    """Result from a single image generation."""
    index: int
    prompt: str
    seed: int
    success: bool
    inference_time_seconds: float
    output_file: Optional[str] = None
    error: Optional[str] = None
    gpu_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "index": self.index,
            "prompt": self.prompt,
            "seed": self.seed,
            "success": self.success,
            "inference_time_seconds": self.inference_time_seconds,
            "images_per_second": 1.0 / self.inference_time_seconds if self.success and self.inference_time_seconds > 0 else 0,
            "output_file": self.output_file,
            "error": self.error,
            "gpu_memory_mb": self.gpu_memory_mb,
            "peak_gpu_memory_mb": self.peak_gpu_memory_mb,
            "cpu_memory_mb": self.cpu_memory_mb,
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a model."""
    model_name: str
    model_id: str
    description: str
    device: str
    dtype: str
    timestamp: str
    config: Dict[str, Any]
    system_info: Dict[str, Any]
    generations: List[GenerationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "description": self.description,
            "device": self.device,
            "dtype": self.dtype,
            "timestamp": self.timestamp,
            "config": self.config,
            "system_info": self.system_info,
            "generations": [g.to_dict() for g in self.generations],
            "summary": self.summary,
        }


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark records."""
    info = {
        "cpu_count": psutil.cpu_count(),
        "total_ram_gb": psutil.virtual_memory().total / (1024**3),
        "pytorch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["cuda_version"] = torch.version.cuda
    
    return info


def calculate_summary_stats(generations: List[GenerationResult]) -> Dict[str, Any]:
    """
    Calculate summary statistics from generation results.
    
    Args:
        generations: List of GenerationResult objects
    
    Returns:
        Dictionary with summary statistics
    """
    successful = [g for g in generations if g.success]
    failed = [g for g in generations if not g.success]
    
    summary = {
        "total_images": len(generations),
        "successful_images": len(successful),
        "failed_images": len(failed),
    }
    
    if not successful:
        summary["error"] = "All generations failed"
        return summary
    
    times = [g.inference_time_seconds for g in successful]
    
    summary["total_time_seconds"] = sum(times)
    summary["mean_time_seconds"] = sum(times) / len(times)
    summary["min_time_seconds"] = min(times)
    summary["max_time_seconds"] = max(times)
    summary["mean_images_per_second"] = len(times) / sum(times) if sum(times) > 0 else 0
    
    # Calculate standard deviation
    mean = summary["mean_time_seconds"]
    variance = sum((t - mean) ** 2 for t in times) / len(times)
    summary["std_time_seconds"] = variance ** 0.5
    
    # GPU memory stats
    peak_memories = [g.peak_gpu_memory_mb for g in successful if g.peak_gpu_memory_mb > 0]
    if peak_memories:
        summary["peak_gpu_memory_mb"] = max(peak_memories)
        summary["mean_gpu_memory_mb"] = sum(peak_memories) / len(peak_memories)
    
    return summary


def run_benchmark(
    pipe,
    model_name: str,
    prompts: List[str],
    height: int,
    width: int,
    seed: int,
    output_dir: Path,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    verbose: bool = True
) -> BenchmarkResult:
    """
    Run benchmark on a model with the given prompts.
    
    Args:
        pipe: The loaded pipeline
        model_name: Name of the model
        prompts: List of prompts to generate
        height: Image height
        width: Image width
        seed: Random seed for reproducibility
        output_dir: Directory to save images
        num_inference_steps: Override for inference steps
        guidance_scale: Override for guidance scale
        verbose: Whether to print progress
    
    Returns:
        BenchmarkResult with all generation results
    """
    config = get_model_config(model_name)
    
    if config is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Use provided values or model defaults
    steps = num_inference_steps if num_inference_steps is not None else config["num_inference_steps"]
    cfg = guidance_scale if guidance_scale is not None else config["guidance_scale"]
    
    # Get dtype from pipeline
    dtype_str = "unknown"
    if hasattr(pipe, 'transformer') and hasattr(pipe.transformer, 'dtype'):
        dtype_str = str(pipe.transformer.dtype)
    elif hasattr(pipe, 'unet') and hasattr(pipe.unet, 'dtype'):
        dtype_str = str(pipe.unet.dtype)
    
    # Create output directory
    model_output_dir = output_dir / model_name / "baseline"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result
    result = BenchmarkResult(
        model_name=model_name,
        model_id=config["model_id"],
        description=config["description"],
        device="cuda",
        dtype=dtype_str,
        timestamp=datetime.now().isoformat(),
        config={
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "seed": seed,
        },
        system_info=get_system_info(),
    )
    
    if verbose:
        print("=" * 70)
        print(f"GENERATING IMAGES - {model_name.upper()}")
        print("=" * 70)
    
    # Generate images
    for idx, prompt in enumerate(prompts, 1):
        if verbose:
            print(f"\n[{idx}/{len(prompts)}] Generating image...")
            print(f"Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"Prompt: {prompt}")
        
        gen_result = _run_single_generation(
            pipe=pipe,
            prompt=prompt,
            height=height,
            width=width,
            seed=seed,
            steps=steps,
            cfg=cfg,
            output_dir=model_output_dir,
            index=idx,
            verbose=verbose,
        )
        
        result.generations.append(gen_result)
    
    # Calculate summary
    result.summary = calculate_summary_stats(result.generations)
    
    if verbose:
        print("\n" + "=" * 70)
    
    return result


def _run_single_generation(
    pipe,
    prompt: str,
    height: int,
    width: int,
    seed: int,
    steps: int,
    cfg: float,
    output_dir: Path,
    index: int,
    verbose: bool = True
) -> GenerationResult:
    """Run a single image generation with benchmarking."""
    
    # Clear cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    reset_peak_memory_stats()
    
    # Track CPU memory
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**2)
    gpu_mem_before = get_gpu_memory_usage()
    
    # Create generator with seed
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    start_time = time.time()
    
    try:
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            ).images[0]
        
        # Synchronize for accurate timing
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        # Get memory usage
        mem_after = process.memory_info().rss / (1024**2)
        peak_gpu_mem = get_peak_gpu_memory()
        
        # Save image
        filename = f"prompt_{index:02d}_seed_{seed}.png"
        filepath = output_dir / filename
        image.save(filepath)
        
        if verbose:
            print(f"✓ Generated in {inference_time:.2f}s ({1.0/inference_time:.2f} img/s)")
            print(f"  Saved to: {filepath}")
            if peak_gpu_mem > 0:
                print(f"  GPU Memory: {peak_gpu_mem:.0f} MB ({peak_gpu_mem/1024:.2f} GB)")
        
        return GenerationResult(
            index=index,
            prompt=prompt,
            seed=seed,
            success=True,
            inference_time_seconds=inference_time,
            output_file=str(filepath),
            gpu_memory_mb=get_gpu_memory_usage() - gpu_mem_before,
            peak_gpu_memory_mb=peak_gpu_mem,
            cpu_memory_mb=mem_after - mem_before,
        )
        
    except Exception as e:
        inference_time = time.time() - start_time
        error_msg = str(e)
        
        if verbose:
            print(f"✗ Generation failed: {error_msg[:100]}")
        
        return GenerationResult(
            index=index,
            prompt=prompt,
            seed=seed,
            success=False,
            inference_time_seconds=inference_time,
            error=error_msg,
        )

