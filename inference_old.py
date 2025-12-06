#!/usr/bin/env python3
"""
Multi-Model Image Generation Benchmark
Compares Z-Image-Turbo, SD-Turbo, and Stable Diffusion 1.5 on challenging prompts.
Supports NVIDIA CUDA, Apple MPS (M1/M2/M3), and CPU with performance benchmarking.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc

import psutil
import torch
from PIL import Image

# ============================================================================
# CONFIGURATION SECTION - Edit your prompts and settings here
# ============================================================================

# Test prompts: challenging but not overly specific
PROMPTS = [
    "Futuristic cityscape in heavy rain at night with neon reflections",
    "Ancient forest with bioluminescent plants and drifting fog",
    "Underwater research base with divers and service robots",
    "Abstract geometric sculpture made of glass, smoke, and colored light",
    "Snowy mountain village at dawn beneath an aurora",
]

# Model configurations
MODEL_CONFIGS = {
    "z-image-turbo": {
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "pipeline_class": "ZImagePipeline",
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
        "description": "6B parameter Turbo DiT model",
    },
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "pipeline_class": "AutoPipelineForText2Image",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "description": "Distilled SD 1.5 for speed",
    },
    "sd-1.5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "pipeline_class": "StableDiffusionPipeline",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "description": "Classic SD 1.5 baseline",
    },
}

# Generation Parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
SEED = None  # None for random generation

# Output Settings
OUTPUT_DIR = "comparison_results"
BENCHMARK_FILE = "benchmark_results.json"

# ============================================================================
# GPU DETECTION AND SETUP
# ============================================================================

def detect_device() -> Tuple[str, torch.dtype]:
    """
    Intelligently detect available GPU/accelerator and select appropriate dtype.
    
    Returns:
        Tuple of (device_name, dtype)
    """
    print("=" * 70)
    print("DEVICE DETECTION")
    print("=" * 70)
    
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ NVIDIA CUDA GPU detected: {gpu_name}")
        print(f"  - Total Memory: {gpu_memory:.2f} GB")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Selected dtype: bfloat16")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS doesn't fully support bfloat16 yet
        print(f"✓ Apple Metal Performance Shaders (MPS) detected")
        print(f"  - Device: Apple Silicon (M1/M2/M3)")
        print(f"  - Selected dtype: float32")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"⚠ No GPU detected, falling back to CPU")
        print(f"  - Selected dtype: float32")
        print(f"  - WARNING: CPU inference will be significantly slower")
    
    print("=" * 70)
    print()
    return device, dtype


def get_gpu_memory_usage(device: str) -> float:
    """Get current GPU memory usage in MB."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def cleanup_memory(device: str, pipe=None):
    """Aggressively clean up memory between model runs."""
    if pipe is not None:
        del pipe
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_name: str, device: str, dtype: torch.dtype):
    """Load the specified model pipeline with appropriate settings."""
    config = MODEL_CONFIGS[model_name]
    
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Model ID: {config['model_id']}")
    print(f"Description: {config['description']}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()
    
    # Clear GPU cache before loading model
    cleanup_memory(device)
    
    print("Downloading/loading model weights... (this may take a while on first run)")
    
    start_time = time.time()
    
    # Load the appropriate pipeline
    if config["pipeline_class"] == "ZImagePipeline":
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            config["model_id"],
            low_cpu_mem_usage=True,
        )
    elif config["pipeline_class"] == "AutoPipelineForText2Image":
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            config["model_id"],
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    else:  # StableDiffusionPipeline
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    
    # Move to device and convert dtype
    pipe = pipe.to(device, dtype=dtype)
    print(f"✓ Model moved to {device} with dtype {dtype}")
    
    # Enable memory-efficient features for CUDA
    if device == "cuda":
        # Enable attention slicing
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            print("✓ Attention slicing enabled")
        except Exception as e:
            print(f"⚠ Attention slicing not available: {e}")
        
        # Enable VAE slicing
        try:
            pipe.enable_vae_slicing()
            print("✓ VAE slicing enabled")
        except Exception as e:
            print(f"⚠ VAE slicing not available: {e}")
        
        # Enable VAE tiling
        try:
            pipe.enable_vae_tiling()
            print("✓ VAE tiling enabled")
        except Exception as e:
            print(f"⚠ VAE tiling not available: {e}")
    
    load_time = time.time() - start_time
    print(f"\n✓ Model loaded successfully in {load_time:.2f}s")
    print("=" * 70)
    print()
    
    return pipe


# ============================================================================
# INFERENCE WITH BENCHMARKING
# ============================================================================

def run_inference_with_benchmark(
    pipe,
    model_name: str,
    prompts: List[str],
    device: str,
    output_dir: Path
) -> Dict:
    """
    Run inference on all prompts with comprehensive performance tracking.
    
    Returns:
        Dictionary containing benchmark results
    """
    config = MODEL_CONFIGS[model_name]
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model_name": model_name,
        "model_id": config["model_id"],
        "description": config["description"],
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
            "num_inference_steps": config["num_inference_steps"],
            "guidance_scale": config["guidance_scale"],
            "dtype": str(pipe.transformer.dtype) if hasattr(pipe, 'transformer') else str(pipe.unet.dtype),
        },
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "pytorch_version": torch.__version__,
        },
        "generations": [],
        "summary": {},
    }
    
    # Add GPU info if available
    if device == "cuda":
        results["system_info"]["gpu_name"] = torch.cuda.get_device_name(0)
        results["system_info"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Main inference loop with benchmarking
    print("=" * 70)
    print(f"GENERATING IMAGES - {model_name.upper()}")
    print("=" * 70)
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Generating image...")
        print(f"Prompt: {prompt}")
        
        # Clear GPU cache before each generation
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Track memory before generation
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**2)
        gpu_mem_before = get_gpu_memory_usage(device)
        
        # Generate image with timing
        start_time = time.time()
        
        # Random seed for each generation (None = random)
        generator = None
        
        with torch.no_grad():  # Disable gradient tracking for inference
            image = pipe(
                prompt=prompt,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                num_inference_steps=config["num_inference_steps"],
                guidance_scale=config["guidance_scale"],
                generator=generator,
            ).images[0]
        
        inference_time = time.time() - start_time
        
        # Synchronize GPU to get accurate memory readings
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Track memory after generation
        mem_after = process.memory_info().rss / (1024**2)
        gpu_mem_after = get_gpu_memory_usage(device)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prompt_{idx:02d}_{timestamp}.png"
        filepath = model_output_dir / filename
        image.save(filepath)
        
        # Record metrics
        gen_result = {
            "index": idx,
            "prompt": prompt,
            "filename": str(filepath.relative_to(output_dir)),
            "inference_time_seconds": inference_time,
            "images_per_second": 1.0 / inference_time,
            "cpu_memory_mb": mem_after - mem_before,
            "peak_cpu_memory_mb": mem_after,
        }
        
        if device == "cuda":
            gen_result["gpu_memory_mb"] = gpu_mem_after - gpu_mem_before
            gen_result["peak_gpu_memory_mb"] = gpu_mem_after
        
        results["generations"].append(gen_result)
        
        print(f"✓ Generated in {inference_time:.2f}s ({1.0/inference_time:.2f} img/s)")
        print(f"  Saved to: {filepath}")
        if device == "cuda" and gpu_mem_after > 0:
            print(f"  GPU Memory: {gpu_mem_after:.0f} MB")
    
    print("\n" + "=" * 70)
    
    # Calculate summary statistics
    inference_times = [g["inference_time_seconds"] for g in results["generations"]]
    results["summary"] = {
        "total_images": len(prompts),
        "total_time_seconds": sum(inference_times),
        "mean_time_seconds": sum(inference_times) / len(inference_times),
        "min_time_seconds": min(inference_times),
        "max_time_seconds": max(inference_times),
        "mean_images_per_second": len(inference_times) / sum(inference_times),
    }
    
    # Calculate standard deviation
    mean_time = results["summary"]["mean_time_seconds"]
    variance = sum((t - mean_time) ** 2 for t in inference_times) / len(inference_times)
    results["summary"]["std_time_seconds"] = variance ** 0.5
    
    return results


# ============================================================================
# RESULTS EXPORT AND DISPLAY
# ============================================================================

def save_benchmark_results(all_results: List[Dict], filepath: str):
    """Save benchmark results to JSON file."""
    output = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "models": all_results,
    }
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"✓ Benchmark results saved to: {filepath}")


def print_summary(results: Dict):
    """Print formatted summary of benchmark results."""
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY - {results['model_name'].upper()}")
    print("=" * 70)
    
    summary = results["summary"]
    config = results["config"]
    
    print(f"\nModel: {results['model_id']}")
    print(f"Description: {results['description']}")
    print(f"Device: {results['device']}")
    print(f"Resolution: {config['height']}x{config['width']}")
    print(f"Inference Steps: {config['num_inference_steps']}")
    print(f"Guidance Scale: {config['guidance_scale']}")
    
    print(f"\nPerformance:")
    print(f"  Total Images: {summary['total_images']}")
    print(f"  Total Time: {summary['total_time_seconds']:.2f}s")
    print(f"  Mean Time: {summary['mean_time_seconds']:.2f}s ± {summary['std_time_seconds']:.2f}s")
    print(f"  Min Time: {summary['min_time_seconds']:.2f}s")
    print(f"  Max Time: {summary['max_time_seconds']:.2f}s")
    print(f"  Throughput: {summary['mean_images_per_second']:.3f} images/second")
    
    if results["device"] == "cuda" and len(results["generations"]) > 0 and "peak_gpu_memory_mb" in results["generations"][0]:
        max_gpu_mem = max(g.get("peak_gpu_memory_mb", 0) for g in results["generations"])
        print(f"\nGPU Memory:")
        print(f"  Peak Usage: {max_gpu_mem:.0f} MB ({max_gpu_mem/1024:.2f} GB)")
    
    print("\n" + "=" * 70)


def update_readme_with_results(all_results: List[Dict]):
    """Update README.md with multi-model comparison results."""
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print("⚠ README.md not found, skipping update")
        return
    
    # Read current README
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Create comparison section
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    comparison_section = f"""## Model Comparison Results

**Last Run:** {timestamp}

**Test Configuration:**
- Resolution: {IMAGE_HEIGHT}x{IMAGE_WIDTH}
- Number of Prompts: {len(PROMPTS)}
- Seed: Random (different for each generation)
- Hardware: {all_results[0]['system_info']['gpu_name'] if all_results[0]['device'] == 'cuda' else 'CPU'}

### Performance Comparison

| Model | Steps | CFG | Mean Time/Image | Throughput | Peak GPU Memory |
|-------|-------|-----|-----------------|------------|-----------------|
"""
    
    for result in all_results:
        summary = result["summary"]
        config = result["config"]
        peak_mem = "N/A"
        
        if result["device"] == "cuda" and len(result["generations"]) > 0 and "peak_gpu_memory_mb" in result["generations"][0]:
            max_gpu_mem = max(g.get("peak_gpu_memory_mb", 0) for g in result["generations"])
            peak_mem = f"{max_gpu_mem/1024:.2f} GB"
        
        comparison_section += f"| {result['model_name']} | {config['num_inference_steps']} | {config['guidance_scale']} | {summary['mean_time_seconds']:.2f}s ± {summary['std_time_seconds']:.2f}s | {summary['mean_images_per_second']:.3f} img/s | {peak_mem} |\n"
    
    comparison_section += "\n### Model Details\n\n"
    
    for result in all_results:
        comparison_section += f"**{result['model_name']}** ({result['model_id']})\n"
        comparison_section += f"- {result['description']}\n"
        comparison_section += f"- Inference steps: {result['config']['num_inference_steps']}\n"
        comparison_section += f"- Guidance scale: {result['config']['guidance_scale']}\n\n"
    
    comparison_section += "### Test Prompts\n\n"
    for idx, prompt in enumerate(PROMPTS, 1):
        comparison_section += f"{idx}. {prompt}\n"
    
    comparison_section += "\n### Sample Outputs\n\n"
    comparison_section += "Images are organized in the `comparison_results/` directory by model name. Each model generated images for all test prompts.\n\n"
    
    for result in all_results:
        comparison_section += f"**{result['model_name']}**: `comparison_results/{result['model_name']}/`\n"
    
    comparison_section += "\n### Analysis\n\n"
    
    # Find fastest model
    fastest = min(all_results, key=lambda x: x["summary"]["mean_time_seconds"])
    comparison_section += f"- **Fastest Model**: {fastest['model_name']} ({fastest['summary']['mean_time_seconds']:.2f}s avg per image)\n"
    
    # Memory usage
    if all_results[0]["device"] == "cuda":
        memory_sorted = []
        for result in all_results:
            if len(result["generations"]) > 0 and "peak_gpu_memory_mb" in result["generations"][0]:
                max_mem = max(g.get("peak_gpu_memory_mb", 0) for g in result["generations"])
                memory_sorted.append((result["model_name"], max_mem))
        
        if memory_sorted:
            memory_sorted.sort(key=lambda x: x[1])
            comparison_section += f"- **Most Memory Efficient**: {memory_sorted[0][0]} ({memory_sorted[0][1]/1024:.2f} GB peak)\n"
    
    comparison_section += "\n"
    
    # Find and replace the model comparison section
    start_marker = "## Model Comparison Results"
    
    # Find the start of any existing section and replace everything from there to the end
    if start_marker in content:
        start_idx = content.find(start_marker)
        content = content[:start_idx] + comparison_section.strip() + "\n"
    else:
        # Find "## Latest Benchmark Results" or "## Generated Examples" and replace from there
        markers_to_try = ["## Latest Benchmark Results", "## Generated Examples", "## Model Information"]
        replaced = False
        for marker in markers_to_try:
            if marker in content:
                start_idx = content.find(marker)
                content = content[:start_idx] + comparison_section.strip() + "\n"
                replaced = True
                break
        
        if not replaced:
            # Append to end
            content += "\n" + comparison_section.strip() + "\n"
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✓ README.md updated with model comparison results")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Multi-Model Image Generation Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Models to benchmark (default: all)"
    )
    args = parser.parse_args()
    
    # Determine which models to run
    if "all" in args.models:
        models_to_run = list(MODEL_CONFIGS.keys())
    else:
        models_to_run = args.models
    
    print("\n" + "=" * 70)
    print("MULTI-MODEL IMAGE GENERATION BENCHMARK")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to benchmark: {', '.join(models_to_run)}")
    print(f"Number of prompts: {len(PROMPTS)}")
    print("=" * 70)
    print()
    
    # Detect device
    device, dtype = detect_device()
    
    # Clear any leftover GPU memory
    cleanup_memory(device)
    
    # Run benchmark for each model
    all_results = []
    output_dir = Path(OUTPUT_DIR)
    
    for model_name in models_to_run:
        print(f"\n{'='*70}")
        print(f"BENCHMARKING MODEL: {model_name.upper()}")
        print(f"{'='*70}\n")
        
        # Load model
        pipe = load_model(model_name, device, dtype)
        
        # Run inference with benchmarking
        results = run_inference_with_benchmark(pipe, model_name, PROMPTS, device, output_dir)
        all_results.append(results)
        
        # Print summary for this model
        print_summary(results)
        
        # Clean up before loading next model
        cleanup_memory(device, pipe)
        
        print(f"\n✓ Completed benchmark for {model_name}")
        print(f"✓ Images saved to: {output_dir / model_name}/")
        
        # Add a pause between models to ensure memory is fully cleared
        if device == "cuda":
            time.sleep(2)
    
    # Save combined results
    save_benchmark_results(all_results, BENCHMARK_FILE)
    
    # Update README with comparison
    update_readme_with_results(all_results)
    
    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"✓ Total models benchmarked: {len(all_results)}")
    print(f"✓ Total images generated: {sum(r['summary']['total_images'] for r in all_results)}")
    print(f"✓ Images saved to: {output_dir}/")
    print(f"✓ Benchmark data: {BENCHMARK_FILE}")
    print(f"✓ README.md updated with comparison")
    print()


if __name__ == "__main__":
    main()
