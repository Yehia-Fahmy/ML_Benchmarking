#!/usr/bin/env python3
"""
Z-Image-Turbo Local Inference Script
Supports NVIDIA CUDA, Apple MPS (M1/M2/M3), and CPU with performance benchmarking.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import psutil
import torch
from PIL import Image

# ============================================================================
# CONFIGURATION SECTION - Edit your prompts and settings here
# ============================================================================

PROMPTS = [
    # Test 9: Underwater scene with complex lighting
    "Underwater coral reef scene with sunlight streaming from above, creating dramatic light rays through the water. Schools of tropical fish swimming between colorful corals and sea anemones. Crystal clear turquoise water with natural depth and refraction effects.",
    
    # Test 10: Dynamic weather and atmospheric effects
    "Storm clouds gathering over a wheat field at dusk. Lightning illuminating dark purple and grey clouds from within. Golden wheat swaying in strong wind, dramatic contrast between warm foreground and ominous sky. Wide cinematic landscape.",
    
    # Test 11: Abstract architectural photography
    "Modern minimalist architecture, geometric concrete structures with sharp angles and clean lines. Interplay of light and shadow on white surfaces. Single figure for scale walking through the space. High contrast black and white photography style.",
    
    # Test 12: Natural phenomena
    "Aurora borealis dancing over a frozen lake surrounded by snow-covered pine forest. Green and purple lights reflecting on ice surface. Stars visible in clear night sky. Long exposure photography capturing light movement.",
    
    # Test 13: Urban street photography at night
    "Busy city street at night after rain, wet pavement reflecting neon signs and streetlights. Blurred motion of people with umbrellas walking past illuminated shop windows. Bokeh lights in background, moody cinematic atmosphere.",
    
    # Test 14: Macro nature photography
    "Morning dewdrops on fresh green leaves backlit by golden sunrise. Each droplet catching and refracting light. Shallow depth of field with soft bokeh background. Delicate plant details visible, vibrant natural colors.",
    
    # Test 15: Industrial and mechanical subjects
    "Vintage steam locomotive at an old railway station, dramatic side lighting highlighting mechanical details. Steam rising from the engine, rust and weathered metal textures. Sense of history and craftsmanship, documentary photography style.",
    
    # Test 16: Dramatic portraiture with elements
    "Portrait of a person with windswept hair against stormy sky backdrop. Fabric or scarf billowing dramatically in wind. Intense natural lighting from the side, raw emotion captured. Environmental portrait connecting subject to nature.",
    
    # Test 17: Food photography with styling
    "Rustic breakfast scene on wooden table by window. Fresh bread, fruits, coffee in ceramic cup, natural morning light casting soft shadows. Steam rising from hot beverage, appetizing composition with natural textures and warm tones.",
    
    # Test 18: Fantasy realism
    "Ancient library with towering bookshelves reaching into darkness above. Floating books and glowing magical particles in the air. Single beam of light from high window illuminating dust motes. Mysterious and enchanting atmosphere, photorealistic rendering.",
]

# Generation Parameters
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
NUM_INFERENCE_STEPS = 9  # Results in 8 DiT forwards for Turbo model
GUIDANCE_SCALE = 0.0  # Should be 0 for Turbo models
SEED = None  # Set to None for random generation

# Model Settings
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
USE_MODEL_COMPILATION = False  # Set to True for faster inference (first run will be slower)

# Output Settings
OUTPUT_DIR = "outputs"
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


def get_gpu_utilization(device: str) -> float:
    """Get GPU utilization percentage (NVIDIA only)."""
    if device == "cuda":
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
    return 0.0


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(device: str, dtype: torch.dtype):
    """Load Z-Image-Turbo pipeline with appropriate settings."""
    from diffusers import ZImagePipeline
    
    print("=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()
    
    # Clear GPU cache before loading model to prevent memory leaks
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ GPU cache cleared before loading")
    
    print("Downloading/loading model weights... (this may take a while on first run)")
    
    start_time = time.time()
    
    # Load pipeline without dtype parameter (it's not supported and will be ignored)
    # We'll convert to the right dtype after loading
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
    )
    
    # Move to device and convert dtype manually
    pipe = pipe.to(device, dtype=dtype)
    print(f"✓ Model moved to {device} with dtype {dtype}")
    
    # Enable memory-efficient features
    if device == "cuda":
        # Enable Flash Attention for better efficiency (if supported)
        try:
            pipe.transformer.set_attention_backend("flash")
            print("✓ Flash Attention enabled")
        except Exception as e:
            print(f"⚠ Flash Attention not available: {e}")
        
        # Enable memory-efficient attention (sliced attention)
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            print("✓ Attention slicing enabled")
        except Exception as e:
            print(f"⚠ Attention slicing not available: {e}")
        
        # Enable VAE slicing to reduce memory usage during decoding
        try:
            pipe.enable_vae_slicing()
            print("✓ VAE slicing enabled")
        except Exception as e:
            print(f"⚠ VAE slicing not available: {e}")
        
        # Enable VAE tiling for large images to reduce memory
        try:
            pipe.enable_vae_tiling()
            print("✓ VAE tiling enabled")
        except Exception as e:
            print(f"⚠ VAE tiling not available: {e}")
    
    # Optional: Model compilation
    if USE_MODEL_COMPILATION:
        print("Compiling model... (first inference will be slower)")
        pipe.transformer.compile()
        print("✓ Model compilation enabled")
    
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
    prompts: List[str],
    device: str,
    output_dir: Path
) -> Dict:
    """
    Run inference on all prompts with comprehensive performance tracking.
    
    Returns:
        Dictionary containing benchmark results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": MODEL_ID,
        "device": device,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "dtype": str(pipe.transformer.dtype),
            "model_compilation": USE_MODEL_COMPILATION,
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
    print("GENERATING IMAGES")
    print("=" * 70)
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Generating image...")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        # Clear GPU cache before each generation to prevent fragmentation
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Track memory before generation
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**2)
        gpu_mem_before = get_gpu_memory_usage(device)
        
        # Generate image with timing and gradient-free inference
        start_time = time.time()
        
        # Only create and seed generator if SEED is provided
        if SEED is not None:
            generator = torch.Generator(device).manual_seed(SEED + idx)
        else:
            generator = None  # Random seed will be used
        
        with torch.no_grad():  # Disable gradient tracking for inference
            image = pipe(
                prompt=prompt,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).images[0]
        
        inference_time = time.time() - start_time
        
        # Synchronize GPU to get accurate memory readings
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Track memory after generation
        mem_after = process.memory_info().rss / (1024**2)
        gpu_mem_after = get_gpu_memory_usage(device)
        gpu_util = get_gpu_utilization(device)
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{idx:03d}_{timestamp}.png"
        filepath = output_dir / filename
        image.save(filepath)
        
        # Record metrics
        gen_result = {
            "index": idx,
            "prompt": prompt,
            "filename": filename,
            "inference_time_seconds": inference_time,
            "images_per_second": 1.0 / inference_time,
            "cpu_memory_mb": mem_after - mem_before,
            "peak_cpu_memory_mb": mem_after,
        }
        
        if device == "cuda":
            gen_result["gpu_memory_mb"] = gpu_mem_after - gpu_mem_before
            gen_result["peak_gpu_memory_mb"] = gpu_mem_after
            if gpu_util > 0:
                gen_result["gpu_utilization_percent"] = gpu_util
        
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

def save_benchmark_results(results: Dict, filepath: str):
    """Save benchmark results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Benchmark results saved to: {filepath}")


def print_summary(results: Dict):
    """Print formatted summary of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    summary = results["summary"]
    config = results["config"]
    
    print(f"\nModel: {results['model']}")
    print(f"Device: {results['device']}")
    print(f"Resolution: {config['height']}x{config['width']}")
    print(f"Inference Steps: {config['num_inference_steps']}")
    
    print(f"\nPerformance:")
    print(f"  Total Images: {summary['total_images']}")
    print(f"  Total Time: {summary['total_time_seconds']:.2f}s")
    print(f"  Mean Time: {summary['mean_time_seconds']:.2f}s ± {summary['std_time_seconds']:.2f}s")
    print(f"  Min Time: {summary['min_time_seconds']:.2f}s")
    print(f"  Max Time: {summary['max_time_seconds']:.2f}s")
    print(f"  Throughput: {summary['mean_images_per_second']:.2f} images/second")
    
    if results["device"] == "cuda" and "peak_gpu_memory_mb" in results["generations"][0]:
        max_gpu_mem = max(g.get("peak_gpu_memory_mb", 0) for g in results["generations"])
        print(f"\nGPU Memory:")
        print(f"  Peak Usage: {max_gpu_mem:.0f} MB ({max_gpu_mem/1024:.2f} GB)")
    
    print("\n" + "=" * 70)


def update_readme_with_results(results: Dict):
    """Update README.md with latest benchmark results."""
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print("⚠ README.md not found, skipping update")
        return
    
    # Read current README
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Create benchmark section
    summary = results["summary"]
    config = results["config"]
    timestamp = datetime.fromisoformat(results["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    
    benchmark_section = f"""
## Latest Benchmark Results

**Last Run:** {timestamp}

**Configuration:**
- Model: {results['model']}
- Device: {results['device']}
- Resolution: {config['height']}x{config['width']}
- Inference Steps: {config['num_inference_steps']}
- Dtype: {config['dtype']}

**Performance Metrics:**
- Total Images Generated: {summary['total_images']}
- Mean Inference Time: {summary['mean_time_seconds']:.2f}s ± {summary['std_time_seconds']:.2f}s
- Throughput: {summary['mean_images_per_second']:.2f} images/second
"""
    
    if results["device"] == "cuda" and "peak_gpu_memory_mb" in results["generations"][0]:
        max_gpu_mem = max(g.get("peak_gpu_memory_mb", 0) for g in results["generations"])
        benchmark_section += f"- Peak GPU Memory: {max_gpu_mem:.0f} MB ({max_gpu_mem/1024:.2f} GB)\n"
    
    # Find and replace benchmark section
    start_marker = "## Latest Benchmark Results"
    if start_marker in content:
        # Find the start of the section
        start_idx = content.find(start_marker)
        # Find the next section (starts with ##) or end of file
        next_section = content.find("\n##", start_idx + len(start_marker))
        if next_section == -1:
            # No next section, replace to end
            content = content[:start_idx] + benchmark_section.strip() + "\n"
        else:
            # Replace up to next section
            content = content[:start_idx] + benchmark_section.strip() + "\n\n" + content[next_section + 1:]
    else:
        # Append to end
        content += "\n" + benchmark_section.strip() + "\n"
    
    # Write updated README
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"✓ README.md updated with benchmark results")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("Z-IMAGE-TURBO LOCAL INFERENCE")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Detect device
    device, dtype = detect_device()
    
    # Clear any leftover GPU memory from previous runs
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ GPU cache cleared from any previous runs")
        print()
    
    # Load model
    pipe = load_model(device, dtype)
    
    # Run inference with benchmarking
    output_dir = Path(OUTPUT_DIR)
    results = run_inference_with_benchmark(pipe, PROMPTS, device, output_dir)
    
    # Save and display results
    save_benchmark_results(results, BENCHMARK_FILE)
    print_summary(results)
    update_readme_with_results(results)
    
    print("\n✓ All tasks completed successfully!")
    print(f"✓ Images saved to: {output_dir}/")
    print(f"✓ Benchmark data: {BENCHMARK_FILE}")
    print()


if __name__ == "__main__":
    main()

