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
    # Original prompts
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.",
    "A serene mountain landscape at sunset, with golden light reflecting off a crystal clear lake, surrounded by pine trees and snow-capped peaks in the distance.",
    "A futuristic cityscape at night with neon lights, flying vehicles, and holographic advertisements in a cyberpunk style.",
    
    # Challenging new prompts
    # Test 1: Bilingual text rendering with complex scene
    "A vintage coffee shop storefront with a neon sign reading 'OPEN 24/7' in red letters above the door, and a wooden chalkboard menu displaying '咖啡 $3.50' in white chalk. Rain-slicked cobblestone street, warm golden light spilling through foggy windows, early morning blue hour atmosphere.",
    
    # Test 2: Complex lighting and reflections
    "A crystal wine glass filled with red wine, sitting on a marble countertop. Dramatic side lighting creates caustics and reflections on the surface. A single rose petal floats in the wine. Shallow depth of field, photorealistic macro photography style, with bokeh lights in the dark background.",
    
    # Test 3: Action and motion with particles
    "A professional dancer mid-leap in an abandoned warehouse, arms extended gracefully. Flour powder explodes around her body, frozen in mid-air, creating dramatic white clouds. Harsh sunlight streams through broken windows, creating god rays through the dust. High-speed photography, every particle visible and sharp.",
    
    # Test 4: Intricate architectural detail
    "Interior of a grand baroque cathedral, ornate golden details on every surface. Sunlight streams through massive stained glass windows, casting colored light patterns on white marble floors. Elaborate ceiling frescoes depicting celestial scenes. Ultra-wide angle perspective looking up towards the dome, emphasizing scale and grandeur.",
    
    # Test 5: Challenging materials and textures
    "Extreme close-up of a water droplet suspended on a spider's web at dawn. The droplet acts as a lens, containing a perfect miniature reflection of a sunrise landscape. Morning dew covers the entire web. Macro photography with perfect focus on water surface tension, iridescent light refractions.",
    
    # Test 6: Complex character interaction with emotion
    "An elderly master calligrapher teaching a young student in a traditional Japanese study room. The master's weathered hands guide the student's brush, mid-stroke on rice paper. Ink bottles, scrolls, and brushes arranged on the low table. Soft natural light from shoji screens, expressions of concentration and wisdom. Photorealistic, intimate moment captured.",
    
    # Test 7: Surreal but photorealistic
    "A giant vintage pocket watch partially buried in desert sand dunes, its face showing roman numerals. The watch is overgrown with lush green vines and blooming flowers emerging from its mechanisms. Golden hour lighting, long shadows, a single bird perched on the watch crown. Hyper-realistic, surreal juxtaposition.",
    
    # Test 8: Environmental storytelling
    "An abandoned astronaut helmet on the surface of Mars, half-buried in red dust. The helmet's visor reflects the pink Martian sky and distant Earth as a blue dot. Small rocks and footprints leading away into the distance. Cinematic composition, sense of isolation and mystery, photorealistic space photography aesthetic.",
]

# Generation Parameters
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
NUM_INFERENCE_STEPS = 9  # Results in 8 DiT forwards for Turbo model
GUIDANCE_SCALE = 0.0  # Should be 0 for Turbo models
SEED = 42  # Set to None for random generation

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
    print("Downloading/loading model weights... (this may take a while on first run)")
    
    start_time = time.time()
    
    # Load pipeline
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe.to(device)
    
    # Optional: Enable Flash Attention for better efficiency (if supported)
    if device == "cuda":
        try:
            pipe.transformer.set_attention_backend("flash")
            print("✓ Flash Attention enabled")
        except Exception as e:
            print(f"⚠ Flash Attention not available: {e}")
    
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
        "warmup": {},
        "generations": [],
        "summary": {},
    }
    
    # Add GPU info if available
    if device == "cuda":
        results["system_info"]["gpu_name"] = torch.cuda.get_device_name(0)
        results["system_info"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print("=" * 70)
    print("WARMUP RUN")
    print("=" * 70)
    print("Running warmup to account for compilation overhead...")
    
    # Warmup run
    warmup_start = time.time()
    generator = torch.Generator(device).manual_seed(SEED) if SEED is not None else None
    
    _ = pipe(
        prompt=prompts[0],
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]
    
    warmup_time = time.time() - warmup_start
    results["warmup"]["time_seconds"] = warmup_time
    print(f"✓ Warmup completed in {warmup_time:.2f}s")
    print("=" * 70)
    print()
    
    # Clear cache after warmup
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Main inference loop with benchmarking
    print("=" * 70)
    print("GENERATING IMAGES")
    print("=" * 70)
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Generating image...")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        # Track memory before generation
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024**2)
        gpu_mem_before = get_gpu_memory_usage(device)
        
        # Generate image with timing
        start_time = time.time()
        generator = torch.Generator(device).manual_seed(SEED + idx if SEED is not None else None)
        
        image = pipe(
            prompt=prompt,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]
        
        inference_time = time.time() - start_time
        
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
    print(f"  Warmup Time: {results['warmup']['time_seconds']:.2f}s")
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
- Warmup Time: {results['warmup']['time_seconds']:.2f}s
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

