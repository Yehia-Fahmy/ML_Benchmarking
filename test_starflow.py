#!/usr/bin/env python3
"""
STARFlow Text-to-Image Benchmark Script
Tests Apple's STARFlow (3B parameter) model with challenging prompts.
Wraps the official ml-starflow sample.py script for easy benchmarking.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import shutil

import psutil
import torch

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Test prompts: 3 challenging prompts + 1 random generation
DEFAULT_PROMPTS = [
    "A hyperrealistic photo of a glass sculpture reflecting rainbow light in a dark room",
    "An ancient temple covered in glowing bioluminescent moss at twilight",
    "A futuristic cyberpunk street with neon signs reflecting on wet pavement in heavy rain",
    "generate a random image",
]

# STARFlow Model Configuration
STARFLOW_REPO_URL = "https://github.com/apple/ml-starflow.git"
MODEL_CONFIG = "configs/starflow_3B_t2i_256x256.yaml"
MODEL_CHECKPOINT = "ckpts/starflow_3B_t2i_256x256.pth"

# Default generation parameters
DEFAULT_CFG_SCALE = 3.6
DEFAULT_SEED = 12
DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_IMAGE_SIZE = 256

# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def check_starflow_repository() -> Path:
    """Check if ml-starflow repository exists, clone if not."""
    repo_path = Path("ml-starflow")
    
    if not repo_path.exists():
        print("=" * 70)
        print("CLONING ML-STARFLOW REPOSITORY")
        print("=" * 70)
        print(f"Repository not found at: {repo_path}")
        print(f"Cloning from: {STARFLOW_REPO_URL}")
        print()
        
        try:
            result = subprocess.run(
                ["git", "clone", STARFLOW_REPO_URL],
                capture_output=True,
                text=True,
                check=True
            )
            print("✓ Successfully cloned ml-starflow repository")
            print("=" * 70)
            print()
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to clone repository: {e.stderr}")
            raise Exception("Unable to clone ml-starflow repository")
    else:
        print(f"✓ Found ml-starflow repository at: {repo_path}")
    
    return repo_path.absolute()


def check_checkpoint(repo_path: Path) -> Path:
    """Check if model checkpoint exists, provide download instructions if not."""
    checkpoint_path = repo_path / MODEL_CHECKPOINT
    
    if not checkpoint_path.exists():
        print()
        print("=" * 70)
        print("MODEL CHECKPOINT NOT FOUND")
        print("=" * 70)
        print(f"Expected location: {checkpoint_path}")
        print()
        print("To use STARFlow, you need to download the checkpoint file:")
        print("1. Visit: https://huggingface.co/apple/starflow/tree/main")
        print("2. Download 'starflow_3B_t2i_256x256.pth' (~6GB)")
        print(f"3. Place it in: {repo_path / 'ckpts'}/")
        print()
        print("Attempting automatic download using huggingface_hub...")
        
        try:
            from huggingface_hub import hf_hub_download
            
            checkpoint_dir = repo_path / "ckpts"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            print("Downloading checkpoint (this may take several minutes, ~6GB)...")
            downloaded_file = hf_hub_download(
                repo_id="apple/starflow",
                filename="starflow_3B_t2i_256x256.pth",
                local_dir=str(checkpoint_dir),
                local_dir_use_symlinks=False,
            )
            
            # Move to expected location if needed
            downloaded_path = Path(downloaded_file)
            if downloaded_path != checkpoint_path:
                shutil.move(str(downloaded_path), str(checkpoint_path))
            
            print(f"✓ Successfully downloaded checkpoint to {checkpoint_path}")
            print("=" * 70)
            print()
            
        except Exception as e:
            print(f"✗ Automatic download failed: {e}")
            print()
            print("Please download the checkpoint manually:")
            print("  https://huggingface.co/apple/starflow/tree/main")
            raise Exception("Checkpoint file not available")
    else:
        print(f"✓ Found checkpoint at: {checkpoint_path}")
    
    return checkpoint_path


def detect_device():
    """Detect available GPU."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ NVIDIA CUDA GPU detected: {gpu_name}")
        print(f"  - Total Memory: {gpu_memory:.2f} GB")
        return "cuda"
    elif torch.backends.mps.is_available():
        print(f"✓ Apple MPS detected")
        print(f"  - Note: STARFlow works best with CUDA GPUs")
        return "mps"
    else:
        print(f"⚠ No GPU detected")
        print(f"  - STARFlow requires a GPU for practical use")
        return "cpu"


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def run_starflow_sample(
    repo_path: Path,
    prompt: str,
    seed: int,
    cfg_scale: float,
    output_dir: Path,
    sample_idx: int,
) -> Dict:
    """
    Run STARFlow sample.py for a single prompt.
    
    Returns:
        Dictionary with timing and output information
    """
    # Create output directory for this sample
    sample_output = output_dir / f"prompt_{sample_idx:02d}"
    sample_output.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node", "1",
        "sample.py",
        "--model_config_path", MODEL_CONFIG,
        "--checkpoint_path", MODEL_CHECKPOINT,
        "--caption", prompt,
        "--sample_batch_size", "1",
        "--cfg", str(cfg_scale),
        "--aspect_ratio", DEFAULT_ASPECT_RATIO,
        "--seed", str(seed),
        "--save_folder", "0",  # Don't create dated folder
        "--finetuned_vae", "none",
        "--jacobi", "1",
        "--jacobi_th", "0.001",
        "--jacobi_block_size", "16",
        "--logdir", str(sample_output),
    ]
    
    print(f"Running STARFlow sample.py...")
    print(f"  Prompt: {prompt[:60]}...")
    
    # Measure time and run
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        inference_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"✗ Generation failed: {result.stderr[-500:]}")
            return {
                "success": False,
                "error": result.stderr[-500:],
                "inference_time": inference_time,
            }
        
        # Find generated image
        generated_images = list(sample_output.rglob("*.png")) + list(sample_output.rglob("*.jpg"))
        
        if not generated_images:
            print(f"✗ No output image found in {sample_output}")
            return {
                "success": False,
                "error": "No output image generated",
                "inference_time": inference_time,
            }
        
        # Use the first generated image
        source_image = generated_images[0]
        
        # Move image to final location
        final_image = output_dir / f"prompt_{sample_idx:02d}_seed_{seed}.png"
        shutil.copy(str(source_image), str(final_image))
        
        print(f"✓ Generated in {inference_time:.2f}s")
        print(f"  Saved to: {final_image}")
        
        # Get GPU memory if available
        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
        
        return {
            "success": True,
            "inference_time": inference_time,
            "output_file": str(final_image),
            "gpu_memory_mb": gpu_mem,
        }
        
    except subprocess.TimeoutExpired:
        print(f"✗ Generation timed out after 300 seconds")
        return {
            "success": False,
            "error": "Timeout",
            "inference_time": 300,
        }
    except Exception as e:
        print(f"✗ Error running sample.py: {e}")
        return {
            "success": False,
            "error": str(e),
            "inference_time": time.time() - start_time,
        }


def run_benchmark(
    repo_path: Path,
    prompts: List[str],
    seed: int,
    cfg_scale: float,
    output_dir: Path,
) -> Dict:
    """Run comprehensive benchmark on all prompts."""
    print()
    print("=" * 70)
    print("RUNNING STARFLOW BENCHMARK")
    print("=" * 70)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Seed: {seed}")
    print(f"CFG Scale: {cfg_scale}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results
    results = {
        "model_name": "starflow",
        "model_id": "apple/starflow",
        "description": "STARFlow 3B - Transformer Autoregressive Flow for T2I (256x256)",
        "device": detect_device(),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "image_size": DEFAULT_IMAGE_SIZE,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "aspect_ratio": DEFAULT_ASPECT_RATIO,
        },
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "pytorch_version": torch.__version__,
        },
        "generations": [],
        "summary": {},
    }
    
    if torch.cuda.is_available():
        results["system_info"]["gpu_name"] = torch.cuda.get_device_name(0)
        results["system_info"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # Run generation for each prompt
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Generating image...")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Run generation
        gen_result = run_starflow_sample(
            repo_path, prompt, seed, cfg_scale, output_dir, idx
        )
        
        # Record result
        result_entry = {
            "index": idx,
            "prompt": prompt,
            "seed": seed,
            **gen_result,
        }
        
        results["generations"].append(result_entry)
        print()
    
    # Calculate summary statistics
    successful = [g for g in results["generations"] if g.get("success", False)]
    
    if successful:
        times = [g["inference_time"] for g in successful]
        
        results["summary"] = {
            "total_images": len(prompts),
            "successful_images": len(successful),
            "failed_images": len(prompts) - len(successful),
            "total_time_seconds": sum(times),
            "mean_time_seconds": sum(times) / len(times),
            "min_time_seconds": min(times),
            "max_time_seconds": max(times),
            "mean_images_per_second": len(times) / sum(times) if sum(times) > 0 else 0,
        }
        
        # Calculate std
        mean = results["summary"]["mean_time_seconds"]
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        results["summary"]["std_time_seconds"] = variance ** 0.5
        
        # GPU memory
        if torch.cuda.is_available():
            max_mem = max(g.get("gpu_memory_mb", 0) for g in successful)
            if max_mem > 0:
                results["summary"]["peak_gpu_memory_mb"] = max_mem
    else:
        results["summary"] = {
            "total_images": len(prompts),
            "successful_images": 0,
            "failed_images": len(prompts),
            "error": "All generations failed",
        }
    
    return results


def print_benchmark_summary(results: Dict):
    """Print formatted benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY - STARFLOW")
    print("=" * 70)
    
    print(f"\nModel: {results['model_id']}")
    print(f"Description: {results['description']}")
    print(f"Device: {results['device']}")
    print(f"Image Size: {results['config']['image_size']}x{results['config']['image_size']}")
    print(f"CFG Scale: {results['config']['cfg_scale']}")
    print(f"Seed: {results['config']['seed']}")
    
    summary = results["summary"]
    
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
    
    print("\n" + "=" * 70)


def save_results(results: Dict, output_file: Path):
    """Save benchmark results to JSON."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Benchmark results saved to: {output_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="STARFlow Text-to-Image Benchmark Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Custom prompts (default: use 3 challenging + 1 random)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_results/starflow",
        help="Output directory for images (default: comparison_results/starflow)"
    )
    
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=DEFAULT_CFG_SCALE,
        help=f"Classifier-free guidance scale (default: {DEFAULT_CFG_SCALE})"
    )
    
    parser.add_argument(
        "--save_json",
        type=str,
        default="starflow_benchmark.json",
        help="Output JSON file for benchmark results (default: starflow_benchmark.json)"
    )
    
    args = parser.parse_args()
    
    # Use custom prompts or default
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    
    print("\n" + "=" * 70)
    print("STARFLOW TEXT-TO-IMAGE BENCHMARK")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Seed: {args.seed}")
    print(f"CFG Scale: {args.cfg_scale}")
    print("=" * 70)
    print()
    
    # Check and setup STARFlow repository
    try:
        repo_path = check_starflow_repository()
        checkpoint_path = check_checkpoint(repo_path)
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        return 1
    
    print()
    
    # Run benchmark
    output_dir = Path(args.output_dir)
    results = run_benchmark(
        repo_path,
        prompts,
        args.seed,
        args.cfg_scale,
        output_dir,
    )
    
    # Print summary
    print_benchmark_summary(results)
    
    # Save results
    save_results(results, Path(args.save_json))
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED!")
    print("=" * 70)
    print(f"✓ Images saved to: {output_dir}/")
    print(f"✓ Results saved to: {args.save_json}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
