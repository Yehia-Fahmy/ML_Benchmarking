#!/usr/bin/env python3
"""
STARFlow Text-to-Image Benchmark Script
Tests Apple's STARFlow (3B parameter) model with configurable prompts.
CUDA GPU required. Requires ml-starflow repository and checkpoint.
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from utils import (
    MODEL_CONFIGS,
    DEFAULT_PROMPTS,
    DEFAULT_SEED,
    OUTPUT_DIR,
    BENCHMARKS_DIR,
    detect_cuda_device,
    cleanup_memory,
)
from utils.benchmark import (
    BenchmarkResult,
    GenerationResult,
    get_system_info,
    calculate_summary_stats,
)
from utils.export import save_single_result, print_summary
from utils.device import get_peak_gpu_memory, reset_peak_memory_stats

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME = "starflow-t2i"
MODEL_CONFIG = MODEL_CONFIGS[MODEL_NAME]

# STARFlow specific settings
STARFLOW_REPO_URL = "https://github.com/apple/ml-starflow.git"
STARFLOW_REPO_PATH = Path("ml-starflow")
MODEL_CONFIG_FILE = "configs/starflow_3B_t2i_256x256.yaml"
MODEL_CHECKPOINT = "ckpts/starflow_3B_t2i_256x256.pth"
DEFAULT_CFG_SCALE = 3.6
DEFAULT_ASPECT_RATIO = "1:1"
IMAGE_SIZE = 256  # STARFlow only supports 256x256


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def check_starflow_repository() -> Path:
    """Check if ml-starflow repository exists, clone if not."""
    if not STARFLOW_REPO_PATH.exists():
        print("=" * 70)
        print("CLONING ML-STARFLOW REPOSITORY")
        print("=" * 70)
        print(f"Repository not found at: {STARFLOW_REPO_PATH}")
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
            raise RuntimeError("Unable to clone ml-starflow repository")
    else:
        print(f"✓ Found ml-starflow repository at: {STARFLOW_REPO_PATH}")
    
    return STARFLOW_REPO_PATH.absolute()


def check_checkpoint(repo_path: Path) -> Path:
    """Check if model checkpoint exists, attempt download if not."""
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
            )
            
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
            raise RuntimeError("Checkpoint file not available")
    else:
        print(f"✓ Found checkpoint at: {checkpoint_path}")
    
    return checkpoint_path


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def run_starflow_sample(
    repo_path: Path,
    prompt: str,
    seed: int,
    cfg_scale: float,
    output_dir: Path,
    sample_idx: int,
) -> GenerationResult:
    """
    Run STARFlow sample.py for a single prompt using subprocess.
    
    Returns:
        GenerationResult with timing and output information
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
        "--model_config_path", MODEL_CONFIG_FILE,
        "--checkpoint_path", MODEL_CHECKPOINT,
        "--caption", prompt,
        "--sample_batch_size", "1",
        "--cfg", str(cfg_scale),
        "--aspect_ratio", DEFAULT_ASPECT_RATIO,
        "--seed", str(seed),
        "--save_folder", "0",
        "--finetuned_vae", "none",
        "--jacobi", "1",
        "--jacobi_th", "0.001",
        "--jacobi_block_size", "16",
        "--logdir", str(sample_output),
    ]
    
    print(f"Running STARFlow sample.py...")
    print(f"  Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"  Prompt: {prompt}")
    
    # Reset GPU memory stats
    reset_peak_memory_stats()
    
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
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            print(f"✗ Generation failed: {error_msg}")
            return GenerationResult(
                index=sample_idx,
                prompt=prompt,
                seed=seed,
                success=False,
                inference_time_seconds=inference_time,
                error=error_msg,
            )
        
        # Find generated image
        generated_images = list(sample_output.rglob("*.png")) + list(sample_output.rglob("*.jpg"))
        
        if not generated_images:
            print(f"✗ No output image found in {sample_output}")
            return GenerationResult(
                index=sample_idx,
                prompt=prompt,
                seed=seed,
                success=False,
                inference_time_seconds=inference_time,
                error="No output image generated",
            )
        
        # Use the first generated image
        source_image = generated_images[0]
        
        # Move image to final location
        final_image = output_dir / f"prompt_{sample_idx:02d}_seed_{seed}.png"
        shutil.copy(str(source_image), str(final_image))
        
        # Get GPU memory
        peak_gpu_mem = get_peak_gpu_memory()
        
        print(f"✓ Generated in {inference_time:.2f}s ({1.0/inference_time:.2f} img/s)")
        print(f"  Saved to: {final_image}")
        if peak_gpu_mem > 0:
            print(f"  GPU Memory: {peak_gpu_mem:.0f} MB ({peak_gpu_mem/1024:.2f} GB)")
        
        return GenerationResult(
            index=sample_idx,
            prompt=prompt,
            seed=seed,
            success=True,
            inference_time_seconds=inference_time,
            output_file=str(final_image),
            peak_gpu_memory_mb=peak_gpu_mem,
        )
        
    except subprocess.TimeoutExpired:
        print(f"✗ Generation timed out after 300 seconds")
        return GenerationResult(
            index=sample_idx,
            prompt=prompt,
            seed=seed,
            success=False,
            inference_time_seconds=300,
            error="Timeout after 300 seconds",
        )
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"✗ Error running sample.py: {e}")
        return GenerationResult(
            index=sample_idx,
            prompt=prompt,
            seed=seed,
            success=False,
            inference_time_seconds=inference_time,
            error=str(e),
        )


def run_benchmark(
    repo_path: Path,
    prompts: List[str],
    seed: int,
    cfg_scale: float,
    output_dir: Path,
) -> BenchmarkResult:
    """Run comprehensive benchmark on all prompts."""
    print()
    print("=" * 70)
    print("RUNNING STARFLOW BENCHMARK")
    print("=" * 70)
    print(f"Number of prompts: {len(prompts)}")
    print(f"Seed: {seed}")
    print(f"CFG Scale: {cfg_scale}")
    print(f"Resolution: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print()
    
    # Create output directory
    model_output_dir = output_dir / MODEL_NAME / "baseline"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result
    result = BenchmarkResult(
        model_name=MODEL_NAME,
        model_id=MODEL_CONFIG["model_id"],
        description=MODEL_CONFIG["description"],
        device="cuda",
        dtype="bfloat16",
        timestamp=datetime.now().isoformat(),
        config={
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
            "num_inference_steps": None,  # Uses Jacobi iterations
            "guidance_scale": cfg_scale,
            "seed": seed,
            "aspect_ratio": DEFAULT_ASPECT_RATIO,
        },
        system_info=get_system_info(),
    )
    
    # Run generation for each prompt
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Generating image...")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        reset_peak_memory_stats()
        
        gen_result = run_starflow_sample(
            repo_path, prompt, seed, cfg_scale, model_output_dir, idx
        )
        
        result.generations.append(gen_result)
        print()
    
    # Calculate summary
    result.summary = calculate_summary_stats(result.generations)
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="STARFlow Text-to-Image Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Custom prompts (default: use built-in test prompts)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    
    parser.add_argument(
        "--cfg",
        type=float,
        default=DEFAULT_CFG_SCALE,
        help=f"Classifier-free guidance scale (default: {DEFAULT_CFG_SCALE})"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for images (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--save_json",
        type=str,
        default=f"{BENCHMARKS_DIR}/{MODEL_NAME}_benchmark.json",
        help=f"Output JSON file for benchmark results (default: {BENCHMARKS_DIR}/{MODEL_NAME}_benchmark.json)"
    )
    
    args = parser.parse_args()
    
    # Use custom prompts or defaults
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    
    print("\n" + "=" * 70)
    print("STARFLOW TEXT-TO-IMAGE BENCHMARK")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Seed: {args.seed}")
    print(f"CFG Scale: {args.cfg}")
    print(f"Resolution: {IMAGE_SIZE}x{IMAGE_SIZE} (fixed)")
    print("=" * 70)
    print()
    
    # Detect device
    try:
        device, dtype = detect_cuda_device()
    except RuntimeError as e:
        print(f"\n✗ {e}")
        return 1
    
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
    
    try:
        result = run_benchmark(
            repo_path,
            prompts,
            args.seed,
            args.cfg,
            output_dir,
        )
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        return 1
    
    # Print summary
    print_summary(result)
    
    # Save results (ensure benchmarks directory exists)
    json_path = Path(args.save_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_single_result(result, args.save_json)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED!")
    print("=" * 70)
    print(f"✓ Images saved to: {output_dir / MODEL_NAME}/")
    print(f"✓ Results saved to: {args.save_json}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

