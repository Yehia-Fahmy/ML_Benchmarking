#!/usr/bin/env python3
"""
Stable Diffusion 1.5 Benchmark Script
Tests the classic SD 1.5 model with configurable prompts.
CUDA GPU required.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

from utils import (
    MODEL_CONFIGS,
    DEFAULT_PROMPTS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DEFAULT_SEED,
    OUTPUT_DIR,
    BENCHMARKS_DIR,
    detect_cuda_device,
    cleanup_memory,
    validate_dimensions,
    apply_optimizations,
    run_benchmark,
    save_single_result,
    print_summary,
)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_NAME = "sd-1.5"
MODEL_CONFIG = MODEL_CONFIGS[MODEL_NAME]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(device: str, dtype: torch.dtype):
    """Load Stable Diffusion 1.5 pipeline."""
    from diffusers import StableDiffusionPipeline
    
    print("=" * 70)
    print("LOADING STABLE DIFFUSION 1.5 MODEL")
    print("=" * 70)
    print(f"Model: {MODEL_CONFIG['model_id']}")
    print(f"Description: {MODEL_CONFIG['description']}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()
    
    cleanup_memory()
    
    print("Downloading/loading model weights... (this may take a while on first run)")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_CONFIG["model_id"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    pipe = pipe.to(device)
    print(f"✓ Model moved to {device} with dtype {dtype}")
    
    # Apply optimizations
    apply_optimizations(pipe, MODEL_NAME, verbose=True)
    
    print(f"\n✓ Model loaded successfully")
    print("=" * 70)
    print()
    
    return pipe


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stable Diffusion 1.5 Image Generation Benchmark",
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
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Image height (default: {DEFAULT_HEIGHT})"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Image width (default: {DEFAULT_WIDTH})"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Inference steps (default: {MODEL_CONFIG['num_inference_steps']})"
    )
    
    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help=f"Guidance scale (default: {MODEL_CONFIG['guidance_scale']})"
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
    print("STABLE DIFFUSION 1.5 BENCHMARK")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Seed: {args.seed}")
    print(f"Resolution: {args.height}x{args.width}")
    print("=" * 70)
    print()
    
    # Detect device
    try:
        device, dtype = detect_cuda_device()
    except RuntimeError as e:
        print(f"\n✗ {e}")
        return 1
    
    # Validate dimensions
    is_valid, message, height, width = validate_dimensions(
        MODEL_NAME, args.height, args.width
    )
    
    if not is_valid:
        print(f"⚠ Dimension adjustment: {message}")
        print(f"  Using: {height}x{width}")
    
    # Load model
    try:
        pipe = load_model(device, dtype)
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        return 1
    
    # Run benchmark
    output_dir = Path(args.output_dir)
    
    try:
        result = run_benchmark(
            pipe=pipe,
            model_name=MODEL_NAME,
            prompts=prompts,
            height=height,
            width=width,
            seed=args.seed,
            output_dir=output_dir,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            verbose=True,
        )
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        cleanup_memory(pipe)
        return 1
    
    # Print summary
    print_summary(result)
    
    # Save results (ensure benchmarks directory exists)
    json_path = Path(args.save_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_single_result(result, args.save_json)
    
    # Cleanup
    cleanup_memory(pipe)
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED!")
    print("=" * 70)
    print(f"✓ Images saved to: {output_dir / MODEL_NAME}/")
    print(f"✓ Results saved to: {args.save_json}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

