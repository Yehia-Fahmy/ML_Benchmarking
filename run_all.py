#!/usr/bin/env python3
"""
Master Benchmark Script - Run All Models
Orchestrates running multiple image generation models with shared parameters.
CUDA GPU required.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    BenchmarkResult,
)
from utils.benchmark import get_system_info, calculate_summary_stats, GenerationResult
from utils.export import (
    save_benchmark_results,
    print_summary,
    print_comparison_summary,
    aggregate_results,
)

# ============================================================================
# AVAILABLE MODELS
# ============================================================================

# Models that use diffusers pipelines directly
DIFFUSERS_MODELS = ["sd-turbo", "sd-1.5", "z-image-turbo"]

# Models that require external repos/special handling
SPECIAL_MODELS = ["starflow-t2i"]

ALL_MODELS = DIFFUSERS_MODELS + SPECIAL_MODELS


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_diffusers_model(model_name: str, device: str, dtype: torch.dtype):
    """Load a diffusers-based model pipeline."""
    config = MODEL_CONFIGS[model_name]
    
    print("=" * 70)
    print(f"LOADING {model_name.upper()} MODEL")
    print("=" * 70)
    print(f"Model: {config['model_id']}")
    print(f"Description: {config['description']}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()
    
    cleanup_memory()
    
    print("Downloading/loading model weights... (this may take a while on first run)")
    
    # Load appropriate pipeline
    if config["pipeline_class"] == "ZImagePipeline":
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            config["model_id"],
            low_cpu_mem_usage=True,
        )
        pipe = pipe.to(device, dtype=dtype)
    elif config["pipeline_class"] == "AutoPipelineForText2Image":
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            config["model_id"],
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        pipe = pipe.to(device)
    else:  # StableDiffusionPipeline
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            config["model_id"],
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        pipe = pipe.to(device)
    
    print(f"✓ Model moved to {device} with dtype {dtype}")
    
    # Apply optimizations
    apply_optimizations(pipe, model_name, verbose=True)
    
    print(f"\n✓ Model loaded successfully")
    print("=" * 70)
    print()
    
    return pipe


def run_diffusers_benchmark(
    model_name: str,
    prompts: List[str],
    height: int,
    width: int,
    seed: int,
    output_dir: Path,
    device: str,
    dtype: torch.dtype,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
) -> Optional[BenchmarkResult]:
    """Run benchmark for a diffusers-based model."""
    
    # Validate dimensions for this model
    is_valid, message, adj_height, adj_width = validate_dimensions(
        model_name, height, width
    )
    
    if not is_valid:
        print(f"⚠ Dimension adjustment for {model_name}: {message}")
        print(f"  Using: {adj_height}x{adj_width}")
    
    # Load model
    try:
        pipe = load_diffusers_model(model_name, device, dtype)
    except Exception as e:
        print(f"✗ Failed to load {model_name}: {e}")
        return None
    
    # Run benchmark
    try:
        result = run_benchmark(
            pipe=pipe,
            model_name=model_name,
            prompts=prompts,
            height=adj_height,
            width=adj_width,
            seed=seed,
            output_dir=output_dir,
            num_inference_steps=steps,
            guidance_scale=cfg,
            verbose=True,
        )
    except Exception as e:
        print(f"✗ Benchmark failed for {model_name}: {e}")
        cleanup_memory(pipe)
        return None
    
    # Cleanup
    cleanup_memory(pipe)
    
    return result


def run_starflow_benchmark(
    prompts: List[str],
    seed: int,
    cfg: float,
    output_dir: Path,
) -> Optional[BenchmarkResult]:
    """Run benchmark for STARFlow model using subprocess."""
    import shutil
    import subprocess
    
    from utils.device import get_peak_gpu_memory, reset_peak_memory_stats
    
    MODEL_NAME = "starflow-t2i"
    config = MODEL_CONFIGS[MODEL_NAME]
    
    STARFLOW_REPO_PATH = Path("ml-starflow")
    MODEL_CONFIG_FILE = "configs/starflow_3B_t2i_256x256.yaml"
    MODEL_CHECKPOINT = "ckpts/starflow_3B_t2i_256x256.pth"
    IMAGE_SIZE = 256
    
    print("=" * 70)
    print("RUNNING STARFLOW BENCHMARK")
    print("=" * 70)
    
    # Check repository
    if not STARFLOW_REPO_PATH.exists():
        print("⚠ ml-starflow repository not found")
        print("  Run `python starflow_t2i.py` first to set up STARFlow")
        return None
    
    # Check checkpoint
    checkpoint_path = STARFLOW_REPO_PATH / MODEL_CHECKPOINT
    if not checkpoint_path.exists():
        print("⚠ STARFlow checkpoint not found")
        print("  Run `python starflow_t2i.py` first to download the checkpoint")
        return None
    
    print(f"✓ Found ml-starflow repository")
    print(f"✓ Found checkpoint")
    print(f"Resolution: {IMAGE_SIZE}x{IMAGE_SIZE} (fixed)")
    print("=" * 70)
    print()
    
    # Create output directory
    model_output_dir = output_dir / MODEL_NAME / "baseline"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result
    result = BenchmarkResult(
        model_name=MODEL_NAME,
        model_id=config["model_id"],
        description=config["description"],
        device="cuda",
        dtype="bfloat16",
        timestamp=datetime.now().isoformat(),
        config={
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
            "num_inference_steps": None,
            "guidance_scale": cfg,
            "seed": seed,
        },
        system_info=get_system_info(),
    )
    
    # Run generation for each prompt
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n[{idx}/{len(prompts)}] Generating image...")
        print(f"Prompt: {prompt[:60]}..." if len(prompt) > 60 else f"Prompt: {prompt}")
        
        sample_output = model_output_dir / f"prompt_{idx:02d}"
        sample_output.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node", "1",
            "sample.py",
            "--model_config_path", MODEL_CONFIG_FILE,
            "--checkpoint_path", MODEL_CHECKPOINT,
            "--caption", prompt,
            "--sample_batch_size", "1",
            "--cfg", str(cfg),
            "--aspect_ratio", "1:1",
            "--seed", str(seed),
            "--save_folder", "0",
            "--finetuned_vae", "none",
            "--jacobi", "1",
            "--jacobi_th", "0.001",
            "--jacobi_block_size", "16",
            "--logdir", str(sample_output),
        ]
        
        torch.cuda.empty_cache()
        reset_peak_memory_stats()
        
        start_time = time.time()
        
        try:
            proc_result = subprocess.run(
                cmd,
                cwd=str(STARFLOW_REPO_PATH.absolute()),
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            inference_time = time.time() - start_time
            
            if proc_result.returncode != 0:
                error_msg = proc_result.stderr[-500:] if proc_result.stderr else "Unknown error"
                print(f"✗ Generation failed: {error_msg[:100]}")
                result.generations.append(GenerationResult(
                    index=idx,
                    prompt=prompt,
                    seed=seed,
                    success=False,
                    inference_time_seconds=inference_time,
                    error=error_msg,
                ))
                continue
            
            # Find generated image
            generated_images = list(sample_output.rglob("*.png")) + list(sample_output.rglob("*.jpg"))
            
            if not generated_images:
                print(f"✗ No output image found")
                result.generations.append(GenerationResult(
                    index=idx,
                    prompt=prompt,
                    seed=seed,
                    success=False,
                    inference_time_seconds=inference_time,
                    error="No output image generated",
                ))
                continue
            
            # Copy to final location
            final_image = model_output_dir / f"prompt_{idx:02d}_seed_{seed}.png"
            shutil.copy(str(generated_images[0]), str(final_image))
            
            peak_gpu_mem = get_peak_gpu_memory()
            
            print(f"✓ Generated in {inference_time:.2f}s ({1.0/inference_time:.2f} img/s)")
            print(f"  Saved to: {final_image}")
            if peak_gpu_mem > 0:
                print(f"  GPU Memory: {peak_gpu_mem:.0f} MB")
            
            result.generations.append(GenerationResult(
                index=idx,
                prompt=prompt,
                seed=seed,
                success=True,
                inference_time_seconds=inference_time,
                output_file=str(final_image),
                peak_gpu_memory_mb=peak_gpu_mem,
            ))
            
        except subprocess.TimeoutExpired:
            print(f"✗ Generation timed out")
            result.generations.append(GenerationResult(
                index=idx,
                prompt=prompt,
                seed=seed,
                success=False,
                inference_time_seconds=300,
                error="Timeout",
            ))
        except Exception as e:
            print(f"✗ Error: {e}")
            result.generations.append(GenerationResult(
                index=idx,
                prompt=prompt,
                seed=seed,
                success=False,
                inference_time_seconds=time.time() - start_time,
                error=str(e),
            ))
    
    # Calculate summary
    result.summary = calculate_summary_stats(result.generations)
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master Benchmark Script - Run All Image Generation Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                          # Run all models with defaults
  python run_all.py --models sd-turbo sd-1.5 # Run specific models
  python run_all.py --seed 42 --height 512   # Custom seed and resolution
  python run_all.py --prompts "a cat" "a dog" # Custom prompts
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS + ["all"],
        default=["all"],
        help=f"Models to benchmark (default: all). Available: {', '.join(ALL_MODELS)}"
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
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for images (default: {OUTPUT_DIR})"
    )
    
    parser.add_argument(
        "--save_json",
        type=str,
        default=f"{BENCHMARKS_DIR}/benchmark_results.json",
        help=f"Output JSON file for combined benchmark results (default: {BENCHMARKS_DIR}/benchmark_results.json)"
    )
    
    parser.add_argument(
        "--starflow_cfg",
        type=float,
        default=3.6,
        help="CFG scale for STARFlow model (default: 3.6)"
    )
    
    args = parser.parse_args()
    
    # Determine which models to run
    if "all" in args.models:
        models_to_run = ALL_MODELS
    else:
        models_to_run = args.models
    
    # Use custom prompts or defaults
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    
    print("\n" + "=" * 70)
    print("MULTI-MODEL IMAGE GENERATION BENCHMARK")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to benchmark: {', '.join(models_to_run)}")
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
    
    # Run benchmarks
    all_results: List[BenchmarkResult] = []
    output_dir = Path(args.output_dir)
    
    for model_name in models_to_run:
        print(f"\n{'='*70}")
        print(f"BENCHMARKING MODEL: {model_name.upper()}")
        print(f"{'='*70}\n")
        
        if model_name in DIFFUSERS_MODELS:
            result = run_diffusers_benchmark(
                model_name=model_name,
                prompts=prompts,
                height=args.height,
                width=args.width,
                seed=args.seed,
                output_dir=output_dir,
                device=device,
                dtype=dtype,
            )
        elif model_name == "starflow-t2i":
            result = run_starflow_benchmark(
                prompts=prompts,
                seed=args.seed,
                cfg=args.starflow_cfg,
                output_dir=output_dir,
            )
        else:
            print(f"⚠ Unknown model: {model_name}")
            continue
        
        if result is not None:
            all_results.append(result)
            print_summary(result)
            print(f"\n✓ Completed benchmark for {model_name}")
            print(f"✓ Images saved to: {output_dir / model_name}/")
        else:
            print(f"\n⚠ Skipped {model_name} due to errors")
        
        # Pause between models to ensure cleanup
        time.sleep(2)
    
    if not all_results:
        print("\n✗ No benchmarks completed successfully")
        return 1
    
    # Print comparison summary
    print_comparison_summary(all_results)
    
    # Save combined results (ensure benchmarks directory exists)
    json_path = Path(args.save_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_benchmark_results(all_results, args.save_json)
    
    # Save aggregated comparison
    comparison = aggregate_results(all_results)
    comparison_file = output_dir / "comparison_summary.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"✓ Comparison summary saved to: {comparison_file}")
    
    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETED!")
    print("=" * 70)
    print(f"✓ Total models benchmarked: {len(all_results)}")
    print(f"✓ Total images generated: {sum(r.summary.get('successful_images', 0) for r in all_results)}")
    print(f"✓ Images saved to: {output_dir}/")
    print(f"✓ Benchmark data: {args.save_json}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

