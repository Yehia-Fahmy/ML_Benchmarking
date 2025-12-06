# Multi-Model Image Generation Benchmark

A modular benchmarking suite comparing popular text-to-image models: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo), [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), and [STARFlow](https://huggingface.co/apple/starflow).

## Features

- **Per-Model Scripts**: Run each model independently with full CLI control
- **Master Orchestrator**: Run multiple models with shared parameters in one command
- **CUDA-Optimized**: Designed specifically for NVIDIA GPUs with automatic optimizations
- **Comprehensive Benchmarking**: Tracks inference time, memory usage, and throughput
- **Robust Error Handling**: Graceful failure with dimension validation per model
- **Reproducible Results**: Consistent seeds across all models for fair comparison
- **JSON Export**: Detailed benchmark data for analysis

## System Requirements

### Required
- Python 3.8 or higher
- **NVIDIA GPU with CUDA support** (8GB+ VRAM recommended)
- 16GB RAM
- 20GB disk space for model weights

### VRAM Guidance for 512x512 Images
- **Z-Image-Turbo**: ~6-8 GB peak (6B parameter model)
- **SD-Turbo**: ~4-6 GB peak
- **Stable Diffusion 1.5**: ~4-6 GB peak
- **STARFlow**: ~6 GB peak (256x256 only)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ML_Benchmarking
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA is available:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Usage

### Quick Start - Run All Models

```bash
python run_all.py
```

This will benchmark all models with default settings (5 prompts, seed 42, 512x512).

### Run Specific Models

```bash
# Run only SD-Turbo and SD 1.5
python run_all.py --models sd-turbo sd-1.5

# Run Z-Image-Turbo only
python run_all.py --models z-image-turbo
```

### Custom Parameters

```bash
# Custom seed and resolution
python run_all.py --seed 123 --height 768 --width 768

# Custom prompts
python run_all.py --prompts "a beautiful sunset" "a cyberpunk city"
```

### Individual Model Scripts

Each model has its own standalone script with full CLI control:

```bash
# SD-Turbo
python sd_turbo.py --seed 42 --height 512 --width 512

# Stable Diffusion 1.5
python sd_1_5.py --seed 42 --steps 30 --cfg 7.5

# Z-Image-Turbo
python z_image_turbo.py --seed 42

# STARFlow (256x256 only)
python starflow_t2i.py --seed 42 --cfg 3.6
```

### CLI Options

All scripts support these common arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--prompts` | Custom prompts (space-separated) | Built-in test prompts |
| `--seed` | Random seed for reproducibility | 42 |
| `--height` | Image height | 512 |
| `--width` | Image width | 512 |
| `--output_dir` | Output directory | `comparison_results` |
| `--save_json` | JSON output file | `{model}_benchmark.json` |

Model-specific options:
- `--steps`: Override inference steps
- `--cfg`: Override guidance scale
- `--starflow_cfg`: CFG scale for STARFlow (run_all.py only)

## Available Models

| Model | Script | Steps | CFG | Resolution | Notes |
|-------|--------|-------|-----|------------|-------|
| Z-Image-Turbo | `z_image_turbo.py` | 9 | 0.0 | 256-1024 | 6B params, DiT |
| SD-Turbo | `sd_turbo.py` | 4 | 0.0 | 256-768 | Distilled SD 1.5 |
| SD 1.5 | `sd_1_5.py` | 25 | 7.5 | 256-768 | Classic baseline |
| STARFlow | `starflow_t2i.py` | N/A | 3.6 | 256 only | Apple's flow model |

## Project Structure

```
ML_Benchmarking/
├── run_all.py           # Master orchestrator script
├── sd_turbo.py          # SD-Turbo standalone script
├── sd_1_5.py            # SD 1.5 standalone script
├── z_image_turbo.py     # Z-Image-Turbo standalone script
├── starflow_t2i.py      # STARFlow standalone script
├── utils/               # Shared utilities
│   ├── __init__.py
│   ├── config.py        # Model configurations
│   ├── device.py        # CUDA detection
│   ├── benchmark.py     # Benchmarking utilities
│   ├── validation.py    # Dimension/optimization validation
│   └── export.py        # JSON/summary export
├── comparison_results/  # Generated images
├── requirements.txt
└── README.md
```

## Output Structure

```
comparison_results/
├── sd-turbo/
│   └── baseline/
│       ├── prompt_01_seed_42.png
│       ├── prompt_02_seed_42.png
│       └── ...
├── sd-1.5/
│   └── baseline/
│       └── ...
├── z-image-turbo/
│   └── baseline/
│       └── ...
└── starflow-t2i/
    └── baseline/
        └── ...
```

## Benchmark Results

Results are saved as JSON files with comprehensive metrics:

```json
{
  "model_name": "sd-turbo",
  "config": {
    "height": 512,
    "width": 512,
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "seed": 42
  },
  "summary": {
    "total_images": 5,
    "successful_images": 5,
    "mean_time_seconds": 2.5,
    "peak_gpu_memory_mb": 4500
  },
  "generations": [...]
}
```

## Memory Optimizations

The suite automatically applies these optimizations for CUDA GPUs:
- **Attention Slicing**: Reduces memory during attention computation
- **VAE Slicing**: Splits VAE operations into smaller chunks
- **VAE Tiling**: Processes images in tiles

These are applied per-model based on what each model supports.

## Troubleshooting

### "CUDA GPU required but not available"
- Ensure you have an NVIDIA GPU with CUDA support
- Install NVIDIA drivers: `nvidia-smi` should work
- Install PyTorch with CUDA: Visit [pytorch.org](https://pytorch.org)

### Out of Memory Errors
- Reduce resolution: `--height 256 --width 256`
- Run models individually instead of all at once
- Close other GPU-intensive applications

### STARFlow Setup
STARFlow requires additional setup:
1. Run `python starflow_t2i.py` once to clone the repository
2. The checkpoint (~6GB) will be downloaded automatically
3. Ensure `ml-starflow/` directory exists with checkpoint

### Model Download Issues
- First run downloads models (~15-20GB total)
- Ensure stable internet and sufficient disk space
- Models cached in `~/.cache/huggingface/hub/`

## Default Test Prompts

1. Futuristic cityscape in heavy rain at night with neon reflections
2. Ancient forest with bioluminescent plants and drifting fog
3. Underwater research base with divers and service robots
4. Abstract geometric sculpture made of glass, smoke, and colored light
5. Snowy mountain village at dawn beneath an aurora

## License

See individual model licenses:
- [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)
- [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [STARFlow](https://huggingface.co/apple/starflow)
