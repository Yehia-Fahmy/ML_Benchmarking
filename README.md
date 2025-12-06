# Multi-Model Image Generation Benchmark

A comprehensive benchmarking suite comparing popular text-to-image models: [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo), [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), and [STARFlow](https://huggingface.co/apple/starflow).

## Features

- **Multi-Model Comparison**: Benchmark Z-Image-Turbo, SD-Turbo, SD 1.5, and STARFlow side-by-side
- **Intelligent GPU Detection**: Automatically detects and uses NVIDIA CUDA, Apple MPS (M1/M2/M3), or falls back to CPU
- **Optimized for 8GB VRAM**: Memory-efficient settings perfect for RTX 3070 Ti and similar GPUs
- **Comprehensive Benchmarking**: Tracks inference time, memory usage, and throughput per model
- **Challenging Test Prompts**: 5 carefully selected prompts testing diverse generation capabilities
- **Aggressive Memory Management**: Proper cleanup between models for stable multi-model runs
- **Automatic Result Export**: Saves benchmark data to JSON and updates this README with comparison tables
- **STARFlow Support**: Dedicated script for Apple's transformer autoregressive flow model

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 16GB RAM
- 20GB disk space for model weights (all three models combined)

### GPU Requirements
- **Recommended**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070 Ti, RTX 3080, RTX 4070)
  - All three models tested at 512x512 resolution
  - Memory-efficient optimizations enabled (attention slicing, VAE tiling)
- **Apple Silicon**: M1/M2/M3 with 16GB+ unified memory
- **Note**: CPU inference is supported but will be significantly slower

### VRAM Guidance for 512x512 Images
- **Z-Image-Turbo**: ~6-8 GB peak
- **SD-Turbo**: ~4-6 GB peak
- **Stable Diffusion 1.5**: ~4-6 GB peak

The benchmark script includes aggressive memory cleanup between models, making it safe to run all three sequentially on 8GB GPUs.

## Installation

1. **Clone or download this repository:**
   ```bash
   cd /home/yehia/image_gen_local
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - `diffusers` (from GitHub source for Z-Image support)
   - PyTorch and related packages
   - Performance monitoring tools (psutil)
   - Image processing libraries

4. **Optional: Install GPUtil for NVIDIA GPU monitoring:**
   ```bash
   pip install gputil
   ```

## Usage

### Quick Start - Benchmark All Models

Run the complete benchmark comparing all three models:

```bash
python inference.py
```

This will:
1. Detect your GPU/device automatically
2. Load each model sequentially (Z-Image-Turbo, SD-Turbo, SD 1.5)
3. Generate 5 test images per model (15 total images)
4. Save images to `comparison results/[model-name]/`
5. Record performance metrics to `benchmark_results.json`
6. Update this README with a comparison table

### Benchmark Individual Models

Run specific models only:

```bash
# Benchmark only Z-Image-Turbo
python inference.py --models z-image-turbo

# Benchmark SD-Turbo and SD 1.5
python inference.py --models sd-turbo sd-1.5

# All models (same as no arguments)
python inference.py --models all
```

### Available Models

- `z-image-turbo`: Tongyi-MAI Z-Image-Turbo (6B params, 9 steps, CFG=0)
- `sd-turbo`: Stability AI SD-Turbo (distilled, 4 steps, CFG=0)
- `sd-1.5`: Stable Diffusion 1.5 (classic baseline, 25 steps, CFG=7.5)

### STARFlow Benchmark (Separate Script)

To test Apple's STARFlow model:

```bash
python test_starflow.py --seed 12
```

See [STARFLOW_README.md](STARFLOW_README.md) for detailed documentation on the STARFlow benchmark script.

Key features:
- Automatic model download and setup
- Tests with 3 challenging prompts + 1 random generation
- Configurable seed (default: 12)
- Comprehensive performance metrics
- GPU memory tracking

### Test Prompts

The benchmark uses 5 challenging prompts designed to test different capabilities:

1. Futuristic cityscape in heavy rain at night with neon reflections
2. Ancient forest with bioluminescent plants and drifting fog
3. Underwater research base with divers and service robots
4. Abstract geometric sculpture made of glass, smoke, and colored light
5. Snowy mountain village at dawn beneath an aurora

All images are generated at **512x512** resolution with **random seeds** for diverse outputs.

### Configuration Options

To modify prompts or settings, edit the configuration section in `inference.py`:

```python
# Test prompts
PROMPTS = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    # Add more...
]

# Generation settings
IMAGE_HEIGHT = 512  # Recommended for 8GB VRAM
IMAGE_WIDTH = 512
SEED = None  # None for random, or set integer for reproducible results
```

Model-specific settings (steps, CFG) are automatically configured per model but can be customized in `MODEL_CONFIGS` dictionary.

## Memory Optimization Features

The benchmark script includes several optimizations for stable operation on 8GB GPUs:

### Automatic Optimizations (CUDA)
- **Attention Slicing**: Reduces memory usage during attention computation
- **VAE Slicing**: Splits VAE operations into smaller chunks
- **VAE Tiling**: Processes images in tiles to reduce peak memory
- **Aggressive Cleanup**: Clears CUDA cache and runs garbage collection between models

### Memory Tips
- **512x512 is optimal** for 8GB VRAM when running all three models
- **Close other GPU applications** before running the benchmark
- **Run models individually** if experiencing OOM: `--models z-image-turbo`
- The script automatically synchronizes CUDA operations for accurate memory tracking

## Understanding the Output

### Directory Structure
After running the benchmark, outputs are organized by model:

```
comparison results/
├── z-image-turbo/
│   ├── prompt_01_20251205_201729.png
│   ├── prompt_02_20251205_202022.png
│   └── ...
├── sd-turbo/
│   ├── prompt_01_20251205_203352.png
│   └── ...
└── sd-1.5/
    ├── prompt_01_20251205_204139.png
    └── ...
```

### Benchmark Results
`benchmark_results.json` contains comprehensive metrics for each model:
- Model configuration (steps, CFG, dtype)
- Per-image timing and memory usage
- Aggregate statistics (mean, std, min, max)
- System information

### README Updates
This README is automatically updated with:
- Performance comparison table across all models
- Model details and configurations
- Analysis highlighting fastest model and most memory-efficient
- Sample output locations

## Troubleshooting

### Out of Memory Errors
- Images are already set to 512x512 (optimal for 8GB VRAM)
- Try running models individually: `python inference.py --models sd-turbo`
- Close other GPU-intensive applications
- Restart your system to clear any GPU memory leaks

### Slow Performance
- Ensure GPU is detected (check console output at start)
- First run downloads models (~15-20GB total) - subsequent runs are faster
- Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Model Download Issues
- First run downloads three models (~15-20GB total)
- Ensure stable internet connection and sufficient disk space
- Models are cached in `~/.cache/huggingface/hub/`
- Set custom cache location: `export HF_HOME=/path/to/cache`

### CUDA/PyTorch Issues
- Verify CUDA is installed: `nvidia-smi`
- Check PyTorch CUDA compatibility: Match PyTorch version to your CUDA version
- Reinstall PyTorch if needed: Visit [pytorch.org](https://pytorch.org)

## Model Comparison Results

**Last Run:** 2025-12-05 20:41:51

**Test Configuration:**
- Resolution: 512x512
- Number of Prompts: 5
- Seed: Random (different for each generation)
- Hardware: NVIDIA GeForce RTX 3070 Ti

### Performance Comparison

| Model | Steps | CFG | Mean Time/Image | Throughput | Peak GPU Memory |
|-------|-------|-----|-----------------|------------|-----------------|
| z-image-turbo | 9 | 0.0 | 161.20s ± 23.46s | 0.006 img/s | 19.29 GB |
| sd-turbo | 4 | 0.0 | 7.71s ± 0.09s | 0.130 img/s | 9.25 GB |
| sd-1.5 | 25 | 7.5 | 4.04s ± 3.02s | 0.248 img/s | 2.60 GB |

### Model Details

**z-image-turbo** (Tongyi-MAI/Z-Image-Turbo)
- 6B parameter Turbo DiT model
- Inference steps: 9
- Guidance scale: 0.0

**sd-turbo** (stabilityai/sd-turbo)
- Distilled SD 1.5 for speed
- Inference steps: 4
- Guidance scale: 0.0

**sd-1.5** (runwayml/stable-diffusion-v1-5)
- Classic SD 1.5 baseline
- Inference steps: 25
- Guidance scale: 7.5

### Test Prompts

1. Futuristic cityscape in heavy rain at night with neon reflections
2. Ancient forest with bioluminescent plants and drifting fog
3. Underwater research base with divers and service robots
4. Abstract geometric sculpture made of glass, smoke, and colored light
5. Snowy mountain village at dawn beneath an aurora

### Sample Outputs

Images are organized in the `comparison results/` directory by model name. Each model generated images for all test prompts.

**z-image-turbo**: `comparison results/z-image-turbo/`
**sd-turbo**: `comparison results/sd-turbo/`
**sd-1.5**: `comparison results/sd-1.5/`

## Visual Comparison Results

This section displays side-by-side comparisons of all three models (Z-Image-Turbo, SD-Turbo, and SD 1.5) for each test prompt, allowing you to visually compare their outputs.

### Prompt 1: Futuristic Cityscape
**Prompt:** Futuristic cityscape in heavy rain at night with neon reflections

| Z-Image-Turbo | SD-Turbo | SD 1.5 |
|---------------|----------|--------|
| ![Z-Image-Turbo](comparison%20results/z-image-turbo/prompt_01_20251205_201729.png) | ![SD-Turbo](comparison%20results/sd-turbo/prompt_01_20251205_203352.png) | ![SD-1.5](comparison%20results/sd-1.5/prompt_01_20251205_204139.png) |

### Prompt 2: Ancient Forest
**Prompt:** Ancient forest with bioluminescent plants and drifting fog

| Z-Image-Turbo | SD-Turbo | SD 1.5 |
|---------------|----------|--------|
| ![Z-Image-Turbo](comparison%20results/z-image-turbo/prompt_02_20251205_202022.png) | ![SD-Turbo](comparison%20results/sd-turbo/prompt_02_20251205_203359.png) | ![SD-1.5](comparison%20results/sd-1.5/prompt_02_20251205_204141.png) |

### Prompt 3: Underwater Research Base
**Prompt:** Underwater research base with divers and service robots

| Z-Image-Turbo | SD-Turbo | SD 1.5 |
|---------------|----------|--------|
| ![Z-Image-Turbo](comparison%20results/z-image-turbo/prompt_03_20251205_202316.png) | ![SD-Turbo](comparison%20results/sd-turbo/prompt_03_20251205_203407.png) | ![SD-1.5](comparison%20results/sd-1.5/prompt_03_20251205_204144.png) |

### Prompt 4: Abstract Geometric Sculpture
**Prompt:** Abstract geometric sculpture made of glass, smoke, and colored light

| Z-Image-Turbo | SD-Turbo | SD 1.5 |
|---------------|----------|--------|
| ![Z-Image-Turbo](comparison%20results/z-image-turbo/prompt_04_20251205_202607.png) | ![SD-Turbo](comparison%20results/sd-turbo/prompt_04_20251205_203415.png) | ![SD-1.5](comparison%20results/sd-1.5/prompt_04_20251205_204146.png) |

### Prompt 5: Snowy Mountain Village
**Prompt:** Snowy mountain village at dawn beneath an aurora

| Z-Image-Turbo | SD-Turbo | SD 1.5 |
|---------------|----------|--------|
| ![Z-Image-Turbo](comparison%20results/z-image-turbo/prompt_05_20251205_202802.png) | ![SD-Turbo](comparison%20results/sd-turbo/prompt_05_20251205_203423.png) | ![SD-1.5](comparison%20results/sd-1.5/prompt_05_20251205_204149.png) |

### Analysis

- **Fastest Model**: sd-1.5 (4.04s avg per image)
- **Most Memory Efficient**: sd-1.5 (2.60 GB peak)

## Z-Image-Turbo Extended Visual Results

For detailed visual results showcasing Z-Image-Turbo's capabilities across diverse photographic styles and scenarios, see [Z-Image-Turbo Visual Results](Z-Image-Turbo-Visual-Results.md).

