# Z-Image-Turbo Local Inference

A high-performance local inference project for [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), a powerful 6B parameter image generation model with sub-second inference latency.

## Features

- **Intelligent GPU Detection**: Automatically detects and uses NVIDIA CUDA, Apple MPS (M1/M2/M3), or falls back to CPU
- **Configurable Prompts**: Easy-to-edit configuration section for batch image generation
- **Comprehensive Benchmarking**: Tracks inference time, memory usage, GPU utilization, and throughput
- **Performance Profiling**: Includes warmup runs and detailed per-image metrics
- **Automatic Result Export**: Saves benchmark data to JSON and updates this README
- **Multi-Platform Support**: Works on Linux, macOS (Apple Silicon), and Windows with appropriate hardware

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 16GB RAM (recommended for 1024x1024 images)
- 10GB disk space for model weights

### GPU Requirements (Recommended)
- **NVIDIA GPU**: Any GPU with 16GB+ VRAM (e.g., RTX 4080, A100, H100)
- **Apple Silicon**: M1/M2/M3 with 16GB+ unified memory
- **Note**: CPU inference is supported but will be significantly slower

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

### Basic Usage

1. **Edit the configuration section** in `inference.py`:
   ```python
   PROMPTS = [
       "Your first prompt here",
       "Your second prompt here",
       # Add more prompts...
   ]
   
   IMAGE_HEIGHT = 1024
   IMAGE_WIDTH = 1024
   NUM_INFERENCE_STEPS = 9  # 8 DiT forwards for Turbo
   ```

2. **Run the inference script:**
   ```bash
   python inference.py
   ```

3. **View results:**
   - Generated images: `outputs/` directory
   - Benchmark data: `benchmark_results.json`
   - Summary: Console output and this README (automatically updated)

### Configuration Options

Edit these variables at the top of `inference.py`:

| Variable | Description | Default |
|----------|-------------|---------|
| `PROMPTS` | List of text prompts to generate | 3 example prompts |
| `IMAGE_HEIGHT` | Output image height in pixels | 1024 |
| `IMAGE_WIDTH` | Output image width in pixels | 1024 |
| `NUM_INFERENCE_STEPS` | Number of denoising steps | 9 (8 NFEs) |
| `GUIDANCE_SCALE` | Classifier-free guidance scale | 0.0 (recommended for Turbo) |
| `SEED` | Random seed for reproducibility | 42 |
| `USE_MODEL_COMPILATION` | Enable torch.compile for speed | False |
| `OUTPUT_DIR` | Directory for generated images | "outputs" |

### Advanced Options

**Model Compilation:**
Enable `USE_MODEL_COMPILATION = True` for faster inference after the first run (first generation will be slower due to compilation overhead).

**Flash Attention:**
Automatically enabled on NVIDIA GPUs if available for improved memory efficiency.

## GPU Compatibility

### NVIDIA CUDA
- **Supported**: All CUDA-compatible GPUs (compute capability 3.5+)
- **Recommended**: RTX 30/40 series, A100, H100 for optimal performance
- **dtype**: bfloat16 (automatic)
- **Features**: Full support including detailed GPU metrics

### Apple Silicon (MPS)
- **Supported**: M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max, M3, M3 Pro, M3 Max
- **dtype**: float32 (MPS doesn't fully support bfloat16 yet)
- **Features**: Hardware acceleration with unified memory

### CPU Fallback
- **Supported**: All x86_64 CPUs
- **dtype**: float32
- **Note**: Very slow; only recommended for testing or if no GPU is available

## Understanding the Output

### Console Output
The script provides detailed progress information:
1. Device detection and selection
2. Model loading status
3. Warmup run timing
4. Per-image generation progress
5. Comprehensive benchmark summary

### Generated Images
Images are saved to `outputs/` with descriptive filenames:
```
image_001_20231205_143052.png
image_002_20231205_143055.png
...
```

### Benchmark Results
`benchmark_results.json` contains:
- System and model configuration
- Per-image metrics (time, memory, throughput)
- Aggregate statistics (mean, std, min, max)
- GPU utilization data (if available)

## Troubleshooting

### Out of Memory Errors
- Reduce `IMAGE_HEIGHT` and `IMAGE_WIDTH` to 512 or 768
- Close other GPU-intensive applications
- Enable CPU offloading (requires code modification)

### Slow Performance
- Ensure you're using a GPU (check device detection output)
- Enable `USE_MODEL_COMPILATION = True` for subsequent runs
- Verify PyTorch is using the correct CUDA/MPS backend

### Model Download Issues
- First run downloads ~12GB of model weights
- Ensure stable internet connection
- Check available disk space
- Try setting `HF_HOME` environment variable to a different location

## Machine Benchmark Results

Comparative performance across different hardware configurations (1024x1024, 9 steps):

| Hardware | VRAM/Memory | Time per Image | Throughput | Date Tested |
|----------|-------------|----------------|------------|-------------|
| NVIDIA GeForce RTX 3070 Ti | 8GB | ~688.75s | 0.00145 img/s | 2025-12-04 |

*Note: This table will be expanded as more hardware configurations are tested. Actual performance varies based on prompt complexity and system configuration.*

## Latest Benchmark Results

**Last Run:** 2025-12-04 23:44:39

**Configuration:**
- Model: Tongyi-MAI/Z-Image-Turbo
- Device: cuda
- Resolution: 1024x1024
- Inference Steps: 9
- Dtype: torch.bfloat16

**Performance Metrics:**
- Total Images Generated: 3
- Mean Inference Time: 688.75s ± 3.07s
- Throughput: 0.00 images/second
- Warmup Time: 689.67s
- Peak GPU Memory: 19753 MB (19.29 GB)

## Generated Examples

Below are example images generated during the latest benchmark run:

### Example 1: Chinese Woman in Red Hanfu
**Prompt:** Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.

![Image 1](outputs/image_001_20251205_000734.png)

### Example 2: Serene Mountain Landscape
**Prompt:** A serene mountain landscape at sunset, with golden light reflecting off a crystal clear lake, surrounded by pine trees and snow-capped peaks in the distance.

![Image 2](outputs/image_002_20251205_001904.png)

### Example 3: Futuristic Cyberpunk Cityscape
**Prompt:** A futuristic cityscape at night with neon lights, flying vehicles, and holographic advertisements in a cyberpunk style.

![Image 3](outputs/image_003_20251205_003036.png)

## Model Information

- **Model**: Z-Image-Turbo by Tongyi-MAI
- **Parameters**: 6 billion
- **Architecture**: Scalable Single-Stream DiT (S3-DiT)
- **Specialties**: 
  - Photorealistic image generation
  - Bilingual text rendering (English & Chinese)
  - Strong instruction adherence
  - Sub-second inference on enterprise GPUs

## References

- [Hugging Face Model Card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [Official GitHub](https://github.com/TencentARC/Z-Image)
- [Research Paper](https://arxiv.org/abs/2511.22699)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)

## License

This project uses the Z-Image-Turbo model which is licensed under Apache 2.0.
See the [model card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) for details.

## Contributing

Feel free to open issues or submit pull requests for improvements!

---

**Created**: 2025-12-05  
**Last Updated**: Auto-updated after each inference run

