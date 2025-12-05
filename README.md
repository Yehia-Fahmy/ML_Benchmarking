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
| NVIDIA GeForce RTX 3070 Ti | 8GB | ~689.38s | 0.00145 img/s | 2025-12-05 |

*Note: This table will be expanded as more hardware configurations are tested. Actual performance varies based on prompt complexity and system configuration.*

## Latest Benchmark Results

**Last Run:** 2025-12-05 09:13:36

**Configuration:**
- Model: Tongyi-MAI/Z-Image-Turbo
- Device: cuda
- Resolution: 1024x1024
- Inference Steps: 9
- Dtype: torch.bfloat16

**Performance Metrics:**
- Total Images Generated: 8
- Mean Inference Time: 689.38s ± 6.65s
- Throughput: 0.00145 images/second
- Warmup Time: 695.00s
- Peak GPU Memory: 19753 MB (19.29 GB)

## Generated Examples

Below are example images generated during the latest benchmark run, showcasing challenging prompts that test various capabilities:

### Example 1: Bilingual Text Rendering - Coffee Shop
**Prompt:** A vintage coffee shop storefront with a neon sign reading 'OPEN 24/7' in red letters above the door, and a wooden chalkboard menu displaying '咖啡 $3.50' in white chalk. Rain-slicked cobblestone street, warm golden light spilling through foggy windows, early morning blue hour atmosphere.

![Image 1](outputs/image_001_20251205_093642.png)

### Example 2: Complex Lighting & Reflections - Wine Glass
**Prompt:** A crystal wine glass filled with red wine, sitting on a marble countertop. Dramatic side lighting creates caustics and reflections on the surface. A single rose petal floats in the wine. Shallow depth of field, photorealistic macro photography style, with bokeh lights in the dark background.

![Image 2](outputs/image_002_20251205_094826.png)

### Example 3: Action & Motion - Dancer with Flour
**Prompt:** A professional dancer mid-leap in an abandoned warehouse, arms extended gracefully. Flour powder explodes around her body, frozen in mid-air, creating dramatic white clouds. Harsh sunlight streams through broken windows, creating god rays through the dust. High-speed photography, every particle visible and sharp.

![Image 3](outputs/image_003_20251205_100003.png)

### Example 4: Architectural Detail - Baroque Cathedral
**Prompt:** Interior of a grand baroque cathedral, ornate golden details on every surface. Sunlight streams through massive stained glass windows, casting colored light patterns on white marble floors. Elaborate ceiling frescoes depicting celestial scenes. Ultra-wide angle perspective looking up towards the dome, emphasizing scale and grandeur.

![Image 4](outputs/image_004_20251205_101128.png)

### Example 5: Materials & Textures - Water Droplet Macro
**Prompt:** Extreme close-up of a water droplet suspended on a spider's web at dawn. The droplet acts as a lens, containing a perfect miniature reflection of a sunrise landscape. Morning dew covers the entire web. Macro photography with perfect focus on water surface tension, iridescent light refractions.

![Image 5](outputs/image_005_20251205_102253.png)

### Example 6: Character Interaction - Japanese Calligraphy Lesson
**Prompt:** An elderly master calligrapher teaching a young student in a traditional Japanese study room. The master's weathered hands guide the student's brush, mid-stroke on rice paper. Ink bottles, scrolls, and brushes arranged on the low table. Soft natural light from shoji screens, expressions of concentration and wisdom. Photorealistic, intimate moment captured.

![Image 6](outputs/image_006_20251205_103418.png)

### Example 7: Surreal Photorealism - Desert Pocket Watch
**Prompt:** A giant vintage pocket watch partially buried in desert sand dunes, its face showing roman numerals. The watch is overgrown with lush green vines and blooming flowers emerging from its mechanisms. Golden hour lighting, long shadows, a single bird perched on the watch crown. Hyper-realistic, surreal juxtaposition.

![Image 7](outputs/image_007_20251205_104543.png)

### Example 8: Environmental Storytelling - Mars Astronaut Helmet
**Prompt:** An abandoned astronaut helmet on the surface of Mars, half-buried in red dust. The helmet's visor reflects the pink Martian sky and distant Earth as a blue dot. Small rocks and footprints leading away into the distance. Cinematic composition, sense of isolation and mystery, photorealistic space photography aesthetic.

![Image 8](outputs/image_008_20251205_105709.png)

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

