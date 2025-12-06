"""
Configuration and model definitions for the benchmarking suite.
"""

from typing import Dict, Any, Optional

# ============================================================================
# DEFAULT PROMPTS
# ============================================================================

DEFAULT_PROMPTS = [
    "Futuristic cityscape in heavy rain at night with neon reflections",
    "Ancient forest with bioluminescent plants and drifting fog",
    "Underwater research base with divers and service robots",
    "Abstract geometric sculpture made of glass, smoke, and colored light",
    "Snowy mountain village at dawn beneath an aurora",
]

# ============================================================================
# DEFAULT GENERATION SETTINGS
# ============================================================================

DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_SEED = 42  # Default seed for reproducibility; use None for random
OUTPUT_DIR = "comparison_results"
BENCHMARKS_DIR = "benchmarks"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "z-image-turbo": {
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "pipeline_class": "ZImagePipeline",
        "num_inference_steps": 9,
        "guidance_scale": 0.0,
        "description": "6B parameter Turbo DiT model",
        "min_resolution": 256,
        "max_resolution": 1024,
        "supported_resolutions": [256, 512, 768, 1024],
        "requires_cfg": False,
        "supports_attention_slicing": True,
        "supports_vae_slicing": True,
        "supports_vae_tiling": True,
    },
    "sd-turbo": {
        "model_id": "stabilityai/sd-turbo",
        "pipeline_class": "AutoPipelineForText2Image",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "description": "Distilled SD 1.5 for speed",
        "min_resolution": 256,
        "max_resolution": 768,
        "supported_resolutions": [256, 512, 768],
        "requires_cfg": False,
        "supports_attention_slicing": True,
        "supports_vae_slicing": True,
        "supports_vae_tiling": True,
    },
    "sd-1.5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "pipeline_class": "StableDiffusionPipeline",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "description": "Classic SD 1.5 baseline",
        "min_resolution": 256,
        "max_resolution": 768,
        "supported_resolutions": [256, 512, 768],
        "requires_cfg": True,
        "supports_attention_slicing": True,
        "supports_vae_slicing": True,
        "supports_vae_tiling": True,
    },
    "starflow-t2i": {
        "model_id": "apple/starflow",
        "pipeline_class": "STARFlowPipeline",
        "num_inference_steps": None,  # Uses Jacobi iterations
        "guidance_scale": 3.6,
        "description": "STARFlow 3B - Transformer Autoregressive Flow for T2I",
        "min_resolution": 256,
        "max_resolution": 256,
        "supported_resolutions": [256],
        "requires_cfg": True,
        "supports_attention_slicing": False,
        "supports_vae_slicing": False,
        "supports_vae_tiling": False,
        "requires_external_repo": True,
        "repo_url": "https://github.com/apple/ml-starflow.git",
        "checkpoint_file": "starflow_3B_t2i_256x256.pth",
        "config_file": "configs/starflow_3B_t2i_256x256.yaml",
    },
}


def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'sd-turbo', 'z-image-turbo')
    
    Returns:
        Model configuration dictionary or None if not found
    """
    return MODEL_CONFIGS.get(model_name)


def get_available_models() -> list:
    """Get list of all available model names."""
    return list(MODEL_CONFIGS.keys())

