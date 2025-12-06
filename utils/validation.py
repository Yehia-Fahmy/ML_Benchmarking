"""
Validation utilities for model dimensions and optimizations.
"""

from typing import Dict, List, Tuple, Any, Optional
from .config import MODEL_CONFIGS, get_model_config


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_dimensions(
    model_name: str,
    height: int,
    width: int,
    strict: bool = False
) -> Tuple[bool, str, int, int]:
    """
    Validate that the requested dimensions are supported by the model.
    
    Args:
        model_name: Name of the model
        height: Requested image height
        width: Requested image width
        strict: If True, raise an error; if False, adjust dimensions
    
    Returns:
        Tuple of (is_valid, message, adjusted_height, adjusted_width)
    
    Raises:
        ValidationError: If strict=True and dimensions are invalid
    """
    config = get_model_config(model_name)
    
    if config is None:
        raise ValidationError(f"Unknown model: {model_name}")
    
    min_res = config.get("min_resolution", 256)
    max_res = config.get("max_resolution", 1024)
    supported = config.get("supported_resolutions", [256, 512, 768, 1024])
    
    adjusted_height = height
    adjusted_width = width
    messages = []
    is_valid = True
    
    # Check minimum resolution
    if height < min_res or width < min_res:
        is_valid = False
        adjusted_height = max(height, min_res)
        adjusted_width = max(width, min_res)
        messages.append(f"Resolution too small (min: {min_res}x{min_res})")
    
    # Check maximum resolution
    if height > max_res or width > max_res:
        is_valid = False
        adjusted_height = min(adjusted_height, max_res)
        adjusted_width = min(adjusted_width, max_res)
        messages.append(f"Resolution too large (max: {max_res}x{max_res})")
    
    # Check if resolution is in supported list (for models that require specific sizes)
    if model_name == "starflow-t2i":
        # STARFlow only supports 256x256
        if height != 256 or width != 256:
            is_valid = False
            adjusted_height = 256
            adjusted_width = 256
            messages.append("STARFlow only supports 256x256 resolution")
    
    # Ensure dimensions are multiples of 8 (required by most diffusion models)
    if adjusted_height % 8 != 0:
        adjusted_height = (adjusted_height // 8) * 8
        if adjusted_height != height:
            is_valid = False
            messages.append(f"Height adjusted to multiple of 8: {adjusted_height}")
    
    if adjusted_width % 8 != 0:
        adjusted_width = (adjusted_width // 8) * 8
        if adjusted_width != width:
            is_valid = False
            messages.append(f"Width adjusted to multiple of 8: {adjusted_width}")
    
    message = "; ".join(messages) if messages else "Dimensions valid"
    
    if strict and not is_valid:
        raise ValidationError(f"Invalid dimensions for {model_name}: {message}")
    
    return is_valid, message, adjusted_height, adjusted_width


def validate_model_support(model_name: str) -> Tuple[bool, str]:
    """
    Check if a model is supported.
    
    Args:
        model_name: Name of the model to check
    
    Returns:
        Tuple of (is_supported, message)
    """
    config = get_model_config(model_name)
    
    if config is None:
        return False, f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}"
    
    return True, f"Model {model_name} is supported"


def get_supported_optimizations(model_name: str) -> Dict[str, bool]:
    """
    Get dictionary of supported optimizations for a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary mapping optimization name to whether it's supported
    """
    config = get_model_config(model_name)
    
    if config is None:
        return {}
    
    return {
        "attention_slicing": config.get("supports_attention_slicing", False),
        "vae_slicing": config.get("supports_vae_slicing", False),
        "vae_tiling": config.get("supports_vae_tiling", False),
    }


def apply_optimizations(
    pipe,
    model_name: str,
    enable_attention_slicing: bool = True,
    enable_vae_slicing: bool = True,
    enable_vae_tiling: bool = True,
    verbose: bool = True
) -> List[str]:
    """
    Apply supported optimizations to a pipeline.
    
    Args:
        pipe: The diffusers pipeline object
        model_name: Name of the model
        enable_attention_slicing: Whether to enable attention slicing
        enable_vae_slicing: Whether to enable VAE slicing
        enable_vae_tiling: Whether to enable VAE tiling
        verbose: Whether to print status messages
    
    Returns:
        List of successfully applied optimizations
    """
    supported = get_supported_optimizations(model_name)
    applied = []
    
    # Attention slicing
    if enable_attention_slicing and supported.get("attention_slicing", False):
        try:
            pipe.enable_attention_slicing(slice_size="auto")
            applied.append("attention_slicing")
            if verbose:
                print("✓ Attention slicing enabled")
        except Exception as e:
            if verbose:
                print(f"⚠ Attention slicing not available: {e}")
    
    # VAE slicing
    if enable_vae_slicing and supported.get("vae_slicing", False):
        try:
            pipe.enable_vae_slicing()
            applied.append("vae_slicing")
            if verbose:
                print("✓ VAE slicing enabled")
        except Exception as e:
            if verbose:
                print(f"⚠ VAE slicing not available: {e}")
    
    # VAE tiling
    if enable_vae_tiling and supported.get("vae_tiling", False):
        try:
            pipe.enable_vae_tiling()
            applied.append("vae_tiling")
            if verbose:
                print("✓ VAE tiling enabled")
        except Exception as e:
            if verbose:
                print(f"⚠ VAE tiling not available: {e}")
    
    return applied


def validate_generation_params(
    model_name: str,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate and get generation parameters for a model.
    
    Args:
        model_name: Name of the model
        num_inference_steps: Override for inference steps (None = use default)
        guidance_scale: Override for guidance scale (None = use default)
    
    Returns:
        Dictionary with validated generation parameters
    """
    config = get_model_config(model_name)
    
    if config is None:
        raise ValidationError(f"Unknown model: {model_name}")
    
    # Use provided values or fall back to model defaults
    steps = num_inference_steps if num_inference_steps is not None else config["num_inference_steps"]
    cfg = guidance_scale if guidance_scale is not None else config["guidance_scale"]
    
    return {
        "num_inference_steps": steps,
        "guidance_scale": cfg,
    }

