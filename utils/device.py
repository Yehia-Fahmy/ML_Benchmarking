"""
CUDA device detection and memory management utilities.
This module only supports NVIDIA CUDA GPUs.
"""

import gc
import sys
from typing import Tuple

import torch


def detect_cuda_device() -> Tuple[str, torch.dtype]:
    """
    Detect NVIDIA CUDA GPU and select appropriate dtype.
    
    Returns:
        Tuple of (device_name, dtype)
    
    Raises:
        RuntimeError: If no CUDA GPU is available
    """
    print("=" * 70)
    print("DEVICE DETECTION")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("✗ No NVIDIA CUDA GPU detected!")
        print("  This benchmarking suite requires an NVIDIA GPU with CUDA support.")
        print("  Please ensure you have:")
        print("    1. An NVIDIA GPU installed")
        print("    2. NVIDIA drivers installed")
        print("    3. PyTorch with CUDA support installed")
        print()
        print("  To check PyTorch CUDA support:")
        print("    python -c \"import torch; print(torch.cuda.is_available())\"")
        print()
        print("  To install PyTorch with CUDA:")
        print("    Visit https://pytorch.org for installation instructions")
        print("=" * 70)
        raise RuntimeError("CUDA GPU required but not available")
    
    device = "cuda"
    dtype = torch.bfloat16
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cuda_version = torch.version.cuda
    
    print(f"✓ NVIDIA CUDA GPU detected: {gpu_name}")
    print(f"  - Total Memory: {gpu_memory:.2f} GB")
    print(f"  - CUDA Version: {cuda_version}")
    print(f"  - Selected dtype: bfloat16")
    print("=" * 70)
    print()
    
    return device, dtype


def require_cuda() -> bool:
    """
    Check if CUDA is available and exit if not.
    
    Returns:
        True if CUDA is available
    
    Note:
        Exits the program with error code 1 if CUDA is not available
    """
    if not torch.cuda.is_available():
        print("ERROR: This script requires an NVIDIA CUDA GPU.")
        print("No CUDA device was detected. Please ensure you have:")
        print("  1. An NVIDIA GPU installed")
        print("  2. NVIDIA drivers installed") 
        print("  3. PyTorch with CUDA support")
        sys.exit(1)
    return True


def get_gpu_memory_usage() -> float:
    """
    Get current GPU memory usage in MB.
    
    Returns:
        Memory usage in megabytes, or 0.0 if CUDA is not available
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0


def get_peak_gpu_memory() -> float:
    """
    Get peak GPU memory usage in MB.
    
    Returns:
        Peak memory usage in megabytes, or 0.0 if CUDA is not available
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def reset_peak_memory_stats():
    """Reset peak memory statistics for accurate per-generation tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cleanup_memory(pipe=None):
    """
    Aggressively clean up GPU memory.
    
    Args:
        pipe: Optional pipeline object to delete
    """
    if pipe is not None:
        del pipe
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


def get_gpu_info() -> dict:
    """
    Get comprehensive GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    props = torch.cuda.get_device_properties(0)
    
    return {
        "available": True,
        "name": props.name,
        "total_memory_gb": props.total_memory / (1024**3),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
    }

