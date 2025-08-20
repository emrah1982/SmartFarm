#!/usr/bin/env python3
# memory_utils.py - Memory management utilities for YOLO11 training

import os
import gc
import torch
import time
import psutil
import numpy as np
from typing import Dict, Any, Callable

def show_memory_usage(label=""):
    """Display memory usage"""
    # Get memory usage for current process
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # Calculate RAM usage in MB
    ram_usage = mem_info.rss / (1024 * 1024)
    
    # Check GPU memory usage (if available)
    gpu_memory = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory.append(torch.cuda.memory_allocated(i) / (1024 * 1024))
    
    # Display information
    print(f"\n--- Memory Status {label} ---")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    
    if gpu_memory:
        for i, mem in enumerate(gpu_memory):
            print(f"GPU {i} Memory Usage: {mem:.2f} MB")
    
    return ram_usage

def clean_memory():
    """Clean memory - both CPU and GPU"""
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Call Python garbage collector
    gc.collect()
    
    # Show status after cleanup
    show_memory_usage("After Cleanup")

# Bu fonksiyon artık gerekli değil, çünkü callbacks ultralytics'in yeni sürümlerinde farklı şekilde çalışıyor
# Bunun yerine training.py içinde doğrudan temizleme yapıyoruz
"""
def run_training_with_memory_cleanup(model, train_args, cleanup_frequency=10):
    # Bu fonksiyon artık kullanılmıyor
    pass
"""

def monitor_memory_usage(interval=60):
    """
    Monitor memory usage at regular intervals in a separate thread
    
    Args:
        interval: Time between memory checks in seconds
    """
    import threading
    
    def _monitor():
        count = 0
        while True:
            count += 1
            show_memory_usage(f"Monitor Check #{count}")
            time.sleep(interval)
    
    # Start monitoring in a daemon thread
    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    print(f"Memory monitoring started (every {interval} seconds)")
    return thread

def estimate_batch_size(model_name, image_size, available_memory_gb=None):
    """
    Estimate appropriate batch size based on model size and available memory
    
    Args:
        model_name: Name of the model (e.g. 'yolo11n.pt', 'yolo11m.pt')
        image_size: Size of input images (e.g. 640)
        available_memory_gb: Available GPU memory in GB (if None, will be detected)
    
    Returns:
        Recommended batch size
    """
    # Get available GPU memory if not specified
    if available_memory_gb is None and torch.cuda.is_available():
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Get allocated memory
        allocated_memory = torch.cuda.memory_allocated(0)
        # Calculate available memory
        available_memory = total_memory - allocated_memory
        available_memory_gb = available_memory / (1024**3)  # Convert to GB
    elif available_memory_gb is None:
        # Default to 4GB if no GPU
        available_memory_gb = 4
    
    # Model complexity factors (estimated relative memory usage)
    model_factors = {
        'yolo11n': 0.3,   # Nano model
        'yolo11s': 0.5,   # Small model
        'yolo11m': 1.0,   # Medium model (reference)
        'yolo11l': 2.0,   # Large model
        'yolo11x': 3.0    # XLarge model
    }
    
    # Get model factor (default to medium if unknown)
    model_type = next((k for k in model_factors.keys() if k in model_name.lower()), 'yolo11m')
    model_factor = model_factors[model_type]
    
    # Calculate image size factor (relative to 640x640)
    img_size_factor = (image_size / 640)**2
    
    # Calculate base batch size based on available memory
    # This is calibrated for yolo11m at 640x640 using ~1.5GB per batch of 16
    base_batch_size = int((available_memory_gb / 1.5) * 16)
    
    # Apply model and image size adjustments
    adjusted_batch_size = int(base_batch_size / (model_factor * img_size_factor))
    
    # Ensure batch size is at least 1 and a power of 2 (for efficiency)
    batch_size = max(1, 2 ** int(np.log2(adjusted_batch_size)))
    
    print(f"Estimated batch size for {model_name} at {image_size}x{image_size}:")
    print(f"  Available memory: {available_memory_gb:.2f} GB")
    print(f"  Recommended batch size: {batch_size}")
    
    return batch_size

def get_system_info():
    """Get detailed system information"""
    info = {}
    
    # Python and OS info
    import platform
    info['python_version'] = platform.python_version()
    info['os'] = platform.system()
    info['os_version'] = platform.version()
    
    # CPU info
    cpu_count = os.cpu_count()
    info['cpu_count'] = cpu_count
    
    # RAM info
    total_ram = psutil.virtual_memory().total / (1024**3)  # GB
    available_ram = psutil.virtual_memory().available / (1024**3)  # GB
    info['total_ram'] = f"{total_ram:.2f} GB"
    info['available_ram'] = f"{available_ram:.2f} GB"
    
    # GPU info
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_version'] = torch.version.cuda
        
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)  # GB
            gpu_info.append({
                'name': props.name,
                'total_memory': f"{total_mem:.2f} GB"
            })
        info['gpus'] = gpu_info
    else:
        info['cuda_available'] = False
    
    # PyTorch info
    info['torch_version'] = torch.__version__
    
    return info

if __name__ == "__main__":
    print("Memory Utilities for YOLO11 Training")
    print("\nCurrent system information:")
    system_info = get_system_info()
    for key, value in system_info.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                for k, v in item.items():
                    print(f"  - {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("\nCurrent memory status:")
    show_memory_usage("Current")
