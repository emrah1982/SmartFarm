#!/usr/bin/env python3
# model_downloader.py - Utility for downloading YOLO11 models

import os
import requests
import sys
from pathlib import Path

def download_yolo11_models(save_dir=None, selected_models=None):
    """
    Download YOLO11 models from GitHub releases
    
    Args:
        save_dir: Directory to save models (default: ./yolo11_models)
        selected_models: List of specific models to download (default: all)
    
    Returns:
        List of paths to downloaded models
    """
    # Use default save directory if not specified
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "yolo11_models")
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Base URL for YOLOv11 models
    base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
    
    # List of model variants
    all_model_variants = [
        "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",  # Detection models
        "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",  # Segmentation models
        "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt",  # Classification models
        "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",  # Pose models
        "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt"  # OBB models
    ]
    
    # Determine which models to download
    model_variants = selected_models if selected_models else all_model_variants
    
    # Keep track of downloaded model paths
    downloaded_models = []
    
    # Download each model
    for model in model_variants:
        download_url = base_url + model
        save_path = os.path.join(save_dir, model)
        
        # Skip if model already exists
        if os.path.exists(save_path):
            print(f"Model {model} already exists at {save_path}")
            downloaded_models.append(save_path)
            continue
        
        print(f"Downloading {model}...")
        try:
            response = requests.get(download_url)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            with open(save_path, "wb") as f:
                f.write(response.content)
                
            print(f"Saved to {save_path}")
            downloaded_models.append(save_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {model}: {e}")
    
    print("YOLO11 models download completed!")
    return downloaded_models

def download_specific_model_type(model_type="detection", size="m", save_dir=None):
    """
    Download a specific type of YOLO11 model
    
    Args:
        model_type: Type of model to download (detection, segmentation, classification, pose, obb)
        size: Model size (s, m, l, x)
        save_dir: Directory to save models
    
    Returns:
        Path to downloaded model
    """
    # Validate model_type
    valid_types = {
        "detection": "",
        "segmentation": "-seg",
        "classification": "-cls",
        "pose": "-pose",
        "obb": "-obb"
    }
    
    if model_type not in valid_types:
        print(f"Invalid model type. Choose from: {', '.join(valid_types.keys())}")
        return None
    
    # Validate size
    valid_sizes = ["s", "m", "l", "x"]
    if size not in valid_sizes:
        print(f"Invalid model size. Choose from: {', '.join(valid_sizes)}")
        return None
    
    # Construct model name
    model_suffix = valid_types[model_type]
    model_name = f"yolo11{size}{model_suffix}.pt"
    
    # Download the model
    models = download_yolo11_models(save_dir, [model_name])
    
    # Return the path to the downloaded model
    return models[0] if models else None

if __name__ == "__main__":
    # If run directly, download all models or specific ones
    print("YOLO11 Model Downloader")
    print("======================")
    
    # Determine save directory
    default_dir = os.path.join(os.getcwd(), "yolo11_models")
    if len(sys.argv) > 1:
        save_dir = sys.argv[1]
    else:
        save_dir = input(f"Enter save directory (default: {default_dir}): ") or default_dir
    
    # Ask whether to download all models
    download_all = input("Download all models? (y/n, default: n): ").lower() == 'y'
    
    if download_all:
        download_yolo11_models(save_dir)
    else:
        # Ask for model type
        print("\nSelect model type:")
        print("1. Detection (default)")
        print("2. Segmentation")
        print("3. Classification")
        print("4. Pose")
        print("5. OBB (Oriented Bounding Box)")
        
        model_type_map = {
            "1": "detection",
            "2": "segmentation",
            "3": "classification",
            "4": "pose",
            "5": "obb"
        }
        
        model_type_choice = input("Enter choice (1-5, default: 1): ") or "1"
        model_type = model_type_map.get(model_type_choice, "detection")
        
        # Ask for model size
        print("\nSelect model size:")
        print("1. Small (s)")
        print("2. Medium (m) (default)")
        print("3. Large (l)")
        print("4. Extra Large (x)")
        
        size_map = {
            "1": "s",
            "2": "m",
            "3": "l",
            "4": "x"
        }
        
        size_choice = input("Enter choice (1-4, default: 2): ") or "2"
        size = size_map.get(size_choice, "m")
        
        # Download the selected model
        model_path = download_specific_model_type(model_type, size, save_dir)
        
        if model_path:
            print(f"\nModel downloaded successfully to: {model_path}")
