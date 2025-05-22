#!/usr/bin/env python3
# hyperparameters.py - Hyperparameter management for YOLO11 training

import os
import yaml

def create_hyperparameters_file():
    """Create hyp.yaml file with optimized hyperparameters"""
    hyp_path = "hyp.yaml"

    # Don't recreate if it already exists
    if os.path.exists(hyp_path):
        print(f"{hyp_path} already exists. Using existing file.")
        return hyp_path

    # Define hyperparameters
    hyperparameters = {
        # Learning rate and optimizer settings
        "lr0": 0.005,
        "lrf": 0.0001,
        "momentum": 0.937,
        "weight_decay": 0.0005,  # Increased to prevent overfitting
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # Loss function weights
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 2.0,

        # Data augmentation settings
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,

        # Early stopping settings
        "patience": 0,
        "min_delta": 0.001,

        # Other settings
        "label_smoothing": 0.0,
        "nbs": 64
    }

    # Create YAML file
    with open(hyp_path, "w") as f:
        yaml.dump(hyperparameters, f, sort_keys=False)

    print(f"Hyperparameters file created: {hyp_path}")
    return hyp_path

def load_hyperparameters(hyp_path):
    """Load hyperparameters from file"""
    try:
        with open(hyp_path, "r") as f:
            hyperparameters = yaml.safe_load(f)
        print(f"Hyperparameters loaded: {hyp_path}")

        # Show summary
        print("Important hyperparameters:")
        important_params = ["lr0", "mosaic", "fliplr", "patience", "box", "cls"]
        for key in important_params:
            if key in hyperparameters:
                print(f"  {key}: {hyperparameters[key]}")

        return hyperparameters
    except Exception as e:
        print(f"Error loading hyperparameters: {e}")
        print("Using default hyperparameters.")
        return {}

def get_recommended_hyperparameters(dataset_size=None, model_size="medium", target_device="gpu"):
    """
    Get recommended hyperparameters based on dataset size, model size, and target device
    
    Args:
        dataset_size: Number of training images (small: <500, medium: 500-5000, large: >5000)
        model_size: Model size (nano, small, medium, large, xlarge)
        target_device: Target device (cpu, gpu)
        
    Returns:
        Dictionary with recommended hyperparameters
    """
    # Base hyperparameters (from default)
    base_hyperparameters = {
        "lr0": 0.01,
        "weight_decay": 0.001,
        "warmup_epochs": 3.0,
        "mosaic": 1.0,
        "fliplr": 0.5,
        "patience": 20
    }
    
    # Adjust based on dataset size
    if dataset_size:
        if dataset_size < 500:  # Small dataset
            base_hyperparameters.update({
                "mosaic": 1.0,  # Full mosaic augmentation
                "fliplr": 0.5,
                "weight_decay": 0.0015,  # Stronger regularization for small datasets
                "patience": 30,  # More patience for small datasets
                "warmup_epochs": 5.0  # Longer warmup for small datasets
            })
        elif dataset_size > 5000:  # Large dataset
            base_hyperparameters.update({
                "mosaic": 0.8,  # Slightly reduce mosaic for large datasets
                "weight_decay": 0.0005,  # Less regularization needed
                "patience": 15  # Less patience needed for large datasets
            })
    
    # Adjust based on model size
    model_size_lower = model_size.lower()
    if model_size_lower == "nano" or model_size_lower == "small":
        base_hyperparameters.update({
            "lr0": 0.01,  # Higher learning rate for smaller models
        })
    elif model_size_lower == "large" or model_size_lower == "xlarge":
        base_hyperparameters.update({
            "lr0": 0.005,  # Lower learning rate for larger models
            "warmup_epochs": 5.0  # Longer warmup for larger models
        })
    
    # Adjust based on target device
    if target_device.lower() == "cpu":
        base_hyperparameters.update({
            "mosaic": 0.5,  # Reduce augmentations to speed up training on CPU
            "warmup_epochs": 1.0  # Shorter warmup for CPU training
        })
    
    return base_hyperparameters

def customize_hyperparameters():
    """Interactively customize hyperparameters"""
    # Start with default hyperparameters
    hyperparameters = {}
    
    if os.path.exists("hyp.yaml"):
        try:
            with open("hyp.yaml", "r") as f:
                hyperparameters = yaml.safe_load(f)
            print("Loaded existing hyperparameters as starting point.")
        except Exception as e:
            print(f"Error loading existing hyperparameters: {e}")
            print("Starting with empty hyperparameters.")
    
    print("\n===== Hyperparameter Customization =====")
    print("Enter new values or press Enter to keep current/default values.")
    
    # Learning rate
    current_lr = hyperparameters.get("lr0", 0.01)
    new_lr = input(f"Learning rate (current: {current_lr}): ")
    if new_lr:
        hyperparameters["lr0"] = float(new_lr)
    
    # Weight decay
    current_wd = hyperparameters.get("weight_decay", 0.001)
    new_wd = input(f"Weight decay (current: {current_wd}): ")
    if new_wd:
        hyperparameters["weight_decay"] = float(new_wd)
    
    # Mosaic augmentation
    current_mosaic = hyperparameters.get("mosaic", 1.0)
    new_mosaic = input(f"Mosaic augmentation (0.0-1.0, current: {current_mosaic}): ")
    if new_mosaic:
        hyperparameters["mosaic"] = float(new_mosaic)
    
    # Patience for early stopping
    current_patience = hyperparameters.get("patience", 20)
    new_patience = input(f"Early stopping patience (current: {current_patience}): ")
    if new_patience:
        hyperparameters["patience"] = int(new_patience)
    
    # Save customized hyperparameters
    save_path = "hyp_custom.yaml"
    with open(save_path, "w") as f:
        yaml.dump(hyperparameters, f, sort_keys=False)
    
    print(f"Customized hyperparameters saved to {save_path}")
    return save_path

if __name__ == "__main__":
    # If this file is run directly, create and load hyperparameters file
    hyp_path = create_hyperparameters_file()
    hyperparameters = load_hyperparameters(hyp_path)
    print("\nTo customize hyperparameters, run with --customize flag")
