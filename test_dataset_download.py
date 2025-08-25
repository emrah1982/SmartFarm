#!/usr/bin/env python3
# test_dataset_download.py - Test dataset download and analyze class names

import os
import yaml
from dataset_utils import download_dataset, analyze_dataset

def test_single_dataset():
    """Test downloading a single dataset to see actual class names"""
    print("===== Testing Dataset Download =====")
    
    # Test with fruit_ripeness dataset (should be simple)
    test_url = "https://universe.roboflow.com/ds/KXxiCfvas4?key=LQed1EPrBo"
    test_dir = "datasets/test_fruit_ripeness"
    
    print(f"Testing URL: {test_url}")
    print(f"Download directory: {test_dir}")
    
    # Download dataset
    success = download_dataset(test_url, test_dir)
    
    if success:
        print("✅ Download successful!")
        
        # Analyze the dataset
        analyze_dataset(test_dir)
        
        # Read the data.yaml to see actual class names
        yaml_path = os.path.join(test_dir, 'data.yaml')
        if os.path.exists(yaml_path):
            print(f"\n===== Actual Class Names from data.yaml =====")
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            print(f"Number of classes: {data.get('nc', 'unknown')}")
            print(f"Class names: {data.get('names', [])}")
            
            # Check if these match our config expectations
            expected_classes = ["ripe", "unripe"]
            actual_classes = data.get('names', [])
            
            print(f"\n===== Comparison =====")
            print(f"Expected: {expected_classes}")
            print(f"Actual: {actual_classes}")
            
            if set(actual_classes) == set(expected_classes):
                print("✅ Classes match expectations!")
            else:
                print("⚠️ Classes don't match - this could cause 'unknown' issues")
                print(f"Missing from config: {set(actual_classes) - set(expected_classes)}")
                print(f"Extra in config: {set(expected_classes) - set(actual_classes)}")
        
        return actual_classes
    else:
        print("❌ Download failed!")
        return None

def test_disease_dataset():
    """Test the main disease dataset"""
    print("\n===== Testing Disease Dataset =====")
    
    test_url = "https://universe.roboflow.com/ds/0UULi7Pnno?key=PU2zi8AslM"
    test_dir = "datasets/test_diseases"
    
    print(f"Testing URL: {test_url}")
    
    success = download_dataset(test_url, test_dir)
    
    if success:
        yaml_path = os.path.join(test_dir, 'data.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            actual_classes = data.get('names', [])
            print(f"Disease dataset classes ({len(actual_classes)}):")
            for i, cls in enumerate(actual_classes):
                print(f"  {i}: {cls}")
            
            return actual_classes
    
    return None

if __name__ == "__main__":
    # Test fruit dataset first (simpler)
    fruit_classes = test_single_dataset()
    
    # Test disease dataset
    disease_classes = test_disease_dataset()
    
    print(f"\n===== Summary =====")
    if fruit_classes:
        print(f"Fruit classes: {fruit_classes}")
    if disease_classes:
        print(f"Disease classes count: {len(disease_classes)}")
        print("First 10 disease classes:")
        for cls in disease_classes[:10]:
            print(f"  - {cls}")
