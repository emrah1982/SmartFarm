#!/usr/bin/env python3
# test_mineral_datasets.py - Test mineral deficiency datasets visibility

try:
    from multi_dataset_manager import YAMLBasedMultiDatasetManager
    print("✅ Successfully imported YAMLBasedMultiDatasetManager")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_mineral_datasets():
    """Test if mineral deficiency datasets are now visible in groups"""
    print("=== Testing Mineral Deficiency Datasets Visibility ===\n")
    
    manager = YAMLBasedMultiDatasetManager()
    
    # Show available groups
    print("Available Dataset Groups:")
    manager.show_available_groups()
    
    # Test unified_agricultural_model group
    print("\n=== Testing unified_agricultural_model group ===")
    success = manager.load_dataset_group('unified_agricultural_model')
    print(f"Load success: {success}")
    
    if success:
        print(f"Loaded {len(manager.datasets)} datasets:")
        mineral_found = False
        for ds in manager.datasets:
            name = ds.get('name', 'Unknown')
            desc = ds.get('description', 'No description')
            print(f"  - {name}: {desc}")
            
            # Check for mineral deficiency datasets
            if 'nutrient_deficiency' in name or 'deficiency_detection' in name:
                mineral_found = True
                print(f"    ✅ MINERAL DATASET FOUND: {name}")
        
        if mineral_found:
            print("\n🎉 SUCCESS: Mineral deficiency datasets are now visible!")
        else:
            print("\n❌ ISSUE: Mineral deficiency datasets still not found")
    
    # Test quick_unified_test group
    print("\n=== Testing quick_unified_test group ===")
    manager.datasets = []  # Clear previous datasets
    success = manager.load_dataset_group('quick_unified_test')
    print(f"Load success: {success}")
    
    if success:
        print(f"Loaded {len(manager.datasets)} datasets:")
        mineral_found = False
        for ds in manager.datasets:
            name = ds.get('name', 'Unknown')
            desc = ds.get('description', 'No description')
            print(f"  - {name}: {desc}")
            
            if 'nutrient_deficiency' in name or 'deficiency_detection' in name:
                mineral_found = True
                print(f"    ✅ MINERAL DATASET FOUND: {name}")
        
        if mineral_found:
            print("\n🎉 SUCCESS: Mineral deficiency datasets found in quick test group!")
        else:
            print("\n❌ ISSUE: Mineral deficiency datasets not found in quick test group")

if __name__ == "__main__":
    test_mineral_datasets()
