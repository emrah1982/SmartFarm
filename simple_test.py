from multi_dataset_manager import YAMLBasedMultiDatasetManager

manager = YAMLBasedMultiDatasetManager()
success = manager.load_dataset_group('unified_agricultural_model')
print(f'Load success: {success}')

if success:
    print(f'Total datasets loaded: {len(manager.datasets)}')
    mineral_found = False
    for ds in manager.datasets:
        name = ds.get('name', 'Unknown')
        print(f'Dataset: {name}')
        if 'nutrient' in name or 'deficiency' in name:
            print(f'✅ FOUND MINERAL DATASET: {name}')
            mineral_found = True
    
    if not mineral_found:
        print('❌ No mineral datasets found')
    else:
        print('🎉 Mineral datasets are now visible!')
