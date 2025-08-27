import os
from multi_dataset_manager import YAMLBasedMultiDatasetManager

def main():
    """
    Main function to download, process, and merge a specified dataset group.
    """
    # Configuration
    output_dir = "datasets/merged_dataset"
    # This group is defined in config_datasets.yaml and contains a comprehensive list of datasets.
    dataset_group_to_download = "unified_agricultural_model" 

    print(f"🚀 Starting dataset processing for group: {dataset_group_to_download}")

    # Initialize the manager
    manager = YAMLBasedMultiDatasetManager(output_dir=output_dir)

    # 1. Load the specified dataset group
    if not manager.load_dataset_group(dataset_group_to_download):
        print(f"❌ Failed to load dataset group '{dataset_group_to_download}'. Aborting.")
        return

    # 2. Download all datasets in the loaded group
    print("\n⬇️  Step 2: Downloading all datasets...")
    download_success = manager.download_all_datasets()
    if not download_success:
        print("❌ Some datasets failed to download. Please check logs. Aborting merge.")
        return
    print("✅ All datasets downloaded successfully.")

    # 3. Create a unified class mapping
    print("\n🗺️  Step 3: Creating unified class mapping...")
    class_map = manager.create_unified_class_mapping()
    if not class_map:
        print("❌ Failed to create class mapping. Aborting.")
        return
    print("✅ Unified class mapping created.")

    # 4. Merge the datasets
    print("\n🔄 Step 4: Merging datasets...")
    merged_path = manager.merge_datasets()
    if not merged_path:
        print("❌ Failed to merge datasets.")
        return
    
    print(f"\n🎉 Success! All datasets for group '{dataset_group_to_download}' have been processed.")
    print(f"📁 Merged dataset is available at: {merged_path}")
    print(f"📝 A report has been generated: {os.path.join(merged_path, 'processing_report.md')}")

if __name__ == "__main__":
    main()
