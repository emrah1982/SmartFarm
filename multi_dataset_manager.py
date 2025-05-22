#!/usr/bin/env python3
# multi_dataset_manager.py - Complete YAML-Based Multi-dataset management for YOLO11 training

import os
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import random
import logging

from dataset_utils import download_dataset, fix_directory_structure

# Augmentation import with fallback
try:
    from augmentation_utils import YOLOAugmentationPipeline, SmartAugmentationRecommender
    AUGMENTATION_AVAILABLE = True
except ImportError:
    print("⚠️  Augmentation utils not available. Basic functionality will be used.")
    AUGMENTATION_AVAILABLE = False

class YAMLBasedMultiDatasetManager:
    """
    Complete YAML-based multi-dataset manager for YOLO11 training
    Uses config_datasets.yaml for all dataset definitions and configurations
    """
    
    def __init__(self, output_dir="datasets/merged_dataset", config_file="config_datasets.yaml"):
        self.output_dir = output_dir
        self.config_file = config_file
        self.datasets = []
        self.class_mapping = {}
        self.hierarchical_classes = {}
        self.class_distribution = {}
        self.merged_stats = {}
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration from YAML file
        self.config = self._load_datasets_config()
        
        # Get class mapping from YAML config
        self.predefined_mapping = self._get_class_mapping_from_config()
        
        # Initialize augmentation pipeline if available
        self.augmentation_pipeline = None
        self.smart_recommender = None
        if AUGMENTATION_AVAILABLE:
            self.smart_recommender = SmartAugmentationRecommender()
        
        self.logger.info(f"✅ YAMLBasedMultiDatasetManager initialized")
        self.logger.info(f"📁 Config file: {self.config_file}")
        self.logger.info(f"📊 Available dataset groups: {list(self.config.get('dataset_groups', {}).keys())}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('multi_dataset_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_datasets_config(self):
        """Load datasets configuration from YAML file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"📋 Loaded config from: {self.config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {self.config_file}")
            print("Using fallback configuration...")
            return self._create_fallback_config()
        except yaml.YAMLError as e:
            print(f"❌ YAML parsing error: {e}")
            return self._create_fallback_config()
    
    def _create_fallback_config(self):
        """Create fallback configuration if config file is not available"""
        return {
            'datasets': {
                'base_datasets': {
                    'plant_village': {
                        'url': "https://universe.roboflow.com/new-workspace-mk0mj/plant-village-yh90l/dataset/1",
                        'description': "Plant Village Dataset",
                        'expected_classes': ["damaged", "healthy", "ripe", "unripe"],
                        'priority': 1,
                        'category': 'diseases'
                    }
                }
            },
            'dataset_groups': {
                'quick_test': {
                    'description': "Quick test group",
                    'datasets': ['plant_village']
                }
            },
            'global_settings': {
                'default_target_count_per_class': 2000,
                'default_image_size': 640,
                'auto_class_mapping': True
            }
        }
    
    def _get_class_mapping_from_config(self):
        """Get class mapping rules directly from YAML configuration"""
        # Get class mapping from config file
        config_mappings = self.config.get('class_mapping', {})
        
        if config_mappings:
            print("📋 Using class mappings from YAML config file")
            return config_mappings
        
        print("⚠️  No class mapping found in config, using default")
        return {}
    
    def get_available_dataset_groups(self):
        """Get list of available dataset groups from config"""
        groups = self.config.get('dataset_groups', {})
        return list(groups.keys())
    
    def get_dataset_group_info(self, group_name):
        """Get information about a specific dataset group"""
        groups = self.config.get('dataset_groups', {})
        return groups.get(group_name, {})
    
    def load_dataset_group(self, group_name):
        """Load datasets from a predefined group"""
        print(f"\n===== Loading Dataset Group: {group_name} =====")
        
        group_info = self.get_dataset_group_info(group_name)
        if not group_info:
            print(f"❌ Dataset group '{group_name}' not found!")
            return False
        
        dataset_names = group_info.get('datasets', [])
        print(f"📊 Group contains {len(dataset_names)} datasets")
        print(f"📝 Description: {group_info.get('description', 'No description')}")
        
        # Show group recommendations
        if 'recommended_model' in group_info:
            print(f"🤖 Recommended model: {group_info['recommended_model']}")
        if 'batch_size' in group_info:
            print(f"📦 Recommended batch size: {group_info['batch_size']}")
        if 'image_size' in group_info:
            print(f"🖼️  Recommended image size: {group_info['image_size']}")
        
        # Clear existing datasets
        self.datasets = []
        
        # Load datasets from all categories
        all_datasets = self._collect_all_datasets()
        
        # Add selected datasets
        for dataset_name in dataset_names:
            if dataset_name in all_datasets:
                dataset_info = all_datasets[dataset_name]
                self.add_dataset_from_config(dataset_name, dataset_info)
            else:
                print(f"⚠️  Dataset '{dataset_name}' not found in configuration")
        
        print(f"✅ Loaded {len(self.datasets)} datasets from group '{group_name}'")
        return True
    
    def _collect_all_datasets(self):
        """Collect all datasets from all categories"""
        all_datasets = {}
        datasets_section = self.config.get('datasets', {})
        
        for category_name, category_datasets in datasets_section.items():
            if isinstance(category_datasets, dict) and category_name not in ['default_target_count_per_class', 'default_image_size', 'auto_class_mapping']:
                all_datasets.update(category_datasets)
        
        return all_datasets
    
    def add_dataset_from_config(self, dataset_name, dataset_config):
        """Add a dataset using configuration from the YAML file"""
        self.datasets.append({
            'url': dataset_config['url'],
            'name': dataset_name,
            'local_path': f"datasets/{dataset_name}",
            'classes': [],
            'class_counts': {},
            'description': dataset_config.get('description', ''),
            'expected_classes': dataset_config.get('expected_classes', []),
            'priority': dataset_config.get('priority', 3),
            'category': dataset_config.get('category', 'general'),
            'resize_info': dataset_config.get('resize_info', 'Not specified'),
            'added_date': dataset_config.get('added_date', datetime.now().strftime('%Y-%m-%d')),
            'class_count': dataset_config.get('class_count', 'unknown'),
            'dataset_type': dataset_config.get('dataset_type', 'general')
        })
        
        print(f"📦 Added dataset: {dataset_name}")
        print(f"   📝 {dataset_config.get('description', 'No description')}")
        print(f"   🎯 Expected classes: {len(dataset_config.get('expected_classes', []))}")
        print(f"   🔄 Resize info: {dataset_config.get('resize_info', 'Not specified')}")
    
    def add_custom_dataset(self, roboflow_url, dataset_name=None, description="", expected_classes=None):
        """Add a custom dataset (not from config)"""
        if dataset_name is None:
            dataset_name = f"custom_{len(self.datasets) + 1}"
        
        if expected_classes is None:
            expected_classes = []
        
        self.datasets.append({
            'url': roboflow_url,
            'name': dataset_name,
            'local_path': f"datasets/{dataset_name}",
            'classes': [],
            'class_counts': {},
            'description': description or 'Custom dataset added by user',
            'expected_classes': expected_classes,
            'priority': 3,
            'category': 'custom',
            'resize_info': 'Not specified',
            'added_date': datetime.now().strftime('%Y-%m-%d'),
            'class_count': len(expected_classes) if expected_classes else 'unknown',
            'dataset_type': 'custom'
        })
        
        print(f"📦 Added custom dataset: {dataset_name}")
    
    def show_available_datasets(self):
        """Display all available datasets from config"""
        print("\n===== Available Datasets =====")
        
        all_datasets = self._collect_all_datasets()
        datasets_by_category = defaultdict(list)
        
        # Group datasets by category
        datasets_section = self.config.get('datasets', {})
        for category_name, category_datasets in datasets_section.items():
            if isinstance(category_datasets, dict) and category_name not in ['default_target_count_per_class', 'default_image_size', 'auto_class_mapping']:
                for dataset_name, dataset_info in category_datasets.items():
                    datasets_by_category[category_name].append((dataset_name, dataset_info))
        
        total_datasets = 0
        
        for category_name, datasets_list in datasets_by_category.items():
            print(f"\n📂 {category_name.upper().replace('_', ' ')}")
            print("-" * 50)
            
            for dataset_name, dataset_info in datasets_list:
                total_datasets += 1
                description = dataset_info.get('description', 'No description')
                class_count = dataset_info.get('class_count', 'Unknown')
                priority = dataset_info.get('priority', 'Unknown')
                resize_info = dataset_info.get('resize_info', 'Not specified')
                
                print(f"{total_datasets:2d}. {dataset_name}")
                print(f"    📝 {description}")
                print(f"    🏷️  Classes: {class_count} | Priority: {priority}")
                print(f"    🔄 Resize: {resize_info}")
                print()
        
        print(f"Total available datasets: {total_datasets}")
    
    def show_available_groups(self):
        """Display all available dataset groups with enhanced information"""
        print("\n===== Available Dataset Groups =====")
        
        groups = self.config.get('dataset_groups', {})
        
        for i, (group_name, group_info) in enumerate(groups.items(), 1):
            description = group_info.get('description', 'No description')
            datasets = group_info.get('datasets', [])
            use_case = group_info.get('use_case', 'General use')
            estimated_time = group_info.get('estimated_training_time', 'Unknown')
            recommended_model = group_info.get('recommended_model', 'Not specified')
            batch_size = group_info.get('batch_size', 'Auto')
            image_size = group_info.get('image_size', 'Auto')
            
            print(f"{i:2d}. {group_name}")
            print(f"    📝 {description}")
            print(f"    📊 Datasets: {len(datasets)} ({', '.join(datasets[:3])}{'...' if len(datasets) > 3 else ''})")
            print(f"    🎯 Use case: {use_case}")
            print(f"    ⏱️  Training time: {estimated_time}")
            print(f"    🤖 Model: {recommended_model} | Batch: {batch_size} | Size: {image_size}")
            print()
    
    def interactive_dataset_selection(self):
        """Interactive dataset selection with config-based options"""
        print("\n🎯 Dataset Selection")
        print("=" * 50)
        
        # Show available groups
        self.show_available_groups()
        
        groups = self.config.get('dataset_groups', {})
        group_list = list(groups.keys())
        
        print(f"\n{len(group_list) + 1}. Custom selection from individual datasets")
        print(f"{len(group_list) + 2}. Add custom Roboflow URL")
        print(f"{len(group_list) + 3}. Show detailed dataset information")
        
        # Get user choice
        while True:
            try:
                choice = input(f"\nSelect option (1-{len(group_list) + 3}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(group_list):
                    selected_group = group_list[choice_num - 1]
                    if self.load_dataset_group(selected_group):
                        return selected_group
                    else:
                        return None
                        
                elif choice_num == len(group_list) + 1:
                    return self._custom_individual_selection()
                    
                elif choice_num == len(group_list) + 2:
                    return self._add_custom_url()
                    
                elif choice_num == len(group_list) + 3:
                    self.show_available_datasets()
                    continue
                    
                else:
                    print(f"❌ Invalid choice. Enter 1-{len(group_list) + 3}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
    
    def _custom_individual_selection(self):
        """Select individual datasets from config"""
        print("\n📋 Individual Dataset Selection")
        self.show_available_datasets()
        
        all_datasets = self._collect_all_datasets()
        dataset_list = list(all_datasets.keys())
        
        print("\nEnter dataset numbers (comma-separated, e.g., 1,3,5-7):")
        selection = input("Selection: ").strip()
        
        selected_datasets = []
        try:
            for part in selection.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    for i in range(start, end + 1):
                        if 1 <= i <= len(dataset_list):
                            selected_datasets.append(dataset_list[i - 1])
                else:
                    i = int(part)
                    if 1 <= i <= len(dataset_list):
                        selected_datasets.append(dataset_list[i - 1])
            
            # Remove duplicates while preserving order
            selected_datasets = list(dict.fromkeys(selected_datasets))
            
            # Add selected datasets
            self.datasets = []
            for dataset_name in selected_datasets:
                dataset_info = all_datasets[dataset_name]
                self.add_dataset_from_config(dataset_name, dataset_info)
            
            print(f"✅ Selected {len(selected_datasets)} datasets:")
            for dataset in selected_datasets:
                print(f"   - {dataset}")
            
            return "custom_selection"
            
        except ValueError:
            print("❌ Invalid selection format")
            return None
    
    def _add_custom_url(self):
        """Add a custom Roboflow URL"""
        print("\n🔗 Add Custom Roboflow URL")
        
        url = input("Enter Roboflow URL: ").strip()
        if not url:
            print("❌ No URL provided")
            return None
        
        name = input("Enter dataset name (optional): ").strip()
        if not name:
            name = f"custom_{len(self.datasets) + 1}"
        
        description = input("Enter description (optional): ").strip()
        
        # Ask for expected classes
        classes_input = input("Enter expected classes (comma-separated, optional): ").strip()
        expected_classes = []
        if classes_input:
            expected_classes = [cls.strip() for cls in classes_input.split(',')]
        
        self.add_custom_dataset(url, name, description, expected_classes)
        return "custom_url"
    
    def get_global_settings(self):
        """Get global settings from config"""
        return self.config.get('global_settings', {
            'default_target_count_per_class': 2000,
            'default_image_size': 640,
            'auto_class_mapping': True,
            'default_augmentation_level': 'medium',
            'preserve_original_data': True
        })
    
    def download_all_datasets(self):
        """Download all datasets with enhanced error handling and progress tracking"""
        print("\n===== Downloading Datasets =====")
        
        successful_downloads = 0
        failed_downloads = []
        download_stats = {}
        
        for i, dataset in enumerate(self.datasets):
            print(f"\n[{i+1}/{len(self.datasets)}] Downloading: {dataset['name']}")
            print(f"📝 {dataset.get('description', 'No description')}")
            print(f"🔗 URL: {dataset['url']}")
            print(f"🔄 Expected resize: {dataset.get('resize_info', 'Not specified')}")
            
            try:
                success = download_dataset(dataset['url'], dataset['local_path'])
                if not success:
                    print(f"❌ ERROR: {dataset['name']} could not be downloaded!")
                    failed_downloads.append(dataset['name'])
                    continue
                
                successful_downloads += 1
                
                # Read class information
                data_yaml = os.path.join(dataset['local_path'], 'data.yaml')
                if os.path.exists(data_yaml):
                    with open(data_yaml, 'r') as f:
                        data = yaml.safe_load(f)
                        dataset['classes'] = data.get('names', [])
                        print(f"✅ Classes found: {dataset['classes']}")
                        
                        # Compare with expected classes
                        expected = dataset.get('expected_classes', [])
                        if expected and expected != ['to_be_analyzed']:
                            actual = set(dataset['classes'])
                            expected_set = set(expected)
                            if actual != expected_set:
                                print(f"⚠️  Class mismatch detected!")
                                print(f"   Expected: {expected}")
                                print(f"   Actual: {list(actual)}")
                
                # Analyze dataset distribution
                stats = self._analyze_dataset_distribution(dataset)
                download_stats[dataset['name']] = stats
                
            except Exception as e:
                print(f"❌ Exception during download of {dataset['name']}: {e}")
                failed_downloads.append(dataset['name'])
        
        # Summary
        print(f"\n📊 Download Summary:")
        print(f"✅ Successful: {successful_downloads}/{len(self.datasets)}")
        if failed_downloads:
            print(f"❌ Failed: {len(failed_downloads)} - {', '.join(failed_downloads)}")
        
        # Save download report
        self._save_download_report(download_stats, failed_downloads)
        
        return successful_downloads > 0
    
    def _analyze_dataset_distribution(self, dataset):
        """Analyze the class distribution of a dataset with detailed statistics"""
        print(f"\n--- {dataset['name']} Class Analysis ---")
        
        train_labels_dir = os.path.join(dataset['local_path'], 'labels', 'train')
        if not os.path.exists(train_labels_dir):
            print(f"❌ Label directory not found: {train_labels_dir}")
            return {}
        
        class_counts = Counter()
        total_annotations = 0
        total_images = 0
        bbox_sizes = []
        
        # Check all label files
        for label_file in os.listdir(train_labels_dir):
            if label_file.endswith('.txt'):
                total_images += 1
                file_path = os.path.join(train_labels_dir, label_file)
                try:
                    with open(file_path, 'r') as f:
                        file_annotations = 0
                        for line in f:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:
                                try:
                                    class_idx = int(parts[0])
                                    if 0 <= class_idx < len(dataset['classes']):
                                        class_name = dataset['classes'][class_idx]
                                        class_counts[class_name] += 1
                                        total_annotations += 1
                                        file_annotations += 1
                                        
                                        # Collect bbox size info
                                        width, height = float(parts[3]), float(parts[4])
                                        bbox_sizes.append(width * height)
                                        
                                except (ValueError, IndexError):
                                    pass
                except Exception as e:
                    print(f"⚠️  Error reading {label_file}: {e}")
        
        # Calculate statistics
        avg_annotations_per_image = total_annotations / total_images if total_images > 0 else 0
        avg_bbox_size = np.mean(bbox_sizes) if bbox_sizes else 0
        
        dataset['class_counts'] = dict(class_counts)
        dataset['total_annotations'] = total_annotations
        dataset['total_images'] = total_images
        dataset['avg_annotations_per_image'] = avg_annotations_per_image
        dataset['avg_bbox_size'] = avg_bbox_size
        
        stats = {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'avg_annotations_per_image': avg_annotations_per_image,
            'avg_bbox_size': avg_bbox_size,
            'class_counts': dict(class_counts)
        }
        
        print(f"📊 Images: {total_images}, Annotations: {total_annotations}")
        print(f"📈 Avg annotations/image: {avg_annotations_per_image:.2f}")
        print(f"📏 Avg bbox size: {avg_bbox_size:.4f}")
        
        if class_counts:
            print("📋 Class distribution:")
            for class_name, count in class_counts.most_common():
                percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        else:
            print("⚠️  No valid annotations found")
        
        return stats
    
    def create_unified_class_mapping(self):
        """Create unified class mapping using YAML configuration"""
        print("\n===== Creating Unified Class Mapping =====")
        
        # Collect all unique classes from datasets
        all_classes = set()
        for dataset in self.datasets:
            all_classes.update(dataset.get('classes', []))
        
        print(f"📊 Total unique classes found: {len(all_classes)}")
        
        # Get class mapping configuration from YAML
        config_mapping = self.config.get('class_mapping', {})
        global_settings = self.get_global_settings()
        auto_mapping = global_settings.get('auto_class_mapping', True)
        
        # Initialize mapping
        self.class_mapping = {}
        mapped_classes = set()
        
        # Apply mapping using YAML configuration
        if auto_mapping and config_mapping:
            print("🔄 Applying YAML-based class mapping...")
            
            for main_class, mapping_info in config_mapping.items():
                sub_classes = mapping_info.get('sub_classes', [])
                keywords = mapping_info.get('keywords', [])
                
                print(f"\n🏷️  Processing main class: {main_class}")
                
                # Direct mapping for sub_classes
                for sub_class in sub_classes:
                    if sub_class in all_classes:
                        self.class_mapping[sub_class] = main_class
                        mapped_classes.add(sub_class)
                        print(f"  ✅ Direct: '{sub_class}' → '{main_class}'")
                
                # Keyword-based mapping for unmapped classes
                for class_name in all_classes:
                    if class_name not in mapped_classes:
                        class_lower = class_name.lower()
                        for keyword in keywords:
                            if keyword.lower() in class_lower:
                                self.class_mapping[class_name] = main_class
                                mapped_classes.add(class_name)
                                print(f"  🔍 Keyword: '{class_name}' → '{main_class}' (keyword: {keyword})")
                                break
        
        # Handle unmapped classes
        unmapped_classes = all_classes - mapped_classes
        if unmapped_classes:
            print(f"\n⚠️  Unmapped classes found: {len(unmapped_classes)}")
            for class_name in unmapped_classes:
                print(f"  - {class_name}")
            
            # Check if 'unknown' category exists in config
            if 'unknown' in config_mapping:
                print("📝 Assigning unmapped classes to 'unknown' category")
                for class_name in unmapped_classes:
                    self.class_mapping[class_name] = 'unknown'
                    print(f"  📝 Mapped '{class_name}' → 'unknown'")
            else:
                # Create 'other' category
                print("📝 Creating 'other' category for unmapped classes")
                for class_name in unmapped_classes:
                    self.class_mapping[class_name] = 'other'
                    print(f"  📝 Mapped '{class_name}' → 'other'")
        
        # Create hierarchical class structure
        self.hierarchical_classes = defaultdict(list)
        for original_class, mapped_class in self.class_mapping.items():
            self.hierarchical_classes[mapped_class].append(original_class)
        
        # Display final mapping with colors and priorities
        print(f"\n📋 Final Class Mapping ({len(self.hierarchical_classes)} main classes):")
        for main_class, sub_classes in self.hierarchical_classes.items():
            mapping_info = config_mapping.get(main_class, {})
            color = mapping_info.get('color', '#808080')
            priority = mapping_info.get('priority', 4)
            description = mapping_info.get('description', 'No description')
            
            print(f"\n🏷️  {main_class} (Priority: {priority}) {color}")
            print(f"    📝 {description}")
            print(f"    📊 {len(sub_classes)} sub-classes:")
            
            for sub_class in sub_classes[:8]:  # Show first 8
                print(f"      ↳ {sub_class}")
            if len(sub_classes) > 8:
                print(f"      ↳ ... and {len(sub_classes) - 8} more")
        
        # Save mapping to file
        self._save_class_mapping()
        
        return len(self.hierarchical_classes)
    
    def merge_datasets(self, target_count_per_class=None):
        """Merge datasets with advanced balancing using YAML configuration"""
        if target_count_per_class is None:
            settings = self.get_global_settings()
            target_count_per_class = settings.get('default_target_count_per_class', 2000)
        
        print(f"\n===== Merging Datasets =====")
        print(f"🎯 Target count per class: {target_count_per_class}")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.output_dir}/labels/val", exist_ok=True)
        
        # Initialize statistics
        merged_class_counts = defaultdict(int)
        file_counter = 0
        augmentation_stats = defaultdict(int)
        
        # Get augmentation settings
        settings = self.get_global_settings()
        augmentation_level = settings.get('default_augmentation_level', 'medium')
        
        # Initialize augmentation pipeline if available
        if AUGMENTATION_AVAILABLE:
            from augmentation_utils import YOLOAugmentationPipeline
            self.augmentation_pipeline = YOLOAugmentationPipeline(
                image_size=settings.get('default_image_size', 640),
                severity_level=augmentation_level
            )
        
        # Process each hierarchical class
        for main_class, sub_classes in self.hierarchical_classes.items():
            print(f"\n🔄 Processing class: {main_class}")
            
            # Collect all samples for this main class
            class_samples = []
            for dataset in self.datasets:
                for sub_class in sub_classes:
                    if sub_class in dataset.get('classes', []):
                        samples = self._collect_class_samples(dataset, sub_class)
                        class_samples.extend(samples)
            
            print(f"📊 Found {len(class_samples)} original samples for {main_class}")
            
            if not class_samples:
                print(f"⚠️  No samples found for {main_class}")
                continue
            
            # Balance and augment class samples
            final_samples, aug_count = self._balance_class_samples(
                class_samples, main_class, target_count_per_class
            )
            
            # Copy samples to merged dataset
            copied_count = self._copy_samples_to_merged(
                final_samples, main_class, file_counter
            )
            
            merged_class_counts[main_class] = copied_count
            augmentation_stats[main_class] = aug_count
            file_counter += copied_count
            
            print(f"✅ {main_class}: {copied_count} total samples ({aug_count} augmented)")
        
        # Create dataset YAML file
        self._create_merged_yaml(merged_class_counts)
        
        # Generate comprehensive analysis report
        self._generate_analysis_report(merged_class_counts, augmentation_stats)
        
        print(f"\n✅ Dataset merging completed!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Total samples: {sum(merged_class_counts.values())}")
        print(f"🔄 Total augmented: {sum(augmentation_stats.values())}")
        
        return merged_class_counts
    
    def _collect_class_samples(self, dataset, class_name):
        """Collect all samples for a specific class from a dataset"""
        samples = []
        
        # Get class index
        if class_name not in dataset.get('classes', []):
            return samples
        
        class_idx = dataset['classes'].index(class_name)
        
        # Paths
        images_dir = os.path.join(dataset['local_path'], 'images', 'train')
        labels_dir = os.path.join(dataset['local_path'], 'labels', 'train')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            return samples
        
        # Find all label files containing this class
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, label_file)
            image_name = os.path.splitext(label_file)[0]
            
            # Find corresponding image
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_path = os.path.join(images_dir, image_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if not image_path:
                continue
            
            # Check if this label file contains the target class
            contains_class = False
            relevant_annotations = []
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:
                            try:
                                file_class_idx = int(parts[0])
                                if file_class_idx == class_idx:
                                    contains_class = True
                                    relevant_annotations.append(line.strip())
                            except (ValueError, IndexError):
                                pass
                
                if contains_class:
                    samples.append({
                        'image_path': image_path,
                        'label_path': label_path,
                        'annotations': relevant_annotations,
                        'dataset_name': dataset['name'],
                        'original_class': class_name
                    })
                    
            except Exception as e:
                print(f"⚠️  Error processing {label_file}: {e}")
        
        return samples
    
    def _balance_class_samples(self, class_samples, main_class, target_count):
        """Balance class samples with intelligent augmentation"""
        original_count = len(class_samples)
        
        if original_count >= target_count:
            # If we have enough samples, randomly select target_count
            selected_samples = random.sample(class_samples, target_count)
            return selected_samples, 0
        
        # We need augmentation
        needed_count = target_count - original_count
        print(f"  📈 Need {needed_count} additional samples for {main_class}")
        
        # Start with original samples
        final_samples = class_samples.copy()
        augmented_count = 0
        
        if AUGMENTATION_AVAILABLE and self.augmentation_pipeline:
            # Apply augmentation to generate needed samples
            augmented_count = self._generate_augmented_samples(
                class_samples, main_class, needed_count, final_samples
            )
        else:
            # Simple duplication if augmentation not available
            print("  ⚠️  Augmentation not available, using duplication")
            while len(final_samples) < target_count:
                sample_to_duplicate = random.choice(class_samples)
                final_samples.append(sample_to_duplicate)
                augmented_count += 1
        
        return final_samples, augmented_count
    
    def _generate_augmented_samples(self, original_samples, main_class, needed_count, final_samples):
        """Generate augmented samples using the augmentation pipeline"""
        augmented_count = 0
        
        # Get class-specific augmentation profile from config
        augmentation_profiles = self.config.get('augmentation_profiles', {})
        
        # Determine appropriate profile based on main class
        profile_name = 'medium'  # default
        if 'disease' in main_class.lower():
            profile_name = 'disease_detection'
        elif 'pest' in main_class.lower():
            profile_name = 'pest_detection'
        elif 'fruit' in main_class.lower():
            profile_name = 'fruit_quality'
        elif 'nutrient' in main_class.lower():
            profile_name = 'nutrient_analysis'
        
        profile = augmentation_profiles.get(profile_name, {})
        severity = profile.get('severity', 'medium')
        techniques = profile.get('techniques', ['brightness', 'contrast', 'flip'])
        
        print(f"  🎨 Using augmentation profile: {profile_name} (severity: {severity})")
        
        # Update augmentation pipeline with class-specific settings
        if hasattr(self.augmentation_pipeline, 'severity_level'):
            self.augmentation_pipeline.severity_level = severity
        
        for i in range(needed_count):
            # Cycle through original samples
            source_sample = original_samples[i % len(original_samples)]
            
            try:
                # Load and augment image
                image = cv2.imread(source_sample['image_path'])
                if image is None:
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Parse bounding boxes for this class
                bboxes = []
                class_labels = []
                
                for annotation in source_sample['annotations']:
                    parts = annotation.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
                            
                        except (ValueError, IndexError):
                            pass
                
                if not bboxes:
                    continue
                
                # Apply augmentation
                aug_image, aug_bboxes, aug_labels = self.augmentation_pipeline.apply_augmentation(
                    image_rgb, bboxes, class_labels, augmentation_type='mixed'
                )
                
                # Create augmented sample
                augmented_sample = {
                    'image_data': aug_image,
                    'bboxes': aug_bboxes,
                    'class_labels': aug_labels,
                    'is_augmented': True,
                    'source_dataset': source_sample['dataset_name'],
                    'original_class': source_sample['original_class']
                }
                
                final_samples.append(augmented_sample)
                augmented_count += 1
                
            except Exception as e:
                print(f"    ⚠️  Augmentation error: {e}")
                continue
        
        return augmented_count
    
    def _copy_samples_to_merged(self, samples, main_class, start_counter):
        """Copy samples to the merged dataset directory"""
        copied_count = 0
        
        for i, sample in enumerate(samples):
            try:
                file_id = start_counter + i
                
                if sample.get('is_augmented', False):
                    # Handle augmented sample
                    img_filename = f"{main_class}_aug_{file_id:06d}.jpg"
                    lbl_filename = f"{main_class}_aug_{file_id:06d}.txt"
                    
                    # Save augmented image
                    img_path = os.path.join(self.output_dir, 'images', 'train', img_filename)
                    image_bgr = cv2.cvtColor(sample['image_data'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_path, image_bgr)
                    
                    # Save augmented labels
                    lbl_path = os.path.join(self.output_dir, 'labels', 'train', lbl_filename)
                    with open(lbl_path, 'w') as f:
                        # Map all classes to the main class index
                        main_class_idx = list(self.hierarchical_classes.keys()).index(main_class)
                        for bbox in sample['bboxes']:
                            if len(bbox) >= 4:
                                f.write(f"{main_class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                else:
                    # Handle original sample
                    img_filename = f"{main_class}_orig_{file_id:06d}.jpg"
                    lbl_filename = f"{main_class}_orig_{file_id:06d}.txt"
                    
                    # Copy original image
                    src_img_path = sample['image_path']
                    dst_img_path = os.path.join(self.output_dir, 'images', 'train', img_filename)
                    shutil.copy2(src_img_path, dst_img_path)
                    
                    # Process and copy labels
                    dst_lbl_path = os.path.join(self.output_dir, 'labels', 'train', lbl_filename)
                    self._process_and_copy_labels(sample, dst_lbl_path, main_class)
                
                copied_count += 1
                
            except Exception as e:
                print(f"    ⚠️  Error copying sample: {e}")
                continue
        
        return copied_count
    
    def _process_and_copy_labels(self, sample, dst_label_path, main_class):
        """Process and copy label file with class remapping"""
        main_class_idx = list(self.hierarchical_classes.keys()).index(main_class)
        
        try:
            with open(sample['label_path'], 'r') as src_f, open(dst_label_path, 'w') as dst_f:
                for line in src_f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:
                        try:
                            original_class_idx = int(parts[0])
                            # Map to main class
                            dst_f.write(f"{main_class_idx} {' '.join(parts[1:])}\n")
                        except (ValueError, IndexError):
                            pass
        except Exception as e:
            print(f"    ⚠️  Error processing labels: {e}")
    
    def _create_merged_yaml(self, merged_class_counts):
        """Create YAML configuration file for the merged dataset"""
        class_names = list(self.hierarchical_classes.keys())
        
        # Create training dataset configuration
        dataset_config = {
            'path': os.path.abspath(self.output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': class_names
        }
        
        # Add metadata from original config
        metadata = self.config.get('metadata', {})
        dataset_config['metadata'] = {
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'framework_version': metadata.get('framework_version', '2.0'),
            'source_datasets': len(self.datasets),
            'total_samples': sum(merged_class_counts.values()),
            'class_mapping': dict(self.class_mapping),
            'hierarchical_classes': dict(self.hierarchical_classes)
        }
        
        # Save merged dataset YAML
        yaml_path = 'merged_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"📄 Created dataset YAML: {yaml_path}")
        
        # Also save to output directory
        output_yaml_path = os.path.join(self.output_dir, 'data.yaml')
        with open(output_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"📄 Created dataset YAML in output: {output_yaml_path}")
    
    def _save_class_mapping(self):
        """Save class mapping to JSON file"""
        mapping_data = {
            'class_mapping': self.class_mapping,
            'hierarchical_classes': dict(self.hierarchical_classes),
            'config_source': self.config_file,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_original_classes': len(self.class_mapping),
            'total_main_classes': len(self.hierarchical_classes)
        }
        
        with open('class_mapping.json', 'w') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved class mapping to: class_mapping.json")
    
    def _save_download_report(self, download_stats, failed_downloads):
        """Save detailed download report"""
        report = {
            'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': self.config_file,
            'total_datasets': len(self.datasets),
            'successful_downloads': len(download_stats),
            'failed_downloads': len(failed_downloads),
            'failed_list': failed_downloads,
            'dataset_statistics': download_stats,
            'datasets_info': [
                {
                    'name': ds['name'],
                    'url': ds['url'],
                    'description': ds['description'],
                    'expected_classes': ds['expected_classes'],
                    'category': ds['category']
                } for ds in self.datasets
            ]
        }
        
        with open('download_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Saved download report to: download_report.json")
    
    def _generate_analysis_report(self, merged_class_counts, augmentation_stats):
        """Generate comprehensive analysis report"""
        print("\n===== Generating Analysis Report =====")
        
        # Calculate statistics
        total_samples = sum(merged_class_counts.values())
        total_augmented = sum(augmentation_stats.values())
        original_samples = total_samples - total_augmented
        
        # Create comprehensive report
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': self.config_file,
            'output_directory': self.output_dir,
            
            # Dataset summary
            'dataset_summary': {
                'source_datasets': len(self.datasets),
                'total_samples': total_samples,
                'original_samples': original_samples,
                'augmented_samples': total_augmented,
                'augmentation_ratio': total_augmented / total_samples if total_samples > 0 else 0,
                'main_classes': len(self.hierarchical_classes),
                'original_classes': len(self.class_mapping)
            },
            
            # Class distribution
            'class_distribution': {
                'merged_counts': dict(merged_class_counts),
                'augmentation_counts': dict(augmentation_stats),
                'class_balance': self._calculate_class_balance(merged_class_counts)
            },
            
            # Source datasets info
            'source_datasets': [
                {
                    'name': ds['name'],
                    'category': ds['category'],
                    'priority': ds['priority'],
                    'classes': ds.get('classes', []),
                    'class_counts': ds.get('class_counts', {}),
                    'total_images': ds.get('total_images', 0),
                    'total_annotations': ds.get('total_annotations', 0)
                } for ds in self.datasets
            ],
            
            # Class mapping details
            'class_mapping': {
                'mapping_rules': self.class_mapping,
                'hierarchical_structure': dict(self.hierarchical_classes),
                'mapping_source': 'YAML configuration'
            },
            
            # Recommendations
            'recommendations': self._generate_recommendations(merged_class_counts, total_samples)
        }
        
        # Save report
        with open('analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Saved analysis report to: analysis_report.json")
        
        # Generate visual plots if matplotlib available
        try:
            self._generate_visual_reports(merged_class_counts, augmentation_stats)
        except ImportError:
            print("⚠️  Matplotlib not available, skipping visual reports")
        
        # Print summary
        self._print_report_summary(report)
    
    def _calculate_class_balance(self, class_counts):
        """Calculate class balance metrics"""
        if not class_counts:
            return {}
        
        counts = list(class_counts.values())
        total = sum(counts)
        
        return {
            'min_samples': min(counts),
            'max_samples': max(counts),
            'mean_samples': np.mean(counts),
            'std_samples': np.std(counts),
            'balance_ratio': min(counts) / max(counts) if max(counts) > 0 else 0,
            'percentages': {class_name: (count / total) * 100 
                          for class_name, count in class_counts.items()}
        }
    
    def _generate_recommendations(self, class_counts, total_samples):
        """Generate training recommendations based on dataset analysis"""
        recommendations = []
        
        # Sample size recommendations
        if total_samples < 1000:
            recommendations.append("Dataset is small (<1000 samples). Consider increasing augmentation or adding more data.")
        elif total_samples > 50000:
            recommendations.append("Large dataset detected. Consider reducing batch size or using gradient accumulation.")
        
        # Class balance recommendations
        balance_info = self._calculate_class_balance(class_counts)
        if balance_info.get('balance_ratio', 0) < 0.3:
            recommendations.append("Significant class imbalance detected. Consider weighted loss or additional augmentation for minority classes.")
        
        # Model recommendations based on dataset size and complexity
        if total_samples < 5000:
            recommendations.append("Recommended model: yolo11s.pt or yolo11m.pt for smaller datasets")
        elif total_samples > 20000:
            recommendations.append("Recommended model: yolo11l.pt or yolo11x.pt for larger datasets")
        
        # Training parameter recommendations
        num_classes = len(class_counts)
        if num_classes > 20:
            recommendations.append("Many classes detected. Consider longer training (500+ epochs) and careful validation monitoring.")
        
        return recommendations
    
    def _generate_visual_reports(self, merged_class_counts, augmentation_stats):
        """Generate visual analysis plots"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution pie chart
        if merged_class_counts:
            axes[0, 0].pie(merged_class_counts.values(), labels=merged_class_counts.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Class Distribution in Merged Dataset')
        
        # Class counts bar chart
        if merged_class_counts:
            class_names = list(merged_class_counts.keys())
            class_counts = list(merged_class_counts.values())
            axes[0, 1].bar(class_names, class_counts)
            axes[0, 1].set_title('Samples per Class')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Original vs Augmented comparison
        if augmentation_stats:
            original_counts = [merged_class_counts[cls] - augmentation_stats.get(cls, 0) 
                             for cls in merged_class_counts.keys()]
            aug_counts = [augmentation_stats.get(cls, 0) for cls in merged_class_counts.keys()]
            
            x = np.arange(len(class_names))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, original_counts, width, label='Original', alpha=0.8)
            axes[1, 0].bar(x + width/2, aug_counts, width, label='Augmented', alpha=0.8)
            axes[1, 0].set_xlabel('Classes')
            axes[1, 0].set_ylabel('Sample Count')
            axes[1, 0].set_title('Original vs Augmented Samples')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(class_names, rotation=45)
            axes[1, 0].legend()
        
        # Dataset source distribution
        source_counts = defaultdict(int)
        for dataset in self.datasets:
            source_counts[dataset['category']] += dataset.get('total_images', 0)
        
        if source_counts:
            axes[1, 1].pie(source_counts.values(), labels=source_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Distribution by Dataset Category')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved visual analysis to: dataset_analysis.png")
    
    def _print_report_summary(self, report):
        """Print a formatted summary of the analysis report"""
        print("\n" + "="*60)
        print("📊 DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        summary = report['dataset_summary']
        print(f"📁 Output Directory: {report['output_directory']}")
        print(f"📦 Source Datasets: {summary['source_datasets']}")
        print(f"📊 Total Samples: {summary['total_samples']:,}")
        print(f"🔄 Augmented Samples: {summary['augmented_samples']:,} ({summary['augmentation_ratio']*100:.1f}%)")
        print(f"🏷️  Main Classes: {summary['main_classes']}")
        print(f"🔀 Original Classes: {summary['original_classes']}")
        
        print(f"\n📋 Class Distribution:")
        for class_name, count in report['class_distribution']['merged_counts'].items():
            percentage = report['class_distribution']['class_balance']['percentages'][class_name]
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print("="*60)
    
    def get_training_recommendations(self, group_name=None):
        """Get training recommendations based on selected dataset group with unified model support"""
        if group_name:
            group_info = self.get_dataset_group_info(group_name)
            if group_info:
                recommendations = {
                    'model': group_info.get('recommended_model', 'yolo11m.pt'),
                    'batch_size': group_info.get('batch_size', 16),
                    'image_size': group_info.get('image_size', 640),
                    'estimated_time': group_info.get('estimated_training_time', 'Unknown'),
                    'group_type': self._determine_group_type(group_name),
                    'target_classes': group_info.get('target_classes', list(self.hierarchical_classes.keys()))
                }
                
                # Add unified model specific recommendations
                if 'unified' in group_name.lower():
                    recommendations.update({
                        'training_strategy': 'unified_model',
                        'class_balancing': 'critical',
                        'augmentation_level': 'heavy',
                        'validation_split': 0.2,
                        'early_stopping_patience': 100,
                        'lr_scheduler': 'cosine',
                        'model_capabilities': group_info.get('model_capabilities', []),
                        'special_notes': [
                            "This is a unified model for multiple agricultural tasks",
                            "Ensure balanced representation of all task types",
                            "Consider task-specific validation metrics",
                            "May require longer training for optimal convergence"
                        ]
                    })
                
                # Add legacy model warnings
                if group_info.get('note'):
                    recommendations['legacy_warning'] = group_info['note']
                
                return recommendations
        
        # Default recommendations based on data analysis
        total_samples = sum(ds.get('total_images', 0) for ds in self.datasets)
        num_classes = len(self.hierarchical_classes)
        
        # Unified model recommendations based on dataset size
        if total_samples < 10000:
            return {
                'model': 'yolo11m.pt',
                'batch_size': 16,
                'image_size': 640,
                'estimated_time': '2-4 hours',
                'training_strategy': 'unified_model',
                'recommendation': 'Consider adding more data for better unified model performance'
            }
        elif total_samples > 30000:
            return {
                'model': 'yolo11x.pt',
                'batch_size': 4,
                'image_size': 640,
                'estimated_time': '8-12 hours',
                'training_strategy': 'unified_model',
                'recommendation': 'Excellent dataset size for comprehensive unified model'
            }
        else:
            return {
                'model': 'yolo11l.pt',
                'batch_size': 8,
                'image_size': 640,
                'estimated_time': '4-8 hours',
                'training_strategy': 'unified_model',
                'recommendation': 'Good dataset size for unified agricultural model'
            }
    
    def _determine_group_type(self, group_name):
        """Determine the type of dataset group for specialized handling"""
        if 'unified' in group_name.lower():
            return 'unified_model'
        elif 'research' in group_name.lower():
            return 'research'
        elif 'production' in group_name.lower():
            return 'production'
        elif 'quick' in group_name.lower() or 'test' in group_name.lower():
            return 'testing'
        else:
            return 'legacy'
    
    def show_available_groups(self):
        """Display all available dataset groups with enhanced information and unified model focus"""
        print("\n===== Available Dataset Groups =====")
        
        groups = self.config.get('dataset_groups', {})
        
        # Prioritize unified models
        unified_groups = []
        legacy_groups = []
        
        for group_name, group_info in groups.items():
            if 'unified' in group_name.lower() or 'production' in group_name.lower():
                unified_groups.append((group_name, group_info))
            else:
                legacy_groups.append((group_name, group_info))
        
        # Show unified models first
        if unified_groups:
            print("\n🎯 RECOMMENDED: Unified Agricultural Models")
            print("-" * 50)
            for i, (group_name, group_info) in enumerate(unified_groups, 1):
                self._display_group_info(i, group_name, group_info, highlight=True)
        
        # Show legacy/specialized models
        if legacy_groups:
            print(f"\n📋 Legacy/Specialized Models")
            print("-" * 50)
            start_idx = len(unified_groups) + 1
            for i, (group_name, group_info) in enumerate(legacy_groups, start_idx):
                self._display_group_info(i, group_name, group_info, highlight=False)
                
    def _display_group_info(self, index, group_name, group_info, highlight=False):
        """Display formatted group information"""
        description = group_info.get('description', 'No description')
        datasets = group_info.get('datasets', [])
        use_case = group_info.get('use_case', 'General use')
        estimated_time = group_info.get('estimated_training_time', 'Unknown')
        recommended_model = group_info.get('recommended_model', 'Not specified')
        batch_size = group_info.get('batch_size', 'Auto')
        image_size = group_info.get('image_size', 'Auto')
        
        prefix = "🌟" if highlight else "  "
        
        print(f"{prefix}{index:2d}. {group_name}")
        if highlight:
            print(f"    ⭐ {description}")
        else:
            print(f"    📝 {description}")
        
        print(f"    📊 Datasets: {len(datasets)} ({', '.join(datasets[:3])}{'...' if len(datasets) > 3 else ''})")
        print(f"    🎯 Use case: {use_case}")
        print(f"    ⏱️  Training time: {estimated_time}")
        print(f"    🤖 Model: {recommended_model} | Batch: {batch_size} | Size: {image_size}")
        
        # Show capabilities for unified models
        if 'model_capabilities' in group_info:
            capabilities = group_info['model_capabilities'][:3]  # Show first 3
            print(f"    🔧 Capabilities: {', '.join(capabilities)}{'...' if len(group_info['model_capabilities']) > 3 else ''}")
        
        # Show legacy warning
        if 'note' in group_info:
            print(f"    ⚠️  {group_info['note']}")
        
        print()
    
    def create_unified_class_mapping(self):
        """Create unified class mapping optimized for multi-task agricultural model"""
        print("\n===== Creating Unified Agricultural Class Mapping =====")
        
        # Collect all unique classes from datasets
        all_classes = set()
        for dataset in self.datasets:
            all_classes.update(dataset.get('classes', []))
        
        print(f"📊 Total unique classes found: {len(all_classes)}")
        
        # Get class mapping configuration from YAML
        config_mapping = self.config.get('class_mapping', {})
        global_settings = self.get_global_settings()
        auto_mapping = global_settings.get('auto_class_mapping', True)
        
        # Initialize mapping
        self.class_mapping = {}
        mapped_classes = set()
        
        # Apply mapping using YAML configuration with unified model priorities
        if auto_mapping and config_mapping:
            print("🔄 Applying unified agricultural class mapping...")
            
            # Process classes by priority for unified model
            priority_order = sorted(config_mapping.items(), 
                                  key=lambda x: x[1].get('priority', 4))
            
            for main_class, mapping_info in priority_order:
                sub_classes = mapping_info.get('sub_classes', [])
                keywords = mapping_info.get('keywords', [])
                priority = mapping_info.get('priority', 4)
                
                print(f"\n🏷️  Processing: {main_class} (Priority: {priority})")
                
                # Direct mapping for sub_classes
                for sub_class in sub_classes:
                    if sub_class in all_classes:
                        self.class_mapping[sub_class] = main_class
                        mapped_classes.add(sub_class)
                        print(f"  ✅ Direct: '{sub_class}' → '{main_class}'")
                
                # Keyword-based mapping for unmapped classes
                for class_name in all_classes:
                    if class_name not in mapped_classes:
                        class_lower = class_name.lower()
                        for keyword in keywords:
                            if keyword.lower() in class_lower:
                                self.class_mapping[class_name] = main_class
                                mapped_classes.add(class_name)
                                print(f"  🔍 Keyword: '{class_name}' → '{main_class}' (keyword: {keyword})")
                                break
        
        # Handle unmapped classes for unified model
        unmapped_classes = all_classes - mapped_classes
        if unmapped_classes:
            print(f"\n⚠️  Unmapped classes found: {len(unmapped_classes)}")
            for class_name in unmapped_classes:
                print(f"  - {class_name}")
            
            # For unified model, try to intelligently assign unmapped classes
            print("🤖 Applying intelligent unified model mapping...")
            for class_name in unmapped_classes:
                assigned_class = self._intelligent_class_assignment(class_name, config_mapping)
                self.class_mapping[class_name] = assigned_class
                print(f"  🧠 Intelligent: '{class_name}' → '{assigned_class}'")
        
        # Create hierarchical class structure
        self.hierarchical_classes = defaultdict(list)
        for original_class, mapped_class in self.class_mapping.items():
            self.hierarchical_classes[mapped_class].append(original_class)
        
        # Display final unified mapping
        print(f"\n📋 Unified Agricultural Model Classes ({len(self.hierarchical_classes)} main classes):")
        
        # Sort by priority for display
        sorted_classes = sorted(
            self.hierarchical_classes.items(),
            key=lambda x: config_mapping.get(x[0], {}).get('priority', 4)
        )
        
        for main_class, sub_classes in sorted_classes:
            mapping_info = config_mapping.get(main_class, {})
            color = mapping_info.get('color', '#808080')
            priority = mapping_info.get('priority', 4)
            description = mapping_info.get('description', 'No description')
            
            print(f"\n🏷️  {main_class} (Priority: {priority}) {color}")
            print(f"    📝 {description}")
            print(f"    📊 {len(sub_classes)} sub-classes:")
            
            for sub_class in sub_classes[:8]:  # Show first 8
                print(f"      ↳ {sub_class}")
            if len(sub_classes) > 8:
                print(f"      ↳ ... and {len(sub_classes) - 8} more")
        
        # Save mapping with unified model metadata
        self._save_unified_class_mapping()
        
        print(f"\n✅ Unified agricultural model mapping completed!")
        print(f"🎯 Ready for multi-task agricultural AI training")
        
        return len(self.hierarchical_classes)
    
    def _intelligent_class_assignment(self, class_name, config_mapping):
        """Intelligently assign unmapped classes for unified model"""
        class_lower = class_name.lower()
        
        # Agricultural intelligence rules for unified model
        if any(term in class_lower for term in ['leaf', 'plant', 'crop']) and 'spot' not in class_lower:
            return 'healthy'  # Default plant parts to healthy
        elif any(term in class_lower for term in ['class', 'category', 'type']) and any(char.isdigit() for char in class_name):
            return 'unknown'  # Generic numbered classes
        elif 'weed' in class_lower:
            return 'weeds'
        elif any(term in class_lower for term in ['bug', 'insect', 'fly', 'mite']):
            return 'pest_damage'
        elif any(term in class_lower for term in ['disease', 'sick', 'infection']):
            return 'fungal_disease'  # Default disease category
        elif any(term in class_lower for term in ['deficiency', 'lack', 'poor']):
            return 'nutrient_deficiency'
        else:
            return 'unknown'
    
    def _save_unified_class_mapping(self):
        """Save unified model class mapping with additional metadata"""
        mapping_data = {
            'model_type': 'unified_agricultural',
            'class_mapping': self.class_mapping,
            'hierarchical_classes': dict(self.hierarchical_classes),
            'config_source': self.config_file,
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_original_classes': len(self.class_mapping),
            'total_main_classes': len(self.hierarchical_classes),
            'unified_model_capabilities': [
                "Plant disease detection and classification",
                "Pest and insect identification", 
                "Fruit ripeness assessment",
                "Nutrient deficiency analysis",
                "Weed detection",
                "General plant health monitoring"
            ],
            'class_priorities': {
                main_class: self.config.get('class_mapping', {}).get(main_class, {}).get('priority', 4)
                for main_class in self.hierarchical_classes.keys()
            },
            'training_recommendations': {
                'model_size': 'yolo11l.pt or larger recommended',
                'batch_size': '8 or smaller for complex unified model',
                'training_time': 'Extended training recommended for convergence',
                'validation': 'Multi-task validation metrics required'
            }
        }
        
        with open('unified_class_mapping.json', 'w') as f:
            json.dump(mapping_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved unified model mapping to: unified_class_mapping.json")
    
    def save_config_updates(self, new_datasets=None):
        """Save any new datasets or updates back to the config file"""
        if new_datasets:
            # Add new datasets to config
            for dataset_info in new_datasets:
                category = dataset_info.get('category', 'custom_datasets')
                
                # Ensure category exists
                if 'datasets' not in self.config:
                    self.config['datasets'] = {}
                if category not in self.config['datasets']:
                    self.config['datasets'][category] = {}
                
                # Add dataset
                self.config['datasets'][category][dataset_info['name']] = {
                    'url': dataset_info['url'],
                    'description': dataset_info.get('description', ''),
                    'expected_classes': dataset_info.get('expected_classes', []),
                    'class_count': len(dataset_info.get('expected_classes', [])),
                    'priority': dataset_info.get('priority', 3),
                    'category': category.replace('_datasets', ''),
                    'dataset_type': 'user_added',
                    'added_date': datetime.now().strftime('%Y-%m-%d'),
                    'resize_info': dataset_info.get('resize_info', 'Not specified'),
                    'notes': f"Added via manager on {datetime.now().strftime('%Y-%m-%d')}"
                }
            
            # Update metadata
            metadata = self.config.get('metadata', {})
            metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            metadata['total_datasets'] = len(self._collect_all_datasets())
            self.config['metadata'] = metadata
            
            # Save updated config
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                print(f"💾 Updated config file: {self.config_file}")
                return True
            except Exception as e:
                print(f"❌ Error saving config: {e}")
                return False
        
        return False

# Convenience functions for backward compatibility
def MultiDatasetManager(*args, **kwargs):
    """Backward compatibility wrapper"""
    return YAMLBasedMultiDatasetManager(*args, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Initialize with YAML config
    manager = YAMLBasedMultiDatasetManager("datasets/agricultural_merged", "config_datasets.yaml")
    
    # Show available options
    print("🎯 Available Dataset Groups:")
    manager.show_available_groups()
    
    # Interactive selection
    selected_group = manager.interactive_dataset_selection()
    
    if selected_group:
        # Get global settings
        settings = manager.get_global_settings()
        target_count = settings.get('default_target_count_per_class', 2000)
        
        print(f"\n1. Downloading datasets...")
        if manager.download_all_datasets():
            
            print(f"\n2. Creating unified class mapping...")
            classes_created = manager.create_unified_class_mapping()
            
            if classes_created > 0:
                print(f"\n3. Merging datasets...")
                merged_counts = manager.merge_datasets(target_count_per_class=target_count)
                
                print(f"\n✅ Process completed successfully!")
                print(f"📁 Merged dataset: {manager.output_dir}")
                print(f"📄 YAML file: merged_dataset.yaml")
                
                # Get training recommendations
                recommendations = manager.get_training_recommendations(selected_group)
                print(f"\n🎯 Training Recommendations:")
                for key, value in recommendations.items():
                    print(f"  {key}: {value}")
            else:
                print("❌ No classes could be mapped. Check your configuration.")
        else:
            print("❌ Dataset download failed. Check your URLs and internet connection.")
    else:
        print("❌ No dataset group selected. Exiting...")