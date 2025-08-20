#!/usr/bin/env python3
# multi_dataset_manager.py - Ana YAML-Based Multi-dataset manager

import os
import yaml
import shutil
from datetime import datetime
import json
import random
import logging
from collections import defaultdict

# Import helper modules
from dataset_utils import download_dataset, fix_directory_structure
from multi_dataset_helpers import (
    DatasetAnalyzer, 
    ClassMapper, 
    DatasetMerger,
    ValidationSplitter,
    ReportGenerator
)

# Augmentation import with fallback
try:
    from augmentation_utils import YOLOAugmentationPipeline, SmartAugmentationRecommender
    AUGMENTATION_AVAILABLE = True
except ImportError:
    print("⚠️  Augmentation utils not available. Basic functionality will be used.")
    AUGMENTATION_AVAILABLE = False

class YAMLBasedMultiDatasetManager:
    """
    Ana YAML-based multi-dataset manager for YOLO11 training
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
        
        # Initialize helper components
        self.analyzer = DatasetAnalyzer(self)
        self.class_mapper = ClassMapper(self)
        self.merger = DatasetMerger(self)
        self.validator = ValidationSplitter(self)
        self.reporter = ReportGenerator(self)
        
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
            return self._create_fallback_config()
        except yaml.YAMLError as e:
            print(f"❌ YAML parsing error: {e}")
            return self._create_fallback_config()
    
    def _create_fallback_config(self):
        """Create fallback configuration if config file is not available"""
        return {
            'datasets': {
                'base_datasets': {
                    'plant_diseases_comprehensive': {
                        'url': "https://universe.roboflow.com/ds/0UULi7Pnno?key=PU2zi8AslM",
                        'description': "Plant diseases dataset",
                        'expected_classes': ["damaged", "healthy", "ripe", "unripe"],
                        'priority': 1,
                        'category': 'diseases'
                    }
                }
            },
            'dataset_groups': {
                'quick_test': {
                    'description': "Quick test group",
                    'datasets': ['plant_diseases_comprehensive']
                }
            },
            'global_settings': {
                'default_target_count_per_class': 2000,
                'default_image_size': 640,
                'auto_class_mapping': True
            }
        }
    
    def get_available_dataset_groups(self):
        """Get list of available dataset groups from config"""
        groups = self.config.get('dataset_groups', {})
        return list(groups.keys())
    
    def get_dataset_group_info(self, group_name):
        """Get information about a specific dataset group"""
        groups = self.config.get('dataset_groups', {})
        return groups.get(group_name, {})
    
    def get_global_settings(self):
        """Get global settings from config"""
        return self.config.get('global_settings', {
            'default_target_count_per_class': 2000,
            'default_image_size': 640,
            'auto_class_mapping': True,
            'default_augmentation_level': 'medium',
            'preserve_original_data': True
        })
    
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
        
        # Clear existing datasets
        self.datasets = []
        
        # Load datasets from all categories
        all_datasets = self._collect_all_datasets()
        
        # Add selected datasets
        added_count = 0
        for dataset_name in dataset_names:
            if dataset_name in all_datasets:
                try:
                    dataset_info = all_datasets[dataset_name]
                    self.add_dataset_from_config(dataset_name, dataset_info)
                    added_count += 1
                except Exception as e:
                    print(f"⚠️  Error adding dataset '{dataset_name}': {e}")
            else:
                print(f"⚠️  Dataset '{dataset_name}' not found in configuration")
        
        print(f"✅ Successfully loaded {added_count}/{len(dataset_names)} datasets")
        return added_count > 0
    
    def _collect_all_datasets(self):
        """Collect all datasets from all categories"""
        all_datasets = {}
        datasets_section = self.config.get('datasets', {})
        
        for category_name, category_datasets in datasets_section.items():
            if isinstance(category_datasets, dict) and category_name not in [
                'default_target_count_per_class', 'default_image_size', 'auto_class_mapping'
            ]:
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
    
    def show_available_groups(self):
        """Display all available dataset groups"""
        print("\n===== Available Dataset Groups =====")
        
        groups = self.config.get('dataset_groups', {})
        
        for i, (group_name, group_info) in enumerate(groups.items(), 1):
            description = group_info.get('description', 'No description')
            datasets = group_info.get('datasets', [])
            use_case = group_info.get('use_case', 'General use')
            estimated_time = group_info.get('estimated_training_time', 'Unknown')
            recommended_model = group_info.get('recommended_model', 'Not specified')
            
            print(f"{i:2d}. {group_name}")
            print(f"    📝 {description}")
            print(f"    📊 Datasets: {len(datasets)}")
            print(f"    🎯 Use case: {use_case}")
            print(f"    ⏱️  Training time: {estimated_time}")
            print(f"    🤖 Recommended model: {recommended_model}")
            print()
    
    def interactive_dataset_selection(self):
        """Interactive dataset selection with config-based options"""
        print("\n🎯 Dataset Selection")
        print("=" * 50)
        
        self.show_available_groups()
        
        groups = self.config.get('dataset_groups', {})
        group_list = list(groups.keys())
        
        print(f"\n{len(group_list) + 1}. Custom selection")
        print(f"{len(group_list) + 2}. Add custom URL")
        
        # Get user choice
        while True:
            try:
                choice = input(f"\nSelect option (1-{len(group_list) + 2}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(group_list):
                    selected_group = group_list[choice_num - 1]
                    if self.load_dataset_group(selected_group):
                        return selected_group
                    else:
                        return None
                        
                elif choice_num == len(group_list) + 1:
                    print("Custom selection not implemented yet")
                    return None
                    
                elif choice_num == len(group_list) + 2:
                    print("Custom URL not implemented yet")
                    return None
                    
                else:
                    print(f"❌ Invalid choice. Enter 1-{len(group_list) + 2}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled")
                return None
    
    # Delegate main functions to helper classes
    def download_all_datasets(self):
        """Download all datasets - delegates to analyzer"""
        return self.analyzer.download_all_datasets()
    
    def create_unified_class_mapping(self):
        """Create unified class mapping - delegates to class_mapper"""
        return self.class_mapper.create_unified_class_mapping()
    
    def merge_datasets(self, target_count_per_class=None):
        """Merge datasets - delegates to merger"""
        result = self.merger.merge_datasets(target_count_per_class)
        
        # *** YENİ: Validation split kontrolü ***
        self.validator.create_validation_split_if_missing()
        
        return result
    
    def get_training_recommendations(self, group_name=None):
        """Get training recommendations"""
        if group_name:
            group_info = self.get_dataset_group_info(group_name)
            if group_info:
                return {
                    'model': group_info.get('recommended_model', 'yolo11l.pt'),
                    'batch_size': group_info.get('batch_size', 8),
                    'image_size': group_info.get('image_size', 640),
                    'estimated_time': group_info.get('estimated_training_time', 'Unknown'),
                    'target_classes': group_info.get('target_classes', []),
                    'special_notes': [
                        "Use validation split for training",
                        "Monitor training progress carefully",
                        "Consider early stopping if overfitting"
                    ]
                }
        
        # Default recommendations
        return {
            'model': 'yolo11l.pt',
            'batch_size': 8,
            'image_size': 640,
            'estimated_time': '4-8 hours',
            'recommendation': 'Good for hierarchical agricultural model'
        }

# Backward compatibility
def MultiDatasetManager(*args, **kwargs):
    """Backward compatibility wrapper"""
    return YAMLBasedMultiDatasetManager(*args, **kwargs)

# Quick test
if __name__ == "__main__":
    print("🎯 Testing Multi-Dataset Manager")
    manager = YAMLBasedMultiDatasetManager()
    print("✅ Manager initialized successfully")
    
    # Show available groups
    manager.show_available_groups()
