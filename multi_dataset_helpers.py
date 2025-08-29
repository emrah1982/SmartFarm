#!/usr/bin/env python3
# multi_dataset_helpers.py - Helper components for multi-dataset manager

import os
import yaml
import shutil
import cv2
import numpy as np
from collections import defaultdict, Counter
import json
import random
from datetime import datetime
from augmentation_utils import YOLOAugmentationPipeline
try:
    from roboflow_api_helper import get_api_key_from_config, download_from_config_entry
except Exception:
    def get_api_key_from_config():
        return None
    def download_from_config_entry(*args, **kwargs):
        return False

class DatasetAnalyzer:
    """Dataset analysis and download operations"""
    
    def __init__(self, manager):
        self.manager = manager
    
    def download_all_datasets(self):
        """Download all datasets with enhanced error handling"""
        print("\n===== Downloading Datasets =====")
        
        successful_downloads = 0
        failed_downloads = []
        download_stats = {}
        
        # SDK-only akışı: Roboflow SDK kullanacağız
        # Try to get a Roboflow API key (required for SDK flow).
        api_key = None
        split_config = None
        try:
            api_key = get_api_key_from_config()
            if api_key:
                print(f"🔑 Roboflow API key bulundu (ilk 10): {api_key[:10]}...")
        except Exception:
            api_key = None
        # Manager üzerinde interaktif girilmiş değer varsa önceliklidir
        if hasattr(self.manager, 'api_key') and self.manager.api_key:
            api_key = self.manager.api_key
        if hasattr(self.manager, 'split_config') and self.manager.split_config:
            split_config = self.manager.split_config
        
        for i, dataset in enumerate(self.manager.datasets):
            print(f"\n[{i+1}/{len(self.manager.datasets)}] Downloading: {dataset['name']}")
            print(f"📝 {dataset.get('description', 'No description')}")
            
            # 0) Skip if disabled
            if dataset.get('enabled') is False:
                print(f"🚫 Atlandı (disabled): {dataset['name']}")
                continue

            # 1) Enforce SDK-only: require api_key + roboflow_canonical
            canonical = dataset.get('roboflow_canonical')
            if not api_key or not canonical or canonical.count('/') != 2:
                print("⚠️  SDK-only mod etkin: Bu veri seti api_key + roboflow_canonical sağlamadığı için atlandı.")
                print(f"   • api_key: {'var' if api_key else 'yok'} | canonical: {canonical if canonical else 'yok'}")
                failed_downloads.append(dataset['name'])
                continue

            # 2) Roboflow SDK ile indirme (yerelde çalışan akışla aynı)
            fmt = dataset.get('format') or 'yolov11'
            print(f"🤖 SDK Strategy: download_from_config_entry (format={fmt})")
            success = bool(download_from_config_entry(dataset, dataset_dir=dataset['local_path'], api_key=api_key, format_name=fmt))
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
            
            # Analyze dataset distribution
            stats = self._analyze_dataset_distribution(dataset)
            download_stats[dataset['name']] = stats
        
        # Summary
        print(f"\n📊 Download Summary:")
        print(f"✅ Successful: {successful_downloads}/{len(self.manager.datasets)}")
        if failed_downloads:
            print(f"❌ Failed: {len(failed_downloads)} - {', '.join(failed_downloads)}")
        
        return successful_downloads > 0
    
    def _analyze_dataset_distribution(self, dataset):
        """Analyze the class distribution of a dataset"""
        print(f"\n--- {dataset['name']} Class Analysis ---")
        
        train_labels_dir = os.path.join(dataset['local_path'], 'labels', 'train')
        if not os.path.exists(train_labels_dir):
            print(f"❌ Label directory not found: {train_labels_dir}")
            return {}
        
        class_counts = Counter()
        total_annotations = 0
        total_images = 0
        
        # Check all label files
        for label_file in os.listdir(train_labels_dir):
            if label_file.endswith('.txt'):
                total_images += 1
                file_path = os.path.join(train_labels_dir, label_file)
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:
                                try:
                                    class_idx = int(parts[0])
                                    if 0 <= class_idx < len(dataset['classes']):
                                        class_name = dataset['classes'][class_idx]
                                        class_counts[class_name] += 1
                                        total_annotations += 1
                                except (ValueError, IndexError):
                                    pass
                except Exception as e:
                    print(f"⚠️  Error reading {label_file}: {e}")
        
        # Store statistics
        dataset['class_counts'] = dict(class_counts)
        dataset['total_annotations'] = total_annotations
        dataset['total_images'] = total_images
        
        stats = {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'class_counts': dict(class_counts)
        }
        
        print(f"📊 Images: {total_images}, Annotations: {total_annotations}")
        
        if class_counts:
            print("📋 Class distribution:")
            for class_name, count in class_counts.most_common():
                percentage = (count / total_annotations) * 100 if total_annotations > 0 else 0
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return stats

class ClassMapper:
    """Class mapping and hierarchical structure creation"""
    
    def __init__(self, manager):
        self.manager = manager
    
    def create_unified_class_mapping(self):
        """Create unified class mapping using YAML configuration"""
        print("\n===== Creating Unified Class Mapping =====")
        
        # Collect all unique classes from datasets
        all_classes = set()
        for dataset in self.manager.datasets:
            all_classes.update(dataset.get('classes', []))
        
        print(f"📊 Total unique classes found: {len(all_classes)}")
        
        # Get class mapping configuration from YAML
        config_mapping = self.manager.config.get('class_mapping', {})
        global_settings = self.manager.get_global_settings()
        auto_mapping = global_settings.get('auto_class_mapping', True)
        
        # Initialize mapping
        self.manager.class_mapping = {}
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
                        self.manager.class_mapping[sub_class] = main_class
                        mapped_classes.add(sub_class)
                        print(f"  ✅ Direct: '{sub_class}' → '{main_class}'")
                
                # Keyword-based mapping for unmapped classes
                for class_name in all_classes:
                    if class_name not in mapped_classes:
                        class_lower = class_name.lower()
                        for keyword in keywords:
                            if keyword.lower() in class_lower:
                                self.manager.class_mapping[class_name] = main_class
                                mapped_classes.add(class_name)
                                print(f"  🔍 Keyword: '{class_name}' → '{main_class}' (keyword: {keyword})")
                                break
        
        # Handle unmapped classes - SKIP unknown mapping to prevent issues
        unmapped_classes = all_classes - mapped_classes
        if unmapped_classes:
            print(f"\n⚠️  Unmapped classes found: {len(unmapped_classes)}")
            for class_name in unmapped_classes:
                print(f"  - {class_name}")
            
            # SKIP unknown mapping - these classes will be ignored
            print(f"  ⚠️  These classes will be IGNORED to prevent 'unknown' issues")
            print(f"  💡 Add them to config_datasets.yaml if needed")
        
        # Create hierarchical class structure
        self.manager.hierarchical_classes = defaultdict(list)
        for original_class, mapped_class in self.manager.class_mapping.items():
            self.manager.hierarchical_classes[mapped_class].append(original_class)
        
        # Display final mapping
        print(f"\n📋 Final Class Mapping ({len(self.manager.hierarchical_classes)} main classes):")
        for main_class, sub_classes in self.manager.hierarchical_classes.items():
            mapping_info = config_mapping.get(main_class, {})
            priority = mapping_info.get('priority', 4)
            description = mapping_info.get('description', 'No description')
            
            print(f"\n🏷️  {main_class} (Priority: {priority})")
            print(f"    📝 {description}")
            print(f"    📊 {len(sub_classes)} sub-classes")
        
        return len(self.manager.hierarchical_classes)

class DatasetMerger:
    """Dataset merging and augmentation operations"""
    
    def __init__(self, manager):
        self.manager = manager
        # Initialize a reusable augmentation pipeline
        try:
            self.augmentation_pipeline = YOLOAugmentationPipeline(image_size=640, severity_level='medium')
        except Exception:
            self.augmentation_pipeline = None
    
    def merge_datasets(self, target_count_per_class=None):
        """Merge datasets with balancing"""
        settings = self.manager.get_global_settings()
        global_default_target = settings.get('default_target_count_per_class', 2000)
        # target_count_per_class can be int or dict per main class
        target_spec = target_count_per_class if target_count_per_class is not None else global_default_target
        
        print(f"\n===== Merging Datasets =====")
        print(f"🎯 Target count per class: {target_spec}")
        
        # Create output directories
        os.makedirs(self.manager.output_dir, exist_ok=True)
        os.makedirs(f"{self.manager.output_dir}/images/train", exist_ok=True)
        os.makedirs(f"{self.manager.output_dir}/images/val", exist_ok=True)
        os.makedirs(f"{self.manager.output_dir}/labels/train", exist_ok=True)
        os.makedirs(f"{self.manager.output_dir}/labels/val", exist_ok=True)
        
        # Initialize statistics
        merged_class_counts = defaultdict(int)
        file_counter = 0
        
        # Process each hierarchical class
        for main_class, sub_classes in self.manager.hierarchical_classes.items():
            print(f"\n🔄 Processing class: {main_class}")
            
            # Collect all samples for this main class
            class_samples = []
            for dataset in self.manager.datasets:
                for sub_class in sub_classes:
                    if sub_class in dataset.get('classes', []):
                        samples = self._collect_class_samples(dataset, sub_class)
                        class_samples.extend(samples)
            
            print(f"📊 Found {len(class_samples)} original samples for {main_class}")
            
            if not class_samples:
                print(f"⚠️  No samples found for {main_class}")
                continue
            
            original_count = len(class_samples)
            # Determine target for this main class
            if isinstance(target_spec, dict):
                class_target = target_spec.get(main_class, global_default_target)
            else:
                class_target = int(target_spec)
            copied_count = 0
            
            if original_count >= class_target:
                # If we have enough samples, randomly select target_count_per_class originals
                selected = random.sample(class_samples, class_target)
                copied_count = self._copy_samples_to_merged(selected, main_class, file_counter)
            else:
                # Copy all originals first
                copied_count = self._copy_samples_to_merged(class_samples, main_class, file_counter)
                needed = max(0, class_target - original_count)
                if needed > 0:
                    print(f"✨ Augmenting {needed} additional samples for {main_class} to reach target {class_target}...")
                    aug_generated = self._augment_and_save(class_samples, needed, main_class, file_counter + copied_count)
                    copied_count += aug_generated
            
            merged_class_counts[main_class] = copied_count
            file_counter += copied_count
            
            print(f"✅ {main_class}: {copied_count} total samples")
        
        # Create dataset YAML file
        self._create_merged_yaml(merged_class_counts)
        
        print(f"\n✅ Dataset merging completed!")
        print(f"📁 Output directory: {self.manager.output_dir}")
        print(f"📊 Total samples: {sum(merged_class_counts.values())}")
        
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
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:
                            try:
                                file_class_idx = int(parts[0])
                                if file_class_idx == class_idx:
                                    contains_class = True
                                    break
                            except (ValueError, IndexError):
                                pass
                
                if contains_class:
                    samples.append({
                        'image_path': image_path,
                        'label_path': label_path,
                        'dataset_name': dataset['name'],
                        'original_class': class_name
                    })
                    
            except Exception as e:
                print(f"⚠️  Error processing {label_file}: {e}")
        
        return samples
    
    def _balance_class_samples_simple(self, class_samples, target_count):
        """Simple balancing without heavy augmentation"""
        original_count = len(class_samples)
        
        if original_count >= target_count:
            # If we have enough samples, randomly select target_count
            return random.sample(class_samples, target_count)
        
        # If we need more samples, duplicate existing ones
        final_samples = class_samples.copy()
        while len(final_samples) < target_count:
            sample_to_duplicate = random.choice(class_samples)
            final_samples.append(sample_to_duplicate)
        
        return final_samples[:target_count]
    
    def _copy_samples_to_merged(self, samples, main_class, start_counter):
        """Copy samples to the merged dataset directory"""
        copied_count = 0
        
        for i, sample in enumerate(samples):
            try:
                file_id = start_counter + i
                
                # Create filenames
                img_filename = f"{main_class}_{file_id:06d}.jpg"
                lbl_filename = f"{main_class}_{file_id:06d}.txt"
                
                # Copy image
                src_img_path = sample['image_path']
                dst_img_path = os.path.join(self.manager.output_dir, 'images', 'train', img_filename)
                shutil.copy2(src_img_path, dst_img_path)
                
                # Process and copy labels
                dst_lbl_path = os.path.join(self.manager.output_dir, 'labels', 'train', lbl_filename)
                self._process_and_copy_labels(sample, dst_lbl_path, main_class)
                
                copied_count += 1
                
            except Exception as e:
                print(f"    ⚠️  Error copying sample: {e}")
                continue
        
        return copied_count

    def _augment_and_save(self, class_samples, needed_count, main_class, start_counter):
        """Generate augmented samples for a main class and save directly into merged dataset.
        Augmented labels are remapped to the main class index.
        """
        if self.augmentation_pipeline is None:
            print("⚠️  Augmentation pipeline could not be initialized. Falling back to duplication.")
            # Fallback: duplicate random samples
            duplicated = 0
            for i in range(needed_count):
                sample = random.choice(class_samples)
                # Create filenames
                file_id = start_counter + duplicated
                img_filename = f"{main_class}_{file_id:06d}.jpg"
                lbl_filename = f"{main_class}_{file_id:06d}.txt"
                # Copy image
                dst_img_path = os.path.join(self.manager.output_dir, 'images', 'train', img_filename)
                shutil.copy2(sample['image_path'], dst_img_path)
                # Remap and copy label
                dst_lbl_path = os.path.join(self.manager.output_dir, 'labels', 'train', lbl_filename)
                self._process_and_copy_labels(sample, dst_lbl_path, main_class)
                duplicated += 1
            return duplicated
        
        saved = 0
        main_class_idx = list(self.manager.hierarchical_classes.keys()).index(main_class)
        images_train_dir = os.path.join(self.manager.output_dir, 'images', 'train')
        labels_train_dir = os.path.join(self.manager.output_dir, 'labels', 'train')
        
        for i in range(needed_count):
            sample = random.choice(class_samples)
            try:
                # Load source image
                image_bgr = cv2.imread(sample['image_path'])
                if image_bgr is None:
                    continue
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                # Read YOLO labels
                bboxes = []
                class_labels = []
                with open(sample['label_path'], 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:
                            try:
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                bboxes.append([x_center, y_center, width, height])
                                # Keep original label list length in sync; values will be overridden to main class on save
                                class_labels.append(main_class_idx)
                            except (ValueError, IndexError):
                                pass
                if not bboxes:
                    continue
                
                # Apply augmentation
                aug_image, aug_bboxes, aug_labels = self.augmentation_pipeline.apply_augmentation(
                    image, bboxes, class_labels
                )
                
                # Save augmented image
                aug_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                file_id = start_counter + saved
                img_filename = f"{main_class}_{file_id:06d}.jpg"
                lbl_filename = f"{main_class}_{file_id:06d}.txt"
                img_path = os.path.join(images_train_dir, img_filename)
                lbl_path = os.path.join(labels_train_dir, lbl_filename)
                cv2.imwrite(img_path, aug_bgr)
                
                # Save remapped labels (all boxes mapped to main class index)
                with open(lbl_path, 'w') as out_f:
                    for bbox in aug_bboxes:
                        if len(bbox) >= 4:
                            out_f.write(f"{main_class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                saved += 1
                if saved % 50 == 0:
                    print(f"  Augmented {saved}/{needed_count} for {main_class}")
            except Exception as e:
                print(f"    ⚠️  Augmentation error for {main_class}: {e}")
                continue
        
        return saved
    
    def _process_and_copy_labels(self, sample, dst_label_path, main_class):
        """Process and copy label file with class remapping"""
        main_class_idx = list(self.manager.hierarchical_classes.keys()).index(main_class)
        
        try:
            with open(sample['label_path'], 'r') as src_f, open(dst_label_path, 'w') as dst_f:
                for line in src_f:
                    parts = line.strip().split()
                    if parts and len(parts) >= 5:
                        try:
                            # Map to main class
                            dst_f.write(f"{main_class_idx} {' '.join(parts[1:])}\n")
                        except (ValueError, IndexError):
                            pass
        except Exception as e:
            print(f"    ⚠️  Error processing labels: {e}")
    
    def _create_merged_yaml(self, merged_class_counts):
        """Create YAML configuration file for the merged dataset"""
        class_names = list(self.manager.hierarchical_classes.keys())
        
        # Create training dataset configuration
        dataset_config = {
            'path': os.path.abspath(self.manager.output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(class_names),
            'names': class_names
        }
        
        # Add metadata
        dataset_config['metadata'] = {
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_datasets': len(self.manager.datasets),
            'total_samples': sum(merged_class_counts.values()),
            'class_mapping': dict(self.manager.class_mapping),
            'hierarchical_classes': dict(self.manager.hierarchical_classes)
        }
        
        # Save merged dataset YAML
        yaml_path = 'merged_dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"📄 Created dataset YAML: {yaml_path}")
        
        # Also save to output directory
        output_yaml_path = os.path.join(self.manager.output_dir, 'data.yaml')
        with open(output_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"📄 Created dataset YAML in output: {output_yaml_path}")

class ValidationSplitter:
    """Validation split creation and management"""
    
    def __init__(self, manager):
        self.manager = manager
    
    def create_validation_split_if_missing(self):
        """Validation split eksikse train'den oluştur"""
        print("\n🔧 Checking validation split...")
        
        train_images_dir = os.path.join(self.manager.output_dir, 'images', 'train')
        train_labels_dir = os.path.join(self.manager.output_dir, 'labels', 'train')
        val_images_dir = os.path.join(self.manager.output_dir, 'images', 'val')
        val_labels_dir = os.path.join(self.manager.output_dir, 'labels', 'val')
        
        # Val klasörlerinin var olup olmadığını kontrol et
        if not os.path.exists(val_images_dir) or len(os.listdir(val_images_dir)) == 0:
            print("⚠️  Validation directory is empty or missing. Creating validation split...")
            
            if os.path.exists(train_images_dir) and len(os.listdir(train_images_dir)) > 0:
                # Train'den validation split oluştur
                os.makedirs(val_images_dir, exist_ok=True)
                os.makedirs(val_labels_dir, exist_ok=True)
                
                # Train dosyalarını listele
                train_images = [f for f in os.listdir(train_images_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                # %15'ini val için ayır
                random.shuffle(train_images)
                val_count = max(1, int(len(train_images) * 0.15))  # En az 1 dosya
                val_images = train_images[:val_count]
                
                moved_count = 0
                for img_file in val_images:
                    # Image dosyasını taşı
                    img_src = os.path.join(train_images_dir, img_file)
                    img_dst = os.path.join(val_images_dir, img_file)
                    
                    if os.path.exists(img_src):
                        shutil.move(img_src, img_dst)
                        
                        # Corresponding label dosyasını taşı
                        label_file = os.path.splitext(img_file)[0] + '.txt'
                        label_src = os.path.join(train_labels_dir, label_file)
                        label_dst = os.path.join(val_labels_dir, label_file)
                        
                        if os.path.exists(label_src):
                            shutil.move(label_src, label_dst)
                            moved_count += 1
                
                print(f"✅ Validation split created: {moved_count} samples moved to validation")
                return True
            else:
                print("❌ No training data found to create validation split")
                return False
        else:
            val_count = len([f for f in os.listdir(val_images_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✅ Validation data already exists: {val_count} images")
            return True

class ReportGenerator:
    """Report generation and analysis"""
    
    def __init__(self, manager):
        self.manager = manager
    
    def generate_analysis_report(self, merged_class_counts):
        """Generate analysis report"""
        print("\n===== Generating Analysis Report =====")
        
        total_samples = sum(merged_class_counts.values())
        
        # Create report
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config_file': self.manager.config_file,
            'output_directory': self.manager.output_dir,
            
            'dataset_summary': {
                'source_datasets': len(self.manager.datasets),
                'total_samples': total_samples,
                'main_classes': len(self.manager.hierarchical_classes),
                'original_classes': len(self.manager.class_mapping)
            },
            
            'class_distribution': {
                'merged_counts': dict(merged_class_counts),
                'class_balance': self._calculate_class_balance(merged_class_counts)
            },
            
            'recommendations': self._generate_recommendations(merged_class_counts, total_samples)
        }
        
        # Save report
        with open('analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Saved analysis report to: analysis_report.json")
        
        # Print summary
        self._print_report_summary(report)
        
        return report
    
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
            'balance_ratio': min(counts) / max(counts) if max(counts) > 0 else 0,
            'percentages': {class_name: (count / total) * 100 
                          for class_name, count in class_counts.items()}
        }
    
    def _generate_recommendations(self, class_counts, total_samples):
        """Generate training recommendations"""
        recommendations = []
        
        if total_samples < 1000:
            recommendations.append("Dataset is small (<1000 samples). Consider increasing augmentation.")
        elif total_samples > 50000:
            recommendations.append("Large dataset detected. Consider reducing batch size.")
        
        balance_info = self._calculate_class_balance(class_counts)
        if balance_info.get('balance_ratio', 0) < 0.3:
            recommendations.append("Class imbalance detected. Consider weighted loss.")
        
        if total_samples < 5000:
            recommendations.append("Recommended model: yolo11s.pt or yolo11m.pt")
        elif total_samples > 20000:
            recommendations.append("Recommended model: yolo11l.pt or yolo11x.pt")
        
        return recommendations
    
    def _print_report_summary(self, report):
        """Print formatted summary"""
        print("\n" + "="*60)
        print("📊 DATASET ANALYSIS SUMMARY")
        print("="*60)
        
        summary = report['dataset_summary']
        print(f"📁 Output Directory: {report['output_directory']}")
        print(f"📦 Source Datasets: {summary['source_datasets']}")
        print(f"📊 Total Samples: {summary['total_samples']:,}")
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
