#!/usr/bin/env python3
# augmentation_utils.py - Advanced augmentation utilities for YOLO11 training

import cv2
import numpy as np
import random
import os
from pathlib import Path
import shutil
import albumentations as A
from albumentations.core.composition import BboxParams
import json

# Albumentations import kontrolü
try:
    import albumentations as A
    from albumentations.core.composition import BboxParams
    ALBUMENTATIONS_AVAILABLE = True
    print("✅ Albumentations library loaded successfully")
except ImportError:
    print("⚠️  Albumentations not installed. Using basic OpenCV augmentation.")
    print("   For advanced augmentation, run: pip install albumentations")
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    BboxParams = None
    
class YOLOAugmentationPipeline:
    """Advanced augmentation pipeline optimized for agricultural datasets"""
    
    def __init__(self, image_size=640, severity_level='medium'):
        self.image_size = image_size
        self.severity_level = severity_level
        # Albumentations major version detection
        try:
            self._albu_major = int(A.__version__.split('.')[0])
        except Exception:
            self._albu_major = 1
        # Load class names for nicer logs (optional)
        self.class_names = self._load_class_names()
        
        # Severity level configurations
        self.severity_configs = {
            'light': {
                'brightness_limit': 0.1,
                'contrast_limit': 0.1,
                'noise_var': 10,
                'blur_limit': 3,
                'rotation_limit': 5,
                'scale_limit': 0.05
            },
            'medium': {
                'brightness_limit': 0.2,
                'contrast_limit': 0.2,
                'noise_var': 15,
                'blur_limit': 5,
                'rotation_limit': 10,
                'scale_limit': 0.1
            },
            'heavy': {
                'brightness_limit': 0.3,
                'contrast_limit': 0.3,
                'noise_var': 25,
                'blur_limit': 7,
                'rotation_limit': 15,
                'scale_limit': 0.2
            }
        }
        
        self.config = self.severity_configs.get(severity_level, self.severity_configs['medium'])
        
        # Preprocess: enforce fixed size with letterbox (keeps aspect ratio, pads to square)
        # Robust to different Albumentations versions: try border_value, fallback to value.
        try:
            self.preprocess = A.Compose([
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size,
                              border_mode=cv2.BORDER_CONSTANT, border_value=(114, 114, 114))
            ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels'], clip=True))
        except TypeError:
            self.preprocess = A.Compose([
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(min_height=self.image_size, min_width=self.image_size,
                              border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
            ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels'], clip=True))
        
        # Agricultural specific augmentations
        self.agricultural_transforms = self._create_agricultural_pipeline()
        self.geometric_transforms = self._create_geometric_pipeline()
        self.color_transforms = self._create_color_pipeline()
        
    def _load_class_names(self):
        """Load class names from config/class_ids.json if available."""
        try:
            cid_path = os.path.join('config', 'class_ids.json')
            if os.path.exists(cid_path):
                with open(cid_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                names = data.get('names')
                if isinstance(names, list) and names:
                    return {i: str(n) for i, n in enumerate(names)}
        except Exception:
            pass
        return None

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.class_names, dict) and class_id in self.class_names:
            return self.class_names[class_id]
        return f"class_{class_id}"

    @staticmethod
    def _print_progress(prefix: str, current: int, total: int, bar_len: int = 30):
        if total <= 0:
            print(f"\r{prefix} 0/0", end='', flush=True)
            return
        filled = int(bar_len * current / total)
        bar = '█' * filled + '-' * (bar_len - filled)
        print(f"\r{prefix} [{bar}] {current}/{total}", end='', flush=True)
        if current >= total:
            print()  # newline at completion

    def _create_agricultural_pipeline(self):
        """Create agriculture-specific augmentation pipeline"""
        config = self.config
        
        return A.Compose([
            # Lighting conditions (very important for agricultural data)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=config['brightness_limit'],
                    contrast_limit=config['contrast_limit'],
                    p=0.8
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.6),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            ], p=0.9),

            # Weather simulation
            A.OneOf([
                # Use safe parameters across versions to avoid deprecation warnings
                A.RandomRain(
                    drop_length=10,
                    drop_width=1,
                    blur_value=1,
                    brightness_coefficient=0.9,
                    rain_type="drizzle",
                    p=0.3
                ),
                # Use safest signature to avoid version warnings
                A.RandomFog(p=0.2),
                # Shadow: avoid lower/upper args; rely on defaults + ROI
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    p=0.3
                ),
            ], p=0.4),

            # Texture and quality variations
            A.OneOf([
                # Some environments warn on var_limit; fallback to defaults if needed
                (A.GaussNoise(var_limit=(10, config['noise_var']), p=0.6) if self._albu_major < 2 else A.GaussNoise(p=0.6)),
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.1, 0.3), p=0.4),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=0.5),
            ], p=0.5),

            # Blur effects (camera focus, movement)
            A.OneOf([
                A.MotionBlur(blur_limit=config['blur_limit'], p=0.4),
                A.GaussianBlur(blur_limit=config['blur_limit'], p=0.6),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.2), p=0.3),
            ], p=0.3),

        ])
    
    def _create_geometric_pipeline(self):
        """Create geometric transformation pipeline"""
        config = self.config
        
        return A.Compose([
            # Perspective and distortion (camera angle changes)
            A.OneOf([
                A.Perspective(scale=(0.02, 0.05), p=0.5),
                # alpha_affine is not supported in some newer versions
                (A.ElasticTransform(alpha=50, sigma=5, p=0.3) if self._albu_major >= 2 else A.ElasticTransform(alpha=50, sigma=5, alpha_affine=10, p=0.3)),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.4),
            ], p=0.4),

            # Rotation and flipping
            A.OneOf([
                A.Rotate(limit=config['rotation_limit'], p=0.8),
                A.SafeRotate(limit=config['rotation_limit'], p=0.6),
            ], p=0.6),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # Rare for agricultural scenes
            
            # Scale and crop variations
            A.OneOf([
                A.RandomScale(scale_limit=config['scale_limit'], p=0.6),
                A.LongestMaxSize(max_size=int(self.image_size * 1.1), p=0.4),
            ], p=0.5),

        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels'], clip=True))
    
    def _create_color_pipeline(self):
        """Create color transformation pipeline"""
        return A.Compose([
            # HSV adjustments (very important for plant health detection)
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.7
            ),
            
            # Channel operations
            A.OneOf([
                A.ChannelShuffle(p=0.3),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.6),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            ], p=0.5),

            # Advanced color manipulations
            A.OneOf([
                A.Posterize(num_bits=6, p=0.3),
                A.Equalize(mode='cv', by_channels=True, mask=None, p=0.3),
                # threshold param may warn in newer versions
                (A.Solarize(p=0.2) if self._albu_major >= 2 else A.Solarize(threshold=128, p=0.2)),
            ], p=0.3),

        ])
    
    def apply_augmentation(self, image, bboxes, class_labels, augmentation_type='mixed'):
        """Apply augmentation to image and bounding boxes"""
        
        # Convert YOLO format bboxes to albumentations format if needed
        if isinstance(bboxes, list) and len(bboxes) > 0:
            # Ensure bboxes are in the correct format
            formatted_bboxes = []
            for bbox in bboxes:
                if len(bbox) >= 4:
                    # YOLO format: [x_center, y_center, width, height] (normalized)
                    x_center, y_center, width, height = bbox[:4]
                    formatted_bboxes.append([x_center, y_center, width, height])
            bboxes = formatted_bboxes
        
        try:
            # Preprocess to fixed size first (letterbox to self.image_size)
            pre = self.preprocess(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes, class_labels = pre['image'], pre['bboxes'], pre['class_labels']
            
            if augmentation_type == 'agricultural':
                transformed = self.agricultural_transforms(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            elif augmentation_type == 'geometric':
                transformed = self.geometric_transforms(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
            elif augmentation_type == 'color':
                # Color-only pipeline does not declare bbox_params; keep bboxes/labels as-is
                transformed = self.color_transforms(image=image)
                return transformed['image'], bboxes, class_labels
            else:  # mixed
                # Randomly choose augmentation type
                aug_type = random.choice(['agricultural', 'geometric', 'color'])
                return self.apply_augmentation(image, bboxes, class_labels, aug_type)
            
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
            
        except Exception as e:
            print(f"Augmentation error: {e}")
            return image, bboxes, class_labels
    
    def augment_dataset_batch(self, image_paths, label_paths, output_dir, target_count_per_class):
        """Batch augment dataset to reach target count per class"""
        print(f"\n===== Batch Augmentation Başlıyor =====")
        print(f"Hedef sınıf başına örnek: {target_count_per_class}")
        
        # Analyze current class distribution
        class_counts = self._analyze_current_distribution(label_paths)
        print(f"Mevcut sınıf dağılımı: {class_counts}")
        
        # Calculate augmentation needs
        augmentation_needs = {}
        for class_id, current_count in class_counts.items():
            if current_count < target_count_per_class:
                augmentation_needs[class_id] = target_count_per_class - current_count
        
        print(f"Augmentation ihtiyaçları: {augmentation_needs}")
        
        # Create output directories
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)
        
        # Copy original files first
        self._copy_original_files(image_paths, label_paths, output_dir)
        
        # Generate augmented samples
        augmented_count = 0
        for class_id, needed_count in augmentation_needs.items():
            cname = self._class_name(class_id)
            print(f"\nSınıf {class_id} ({cname}) için {needed_count} augmentation yapılıyor...")
            
            # Find files containing this class
            class_files = self._find_files_with_class(label_paths, class_id)
            
            if not class_files:
                print(f"Sınıf {class_id} için dosya bulunamadı!")
                continue
            
            # Generate augmented samples
            generated = self._generate_augmented_samples(
                class_files, image_paths, needed_count, class_id, output_dir, augmented_count,
                progress_prefix=f"Sınıf {class_id} ({cname})"
            )
            augmented_count += generated
        
        print(f"\n✅ Batch augmentation tamamlandı!")
        print(f"Toplam {augmented_count} augmented sample oluşturuldu.")
        
        return augmented_count
    
    def _analyze_current_distribution(self, label_paths):
        """Analyze current class distribution"""
        class_counts = {}
        
        for label_path in label_paths:
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                            except ValueError:
                                pass
        
        return class_counts
    
    def _find_files_with_class(self, label_paths, target_class_id):
        """Find label files that contain the target class"""
        matching_files = []
        
        for label_path in label_paths:
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                if class_id == target_class_id:
                                    matching_files.append(label_path)
                                    break
                            except ValueError:
                                pass
        
        return matching_files
    
    def _copy_original_files(self, image_paths, label_paths, output_dir):
        """Copy original files to output directory"""
        print("Orijinal dosyalar kopyalanıyor...")
        
        for i, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                # Create new filenames
                base_name = f"orig_{i:06d}"
                
                # Copy image
                img_ext = os.path.splitext(img_path)[1]
                new_img_path = f"{output_dir}/images/{base_name}{img_ext}"
                shutil.copy2(img_path, new_img_path)
                
                # Copy label
                new_lbl_path = f"{output_dir}/labels/{base_name}.txt"
                shutil.copy2(lbl_path, new_lbl_path)
    
    def _generate_augmented_samples(self, class_files, image_paths, needed_count, class_id, output_dir, start_counter, progress_prefix: str = None):
        """Generate augmented samples for a specific class"""
        generated_count = 0
        
        # Create mapping from label paths to image paths
        label_to_image = {}
        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Find corresponding label file
            for lbl_path in class_files:
                lbl_base = os.path.splitext(os.path.basename(lbl_path))[0]
                if base_name == lbl_base:
                    label_to_image[lbl_path] = img_path
                    break
        
        for i in range(needed_count):
            # Cyclically select source file
            source_label = class_files[i % len(class_files)]
            source_image = label_to_image.get(source_label)
            
            if not source_image or not os.path.exists(source_image):
                continue
            
            try:
                # Load image
                image = cv2.imread(source_image)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Load bboxes and class labels
                bboxes = []
                class_labels = []
                
                with open(source_label, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cls_id = int(parts[0])
                        except (ValueError, IndexError):
                            continue

                        # Detection format: class x y w h
                        if len(parts) == 5:
                            try:
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                            except ValueError:
                                continue
                        else:
                            # Segmentation format: class x1 y1 x2 y2 ... (normalized)
                            coords = parts[1:]
                            if len(coords) < 6 or len(coords) % 2 != 0:
                                # Not enough points to form a polygon; skip
                                continue
                            try:
                                xs = [float(coords[i]) for i in range(0, len(coords), 2)]
                                ys = [float(coords[i+1]) for i in range(0, len(coords), 2)]
                            except ValueError:
                                continue
                            # Compute bbox from polygon
                            x_min = max(0.0, min(xs))
                            y_min = max(0.0, min(ys))
                            x_max = min(1.0, max(xs))
                            y_max = min(1.0, max(ys))
                            width = max(0.0, x_max - x_min)
                            height = max(0.0, y_max - y_min)
                            if width <= 0.0 or height <= 0.0:
                                continue
                            x_center = (x_min + x_max) / 2.0
                            y_center = (y_min + y_max) / 2.0

                        # Clip to [0,1] and filter tiny boxes
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        width = max(0.0, min(1.0, width))
                        height = max(0.0, min(1.0, height))
                        if width < 1e-4 or height < 1e-4:
                            continue

                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(cls_id)
                
                if not bboxes:
                    continue
                
                # Apply augmentation
                aug_image, aug_bboxes, aug_labels = self.apply_augmentation(
                    image, bboxes, class_labels
                )
                
                # Save augmented image
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                
                file_counter = start_counter + generated_count
                aug_img_name = f"aug_{class_id}_{file_counter:06d}.jpg"
                aug_lbl_name = f"aug_{class_id}_{file_counter:06d}.txt"
                
                aug_img_path = f"{output_dir}/images/{aug_img_name}"
                aug_lbl_path = f"{output_dir}/labels/{aug_lbl_name}"
                
                cv2.imwrite(aug_img_path, aug_image_bgr)
                
                # Save augmented labels
                with open(aug_lbl_path, 'w') as f:
                    for bbox, label in zip(aug_bboxes, aug_labels):
                        if len(bbox) >= 4:
                            f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                generated_count += 1
                # Inline progress bar update
                if progress_prefix:
                    self._print_progress(progress_prefix, generated_count, needed_count)
                
            except Exception as e:
                print(f"Augmentation hatası: {e}")
                continue
        
        # Ensure progress ends with a newline and full bar
        if progress_prefix and generated_count >= 0:
            self._print_progress(progress_prefix, needed_count, needed_count)
        return generated_count

class SmartAugmentationRecommender:
    """Smart augmentation recommender based on dataset characteristics"""
    
    def __init__(self):
        self.recommendations = {}
    
    def analyze_and_recommend(self, dataset_path, class_names):
        """Analyze dataset and recommend augmentation strategy"""
        print(f"\n===== Smart Augmentation Analysis =====")
        
        # Analyze dataset characteristics
        characteristics = self._analyze_dataset_characteristics(dataset_path, class_names)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(characteristics)
        
        # Display recommendations
        self._display_recommendations(recommendations)
        
        return recommendations
    
    def _analyze_dataset_characteristics(self, dataset_path, class_names):
        """Analyze dataset characteristics"""
        characteristics = {
            'class_distribution': {},
            'image_qualities': [],
            'lighting_conditions': [],
            'image_sizes': [],
            'bbox_sizes': [],
            'dataset_type': 'mixed'
        }
        
        # Analyze images and labels
        images_dir = f"{dataset_path}/images/train"
        labels_dir = f"{dataset_path}/labels/train"
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sample analysis (limit to 100 images for speed)
            sample_files = random.sample(image_files, min(100, len(image_files)))
            
            for img_file in sample_files:
                img_path = os.path.join(images_dir, img_file)
                lbl_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
                
                # Analyze image
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    characteristics['image_sizes'].append((w, h))
                    
                    # Estimate lighting condition
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)
                    characteristics['lighting_conditions'].append(brightness)
                    
                    # Estimate image quality (using Laplacian variance)
                    quality = cv2.Laplacian(gray, cv2.CV_64F).var()
                    characteristics['image_qualities'].append(quality)
                
                # Analyze labels
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:
                                try:
                                    class_id = int(parts[0])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    
                                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                                    characteristics['class_distribution'][class_name] = characteristics['class_distribution'].get(class_name, 0) + 1
                                    characteristics['bbox_sizes'].append((width, height))
                                except (ValueError, IndexError):
                                    pass
        
        # Determine dataset type
        if any('disease' in name.lower() or 'pest' in name.lower() for name in class_names):
            characteristics['dataset_type'] = 'agricultural'
        elif any('deficiency' in name.lower() for name in class_names):
            characteristics['dataset_type'] = 'nutrient'
        
        return characteristics
    
    def _generate_recommendations(self, characteristics):
        """Generate augmentation recommendations"""
        recommendations = {
            'severity_level': 'medium',
            'primary_augmentations': [],
            'secondary_augmentations': [],
            'avoid_augmentations': [],
            'special_considerations': []
        }
        
        # Analyze class imbalance
        class_counts = list(characteristics['class_distribution'].values())
        if class_counts:
            max_count = max(class_counts)
            min_count = min(class_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 5:
                recommendations['severity_level'] = 'heavy'
                recommendations['special_considerations'].append(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}). Use heavy augmentation for minority classes."
                )
            elif imbalance_ratio > 2:
                recommendations['severity_level'] = 'medium'
                recommendations['special_considerations'].append(
                    f"Moderate class imbalance detected (ratio: {imbalance_ratio:.1f})."
                )
        
        # Lighting analysis
        if characteristics['lighting_conditions']:
            avg_brightness = np.mean(characteristics['lighting_conditions'])
            brightness_std = np.std(characteristics['lighting_conditions'])
            
            if brightness_std < 20:  # Low variance in lighting
                recommendations['primary_augmentations'].extend([
                    'RandomBrightnessContrast',
                    'RandomGamma',
                    'CLAHE'
                ])
                recommendations['special_considerations'].append(
                    "Limited lighting variation detected. Focus on brightness/contrast augmentations."
                )
            
            if avg_brightness < 100:  # Dark images
                recommendations['avoid_augmentations'].append('RandomShadow')
                recommendations['special_considerations'].append(
                    "Dark images detected. Avoid shadow augmentations."
                )
        
        # Image quality analysis
        if characteristics['image_qualities']:
            avg_quality = np.mean(characteristics['image_qualities'])
            
            if avg_quality < 100:  # Low quality images
                recommendations['avoid_augmentations'].extend(['GaussianBlur', 'MotionBlur'])
                recommendations['special_considerations'].append(
                    "Low image quality detected. Avoid blur augmentations."
                )
            else:
                recommendations['secondary_augmentations'].extend(['GaussianBlur', 'MotionBlur'])
        
        # Bbox size analysis
        if characteristics['bbox_sizes']:
            avg_bbox_area = np.mean([w*h for w, h in characteristics['bbox_sizes']])
            
            if avg_bbox_area < 0.01:  # Very small objects
                recommendations['avoid_augmentations'].extend(['RandomScale', 'Perspective'])
                recommendations['special_considerations'].append(
                    "Small objects detected. Avoid scale and perspective transformations."
                )
        
        # Dataset type specific recommendations
        if characteristics['dataset_type'] == 'agricultural':
            recommendations['primary_augmentations'].extend([
                'HueSaturationValue',
                'RandomRain',
                'RandomShadow'
            ])
            recommendations['special_considerations'].append(
                "Agricultural dataset detected. Use weather and lighting augmentations."
            )
        
        elif characteristics['dataset_type'] == 'nutrient':
            recommendations['primary_augmentations'].extend([
                'HueSaturationValue',
                'ColorJitter',
                'RGBShift'
            ])
            recommendations['avoid_augmentations'].append('RandomRain')
            recommendations['special_considerations'].append(
                "Nutrient deficiency dataset detected. Focus on color augmentations."
            )
        
        return recommendations
    
    def _display_recommendations(self, recommendations):
        """Display augmentation recommendations"""
        print(f"\n🎯 Augmentation Recommendations:")
        print(f"Severity Level: {recommendations['severity_level']}")
        
        if recommendations['primary_augmentations']:
            print(f"\n✅ Primary Augmentations:")
            for aug in recommendations['primary_augmentations']:
                print(f"  - {aug}")
        
        if recommendations['secondary_augmentations']:
            print(f"\n🔄 Secondary Augmentations:")
            for aug in recommendations['secondary_augmentations']:
                print(f"  - {aug}")
        
        if recommendations['avoid_augmentations']:
            print(f"\n❌ Avoid These Augmentations:")
            for aug in recommendations['avoid_augmentations']:
                print(f"  - {aug}")
        
        if recommendations['special_considerations']:
            print(f"\n💡 Special Considerations:")
            for consideration in recommendations['special_considerations']:
                print(f"  - {consideration}")

# Example usage and testing
if __name__ == "__main__":
    # Test the augmentation pipeline
    pipeline = YOLOAugmentationPipeline(image_size=640, severity_level='medium')
    
    print("YOLOAugmentationPipeline test...")
    print(f"Severity level: {pipeline.severity_level}")
    print(f"Config: {pipeline.config}")
    
    # Test smart recommender
    recommender = SmartAugmentationRecommender()
    
    # Example class names for agricultural dataset
    example_classes = [
        'healthy', 'fungal_disease', 'viral_disease', 'pest_damage',
        'nutrient_deficiency', 'fruit_ripe', 'fruit_unripe'
    ]
    
    print(f"\nSmart recommender initialized.")
    print(f"Example classes: {example_classes}")
    
    print(f"\n✅ Augmentation utilities ready!")
