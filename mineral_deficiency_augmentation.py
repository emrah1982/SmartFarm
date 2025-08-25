#!/usr/bin/env python3
# mineral_deficiency_augmentation.py - Mineral Deficiency Detection Augmentation System

import albumentations as A
import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from datetime import datetime
from typing import List, Dict, Tuple

class MineralDeficiencyAugmentation:
    """Mineral eksikliği tespiti için augmentation sistemi"""
    
    def __init__(self, images_dir: str, labels_dir: str, output_images_dir: str, output_labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) 
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # CSV raporlama için
        self.missing_data_log = []
        self.processing_stats = {
            'total_images': 0,
            'successful_augmentations': 0,
            'failed_augmentations': 0,
            'missing_minerals': {},
            'start_time': datetime.now()
        }
        
        # Çıkış dizinlerini oluştur
        self.output_images_dir.mkdir(exist_ok=True, parents=True)
        self.output_labels_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"🔬 Mineral Eksikliği Augmentation Sistemi Başlatıldı")
        print(f"📁 Giriş: {self.images_dir}")
        print(f"📁 Çıkış: {self.output_images_dir}")
        
        # Sabit boyut ön işleme (letterbox) - YOLO uyumlu gri arkaplan
        self.target_size = 512
        self.preprocess = A.Compose([
            A.LongestMaxSize(max_size=self.target_size),
            A.PadIfNeeded(min_height=self.target_size, min_width=self.target_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def _clip_and_filter_bboxes(self, bboxes, class_labels):
        """YOLO bbox'ları [0,1] aralığına kırp ve sıfır/alakasız kutuları filtrele"""
        if not bboxes:
            return [], []
        clipped, labels = [], []
        for bbox, cid in zip(bboxes, class_labels):
            x, y, w, h = bbox
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            if w <= 0 or h <= 0:
                continue
            # Kutunun görüntü sınırları içinde kalmasını sağla
            if x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1:
                # Aşımı kırp: Merkez sabit, genişlik/yükseklik azalt
                left = max(0.0, x - w/2)
                right = min(1.0, x + w/2)
                top = max(0.0, y - h/2)
                bottom = min(1.0, y + h/2)
                w = max(0.0, right - left)
                h = max(0.0, bottom - top)
                if w <= 0 or h <= 0:
                    continue
                x = (left + right) / 2
                y = (top + bottom) / 2
            clipped.append([x, y, w, h])
            labels.append(cid)
        return clipped, labels
    
    def get_nitrogen_deficiency_transforms(self):
        """Azot (N) eksikliği - Yaşlı yapraklar sarı, genç yapraklar açık yeşil"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(10, 25), sat_shift_limit=(-30, -10), val_shift_limit=(-10, 5), p=0.8),
                A.ColorJitter(brightness=(-0.1, 0.1), contrast=(0.9, 1.1), saturation=(0.6, 0.9), hue=(0.02, 0.08), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.6),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(4, 4), p=0.4),
            ], p=0.7),
            A.OneOf([
                A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.4),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=5, p=0.5),
            ], p=0.6),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
    
    def get_phosphorus_deficiency_transforms(self):
        """Fosfor (P) eksikliği - Mor/kırmızımsı renk, büyüme geriliği"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(-25, -5), sat_shift_limit=(5, 25), val_shift_limit=(-15, -5), p=0.8),
                A.ColorJitter(brightness=(-0.15, 0.05), contrast=(1.1, 1.3), saturation=(1.1, 1.4), hue=(-0.1, -0.02), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=3, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(0.1, 0.3), p=0.6),
            ], p=0.5),
            A.OneOf([A.Rotate(limit=10, p=0.4), A.HorizontalFlip(p=0.5), A.Perspective(scale=(0.02, 0.04), p=0.3)], p=0.7),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    def get_potassium_deficiency_transforms(self):
        """Potasyum (K) eksikliği - Yaprak kenarlarında kahverengi yanık"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(5, 20), sat_shift_limit=(-20, 10), val_shift_limit=(-20, -5), p=0.8),
                A.ColorJitter(brightness=(-0.2, 0.05), contrast=(1.0, 1.25), saturation=(0.8, 1.2), hue=(0.01, 0.06), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.1), contrast_limit=(0.1, 0.2), p=0.7),
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.4),
            ], p=0.8),
            A.OneOf([A.OpticalDistortion(distort_limit=0.05, p=0.3), A.ElasticTransform(alpha=30, sigma=5, p=0.2)], p=0.4),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
    
    def get_magnesium_deficiency_transforms(self):
        """Magnezyum (Mg) eksikliği - Damarlar arası sarılaşma"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(15, 30), sat_shift_limit=(-25, -5), val_shift_limit=(5, 15), p=0.8),
                A.ColorJitter(brightness=(0.05, 0.15), contrast=(0.9, 1.1), saturation=(0.7, 0.9), hue=(0.04, 0.1), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.CLAHE(clip_limit=3.0, tile_grid_size=(6, 6), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.6),
            ], p=0.7),
            A.UnsharpMask(blur_limit=(3, 5), sigma_limit=(1.0, 1.5), alpha=(0.1, 0.25), threshold=5, p=0.4),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
    
    def get_calcium_deficiency_transforms(self):
        """Kalsiyum (Ca) eksikliği - Yaprak ucu yanığı, nekrotik lekeler"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(-10, 15), sat_shift_limit=(-30, 5), val_shift_limit=(-25, -10), p=0.8),
                A.ColorJitter(brightness=(-0.2, 0.1), contrast=(1.1, 1.3), saturation=(0.6, 1.0), hue=(-0.02, 0.04), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(0.15, 0.3), p=0.7),
                A.GaussNoise(var_limit=(10, 25), p=0.3),
            ], p=0.6),
            A.OneOf([A.ElasticTransform(alpha=25, sigma=4, p=0.2), A.OpticalDistortion(distort_limit=0.08, p=0.3)], p=0.4),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    def get_iron_deficiency_transforms(self):
        """Demir (Fe) eksikliği - Genç yapraklarda kloroz"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(20, 35), sat_shift_limit=(-35, -15), val_shift_limit=(10, 25), p=0.8),
                A.ColorJitter(brightness=(0.1, 0.2), contrast=(0.8, 1.0), saturation=(0.5, 0.8), hue=(0.06, 0.12), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.CLAHE(clip_limit=2.8, tile_grid_size=(5, 5), p=0.5),
                A.Sharpen(alpha=(0.15, 0.35), lightness=(1.0, 1.3), p=0.4),
            ], p=0.6),
            A.OneOf([A.Rotate(limit=6, p=0.4), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=3, p=0.5)], p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
    
    def get_sulfur_deficiency_transforms(self):
        """Kükürt (S) eksikliği - Genç yapraklar soluk yeşil-sarı"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(12, 28), sat_shift_limit=(-25, -8), val_shift_limit=(5, 18), p=0.8),
                A.ColorJitter(brightness=(0.05, 0.12), contrast=(0.85, 1.05), saturation=(0.65, 0.85), hue=(0.03, 0.08), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(0.05, 0.15), contrast_limit=(-0.1, 0.1), p=0.6),
                A.RandomGamma(gamma_limit=(90, 110), p=0.4),
            ], p=0.7),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
    
    def get_zinc_deficiency_transforms(self):
        """Çinko (Zn) eksikliği - Küçük yaprak, damarlar arası kloroz"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(18, 32), sat_shift_limit=(-20, -5), val_shift_limit=(8, 20), p=0.8),
                A.ColorJitter(brightness=(0.08, 0.18), contrast=(0.9, 1.15), saturation=(0.7, 0.9), hue=(0.05, 0.1), p=0.7),
            ], p=0.9),
            A.OneOf([A.ShiftScaleRotate(scale_limit=(-0.15, -0.05), rotate_limit=5, p=0.4)], p=0.5),
            A.UnsharpMask(blur_limit=(2, 4), sigma_limit=(0.8, 1.2), alpha=(0.15, 0.3), threshold=8, p=0.4),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    def get_manganese_deficiency_transforms(self):
        """Mangan (Mn) eksikliği - Damarlar arası kloroz, nekrotik lekeler"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(10, 25), sat_shift_limit=(-28, -8), val_shift_limit=(-5, 12), p=0.8),
                A.ColorJitter(brightness=(-0.05, 0.15), contrast=(1.0, 1.2), saturation=(0.6, 0.9), hue=(0.02, 0.07), p=0.7),
            ], p=0.9),
            A.OneOf([A.GaussNoise(var_limit=(5, 15), p=0.4), A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.25), p=0.3)], p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
    
    def get_boron_deficiency_transforms(self):
        """Bor (B) eksikliği - Yaprak deformasyonu, büyüme noktası ölümü"""
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=(-15, 10), sat_shift_limit=(-25, 0), val_shift_limit=(-20, 5), p=0.8),
                A.ColorJitter(brightness=(-0.15, 0.1), contrast=(1.05, 1.25), saturation=(0.7, 1.0), hue=(-0.04, 0.03), p=0.7),
            ], p=0.9),
            A.OneOf([
                A.ElasticTransform(alpha=35, sigma=6, p=0.3),
                A.OpticalDistortion(distort_limit=0.1, p=0.4),
                A.GridDistortion(num_steps=3, distort_limit=0.05, p=0.3),
            ], p=0.6),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(0.1, 0.25), p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.25))
    
    # Mineral eksikliği türlerini mapping
    MINERAL_TRANSFORMS = {
        'nitrogen': get_nitrogen_deficiency_transforms,
        'phosphorus': get_phosphorus_deficiency_transforms, 
        'potassium': get_potassium_deficiency_transforms,
        'magnesium': get_magnesium_deficiency_transforms,
        'calcium': get_calcium_deficiency_transforms,
        'iron': get_iron_deficiency_transforms,
        'sulfur': get_sulfur_deficiency_transforms,
        'zinc': get_zinc_deficiency_transforms,
        'manganese': get_manganese_deficiency_transforms,
        'boron': get_boron_deficiency_transforms,
    }
    
    def read_yolo_annotation(self, txt_file):
        """YOLO formatındaki annotation dosyasını oku"""
        bboxes, class_labels = [], []
        if not txt_file.exists():
            return bboxes, class_labels
            
        try:
            with open(txt_file, 'r') as f:
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id = int(values[0])
                        x_center, y_center, width, height = map(float, values[1:5])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        except Exception as e:
            print(f"⚠️  Annotation okuma hatası {txt_file}: {e}")
        return bboxes, class_labels
    
    def save_yolo_annotation(self, txt_file, bboxes, class_labels):
        """Güncellenmiş annotationları YOLO formatında kaydet"""
        try:
            with open(txt_file, 'w') as f:
                for bbox, class_id in zip(bboxes, class_labels):
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        except Exception as e:
            print(f"⚠️  Annotation kaydetme hatası {txt_file}: {e}")
    
    def log_missing_data(self, image_file, mineral_type, reason):
        """Eksik veri durumlarını logla"""
        self.missing_data_log.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_file': image_file,
            'mineral_type': mineral_type,
            'reason': reason
        })
        if mineral_type not in self.processing_stats['missing_minerals']:
            self.processing_stats['missing_minerals'][mineral_type] = 0
        self.processing_stats['missing_minerals'][mineral_type] += 1
    
    def save_missing_data_report(self):
        """Eksik veri raporunu CSV olarak kaydet"""
        if not self.missing_data_log:
            print("📊 Eksik veri bulunamadı - rapor oluşturulmadı")
            return
            
        csv_file = self.output_images_dir.parent / f"missing_data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            df = pd.DataFrame(self.missing_data_log)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"📋 Eksik veri raporu kaydedildi: {csv_file}")
            print(f"📊 Toplam eksik veri: {len(self.missing_data_log)} kayıt")
        except Exception as e:
            print(f"⚠️  CSV rapor kaydetme hatası: {e}")
    
    def check_mineral_compatibility(self, image_file, mineral_type):
        """Görüntünün belirtilen mineral eksikliği için uygun olup olmadığını kontrol et"""
        # Basit uyumluluk kontrolü - geliştirilmeye açık
        return True  # Şimdilik tüm görüntüler işlenebilir
    
    def augment_mineral_deficiency(self, mineral_type, multiplier=4):
        """Belirli bir mineral eksikliği için augmentation"""
        if mineral_type not in self.MINERAL_TRANSFORMS:
            available = ', '.join(self.MINERAL_TRANSFORMS.keys())
            print(f"❌ Desteklenmeyen mineral türü: {mineral_type}")
            print(f"✅ Mevcut mineral türleri: {available}")
            return False
        
        transform_method = self.MINERAL_TRANSFORMS[mineral_type]
        transform = transform_method(self)
        
        # Görüntü dosyalarını bul
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_dir.glob(ext)))
        
        if not image_files:
            print(f"❌ {self.images_dir} dizininde görüntü dosyası bulunamadı")
            return False
        
        successful_augmentations = 0
        total_attempts = 0
        skipped_images = 0
        
        print(f"\n🔬 {mineral_type.upper()} eksikliği augmentation başlıyor...")
        print(f"📁 Toplam {len(image_files)} görüntü bulundu")
        
        for img_file in image_files:
            self.processing_stats['total_images'] += 1
            
            # Mineral uyumluluğu kontrolü
            if not self.check_mineral_compatibility(img_file, mineral_type):
                self.log_missing_data(img_file.name, mineral_type, "Mineral tipi uyumsuzluğu")
                skipped_images += 1
                continue
            
            # Görüntüyü yükle
            try:
                image = cv2.imread(str(img_file))
                if image is None:
                    self.log_missing_data(img_file.name, mineral_type, "Görüntü yüklenemedi")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.log_missing_data(img_file.name, mineral_type, f"Görüntü okuma hatası: {e}")
                continue
            
            # Annotation dosyasını oku
            txt_file = self.labels_dir / (img_file.stem + '.txt')
            bboxes, class_labels = self.read_yolo_annotation(txt_file)
            
            # Orijinali kopyala
            try:
                cv2.imwrite(str(self.output_images_dir / img_file.name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if txt_file.exists():
                    with open(txt_file, 'r') as src, open(self.output_labels_dir / txt_file.name, 'w') as dst:
                        dst.write(src.read())
            except Exception as e:
                print(f"⚠️  Orijinal dosya kopyalama hatası {img_file.name}: {e}")
            
            # Augmente edilmiş versiyonları üret
            for i in range(multiplier):
                total_attempts += 1
                try:
                    # Giriş bboxlarını doğrula/temizle
                    bboxes_valid, class_labels_valid = self._clip_and_filter_bboxes(bboxes, class_labels)
                    if not bboxes_valid:
                        self.log_missing_data(img_file.name, mineral_type, "Geçerli bbox bulunamadı (giriş)")
                        continue

                    # Önce preprocess (sabit boyut letterbox)
                    pre = self.preprocess(image=image, bboxes=bboxes_valid, class_labels=class_labels_valid)
                    pre_image, pre_bboxes, pre_labels = pre['image'], pre['bboxes'], pre['class_labels']
                    # Preprocess sonrası boyut doğrulaması
                    if pre_image is None:
                        self.log_missing_data(img_file.name, mineral_type, "Preprocess sonrası görüntü None")
                        continue
                    if pre_image.shape[:2] != (self.target_size, self.target_size):
                        self.logger.warning(f"Preprocess boyut hatası: {img_file.name} - Beklenen: {(self.target_size, self.target_size)}, Gerçek: {pre_image.shape[:2]}")
                        self.log_missing_data(img_file.name, mineral_type, f"Preprocess boyutu: beklenen {self.target_size}x{self.target_size}, gerçek {pre_image.shape[:2]}")
                        continue

                    transformed = transform(image=pre_image, bboxes=pre_bboxes, class_labels=pre_labels)
                    augmented_image = transformed['image']
                    augmented_bboxes = transformed['bboxes']
                    augmented_class_labels = transformed['class_labels']
                    # Transform sonrası bbox doğrulaması ve boyut kontrolü
                    augmented_bboxes, augmented_class_labels = self._clip_and_filter_bboxes(augmented_bboxes, augmented_class_labels)
                    if augmented_image is None:
                        self.log_missing_data(img_file.name, mineral_type, "Transform sonrası görüntü None")
                        continue
                    if augmented_image.shape[:2] != (self.target_size, self.target_size):
                        self.logger.warning(f"Transform boyut hatası: {img_file.name} - Beklenen: {(self.target_size, self.target_size)}, Gerçek: {augmented_image.shape[:2]}")
                        self.log_missing_data(img_file.name, mineral_type, f"Transform boyutu: beklenen {self.target_size}x{self.target_size}, gerçek {augmented_image.shape[:2]}")
                        continue
                    if not augmented_bboxes:
                        self.log_missing_data(img_file.name, mineral_type, "Geçerli bbox bulunamadı (çıkış)")
                        continue
                    
                    # Augmente edilmiş görüntüyü kaydet
                    output_img_name = f"{img_file.stem}_{mineral_type}_aug_{i+1}.jpg"
                    output_img_path = self.output_images_dir / output_img_name
                    cv2.imwrite(str(output_img_path), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                    
                    # Güncellenmiş annotationları kaydet
                    output_txt_name = f"{img_file.stem}_{mineral_type}_aug_{i+1}.txt"
                    output_txt_path = self.output_labels_dir / output_txt_name
                    self.save_yolo_annotation(output_txt_path, augmented_bboxes, augmented_class_labels)
                    
                    successful_augmentations += 1
                    self.processing_stats['successful_augmentations'] += 1
                    print(f"✅ {output_img_name} (bbox: {len(augmented_bboxes)})")
                    
                except Exception as e:
                    print(f"⚠️  {img_file.name} aug_{i+1} başarısız: {e}")
                    self.processing_stats['failed_augmentations'] += 1
                    self.log_missing_data(img_file.name, mineral_type, f"Augmentation hatası: {e}")
                    continue
        
        success_rate = (successful_augmentations / total_attempts) * 100 if total_attempts > 0 else 0
        print(f"\n📊 {mineral_type.upper()} Özet:")
        print(f"✅ Başarılı: {successful_augmentations}/{total_attempts} (%{success_rate:.1f})")
        print(f"⏭️  Atlanan: {skipped_images}")
        print(f"📁 Toplam üretilen dosya: {successful_augmentations + len(image_files)}")
        return True
    
    def augment_all_minerals(self, multiplier_per_mineral=3):
        """Tüm mineral eksiklikleri için augmentation"""
        print("🌱 TÜM MİNERAL EKSİKLİKLERİ İÇİN AUGMENTATION BAŞLIYOR...")
        print("=" * 60)
        
        for mineral_type in self.MINERAL_TRANSFORMS.keys():
            try:
                # Her mineral için farklı çıkış klasörü oluştur
                mineral_output_images = self.output_images_dir / mineral_type
                mineral_output_labels = self.output_labels_dir / mineral_type
                
                # Geçici olarak output klasörlerini değiştir
                original_img_dir = self.output_images_dir
                original_lbl_dir = self.output_labels_dir
                
                self.output_images_dir = mineral_output_images
                self.output_labels_dir = mineral_output_labels
                
                # Klasörleri oluştur
                self.output_images_dir.mkdir(exist_ok=True, parents=True)
                self.output_labels_dir.mkdir(exist_ok=True, parents=True)
                
                # Bu mineral için augmentation yap
                self.augment_mineral_deficiency(mineral_type, multiplier_per_mineral)
                
                # Orijinal klasörleri geri yükle
                self.output_images_dir = original_img_dir
                self.output_labels_dir = original_lbl_dir
                
            except Exception as e:
                print(f"❌ {mineral_type} için augmentation hatası: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("🎉 TÜM MİNERAL AUGMENTASYON İŞLEMİ TAMAMLANDI!")
        print(f"📂 Çıkış dizinini kontrol edin: {self.output_images_dir}")
        
        # Final rapor
        self.save_missing_data_report()
        self.print_final_report()
    
    def print_final_report(self):
        """Final işlem raporunu yazdır"""
        end_time = datetime.now()
        duration = end_time - self.processing_stats['start_time']
        
        print(f"\n📊 FINAL RAPOR")
        print(f"⏱️  İşlem süresi: {duration}")
        print(f"📁 Toplam işlenen görüntü: {self.processing_stats['total_images']}")
        print(f"✅ Başarılı augmentation: {self.processing_stats['successful_augmentations']}")
        print(f"❌ Başarısız augmentation: {self.processing_stats['failed_augmentations']}")
        
        if self.processing_stats['missing_minerals']:
            print(f"\n⚠️  Mineral bazında eksik veriler:")
            for mineral, count in self.processing_stats['missing_minerals'].items():
                print(f"   {mineral}: {count} eksik")
        
        print(f"\n🎯 Sonuç hazırlandı! Çıkış klasörünü kontrol edin: {self.output_images_dir}")

# Kullanım örneği
if __name__ == "__main__":
    # Ana pipeline oluştur
    pipeline = MineralDeficiencyAugmentation(
        images_dir="original_images",
        labels_dir="original_labels", 
        output_images_dir="augmented_mineral_images",
        output_labels_dir="augmented_mineral_labels"
    )
    
    # Tek mineral için test
    print("🧪 TEK MİNERAL TEST - AZOT EKSİKLİĞİ")
    pipeline.augment_mineral_deficiency('nitrogen', multiplier=2)
    
    # Tüm mineraller için
    print("\n" * 2)
    pipeline.augment_all_minerals(multiplier_per_mineral=3)
    
    # Manuel seçim örneği
    print("\n" * 2)
    print("🎯 MANUEL SEÇİM ÖRNEĞİ")
    critical_minerals = ['nitrogen', 'phosphorus', 'potassium', 'iron']
    for mineral in critical_minerals:
        print(f"\n⚗️  {mineral.upper()} işleniyor...")
        pipeline.augment_mineral_deficiency(mineral, multiplier=4)
