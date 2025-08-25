"""
Domates Hastalıkları Augmentation Sistemi
==========================================

Bu modül, domates hastalıkları için özel augmentation işlemleri gerçekleştirir.
10 farklı domates hastalığı için gerçekçi görsel transformasyonlar uygular.

Desteklenen Hastalıklar:
- Early Blight (Erken Yanıklık)
- Late Blight (Geç Yanıklık) 
- Leaf Mold (Yaprak Küfü)
- Septoria Leaf Spot (Septoria Yaprak Lekesi)
- Spider Mites (Kırmızı Örümcek)
- Target Spot (Hedef Leke)
- Yellow Leaf Curl Virus (Sarı Yaprak Kıvrılma Virüsü)
- Mosaic Virus (Mozaik Virüs)
- Bacterial Spot (Bakteriyel Leke)
- Healthy (Sağlıklı)

Kullanım:
    augmenter = TomatoDiseaseAugmentation()
    augmenter.augment_disease('early_blight', 'input_dir', 'output_dir', num_augmentations=5)
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations import BboxParams
import csv
import json
from datetime import datetime
from pathlib import Path
import logging

class TomatoDiseaseAugmentation:
    def __init__(self, log_level=logging.INFO):
        """
        Domates hastalığı augmentation sınıfı
        
        Args:
            log_level: Logging seviyesi
        """
        self.setup_logging(log_level)
        self.supported_diseases = [
            'early_blight', 'late_blight', 'leaf_mold', 'septoria_leaf_spot',
            'spider_mites', 'target_spot', 'yellow_leaf_curl', 'mosaic_virus',
            'bacterial_spot', 'healthy'
        ]
        
        # CSV raporlama için başlıklar
        self.csv_headers = [
            'timestamp', 'disease_type', 'image_path', 'status', 
            'error_message', 'augmentation_count'
        ]
        
        # İstatistikler
        self.stats = {
            'total_processed': 0,
            'successful_augmentations': 0,
            'skipped_images': 0,
            'errors': 0
        }
        # Görüntüleri sabit boyuta getirmek için ön işleme (letterbox)
        # 512x512 önerilen; YOLO uyumlu gri arkaplan ile pad edilir
        self.target_size = 512
        self.preprocess = A.Compose([
            A.LongestMaxSize(max_size=self.target_size),
            A.PadIfNeeded(min_height=self.target_size, min_width=self.target_size,
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

        # BBox filtreleme için minimum genişlik/yükseklik eşikleri (normalized YOLO formatında)
        self.min_box_w = 1e-3
        self.min_box_h = 1e-3

    def setup_logging(self, log_level):
        """Logging konfigürasyonu"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('tomato_disease_augmentation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _clip_and_filter_bboxes(self, bboxes, class_labels):
        """
        YOLO formatındaki bbox'ları [0,1] aralığına kırpar ve geçersiz/boş kutuları eler.

        Args:
            bboxes (list[list[float]]): [x, y, w, h] listesi
            class_labels (list[int]): sınıf etiketleri

        Returns:
            tuple: (filtered_bboxes, filtered_labels)
        """
        if not bboxes:
            return [], (class_labels if class_labels else [])

        filtered_bboxes = []
        filtered_labels = []
        for i, bbox in enumerate(bboxes):
            try:
                x, y, w, h = bbox
                # NaN/inf kontrolü
                if not all(np.isfinite([x, y, w, h])):
                    continue
                # [0,1] sınırına kırp
                x = float(np.clip(x, 0.0, 1.0))
                y = float(np.clip(y, 0.0, 1.0))
                w = float(np.clip(w, 0.0, 1.0))
                h = float(np.clip(h, 0.0, 1.0))
                # Degenerate kutuları ele
                if w < self.min_box_w or h < self.min_box_h:
                    continue
                filtered_bboxes.append([x, y, w, h])
                if class_labels:
                    filtered_labels.append(class_labels[i])
            except Exception:
                continue
        # Her zaman (bboxes, labels) döndür
        return filtered_bboxes, filtered_labels

    def _is_valid_size(self, image, expected=512):
        """Görüntü boyutunun expected x expected olup olmadığını doğrular."""
        try:
            h, w = image.shape[:2]
            return (h == expected and w == expected)
        except Exception:
            return False

    def get_early_blight_transform(self):
        """
        Early Blight (Erken Yanıklık) transformasyonu
        - Koyu kahverengi/siyah konsantrik halkalar
        - Yaprak kenarlarından başlayan nekroz
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.3, -0.1), contrast=(0.8, 1.2), 
                         saturation=(0.6, 0.9), hue=(-0.1, 0.1), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.1), 
                                     contrast_limit=(0.1, 0.3), p=0.9),
            A.HueSaturationValue(hue_shift_limit=(-20, 10), 
                               sat_shift_limit=(-30, -10), 
                               val_shift_limit=(-40, -10), p=0.8),
            A.GaussNoise(var_limit=(10, 30), p=0.6),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_late_blight_transform(self):
        """
        Late Blight (Geç Yanıklık) transformasyonu
        - Su emmiş görünüm, koyu yeşil-kahverengi lekeler
        - Hızlı yayılan nekrotik alanlar
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.4, -0.2), contrast=(0.7, 1.1), 
                         saturation=(0.5, 0.8), hue=(-0.15, 0.05), p=0.9),
            A.Blur(blur_limit=(3, 7), p=0.7),
            A.GaussNoise(var_limit=(15, 35), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.2), 
                                     contrast_limit=(0.2, 0.4), p=0.9),
            A.HueSaturationValue(hue_shift_limit=(-25, 5), 
                               sat_shift_limit=(-40, -15), 
                               val_shift_limit=(-50, -20), p=0.8),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_leaf_mold_transform(self):
        """
        Leaf Mold (Yaprak Küfü) transformasyonu
        - Sarı lekeler, alt yüzeyde gri-kahverengi küf
        - Yüksek nem koşullarında gelişir
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.2, 0.1), contrast=(0.9, 1.3), 
                         saturation=(0.7, 1.1), hue=(-0.1, 0.2), p=0.8),
            A.HueSaturationValue(hue_shift_limit=(-10, 30), 
                               sat_shift_limit=(-20, 10), 
                               val_shift_limit=(-30, 10), p=0.8),
            A.GaussNoise(var_limit=(5, 20), p=0.6),
            A.Blur(blur_limit=(1, 3), p=0.5),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_septoria_leaf_spot_transform(self):
        """
        Septoria Leaf Spot transformasyonu
        - Küçük, yuvarlak, koyu kenarlı lekeler
        - Merkezi açık renkli, kenarları koyu
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.3, 0.0), contrast=(1.0, 1.4), 
                         saturation=(0.6, 0.9), hue=(-0.1, 0.1), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.0), 
                                     contrast_limit=(0.2, 0.5), p=0.9),
            A.HueSaturationValue(hue_shift_limit=(-15, 15), 
                               sat_shift_limit=(-25, -5), 
                               val_shift_limit=(-35, -5), p=0.8),
            A.GaussNoise(var_limit=(8, 25), p=0.7),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_spider_mites_transform(self):
        """
        Spider Mites (Kırmızı Örümcek) transformasyonu
        - Yapraklarda sarı benekler, bronzlaşma
        - İnce ağ yapıları görülebilir
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.2, 0.2), contrast=(0.8, 1.2), 
                         saturation=(0.7, 1.2), hue=(-0.05, 0.15), p=0.8),
            A.HueSaturationValue(hue_shift_limit=(-5, 25), 
                               sat_shift_limit=(-15, 15), 
                               val_shift_limit=(-25, 15), p=0.8),
            A.GaussNoise(var_limit=(10, 30), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), 
                                     contrast_limit=(0.1, 0.3), p=0.7),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_target_spot_transform(self):
        """
        Target Spot (Hedef Leke) transformasyonu
        - Konsantrik halkalı lekeler (hedef tahtası görünümü)
        - Koyu kahverengi kenarlar, açık merkez
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.3, -0.1), contrast=(1.1, 1.5), 
                         saturation=(0.6, 0.9), hue=(-0.1, 0.1), p=0.9),
            A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.1), 
                                     contrast_limit=(0.3, 0.6), p=0.9),
            A.HueSaturationValue(hue_shift_limit=(-20, 10), 
                               sat_shift_limit=(-30, -10), 
                               val_shift_limit=(-40, -10), p=0.8),
            A.GaussNoise(var_limit=(12, 28), p=0.7),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_yellow_leaf_curl_transform(self):
        """
        Yellow Leaf Curl Virus transformasyonu
        - Yaprak sararmasi ve kıvrılma
        - Büyüme geriliği, küçük yapraklar
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.1, 0.3), contrast=(0.8, 1.2), 
                         saturation=(0.8, 1.3), hue=(0.0, 0.2), p=0.9),
            A.HueSaturationValue(hue_shift_limit=(0, 40), 
                               sat_shift_limit=(-10, 20), 
                               val_shift_limit=(-20, 20), p=0.9),
            A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.6),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_mosaic_virus_transform(self):
        """
        Mosaic Virus transformasyonu
        - Yapraklarda mozaik desenli sarı-yeşil lekeler
        - Düzensiz renk dağılımı
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.1, 0.2), contrast=(0.9, 1.3), 
                         saturation=(0.7, 1.2), hue=(-0.1, 0.2), p=0.9),
            A.HueSaturationValue(hue_shift_limit=(-15, 35), 
                               sat_shift_limit=(-20, 15), 
                               val_shift_limit=(-25, 15), p=0.8),
            A.GaussNoise(var_limit=(8, 22), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), 
                                     contrast_limit=(0.1, 0.4), p=0.8),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_bacterial_spot_transform(self):
        """
        Bacterial Spot (Bakteriyel Leke) transformasyonu
        - Küçük, koyu, yağlı görünümlü lekeler
        - Sarı hale çevrili koyu lekeler
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.3, 0.0), contrast=(1.0, 1.4), 
                         saturation=(0.6, 1.0), hue=(-0.1, 0.15), p=0.8),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.0), 
                                     contrast_limit=(0.2, 0.5), p=0.9),
            A.HueSaturationValue(hue_shift_limit=(-15, 20), 
                               sat_shift_limit=(-25, 0), 
                               val_shift_limit=(-35, -5), p=0.8),
            A.GaussNoise(var_limit=(10, 25), p=0.7),
            A.Blur(blur_limit=(1, 3), p=0.4),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_healthy_transform(self):
        """
        Healthy (Sağlıklı) transformasyonu
        - Minimal değişiklikler, doğal görünümü korur
        - Sadece temel augmentasyonlar
        """
        return A.Compose([
            A.ColorJitter(brightness=(-0.1, 0.1), contrast=(0.9, 1.1), 
                         saturation=(0.9, 1.1), hue=(-0.05, 0.05), p=0.6),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), 
                                     contrast_limit=(0.0, 0.2), p=0.5),
            A.HueSaturationValue(hue_shift_limit=(-5, 5), 
                               sat_shift_limit=(-10, 10), 
                               val_shift_limit=(-10, 10), p=0.5),
            A.GaussNoise(var_limit=(0, 10), p=0.3),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5)
        ], bbox_params=BboxParams(format='yolo', label_fields=['class_labels']))

    def get_transform_for_disease(self, disease_type):
        """
        Hastalık tipine göre uygun transformasyonu döndürür
        
        Args:
            disease_type (str): Hastalık tipi
            
        Returns:
            albumentations.Compose: Transform pipeline
        """
        transform_map = {
            'early_blight': self.get_early_blight_transform,
            'late_blight': self.get_late_blight_transform,
            'leaf_mold': self.get_leaf_mold_transform,
            'septoria_leaf_spot': self.get_septoria_leaf_spot_transform,
            'spider_mites': self.get_spider_mites_transform,
            'target_spot': self.get_target_spot_transform,
            'yellow_leaf_curl': self.get_yellow_leaf_curl_transform,
            'mosaic_virus': self.get_mosaic_virus_transform,
            'bacterial_spot': self.get_bacterial_spot_transform,
            'healthy': self.get_healthy_transform
        }
        
        if disease_type not in transform_map:
            raise ValueError(f"Desteklenmeyen hastalık tipi: {disease_type}")
            
        return transform_map[disease_type]()

    def read_yolo_annotation(self, annotation_path):
        """
        YOLO formatındaki annotation dosyasını okur
        
        Args:
            annotation_path (str): Annotation dosya yolu
            
        Returns:
            list: Bounding box listesi
        """
        bboxes = []
        class_labels = []
        
        try:
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
        except Exception as e:
            self.logger.warning(f"Annotation okunamadı {annotation_path}: {e}")
            
        return bboxes, class_labels

    def save_yolo_annotation(self, annotation_path, bboxes, class_labels):
        """
        YOLO formatında annotation dosyasını kaydeder
        
        Args:
            annotation_path (str): Kayıt yolu
            bboxes (list): Bounding box listesi
            class_labels (list): Sınıf etiketleri
        """
        try:
            with open(annotation_path, 'w') as f:
                for bbox, class_label in zip(bboxes, class_labels):
                    line = f"{class_label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    f.write(line)
        except Exception as e:
            self.logger.error(f"Annotation kaydedilemedi {annotation_path}: {e}")

    def log_to_csv(self, csv_path, disease_type, image_path, status, error_message="", augmentation_count=0):
        """
        CSV dosyasına log kaydı yapar
        
        Args:
            csv_path (str): CSV dosya yolu
            disease_type (str): Hastalık tipi
            image_path (str): Görüntü dosya yolu
            status (str): İşlem durumu
            error_message (str): Hata mesajı
            augmentation_count (int): Augmentation sayısı
        """
        try:
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Başlık satırını yaz (dosya yoksa)
                if not file_exists:
                    writer.writerow(self.csv_headers)
                
                # Veri satırını yaz
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    disease_type,
                    image_path,
                    status,
                    error_message,
                    augmentation_count
                ])
        except Exception as e:
            self.logger.error(f"CSV log hatası: {e}")

    def is_image_compatible(self, image_path):
        """
        Görüntünün augmentation için uygun olup olmadığını kontrol eder
        
        Args:
            image_path (str): Görüntü dosya yolu
            
        Returns:
            bool: Uygunluk durumu
        """
        try:
            # Dosya varlığı kontrolü
            if not os.path.exists(image_path):
                return False
                
            # Görüntü okuma testi
            image = cv2.imread(image_path)
            if image is None:
                return False
                
            # Boyut kontrolü
            height, width = image.shape[:2]
            if height < 32 or width < 32:
                return False
                
            # Kanal kontrolü
            if len(image.shape) != 3:
                return False
                
            return True
            
        except Exception:
            return False

    def augment_disease(self, disease_type, input_dir, output_dir, num_augmentations=5):
        """
        Belirli bir hastalık için augmentation işlemi gerçekleştirir
        
        Args:
            disease_type (str): Hastalık tipi
            input_dir (str): Giriş dizini
            output_dir (str): Çıkış dizini
            num_augmentations (int): Her görüntü için augmentation sayısı
        """
        if disease_type not in self.supported_diseases:
            raise ValueError(f"Desteklenmeyen hastalık tipi: {disease_type}")
        
        # Çıkış dizinlerini oluştur
        images_output_dir = os.path.join(output_dir, 'images')
        labels_output_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)
        
        # CSV log dosyası
        csv_path = os.path.join(output_dir, f'{disease_type}_augmentation_log.csv')
        
        # Transform pipeline'ı al
        transform = self.get_transform_for_disease(disease_type)
        
        # Giriş görüntülerini bul
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        self.logger.info(f"{disease_type} için {len(image_files)} görüntü bulundu")
        
        processed_count = 0
        successful_count = 0
        
        for image_path in image_files:
            image_path = str(image_path)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Uygunluk kontrolü
            if not self.is_image_compatible(image_path):
                self.logger.warning(f"Uyumsuz görüntü atlandı: {image_path}")
                self.log_to_csv(csv_path, disease_type, image_path, 'SKIPPED', 'Incompatible image format or size')
                self.stats['skipped_images'] += 1
                continue
            
            try:
                # Görüntüyü oku
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Annotation dosyasını oku
                annotation_path = os.path.join(os.path.dirname(image_path), f'{image_name}.txt')
                bboxes, class_labels = self.read_yolo_annotation(annotation_path)
                
                # Augmentation işlemleri
                for i in range(num_augmentations):
                    try:
                        # Önce preprocess (sabit boyut letterbox)
                        pre = self.preprocess(
                            image=image_rgb,
                            bboxes=bboxes or [],
                            class_labels=class_labels or []
                        )
                        pre_image = pre['image']
                        pre_bboxes = pre['bboxes']
                        pre_labels = pre['class_labels']

                        # Boyut doğrulama (preprocess sonrası mutlaka 512x512 olmalı)
                        if not self._is_valid_size(pre_image, self.target_size):
                            self.logger.warning(f"Preprocess sonrası beklenmeyen boyut, atlandı: {image_path}")
                            self.log_to_csv(csv_path, disease_type, image_path, 'SKIPPED', 'Invalid size after preprocess')
                            self.stats['skipped_images'] += 1
                            continue

                        # BBox kırpma/filtreleme (preprocess sonrası)
                        if pre_bboxes:
                            pre_bboxes, pre_labels = self._clip_and_filter_bboxes(pre_bboxes, pre_labels)
                            if not pre_bboxes:
                                # Eğer tüm kutular elendiyse, etiketsiz devam edebiliriz
                                pre_labels = []

                        # Transform uygula
                        if pre_bboxes:
                            transformed = transform(image=pre_image, bboxes=pre_bboxes, class_labels=pre_labels)
                            augmented_image = transformed['image']
                            augmented_bboxes = transformed['bboxes']
                            augmented_labels = transformed['class_labels']
                        else:
                            transformed = transform(image=pre_image)
                            augmented_image = transformed['image']
                            augmented_bboxes = []
                            augmented_labels = []

                        # Boyut doğrulama (transform sonrası)
                        if not self._is_valid_size(augmented_image, self.target_size):
                            self.logger.warning(f"Transform sonrası beklenmeyen boyut, atlandı: {image_path}")
                            self.log_to_csv(csv_path, disease_type, image_path, 'SKIPPED', 'Invalid size after transform')
                            self.stats['skipped_images'] += 1
                            continue

                        # BBox kırpma/filtreleme (transform sonrası)
                        if augmented_bboxes:
                            augmented_bboxes, augmented_labels = self._clip_and_filter_bboxes(augmented_bboxes, augmented_labels)

                        # Augmented görüntüyü kaydet
                        output_image_name = f"{image_name}_{disease_type}_aug_{i+1}.jpg"
                        output_image_path = os.path.join(images_output_dir, output_image_name)
                        
                        augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(output_image_path, augmented_image_bgr)
                        
                        # Annotation kaydet
                        if augmented_bboxes:
                            output_annotation_name = f"{image_name}_{disease_type}_aug_{i+1}.txt"
                            output_annotation_path = os.path.join(labels_output_dir, output_annotation_name)
                            self.save_yolo_annotation(output_annotation_path, augmented_bboxes, augmented_labels)
                        
                        successful_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Augmentation hatası {image_path} (aug {i+1}): {e}")
                        self.stats['errors'] += 1
                
                # Başarılı işlem logu
                self.log_to_csv(csv_path, disease_type, image_path, 'SUCCESS', '', num_augmentations)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    self.logger.info(f"İşlenen: {processed_count}/{len(image_files)}")
                    
            except Exception as e:
                self.logger.error(f"Görüntü işleme hatası {image_path}: {e}")
                self.log_to_csv(csv_path, disease_type, image_path, 'ERROR', str(e))
                self.stats['errors'] += 1
        
        # İstatistikleri güncelle
        self.stats['total_processed'] += processed_count
        self.stats['successful_augmentations'] += successful_count
        
        # Özet rapor
        self.logger.info(f"\n=== {disease_type.upper()} AUGMENTATION ÖZET ===")
        self.logger.info(f"Toplam görüntü: {len(image_files)}")
        self.logger.info(f"İşlenen: {processed_count}")
        self.logger.info(f"Başarılı augmentation: {successful_count}")
        self.logger.info(f"Atlanan: {self.stats['skipped_images']}")
        self.logger.info(f"Hata: {self.stats['errors']}")
        self.logger.info(f"Çıkış dizini: {output_dir}")
        self.logger.info(f"Log dosyası: {csv_path}")

    def augment_all_diseases(self, base_input_dir, base_output_dir, num_augmentations=5):
        """
        Tüm desteklenen hastalıklar için augmentation işlemi gerçekleştirir
        
        Args:
            base_input_dir (str): Ana giriş dizini
            base_output_dir (str): Ana çıkış dizini
            num_augmentations (int): Her görüntü için augmentation sayısı
        """
        self.logger.info("=== TÜM DOMATES HASTALIKLARI AUGMENTATION BAŞLADI ===")
        
        # İstatistikleri sıfırla
        self.stats = {
            'total_processed': 0,
            'successful_augmentations': 0,
            'skipped_images': 0,
            'errors': 0
        }
        
        for disease in self.supported_diseases:
            disease_input_dir = os.path.join(base_input_dir, disease)
            disease_output_dir = os.path.join(base_output_dir, disease)
            
            if os.path.exists(disease_input_dir):
                self.logger.info(f"\n--- {disease.upper()} işleniyor ---")
                self.augment_disease(disease, disease_input_dir, disease_output_dir, num_augmentations)
            else:
                self.logger.warning(f"Dizin bulunamadı: {disease_input_dir}")
        
        # Genel özet rapor
        self.logger.info(f"\n=== GENEL AUGMENTATION ÖZET ===")
        self.logger.info(f"Toplam işlenen görüntü: {self.stats['total_processed']}")
        self.logger.info(f"Toplam başarılı augmentation: {self.stats['successful_augmentations']}")
        self.logger.info(f"Toplam atlanan: {self.stats['skipped_images']}")
        self.logger.info(f"Toplam hata: {self.stats['errors']}")
        
        # Genel log dosyası
        general_log_path = os.path.join(base_output_dir, 'all_diseases_summary.json')
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'diseases_processed': self.supported_diseases,
            'statistics': self.stats,
            'augmentations_per_image': num_augmentations
        }
        
        try:
            with open(general_log_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Genel özet raporu kaydedildi: {general_log_path}")
        except Exception as e:
            self.logger.error(f"Özet raporu kaydedilemedi: {e}")


# Kullanım örnekleri
if __name__ == "__main__":
    # Augmentation sınıfını oluştur
    augmenter = TomatoDiseaseAugmentation()
    
    # Tek hastalık için augmentation
    # augmenter.augment_disease(
    #     disease_type='early_blight',
    #     input_dir='data/tomato_diseases/early_blight',
    #     output_dir='data/augmented/early_blight',
    #     num_augmentations=5
    # )
    
    # Tüm hastalıklar için augmentation
    # augmenter.augment_all_diseases(
    #     base_input_dir='data/tomato_diseases',
    #     base_output_dir='data/augmented',
    #     num_augmentations=3
    # )
