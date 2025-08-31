"""
Domates Zararlıları Augmentation Sistemi
========================================

Bu modül, domates zararlıları için özel augmentation işlemleri gerçekleştirir.
10 farklı domates zararlısı için gerçekçi görsel transformasyonlar uygular.

Desteklenen Zararlılar:
- Whitefly (Beyaz Sinek) - Bemisia tabaci
- Aphid (Yaprak Biti) - Aphis gossypii  
- Thrips (Trips) - Frankliniella occidentalis
- Spider Mite (Kırmızı Örümcek) - Tetranychus urticae
- Hornworm (Tütün Kurdu) - Manduca sexta
- Cutworm (Kök Kurdu) - Agrotis spp.
- Leafhopper (Yaprak Piresi) - Empoasca spp.
- Flea Beetle (Pire Böceği) - Epitrix spp.
- Leaf Miner (Yaprak Madencisi) - Liriomyza spp.
- Stink Bug (Kokarca) - Nezara viridula

Kullanım:
    augmenter = TomatoPestAugmentation('input_images', 'input_labels', 'output_images', 'output_labels')
    augmenter.augment_pest('whitefly', multiplier=5)
"""

import albumentations as A
import cv2
import numpy as np
import os
import json
import csv
from pathlib import Path
import random
from datetime import datetime
import logging

class TomatoPestAugmentation:
    def __init__(self, images_dir, labels_dir, output_images_dir, output_labels_dir, log_level=logging.INFO):
        """
        Domates zararlıları augmentation sınıfı
        
        Args:
            images_dir: Giriş görüntü dizini
            labels_dir: Giriş etiket dizini
            output_images_dir: Çıkış görüntü dizini
            output_labels_dir: Çıkış etiket dizini
            log_level: Logging seviyesi
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir) 
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        # Çıkış dizinlerini oluştur
        self.output_images_dir.mkdir(exist_ok=True, parents=True)
        self.output_labels_dir.mkdir(exist_ok=True, parents=True)
        
        # Logging kurulumu
        self.setup_logging(log_level)
        
        # Desteklenen zararlılar
        self.supported_pests = [
            'whitefly', 'aphid', 'thrips', 'spider_mite', 'hornworm',
            'cutworm', 'leafhopper', 'flea_beetle', 'leaf_miner', 'stink_bug'
        ]
        
        # CSV raporlama için başlıklar
        self.csv_headers = [
            'timestamp', 'pest_type', 'image_path', 'status', 
            'error_message', 'augmentation_count', 'bbox_count'
        ]
        
        # İstatistikler
        self.stats = {
            'total_processed': 0,
            'successful_augmentations': 0,
            'skipped_images': 0,
            'errors': 0
        }
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
            x_center, y_center, width, height = bbox
            
            # Önce tüm değerleri [0,1] aralığına zorla kırp
            x_center = max(0.0, min(1.0, float(x_center)))
            y_center = max(0.0, min(1.0, float(y_center)))
            width = max(0.0, min(1.0, float(width)))
            height = max(0.0, min(1.0, float(height)))
            
            # Minimum boyut kontrolü
            if width < 1e-4 or height < 1e-4:
                continue
                
            # Bbox'un görüntü sınırları içinde kalmasını sağla
            x1 = x_center - width / 2.0
            y1 = y_center - height / 2.0
            x2 = x_center + width / 2.0
            y2 = y_center + height / 2.0
            
            # Sınırları kırp
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(1.0, x2)
            y2 = min(1.0, y2)
            
            # Yeni boyutları hesapla
            new_width = x2 - x1
            new_height = y2 - y1
            
            # Çok küçükse atla
            if new_width < 1e-4 or new_height < 1e-4:
                continue
                
            # Yeni merkezi hesapla
            new_x_center = (x1 + x2) / 2.0
            new_y_center = (y1 + y2) / 2.0
            
            # Son kontrol: tüm değerlerin [0,1] aralığında olduğundan emin ol
            new_x_center = max(0.0, min(1.0, new_x_center))
            new_y_center = max(0.0, min(1.0, new_y_center))
            new_width = max(0.0, min(1.0, new_width))
            new_height = max(0.0, min(1.0, new_height))
            
            clipped.append([new_x_center, new_y_center, new_width, new_height])
            labels.append(cid)
        return clipped, labels

    def setup_logging(self, log_level):
        """Logging konfigürasyonu"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('tomato_pest_augmentation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_whitefly_transforms(self):
        """Beyaz sinek (Whitefly) - Bemisia tabaci"""
        return A.Compose([
            # Küçük beyaz noktalar için kontrast artırma
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.25),  # Beyaz sinekler daha görünür
                    contrast_limit=(0.2, 0.4),     # Yüksek kontrast
                    p=0.7
                ),
            ], p=0.8),
            
            # Beyaz noktaları vurgulama
            A.OneOf([
                A.Sharpen(alpha=(0.3, 0.6), lightness=(1.1, 1.4), p=0.6),
                A.UnsharpMask(
                    blur_limit=(2, 4),
                    sigma_limit=(0.8, 1.5),
                    alpha=(0.3, 0.5),
                    threshold=5,
                    p=0.4
                ),
            ], p=0.7),
            
            # Hafif noise (sinek hareketi)
            A.OneOf([
                A.GaussNoise(var_limit=(3, 8), p=0.3),
                A.ISONoise(
                    color_shift=(0.005, 0.02),
                    intensity=(0.05, 0.15),
                    p=0.3
                ),
            ], p=0.4),
            
            # Geometrik transformasyonlar (sinekler hareket halinde)
            A.OneOf([
                A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.6),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    p=0.5
                ),
            ], p=0.8),
            
            # Hafif blur (hareket etkisi)
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 5), p=0.2),
                A.GaussianBlur(blur_limit=(1, 2), p=0.2),
            ], p=0.3),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.2  # Çok küçük sinekler için düşük threshold
        ))
    
    def get_aphid_transforms(self):
        """Yaprak biti (Aphid) - Aphis gossypii"""
        return A.Compose([
            # Yeşil-siyah küçük kümeler
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(-10, 10),    # Hafif renk varyasyonu
                    sat_shift_limit=(5, 25),      # Doygunluk artışı
                    val_shift_limit=(-15, 5),     # Hafif koyuluk
                    p=0.7
                ),
                A.ColorJitter(
                    brightness=(-0.1, 0.1),
                    contrast=(1.1, 1.3),          # Orta kontrast
                    saturation=(1.0, 1.3),
                    hue=(-0.03, 0.03),
                    p=0.6
                ),
            ], p=0.8),
            
            # Küme halindeki yaprak bitleri için doku
            A.OneOf([
                A.GaussNoise(var_limit=(5, 12), p=0.4),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    elementwise=True,
                    p=0.3
                ),
            ], p=0.5),
            
            # Detay keskinleştirme
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.4), lightness=(0.9, 1.2), p=0.5),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(6, 6), p=0.4),
            ], p=0.6),
            
            # Küçük hareket
            A.OneOf([
                A.Rotate(limit=10, p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=8,
                    p=0.4
                ),
            ], p=0.6),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.25
        ))
    
    def get_thrips_transforms(self):
        """Trips (Thrips) - Frankliniella occidentalis"""
        return A.Compose([
            # Küçük, ince, hızlı hareket eden böcekler
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(-5, 15),     # Sarı-kahve tonları
                    sat_shift_limit=(0, 20),      # Orta doygunluk
                    val_shift_limit=(-10, 10),    # Değişken parlaklık
                    p=0.7
                ),
                A.ColorJitter(
                    brightness=(-0.08, 0.15),
                    contrast=(1.0, 1.25),
                    saturation=(0.9, 1.2),
                    hue=(-0.02, 0.04),
                    p=0.6
                ),
            ], p=0.8),
            
            # Hareket bulanıklığı (hızlı hareket)
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7), p=0.4),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                A.Defocus(radius=(1, 2), alias_blur=(0.1, 0.2), p=0.2),
            ], p=0.6),
            
            # İnce çizgi benzeri form için keskinleştirme
            A.OneOf([
                A.Sharpen(alpha=(0.25, 0.5), lightness=(0.9, 1.1), p=0.5),
                A.UnsharpMask(
                    blur_limit=(2, 4),
                    sigma_limit=(0.8, 1.2),
                    alpha=(0.2, 0.4),
                    threshold=8,
                    p=0.4
                ),
            ], p=0.6),
            
            # Dinamik hareket
            A.OneOf([
                A.Rotate(limit=25, p=0.6),
                A.ShiftScaleRotate(
                    shift_limit=0.12,
                    scale_limit=0.15,
                    rotate_limit=20,
                    p=0.5
                ),
            ], p=0.8),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.15  # Çok hızlı hareket eden böcekler
        ))
    
    def get_spider_mite_transforms(self):
        """Kırmızı örümcek (Spider Mite) - Tetranychus urticae"""
        return A.Compose([
            # Çok küçük, kırmızımsı noktalar
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(-15, 5),     # Kırmızımsı tonlar
                    sat_shift_limit=(15, 40),     # Yüksek doygunluk
                    val_shift_limit=(-5, 15),     # Hafif parlaklık
                    p=0.8
                ),
                A.ColorJitter(
                    brightness=(-0.05, 0.12),
                    contrast=(1.2, 1.5),          # Yüksek kontrast (küçük noktalar)
                    saturation=(1.2, 1.6),       # Kırmızı vurgu
                    hue=(-0.05, 0.02),
                    p=0.7
                ),
            ], p=0.9),
            
            # Çok küçük noktalar için maksimum keskinleştirme
            A.OneOf([
                A.Sharpen(alpha=(0.4, 0.7), lightness=(0.8, 1.0), p=0.7),
                A.CLAHE(clip_limit=5.0, tile_grid_size=(10, 10), p=0.5),
                A.UnsharpMask(
                    blur_limit=(1, 3),
                    sigma_limit=(0.5, 1.0),
                    alpha=(0.4, 0.6),
                    threshold=3,
                    p=0.4
                ),
            ], p=0.8),
            
            # İnce web dokusu efekti
            A.OneOf([
                A.GaussNoise(var_limit=(2, 6), p=0.3),
                A.ISONoise(
                    color_shift=(0.005, 0.015),
                    intensity=(0.03, 0.1),
                    p=0.3
                ),
            ], p=0.4),
            
            # Minimal hareket (çok küçük)
            A.OneOf([
                A.Rotate(limit=8, p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.08,
                    rotate_limit=5,
                    p=0.4
                ),
            ], p=0.5),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.1  # En küçük zararlılar
        ))

    def get_hornworm_transforms(self):
        """Tütün kurdu (Hornworm) - Manduca sexta"""
        return A.Compose([
            # Büyük, yeşil tırtıl
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(-10, 20),    # Yeşil tonları
                    sat_shift_limit=(10, 30),     # Orta-yüksek doygunluk
                    val_shift_limit=(-8, 12),     # Hafif değişim
                    p=0.8
                ),
                A.ColorJitter(
                    brightness=(-0.1, 0.1),
                    contrast=(1.0, 1.2),
                    saturation=(1.1, 1.4),       # Yeşil vurgu
                    hue=(-0.03, 0.06),
                    p=0.7
                ),
            ], p=0.8),
            
            # Tırtıl dokusu
            A.OneOf([
                A.GaussNoise(var_limit=(3, 8), p=0.4),
                A.MultiplicativeNoise(
                    multiplier=(0.95, 1.05),
                    elementwise=True,
                    p=0.3
                ),
            ], p=0.5),
            
            # Şekil detayları için hafif keskinleştirme
            A.OneOf([
                A.Sharpen(alpha=(0.15, 0.35), lightness=(0.9, 1.1), p=0.5),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(4, 4), p=0.4),
            ], p=0.6),
            
            # Büyük böcek için normal hareketler
            A.OneOf([
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.12,
                    rotate_limit=12,
                    p=0.4
                ),
            ], p=0.7),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.4
        ))
    
    def get_cutworm_transforms(self):
        """Kök kurdu (Cutworm) - Agrotis spp."""
        return A.Compose([
            # Kahverengi-gri tırtıl (toprakta yaşar)
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(-5, 15),     # Kahve-gri tonlar
                    sat_shift_limit=(-10, 15),    # Düşük-orta doygunluk
                    val_shift_limit=(-15, 5),     # Koyu tonlar
                    p=0.8
                ),
                A.ColorJitter(
                    brightness=(-0.15, 0.05),
                    contrast=(1.0, 1.3),
                    saturation=(0.8, 1.2),
                    hue=(-0.02, 0.04),
                    p=0.7
                ),
            ], p=0.8),
            
            # Toprak parçacıkları efekti
            A.OneOf([
                A.GaussNoise(var_limit=(8, 18), p=0.5),
                A.ISONoise(
                    color_shift=(0.01, 0.04),
                    intensity=(0.1, 0.25),
                    p=0.4
                ),
            ], p=0.6),
            
            # Hafif bulanıklık (toprağa gizlenmiş)
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 2), p=0.3),
                A.Defocus(radius=(1, 2), alias_blur=(0.05, 0.15), p=0.2),
            ], p=0.4),
            
            # Orta derecede hareket
            A.OneOf([
                A.Rotate(limit=12, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.06,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=0.4
                ),
            ], p=0.6),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def get_leafhopper_transforms(self):
        """Yaprak piresi (Leafhopper) - Empoasca spp."""
        return A.Compose([
            # Küçük, yeşil, zıplayan böcekler
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(-8, 25),     # Yeşil-sarı tonları
                    sat_shift_limit=(10, 30),     # Orta doygunluk
                    val_shift_limit=(-5, 15),     # Hafif değişim
                    p=0.8
                ),
                A.ColorJitter(
                    brightness=(-0.05, 0.15),
                    contrast=(1.1, 1.3),
                    saturation=(1.0, 1.3),
                    hue=(-0.02, 0.08),
                    p=0.7
                ),
            ], p=0.8),
            
            # Zıplama hareketi bulanıklığı
            A.OneOf([
                A.MotionBlur(blur_limit=(4, 8), p=0.5),
                A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            ], p=0.6),
            
            # Küçük böcek için keskinleştirme
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.45), lightness=(0.9, 1.2), p=0.5),
                A.CLAHE(clip_limit=3.5, tile_grid_size=(6, 6), p=0.4),
            ], p=0.6),
            
            # Dinamik hareket (zıplama)
            A.OneOf([
                A.Rotate(limit=20, p=0.6),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.2,
                    rotate_limit=18,
                    p=0.5
                ),
            ], p=0.8),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.2
        ))
    
    def get_flea_beetle_transforms(self):
        """Pire böceği (Flea Beetle) - Epitrix spp."""
        return A.Compose([
            # Çok küçük, siyah, zıplayan böcekler
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(-10, 10),    # Siyah-koyu tonlar
                    sat_shift_limit=(-5, 15),     # Düşük-orta doygunluk
                    val_shift_limit=(-25, -5),    # Koyu renkler
                    p=0.8
                ),
                A.ColorJitter(
                    brightness=(-0.2, 0.05),
                    contrast=(1.3, 1.6),          # Çok yüksek kontrast
                    saturation=(0.8, 1.2),
                    hue=(-0.03, 0.03),
                    p=0.7
                ),
            ], p=0.9),
            
            # Maksimum keskinleştirme (çok küçük böcekler)
            A.OneOf([
                A.Sharpen(alpha=(0.5, 0.8), lightness=(0.7, 1.0), p=0.7),
                A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.6),
                A.UnsharpMask(
                    blur_limit=(1, 2),
                    sigma_limit=(0.5, 1.0),
                    alpha=(0.5, 0.7),
                    threshold=2,
                    p=0.4
                ),
            ], p=0.8),
            
            # Hızlı hareket bulanıklığı
            A.OneOf([
                A.MotionBlur(blur_limit=(5, 10), p=0.5),
                A.GaussianBlur(blur_limit=(1, 2), p=0.3),
            ], p=0.6),
            
            # Çok dinamik hareket
            A.OneOf([
                A.Rotate(limit=30, p=0.7),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.25,
                    rotate_limit=25,
                    p=0.6
                ),
            ], p=0.9),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.1  # En küçük ve hızlı böcekler
        ))
    
    def get_leaf_miner_transforms(self):
        """Yaprak madencisi (Leaf Miner) - Liriomyza spp."""
        return A.Compose([
            # Yaprak içi tüneller (beyazımsı çizgiler)
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(10, 25),     # Sarı-beyaz tonları
                    sat_shift_limit=(-30, -10),   # Düşük doygunluk
                    val_shift_limit=(10, 25),     # Açık renkler
                    p=0.8
                ),
                A.ColorJitter(
                    brightness=(0.1, 0.25),
                    contrast=(1.0, 1.3),
                    saturation=(0.5, 0.8),       # Çok düşük doygunluk
                    hue=(0.03, 0.08),
                    p=0.7
                ),
            ], p=0.8),
            
            # Çizgi/tünel dokusu
            A.OneOf([
                A.GaussNoise(var_limit=(4, 10), p=0.4),
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    elementwise=True,
                    p=0.3
                ),
            ], p=0.5),
            
            # Çizgi keskinleştirme
            A.OneOf([
                A.Sharpen(alpha=(0.3, 0.6), lightness=(1.0, 1.3), p=0.6),
                A.UnsharpMask(
                    blur_limit=(2, 4),
                    sigma_limit=(0.8, 1.5),
                    alpha=(0.3, 0.5),
                    threshold=5,
                    p=0.4
                ),
            ], p=0.7),
            
            # Çizgisel hareket
            A.OneOf([
                A.Rotate(limit=5, p=0.4),       # Minimal rotasyon
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.08,
                    rotate_limit=3,
                    p=0.3
                ),
            ], p=0.5),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def get_stink_bug_transforms(self):
        """Kokarca (Stink Bug) - Nezara viridula"""
        return A.Compose([
            # Orta büyüklükte, yeşil-kahve böcek
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=(-15, 20),    # Yeşil-kahve aralığı
                    sat_shift_limit=(5, 25),      # Orta doygunluk
                    val_shift_limit=(-10, 10),    # Değişken parlaklık
                    p=0.8
                ),
                A.ColorJitter(
                    brightness=(-0.1, 0.1),
                    contrast=(1.0, 1.25),
                    saturation=(0.9, 1.3),
                    hue=(-0.05, 0.06),
                    p=0.7
                ),
            ], p=0.8),
            
            # Böcek kabuğu dokusu
            A.OneOf([
                A.GaussNoise(var_limit=(4, 10), p=0.4),
                A.MultiplicativeNoise(
                    multiplier=(0.95, 1.05),
                    elementwise=True,
                    p=0.3
                ),
            ], p=0.5),
            
            # Orta derecede keskinleştirme
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.4), lightness=(0.9, 1.1), p=0.5),
                A.CLAHE(clip_limit=2.8, tile_grid_size=(5, 5), p=0.4),
            ], p=0.6),
            
            # Normal böcek hareketleri
            A.OneOf([
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.12,
                    rotate_limit=12,
                    p=0.4
                ),
            ], p=0.7),
            
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.35
        ))

    def get_pest_transforms(self, pest_type):
        """Zararlı türüne göre transform pipeline döndür"""
        transform_methods = {
            'whitefly': self.get_whitefly_transforms,
            'aphid': self.get_aphid_transforms,
            'thrips': self.get_thrips_transforms,
            'spider_mite': self.get_spider_mite_transforms,
            'hornworm': self.get_hornworm_transforms,
            'cutworm': self.get_cutworm_transforms,
            'leafhopper': self.get_leafhopper_transforms,
            'flea_beetle': self.get_flea_beetle_transforms,
            'leaf_miner': self.get_leaf_miner_transforms,
            'stink_bug': self.get_stink_bug_transforms
        }
        
        if pest_type not in transform_methods:
            raise ValueError(f"Desteklenmeyen zararlı türü: {pest_type}. Desteklenen türler: {list(transform_methods.keys())}")
        
        return transform_methods[pest_type]()

    def get_pest_size_category_transforms(self, size_category):
        """Zararlı büyüklük kategorisine göre augmentation"""
        if size_category == 'very_small':  # Çok küçük (whitefly, spider_mite, flea_beetle)
            return A.Compose([
                A.CLAHE(clip_limit=6.0, tile_grid_size=(10, 10), p=0.8),
                A.Sharpen(alpha=(0.5, 0.8), lightness=(0.7, 1.0), p=0.8),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3),
                    contrast_limit=(0.3, 0.6),
                    p=0.9
                ),
                A.OneOf([
                    A.MotionBlur(blur_limit=(3, 8), p=0.4),
                    A.GaussianBlur(blur_limit=(1, 2), p=0.3),
                ], p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.2,
                    rotate_limit=25,
                    p=0.8
                ),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.1
            ))
        
        elif size_category == 'small':  # Küçük (aphid, thrips, leafhopper, leaf_miner)
            return A.Compose([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.7),
                A.Sharpen(alpha=(0.3, 0.6), lightness=(0.8, 1.2), p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.05, 0.2),
                    contrast_limit=(0.2, 0.4),
                    p=0.8
                ),
                A.OneOf([
                    A.MotionBlur(blur_limit=(4, 7), p=0.3),
                    A.GaussianBlur(blur_limit=(1, 3), p=0.2),
                ], p=0.4),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=20,
                    p=0.7
                ),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.2
            ))
        
        elif size_category == 'medium':  # Orta (cutworm, stink_bug)
            return A.Compose([
                A.CLAHE(clip_limit=3.0, tile_grid_size=(6, 6), p=0.6),
                A.Sharpen(alpha=(0.2, 0.4), lightness=(0.9, 1.1), p=0.6),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.15),
                    contrast_limit=(0.1, 0.3),
                    p=0.7
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(1, 2), p=0.2),
                    A.Defocus(radius=(1, 2), alias_blur=(0.05, 0.15), p=0.2),
                ], p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.12,
                    rotate_limit=15,
                    p=0.6
                ),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        
        elif size_category == 'large':  # Büyük (hornworm)
            return A.Compose([
                A.CLAHE(clip_limit=2.5, tile_grid_size=(4, 4), p=0.5),
                A.Sharpen(alpha=(0.15, 0.35), lightness=(0.9, 1.1), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1),
                    contrast_limit=(0.05, 0.25),
                    p=0.6
                ),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(1, 2), p=0.1),
                    A.Defocus(radius=(1, 2), alias_blur=(0.02, 0.1), p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.06,
                    scale_limit=0.1,
                    rotate_limit=12,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.4
            ))
        
        else:
            raise ValueError(f"Desteklenmeyen büyüklük kategorisi: {size_category}. Desteklenen: very_small, small, medium, large")

    def read_yolo_annotation(self, label_path):
        """YOLO formatında etiket dosyasını oku"""
        try:
            bboxes = []
            class_labels = []
            
            if not label_path.exists():
                self.logger.warning(f"Etiket dosyası bulunamadı: {label_path}")
                return bboxes, class_labels
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
            
            return bboxes, class_labels
        
        except Exception as e:
            self.logger.error(f"Etiket okuma hatası {label_path}: {str(e)}")
            return [], []

    def save_yolo_annotation(self, label_path, bboxes, class_labels):
        """YOLO formatında etiket dosyasını kaydet"""
        try:
            with open(label_path, 'w') as f:
                for bbox, class_id in zip(bboxes, class_labels):
                    f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        except Exception as e:
            self.logger.error(f"Etiket kaydetme hatası {label_path}: {str(e)}")

    def log_to_csv(self, pest_type, image_path, status, error_message="", augmentation_count=0, bbox_count=0):
        """CSV dosyasına log kaydet"""
        csv_path = self.output_images_dir.parent / f"tomato_pest_augmentation_{pest_type}.csv"
        
        # CSV dosyası yoksa başlık satırını ekle
        write_header = not csv_path.exists()
        
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(self.csv_headers)
                
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    pest_type,
                    str(image_path),
                    status,
                    error_message,
                    augmentation_count,
                    bbox_count
                ])
        except Exception as e:
            self.logger.error(f"CSV log hatası: {str(e)}")

    def augment_pest(self, pest_type, multiplier=3, max_images=None):
        """Belirli bir zararlı türü için augmentation gerçekleştir"""
        if pest_type not in self.supported_pests:
            raise ValueError(f"Desteklenmeyen zararlı türü: {pest_type}")
        
        self.logger.info(f"🐛 {pest_type.upper()} zararlısı için augmentation başlatılıyor...")
        self.logger.info(f"📊 Multiplier: {multiplier}, Max images: {max_images}")
        
        # Transform pipeline al
        transform = self.get_pest_transforms(pest_type)
        
        # Görüntü dosyalarını bul
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(ext))
            image_files.extend(self.images_dir.glob(ext.upper()))
        
        if max_images:
            image_files = image_files[:max_images]
        
        self.logger.info(f"📁 {len(image_files)} görüntü dosyası bulundu")
        
        processed_count = 0
        successful_count = 0
        
        for image_path in image_files:
            try:
                # Etiket dosyası yolu
                label_path = self.labels_dir / f"{image_path.stem}.txt"
                
                # Görüntüyü oku
                image = cv2.imread(str(image_path))
                if image is None:
                    self.logger.warning(f"Görüntü okunamadı: {image_path}")
                    self.log_to_csv(pest_type, image_path, "SKIPPED", "Görüntü okunamadı")
                    continue
                
                # RGB'ye çevir
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Etiketleri oku
                bboxes, class_labels = self.read_yolo_annotation(label_path)
                
                if not bboxes:
                    self.logger.warning(f"Etiket bulunamadı: {label_path}")
                    self.log_to_csv(pest_type, image_path, "SKIPPED", "Etiket bulunamadı")
                    continue
                
                # Augmentation uygula
                for i in range(multiplier):
                    try:
                        # Giriş bboxlarını doğrula/temizle
                        bboxes_valid, labels_valid = self._clip_and_filter_bboxes(bboxes, class_labels)
                        if not bboxes_valid:
                            self.log_to_csv(pest_type, image_path, "SKIPPED", "Geçerli bbox bulunamadı (giriş)")
                            continue

                        # Önce preprocess (sabit boyut letterbox)
                        pre = self.preprocess(image=image, bboxes=bboxes_valid, class_labels=labels_valid)
                        pre_image, pre_bboxes, pre_labels = pre['image'], pre['bboxes'], pre['class_labels']
                        # Preprocess sonrası boyut doğrulaması
                        if pre_image is None:
                            self.log_to_csv(pest_type, image_path, "ERROR", "Preprocess sonrası görüntü None")
                            continue
                        if pre_image.shape[:2] != (self.target_size, self.target_size):
                            self.logger.warning(f"Preprocess boyut hatası: {image_path} - Beklenen: {(self.target_size, self.target_size)}, Gerçek: {pre_image.shape[:2]}")
                            self.log_to_csv(pest_type, image_path, "ERROR", f"Preprocess boyutu: beklenen {self.target_size}x{self.target_size}, gerçek {pre_image.shape[:2]}")
                            continue

                        # Transform uygula
                        transformed = transform(
                            image=pre_image,
                            bboxes=pre_bboxes,
                            class_labels=pre_labels
                        )
                        # Çıkış doğrulama
                        out_img = transformed['image']
                        out_bboxes, out_labels = self._clip_and_filter_bboxes(transformed['bboxes'], transformed['class_labels'])
                        if out_img is None:
                            self.log_to_csv(pest_type, image_path, "ERROR", "Transform sonrası görüntü None")
                            continue
                        if out_img.shape[:2] != (self.target_size, self.target_size):
                            self.logger.warning(f"Transform boyut hatası: {image_path} - Beklenen: {(self.target_size, self.target_size)}, Gerçek: {out_img.shape[:2]}")
                            self.log_to_csv(pest_type, image_path, "ERROR", f"Transform boyutu: beklenen {self.target_size}x{self.target_size}, gerçek {out_img.shape[:2]}")
                            continue
                        if not out_bboxes:
                            self.log_to_csv(pest_type, image_path, "SKIPPED", "Geçerli bbox bulunamadı (çıkış)")
                            continue
                        
                        # Çıkış dosya adları
                        output_image_name = f"{image_path.stem}_{pest_type}_aug_{i+1}{image_path.suffix}"
                        output_label_name = f"{image_path.stem}_{pest_type}_aug_{i+1}.txt"
                        
                        output_image_path = self.output_images_dir / output_image_name
                        output_label_path = self.output_labels_dir / output_label_name
                        
                        # Görüntüyü kaydet
                        transformed_image = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_image_path), transformed_image)
                        
                        # Etiketleri kaydet
                        self.save_yolo_annotation(
                            output_label_path,
                            out_bboxes,
                            out_labels
                        )
                        
                        successful_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Augmentation hatası {image_path} (iter {i+1}): {str(e)}")
                        self.log_to_csv(pest_type, image_path, "ERROR", str(e))
                
                processed_count += 1
                self.log_to_csv(
                    pest_type, image_path, "SUCCESS", "",
                    multiplier, len(bboxes)
                )
                
                if processed_count % 10 == 0:
                    self.logger.info(f"📈 İşlenen: {processed_count}/{len(image_files)}")
                
            except Exception as e:
                self.logger.error(f"Genel hata {image_path}: {str(e)}")
                self.log_to_csv(pest_type, image_path, "ERROR", str(e))
        
        # İstatistikleri güncelle
        self.stats['total_processed'] += processed_count
        self.stats['successful_augmentations'] += successful_count
        self.stats['skipped_images'] += len(image_files) - processed_count
        
        self.logger.info(f"✅ {pest_type.upper()} augmentation tamamlandı!")
        self.logger.info(f"📊 İşlenen görüntü: {processed_count}")
        self.logger.info(f"🎯 Başarılı augmentation: {successful_count}")
        
        return {
            'pest_type': pest_type,
            'processed_images': processed_count,
            'successful_augmentations': successful_count,
            'total_images_found': len(image_files)
        }

    def augment_all_pests(self, multiplier=3, max_images_per_pest=None):
        """Tüm zararlı türleri için toplu augmentation"""
        self.logger.info("🚀 Tüm domates zararlıları için toplu augmentation başlatılıyor...")
        
        results = {}
        total_start_time = datetime.now()
        
        for pest_type in self.supported_pests:
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"🐛 {pest_type.upper()} işleniyor...")
                
                result = self.augment_pest(pest_type, multiplier, max_images_per_pest)
                results[pest_type] = result
                
            except Exception as e:
                self.logger.error(f"❌ {pest_type} augmentation hatası: {str(e)}")
                results[pest_type] = {
                    'pest_type': pest_type,
                    'processed_images': 0,
                    'successful_augmentations': 0,
                    'total_images_found': 0,
                    'error': str(e)
                }
        
        total_end_time = datetime.now()
        total_duration = total_end_time - total_start_time
        
        # Genel özet
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📋 TOPLU AUGMENTATION ÖZET")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"⏱️  Toplam süre: {total_duration}")
        self.logger.info(f"📊 Toplam işlenen görüntü: {self.stats['total_processed']}")
        self.logger.info(f"🎯 Toplam başarılı augmentation: {self.stats['successful_augmentations']}")
        self.logger.info(f"⚠️  Atlanan görüntü: {self.stats['skipped_images']}")
        self.logger.info(f"❌ Hata sayısı: {self.stats['errors']}")
        
        # Zararlı bazında özet
        for pest_type, result in results.items():
            if 'error' not in result:
                self.logger.info(f"  • {pest_type}: {result['successful_augmentations']} augmentation")
            else:
                self.logger.info(f"  • {pest_type}: HATA - {result['error']}")
        
        return results

    def augment_by_size_category(self, size_category, multiplier=3, max_images=None):
        """Büyüklük kategorisine göre augmentation"""
        size_pest_mapping = {
            'very_small': ['whitefly', 'spider_mite', 'flea_beetle'],
            'small': ['aphid', 'thrips', 'leafhopper', 'leaf_miner'],
            'medium': ['cutworm', 'stink_bug'],
            'large': ['hornworm']
        }
        
        if size_category not in size_pest_mapping:
            raise ValueError(f"Desteklenmeyen büyüklük kategorisi: {size_category}")
        
        self.logger.info(f"📏 {size_category.upper()} kategorisi zararlıları için augmentation başlatılıyor...")
        
        # Kategori-spesifik transform al
        transform = self.get_pest_size_category_transforms(size_category)
        pests_in_category = size_pest_mapping[size_category]
        
        self.logger.info(f"🐛 Bu kategorideki zararlılar: {', '.join(pests_in_category)}")
        
        # Görüntü dosyalarını bul
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.images_dir.glob(ext))
            image_files.extend(self.images_dir.glob(ext.upper()))
        
        if max_images:
            image_files = image_files[:max_images]
        
        self.logger.info(f"📁 {len(image_files)} görüntü dosyası bulundu")
        
        processed_count = 0
        successful_count = 0
        
        for image_path in image_files:
            try:
                # Etiket dosyası yolu
                label_path = self.labels_dir / f"{image_path.stem}.txt"
                
                # Görüntüyü oku
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # RGB'ye çevir
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Etiketleri oku
                bboxes, class_labels = self.read_yolo_annotation(label_path)
                
                if not bboxes:
                    continue
                
                # Augmentation uygula
                for i in range(multiplier):
                    try:
                        # Önce preprocess (sabit boyut letterbox)
                        pre = self.preprocess(image=image, bboxes=bboxes or [], class_labels=class_labels or [])
                        pre_image, pre_bboxes, pre_labels = pre['image'], pre['bboxes'], pre['class_labels']

                        # Transform uygula
                        transformed = transform(
                            image=pre_image,
                            bboxes=pre_bboxes,
                            class_labels=pre_labels
                        )
                        
                        # Çıkış dosya adları
                        output_image_name = f"{image_path.stem}_{size_category}_aug_{i+1}{image_path.suffix}"
                        output_label_name = f"{image_path.stem}_{size_category}_aug_{i+1}.txt"
                        
                        output_image_path = self.output_images_dir / output_image_name
                        output_label_path = self.output_labels_dir / output_label_name
                        
                        # Görüntüyü kaydet
                        transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(output_image_path), transformed_image)
                        
                        # Etiketleri kaydet
                        self.save_yolo_annotation(
                            output_label_path,
                            transformed['bboxes'],
                            transformed['class_labels']
                        )
                        
                        successful_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"Augmentation hatası {image_path} (iter {i+1}): {str(e)}")
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    self.logger.info(f"📈 İşlenen: {processed_count}/{len(image_files)}")
                
            except Exception as e:
                self.logger.error(f"Genel hata {image_path}: {str(e)}")
        
        self.logger.info(f"✅ {size_category.upper()} kategori augmentation tamamlandı!")
        self.logger.info(f"📊 İşlenen görüntü: {processed_count}")
        self.logger.info(f"🎯 Başarılı augmentation: {successful_count}")
        
        return {
            'size_category': size_category,
            'processed_images': processed_count,
            'successful_augmentations': successful_count,
            'total_images_found': len(image_files),
            'pests_in_category': pests_in_category
        }

    def get_augmentation_summary(self):
        """Augmentation işlemlerinin özetini döndür"""
        return {
            'stats': self.stats.copy(),
            'supported_pests': self.supported_pests.copy(),
            'size_categories': {
                'very_small': ['whitefly', 'spider_mite', 'flea_beetle'],
                'small': ['aphid', 'thrips', 'leafhopper', 'leaf_miner'],
                'medium': ['cutworm', 'stink_bug'],
                'large': ['hornworm']
            },
            'directories': {
                'input_images': str(self.images_dir),
                'input_labels': str(self.labels_dir),
                'output_images': str(self.output_images_dir),
                'output_labels': str(self.output_labels_dir)
            }
        }


# Kullanım Örnekleri
if __name__ == "__main__":
    """
    Domates Zararlıları Augmentation Sistemi - Kullanım Örnekleri
    """
    
    # Temel kullanım
    print("🌿 Domates Zararlıları Augmentation Sistemi")
    print("=" * 50)
    
    # Augmentation sınıfını başlat
    augmenter = TomatoPestAugmentation(
        images_dir='data/images',
        labels_dir='data/labels',
        output_images_dir='output/images',
        output_labels_dir='output/labels'
    )
    
    print("\n📋 Desteklenen zararlı türleri:")
    for i, pest in enumerate(augmenter.supported_pests, 1):
        print(f"  {i:2d}. {pest}")
    
    # Örnek 1: Tek zararlı türü için augmentation
    print("\n" + "="*50)
    print("📝 ÖRNEK 1: Tek Zararlı Türü Augmentation")
    print("="*50)
    print("""
# Beyaz sinek (whitefly) için 5x augmentation
result = augmenter.augment_pest('whitefly', multiplier=5)
print(f"İşlenen görüntü: {result['processed_images']}")
print(f"Başarılı augmentation: {result['successful_augmentations']}")
    """)
    
    # Örnek 2: Tüm zararlılar için toplu augmentation
    print("\n" + "="*50)
    print("📝 ÖRNEK 2: Tüm Zararlılar İçin Toplu Augmentation")
    print("="*50)
    print("""
# Tüm zararlı türleri için 3x augmentation
results = augmenter.augment_all_pests(multiplier=3, max_images_per_pest=50)

# Sonuçları görüntüle
for pest_type, result in results.items():
    if 'error' not in result:
        print(f"{pest_type}: {result['successful_augmentations']} augmentation")
    else:
        print(f"{pest_type}: HATA - {result['error']}")
    """)
    
    # Örnek 3: Büyüklük kategorisine göre augmentation
    print("\n" + "="*50)
    print("📝 ÖRNEK 3: Büyüklük Kategorisine Göre Augmentation")
    print("="*50)
    print("""
# Çok küçük zararlılar için özel augmentation
result = augmenter.augment_by_size_category('very_small', multiplier=4)
print(f"Kategori: {result['size_category']}")
print(f"Bu kategorideki zararlılar: {result['pests_in_category']}")
print(f"Başarılı augmentation: {result['successful_augmentations']}")

# Diğer kategoriler: 'small', 'medium', 'large'
    """)
    
    # Örnek 4: Özet bilgileri
    print("\n" + "="*50)
    print("📝 ÖRNEK 4: Augmentation Özet Bilgileri")
    print("="*50)
    print("""
# Sistem özeti
summary = augmenter.get_augmentation_summary()
print("Desteklenen zararlılar:", summary['supported_pests'])
print("Büyüklük kategorileri:", summary['size_categories'])
print("Dizinler:", summary['directories'])
print("İstatistikler:", summary['stats'])
    """)
    
    # Örnek 5: Özel transform pipeline
    print("\n" + "="*50)
    print("📝 ÖRNEK 5: Özel Transform Pipeline Kullanımı")
    print("="*50)
    print("""
# Belirli bir zararlı için transform pipeline al
whitefly_transform = augmenter.get_pest_transforms('whitefly')

# Büyüklük kategorisi için transform pipeline al
small_pest_transform = augmenter.get_pest_size_category_transforms('small')

# Manuel augmentation (gelişmiş kullanım)
import cv2
image = cv2.imread('sample_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[0.5, 0.5, 0.2, 0.3]]  # YOLO format
class_labels = [0]

transformed = whitefly_transform(
    image=image,
    bboxes=bboxes,
    class_labels=class_labels
)

# Sonucu kaydet
transformed_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
cv2.imwrite('output_image.jpg', transformed_image)
    """)
    
    print("\n" + "="*60)
    print("✅ Domates Zararlıları Augmentation Sistemi Hazır!")
    print("📚 Detaylı kullanım için yukarıdaki örnekleri inceleyiniz.")
    print("🐛 10 farklı zararlı türü için özelleştirilmiş augmentation")
    print("📏 4 büyüklük kategorisi desteği (very_small, small, medium, large)")
    print("📊 CSV raporlama ve detaylı logging")
    print("🎯 YOLO format bbox desteği")
    print("="*60)
