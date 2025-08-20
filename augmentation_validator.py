"""
Augmentation Validation ve Kalite Kontrol Sistemi
================================================

Bu modül, augmentation işlemlerinin kalitesini otomatik olarak değerlendirir ve
performance optimizasyonu sağlar. Paralel işlem desteği ile büyük veri setleri
için hızlandırılmış augmentation sunar.

Özellikler:
- Augmentation kalite metrikleri hesaplama
- Görsel benzerlik analizi
- Bounding box doğrulama
- Performance monitoring
- Paralel batch processing
- Otomatik kalite kontrol
- Detaylı raporlama

Kullanım:
    validator = AugmentationValidator()
    quality_report = validator.validate_augmentation_quality('output_dir')
    performance_report = validator.benchmark_augmentation_performance()
"""

import cv2
import numpy as np
import os
import json
import csv
from pathlib import Path
from datetime import datetime
import logging
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from typing import List, Dict, Tuple, Optional, Any
import statistics
from dataclasses import dataclass
import psutil
import gc

# Image quality metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import measure
import albumentations as A

@dataclass
class QualityMetrics:
    """Kalite metrikleri veri sınıfı"""
    ssim_score: float
    psnr_score: float
    brightness_diff: float
    contrast_diff: float
    bbox_preservation: float
    visual_similarity: float
    processing_time: float
    memory_usage: float

@dataclass
class PerformanceMetrics:
    """Performance metrikleri veri sınıfı"""
    total_time: float
    avg_time_per_image: float
    images_per_second: float
    memory_peak: float
    memory_avg: float
    cpu_usage_avg: float
    success_rate: float
    error_count: int

class AugmentationValidator:
    def __init__(self, log_level=logging.INFO):
        """
        Augmentation validator sınıfı
        
        Args:
            log_level: Logging seviyesi
        """
        self.setup_logging(log_level)
        
        # Kalite threshold'ları
        self.quality_thresholds = {
            'ssim_min': 0.3,           # Minimum structural similarity
            'psnr_min': 15.0,          # Minimum peak signal-to-noise ratio
            'brightness_max_diff': 0.3, # Maksimum parlaklık farkı
            'contrast_max_diff': 0.4,   # Maksimum kontrast farkı
            'bbox_preservation_min': 0.8, # Minimum bbox korunma oranı
            'visual_similarity_min': 0.4   # Minimum görsel benzerlik
        }
        
        # Performance monitoring
        self.performance_data = []
        self.memory_monitor = None
        self.cpu_monitor = None
        
        # Thread safety
        self.lock = threading.Lock()
        
    def setup_logging(self, log_level):
        """Logging konfigürasyonu"""
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('augmentation_validator.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_image_quality_metrics(self, original_img, augmented_img) -> QualityMetrics:
        """İki görüntü arasında kalite metrikleri hesapla"""
        try:
            # Görüntüleri aynı boyuta getir
            if original_img.shape != augmented_img.shape:
                augmented_img = cv2.resize(augmented_img, (original_img.shape[1], original_img.shape[0]))
            
            # Gri tonlamaya çevir
            if len(original_img.shape) == 3:
                orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                aug_gray = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original_img
                aug_gray = augmented_img
            
            # SSIM hesapla
            ssim_score = ssim(orig_gray, aug_gray, data_range=255)
            
            # PSNR hesapla
            psnr_score = psnr(orig_gray, aug_gray, data_range=255)
            
            # Parlaklık farkı
            orig_brightness = np.mean(orig_gray)
            aug_brightness = np.mean(aug_gray)
            brightness_diff = abs(orig_brightness - aug_brightness) / 255.0
            
            # Kontrast farkı
            orig_contrast = np.std(orig_gray)
            aug_contrast = np.std(aug_gray)
            contrast_diff = abs(orig_contrast - aug_contrast) / 255.0
            
            # Görsel benzerlik (histogram karşılaştırması)
            orig_hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
            aug_hist = cv2.calcHist([aug_gray], [0], None, [256], [0, 256])
            visual_similarity = cv2.compareHist(orig_hist, aug_hist, cv2.HISTCMP_CORREL)
            
            return QualityMetrics(
                ssim_score=ssim_score,
                psnr_score=psnr_score,
                brightness_diff=brightness_diff,
                contrast_diff=contrast_diff,
                bbox_preservation=1.0,  # Ayrıca hesaplanacak
                visual_similarity=visual_similarity,
                processing_time=0.0,    # Ayrıca hesaplanacak
                memory_usage=0.0        # Ayrıca hesaplanacak
            )
            
        except Exception as e:
            self.logger.error(f"Kalite metrikleri hesaplama hatası: {str(e)}")
            return QualityMetrics(0, 0, 1, 1, 0, 0, 0, 0)
    
    def validate_bbox_preservation(self, original_bboxes, augmented_bboxes) -> float:
        """Bounding box korunma oranını hesapla"""
        try:
            if not original_bboxes or not augmented_bboxes:
                return 0.0
            
            if len(original_bboxes) != len(augmented_bboxes):
                return 0.0
            
            preservation_scores = []
            
            for orig_bbox, aug_bbox in zip(original_bboxes, augmented_bboxes):
                # YOLO format: [x_center, y_center, width, height]
                orig_x, orig_y, orig_w, orig_h = orig_bbox
                aug_x, aug_y, aug_w, aug_h = aug_bbox
                
                # IoU hesapla
                orig_x1 = orig_x - orig_w/2
                orig_y1 = orig_y - orig_h/2
                orig_x2 = orig_x + orig_w/2
                orig_y2 = orig_y + orig_h/2
                
                aug_x1 = aug_x - aug_w/2
                aug_y1 = aug_y - aug_h/2
                aug_x2 = aug_x + aug_w/2
                aug_y2 = aug_y + aug_h/2
                
                # Intersection
                int_x1 = max(orig_x1, aug_x1)
                int_y1 = max(orig_y1, aug_y1)
                int_x2 = min(orig_x2, aug_x2)
                int_y2 = min(orig_y2, aug_y2)
                
                if int_x2 > int_x1 and int_y2 > int_y1:
                    intersection = (int_x2 - int_x1) * (int_y2 - int_y1)
                else:
                    intersection = 0
                
                # Union
                orig_area = orig_w * orig_h
                aug_area = aug_w * aug_h
                union = orig_area + aug_area - intersection
                
                # IoU
                if union > 0:
                    iou = intersection / union
                    preservation_scores.append(iou)
                else:
                    preservation_scores.append(0.0)
            
            return np.mean(preservation_scores) if preservation_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Bbox korunma hesaplama hatası: {str(e)}")
            return 0.0
    
    def read_yolo_annotation(self, label_path):
        """YOLO annotation dosyasını oku"""
        try:
            bboxes = []
            class_labels = []
            
            if not os.path.exists(label_path):
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
            self.logger.error(f"Annotation okuma hatası {label_path}: {str(e)}")
            return [], []
    
    def validate_single_augmentation(self, original_img_path, augmented_img_path, 
                                   original_label_path, augmented_label_path) -> QualityMetrics:
        """Tek bir augmentation işlemini validate et"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Görüntüleri oku
            original_img = cv2.imread(original_img_path)
            augmented_img = cv2.imread(augmented_img_path)
            
            if original_img is None or augmented_img is None:
                self.logger.warning(f"Görüntü okunamadı: {original_img_path} veya {augmented_img_path}")
                return QualityMetrics(0, 0, 1, 1, 0, 0, 0, 0)
            
            # Kalite metrikleri hesapla
            quality_metrics = self.calculate_image_quality_metrics(original_img, augmented_img)
            
            # Bbox korunma oranını hesapla
            orig_bboxes, _ = self.read_yolo_annotation(original_label_path)
            aug_bboxes, _ = self.read_yolo_annotation(augmented_label_path)
            bbox_preservation = self.validate_bbox_preservation(orig_bboxes, aug_bboxes)
            
            # Timing ve memory
            processing_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory
            
            # Metrikleri güncelle
            quality_metrics.bbox_preservation = bbox_preservation
            quality_metrics.processing_time = processing_time
            quality_metrics.memory_usage = memory_usage
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Validation hatası: {str(e)}")
            return QualityMetrics(0, 0, 1, 1, 0, 0, 0, 0)
    
    def validate_augmentation_batch(self, validation_pairs: List[Tuple[str, str, str, str]]) -> List[QualityMetrics]:
        """Batch validation işlemi"""
        results = []
        
        for orig_img, aug_img, orig_label, aug_label in validation_pairs:
            metrics = self.validate_single_augmentation(orig_img, aug_img, orig_label, aug_label)
            results.append(metrics)
        
        return results
    
    def validate_augmentation_batch_parallel(self, validation_pairs: List[Tuple[str, str, str, str]], 
                                           max_workers: int = None) -> List[QualityMetrics]:
        """Paralel batch validation işlemi"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(validation_pairs))
        
        self.logger.info(f"🚀 Paralel validation başlatılıyor - {max_workers} worker")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for orig_img, aug_img, orig_label, aug_label in validation_pairs:
                future = executor.submit(
                    self.validate_single_augmentation, 
                    orig_img, aug_img, orig_label, aug_label
                )
                futures.append(future)
            
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=30)  # 30 saniye timeout
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"📈 Validation progress: {i+1}/{len(futures)}")
                        
                except Exception as e:
                    self.logger.error(f"Paralel validation hatası: {str(e)}")
                    results.append(QualityMetrics(0, 0, 1, 1, 0, 0, 0, 0))
        
        return results
    
    def analyze_quality_metrics(self, metrics_list: List[QualityMetrics]) -> Dict[str, Any]:
        """Kalite metriklerini analiz et"""
        if not metrics_list:
            return {}
        
        analysis = {
            'total_samples': len(metrics_list),
            'ssim': {
                'mean': statistics.mean([m.ssim_score for m in metrics_list]),
                'median': statistics.median([m.ssim_score for m in metrics_list]),
                'std': statistics.stdev([m.ssim_score for m in metrics_list]) if len(metrics_list) > 1 else 0,
                'min': min([m.ssim_score for m in metrics_list]),
                'max': max([m.ssim_score for m in metrics_list])
            },
            'psnr': {
                'mean': statistics.mean([m.psnr_score for m in metrics_list]),
                'median': statistics.median([m.psnr_score for m in metrics_list]),
                'std': statistics.stdev([m.psnr_score for m in metrics_list]) if len(metrics_list) > 1 else 0,
                'min': min([m.psnr_score for m in metrics_list]),
                'max': max([m.psnr_score for m in metrics_list])
            },
            'bbox_preservation': {
                'mean': statistics.mean([m.bbox_preservation for m in metrics_list]),
                'median': statistics.median([m.bbox_preservation for m in metrics_list]),
                'std': statistics.stdev([m.bbox_preservation for m in metrics_list]) if len(metrics_list) > 1 else 0,
                'min': min([m.bbox_preservation for m in metrics_list]),
                'max': max([m.bbox_preservation for m in metrics_list])
            },
            'visual_similarity': {
                'mean': statistics.mean([m.visual_similarity for m in metrics_list]),
                'median': statistics.median([m.visual_similarity for m in metrics_list]),
                'std': statistics.stdev([m.visual_similarity for m in metrics_list]) if len(metrics_list) > 1 else 0,
                'min': min([m.visual_similarity for m in metrics_list]),
                'max': max([m.visual_similarity for m in metrics_list])
            },
            'performance': {
                'avg_processing_time': statistics.mean([m.processing_time for m in metrics_list]),
                'total_processing_time': sum([m.processing_time for m in metrics_list]),
                'avg_memory_usage': statistics.mean([m.memory_usage for m in metrics_list]),
                'total_memory_usage': sum([m.memory_usage for m in metrics_list])
            }
        }
        
        # Kalite değerlendirmesi
        analysis['quality_assessment'] = self.assess_quality(analysis)
        
        return analysis
    
    def assess_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Kalite değerlendirmesi yap"""
        assessment = {
            'overall_quality': 'UNKNOWN',
            'passed_tests': 0,
            'total_tests': 6,
            'issues': []
        }
        
        # SSIM kontrolü
        if analysis['ssim']['mean'] >= self.quality_thresholds['ssim_min']:
            assessment['passed_tests'] += 1
        else:
            assessment['issues'].append(f"Düşük SSIM skoru: {analysis['ssim']['mean']:.3f}")
        
        # PSNR kontrolü
        if analysis['psnr']['mean'] >= self.quality_thresholds['psnr_min']:
            assessment['passed_tests'] += 1
        else:
            assessment['issues'].append(f"Düşük PSNR skoru: {analysis['psnr']['mean']:.3f}")
        
        # Bbox preservation kontrolü
        if analysis['bbox_preservation']['mean'] >= self.quality_thresholds['bbox_preservation_min']:
            assessment['passed_tests'] += 1
        else:
            assessment['issues'].append(f"Düşük bbox korunma: {analysis['bbox_preservation']['mean']:.3f}")
        
        # Visual similarity kontrolü
        if analysis['visual_similarity']['mean'] >= self.quality_thresholds['visual_similarity_min']:
            assessment['passed_tests'] += 1
        else:
            assessment['issues'].append(f"Düşük görsel benzerlik: {analysis['visual_similarity']['mean']:.3f}")
        
        # Performance kontrolü (örnek kriterler)
        if analysis['performance']['avg_processing_time'] < 5.0:  # 5 saniyeden az
            assessment['passed_tests'] += 1
        else:
            assessment['issues'].append(f"Yavaş işlem: {analysis['performance']['avg_processing_time']:.2f}s")
        
        if analysis['performance']['avg_memory_usage'] < 100:  # 100MB'dan az
            assessment['passed_tests'] += 1
        else:
            assessment['issues'].append(f"Yüksek bellek kullanımı: {analysis['performance']['avg_memory_usage']:.1f}MB")
        
        # Genel kalite değerlendirmesi
        pass_rate = assessment['passed_tests'] / assessment['total_tests']
        if pass_rate >= 0.8:
            assessment['overall_quality'] = 'EXCELLENT'
        elif pass_rate >= 0.6:
            assessment['overall_quality'] = 'GOOD'
        elif pass_rate >= 0.4:
            assessment['overall_quality'] = 'FAIR'
        else:
            assessment['overall_quality'] = 'POOR'
        
        return assessment

    def validate_augmentation_directory(self, original_dir: str, augmented_dir: str, 
                                      original_labels_dir: str, augmented_labels_dir: str,
                                      parallel: bool = True, max_workers: int = None) -> Dict[str, Any]:
        """Tüm augmentation dizinini validate et"""
        self.logger.info("🔍 Augmentation dizini validation başlatılıyor...")
        
        # Dosya çiftlerini bul
        validation_pairs = []
        original_path = Path(original_dir)
        augmented_path = Path(augmented_dir)
        original_labels_path = Path(original_labels_dir)
        augmented_labels_path = Path(augmented_labels_dir)
        
        # Augmented görüntüleri tara
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        augmented_images = []
        for ext in image_extensions:
            augmented_images.extend(augmented_path.glob(f"*{ext}"))
            augmented_images.extend(augmented_path.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"📁 {len(augmented_images)} augmented görüntü bulundu")
        
        for aug_img_path in augmented_images:
            # Original görüntü adını çıkar (augmentation suffix'ini kaldır)
            aug_name = aug_img_path.stem
            
            # Augmentation pattern'lerini temizle
            patterns_to_remove = [
                '_whitefly_aug_', '_aphid_aug_', '_thrips_aug_', '_spider_mite_aug_',
                '_hornworm_aug_', '_cutworm_aug_', '_leafhopper_aug_', '_flea_beetle_aug_',
                '_leaf_miner_aug_', '_stink_bug_aug_', '_disease_aug_', '_mineral_aug_',
                '_very_small_aug_', '_small_aug_', '_medium_aug_', '_large_aug_'
            ]
            
            original_name = aug_name
            for pattern in patterns_to_remove:
                if pattern in original_name:
                    original_name = original_name.split(pattern)[0]
                    break
            
            # Original dosyaları bul
            original_img_path = None
            for ext in image_extensions:
                potential_path = original_path / f"{original_name}{ext}"
                if potential_path.exists():
                    original_img_path = potential_path
                    break
                potential_path = original_path / f"{original_name}{ext.upper()}"
                if potential_path.exists():
                    original_img_path = potential_path
                    break
            
            if original_img_path is None:
                continue
            
            # Label dosyalarını bul
            original_label_path = original_labels_path / f"{original_name}.txt"
            augmented_label_path = augmented_labels_path / f"{aug_name}.txt"
            
            validation_pairs.append((
                str(original_img_path),
                str(aug_img_path),
                str(original_label_path),
                str(augmented_label_path)
            ))
        
        self.logger.info(f"🎯 {len(validation_pairs)} validation çifti hazırlandı")
        
        # Validation işlemini çalıştır
        start_time = time.time()
        
        if parallel and len(validation_pairs) > 1:
            metrics_list = self.validate_augmentation_batch_parallel(validation_pairs, max_workers)
        else:
            metrics_list = self.validate_augmentation_batch(validation_pairs)
        
        total_time = time.time() - start_time
        
        # Analiz yap
        analysis = self.analyze_quality_metrics(metrics_list)
        analysis['validation_info'] = {
            'total_pairs_validated': len(validation_pairs),
            'total_validation_time': total_time,
            'validation_method': 'parallel' if parallel else 'sequential',
            'timestamp': datetime.now().isoformat()
        }
        
        # Raporu kaydet
        self.save_validation_report(analysis, 'validation_report.json')
        
        self.logger.info("✅ Validation tamamlandı!")
        self.logger.info(f"📊 Genel kalite: {analysis['quality_assessment']['overall_quality']}")
        self.logger.info(f"🎯 Başarı oranı: {analysis['quality_assessment']['passed_tests']}/{analysis['quality_assessment']['total_tests']}")
        
        return analysis
    
    def save_validation_report(self, analysis: Dict[str, Any], output_path: str):
        """Validation raporunu kaydet"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
            
            # CSV formatında da kaydet
            csv_path = output_path.replace('.json', '.csv')
            self.save_validation_report_csv(analysis, csv_path)
            
            self.logger.info(f"📄 Validation raporu kaydedildi: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Rapor kaydetme hatası: {str(e)}")
    
    def save_validation_report_csv(self, analysis: Dict[str, Any], csv_path: str):
        """Validation raporunu CSV formatında kaydet"""
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Başlık
                writer.writerow(['Metric', 'Mean', 'Median', 'Std', 'Min', 'Max'])
                
                # Metrikler
                for metric_name in ['ssim', 'psnr', 'bbox_preservation', 'visual_similarity']:
                    if metric_name in analysis:
                        metric_data = analysis[metric_name]
                        writer.writerow([
                            metric_name.upper(),
                            f"{metric_data['mean']:.4f}",
                            f"{metric_data['median']:.4f}",
                            f"{metric_data['std']:.4f}",
                            f"{metric_data['min']:.4f}",
                            f"{metric_data['max']:.4f}"
                        ])
                
                # Performance metrikleri
                writer.writerow([])
                writer.writerow(['Performance Metric', 'Value', '', '', '', ''])
                if 'performance' in analysis:
                    perf = analysis['performance']
                    writer.writerow(['Avg Processing Time (s)', f"{perf['avg_processing_time']:.3f}", '', '', '', ''])
                    writer.writerow(['Total Processing Time (s)', f"{perf['total_processing_time']:.3f}", '', '', '', ''])
                    writer.writerow(['Avg Memory Usage (MB)', f"{perf['avg_memory_usage']:.2f}", '', '', '', ''])
                
                # Kalite değerlendirmesi
                writer.writerow([])
                writer.writerow(['Quality Assessment', 'Result', '', '', '', ''])
                if 'quality_assessment' in analysis:
                    qa = analysis['quality_assessment']
                    writer.writerow(['Overall Quality', qa['overall_quality'], '', '', '', ''])
                    writer.writerow(['Passed Tests', f"{qa['passed_tests']}/{qa['total_tests']}", '', '', '', ''])
                    
                    if qa['issues']:
                        writer.writerow([])
                        writer.writerow(['Issues Found', '', '', '', '', ''])
                        for issue in qa['issues']:
                            writer.writerow([issue, '', '', '', '', ''])
            
        except Exception as e:
            self.logger.error(f"CSV rapor kaydetme hatası: {str(e)}")

class PerformanceOptimizer:
    """Performance optimizasyon sınıfı"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def optimize_batch_size(self, total_images: int, available_memory_gb: float = None) -> int:
        """Optimal batch size hesapla"""
        if available_memory_gb is None:
            # Mevcut bellek miktarını al
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Görüntü başına ortalama bellek kullanımı (MB)
        avg_memory_per_image = 50  # Ortalama değer
        
        # Güvenli batch size hesapla (%70 bellek kullanımı)
        safe_memory_gb = available_memory_gb * 0.7
        safe_memory_mb = safe_memory_gb * 1024
        
        optimal_batch_size = int(safe_memory_mb / avg_memory_per_image)
        optimal_batch_size = max(1, min(optimal_batch_size, total_images))
        
        self.logger.info(f"🚀 Optimal batch size: {optimal_batch_size} (Mevcut bellek: {available_memory_gb:.1f}GB)")
        
        return optimal_batch_size
    
    def calculate_optimal_workers(self, cpu_intensive: bool = False) -> int:
        """Optimal worker sayısı hesapla"""
        cpu_count = mp.cpu_count()
        
        if cpu_intensive:
            # CPU-intensive işlemler için CPU sayısı kadar worker
            optimal_workers = cpu_count
        else:
            # I/O-intensive işlemler için daha fazla worker
            optimal_workers = min(cpu_count * 2, 32)
        
        self.logger.info(f"🔧 Optimal worker sayısı: {optimal_workers} (CPU count: {cpu_count})")
        
        return optimal_workers
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Sistem kaynaklarını monitör et"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resources = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
        
        return resources
    
    def suggest_optimizations(self, current_performance: Dict[str, Any]) -> List[str]:
        """Performance optimizasyon önerileri"""
        suggestions = []
        
        resources = self.monitor_system_resources()
        
        # Bellek önerileri
        if resources['memory_percent'] > 80:
            suggestions.append("⚠️ Yüksek bellek kullanımı - batch size'ı azaltın")
        elif resources['memory_percent'] < 50:
            suggestions.append("💡 Düşük bellek kullanımı - batch size'ı artırabilirsiniz")
        
        # CPU önerileri
        if resources['cpu_percent'] > 90:
            suggestions.append("⚠️ Yüksek CPU kullanımı - worker sayısını azaltın")
        elif resources['cpu_percent'] < 30:
            suggestions.append("💡 Düşük CPU kullanımı - paralel işlemi artırabilirsiniz")
        
        # Disk önerileri
        if resources['disk_percent'] > 90:
            suggestions.append("⚠️ Disk alanı düşük - geçici dosyaları temizleyin")
        
        # Performance önerileri
        if 'performance' in current_performance:
            perf = current_performance['performance']
            if perf['avg_processing_time'] > 10:
                suggestions.append("🐌 Yavaş işlem - görüntü boyutlarını kontrol edin")
            
            if perf['avg_memory_usage'] > 200:
                suggestions.append("🧠 Yüksek bellek kullanımı - augmentation karmaşıklığını azaltın")
        
        return suggestions

# Kullanım örnekleri
if __name__ == "__main__":
    """
    Augmentation Validation ve Performance Optimization - Kullanım Örnekleri
    """
    
    print("🔍 Augmentation Validation ve Performance Optimization Sistemi")
    print("=" * 70)
    
    # Validator oluştur
    validator = AugmentationValidator()
    optimizer = PerformanceOptimizer()
    
    print("\n📋 Sistem Özellikleri:")
    print("  • Augmentation kalite metrikleri (SSIM, PSNR, Bbox preservation)")
    print("  • Paralel validation desteği")
    print("  • Performance monitoring")
    print("  • Otomatik optimizasyon önerileri")
    print("  • Detaylı raporlama (JSON + CSV)")
    
    # Örnek 1: Tek dosya validation
    print("\n" + "="*50)
    print("📝 ÖRNEK 1: Tek Augmentation Validation")
    print("="*50)
    print("""
# Tek bir augmentation işlemini validate et
metrics = validator.validate_single_augmentation(
    'original_images/tomato_001.jpg',
    'augmented_images/tomato_001_whitefly_aug_1.jpg',
    'original_labels/tomato_001.txt',
    'augmented_labels/tomato_001_whitefly_aug_1.txt'
)

print(f"SSIM Score: {metrics.ssim_score:.3f}")
print(f"PSNR Score: {metrics.psnr_score:.3f}")
print(f"Bbox Preservation: {metrics.bbox_preservation:.3f}")
    """)
    
    # Örnek 2: Dizin validation
    print("\n" + "="*50)
    print("📝 ÖRNEK 2: Tüm Dizin Validation")
    print("="*50)
    print("""
# Tüm augmentation dizinini validate et
analysis = validator.validate_augmentation_directory(
    original_dir='data/original_images',
    augmented_dir='data/augmented_images',
    original_labels_dir='data/original_labels',
    augmented_labels_dir='data/augmented_labels',
    parallel=True,
    max_workers=8
)

print(f"Genel Kalite: {analysis['quality_assessment']['overall_quality']}")
print(f"Başarı Oranı: {analysis['quality_assessment']['passed_tests']}/6")
    """)
    
    # Örnek 3: Performance optimization
    print("\n" + "="*50)
    print("📝 ÖRNEK 3: Performance Optimization")
    print("="*50)
    print("""
# Optimal batch size hesapla
optimal_batch = optimizer.optimize_batch_size(total_images=1000)
print(f"Optimal batch size: {optimal_batch}")

# Optimal worker sayısı
optimal_workers = optimizer.calculate_optimal_workers(cpu_intensive=False)
print(f"Optimal worker sayısı: {optimal_workers}")

# Sistem kaynaklarını monitör et
resources = optimizer.monitor_system_resources()
print(f"CPU: {resources['cpu_percent']:.1f}%")
print(f"Memory: {resources['memory_percent']:.1f}%")

# Optimizasyon önerileri al
suggestions = optimizer.suggest_optimizations(analysis)
for suggestion in suggestions:
    print(f"  {suggestion}")
    """)
    
    # Örnek 4: Entegre kullanım
    print("\n" + "="*50)
    print("📝 ÖRNEK 4: Augmentation + Validation Entegrasyonu")
    print("="*50)
    print("""
from tomato_pest_augmentation import TomatoPestAugmentation

# Augmentation yap
augmenter = TomatoPestAugmentation('input_images', 'input_labels', 
                                 'output_images', 'output_labels')
result = augmenter.augment_pest('whitefly', multiplier=5)

# Sonuçları validate et
analysis = validator.validate_augmentation_directory(
    'input_images', 'output_images',
    'input_labels', 'output_labels'
)

# Kalite kontrolü
if analysis['quality_assessment']['overall_quality'] in ['EXCELLENT', 'GOOD']:
    print("✅ Augmentation kalitesi kabul edilebilir")
else:
    print("⚠️ Augmentation kalitesi düşük - parametreleri gözden geçirin")
    for issue in analysis['quality_assessment']['issues']:
        print(f"  • {issue}")
    """)
    
    print("\n" + "="*70)
    print("✅ Augmentation Validation ve Performance Optimization Sistemi Hazır!")
    print("🔍 Kalite metrikleri: SSIM, PSNR, Bbox preservation, Visual similarity")
    print("🚀 Performance: Paralel işlem, bellek optimizasyonu, CPU monitoring")
    print("📊 Raporlama: JSON ve CSV formatında detaylı raporlar")
    print("💡 Optimizasyon: Otomatik batch size ve worker sayısı hesaplama")
    print("="*70)
