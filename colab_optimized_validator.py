#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Google Colab Optimized Augmentation Validator

Google Colab ortamı için optimize edilmiş augmentation validation sistemi.
Sınırlı kaynaklarda verimli çalışacak şekilde tasarlanmıştır.

Özellikler:
- Colab GPU/CPU kaynaklarını verimli kullanım
- Memory-friendly batch processing
- Adaptive resource management
- Progress tracking ve ETA hesaplama
- Colab session timeout koruması

Kullanım:
    from colab_optimized_validator import ColabAugmentationValidator
    
    validator = ColabAugmentationValidator()
    result = validator.validate_colab_friendly(...)
"""

import os
import sys
import time
import gc
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import csv
from datetime import datetime, timedelta

try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import psutil
except ImportError as e:
    print(f"⚠️ Gerekli kütüphane eksik: {e}")
    print("Colab'da şu komutu çalıştırın: !pip install opencv-python scikit-image psutil")

# Colab ortamı tespiti
def is_colab():
    """Google Colab ortamında çalışıp çalışmadığını kontrol et"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_colab_resources():
    """Colab kaynaklarını analiz et"""
    resources = {
        'is_colab': is_colab(),
        'gpu_available': False,
        'gpu_memory_gb': 0,
        'ram_gb': 0,
        'disk_space_gb': 0
    }
    
    if is_colab():
        try:
            # GPU kontrolü
            import torch
            if torch.cuda.is_available():
                resources['gpu_available'] = True
                resources['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
        
        # RAM kontrolü
        try:
            mem_info = psutil.virtual_memory()
            resources['ram_gb'] = mem_info.total / (1024**3)
        except:
            resources['ram_gb'] = 12.7  # Colab varsayılan
        
        # Disk alanı kontrolü
        try:
            disk_info = psutil.disk_usage('/')
            resources['disk_space_gb'] = disk_info.free / (1024**3)
        except:
            resources['disk_space_gb'] = 25  # Colab varsayılan
    
    return resources


class ColabProgressTracker:
    """Colab için optimize edilmiş progress tracking"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.processed_items = 0
        self.start_time = time.time()
        self.last_update = time.time()
        self.update_interval = 5  # 5 saniyede bir güncelle
        
        # Colab için tqdm kullan
        if is_colab():
            try:
                from tqdm.notebook import tqdm
                self.pbar = tqdm(total=total_items, desc=description)
            except ImportError:
                self.pbar = None
        else:
            self.pbar = None
    
    def update(self, increment: int = 1):
        """Progress güncelle"""
        self.processed_items += increment
        current_time = time.time()
        
        if self.pbar:
            self.pbar.update(increment)
        
        # Periyodik konsol güncellemesi
        if current_time - self.last_update >= self.update_interval:
            self._print_progress()
            self.last_update = current_time
    
    def _print_progress(self):
        """Progress bilgisini yazdır"""
        elapsed_time = time.time() - self.start_time
        progress_ratio = self.processed_items / self.total_items
        
        if progress_ratio > 0:
            estimated_total_time = elapsed_time / progress_ratio
            eta = estimated_total_time - elapsed_time
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "Hesaplanıyor..."
        
        print(f"\r{self.description}: {self.processed_items}/{self.total_items} "
              f"({progress_ratio*100:.1f}%) - ETA: {eta_str}", end="", flush=True)
    
    def close(self):
        """Progress tracker'ı kapat"""
        if self.pbar:
            self.pbar.close()
        
        total_time = time.time() - self.start_time
        print(f"\n✅ {self.description} tamamlandı! Toplam süre: {total_time:.2f} saniye")


class ColabMemoryManager:
    """Colab için memory management"""
    
    def __init__(self, memory_threshold_gb: float = 10.0):
        self.memory_threshold_gb = memory_threshold_gb
        self.logger = logging.getLogger(__name__)
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Memory kullanımını kontrol et"""
        try:
            mem_info = psutil.virtual_memory()
            memory_usage = {
                'used_gb': mem_info.used / (1024**3),
                'available_gb': mem_info.available / (1024**3),
                'percent': mem_info.percent
            }
            return memory_usage
        except:
            return {'used_gb': 0, 'available_gb': 12, 'percent': 0}
    
    def cleanup_if_needed(self) -> bool:
        """Gerekirse memory temizliği yap"""
        memory_info = self.check_memory_usage()
        
        if memory_info['used_gb'] > self.memory_threshold_gb:
            self.logger.info(f"🧹 Memory kullanımı yüksek ({memory_info['used_gb']:.1f}GB), temizlik yapılıyor...")
            
            # Garbage collection
            gc.collect()
            
            # GPU memory temizliği (varsa)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            return True
        
        return False
    
    def get_optimal_batch_size(self, base_batch_size: int = 16) -> int:
        """Memory durumuna göre optimal batch size hesapla"""
        memory_info = self.check_memory_usage()
        available_ratio = memory_info['available_gb'] / 12.7  # Colab total RAM
        
        if available_ratio > 0.7:
            return base_batch_size
        elif available_ratio > 0.5:
            return max(8, base_batch_size // 2)
        elif available_ratio > 0.3:
            return max(4, base_batch_size // 4)
        else:
            return 2  # Minimum batch size


class ColabAugmentationValidator:
    """Google Colab için optimize edilmiş augmentation validator"""
    
    def __init__(self, 
                 memory_threshold_gb: float = 10.0,
                 max_workers: int = 2,
                 batch_size: int = 8):
        
        self.resources = get_colab_resources()
        self.memory_manager = ColabMemoryManager(memory_threshold_gb)
        self.max_workers = min(max_workers, 2) if is_colab() else max_workers
        self.base_batch_size = batch_size
        
        # Logger ayarla
        self.logger = logging.getLogger(__name__)
        
        # Colab uyarıları
        if is_colab():
            print("🚀 Colab ortamı tespit edildi!")
            print(f"📊 RAM: {self.resources['ram_gb']:.1f}GB")
            print(f"🎮 GPU: {'Evet' if self.resources['gpu_available'] else 'Hayır'}")
            print(f"💾 Disk: {self.resources['disk_space_gb']:.1f}GB boş")
    
    def validate_single_augmentation_colab(self, 
                                         original_path: str, 
                                         augmented_path: str) -> Dict[str, Any]:
        """Tek augmentation'ı Colab-friendly şekilde validate et"""
        
        try:
            # Görüntüleri yükle
            original = cv2.imread(original_path)
            augmented = cv2.imread(augmented_path)
            
            if original is None or augmented is None:
                return {'error': 'Görüntü yüklenemedi', 'valid': False}
            
            # Görüntüleri aynı boyuta getir
            if original.shape != augmented.shape:
                augmented = cv2.resize(augmented, (original.shape[1], original.shape[0]))
            
            # Grayscale'e çevir (memory tasarrufu için)
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            augmented_gray = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)
            
            # SSIM hesapla (memory-friendly)
            ssim_score = ssim(original_gray, augmented_gray, data_range=255)
            
            # PSNR hesapla
            psnr_score = psnr(original_gray, augmented_gray, data_range=255)
            
            # Brightness ve contrast farkları
            orig_brightness = np.mean(original_gray)
            aug_brightness = np.mean(augmented_gray)
            brightness_diff = abs(orig_brightness - aug_brightness) / 255.0
            
            orig_contrast = np.std(original_gray)
            aug_contrast = np.std(augmented_gray)
            contrast_diff = abs(orig_contrast - aug_contrast) / 255.0
            
            # Overall quality score
            quality_score = (ssim_score * 0.5 + 
                           min(psnr_score / 30.0, 1.0) * 0.3 + 
                           (1 - brightness_diff) * 0.1 + 
                           (1 - contrast_diff) * 0.1)
            
            # Memory temizliği
            del original, augmented, original_gray, augmented_gray
            gc.collect()
            
            return {
                'ssim': float(ssim_score),
                'psnr': float(psnr_score),
                'brightness_diff': float(brightness_diff),
                'contrast_diff': float(contrast_diff),
                'overall_quality': float(quality_score),
                'valid': True,
                'passed': quality_score > 0.7  # Colab için daha esnek threshold
            }
            
        except Exception as e:
            self.logger.error(f"Validation hatası: {str(e)}")
            return {'error': str(e), 'valid': False}
    
    def validate_directory_colab_friendly(self,
                                        original_images_dir: str,
                                        augmented_images_dir: str,
                                        sample_rate: float = 0.1,
                                        save_report: bool = True) -> Dict[str, Any]:
        """Dizin bazlı validation - Colab için optimize edilmiş"""
        
        self.logger.info("🔍 Colab-friendly directory validation başlatılıyor...")
        
        # Dosyaları bul
        original_images = []
        augmented_images = []
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            original_images.extend(Path(original_images_dir).glob(f'*{ext}'))
            original_images.extend(Path(original_images_dir).glob(f'*{ext.upper()}'))
        
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            augmented_images.extend(Path(augmented_images_dir).glob(f'*{ext}'))
            augmented_images.extend(Path(augmented_images_dir).glob(f'*{ext.upper()}'))
        
        # Sampling (Colab için)
        if sample_rate < 1.0:
            sample_size = max(1, int(len(original_images) * sample_rate))
            original_images = original_images[:sample_size]
            self.logger.info(f"📊 Sampling: {sample_size}/{len(original_images)} görüntü test edilecek")
        
        # Eşleştir
        validation_pairs = []
        for orig_path in original_images:
            orig_name = orig_path.stem
            
            # Augmented karşılığını bul
            for aug_path in augmented_images:
                if orig_name in aug_path.stem:
                    validation_pairs.append((str(orig_path), str(aug_path)))
                    break
        
        if not validation_pairs:
            return {
                'error': 'Eşleşen görüntü çifti bulunamadı',
                'total_validated': 0,
                'passed_images': 0,
                'failed_images': 0
            }
        
        # Progress tracker
        progress = ColabProgressTracker(len(validation_pairs), "Validation")
        
        # Validation sonuçları
        results = []
        passed_count = 0
        failed_count = 0
        
        # Batch processing
        current_batch_size = self.memory_manager.get_optimal_batch_size(self.base_batch_size)
        
        try:
            for i in range(0, len(validation_pairs), current_batch_size):
                batch_pairs = validation_pairs[i:i + current_batch_size]
                
                # Memory kontrolü
                self.memory_manager.cleanup_if_needed()
                
                # Batch'i işle
                for orig_path, aug_path in batch_pairs:
                    result = self.validate_single_augmentation_colab(orig_path, aug_path)
                    
                    if result.get('valid', False):
                        results.append(result)
                        if result.get('passed', False):
                            passed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                    
                    progress.update(1)
                
                # Colab session timeout koruması
                if i % (current_batch_size * 10) == 0:
                    time.sleep(0.1)  # Kısa break
        
        finally:
            progress.close()
        
        # Sonuç özeti
        total_validated = len(results)
        avg_ssim = np.mean([r['ssim'] for r in results if 'ssim' in r]) if results else 0
        avg_psnr = np.mean([r['psnr'] for r in results if 'psnr' in r]) if results else 0
        avg_quality = np.mean([r['overall_quality'] for r in results if 'overall_quality' in r]) if results else 0
        
        summary = {
            'total_validated': total_validated,
            'passed_images': passed_count,
            'failed_images': failed_count,
            'pass_rate': passed_count / max(1, total_validated),
            'avg_ssim': float(avg_ssim),
            'avg_psnr': float(avg_psnr),
            'avg_quality': float(avg_quality),
            'validation_time': time.time(),
            'colab_optimized': True
        }
        
        # Rapor kaydet
        if save_report:
            self._save_colab_report(summary, results)
        
        self.logger.info(f"✅ Validation tamamlandı: {passed_count}/{total_validated} geçti")
        return summary
    
    def _save_colab_report(self, summary: Dict[str, Any], results: List[Dict[str, Any]]):
        """Colab için rapor kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON raporu
        report_path = f"colab_validation_report_{timestamp}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'detailed_results': results,
                'colab_resources': self.resources
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Validation raporu kaydedildi: {report_path}")
        
        # Colab'da görüntüle
        if is_colab():
            try:
                from google.colab import files
                print(f"📥 Raporu indirmek için: files.download('{report_path}')")
            except ImportError:
                pass


# Kullanım örneği
if __name__ == "__main__":
    # Colab için örnek kullanım
    validator = ColabAugmentationValidator(
        memory_threshold_gb=8.0,
        max_workers=2,
        batch_size=4
    )
    
    # Test validation
    if is_colab():
        print("🧪 Colab validation testi...")
        
        # Örnek dizinler (kullanıcı kendi dizinlerini belirtmeli)
        original_dir = "/content/data/original/images"
        augmented_dir = "/content/data/augmented/images"
        
        if os.path.exists(original_dir) and os.path.exists(augmented_dir):
            result = validator.validate_directory_colab_friendly(
                original_dir, augmented_dir,
                sample_rate=0.1,  # %10 sampling
                save_report=True
            )
            
            print(f"✅ Validation tamamlandı!")
            print(f"📊 Geçen: {result['passed_images']}/{result['total_validated']}")
            print(f"🎯 Başarı oranı: {result['pass_rate']*100:.1f}%")
        else:
            print("⚠️ Test dizinleri bulunamadı. Lütfen kendi dizinlerinizi belirtin.")
    else:
        print("ℹ️ Bu script Google Colab için optimize edilmiştir.")
