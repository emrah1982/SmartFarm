"""
Batch Augmentation Processor - Paralel ve Optimize Edilmiş Augmentation Sistemi
==============================================================================

Bu modül, büyük veri setleri için optimize edilmiş paralel augmentation işlemleri
gerçekleştirir. Performance monitoring ve otomatik optimizasyon özellikleri içerir.

Özellikler:
- Paralel batch processing
- Bellek optimizasyonu
- Progress tracking
- Error recovery
- Performance monitoring
- Adaptive batch sizing
- Resource management
- Quality control integration

Kullanım:
    processor = BatchAugmentationProcessor()
    results = processor.process_dataset_parallel(
        'input_images', 'input_labels', 'output_images', 'output_labels',
        augmentation_configs=['whitefly', 'aphid'], multiplier=3
    )
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
import time
import os
import glob
import logging
import json
import csv
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

try:
    from augmentation import TomatoPestAugmentation
except ImportError:
    TomatoPestAugmentation = None

try:
    from augmentation_validator import AugmentationValidator
except ImportError:
    AugmentationValidator = None

try:
    from augmentation_validator import AugmentationValidator, PerformanceOptimizer
    AUGMENTATION_SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Augmentation systems import hatası: {e}")
    AUGMENTATION_SYSTEMS_AVAILABLE = False

@dataclass
class BatchProcessingConfig:
    """Batch processing konfigürasyonu"""
    batch_size: int = 32
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    enable_validation: bool = True
    validation_sample_rate: float = 0.1
    error_tolerance: float = 0.1
    progress_callback: Optional[Callable] = None
    temp_dir: Optional[str] = None
    cleanup_temp: bool = True

@dataclass
class ProcessingResult:
    """İşlem sonucu veri sınıfı"""
    total_images: int = 0
    processed_images: int = 0
    successful_augmentations: int = 0
    failed_augmentations: int = 0
    skipped_images: int = 0
    processing_time: float = 0.0
    avg_time_per_image: float = 0.0
    peak_memory_usage: float = 0.0
    error_details: List[str] = None
    validation_results: Optional[Dict] = None

    def __post_init__(self):
        if self.error_details is None:
            self.error_details = []

@dataclass
class BatchTask:
    """Batch görev veri sınıfı"""
    task_id: str
    image_paths: List[str]
    label_paths: List[str]
    output_image_dir: str
    output_label_dir: str
    augmentation_type: str
    multiplier: int
    config: Dict[str, Any]

class ResourceMonitor:
    """Sistem kaynak monitörü"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.resource_data = []
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Monitoring başlat"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Monitoring durdur"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Monitoring döngüsü"""
        while self.is_monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                
                resource_info = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3)
                }
                
                with self.lock:
                    self.resource_data.append(resource_info)
                    # Son 1000 kayıtı tut
                    if len(self.resource_data) > 1000:
                        self.resource_data = self.resource_data[-1000:]
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Resource monitoring hatası: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Peak kullanım değerlerini al"""
        with self.lock:
            if not self.resource_data:
                return {'cpu_peak': 0, 'memory_peak_gb': 0}
            
            cpu_peak = max(data['cpu_percent'] for data in self.resource_data)
            memory_peak = max(data['memory_used_gb'] for data in self.resource_data)
            
            return {
                'cpu_peak': cpu_peak,
                'memory_peak_gb': memory_peak
            }

class BatchAugmentationProcessor:
    """Batch augmentation işlemcisi"""
    
    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or BatchProcessingConfig()
        self.setup_logging()
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Performance optimizer
        self.optimizer = PerformanceOptimizer() if AUGMENTATION_SYSTEMS_AVAILABLE else None
        
        # Validator
        self.validator = AugmentationValidator() if AUGMENTATION_SYSTEMS_AVAILABLE else None
        
        # Progress tracking
        self.progress_queue = queue.Queue()
        self.error_queue = queue.Queue()
        
        # Temp directory
        if self.config.temp_dir:
            self.temp_dir = Path(self.config.temp_dir)
        else:
            self.temp_dir = Path("temp_batch_processing")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.processing_stats = defaultdict(int)
        
    def setup_logging(self):
        """Logging konfigürasyonu"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('batch_augmentation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def optimize_batch_config(self, total_images: int, image_sample_path: str = None) -> BatchProcessingConfig:
        """Batch konfigürasyonunu optimize et"""
        if not self.optimizer:
            return self.config
        
        # Sistem kaynaklarını kontrol et
        resources = self.optimizer.monitor_system_resources()
        
        # Optimal batch size hesapla
        optimal_batch_size = self.optimizer.optimize_batch_size(total_images)
        
        # Optimal worker sayısı hesapla
        optimal_workers = self.optimizer.calculate_optimal_workers(cpu_intensive=True)
        
        # Örnek görüntü boyutunu kontrol et
        if image_sample_path and os.path.exists(image_sample_path):
            try:
                img = cv2.imread(image_sample_path)
                if img is not None:
                    img_size_mb = img.nbytes / (1024 * 1024)
                    # Görüntü boyutuna göre batch size ayarla
                    if img_size_mb > 10:  # 10MB'dan büyük görüntüler
                        optimal_batch_size = max(1, optimal_batch_size // 4)
                    elif img_size_mb > 5:  # 5MB'dan büyük görüntüler
                        optimal_batch_size = max(1, optimal_batch_size // 2)
            except Exception as e:
                self.logger.warning(f"Görüntü boyutu analizi hatası: {e}")
        
        # Konfigürasyonu güncelle
        optimized_config = BatchProcessingConfig(
            batch_size=min(optimal_batch_size, self.config.batch_size),
            max_workers=min(optimal_workers, self.config.max_workers),
            memory_limit_gb=resources['memory_available_gb'] * 0.8,  # %80 güvenli kullanım
            enable_validation=self.config.enable_validation,
            validation_sample_rate=self.config.validation_sample_rate,
            error_tolerance=self.config.error_tolerance,
            progress_callback=self.config.progress_callback,
            temp_dir=self.config.temp_dir,
            cleanup_temp=self.config.cleanup_temp
        )
        
        self.logger.info(f"🔧 Optimized batch config:")
        self.logger.info(f"  • Batch size: {optimized_config.batch_size}")
        self.logger.info(f"  • Max workers: {optimized_config.max_workers}")
        self.logger.info(f"  • Memory limit: {optimized_config.memory_limit_gb:.1f}GB")
        
        return optimized_config
    
    def create_batch_tasks(self, image_paths: List[str], label_paths: List[str],
                          output_image_dir: str, output_label_dir: str,
                          augmentation_configs: List[str], multiplier: int) -> List[BatchTask]:
        """Batch görevlerini oluştur"""
        tasks = []
        batch_size = self.config.batch_size
        
        # Görüntüleri batch'lere böl
        for i in range(0, len(image_paths), batch_size):
            batch_images = image_paths[i:i + batch_size]
            batch_labels = label_paths[i:i + batch_size]
            
            # Her augmentation tipi için ayrı task
            for aug_type in augmentation_configs:
                task_id = f"batch_{i//batch_size}_{aug_type}"
                
                task = BatchTask(
                    task_id=task_id,
                    image_paths=batch_images,
                    label_paths=batch_labels,
                    output_image_dir=output_image_dir,
                    output_label_dir=output_label_dir,
                    augmentation_type=aug_type,
                    multiplier=multiplier,
                    config={}
                )
                
                tasks.append(task)
        
        return tasks
    
    def process_single_batch(self, task: BatchTask) -> ProcessingResult:
        """Tek batch'i işle"""
        result = ProcessingResult()
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 Processing batch {task.task_id} ({len(task.image_paths)} images)")
            
            # Augmentation sistemi seç
            if task.augmentation_type in ['whitefly', 'aphid', 'thrips', 'spider_mite', 
                                        'hornworm', 'cutworm', 'leafhopper', 'flea_beetle',
                                        'leaf_miner', 'stink_bug']:
                # Geçici dizinler oluştur
                temp_images = self.temp_dir / f"{task.task_id}_images"
                temp_labels = self.temp_dir / f"{task.task_id}_labels"
                temp_images.mkdir(exist_ok=True)
                temp_labels.mkdir(exist_ok=True)
                
                # Dosyaları geçici dizine kopyala
                import shutil
                for img_path, label_path in zip(task.image_paths, task.label_paths):
                    if os.path.exists(img_path):
                        shutil.copy2(img_path, temp_images)
                    if os.path.exists(label_path):
                        shutil.copy2(label_path, temp_labels)
                
                augmenter = TomatoPestAugmentation(
                    str(temp_images), str(temp_labels),
                    task.output_image_dir, task.output_label_dir
                )
                
                # Augmentation işlemini çalıştır
                aug_result = augmenter.augment_pest(
                    task.augmentation_type, 
                    task.multiplier,
                    max_images=len(task.image_paths)
                )
                
                result.processed_images = aug_result['processed_images']
                result.successful_augmentations = aug_result['successful_augmentations']
                
                # Geçici dosyaları temizle
                if self.config.cleanup_temp:
                    shutil.rmtree(temp_images, ignore_errors=True)
                    shutil.rmtree(temp_labels, ignore_errors=True)
            
            result.total_images = len(task.image_paths)
            result.processing_time = time.time() - start_time
            result.avg_time_per_image = result.processing_time / max(1, result.processed_images)
            
            # Memory usage
            process = psutil.Process()
            result.peak_memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
            
            self.logger.info(f"✅ Batch {task.task_id} completed: {result.successful_augmentations} augmentations")
            
        except Exception as e:
            error_msg = f"Batch {task.task_id} error: {str(e)}"
            self.logger.error(error_msg)
            result.error_details.append(error_msg)
            result.failed_augmentations = len(task.image_paths)
        
        return result
    
    def process_dataset_parallel(self, images_dir: str, labels_dir: str,
                               output_images_dir: str, output_labels_dir: str,
                               augmentation_configs: List[str], multiplier: int = 3,
                               optimize_config: bool = True) -> ProcessingResult:
        """Veri setini paralel olarak işle"""
        
        self.logger.info("🚀 Paralel batch augmentation başlatılıyor...")
        
        # Dizinleri hazırla
        Path(output_images_dir).mkdir(exist_ok=True, parents=True)
        Path(output_labels_dir).mkdir(exist_ok=True, parents=True)
        
        # Dosyaları bul
        image_paths, label_paths = self._find_image_label_pairs(images_dir, labels_dir)
        
        if not image_paths:
            self.logger.warning("Hiç görüntü dosyası bulunamadı!")
            return ProcessingResult()
        
        self.logger.info(f"📁 {len(image_paths)} görüntü dosyası bulundu")
        
        # Konfigürasyonu optimize et
        if optimize_config:
            sample_image = image_paths[0] if image_paths else None
            self.config = self.optimize_batch_config(len(image_paths), sample_image)
        
        # Batch görevlerini oluştur
        tasks = self.create_batch_tasks(
            image_paths, label_paths, output_images_dir, output_labels_dir,
            augmentation_configs, multiplier
        )
        
        self.logger.info(f"🎯 {len(tasks)} batch görevi oluşturuldu")
        
        # Resource monitoring başlat
        self.resource_monitor.start_monitoring()
        
        # Paralel işlemi başlat
        overall_result = ProcessingResult()
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Görevleri gönder
                future_to_task = {
                    executor.submit(self.process_single_batch, task): task 
                    for task in tasks
                }
                
                # Sonuçları topla
                completed_tasks = 0
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result(timeout=300)  # 5 dakika timeout
                        
                        # Sonuçları birleştir
                        overall_result.total_images += result.total_images
                        overall_result.processed_images += result.processed_images
                        overall_result.successful_augmentations += result.successful_augmentations
                        overall_result.failed_augmentations += result.failed_augmentations
                        overall_result.skipped_images += result.skipped_images
                        overall_result.error_details.extend(result.error_details)
                        
                        completed_tasks += 1
                        
                        # Progress callback
                        if self.config.progress_callback:
                            progress = completed_tasks / len(tasks)
                            self.config.progress_callback(progress, task.task_id, result)
                        
                        self.logger.info(f"📈 Progress: {completed_tasks}/{len(tasks)} batches completed")
                        
                    except Exception as e:
                        error_msg = f"Task {task.task_id} failed: {str(e)}"
                        self.logger.error(error_msg)
                        overall_result.error_details.append(error_msg)
                        overall_result.failed_augmentations += len(task.image_paths)
        
        except Exception as e:
            self.logger.error(f"Paralel işlem hatası: {str(e)}")
            overall_result.error_details.append(f"Paralel işlem hatası: {str(e)}")
        
        finally:
            # Resource monitoring durdur
            self.resource_monitor.stop_monitoring()
        
        # Timing ve resource bilgileri
        overall_result.processing_time = time.time() - start_time
        if overall_result.processed_images > 0:
            overall_result.avg_time_per_image = overall_result.processing_time / overall_result.processed_images
        
        peak_usage = self.resource_monitor.get_peak_usage()
        overall_result.peak_memory_usage = peak_usage['memory_peak_gb'] * 1024  # MB
        
        # Validation (opsiyonel)
        if self.config.enable_validation and self.validator:
            self.logger.info("🔍 Validation başlatılıyor...")
            try:
                validation_results = self.validator.validate_augmentation_directory(
                    images_dir, output_images_dir, labels_dir, output_labels_dir,
                    parallel=True, max_workers=min(4, self.config.max_workers)
                )
                overall_result.validation_results = validation_results
            except Exception as e:
                self.logger.warning(f"Validation hatası: {str(e)}")
        
        # Sonuç raporu
        self._log_final_results(overall_result, augmentation_configs)
        
        # Geçici dosyaları temizle
        if self.config.cleanup_temp:
            self._cleanup_temp_files()
        
        return overall_result
    
    def _find_image_label_pairs(self, images_dir: str, labels_dir: str) -> Tuple[List[str], List[str]]:
        """Görüntü ve etiket dosyalarını eşleştir"""
        image_paths = []
        label_paths = []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for ext in image_extensions:
            pattern = os.path.join(images_dir, f"*{ext}")
            for img_path in glob.glob(pattern):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(labels_dir, f"{base_name}.txt")
                
                if os.path.exists(label_path):
                    image_paths.append(img_path)
                    label_paths.append(label_path)
        
        return image_paths, label_paths
    
    def _log_final_results(self, result: ProcessingResult, augmentation_configs: List[str]):
        """Final sonuçları logla"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🎉 BATCH AUGMENTATION TAMAMLANDI")
        self.logger.info("="*60)
        
        self.logger.info(f"📊 İşlem Özeti:")
        self.logger.info(f"  • Toplam görüntü: {result.total_images}")
        self.logger.info(f"  • İşlenen görüntü: {result.processed_images}")
        self.logger.info(f"  • Başarılı augmentation: {result.successful_augmentations}")
        self.logger.info(f"  • Başarısız augmentation: {result.failed_augmentations}")
        self.logger.info(f"  • Atlanan görüntü: {result.skipped_images}")
        
        self.logger.info(f"\n⏱️ Performans:")
        self.logger.info(f"  • Toplam süre: {result.processing_time:.2f} saniye")
        self.logger.info(f"  • Görüntü başına ortalama: {result.avg_time_per_image:.3f} saniye")
        self.logger.info(f"  • Peak memory: {result.peak_memory_usage:.1f} MB")
        
        if result.validation_results:
            val_results = result.validation_results
            self.logger.info(f"\n🔍 Validation Sonuçları:")
            self.logger.info(f"  • Doğrulanan görüntü: {val_results.get('total_validated', 0)}")
            self.logger.info(f"  • Geçen görüntü: {val_results.get('passed_images', 0)}")
            self.logger.info(f"  • Başarısız görüntü: {val_results.get('failed_images', 0)}")
            self.logger.info(f"  • Ortalama SSIM: {val_results.get('avg_ssim', 0):.3f}")
            self.logger.info(f"  • Ortalama PSNR: {val_results.get('avg_psnr', 0):.2f} dB")
        
        if result.error_details:
            self.logger.warning(f"\n⚠️ Hatalar ({len(result.error_details)}):")
            for i, error in enumerate(result.error_details[:5], 1):
                self.logger.warning(f"  {i}. {error}")
            if len(result.error_details) > 5:
                self.logger.warning(f"  ... ve {len(result.error_details) - 5} hata daha")
        
        self.logger.info(f"\n🎯 Augmentation Tipleri: {', '.join(augmentation_configs)}")
        self.logger.info("="*60)
    
    def _cleanup_temp_files(self):
        """Geçici dosyaları temizle"""
        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.logger.info("🧹 Geçici dosyalar temizlendi")
        except Exception as e:
            self.logger.warning(f"Geçici dosya temizleme hatası: {str(e)}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """İşlem istatistiklerini al"""
        peak_usage = self.resource_monitor.get_peak_usage()
        
        return {
            'config': {
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'memory_limit_gb': self.config.memory_limit_gb,
                'enable_validation': self.config.enable_validation
            },
            'resource_usage': peak_usage,
            'temp_dir': str(self.temp_dir),
            'validator_enabled': self.validator is not None
        }


def create_sample_batch_config() -> BatchProcessingConfig:
    """Örnek batch konfigürasyonu oluştur"""
    return BatchProcessingConfig(
        batch_size=16,
        max_workers=4,
        memory_limit_gb=8.0,
        enable_validation=True,
        validation_sample_rate=0.1,
        error_tolerance=0.1,
        temp_dir="./temp_batch",
        cleanup_temp=True
    )


# Kullanım örneği
if __name__ == "__main__":
    import logging
    
    # Logging ayarla
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('batch_augmentation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Batch processor oluştur
    config = create_sample_batch_config()
    processor = BatchAugmentationProcessor(config)
    
    # Örnek kullanım
    images_dir = "./data/images"
    labels_dir = "./data/labels"
    output_images_dir = "./data/augmented/images"
    output_labels_dir = "./data/augmented/labels"
    
    # Domates zararlıları için augmentation
    pest_configs = ['whitefly', 'aphid', 'thrips', 'spider_mite']
    
    try:
        result = processor.process_dataset_parallel(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            augmentation_configs=pest_configs,
            multiplier=3,
            optimize_config=True
        )
        
        print(f"\n🎉 Batch augmentation tamamlandı!")
        print(f"📊 {result.successful_augmentations} başarılı augmentation")
        print(f"⏱️ Toplam süre: {result.processing_time:.2f} saniye")
        
        # İstatistikleri kaydet
        stats = processor.get_processing_statistics()
        with open('batch_processing_stats.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        logging.error(f"Batch processing hatası: {str(e)}", exc_info=True)
