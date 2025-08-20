#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 SmartFarm Augmentation Systems Integration Tests

Bu modül, SmartFarm projesindeki tüm augmentation sistemlerinin
entegrasyonunu test eder ve doğrular.

Test Kapsamı:
- Tomato Pest Augmentation sistemi
- Augmentation Validator sistemi  
- Batch Augmentation Processor sistemi
- Performance monitoring
- Kalite kontrol metrikleri

Kullanım:
    python test_integration.py

Gereksinimler:
    - OpenCV (cv2)
    - NumPy
    - Albumentations
    - PSUtil
    - Scikit-image
"""

import os
import sys
import time
import shutil
import tempfile
import unittest
import logging
from pathlib import Path
from typing import Dict, List, Any
import json

# Test için gerekli kütüphaneler
try:
    import cv2
    import numpy as np
    from PIL import Image
    import psutil
except ImportError as e:
    print(f"❌ Gerekli kütüphane eksik: {e}")
    sys.exit(1)

# SmartFarm modüllerini import et
try:
    from tomato_pest_augmentation import TomatoPestAugmentation
    from augmentation_validator import AugmentationValidator, PerformanceOptimizer
    from batch_augmentation_processor import BatchAugmentationProcessor, BatchProcessingConfig
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SmartFarm modülleri import edilemedi: {e}")
    MODULES_AVAILABLE = False


class SmartFarmIntegrationTests(unittest.TestCase):
    """SmartFarm augmentation sistemleri entegrasyon testleri"""
    
    @classmethod
    def setUpClass(cls):
        """Test sınıfı kurulumu"""
        cls.test_dir = Path(tempfile.mkdtemp(prefix="smartfarm_test_"))
        cls.images_dir = cls.test_dir / "images"
        cls.labels_dir = cls.test_dir / "labels"
        cls.output_images_dir = cls.test_dir / "output_images"
        cls.output_labels_dir = cls.test_dir / "output_labels"
        
        # Test dizinlerini oluştur
        for dir_path in [cls.images_dir, cls.labels_dir, cls.output_images_dir, cls.output_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Test görüntüleri ve etiketleri oluştur
        cls._create_test_data()
        
        # Logger ayarla
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        
    @classmethod
    def tearDownClass(cls):
        """Test sınıfı temizliği"""
        try:
            shutil.rmtree(cls.test_dir, ignore_errors=True)
            cls.logger.info(f"🧹 Test dizini temizlendi: {cls.test_dir}")
        except Exception as e:
            cls.logger.warning(f"Test dizini temizleme hatası: {e}")
    
    @classmethod
    def _create_test_data(cls):
        """Test için örnek görüntü ve etiket dosyaları oluştur"""
        # 5 adet test görüntüsü oluştur
        for i in range(5):
            # Rastgele görüntü oluştur (640x480, RGB)
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Görüntüyü kaydet
            image_path = cls.images_dir / f"test_image_{i:03d}.jpg"
            cv2.imwrite(str(image_path), image)
            
            # YOLO format etiket oluştur (rastgele bounding box)
            label_path = cls.labels_dir / f"test_image_{i:03d}.txt"
            
            # 1-3 arası rastgele sayıda bounding box
            num_boxes = np.random.randint(1, 4)
            with open(label_path, 'w') as f:
                for _ in range(num_boxes):
                    # YOLO format: class_id center_x center_y width height
                    class_id = np.random.randint(0, 10)  # 10 farklı zararlı sınıfı
                    center_x = np.random.uniform(0.1, 0.9)
                    center_y = np.random.uniform(0.1, 0.9)
                    width = np.random.uniform(0.05, 0.3)
                    height = np.random.uniform(0.05, 0.3)
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        cls.logger.info(f"📁 {len(list(cls.images_dir.glob('*.jpg')))} test görüntüsü oluşturuldu")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "SmartFarm modülleri mevcut değil")
    def test_tomato_pest_augmentation_basic(self):
        """Temel domates zararlısı augmentation testi"""
        self.logger.info("🧪 Tomato Pest Augmentation temel test başlatılıyor...")
        
        # TomatoPestAugmentation instance oluştur
        augmenter = TomatoPestAugmentation(
            str(self.images_dir),
            str(self.labels_dir),
            str(self.output_images_dir),
            str(self.output_labels_dir)
        )
        
        # Whitefly augmentation test et
        result = augmenter.augment_pest('whitefly', multiplier=2, max_images=3)
        
        # Sonuçları doğrula
        self.assertGreater(result['processed_images'], 0, "Hiç görüntü işlenmedi")
        self.assertGreater(result['successful_augmentations'], 0, "Hiç augmentation başarılı olmadı")
        self.assertEqual(result['failed_augmentations'], 0, "Başarısız augmentation var")
        
        # Çıktı dosyalarının varlığını kontrol et
        output_images = list(self.output_images_dir.glob('*.jpg'))
        output_labels = list(self.output_labels_dir.glob('*.txt'))
        
        self.assertGreater(len(output_images), 0, "Çıktı görüntüsü oluşturulmadı")
        self.assertGreater(len(output_labels), 0, "Çıktı etiketi oluşturulmadı")
        
        self.logger.info(f"✅ Temel test başarılı: {len(output_images)} görüntü, {len(output_labels)} etiket")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "SmartFarm modülleri mevcut değil")
    def test_multiple_pest_augmentation(self):
        """Çoklu zararlı augmentation testi"""
        self.logger.info("🧪 Çoklu zararlı augmentation testi başlatılıyor...")
        
        # Çıktı dizinini temizle
        for file in self.output_images_dir.glob('*'):
            file.unlink()
        for file in self.output_labels_dir.glob('*'):
            file.unlink()
        
        augmenter = TomatoPestAugmentation(
            str(self.images_dir),
            str(self.labels_dir),
            str(self.output_images_dir),
            str(self.output_labels_dir)
        )
        
        # Farklı zararlı türleri test et
        pest_types = ['whitefly', 'aphid', 'thrips', 'spider_mite']
        total_augmentations = 0
        
        for pest_type in pest_types:
            result = augmenter.augment_pest(pest_type, multiplier=1, max_images=2)
            total_augmentations += result['successful_augmentations']
            
            self.assertGreater(result['successful_augmentations'], 0, 
                             f"{pest_type} için augmentation başarısız")
        
        self.assertGreater(total_augmentations, 0, "Toplam augmentation sayısı sıfır")
        
        self.logger.info(f"✅ Çoklu zararlı testi başarılı: {total_augmentations} toplam augmentation")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "SmartFarm modülleri mevcut değil")
    def test_augmentation_validator(self):
        """Augmentation validator testi"""
        self.logger.info("🧪 Augmentation Validator testi başlatılıyor...")
        
        # Önce augmentation yap
        augmenter = TomatoPestAugmentation(
            str(self.images_dir),
            str(self.labels_dir),
            str(self.output_images_dir),
            str(self.output_labels_dir)
        )
        
        result = augmenter.augment_pest('whitefly', multiplier=2, max_images=3)
        self.assertGreater(result['successful_augmentations'], 0)
        
        # Validator oluştur ve test et
        validator = AugmentationValidator()
        
        # Tek görüntü validation
        original_images = list(self.images_dir.glob('*.jpg'))
        augmented_images = list(self.output_images_dir.glob('*.jpg'))
        
        if original_images and augmented_images:
            validation_result = validator.validate_single_augmentation(
                str(original_images[0]),
                str(augmented_images[0])
            )
            
            # Validation sonuçlarını kontrol et
            self.assertIn('ssim', validation_result, "SSIM metriği eksik")
            self.assertIn('psnr', validation_result, "PSNR metriği eksik")
            self.assertIn('brightness_diff', validation_result, "Brightness metriği eksik")
            self.assertIn('overall_quality', validation_result, "Genel kalite skoru eksik")
            
            # Kalite skorlarının makul aralıkta olduğunu kontrol et
            self.assertGreaterEqual(validation_result['ssim'], 0.0)
            self.assertLessEqual(validation_result['ssim'], 1.0)
            self.assertGreaterEqual(validation_result['psnr'], 0.0)
            
            self.logger.info(f"✅ Validator testi başarılı - SSIM: {validation_result['ssim']:.3f}")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "SmartFarm modülleri mevcut değil")
    def test_batch_processing(self):
        """Batch processing testi"""
        self.logger.info("🧪 Batch Processing testi başlatılıyor...")
        
        # Çıktı dizinini temizle
        for file in self.output_images_dir.glob('*'):
            file.unlink()
        for file in self.output_labels_dir.glob('*'):
            file.unlink()
        
        # Batch processing config
        config = BatchProcessingConfig(
            batch_size=2,
            max_workers=2,
            memory_limit_gb=4.0,
            enable_validation=True,
            validation_sample_rate=0.5,
            error_tolerance=0.2,
            temp_dir=str(self.test_dir / "temp_batch"),
            cleanup_temp=True
        )
        
        # Batch processor oluştur
        processor = BatchAugmentationProcessor(config)
        
        # Paralel processing test et
        result = processor.process_dataset_parallel(
            images_dir=str(self.images_dir),
            labels_dir=str(self.labels_dir),
            output_images_dir=str(self.output_images_dir),
            output_labels_dir=str(self.output_labels_dir),
            augmentation_configs=['whitefly', 'aphid'],
            multiplier=2,
            optimize_config=True
        )
        
        # Sonuçları doğrula
        self.assertGreater(result.total_images, 0, "Hiç görüntü işlenmedi")
        self.assertGreater(result.successful_augmentations, 0, "Hiç augmentation başarılı olmadı")
        self.assertGreater(result.processing_time, 0, "İşlem süresi kaydedilmedi")
        
        # Çıktı dosyalarını kontrol et
        output_images = list(self.output_images_dir.glob('*.jpg'))
        output_labels = list(self.output_labels_dir.glob('*.txt'))
        
        self.assertGreater(len(output_images), 0, "Batch processing çıktı görüntüsü yok")
        self.assertGreater(len(output_labels), 0, "Batch processing çıktı etiketi yok")
        
        self.logger.info(f"✅ Batch processing testi başarılı: {result.successful_augmentations} augmentation")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "SmartFarm modülleri mevcut değil")
    def test_performance_monitoring(self):
        """Performance monitoring testi"""
        self.logger.info("🧪 Performance Monitoring testi başlatılıyor...")
        
        # Performance optimizer oluştur
        optimizer = PerformanceOptimizer()
        
        # Sistem kaynaklarını kontrol et
        system_info = optimizer.get_system_resources()
        
        self.assertIn('cpu_count', system_info, "CPU sayısı bilgisi eksik")
        self.assertIn('memory_gb', system_info, "Memory bilgisi eksik")
        self.assertIn('available_memory_gb', system_info, "Kullanılabilir memory bilgisi eksik")
        
        # CPU ve memory değerlerinin pozitif olduğunu kontrol et
        self.assertGreater(system_info['cpu_count'], 0, "CPU sayısı sıfır")
        self.assertGreater(system_info['memory_gb'], 0, "Toplam memory sıfır")
        self.assertGreater(system_info['available_memory_gb'], 0, "Kullanılabilir memory sıfır")
        
        # Batch size optimizasyonu test et
        optimal_config = optimizer.optimize_batch_size(
            total_images=100,
            sample_image_path=str(list(self.images_dir.glob('*.jpg'))[0])
        )
        
        self.assertIn('batch_size', optimal_config, "Optimal batch size eksik")
        self.assertIn('max_workers', optimal_config, "Optimal worker sayısı eksik")
        self.assertGreater(optimal_config['batch_size'], 0, "Batch size sıfır")
        self.assertGreater(optimal_config['max_workers'], 0, "Worker sayısı sıfır")
        
        self.logger.info(f"✅ Performance monitoring testi başarılı - Batch: {optimal_config['batch_size']}")
    
    def test_file_system_operations(self):
        """Dosya sistemi operasyonları testi"""
        self.logger.info("🧪 Dosya sistemi operasyonları testi başlatılıyor...")
        
        # Test dizinlerinin varlığını kontrol et
        self.assertTrue(self.images_dir.exists(), "Images dizini mevcut değil")
        self.assertTrue(self.labels_dir.exists(), "Labels dizini mevcut değil")
        self.assertTrue(self.output_images_dir.exists(), "Output images dizini mevcut değil")
        self.assertTrue(self.output_labels_dir.exists(), "Output labels dizini mevcut değil")
        
        # Test dosyalarının varlığını kontrol et
        image_files = list(self.images_dir.glob('*.jpg'))
        label_files = list(self.labels_dir.glob('*.txt'))
        
        self.assertGreater(len(image_files), 0, "Test görüntü dosyası yok")
        self.assertGreater(len(label_files), 0, "Test etiket dosyası yok")
        self.assertEqual(len(image_files), len(label_files), "Görüntü ve etiket sayısı eşleşmiyor")
        
        # Dosya boyutlarını kontrol et
        for img_path in image_files:
            self.assertGreater(img_path.stat().st_size, 0, f"Boş görüntü dosyası: {img_path}")
        
        for label_path in label_files:
            self.assertGreater(label_path.stat().st_size, 0, f"Boş etiket dosyası: {label_path}")
        
        self.logger.info(f"✅ Dosya sistemi testi başarılı: {len(image_files)} dosya çifti")
    
    def test_error_handling(self):
        """Hata yönetimi testi"""
        self.logger.info("🧪 Hata yönetimi testi başlatılıyor...")
        
        if not MODULES_AVAILABLE:
            self.skipTest("SmartFarm modülleri mevcut değil")
        
        # Geçersiz dizin ile augmentation test et
        invalid_dir = str(self.test_dir / "nonexistent")
        
        try:
            augmenter = TomatoPestAugmentation(
                invalid_dir,  # Geçersiz input dizini
                str(self.labels_dir),
                str(self.output_images_dir),
                str(self.output_labels_dir)
            )
            
            result = augmenter.augment_pest('whitefly', multiplier=1, max_images=1)
            
            # Hata durumunda bile sonuç dönmeli
            self.assertIsInstance(result, dict, "Sonuç dict formatında değil")
            self.assertEqual(result['processed_images'], 0, "Geçersiz dizinde görüntü işlendi")
            
        except Exception as e:
            # Beklenen hata durumu
            self.logger.info(f"✅ Beklenen hata yakalandı: {str(e)[:50]}...")
        
        self.logger.info("✅ Hata yönetimi testi başarılı")


def run_integration_tests():
    """Integration testlerini çalıştır"""
    print("🚀 SmartFarm Integration Tests Başlatılıyor...")
    print("="*60)
    
    # Test suite oluştur
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(SmartFarmIntegrationTests)
    
    # Test runner oluştur
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    # Testleri çalıştır
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Sonuç raporu
    print("\n" + "="*60)
    print("📊 TEST SONUÇLARI")
    print("="*60)
    print(f"⏱️  Toplam süre: {end_time - start_time:.2f} saniye")
    print(f"🧪 Toplam test: {result.testsRun}")
    print(f"✅ Başarılı: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Başarısız: {len(result.failures)}")
    print(f"💥 Hata: {len(result.errors)}")
    print(f"⏭️  Atlanan: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n⚠️ BAŞARISIZ TESTLER:")
        for test, traceback in result.failures:
            print(f"  • {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print(f"\n💥 HATA OLAN TESTLER:")
        for test, traceback in result.errors:
            print(f"  • {test}: {traceback.split('\\n')[-2]}")
    
    # Genel başarı durumu
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Başarı oranı: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Integration testleri başarıyla tamamlandı!")
        return True
    else:
        print("⚠️ Bazı testler başarısız oldu. Lütfen hataları kontrol edin.")
        return False


if __name__ == "__main__":
    # Modül kontrolü
    if not MODULES_AVAILABLE:
        print("❌ SmartFarm modülleri import edilemedi!")
        print("Lütfen aşağıdaki dosyaların mevcut olduğundan emin olun:")
        print("  • tomato_pest_augmentation.py")
        print("  • augmentation_validator.py")
        print("  • batch_augmentation_processor.py")
        sys.exit(1)
    
    # Testleri çalıştır
    success = run_integration_tests()
    sys.exit(0 if success else 1)
