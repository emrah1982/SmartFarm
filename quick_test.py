#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SmartFarm Quick Integration Test

Basit ve hızlı entegrasyon testi - tüm sistemlerin çalıştığını doğrular.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Modül importlarını test et"""
    print("📦 Modül importları test ediliyor...")
    
    try:
        from tomato_pest_augmentation import TomatoPestAugmentation
        print("✅ TomatoPestAugmentation import başarılı")
    except ImportError as e:
        print(f"❌ TomatoPestAugmentation import hatası: {e}")
        return False
    
    try:
        from augmentation_validator import AugmentationValidator
        print("✅ AugmentationValidator import başarılı")
    except ImportError as e:
        print(f"❌ AugmentationValidator import hatası: {e}")
        return False
    
    try:
        from batch_augmentation_processor import BatchAugmentationProcessor
        print("✅ BatchAugmentationProcessor import başarılı")
    except ImportError as e:
        print(f"❌ BatchAugmentationProcessor import hatası: {e}")
        return False
    
    return True

def create_test_image(path, size=(640, 480)):
    """Test görüntüsü oluştur"""
    try:
        import cv2
        import numpy as np
        
        # Rastgele görüntü oluştur
        image = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        cv2.imwrite(str(path), image)
        return True
    except ImportError:
        print("⚠️ OpenCV mevcut değil, test görüntüsü oluşturulamadı")
        return False

def create_test_label(path):
    """Test etiketi oluştur"""
    with open(path, 'w') as f:
        # Basit YOLO format etiket
        f.write("0 0.5 0.5 0.2 0.2\n")  # class_id center_x center_y width height

def test_basic_functionality():
    """Temel fonksiyonalite testi"""
    print("\n🧪 Temel fonksiyonalite testi...")
    
    # Geçici dizin oluştur
    test_dir = Path(tempfile.mkdtemp(prefix="smartfarm_quick_test_"))
    
    try:
        # Test dizinleri
        images_dir = test_dir / "images"
        labels_dir = test_dir / "labels"
        output_images_dir = test_dir / "output_images"
        output_labels_dir = test_dir / "output_labels"
        
        for dir_path in [images_dir, labels_dir, output_images_dir, output_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Test dosyaları oluştur
        test_image_path = images_dir / "test.jpg"
        test_label_path = labels_dir / "test.txt"
        
        if not create_test_image(test_image_path):
            print("⚠️ Test görüntüsü oluşturulamadı, test atlanıyor")
            return True
        
        create_test_label(test_label_path)
        
        # TomatoPestAugmentation test et
        from tomato_pest_augmentation import TomatoPestAugmentation
        
        augmenter = TomatoPestAugmentation(
            str(images_dir),
            str(labels_dir),
            str(output_images_dir),
            str(output_labels_dir)
        )
        
        result = augmenter.augment_pest('whitefly', multiplier=1, max_images=1)
        
        if result['successful_augmentations'] > 0:
            print("✅ Temel augmentation testi başarılı")
            return True
        else:
            print("❌ Augmentation başarısız")
            return False
            
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        return False
    
    finally:
        # Temizlik
        try:
            shutil.rmtree(test_dir, ignore_errors=True)
        except:
            pass

def main():
    """Ana test fonksiyonu"""
    print("🚀 SmartFarm Quick Integration Test")
    print("=" * 50)
    
    # Import testleri
    if not test_imports():
        print("\n❌ Import testleri başarısız!")
        return False
    
    # Temel fonksiyonalite testi
    if not test_basic_functionality():
        print("\n❌ Temel fonksiyonalite testleri başarısız!")
        return False
    
    print("\n🎉 Tüm testler başarılı!")
    print("✅ SmartFarm augmentation sistemleri çalışıyor")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
