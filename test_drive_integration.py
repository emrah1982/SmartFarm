#!/usr/bin/env python3
"""
Drive entegrasyonu mini test scripti
Drive klasör yapısı ve dosya kaydetme işlevselliğini test eder
"""

import os
import sys
import tempfile
from pathlib import Path

# Proje modüllerini import et (yalnızca DriveManager, torch bağımlılığını tetiklememek için training'i import etmiyoruz)
try:
    from drive_manager import DriveManager
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("Bu scripti SmartFarm ana klasöründe çalıştırın.")
    sys.exit(1)

def create_test_files():
    """Test için geçici dosyalar oluştur"""
    test_dir = Path("test_temp")
    test_dir.mkdir(exist_ok=True)
    
    # Sahte model dosyaları oluştur
    (test_dir / "test_best.pt").write_text("fake best model data")
    (test_dir / "test_last.pt").write_text("fake last model data")
    (test_dir / "test_config.yaml").write_text("fake config data")
    
    print(f"✅ Test dosyaları oluşturuldu: {test_dir}")
    return test_dir

def test_drive_setup():
    """Drive kurulumu ve klasör yapısı testi"""
    print("\n🔧 Drive Manager Testi Başlıyor...")
    
    dm = DriveManager()
    
    # 1) Drive kimlik doğrulama/mount
    print("1️⃣ Drive kimlik doğrulama...")
    if not dm.authenticate():
        print("❌ Drive kimlik doğrulama başarısız!")
        return None
    print("✅ Drive kimlik doğrulama başarılı!")
    
    # 2) Klasör kurulumu (otomatik mod)
    print("2️⃣ Klasör kurulumu...")
    # Otomatik kurulum için input'u simulate et
    import builtins
    original_input = builtins.input
    builtins.input = lambda prompt: "e"  # Otomatik kurulum seç
    
    try:
        if not dm._setup_colab_folder():
            print("❌ Klasör kurulumu başarısız!")
            return None
        print(f"✅ Klasör kurulumu başarılı: {dm.project_folder}")
    finally:
        builtins.input = original_input
    
    return dm

def test_file_uploads(dm, test_dir):
    """Dosya yükleme testi"""
    print("\n📤 Dosya Yükleme Testi...")
    
    test_files = [
        (test_dir / "test_best.pt", "models/test_best.pt"),
        (test_dir / "test_last.pt", "models/test_last.pt"),
        (test_dir / "test_config.yaml", "configs/test_config.yaml"),
    ]
    
    success_count = 0
    for local_file, drive_path in test_files:
        print(f"📁 Yükleniyor: {local_file} → {drive_path}")
        if dm.upload_file(str(local_file), drive_path):
            print(f"✅ Başarılı: {drive_path}")
            success_count += 1
        else:
            print(f"❌ Başarısız: {drive_path}")
    
    print(f"\n📊 Yükleme Sonucu: {success_count}/{len(test_files)} başarılı")
    return success_count == len(test_files)

def verify_folder_structure(dm):
    """Klasör yapısını doğrula"""
    print("\n🔍 Klasör Yapısı Doğrulama...")
    
    expected_folders = ["models", "logs", "checkpoints", "configs"]
    
    for folder in expected_folders:
        folder_path = os.path.join(dm.project_folder, folder)
        if os.path.exists(folder_path):
            print(f"✅ {folder}/ klasörü mevcut")
        else:
            print(f"❌ {folder}/ klasörü eksik")
    
    # Yüklenen dosyaları kontrol et
    print("\n📋 Yüklenen Dosyalar:")
    for root, dirs, files in os.walk(dm.project_folder):
        level = root.replace(dm.project_folder, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

def cleanup_test_files(test_dir):
    """Test dosyalarını temizle"""
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"🗑️ Test dosyaları temizlendi: {test_dir}")

def main():
    """Ana test fonksiyonu"""
    print("🌱 SmartFarm Drive Entegrasyonu Mini Test")
    print("=" * 50)
    
    test_dir = None
    dm = None
    
    try:
        # Test dosyaları oluştur
        test_dir = create_test_files()
        
        # Drive kurulumu test et
        dm = test_drive_setup()
        if not dm:
            print("❌ Drive kurulumu başarısız, test sonlandırılıyor.")
            return False
        
        # Dosya yükleme test et
        upload_success = test_file_uploads(dm, test_dir)
        
        # Klasör yapısını doğrula
        verify_folder_structure(dm)
        
        # Sonuç
        if upload_success:
            print("\n🎉 Tüm testler başarılı!")
            print(f"📁 Drive klasörü: {dm.project_folder}")
            print("✅ Drive entegrasyonu çalışıyor.")
            return True
        else:
            print("\n⚠️ Bazı testler başarısız!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test hatası: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Temizlik
        if test_dir:
            cleanup_test_files(test_dir)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
