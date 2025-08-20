#!/usr/bin/env python3
# colab_setup.py - Google Colab için özel kurulum scripti

import subprocess
import sys
import os
import pkg_resources
from packaging import version

def is_colab():
    """Google Colab ortamında mı kontrol et"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_installed_version(package_name):
    """Yüklü paket versiyonunu al"""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def install_package(package_spec, force_reinstall=False):
    """Paketi güvenli şekilde yükle"""
    try:
        if force_reinstall:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", package_spec]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package_spec]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {package_spec} başarıyla yüklendi")
            return True
        else:
            print(f"❌ {package_spec} yüklenemedi: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {package_spec} yükleme zaman aşımı")
        return False
    except Exception as e:
        print(f"❌ {package_spec} yükleme hatası: {e}")
        return False

def setup_colab_environment():
    """Colab ortamını SmartFarm için hazırla"""
    print("🚀 Google Colab SmartFarm Kurulumu Başlıyor...")
    print("=" * 50)
    
    if not is_colab():
        print("⚠️ Bu script sadece Google Colab için tasarlanmıştır!")
        return False
    
    # Colab'da önceden yüklü paketleri kontrol et
    print("\n📋 Mevcut paket versiyonları:")
    critical_packages = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'opencv-python', 'pillow', 'requests', 'tqdm'
    ]
    
    for pkg in critical_packages:
        ver = get_installed_version(pkg)
        if ver:
            print(f"  {pkg}: {ver}")
        else:
            print(f"  {pkg}: Yüklü değil")
    
    # Gerekli paketleri yükle
    print("\n📦 Gerekli paketleri yükleniyor...")
    
    packages_to_install = [
        # YOLO ve ML kütüphaneleri
        "ultralytics>=8.2.0",
        
        # Görüntü işleme (opencv-python yerine headless versiyon)
        "opencv-python-headless>=4.7.0",
        
        # Augmentation
        "albumentations>=1.3.0",
        
        # Colab'da sorun çıkarabilecek paketler için özel versiyonlar
        "psutil>=5.8.0",
        "pyyaml>=6.0",
        
        # Google Drive API (Colab'da genelde yüklü ama güncel versiyon için)
        "google-api-python-client>=2.70.0",
        "google-auth-httplib2>=0.1.0",
        "google-auth-oauthlib>=0.8.0",
    ]
    
    failed_packages = []
    
    for package in packages_to_install:
        print(f"\n🔄 Yükleniyor: {package}")
        if not install_package(package):
            failed_packages.append(package)
    
    # Numpy özel kontrolü (Colab'da sorun çıkarabilir)
    print("\n🔍 NumPy versiyonu kontrol ediliyor...")
    numpy_ver = get_installed_version('numpy')
    if numpy_ver:
        if version.parse(numpy_ver) >= version.parse('2.0.0'):
            print("⚠️ NumPy 2.0+ tespit edildi, 1.x'e downgrade yapılıyor...")
            if install_package("numpy>=1.21.0,<2.0.0", force_reinstall=True):
                print("✅ NumPy başarıyla downgrade edildi")
            else:
                print("❌ NumPy downgrade başarısız")
                failed_packages.append("numpy<2.0.0")
    
    # Sonuçları raporla
    print("\n" + "=" * 50)
    if failed_packages:
        print("❌ Bazı paketler yüklenemedi:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\n💡 Bu paketleri manuel olarak yüklemeyi deneyin:")
        for pkg in failed_packages:
            print(f"  !pip install {pkg}")
    else:
        print("✅ Tüm paketler başarıyla yüklendi!")
    
    # Restart runtime uyarısı
    print("\n🔄 ÖNEMLI: Kurulum tamamlandıktan sonra runtime'ı yeniden başlatın!")
    print("   Runtime → Restart runtime")
    
    return len(failed_packages) == 0

def verify_installation():
    """Kurulumun başarılı olup olmadığını kontrol et"""
    print("\n🔍 Kurulum doğrulaması...")
    
    test_imports = [
        ('ultralytics', 'YOLO'),
        ('cv2', 'OpenCV'),
        ('albumentations', 'Albumentations'),
        ('google.auth', 'Google Auth'),
        ('googleapiclient.discovery', 'Google API Client'),
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} import başarılı")
        except ImportError as e:
            print(f"❌ {name} import başarısız: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} modül import edilemedi")
        return False
    else:
        print("\n✅ Tüm modüller başarıyla import edildi!")
        return True

def setup_drive_credentials():
    """Google Drive credentials kurulumu için rehber"""
    print("\n📁 Google Drive Kurulum Rehberi:")
    print("1. Google Cloud Console'a gidin: https://console.cloud.google.com/")
    print("2. Yeni proje oluşturun")
    print("3. Google Drive API'yi etkinleştirin")
    print("4. OAuth 2.0 credentials oluşturun (Desktop app)")
    print("5. credentials.json dosyasını indirin")
    print("6. Colab'a yükleyin:")
    print("   from google.colab import files")
    print("   files.upload()  # credentials.json seçin")

if __name__ == "__main__":
    print("🌱 SmartFarm Colab Setup")
    print("Bu script Google Colab ortamını SmartFarm için hazırlar")
    
    if setup_colab_environment():
        print("\n🎉 Kurulum başarılı!")
        verify_installation()
        setup_drive_credentials()
    else:
        print("\n💥 Kurulum sırasında hatalar oluştu")
        print("Lütfen hataları düzeltin ve tekrar deneyin")
