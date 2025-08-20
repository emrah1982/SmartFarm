#!/usr/bin/env python3
# quick_colab_fix.py - Hızlı Colab kurulum düzeltmesi

import subprocess
import sys

def quick_fix_colab():
    """Colab'da numpy sorununu hızlıca çöz"""
    print("🔧 Colab numpy sorunu düzeltiliyor...")
    
    # Kritik paketleri tek tek yükle
    packages = [
        # Önce numpy'i düzelt
        "numpy>=1.21.0,<2.0.0",
        
        # Sonra diğer temel paketler
        "ultralytics>=8.2.0",
        "opencv-python-headless",
        "albumentations",
        
        # Google Drive API
        "google-api-python-client",
        "google-auth-oauthlib",
    ]
    
    for package in packages:
        try:
            print(f"📦 Yükleniyor: {package}")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "--no-deps", package
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"✅ {package} başarılı")
            else:
                print(f"⚠️ {package} sorunlu, devam ediliyor...")
                
        except Exception as e:
            print(f"❌ {package} hatası: {e}")
    
    print("\n🔄 Runtime'ı yeniden başlatın: Runtime → Restart runtime")

if __name__ == "__main__":
    quick_fix_colab()
