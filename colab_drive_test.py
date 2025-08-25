#!/usr/bin/env python3
# colab_drive_test.py - Colab ortamında Google Drive kaydetme sorunlarını test et

"""
Colab Google Drive Test Scripti
===============================

Bu script Colab ortamında Google Drive kaydetme sorunlarını tespit etmek için kullanılır.

Kullanım:
1. Colab'de bu scripti çalıştırın
2. Debug raporu ve test sonuçlarını inceleyin
3. Sorunları tespit edin ve çözümler uygulayın

Örnek Kullanım:
```python
from colab_drive_test import run_full_test
run_full_test()
```
"""

import os
import sys
from pathlib import Path

# Drive manager'ı import et
try:
    from drive_manager import DriveManager, debug_colab_environment, test_drive_operations
except ImportError:
    print("❌ drive_manager modülü bulunamadı!")
    print("Bu scripti SmartFarm ana dizininde çalıştırın.")
    sys.exit(1)

def quick_colab_check():
    """Hızlı Colab ortam kontrolü"""
    print("🚀 Hızlı Colab Kontrol")
    print("=" * 30)
    
    # 1. Colab tespiti
    try:
        from google.colab import drive
        print("✅ Google Colab ortamı tespit edildi")
        colab_detected = True
    except ImportError:
        print("❌ Google Colab ortamı tespit edilemedi")
        colab_detected = False
    
    # 2. Drive mount kontrolü
    drive_mounted = os.path.exists('/content/drive/MyDrive')
    print(f"📁 Drive mount durumu: {'✅ Mount edilmiş' if drive_mounted else '❌ Mount edilmemiş'}")
    
    # 3. Yazma izni kontrolü
    if drive_mounted:
        try:
            test_file = '/content/drive/MyDrive/test_write_permission.txt'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print("✅ Drive yazma izni var")
            write_permission = True
        except Exception as e:
            print(f"❌ Drive yazma izni yok: {e}")
            write_permission = False
    else:
        write_permission = False
    
    return {
        'colab_detected': colab_detected,
        'drive_mounted': drive_mounted,
        'write_permission': write_permission
    }

def test_model_upload_simulation():
    """Model yükleme simülasyonu"""
    print("\n🧪 Model Yükleme Simülasyonu")
    print("=" * 40)
    
    # Sahte model dosyası oluştur
    temp_model_path = '/tmp/test_model.pt'
    try:
        with open(temp_model_path, 'w') as f:
            f.write('Bu sahte bir model dosyasıdır - test amaçlı')
        
        print(f"✅ Test model dosyası oluşturuldu: {temp_model_path}")
        
        # Drive Manager ile yükleme testi
        dm = DriveManager()
        
        if dm.authenticate():
            print("✅ Drive kimlik doğrulama başarılı")
            
            # Otomatik klasör kurulumu için test
            if dm.is_colab and dm.is_mounted:
                # Test klasörü oluştur
                test_project_folder = os.path.join(dm.base_drive_path, 'SmartFarm_Test_Upload')
                os.makedirs(test_project_folder, exist_ok=True)
                os.makedirs(os.path.join(test_project_folder, 'models'), exist_ok=True)
                
                dm.project_folder = test_project_folder
                
                # Model yükleme testi
                success = dm.upload_model(temp_model_path, 'test_model.pt')
                
                if success:
                    print("✅ Model yükleme testi başarılı")
                    
                    # Temizlik
                    uploaded_file = os.path.join(test_project_folder, 'models', 'test_model.pt')
                    if os.path.exists(uploaded_file):
                        os.remove(uploaded_file)
                    
                    models_dir = os.path.join(test_project_folder, 'models')
                    if os.path.exists(models_dir) and not os.listdir(models_dir):
                        os.rmdir(models_dir)
                    
                    if os.path.exists(test_project_folder) and not os.listdir(test_project_folder):
                        os.rmdir(test_project_folder)
                    print("🧹 Test dosyaları temizlendi")
                else:
                    print("❌ Model yükleme testi başarısız")
            else:
                print("❌ Drive mount edilmemiş, model yükleme testi yapılamadı")
        else:
            print("❌ Drive kimlik doğrulama başarısız")
        
        # Temp dosyayı temizle
        os.remove(temp_model_path)
        
    except Exception as e:
        print(f"❌ Model yükleme simülasyon hatası: {e}")

def diagnose_common_issues():
    """Yaygın sorunları teşhis et"""
    print("\n🔍 Yaygın Sorun Teşhisi")
    print("=" * 30)
    
    issues_found = []
    solutions = []
    
    # 1. Colab ortam kontrolü
    try:
        from google.colab import drive
    except ImportError:
        issues_found.append("Google Colab kütüphanesi bulunamadı")
        solutions.append("Bu scripti Google Colab ortamında çalıştırın")
    
    # 2. Drive mount kontrolü
    if not os.path.exists('/content/drive'):
        issues_found.append("Drive mount edilmemiş")
        solutions.append("from google.colab import drive; drive.mount('/content/drive') komutunu çalıştırın")
    
    # 3. MyDrive klasörü kontrolü
    if not os.path.exists('/content/drive/MyDrive'):
        issues_found.append("MyDrive klasörü bulunamadı")
        solutions.append("Drive mount işlemini tamamlayın ve Google hesabınızı doğrulayın")
    
    # 4. Yazma izni kontrolü
    if os.path.exists('/content/drive/MyDrive'):
        try:
            test_file = '/content/drive/MyDrive/.write_test'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception:
            issues_found.append("Drive yazma izni yok")
            solutions.append("Google hesabınızın Drive erişim iznini kontrol edin")
    
    # Sonuçları göster
    if issues_found:
        print("❌ Tespit edilen sorunlar:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
        
        print("\n💡 Önerilen çözümler:")
        for i, solution in enumerate(solutions, 1):
            print(f"   {i}. {solution}")
    else:
        print("✅ Yaygın sorun tespit edilmedi")
    
    return issues_found, solutions

def run_full_test():
    """Tam test süitini çalıştır"""
    print("🚀 Colab Google Drive Tam Test Süiti")
    print("=" * 50)
    
    # 1. Hızlı kontrol
    quick_results = quick_colab_check()
    
    # 2. Detaylı debug
    print("\n" + "="*50)
    debug_colab_environment()
    
    # 3. Drive işlemleri testi
    print("\n" + "="*50)
    test_drive_operations()
    
    # 4. Model yükleme simülasyonu
    if quick_results['colab_detected'] and quick_results['drive_mounted']:
        test_model_upload_simulation()
    
    # 5. Sorun teşhisi
    print("\n" + "="*50)
    issues, solutions = diagnose_common_issues()
    
    # 6. Özet rapor
    print("\n" + "="*50)
    print("📋 TEST SONUÇ ÖZETİ")
    print("=" * 20)
    
    print(f"🔍 Colab Tespit: {'✅' if quick_results['colab_detected'] else '❌'}")
    print(f"📁 Drive Mount: {'✅' if quick_results['drive_mounted'] else '❌'}")
    print(f"✏️ Yazma İzni: {'✅' if quick_results['write_permission'] else '❌'}")
    print(f"⚠️ Tespit Edilen Sorun: {len(issues)} adet")
    
    if all(quick_results.values()) and not issues:
        print("\n🎉 Tüm testler başarılı! Google Drive kaydetme çalışmalı.")
    else:
        print("\n⚠️ Sorunlar tespit edildi. Yukarıdaki çözümleri uygulayın.")

def run_quick_fix():
    """Hızlı sorun giderme"""
    print("🔧 Hızlı Sorun Giderme")
    print("=" * 25)
    
    try:
        # Drive mount
        from google.colab import drive
        print("🔄 Drive mount işlemi başlatılıyor...")
        drive.mount('/content/drive', force_remount=True)
        
        # Test
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Drive başarıyla mount edildi")
            
            # Yazma testi
            test_file = '/content/drive/MyDrive/.smartfarm_test'
            with open(test_file, 'w') as f:
                f.write('SmartFarm test')
            os.remove(test_file)
            print("✅ Yazma izni doğrulandı")
            
            return True
        else:
            print("❌ Drive mount başarısız")
            return False
            
    except Exception as e:
        print(f"❌ Hızlı düzeltme hatası: {e}")
        return False

if __name__ == "__main__":
    print("Colab Drive Test Scripti")
    print("Kullanılabilir fonksiyonlar:")
    print("- run_full_test(): Tam test süiti")
    print("- quick_colab_check(): Hızlı kontrol")
    print("- run_quick_fix(): Hızlı sorun giderme")
    print("- diagnose_common_issues(): Sorun teşhisi")
    
    # Etkileşimli mod
    print("\nHangi testi çalıştırmak istiyorsunuz?")
    print("1. Tam test süiti (önerilen)")
    print("2. Hızlı kontrol")
    print("3. Hızlı sorun giderme")
    
    try:
        choice = input("Seçiminiz (1-3): ").strip()
        
        if choice == "1":
            run_full_test()
        elif choice == "2":
            quick_colab_check()
        elif choice == "3":
            run_quick_fix()
        else:
            print("Geçersiz seçim, tam test çalıştırılıyor...")
            run_full_test()
            
    except (EOFError, KeyboardInterrupt):
        print("\nTam test çalıştırılıyor...")
        run_full_test()
