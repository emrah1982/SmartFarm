#!/usr/bin/env python3
# test_drive.py - Google Drive entegrasyon testi

import os
from drive_manager import DriveManager, setup_drive_integration

def main():
    print("🔍 Google Drive Entegrasyon Testi")
    print("=" * 50)
    
    # 1. Drive Manager oluştur
    print("\n1. Drive Manager oluşturuluyor...")
    drive_manager = DriveManager()
    
    # 2. Kimlik doğrulama
    print("\n2. Kimlik doğrulama yapılıyor...")
    if not drive_manager.authenticate():
        print("❌ Kimlik doğrulama başarısız!")
        return
    
    print("✅ Kimlik doğrulama başarılı!")
    
    # 3. Mevcut konfigürasyonu yükle
    print("\n3. Mevcut konfigürasyon kontrol ediliyor...")
    if drive_manager.load_drive_config():
        print("✅ Mevcut konfigürasyon yüklendi!")
        print(f"   - Klasör ID: {drive_manager.drive_folder_id}")
        print(f"   - Proje Adı: {drive_manager.project_name}")
    else:
        print("ℹ️ Mevcut konfigürasyon bulunamadı veya yüklenemedi.")
        
        # 4. Yeni klasör oluştur
        print("\n4. Yeni klasör oluşturuluyor...")
        if drive_manager.setup_drive_folder():
            print("✅ Klasör başarıyla oluşturuldu!")
        else:
            print("❌ Klasör oluşturulamadı!")
            return
    
    # 5. Klasör içeriğini listele
    print("\n5. Klasör içeriği listeleniyor...")
    items = drive_manager.list_files()
    if items:
        print("\n📂 Klasör İçeriği:")
        for item in items:
            print(f"   - {item['name']} ({item['id']})")
    else:
        print("ℹ️ Klasör boş veya içerik listelenemedi.")
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    main()
