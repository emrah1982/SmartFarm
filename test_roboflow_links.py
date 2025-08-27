#!/usr/bin/env python3
# test_roboflow_links.py - Roboflow linklerini test etmek için

import os
import sys
from dataset_utils import download_dataset, build_roboflow_download_url

def test_roboflow_links():
    """Roboflow linklerini test et"""
    print("🧪 Roboflow Link Test Başlatılıyor...")
    
    # Test edilecek linkler (config_datasets.yaml'den)
    test_links = [
        "https://universe.roboflow.com/ds/nKPr1UgofJ?key=a2sSLftQC8",
        "https://universe.roboflow.com/ds/0UULi7Pnno?key=PU2zi8AslM",
        "https://universe.roboflow.com/ds/KXxiCfvas4?key=LQed1EPrBo"
    ]
    
    # Test klasörü oluştur
    test_dir = "test_downloads"
    os.makedirs(test_dir, exist_ok=True)
    
    for i, url in enumerate(test_links, 1):
        print(f"\n{'='*60}")
        print(f"🔗 Test {i}/{len(test_links)}: {url}")
        print(f"{'='*60}")
        
        # URL yapısını analiz et
        print(f"📋 URL Analizi:")
        built_url = build_roboflow_download_url(url, None, None)
        print(f"   Orijinal: {url}")
        print(f"   İşlenmiş: {built_url}")
        
        # Test indirme klasörü
        dataset_name = f"test_dataset_{i}"
        dataset_path = os.path.join(test_dir, dataset_name)
        
        print(f"\n📥 İndirme Testi:")
        print(f"   Hedef klasör: {dataset_path}")
        
        try:
            # Sadece ilk birkaç saniye test et (timeout kısa)
            success = download_dataset(url, dataset_path, api_key=None, split_config=None)
            
            if success:
                print(f"✅ İndirme başarılı!")
                # Klasör içeriğini kontrol et
                if os.path.exists(dataset_path):
                    files = os.listdir(dataset_path)
                    print(f"   İndirilen dosyalar: {len(files)} adet")
                    for f in files[:5]:  # İlk 5 dosyayı göster
                        print(f"     - {f}")
                    if len(files) > 5:
                        print(f"     ... ve {len(files)-5} dosya daha")
            else:
                print(f"❌ İndirme başarısız")
                
        except Exception as e:
            print(f"❌ Hata: {e}")
        
        print(f"\n⏸️  Test {i} tamamlandı")
    
    print(f"\n{'='*60}")
    print(f"🏁 Tüm testler tamamlandı")
    print(f"📁 Test dosyaları: {os.path.abspath(test_dir)}")
    print(f"{'='*60}")

def test_url_building():
    """URL oluşturma fonksiyonunu test et"""
    print("\n🔧 URL Oluşturma Testi:")
    
    test_cases = [
        {
            "url": "https://universe.roboflow.com/ds/nKPr1UgofJ?key=a2sSLftQC8",
            "api_key": None,
            "split_config": None
        },
        {
            "url": "https://universe.roboflow.com/ds/nKPr1UgofJ?key=a2sSLftQC8",
            "api_key": "test_api_key_123",
            "split_config": {"train": 70, "test": 20, "val": 10}
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n  Test Case {i}:")
        print(f"    Giriş URL: {case['url']}")
        print(f"    API Key: {case['api_key']}")
        print(f"    Split Config: {case['split_config']}")
        
        result = build_roboflow_download_url(
            case['url'], 
            case['api_key'], 
            case['split_config']
        )
        print(f"    Sonuç URL: {result}")

if __name__ == "__main__":
    print("🚀 Roboflow Link Test Aracı")
    print("=" * 50)
    
    # URL oluşturma testini çalıştır
    test_url_building()
    
    # Kullanıcıya sor
    choice = input("\n📥 Gerçek indirme testi yapmak istiyor musunuz? (e/h): ").lower().strip()
    
    if choice in ['e', 'evet', 'yes', 'y']:
        test_roboflow_links()
    else:
        print("⏭️  İndirme testi atlandı")
    
    print("\n✅ Test tamamlandı!")
