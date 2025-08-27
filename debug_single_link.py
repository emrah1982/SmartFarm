#!/usr/bin/env python3
# debug_single_link.py - Tek bir Roboflow linkini detaylı debug et

import requests
from dataset_utils import download_dataset

def debug_single_roboflow_link():
    """Tek bir Roboflow linkini detaylı debug et"""
    
    # Test URL - config_datasets.yaml'den
    test_url = "https://universe.roboflow.com/ds/nKPr1UgofJ?key=a2sSLftQC8"
    
    print("🔍 ROBOFLOW LINK DEBUG")
    print("=" * 50)
    print(f"Test URL: {test_url}")
    print("=" * 50)
    
    # 1. Manuel HTTP isteği ile test
    print("\n1️⃣ Manuel HTTP İsteği:")
    try:
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        session.headers.update(headers)
        
        response = session.get(test_url, timeout=30, allow_redirects=True)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('content-type', 'Unknown')}")
        print(f"   Content-Length: {response.headers.get('content-length', 'Unknown')}")
        print(f"   Final URL: {response.url}")
        
        # İlk 1000 karakter
        if response.text:
            preview = response.text[:1000].replace('\n', ' ').replace('\r', '')
            print(f"   Body preview: {preview}...")
            
            # ZIP link arama
            import re
            zip_links = re.findall(r'href=["\']([^"\']*\.zip[^"\']*)["\']', response.text, re.IGNORECASE)
            if zip_links:
                print(f"   ZIP links bulundu: {zip_links[:3]}")
            else:
                print(f"   ZIP link bulunamadı")
                
    except Exception as e:
        print(f"   Hata: {e}")
    
    # 2. dataset_utils.py ile test
    print(f"\n2️⃣ dataset_utils.py ile İndirme:")
    try:
        success = download_dataset(test_url, "debug_test", api_key=None, split_config=None)
        print(f"   Sonuç: {'Başarılı' if success else 'Başarısız'}")
    except Exception as e:
        print(f"   Hata: {e}")
    
    # 3. API key ile test
    print(f"\n3️⃣ API Key Prompt:")
    api_key = input("Roboflow API key girin (boş geçebilirsiniz): ").strip()
    if api_key:
        print(f"   API key ile test ediliyor...")
        try:
            success = download_dataset(test_url, "debug_test_api", api_key=api_key, split_config=None)
            print(f"   Sonuç: {'Başarılı' if success else 'Başarısız'}")
        except Exception as e:
            print(f"   Hata: {e}")
    else:
        print(f"   API key atlandı")
    
    print(f"\n🏁 Debug tamamlandı")

if __name__ == "__main__":
    debug_single_roboflow_link()
