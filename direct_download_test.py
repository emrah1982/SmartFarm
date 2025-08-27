#!/usr/bin/env python3
# direct_download_test.py - Roboflow direkt indirme testi

import requests
import os
from urllib.parse import urlparse

def test_direct_download(url):
    """Browser'da çalışan URL'yi direkt indirme testi"""
    print(f"🧪 Direkt indirme testi başlatılıyor...")
    print(f"📎 URL: {url}")
    
    # Browser benzeri session oluştur
    session = requests.Session()
    
    # Tam browser headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Cache-Control': 'max-age=0'
    }
    
    session.headers.update(headers)
    
    try:
        # İlk olarak HEAD request ile kontrol et
        print("🔍 URL erişilebilirlik kontrolü...")
        head_response = session.head(url, allow_redirects=True, timeout=30)
        print(f"📊 Status Code: {head_response.status_code}")
        print(f"📄 Content-Type: {head_response.headers.get('content-type', 'N/A')}")
        print(f"📏 Content-Length: {head_response.headers.get('content-length', 'N/A')}")
        
        if head_response.status_code == 200:
            print("✅ URL erişilebilir!")
            
            # Gerçek indirme testi (sadece ilk 1MB)
            print("📥 İndirme testi (ilk 1MB)...")
            response = session.get(url, stream=True, timeout=60)
            
            if response.status_code == 200:
                # İlk 1MB'ı indir
                downloaded = 0
                max_download = 1024 * 1024  # 1MB
                
                test_file = "test_download.zip"
                with open(test_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded >= max_download:
                                break
                
                print(f"✅ Test indirme başarılı: {downloaded} bytes")
                
                # Test dosyasını sil
                if os.path.exists(test_file):
                    os.remove(test_file)
                
                return True
            else:
                print(f"❌ İndirme başarısız: {response.status_code}")
                return False
                
        elif head_response.status_code == 403:
            print("❌ 403 Forbidden - Erişim reddedildi")
            print("🔍 Redirect zincirini kontrol ediliyor...")
            
            # Redirect zincirini takip et
            response = session.get(url, allow_redirects=False, timeout=30)
            if response.status_code in [301, 302, 303, 307, 308]:
                redirect_url = response.headers.get('location')
                print(f"🔄 Redirect tespit edildi: {redirect_url}")
                if redirect_url:
                    return test_direct_download(redirect_url)
            
            return False
        else:
            print(f"❌ Erişim başarısız: {head_response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network hatası: {e}")
        return False
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        return False

def extract_download_link(roboflow_url):
    """Roboflow URL'sinden direkt indirme linkini çıkar"""
    print("🔗 Direkt indirme linki çıkarılıyor...")
    
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    session.headers.update(headers)
    
    try:
        # Sayfayı al ve indirme linkini bul
        response = session.get(roboflow_url, timeout=30)
        if response.status_code == 200:
            # HTML içinde ZIP link arama
            content = response.text
            
            # Yaygın ZIP link formatları
            import re
            zip_patterns = [
                r'href="([^"]*\.zip[^"]*)"',
                r'href="([^"]*format=yolov5[^"]*)"',
                r'href="([^"]*download[^"]*)"'
            ]
            
            for pattern in zip_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    print(f"✅ {len(matches)} indirme linki bulundu")
                    for i, match in enumerate(matches[:3]):  # İlk 3'ünü göster
                        print(f"   {i+1}. {match[:100]}...")
                    return matches[0]  # İlkini döndür
            
            print("❌ HTML içinde indirme linki bulunamadı")
            return None
        else:
            print(f"❌ Sayfa erişimi başarısız: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Link çıkarma hatası: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Roboflow Direkt İndirme Test Aracı")
    print("=" * 50)
    
    url = input("🔗 Test edilecek Roboflow URL'sini girin: ").strip()
    
    if url:
        print(f"\n1️⃣ Direkt URL testi...")
        if test_direct_download(url):
            print("✅ Direkt indirme çalışıyor!")
        else:
            print("\n2️⃣ HTML'den indirme linki çıkarma...")
            download_link = extract_download_link(url)
            if download_link:
                print(f"🔗 Bulunan link: {download_link}")
                if test_direct_download(download_link):
                    print("✅ Çıkarılan link ile indirme çalışıyor!")
                else:
                    print("❌ Çıkarılan link de çalışmıyor")
            else:
                print("❌ İndirme linki bulunamadı")
    else:
        print("❌ URL girilmedi")
