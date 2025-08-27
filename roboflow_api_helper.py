#!/usr/bin/env python3
# roboflow_api_helper.py - Roboflow API key yönetimi ve dataset indirme yardımcısı

import os
import json
from pathlib import Path

def get_api_key_from_config():
    """Yapılandırma dosyasından API key'i oku"""
    config_paths = [
        'roboflow_config.json',
        os.path.expanduser('~/.roboflow/config.json'),
        'config/roboflow.json'
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('api_key')
            except:
                continue
    return None

def save_api_key(api_key):
    """API key'i yapılandırma dosyasına kaydet"""
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / 'roboflow.json'
    config = {'api_key': api_key}
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ API key kaydedildi: {config_path}")

def setup_roboflow_api():
    """Roboflow API key kurulumu"""
    print("\n🔑 Roboflow API Key Kurulumu")
    print("=" * 40)
    
    # Mevcut API key kontrol et
    existing_key = get_api_key_from_config()
    if existing_key:
        print(f"✅ Mevcut API key bulundu: {existing_key[:10]}...")
        use_existing = input("Mevcut API key'i kullanmak istiyor musunuz? (e/h): ").lower()
        if use_existing.startswith('e'):
            return existing_key
    
    print("\n📋 API Key alma adımları:")
    print("1. https://roboflow.com adresine gidin")
    print("2. Hesabınıza giriş yapın")
    print("3. Settings > API sayfasına gidin")
    print("4. Private API Key'inizi kopyalayın")
    
    api_key = input("\n🔑 API Key'inizi girin: ").strip()
    
    if api_key:
        save_api_key(api_key)
        return api_key
    else:
        print("❌ API key girilmedi")
        return None

def download_with_api_key(url, dataset_dir='datasets/roboflow_dataset'):
    """API key ile dataset indir"""
    from dataset_utils import download_dataset
    
    # API key al
    api_key = get_api_key_from_config()
    if not api_key:
        print("⚠️ API key bulunamadı. Kurulum başlatılıyor...")
        api_key = setup_roboflow_api()
    
    if api_key:
        print(f"🔑 API key ile indirme başlatılıyor...")
        return download_dataset(url, dataset_dir, api_key=api_key)
    else:
        print("❌ API key olmadan indirme deneniyor...")
        return download_dataset(url, dataset_dir)

def check_dataset_access(url):
    """Dataset erişimini kontrol et"""
    import requests
    
    try:
        # Sadece HEAD request ile kontrol et
        response = requests.head(url, timeout=10)
        
        if response.status_code == 200:
            print("✅ Dataset erişilebilir (Public)")
            return True
        elif response.status_code == 403:
            print("🔒 Dataset private - API key gerekli")
            return False
        elif response.status_code == 404:
            print("❌ Dataset bulunamadı")
            return False
        else:
            print(f"⚠️ Bilinmeyen durum: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erişim kontrolü başarısız: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Roboflow API Helper")
    print("Bu modül Roboflow dataset indirme sorunlarını çözer")
    
    # Test URL'si
    test_url = input("Test etmek istediğiniz Roboflow URL'sini girin: ").strip()
    if test_url:
        print(f"\n🔍 Dataset erişimi kontrol ediliyor...")
        if check_dataset_access(test_url):
            print("✅ Dataset public, API key gerekmeyebilir")
        else:
            print("🔑 API key kurulumu gerekli")
            setup_roboflow_api()
