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

def setup_roboflow_api(api_key=None, use_existing=None):
    """Roboflow API key kurulumu - main_multi_dataset.py'dan yönetilebilir"""
    print("\n🔑 Roboflow API Key Kurulumu")
    print("=" * 40)
    
    # Mevcut API key kontrol et
    existing_key = get_api_key_from_config()
    if existing_key and use_existing is None:
        print(f"✅ Mevcut API key bulundu: {existing_key[:10]}...")
        return existing_key  # Otomatik olarak mevcut key'i kullan
    elif existing_key and use_existing:
        return existing_key
    
    # API key parametre olarak verilmişse
    if api_key:
        save_api_key(api_key)
        return api_key
    
    print("\n📋 API Key alma adımları:")
    print("1. https://roboflow.com adresine gidin")
    print("2. Hesabınıza giriş yapın")
    print("3. Settings > API sayfasına gidin")
    print("4. Private API Key'inizi kopyalayın")
    
    return None  # Input gerekli ama parametre verilmemiş
    
    # Bu durumda main_multi_dataset.py'dan input alınması gerekiyor
    print("⚠️ API key main_multi_dataset.py'dan sağlanmalı")
    return None

def download_with_api_key(url, dataset_dir='datasets/roboflow_dataset', api_key=None):
    """API key ile dataset indir - main_multi_dataset.py'dan yönetilebilir"""
    from dataset_utils import download_dataset
    
    # Parametre olarak API key verilmişse kullan
    if api_key:
        print(f"🔑 Verilen API key ile indirme başlatılıyor...")
        return download_dataset(url, dataset_dir, api_key=api_key)
    
    # Kayıtlı API key kontrol et
    saved_api_key = get_api_key_from_config()
    if saved_api_key:
        print(f"🔑 Kayıtlı API key ile indirme başlatılıyor...")
        return download_dataset(url, dataset_dir, api_key=saved_api_key)
    
    # API key yok, public dataset olarak dene
    print("⚠️ API key bulunamadı. Public dataset olarak deneniyor...")
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

def get_roboflow_menu_choice():
    """Roboflow menü seçeneklerini döndür (main_multi_dataset.py için)"""
    menu_options = {
        '1': 'API Key Kurulumu',
        '2': 'Mevcut API Key Görüntüle',
        '3': 'API Key Sil',
        '4': 'Dataset Erişim Testi',
        '5': 'Dataset İndir (API Key ile)',
        '6': 'Dataset İndir (Public)'
    }
    return menu_options

def handle_roboflow_action(choice, **kwargs):
    """Roboflow aksiyonlarını handle et (main_multi_dataset.py için)"""
    if choice == '1':  # API Key Kurulumu
        api_key = kwargs.get('api_key')
        return setup_roboflow_api(api_key=api_key)
    
    elif choice == '2':  # Mevcut API Key Görüntüle
        existing_key = get_api_key_from_config()
        if existing_key:
            print(f"✅ Mevcut API Key: {existing_key[:10]}...{existing_key[-4:]}")
            return existing_key
        else:
            print("❌ Kayıtlı API key bulunamadı")
            return None
    
    elif choice == '3':  # API Key Sil
        config_paths = ['config/roboflow.json']
        for path in config_paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"✅ API key silindi: {path}")
                return True
        print("❌ Silinecek API key bulunamadı")
        return False
    
    elif choice == '4':  # Dataset Erişim Testi
        url = kwargs.get('url')
        if url:
            return check_dataset_access(url)
        return False
    
    elif choice == '5':  # Dataset İndir (API Key ile)
        url = kwargs.get('url')
        dataset_dir = kwargs.get('dataset_dir', 'datasets/roboflow_dataset')
        api_key = kwargs.get('api_key')
        if url:
            return download_with_api_key(url, dataset_dir, api_key=api_key)
        return False
    
    elif choice == '6':  # Dataset İndir (Public)
        url = kwargs.get('url')
        dataset_dir = kwargs.get('dataset_dir', 'datasets/roboflow_dataset')
        if url:
            from dataset_utils import download_dataset
            return download_dataset(url, dataset_dir)
        return False
    
    return False

if __name__ == "__main__":
    print("🤖 Roboflow API Helper")
    print("Bu modül Roboflow dataset indirme sorunlarını çözer")
    print("Ana kullanım main_multi_dataset.py üzerinden yapılmalıdır.")
