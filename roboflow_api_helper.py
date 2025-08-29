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

def download_fieldplant_v11_yolov11(api_key: str = "vzhHWH8uT44rO0V25x5n",
                                     workspace: str = "plant-disease-detection",
                                     project: str = "fieldplant",
                                     version_number: int = 11,
                                     format_name: str = "yolov11"):
    """Roboflow SDK kullanarak dataset indirir.

    Varsayılan değerler, talep ettiğiniz örnekle birebir uyumludur:
    - api_key: "vzhHWH8uT44rO0V25x5n"
    - workspace: "plant-disease-detection"
    - project: "fieldplant"
    - version: 11
    - format: "yolov11"

    Not: Jupyter dışındaki ortamlarda "!pip install roboflow" yerine
    eksikse paket otomatik kurulmaya çalışılır.
    """
    import sys
    import subprocess

    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        print("📦 Roboflow paketi bulunamadı. Kurulum deneniyor: pip install roboflow")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])  # nosec
            from roboflow import Roboflow  # type: ignore
        except Exception as e:
            print(f"❌ Roboflow kurulumu başarısız: {e}")
            return False

    try:
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        ver = proj.version(version_number)
        print(f"⬇️ İndirme başlıyor: {workspace}/{project} v{version_number} ({format_name})")
        dataset = ver.download(format_name)
        # SDK genellikle indirilen klasörü stdout'a yazar. Burada sadece başarı bilgisini dönüyoruz.
        print("✅ İndirme tamamlandı.")
        return True
    except Exception as e:
        print(f"❌ İndirme sırasında hata: {e}")
        return False

def _ensure_package(import_name: str, install_name: str = None):
    """Gerekli paketi import etmeyi dener, yoksa pip ile kurmaya çalışır.

    import_name: Python'da import edilen modül adı (örn. 'yaml')
    install_name: pip paket adı (örn. 'pyyaml'). Boş ise import_name kullanılır.
    """
    import importlib
    import sys
    import subprocess

    # Bilinen farklı isim eşleştirmeleri
    if install_name is None:
        mapping = {
            'yaml': 'pyyaml',
        }
        install_name = mapping.get(import_name, import_name)

    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"📦 '{import_name}' modülü bulunamadı. Kurulum deneniyor: pip install {install_name}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])  # nosec
            return importlib.import_module(import_name)
        except Exception as e:
            print(f"❌ {install_name} kurulumu başarısız: {e}")
            return None

def _parse_roboflow_canonical(canonical: str):
    """'workspace/project/version' formatını (ws, proj, ver) olarak döndürür."""
    try:
        parts = [p for p in canonical.strip().split('/') if p]
        if len(parts) != 3:
            raise ValueError(f"Beklenen format 'workspace/project/version', gelen: {canonical}")
        ws, proj, ver_str = parts
        ver = int(ver_str)
        return ws, proj, ver
    except Exception as e:
        print(f"❌ roboflow_canonical parse hatası: {e}")
        return None

def download_from_config_entry(entry: dict,
                               dataset_dir: str = 'datasets/roboflow_dataset',
                               api_key: str = None,
                               format_name: str = 'yolov11'):
    """config_datasets.yaml içindeki tek dataset kaydını SADECE Roboflow SDK ile indirir.

    - Yalnızca `roboflow_canonical` ("workspace/project/version") alanını kabul eder.
    - API key zorunludur (parametre veya `config/roboflow.json`).
    - İndirme dizini `dataset_dir` olur; SDK çağrısı sırasında çalışma dizini buraya alınır.
    """
    # Debug: hangi alanlar var?
    try:
        print(f"🧾 YAML entry anahtarları: {sorted(list(entry.keys()))}")
    except Exception:
        pass

    # roboflow_canonical + API (SDK-only)
    canonical = entry.get('roboflow_canonical')
    if canonical:
        print("🧭 roboflow_canonical bulundu. SDK ile indirme denenecek...")
        parsed = _parse_roboflow_canonical(canonical)
        if not parsed:
            return False
        ws, proj, ver = parsed
        # API key temini: parametre > config
        use_key = api_key or get_api_key_from_config()
        if not use_key:
            print("🔒 API key bulunamadı. SDK ile indirme atlandı.")
        else:
            # Roboflow SDK'yı garanti altına al
            rf_mod = _ensure_package('roboflow')
            if rf_mod is None:
                return False
            try:
                from roboflow import Roboflow  # type: ignore
                rf = Roboflow(api_key=use_key)
                ver_obj = rf.workspace(ws).project(proj).version(ver)
                print(f"⬇️ İndirme başlıyor: {ws}/{proj} v{ver} ({format_name})")
                import os
                os.makedirs(dataset_dir, exist_ok=True)
                cwd_backup = os.getcwd()
                try:
                    os.chdir(dataset_dir)
                    print(f"📂 Çalışma dizini: {os.getcwd()}")
                    # Roboflow SDK indirmeyi mevcut çalışma dizinine yapar
                    out = ver_obj.download(format_name)
                    # İndirme sonrası kalan zip'leri otomatik çıkar
                    try:
                        import pathlib, zipfile
                        # Roboflow SDK çoğu zaman bir Dataset objesi döndürür; yol için `.location` kullan
                        p = None
                        if out is None:
                            p = pathlib.Path(os.getcwd())
                        else:
                            # Dataset objesi ise .location olabilir
                            loc = getattr(out, 'location', None)
                            if isinstance(loc, (str, bytes, os.PathLike)):
                                p = pathlib.Path(loc)
                            else:
                                # out bir string veya Path olabilir
                                try:
                                    p = pathlib.Path(out)
                                except Exception:
                                    p = pathlib.Path(os.getcwd())
                        p = p.resolve()
                        print(f"📦 İndirilen yol: {p}")
                        # roboflow.zip veya diğer zipler
                        zips = list(p.rglob('*.zip'))
                        if zips:
                            for z in zips:
                                try:
                                    target_dir = z.with_suffix('')
                                    target_dir.mkdir(parents=True, exist_ok=True)
                                    print(f"🗜️  Zip çıkarılıyor: {z} -> {target_dir}")
                                    with zipfile.ZipFile(z, 'r') as zf:
                                        zf.extractall(target_dir)
                                except Exception as ez:
                                    print(f"⚠️ Zip çıkarma hatası ({z}): {ez}")
                        # İçerik önizleme
                        if p.exists():
                            entries = list(p.glob('*'))[:10]
                            print("🗂 İçerik örnekleri:", [e.name for e in entries])
                    except Exception as ep:
                        print(f"⚠️ İndirme sonrası kontrol/çıkarma sırasında hata: {ep}")
                finally:
                    os.chdir(cwd_backup)
                print("✅ İndirme tamamlandı (SDK)")
                return True
            except Exception as e:
                print(f"❌ SDK ile indirme hatası: {e}")
                return False
    print("❌ 'roboflow_canonical' alanı zorunludur ve bulunamadı. SDK-only modda indirme yapılamaz.")
    return False

def _find_dataset_entry_in_yaml(dataset_name: str, yaml_path: str = 'config_datasets.yaml'):
    """YAML içinde verilen dataset adını arar ve kaydını döndürür.

    Aranan path: root['datasets'][<group_name>][dataset_name]
    group_name örnekleri: base_datasets, pest_datasets, specialized_datasets, experimental_datasets
    """
    yaml_mod = _ensure_package('yaml')
    if yaml_mod is None:
        return None
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml_mod.safe_load(f)
    except Exception as e:
        print(f"❌ YAML okuma hatası: {e}")
        return None

    datasets_root = (cfg or {}).get('datasets') or {}
    if not isinstance(datasets_root, dict):
        return None

    for group_name, group in datasets_root.items():
        if not isinstance(group, dict):
            continue
        if dataset_name in group:
            print(f"🔍 Bulunan grup: {group_name} -> dataset: {dataset_name}")
            entry = group.get(dataset_name)
            if isinstance(entry, dict):
                try:
                    print(f"🧾 Entry anahtarları: {sorted(list(entry.keys()))}")
                except Exception:
                    pass
                return entry
    print(f"❌ YAML içinde '{dataset_name}' bulunamadı.")
    return None

def download_from_config_yaml(dataset_name: str,
                              yaml_path: str = 'config_datasets.yaml',
                              dataset_dir: str = 'datasets/roboflow_dataset',
                              api_key: str = None,
                              format_name: str = 'yolov11'):
    """config_datasets.yaml dosyasında adı verilen dataset'i indirir (SDK-only).

    - Yalnızca `roboflow_canonical` + API key ile indirme desteklenir.
    """
    entry = _find_dataset_entry_in_yaml(dataset_name, yaml_path=yaml_path)
    if not entry:
        return False
    return download_from_config_entry(entry, dataset_dir=dataset_dir, api_key=api_key, format_name=format_name)

if __name__ == "__main__":
    print("🤖 Roboflow API Helper")
    print("Bu modül Roboflow dataset indirme sorunlarını çözer")
    print("Ana kullanım main_multi_dataset.py üzerinden yapılmalıdır.")
