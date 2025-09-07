"""
Config Utilities - Batch size ve diğer ayarları config_datasets.yaml'dan okuma
"""
import yaml
import os
from typing import Optional, Dict, Any

def load_config_datasets(config_path: str = "config_datasets.yaml") -> Optional[Dict[str, Any]]:
    """
    config_datasets.yaml dosyasını yükle
    
    Args:
        config_path: Config dosyasının yolu
        
    Returns:
        Dict: Config verisi veya None
    """
    try:
        # Mutlak yol kontrolü
        if not os.path.isabs(config_path):
            # Mevcut dosyanın bulunduğu dizini bul
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, config_path)
        
        if not os.path.exists(config_path):
            print(f"⚠️ Config dosyası bulunamadı: {config_path}")
            return None
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        print(f"⚠️ Config dosyası okuma hatası: {e}")
        return None

def get_default_batch_size(config_path: str = "config_datasets.yaml") -> int:
    """
    Config dosyasından default batch size değerini al
    
    Args:
        config_path: Config dosyasının yolu
        
    Returns:
        int: Batch size değeri (default: 8)
    """
    config = load_config_datasets(config_path)
    
    if config and 'global_settings' in config:
        batch_size = config['global_settings'].get('default_batch_size', 8)
        print(f"📊 Config'den batch size okundu: {batch_size}")
        return batch_size
    
    print("📊 Config okunamadı, default batch size kullanılıyor: 8")
    return 8

def get_dataset_group_batch_size(group_name: str, config_path: str = "config_datasets.yaml") -> int:
    """
    Belirli bir dataset grubu için batch size al
    
    Args:
        group_name: Dataset grup adı
        config_path: Config dosyasının yolu
        
    Returns:
        int: Batch size değeri
    """
    config = load_config_datasets(config_path)
    
    if config and 'dataset_groups' in config:
        groups = config['dataset_groups']
        if group_name in groups:
            group_batch = groups[group_name].get('batch_size')
            if group_batch:
                print(f"📊 {group_name} grubu için batch size: {group_batch}")
                return group_batch
    
    # Fallback to default
    return get_default_batch_size(config_path)

def get_default_image_size(config_path: str = "config_datasets.yaml") -> int:
    """
    Config dosyasından default image size değerini al
    
    Args:
        config_path: Config dosyasının yolu
        
    Returns:
        int: Image size değeri (default: 640)
    """
    config = load_config_datasets(config_path)
    
    if config and 'global_settings' in config:
        img_size = config['global_settings'].get('default_image_size', 640)
        print(f"🖼️ Config'den image size okundu: {img_size}")
        return img_size
    
    print("🖼️ Config okunamadı, default image size kullanılıyor: 640")
    return 640

def get_training_config_from_yaml(dataset_group: str = None, config_path: str = "config_datasets.yaml") -> Dict[str, Any]:
    """
    Config dosyasından eğitim ayarlarını al
    
    Args:
        dataset_group: Dataset grup adı (opsiyonel)
        config_path: Config dosyasının yolu
        
    Returns:
        Dict: Eğitim konfigürasyonu
    """
    config = load_config_datasets(config_path)
    
    # Default değerler
    training_config = {
        'batch_size': 8,
        'image_size': 640,
        'model': 'yolo11l.pt',
        'estimated_time': 'Unknown'
    }
    
    if not config:
        return training_config
    
    # Global settings'den default değerleri al
    if 'global_settings' in config:
        global_settings = config['global_settings']
        training_config['batch_size'] = global_settings.get('default_batch_size', 8)
        training_config['image_size'] = global_settings.get('default_image_size', 640)
    
    # Dataset grup ayarları varsa onları kullan
    if dataset_group and 'dataset_groups' in config:
        groups = config['dataset_groups']
        if dataset_group in groups:
            group_config = groups[dataset_group]
            training_config['batch_size'] = group_config.get('batch_size', training_config['batch_size'])
            training_config['image_size'] = group_config.get('image_size', training_config['image_size'])
            training_config['model'] = group_config.get('recommended_model', training_config['model'])
            training_config['estimated_time'] = group_config.get('estimated_training_time', training_config['estimated_time'])
    
    print(f"⚙️ Eğitim config hazırlandı: batch={training_config['batch_size']}, img_size={training_config['image_size']}")
    return training_config

def update_config_batch_size(new_batch_size: int, config_path: str = "config_datasets.yaml") -> bool:
    """
    Config dosyasındaki batch size değerini güncelle
    
    Args:
        new_batch_size: Yeni batch size değeri
        config_path: Config dosyasının yolu
        
    Returns:
        bool: Başarılı olup olmadığı
    """
    try:
        config = load_config_datasets(config_path)
        if not config:
            return False
        
        # Global settings'i güncelle
        if 'global_settings' not in config:
            config['global_settings'] = {}
        
        config['global_settings']['default_batch_size'] = new_batch_size
        
        # Tüm dataset gruplarını güncelle
        if 'dataset_groups' in config:
            for group_name, group_config in config['dataset_groups'].items():
                group_config['batch_size'] = new_batch_size
        
        # Dosyayı kaydet
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"✅ Config dosyasında batch size {new_batch_size} olarak güncellendi")
        return True
        
    except Exception as e:
        print(f"❌ Config güncelleme hatası: {e}")
        return False
