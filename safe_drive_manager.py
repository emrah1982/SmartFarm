"""
Güvenli Drive Yöneticisi

Bu modül, DriveManager sınıfı için hata yönetimli bir arayüz sağlar.
Hataları yakalar, yerel yedekleme yapar ve kullanıcı dostu mesajlar sunar.
"""

import os
import time
import shutil
from datetime import datetime
from typing import Optional, Tuple, Any, Dict, List
from pathlib import Path

# DriveManager'ı içe aktar
from drive_manager import DriveManager, setup_drive_integration

# Global DriveManager örneği
_drive_manager = None

# Yerel yedekleme klasörü
LOCAL_BACKUP_DIR = Path("local_backup")

# Hata mesajları
error_messages = {
    'not_initialized': "❌ Drive yöneticisi başlatılmamış! Lütfen önce 'initialize_drive()' fonksiyonunu çağırın.",
    'file_not_found': "❌ Dosya bulunamadı: {}",
    'drive_error': "❌ Drive hatası: {}",
    'success': "✅ {} başarıyla yüklendi: {}",
    'local_backup': "📦 Yerel yedekleme yapıldı: {}"
}

def initialize_drive() -> bool:
    """Drive yöneticisini başlatır.
    
    Returns:
        bool: Başarılıysa True, değilse False
    """
    global _drive_manager
    
    try:
        _drive_manager = setup_drive_integration()
        if _drive_manager:
            print("✅ Drive yöneticisi başarıyla başlatıldı!")
            return True
        else:
            print("❌ Drive yöneticisi başlatılamadı!")
            return False
    except Exception as e:
        print(f"❌ Drive başlatma hatası: {e}")
        return False

def get_safe_drive_manager() -> Optional[DriveManager]:
    """Mevcut Drive yöneticisini döndürür.
    
    Returns:
        Optional[DriveManager]: Drive yöneticisi veya None
    """
    return _drive_manager

def safe_upload_model(local_path: str, drive_filename: str) -> bool:
    """Modeli Drive'a güvenli bir şekilde yükler.
    
    Args:
        local_path: Yüklenecek dosyanın yolu
        drive_filename: Drive'da görünecek dosya adı
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    if _drive_manager is None:
        print(error_messages['not_initialized'])
        return _local_backup(local_path, drive_filename)
    
    if not os.path.exists(local_path):
        error_msg = error_messages['file_not_found'].format(local_path)
        print(error_msg)
        return _local_backup(local_path, drive_filename)
    
    try:
        # Yerel yedekleme yap
        local_success = _local_backup(local_path, drive_filename)
        
        # Drive'a yükle
        success = _drive_manager.upload_model(local_path, drive_filename)
        
        if success:
            print(error_messages['success'].format(drive_filename, local_path))
        else:
            print(f"⚠️ {drive_filename} Drive'a yüklenemedi, sadece yerel yedek kullanılıyor.")
        
        return success or local_success
        
    except Exception as e:
        error_msg = error_messages['drive_error'].format(str(e))
        print(error_msg)
        return _local_backup(local_path, drive_filename)

def safe_find_checkpoint() -> Tuple[Optional[str], Optional[str]]:
    """En son checkpoint'i güvenli bir şekilde bulur.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (checkpoint yolu, dosya adı) veya (None, None)
    """
    if _drive_manager is None:
        print(error_messages['not_initialized'])
        return _find_local_checkpoint()
    
    try:
        # Önce Drive'da ara
        checkpoint_path, filename = _drive_manager.find_latest_checkpoint()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            return checkpoint_path, filename
        else:
            # Drive'da bulunamazsa yerel yedekleri kontrol et
            print("⚠️ Drive'da checkpoint bulunamadı, yerel yedekler kontrol ediliyor...")
            return _find_local_checkpoint()
            
    except Exception as e:
        print(f"❌ Checkpoint arama hatası: {e}")
        return _find_local_checkpoint()

def _local_backup(local_path: str, filename: str) -> bool:
    """Dosyayı yerel yedekleme klasörüne kopyalar.
    
    Args:
        local_path: Kaynak dosya yolu
        filename: Hedef dosya adı
        
    Returns:
        bool: İşlem başarılıysa True, değilse False
    """
    try:
        if not os.path.exists(local_path):
            return False
            
        # Yedekleme klasörünü oluştur
        os.makedirs(LOCAL_BACKUP_DIR, exist_ok=True)
        
        # Dosyayı kopyala
        target_path = LOCAL_BACKUP_DIR / filename
        shutil.copy2(local_path, target_path)
        
        print(error_messages['local_backup'].format(target_path))
        return True
        
    except Exception as e:
        print(f"❌ Yerel yedekleme hatası: {e}")
        return False

def _find_local_checkpoint() -> Tuple[Optional[str], Optional[str]]:
    """Yerel yedekleme klasöründeki en son checkpoint'i bulur.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: (checkpoint yolu, dosya adı) veya (None, None)
    """
    try:
        if not os.path.exists(LOCAL_BACKUP_DIR):
            return None, None
            
        # En son değiştirilen .pt dosyasını bul
        pt_files = list(LOCAL_BACKUP_DIR.glob("*.pt"))
        if not pt_files:
            return None, None
            
        # En son değiştirilen dosyayı bul
        latest_file = max(pt_files, key=os.path.getmtime)
        return str(latest_file), latest_file.name
        
    except Exception as e:
        print(f"❌ Yerel checkpoint arama hatası: {e}")
        return None, None

def print_status():
    """Mevcut Drive durumunu yazdırır."""
    if _drive_manager is None:
        print("🔴 Drive durumu: Başlatılmadı")
        return
        
    try:
        print("\n📊 Drive Durum Raporu")
        print("=" * 40)
        
        # Drive bağlantı durumu
        if _drive_manager.is_colab:
            print(f"🔵 Mod: Google Colab")
            print(f"📂 Proje Klasörü: {getattr(_drive_manager, 'project_folder', 'Belirlenemedi')}")
        else:
            print(f"🔵 Mod: Standart API")
            print(f"📂 Klasör ID: {getattr(_drive_manager, 'drive_folder_id', 'Belirlenemedi')}")
        
        # Yerel yedekleme durumu
        local_backup_count = len(list(LOCAL_BACKUP_DIR.glob("*.pt"))) if os.path.exists(LOCAL_BACKUP_DIR) else 0
        print(f"💾 Yerel Yedekler: {local_backup_count} adet")
        
        # Drive'daki modeller
        if _drive_manager.is_colab:
            models_dir = os.path.join(getattr(_drive_manager, 'project_folder', ''), 'models')
            drive_models = list(Path(models_dir).glob("*.pt")) if os.path.exists(models_dir) else []
        else:
            drive_models = _drive_manager.list_drive_models() if hasattr(_drive_manager, 'list_drive_models') else []
        
        print(f"🚀 Drive'daki Modeller: {len(drive_models)} adet")
        
        # Son 3 modeli göster
        for i, model in enumerate(drive_models[-3:], 1):
            if _drive_manager.is_colab:
                print(f"   {i}. {model.name} - {time.ctime(model.stat().st_mtime)}")
            else:
                print(f"   {i}. {model.get('name')} - {model.get('modifiedTime')}")
        
        print("=" * 40)
        
    except Exception as e:
        print(f"❌ Durum raporu alınamadı: {e}")

# Modül yüklendiğinde yerel yedekleme klasörünü oluştur
os.makedirs(LOCAL_BACKUP_DIR, exist_ok=True)
