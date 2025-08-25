"""
Training Utilities for SmartFarm

This module provides utility functions for training models with safe Drive integration.
"""

import os
import torch
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Import the safe Drive manager
from safe_drive_manager import (
    initialize_drive,
    safe_upload_model,
    safe_find_checkpoint,
    get_safe_drive_manager,
    print_status as print_drive_status
)

# Error messages
error_messages = {
    'model_not_found': "❌ Model dosyası bulunamadı: {}",
    'checkpoint_not_found': "❌ Checkpoint bulunamadı, yeni eğitime başlanıyor",
    'checkpoint_loaded': "✅ Checkpoint yüklendi: {}",
    'training_started': "🚀 Eğitim başlatılıyor (Epoch {}/{})",
    'epoch_completed': "✅ Epoch {}/{} tamamlandı - Loss: {:.4f}",
    'model_saved': "💾 Model kaydedildi: {}",
    'drive_error': "❌ Drive hatası: {}",
}

def setup_training_environment(use_drive: bool = True) -> bool:
    """Eğitim ortamını hazırlar ve Drive entegrasyonunu başlatır.
    
    Args:
        use_drive: Drive entegrasyonu kullanılsın mı?
        
    Returns:
        bool: Drive başarıyla başlatıldıysa True, değilse False
    """
    print("\n🔧 Eğitim Ortamı Hazırlanıyor...")
    
    # GPU kullanılabilir mi kontrol et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Kullanılan Cihaz: {device}")
    
    # Drive entegrasyonu
    drive_initialized = False
    if use_drive:
        print("🔄 Drive entegrasyonu başlatılıyor...")
        drive_initialized = initialize_drive()
        
        if drive_initialized:
            print("✅ Drive entegrasyonu başarılı!")
        else:
            print("⚠️ Drive entegrasyonu başarısız, sadece yerel kaydetme kullanılacak")
    
    return drive_initialized

def load_checkpoint(checkpoint_path: Optional[str] = None) -> Tuple[Optional[Dict], Optional[int]]:
    """Checkpoint yükler veya en son checkpoint'i arar.
    
    Args:
        checkpoint_path: Özel bir checkpoint yolu belirtmek için
        
    Returns:
        Tuple[Optional[Dict], Optional[int]]: (checkpoint, start_epoch)
    """
    print("\n🔍 Checkpoint kontrol ediliyor...")
    
    # Eğer özel bir checkpoint yolu verilmişse onu kullan
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(error_messages['checkpoint_loaded'].format(checkpoint_path))
            return checkpoint, start_epoch
        except Exception as e:
            print(f"❌ Checkpoint yükleme hatası: {e}")
    
    # Eğer özel bir yol verilmemişse veya yükleme başarısız olmuşsa
    # Drive'dan veya yerel yedeklerden en son checkpoint'i bul
    checkpoint_path, filename = safe_find_checkpoint()
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(error_messages['checkpoint_loaded'].format(checkpoint_path))
            return checkpoint, start_epoch
        except Exception as e:
            print(f"❌ Checkpoint yükleme hatası: {e}")
    
    # Hiçbir checkpoint bulunamadı
    print(error_messages['checkpoint_not_found'])
    return None, 0

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    is_best: bool = False,
    model_dir: str = 'runs/train/weights',
    model_name: str = 'model'
) -> Tuple[bool, str]:
    """Model checkpoint'ini kaydeder.
    
    Args:
        model: Eğitilen model
        optimizer: Optimizer
        epoch: Mevcut epoch
        loss: Mevcut loss değeri
        is_best: Bu en iyi model mi?
        model_dir: Modelin kaydedileceği dizin
        model_name: Model dosya adı öneki
        
    Returns:
        Tuple[bool, str]: (başarılı mı, kaydedilen dosya yolu)
    """
    try:
        # Dizini oluştur
        os.makedirs(model_dir, exist_ok=True)
        
        # Checkpoint verisini hazırla
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': time.time()
        }
        
        # Dosya adını belirle (standart YOLO formatında)
        if is_best:
            filename = "best.pt"
        else:
            filename = "last.pt"
            
        # Dosya yolunu oluştur
        filepath = os.path.join(model_dir, filename)
        
        # Modeli kaydet
        torch.save(checkpoint, filepath)
        print(error_messages['model_saved'].format(filepath))
        
        return True, filepath
        
    except Exception as e:
        print(f"❌ Model kaydetme hatası: {e}")
        return False, ""

def upload_to_drive(
    local_path: str,
    drive_filename: Optional[str] = None,
    is_best: bool = False
) -> bool:
    """Modeli Drive'a yükler.
    
    Args:
        local_path: Yüklenecek dosyanın yolu
        drive_filename: Drive'da görünecek dosya adı (None ise local_path'in dosya adı kullanılır)
        is_best: Bu en iyi model mi?
        
    Returns:
        bool: Yükleme başarılı olduysa True, değilse False
    """
    if not os.path.exists(local_path):
        print(error_messages['model_not_found'].format(local_path))
        return False
    
    # Eğer dosya adı belirtilmemişse, local_path'ten al
    if not drive_filename:
        drive_filename = os.path.basename(local_path)
    
    # Dosya boyutunu göster
    file_size = os.path.getsize(local_path) / (1024*1024)
    print(f"☁️ Drive'a yükleniyor: {drive_filename} ({file_size:.1f} MB)...")
    
    return safe_upload_model(local_path, drive_filename)

def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> float:
    """Tek bir epoch için eğitim döngüsü.
    
    Args:
        model: Eğitilecek model
        train_loader: Eğitim veri yükleyicisi
        criterion: Kayıp fonksiyonu
        optimizer: Optimizer
        device: Kullanılacak cihaz (CPU/GPU)
        epoch: Mevcut epoch numarası
        num_epochs: Toplam epoch sayısı
        
    Returns:
        float: Ortalama kayıp değeri
    """
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    
    print(error_messages['training_started'].format(epoch + 1, num_epochs))
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Sıfırla gradyanları
        optimizer.zero_grad()
        
        # İleri geçiş
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Geri yayılım
        loss.backward()
        optimizer.step()
        
        # İstatistikleri güncelle
        running_loss += loss.item()
        
        # İlerlemeyi göster
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Epoch: {epoch+1}/{num_epochs}, "
                  f"Batch: {batch_idx+1}/{total_batches}, "
                  f"Loss: {avg_loss:.4f}", end='\r')
    
    # Ortalama kaybı hesapla
    avg_loss = running_loss / len(train_loader)
    print(error_messages['epoch_completed'].format(epoch + 1, num_epochs, avg_loss))
    
    return avg_loss

def validate_model(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> float:
    """Modeli doğrulama setinde değerlendirir.
    
    Args:
        model: Değerlendirilecek model
        val_loader: Doğrulama veri yükleyicisi
        criterion: Kayıp fonksiyonu
        device: Kullanılacak cihaz (CPU/GPU)
        
    Returns:
        float: Doğrulama kaybı
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
    
    return val_loss / len(val_loader)

def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    model_dir: str = 'runs/train/weights',
    model_name: str = 'model',
    use_drive: bool = True,
    save_interval: int = 1
) -> None:
    """Model eğitimi için ana fonksiyon.
    
    Args:
        model: Eğitilecek model
        train_loader: Eğitim veri yükleyicisi
        val_loader: Doğrulama veri yükleyicisi
        criterion: Kayıp fonksiyonu
        optimizer: Optimizer
        num_epochs: Eğitilecek epoch sayısı
        model_dir: Modellerin kaydedileceği dizin
        model_name: Model dosya adı öneki
        use_drive: Drive'a yedekleme yapılsın mı?
        save_interval: Kaç epoch'ta bir kaydedilecek
    """
    # Cihazı belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Eğitim ortamını hazırla
    drive_initialized = setup_training_environment(use_drive)
    
    # Checkpoint yükle
    checkpoint, start_epoch = load_checkpoint()
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # En iyi modeli takip et
    best_loss = float('inf')
    
    # Eğitim döngüsü
    for epoch in range(start_epoch, num_epochs):
        # Eğitim
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        
        # Doğrulama
        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"✅ Doğrulama Kaybı: {val_loss:.4f}")
        
        # Checkpoint kaydet
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
            # Her zaman son modeli kaydet
            success, last_model_path = save_checkpoint(
                model, optimizer, epoch, train_loss,
                is_best=False, model_dir=model_dir, model_name=model_name
            )
            
            # Eğer Drive aktifse yükle
            if success and drive_initialized:
                upload_to_drive(last_model_path, is_best=False)
            
            # En iyi modeli kaydet
            if val_loss < best_loss:
                best_loss = val_loss
                success, best_model_path = save_checkpoint(
                    model, optimizer, epoch, train_loss,
                    is_best=True, model_dir=model_dir, model_name=model_name
                )
                
                # Eğer Drive aktifse yükle
                if success and drive_initialized:
                    upload_to_drive(best_model_path, is_best=True)
        
        # Drive durumunu göster
        if drive_initialized and (epoch + 1) % 5 == 0:
            print_drive_status()
    
    print("\n🎉 Eğitim tamamlandı!")
    
    # Son durumu göster
    if drive_initialized:
        print("\n📊 Son Durum:")
        print_drive_status()
