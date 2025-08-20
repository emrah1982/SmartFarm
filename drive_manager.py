#!/usr/bin/env python3
# drive_manager.py - Google Drive integration for SmartFarm model management

import os
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    import io
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    print("Google Drive kütüphaneleri bulunamadı. 'pip install -r requirements.txt' çalıştırın.")

# Google Drive API kapsamları
SCOPES = ['https://www.googleapis.com/auth/drive.file']

class DriveManager:
    """Google Drive ile model yönetimi için sınıf"""
    
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.pickle"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None
        self.drive_folder_id = None
        self.project_name = None
        
    def authenticate(self) -> bool:
        """Google Drive kimlik doğrulama"""
        if not GOOGLE_DRIVE_AVAILABLE:
            print("❌ Google Drive kütüphaneleri yüklü değil!")
            return False
            
        creds = None
        
        # Token dosyası varsa yükle
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # Geçerli kimlik bilgileri yoksa veya süresi dolmuşsa
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    print(f"❌ Kimlik dosyası bulunamadı: {self.credentials_path}")
                    print("Google Cloud Console'dan OAuth 2.0 credentials indirin ve 'credentials.json' olarak kaydedin.")
                    return False
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Token'ı kaydet
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        try:
            self.service = build('drive', 'v3', credentials=creds)
            print("✅ Google Drive kimlik doğrulama başarılı!")
            return True
        except Exception as e:
            print(f"❌ Google Drive bağlantı hatası: {e}")
            return False
    
    def setup_drive_folder(self) -> bool:
        """Drive'da proje klasörü yapısını oluştur"""
        if not self.service:
            print("❌ Google Drive servisi başlatılmamış!")
            return False
        
        # Kullanıcıdan klasör yolu iste
        print("\n🔧 Google Drive Klasör Ayarları")
        print("Örnek: Tarım/SmartFarm")
        print("Bu, Drive'ınızda şu yapıyı oluşturacak:")
        print("  📁 Tarım/")
        print("    📁 SmartFarm/")
        print("      📁 [timestamp]_model/")
        
        folder_path = input("\nDrive'da oluşturulacak klasör yolu: ").strip()
        if not folder_path:
            folder_path = "Tarım/SmartFarm"
            print(f"Varsayılan klasör kullanılıyor: {folder_path}")
        
        # Proje adı al
        self.project_name = input("Proje adı (varsayılan: SmartFarm_Training): ").strip()
        if not self.project_name:
            self.project_name = "SmartFarm_Training"
        
        try:
            # Klasör yapısını oluştur
            folder_parts = folder_path.split('/')
            parent_id = 'root'
            
            for folder_name in folder_parts:
                folder_id = self._find_or_create_folder(folder_name, parent_id)
                if not folder_id:
                    return False
                parent_id = folder_id
            
            # Zaman damgalı proje klasörü oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_folder_name = f"{timestamp}_{self.project_name}"
            self.drive_folder_id = self._find_or_create_folder(project_folder_name, parent_id)
            
            if self.drive_folder_id:
                print(f"✅ Drive klasörü oluşturuldu: {folder_path}/{project_folder_name}")
                
                # Klasör bilgilerini kaydet
                self._save_drive_config(folder_path, project_folder_name)
                return True
            else:
                print("❌ Proje klasörü oluşturulamadı!")
                return False
                
        except Exception as e:
            print(f"❌ Klasör oluşturma hatası: {e}")
            return False
    
    def _find_or_create_folder(self, folder_name: str, parent_id: str) -> Optional[str]:
        """Klasör bul veya oluştur"""
        try:
            # Önce klasörün var olup olmadığını kontrol et
            query = f"name='{folder_name}' and parents in '{parent_id}' and mimeType='application/vnd.google-apps.folder'"
            results = self.service.files().list(q=query).execute()
            items = results.get('files', [])
            
            if items:
                print(f"📁 Mevcut klasör bulundu: {folder_name}")
                return items[0]['id']
            
            # Klasör yoksa oluştur
            folder_metadata = {
                'name': folder_name,
                'parents': [parent_id],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            folder = self.service.files().create(body=folder_metadata).execute()
            print(f"📁 Yeni klasör oluşturuldu: {folder_name}")
            return folder.get('id')
            
        except Exception as e:
            print(f"❌ Klasör işlemi hatası ({folder_name}): {e}")
            return None
    
    def _save_drive_config(self, folder_path: str, project_folder_name: str):
        """Drive konfigürasyonunu kaydet"""
        config = {
            'folder_path': folder_path,
            'project_folder_name': project_folder_name,
            'drive_folder_id': self.drive_folder_id,
            'project_name': self.project_name,
            'created_at': datetime.now().isoformat()
        }
        
        with open('drive_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Drive konfigürasyonu kaydedildi: drive_config.json")
    
    def load_drive_config(self) -> bool:
        """Kaydedilmiş Drive konfigürasyonunu yükle"""
        if not os.path.exists('drive_config.json'):
            return False
        
        try:
            with open('drive_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.drive_folder_id = config.get('drive_folder_id')
            self.project_name = config.get('project_name')
            
            print(f"📂 Drive konfigürasyonu yüklendi: {config.get('folder_path')}/{config.get('project_folder_name')}")
            return True
            
        except Exception as e:
            print(f"❌ Drive konfigürasyonu yükleme hatası: {e}")
            return False
    
    def upload_model(self, model_path: str, epoch: int, is_best: bool = False) -> bool:
        """Modeli Drive'a yükle"""
        if not self.service or not self.drive_folder_id:
            print("❌ Drive servisi veya klasör ID'si bulunamadı!")
            return False
        
        if not os.path.exists(model_path):
            print(f"❌ Model dosyası bulunamadı: {model_path}")
            return False
        
        try:
            # Dosya adını oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_best:
                filename = f"best_model_epoch_{epoch}_{timestamp}.pt"
            else:
                filename = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
            
            # Dosya metadata'sı
            file_metadata = {
                'name': filename,
                'parents': [self.drive_folder_id]
            }
            
            # Dosyayı yükle
            media = MediaFileUpload(model_path, resumable=True)
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,size'
            ).execute()
            
            # Dosya boyutunu hesapla
            file_size = os.path.getsize(model_path)
            size_mb = file_size / (1024 * 1024)
            
            print(f"✅ Model Drive'a yüklendi:")
            print(f"   📄 Dosya: {filename}")
            print(f"   📊 Boyut: {size_mb:.2f} MB")
            print(f"   🆔 Drive ID: {file.get('id')}")
            
            # Yükleme kaydını tut
            self._log_upload(filename, epoch, file.get('id'), is_best)
            
            return True
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def _log_upload(self, filename: str, epoch: int, file_id: str, is_best: bool):
        """Yükleme kaydını tut"""
        log_entry = {
            'filename': filename,
            'epoch': epoch,
            'file_id': file_id,
            'is_best': is_best,
            'uploaded_at': datetime.now().isoformat()
        }
        
        # Log dosyasını güncelle
        log_file = 'drive_uploads.json'
        uploads = []
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                uploads = json.load(f)
        
        uploads.append(log_entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(uploads, f, indent=2, ensure_ascii=False)
    
    def find_latest_checkpoint(self) -> Tuple[Optional[str], int]:
        """En son checkpoint'i bul"""
        if not os.path.exists('drive_uploads.json'):
            return None, 0
        
        try:
            with open('drive_uploads.json', 'r', encoding='utf-8') as f:
                uploads = json.load(f)
            
            # En son epoch'u bul
            latest_epoch = 0
            latest_file_id = None
            
            for upload in uploads:
                if upload['epoch'] > latest_epoch:
                    latest_epoch = upload['epoch']
                    latest_file_id = upload['file_id']
            
            return latest_file_id, latest_epoch
            
        except Exception as e:
            print(f"❌ Checkpoint arama hatası: {e}")
            return None, 0
    
    def download_checkpoint(self, file_id: str, local_path: str) -> bool:
        """Checkpoint'i Drive'dan indir"""
        if not self.service:
            print("❌ Drive servisi başlatılmamış!")
            return False
        
        try:
            # Dosyayı indir
            request = self.service.files().get_media(fileId=file_id)
            
            with open(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"📥 İndiriliyor: {int(status.progress() * 100)}%")
            
            print(f"✅ Checkpoint indirildi: {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ Checkpoint indirme hatası: {e}")
            return False
    
    def list_drive_models(self) -> List[Dict]:
        """Drive'daki modelleri listele"""
        if not self.service or not self.drive_folder_id:
            return []
        
        try:
            query = f"parents in '{self.drive_folder_id}' and name contains '.pt'"
            results = self.service.files().list(
                q=query,
                fields="files(id,name,size,createdTime,modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            
            print(f"\n📋 Drive'daki modeller ({len(files)} adet):")
            for i, file in enumerate(files, 1):
                size_mb = int(file.get('size', 0)) / (1024 * 1024)
                created = file.get('createdTime', '')[:19].replace('T', ' ')
                print(f"   {i}. {file['name']} ({size_mb:.1f} MB) - {created}")
            
            return files
            
        except Exception as e:
            print(f"❌ Model listeleme hatası: {e}")
            return []

def setup_drive_integration() -> Optional[DriveManager]:
    """Drive entegrasyonunu kur"""
    print("\n🚀 Google Drive Entegrasyonu Kurulumu")
    print("=" * 50)
    
    # Credentials dosyası kontrolü
    if not os.path.exists("credentials.json"):
        print("❌ credentials.json dosyası bulunamadı!")
        print("\n📋 Kurulum Adımları:")
        print("1. Google Cloud Console'a gidin (https://console.cloud.google.com/)")
        print("2. Yeni proje oluşturun veya mevcut projeyi seçin")
        print("3. Google Drive API'yi etkinleştirin")
        print("4. OAuth 2.0 Client ID oluşturun (Desktop Application)")
        print("5. credentials.json dosyasını indirin ve bu klasöre koyun")
        print("6. Tekrar çalıştırın")
        return None
    
    # Drive Manager oluştur
    drive_manager = DriveManager()
    
    # Kimlik doğrulama
    if not drive_manager.authenticate():
        return None
    
    # Mevcut konfigürasyon var mı kontrol et
    if drive_manager.load_drive_config():
        use_existing = input("\n📂 Mevcut Drive konfigürasyonu bulundu. Kullanılsın mı? (y/n): ").lower()
        if use_existing.startswith('y'):
            return drive_manager
    
    # Yeni klasör yapısı kur
    if not drive_manager.setup_drive_folder():
        return None
    
    return drive_manager

if __name__ == "__main__":
    print("Drive Manager - Google Drive entegrasyon modülü")
    print("Bu modül doğrudan çalıştırılamaz.")
