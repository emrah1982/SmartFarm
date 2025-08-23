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
    
    def select_existing_folder(self, folder_path: str, project_name: Optional[str] = None) -> bool:
        """Var olan bir klasörü proje klasörü olarak ayarla (timestamp oluşturmadan).
        folder_path: Drive kökünden itibaren yol, ör: "SmartFarm/colab_learn/yolo11_models"
        """
        if not self.service:
            print("❌ Google Drive servisi başlatılmamış!")
            return False
        try:
            folder_parts = [p for p in folder_path.split('/') if p]
            parent_id = 'root'
            for part in folder_parts:
                # Bul veya oluştur (mevcutsa bulur, yoksa oluşturur)
                fid = self._find_or_create_folder(part, parent_id)
                if not fid:
                    return False
                parent_id = fid
            self.drive_folder_id = parent_id
            # Proje adı ayarla
            self.project_name = project_name or folder_parts[-1]
            # Konfigürasyonu kaydet
            self._save_drive_config('/'.join(folder_parts[:-1]) if len(folder_parts) > 1 else '', folder_parts[-1])
            print(f"✅ Var olan klasör proje klasörü olarak ayarlandı: {folder_path}")
            return True
        except Exception as e:
            print(f"❌ Var olan klasör ayarlanamadı: {e}")
            return False
    
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
    
    def upload_model(self, local_path: str, drive_filename: str) -> bool:
        """Uploads a model file to Google Drive, updating it if it already exists."""
        if not self.service or not self.drive_folder_id:
            print("❌ Drive service or folder ID not found!")
            return False

        if not os.path.exists(local_path):
            print(f"❌ Model file not found: {local_path}")
            return False

        try:
            # Check if the file already exists in Drive
            query = f"name='{drive_filename}' and parents in '{self.drive_folder_id}' and trashed=false"
            response = self.service.files().list(q=query, fields='files(id)').execute()
            existing_files = response.get('files', [])

            media = MediaFileUpload(local_path, resumable=True)

            if existing_files:
                # Update existing file
                file_id = existing_files[0]['id']
                self.service.files().update(fileId=file_id, media_body=media).execute()
                print(f"✅ Model güncellendi: {drive_filename}")
            else:
                # Create new file
                file_metadata = {'name': drive_filename, 'parents': [self.drive_folder_id]}
                self.service.files().create(body=file_metadata, media_body=media).execute()
                print(f"✅ Model Drive'a yüklendi: {drive_filename}")
            
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
    
    def find_latest_checkpoint(self) -> Tuple[Optional[str], Optional[str]]:
        """Drive klasöründe ve alt klasörlerde 'last.pt' veya 'best.pt' dosyalarını recursive ara.
        Önce 'last.pt' için en güncelini, yoksa 'best.pt' için en güncelini döndür.
        """
        if not self.service or not self.drive_folder_id:
            return None, None

        try:
            # BFS ile tüm alt klasörleri dolaş
            from collections import deque
            queue = deque([self.drive_folder_id])
            found_last = []  # (file_id, name, modifiedTime)
            found_best = []

            while queue:
                parent = queue.popleft()
                # Çocukları getir (klasör ve dosyalar)
                results = self.service.files().list(
                    q=f"parents in '{parent}' and trashed=false",
                    fields="files(id,name,mimeType,modifiedTime)"
                ).execute()
                items = results.get('files', [])
                for item in items:
                    mime = item.get('mimeType', '')
                    if mime == 'application/vnd.google-apps.folder':
                        queue.append(item['id'])
                    else:
                        name = item.get('name', '')
                        if name == 'last.pt':
                            found_last.append((item['id'], name, item.get('modifiedTime', '')))
                        elif name == 'best.pt':
                            found_best.append((item['id'], name, item.get('modifiedTime', '')))

            def pick_latest(files):
                if not files:
                    return None
                # modifiedTime ISO8601, string olarak karşılaştırmak da çoğu zaman yeterlidir
                files.sort(key=lambda x: x[2], reverse=True)
                return files[0][0], files[0][1]

            latest = pick_latest(found_last) or pick_latest(found_best)
            if latest:
                print(f"🔍 Drive'da bulundu: {latest[1]}")
                return latest[0], latest[1]
            return None, None

        except Exception as e:
            print(f"❌ Drive'da checkpoint arama hatası: {e}")
            return None, None
    
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

    # Var olan bir klasörü kullanmak ister misiniz?
    use_existing_folder = input("\n📁 Var olan bir klasörü kullanmak ister misiniz? (y/n): ").lower()
    if use_existing_folder.startswith('y'):
        print("Örnek yol: SmartFarm/colab_learn/yolo11_models")
        folder_path = input("Drive klasör yolu: ").strip()
        if not folder_path:
            print("❌ Geçerli bir klasör yolu girilmedi.")
            return None
        if not drive_manager.select_existing_folder(folder_path):
            return None
        return drive_manager

    # Yeni klasör yapısı kur
    if not drive_manager.setup_drive_folder():
        return None
    
    return drive_manager

if __name__ == "__main__":
    print("Drive Manager - Google Drive entegrasyon modülü")
    print("Bu modül doğrudan çalıştırılamaz.")
