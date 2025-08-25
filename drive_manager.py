#!/usr/bin/env python3
# drive_manager.py - Google Drive integration for SmartFarm model management

import os
import json
import pickle
import shutil
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Ortam tespiti
IS_COLAB = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False

GOOGLE_DRIVE_AVAILABLE = False
try:
    if not IS_COLAB:
        # Normal Python ortamı için API kütüphaneleri
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
        import io
        GOOGLE_DRIVE_AVAILABLE = True
except ImportError as e:
    if not IS_COLAB:
        print(f"⚠️ Google Drive kütüphane hatası: {e}")
        print("Lütfen aşağıdaki komutları çalıştırarak gerekli kütüphaneleri yükleyin:")
        print("pip install --upgrade google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

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
        self.is_colab = IS_COLAB
        
        # Colab için ek özellikler
        if self.is_colab:
            self.base_drive_path = "/content/drive/MyDrive"
            self.project_folder = None
            self.is_mounted = False
        
    def authenticate(self) -> bool:
        """Google Drive kimlik doğrulama"""
        if self.is_colab:
            return self._authenticate_colab()
        else:
            return self._authenticate_api()
    
    def _authenticate_colab(self) -> bool:
        """Colab için Drive bağlama"""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Drive'ın bağlandığını kontrol et
            if os.path.exists(self.base_drive_path):
                self.is_mounted = True
                print("✅ Google Drive başarıyla bağlandı!")
                return True
            else:
                print("❌ Drive bağlanamadı!")
                return False
                
        except ImportError:
            print("❌ Bu kod Google Colab dışında çalışıyor!")
            return False
        except Exception as e:
            print(f"❌ Drive bağlama hatası: {e}")
            return False
    
    def _authenticate_api(self) -> bool:
        """API ile kimlik doğrulama (normal Python ortamı)"""
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
        if self.is_colab:
            return self._setup_colab_folder()
        else:
            return self._setup_api_folder()
    
    def _setup_colab_folder(self) -> bool:
        """Colab için klasör kurulumu"""
        if not self.is_mounted:
            print("❌ Drive bağlı değil! Önce authenticate() çalıştırın.")
            return False
        
        try:
            # Kullanıcıdan bilgileri al
            print("\n🔧 Google Drive Klasör Ayarları")
            folder_path = input("Klasör yolu (örn: SmartFarm/Training): ").strip()
            if not folder_path:
                folder_path = "SmartFarm/Training"
            
            self.project_name = input("Proje adı (varsayılan: SmartFarm_Training): ").strip()
            if not self.project_name:
                self.project_name = "SmartFarm_Training"
            
            # Zaman damgası oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_folder_name = f"{timestamp}_{self.project_name}"
            
            # Tam klasör yolu
            self.project_folder = os.path.join(self.base_drive_path, folder_path, project_folder_name)
            
            # Klasörleri oluştur
            os.makedirs(self.project_folder, exist_ok=True)
            
            # Alt klasörleri oluştur
            sub_folders = ['models', 'checkpoints', 'logs', 'configs']
            for sub_folder in sub_folders:
                os.makedirs(os.path.join(self.project_folder, sub_folder), exist_ok=True)
            
            print(f"✅ Drive klasörü oluşturuldu: {self.project_folder}")
            
            # Konfigürasyonu kaydet
            self._save_drive_config(folder_path, project_folder_name)
            return True
            
        except Exception as e:
            print(f"❌ Klasör oluşturma hatası: {e}")
            return False
    
    def _setup_api_folder(self) -> bool:
        """API ile klasör kurulumu (orijinal kod)"""
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
    
    def select_existing_folder(self, folder_path: str, project_name: Optional[str] = None) -> bool:
        """Var olan bir klasörü proje klasörü olarak ayarla"""
        if self.is_colab:
            return self._select_existing_colab(folder_path, project_name)
        else:
            return self._select_existing_api(folder_path, project_name)
    
    def _select_existing_colab(self, folder_path: str, project_name: Optional[str] = None) -> bool:
        """Colab için var olan klasör seçimi"""
        if not self.is_mounted:
            print("❌ Drive bağlı değil!")
            return False
        
        full_path = os.path.join(self.base_drive_path, folder_path)
        
        if os.path.exists(full_path):
            self.project_folder = full_path
            self.project_name = project_name or os.path.basename(folder_path)
            print(f"✅ Var olan klasör kullanılacak: {self.project_folder}")
            self._save_drive_config(os.path.dirname(folder_path), os.path.basename(folder_path))
            return True
        else:
            # Klasör yoksa oluştur
            try:
                os.makedirs(full_path, exist_ok=True)
                self.project_folder = full_path
                self.project_name = project_name or os.path.basename(folder_path)
                print(f"✅ Yeni klasör oluşturuldu: {self.project_folder}")
                self._save_drive_config(os.path.dirname(folder_path), os.path.basename(folder_path))
                return True
            except Exception as e:
                print(f"❌ Klasör oluşturulamadı: {e}")
                return False
    
    def _select_existing_api(self, folder_path: str, project_name: Optional[str] = None) -> bool:
        """API için var olan klasör seçimi (orijinal kod)"""
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
    
    def _find_or_create_folder(self, folder_name: str, parent_id: str) -> Optional[str]:
        """Klasör bul veya oluştur (sadece API modu için)"""
        if self.is_colab:
            return None  # Colab modunda bu fonksiyon kullanılmaz
            
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
            'project_name': self.project_name,
            'created_at': datetime.now().isoformat(),
            'is_colab': self.is_colab
        }
        
        if self.is_colab:
            config['project_folder'] = self.project_folder
            config['base_drive_path'] = self.base_drive_path
        else:
            config['drive_folder_id'] = self.drive_folder_id
        
        config_file = '/content/drive_config.json' if self.is_colab else 'drive_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Drive konfigürasyonu kaydedildi: {config_file}")
    
    def load_drive_config(self) -> bool:
        """Kaydedilmiş Drive konfigürasyonunu yükle"""
        config_file = '/content/drive_config.json' if self.is_colab else 'drive_config.json'
        
        if not os.path.exists(config_file):
            return False
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.project_name = config.get('project_name')
            
            if self.is_colab:
                self.project_folder = config.get('project_folder')
                self.base_drive_path = config.get('base_drive_path', '/content/drive/MyDrive')
                # Drive'ın mount edilmiş olup olmadığını kontrol et
                if os.path.exists(self.base_drive_path):
                    self.is_mounted = True
                    print(f"📂 Konfigürasyon yüklendi: {self.project_folder}")
                    return True
                else:
                    print("❌ Drive mount edilmemiş!")
                    return False
            else:
                self.drive_folder_id = config.get('drive_folder_id')
                print(f"📂 Drive konfigürasyonu yüklendi: {config.get('folder_path')}/{config.get('project_folder_name')}")
                return True
                
        except Exception as e:
            print(f"❌ Drive konfigürasyonu yükleme hatası: {e}")
            return False
    
    def upload_model(self, local_path: str, drive_filename: str) -> bool:
        """Model dosyasını Drive'a yükle"""
        if self.is_colab:
            return self._upload_model_colab(local_path, drive_filename)
        else:
            return self._upload_model_api(local_path, drive_filename)
    
    def _upload_model_colab(self, local_path: str, drive_filename: str) -> bool:
        """Colab için model yükleme"""
        if not self.project_folder:
            print("❌ Proje klasörü ayarlanmamış!")
            return False
        
        if not os.path.exists(local_path):
            print(f"❌ Model dosyası bulunamadı: {local_path}")
            return False
        
        try:
            # Hedef yol
            target_path = os.path.join(self.project_folder, 'models', drive_filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Değişiklik algılama: varsa boyut karşılaştır
            local_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            target_exists = os.path.exists(target_path)
            if target_exists:
                target_size = os.path.getsize(target_path)
                if local_size == target_size:
                    print(f"⏭️ Atlandı (değişiklik yok): {drive_filename} ({local_size/(1024*1024):.1f} MB)")
                    return True

            t0 = time.time()
            shutil.copy2(local_path, target_path)
            dt = time.time() - t0

            mb = os.path.getsize(target_path) / (1024*1024)
            speed = (mb / dt) if dt > 0 else 0
            print(f"✅ Model Drive'a kaydedildi: {target_path}")
            print(f"📁 Boyut: {mb:.1f} MB | ⏱️ Süre: {dt:.2f}s | 🚀 Hız: {speed:.2f} MB/s")

            # Log tut
            self._log_upload_colab(drive_filename, local_path, target_path)
            return True

        except Exception as e:
            print(f"❌ Model kaydetme hatası: {e}")
            return False
    
    def _upload_model_api(self, local_path: str, drive_filename: str) -> bool:
        """API ile model yükleme (geliştirilmiş: retry, değişiklik algı, log)"""
        if not self.service or not self.drive_folder_id:
            print("❌ Drive service or folder ID not found!")
            return False

        if not os.path.exists(local_path):
            print(f"❌ Model file not found: {local_path}")
            return False

        try:
            # Yardımcılar
            def _md5(path: str) -> Optional[str]:
                try:
                    h = hashlib.md5()
                    with open(path, 'rb') as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b''):
                            h.update(chunk)
                    return h.hexdigest()
                except Exception:
                    return None

            def _retry(fn, attempts=3):
                delay = 1.0
                last_exc = None
                for i in range(attempts):
                    try:
                        return fn()
                    except Exception as e:
                        last_exc = e
                        print(f"⚠️ Deneme {i+1}/{attempts} hata: {e}. {delay:.1f}s sonra tekrar denenecek...")
                        time.sleep(delay)
                        delay = delay * 2 + 0.5
                if last_exc:
                    raise last_exc

            # Mevcut dosyayı bul ve md5/size al
            query = f"name='{drive_filename}' and parents in '{self.drive_folder_id}' and trashed=false"
            response = self.service.files().list(q=query, fields='files(id, name, md5Checksum, size)').execute()
            existing_files = response.get('files', [])

            local_size = os.path.getsize(local_path)
            local_md5 = _md5(local_path)

            # Değişiklik algıla: aynı md5 veya aynı size ise (md5 yoksa) yüklemeyi atlayabiliriz
            if existing_files:
                meta = existing_files[0]
                file_id = meta['id']
                remote_md5 = meta.get('md5Checksum')
                remote_size = int(meta.get('size', 0)) if meta.get('size') is not None else None

                if (remote_md5 and local_md5 and remote_md5 == local_md5) or (remote_md5 is None and remote_size == local_size):
                    print(f"⏭️ Atlandı (değişiklik yok): {drive_filename} ({local_size/(1024*1024):.1f} MB)")
                    return True

            media = MediaFileUpload(local_path, resumable=True)

            t0 = time.time()
            if existing_files:
                file_id = existing_files[0]['id']

                def _do_update():
                    return self.service.files().update(fileId=file_id, media_body=media).execute()

                result = _retry(_do_update)
                dt = time.time() - t0
                mb = local_size / (1024*1024)
                speed = (mb / dt) if dt > 0 else 0
                print(f"✅ Model güncellendi: {drive_filename} | ⏱️ {dt:.2f}s | 🚀 {speed:.2f} MB/s")
                self._log_upload(drive_filename, 0, file_id, False)
            else:
                file_metadata = {'name': drive_filename, 'parents': [self.drive_folder_id]}

                def _do_create():
                    return self.service.files().create(body=file_metadata, media_body=media).execute()

                result = _retry(_do_create)
                dt = time.time() - t0
                mb = local_size / (1024*1024)
                speed = (mb / dt) if dt > 0 else 0
                print(f"✅ Model Drive'a yüklendi: {drive_filename} | ⏱️ {dt:.2f}s | 🚀 {speed:.2f} MB/s")
                self._log_upload(drive_filename, 0, result.get('id'), False)

            return True

        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            return False
    
    def _log_upload_colab(self, filename: str, source_path: str, target_path: str):
        """Colab için yükleme kaydı"""
        log_entry = {
            'filename': filename,
            'source_path': source_path,
            'target_path': target_path,
            'uploaded_at': datetime.now().isoformat(),
            'file_size': os.path.getsize(target_path) if os.path.exists(target_path) else 0
        }
        
        log_file = os.path.join(self.project_folder, 'logs', 'uploads.json')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Mevcut logları yükle
        uploads = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    uploads = json.load(f)
            except:
                uploads = []
        
        uploads.append(log_entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(uploads, f, indent=2, ensure_ascii=False)
    
    def _log_upload(self, filename: str, epoch: int, file_id: str, is_best: bool):
        """API için yükleme kaydını tut (orijinal kod)"""
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
        """En son checkpoint'i bul"""
        if self.is_colab:
            return self._find_checkpoint_colab()
        else:
            return self._find_checkpoint_api()
    
    def _find_checkpoint_colab(self) -> Tuple[Optional[str], Optional[str]]:
        """Colab için checkpoint arama"""
        if not self.project_folder:
            print("❌ Proje klasörü ayarlanmamış!")
            return None, None
        
        # Önce checkpoints klasörünü kontrol et
        checkpoint_dir = os.path.join(self.project_folder, 'checkpoints')
        
        # Eğer checkpoints klasörü yoksa models klasörünü kontrol et
        if not os.path.exists(checkpoint_dir):
            checkpoint_dir = os.path.join(self.project_folder, 'models')
            if not os.path.exists(checkpoint_dir):
                print("❌ Ne checkpoint ne de models klasörü bulunamadı!")
                return None, None
        
        try:
            # Önce last.pt, sonra best.pt ara
            for filename in ['last.pt', 'best.pt']:
                checkpoint_path = os.path.join(checkpoint_dir, filename)
                if os.path.exists(checkpoint_path):
                    file_size = os.path.getsize(checkpoint_path) / (1024*1024)
                    print(f"✅ Checkpoint bulundu: {checkpoint_path} ({file_size:.1f} MB)")
                    return checkpoint_path, filename
            
            # Diğer .pt dosyalarını ara
            pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if pt_files:
                # En yeni dosyayı al
                latest_file = max(pt_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
                latest_path = os.path.join(checkpoint_dir, latest_file)
                file_size = os.path.getsize(latest_path) / (1024*1024)
                print(f"✅ En yeni checkpoint bulundu: {latest_path} ({file_size:.1f} MB)")
                return latest_path, latest_file
            
            print(f"❌ {checkpoint_dir} klasöründe hiçbir checkpoint bulunamadı!")
            return None, None
            
        except Exception as e:
            print(f"❌ Checkpoint arama hatası: {e}")
            return None, None
    
    def _find_checkpoint_api(self) -> Tuple[Optional[str], Optional[str]]:
        """API ile checkpoint arama (orijinal kod)"""
        if not self.service or not self.drive_folder_id:
            print("❌ Drive servisi veya klasör ID'si bulunamadı!")
            return None, None

        try:
            print(f"🔍 Drive'da checkpoint aranıyor (Klasör ID: {self.drive_folder_id})...")
            
            # BFS ile tüm alt klasörleri dolaş
            from collections import deque
            queue = deque([(self.drive_folder_id, "")])  # (folder_id, path)
            found_last = []  # (file_id, name, modifiedTime, path)
            found_best = []
            processed_folders = set()
            processed_files = set()

            while queue:
                parent_id, parent_path = queue.popleft()
                
                # Aynı klasörü tekrar işleme
                if parent_id in processed_folders:
                    continue
                processed_folders.add(parent_id)
                
                try:
                    # Çocukları getir (klasör ve dosyalar)
                    results = self.service.files().list(
                        q=f"'{parent_id}' in parents and trashed=false",
                        fields="files(id,name,mimeType,modifiedTime)",
                        pageSize=1000
                    ).execute()
                    
                    items = results.get('files', [])
                    
                    for item in items:
                        item_id = item['id']
                        mime = item.get('mimeType', '')
                        name = item.get('name', '')
                        
                        if mime == 'application/vnd.google-apps.folder':
                            # Klasörse queue'ya ekle
                            folder_name = name
                            folder_path = f"{parent_path}/{folder_name}" if parent_path else folder_name
                            queue.append((item_id, folder_path))
                        else:
                            # Dosyaysa ve daha önce işlenmediyse kontrol et
                            if item_id not in processed_files:
                                processed_files.add(item_id)
                                
                                if name == 'last.pt' or name == 'best.pt':
                                    file_path = f"{parent_path}/{name}" if parent_path else name
                                    file_info = (item_id, name, item.get('modifiedTime', ''), file_path)
                                    
                                    if name == 'last.pt':
                                        found_last.append(file_info)
                                    elif name == 'best.pt':
                                        found_best.append(file_info)
                                    
                                    print(f"✅ Bulundu: {file_path} (Son değişiklik: {item.get('modifiedTime', 'bilinmiyor')})")
                
                except Exception as e:
                    print(f"❌ Klasör içeriği alınırken hata (ID: {parent_id}, Yol: {parent_path}): {str(e)}")
                    continue

            def pick_latest(files):
                if not files:
                    return None, None
                # modifiedTime'a göre sırala (en yeni en başta)
                files.sort(key=lambda x: x[2], reverse=True)
                print(f"📊 En güncel dosya seçildi: {files[0][3]} (Tarih: {files[0][2]})")
                return files[0][0], files[0][1]  # (file_id, filename)

            # Önce last.pt, yoksa best.pt'yi dene
            latest = pick_latest(found_last) or pick_latest(found_best)
            
            if not latest[0]:
                print("❌ Drive'da uygun bir checkpoint dosyası bulunamadı.")
                
            return latest

        except Exception as e:
            print(f"❌ Drive'da checkpoint arama sırasında beklenmeyen bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def download_checkpoint(self, file_id_or_path: str, local_path: str) -> bool:
        """Checkpoint'i indir"""
        if self.is_colab:
            return self._download_checkpoint_colab(file_id_or_path, local_path)
        else:
            return self._download_checkpoint_api(file_id_or_path, local_path)
    
    def _download_checkpoint_colab(self, checkpoint_path: str, local_path: str) -> bool:
        """Colab için checkpoint kopyalama"""
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint dosyası bulunamadı: {checkpoint_path}")
            return False
        
        try:
            # Dosyayı kopyala
            shutil.copy2(checkpoint_path, local_path)
            print(f"✅ Checkpoint kopyalandı: {local_path}")
            return True
            
        except Exception as e:
            print(f"❌ Checkpoint kopyalama hatası: {e}")
            return False
    
    def _download_checkpoint_api(self, file_id: str, local_path: str) -> bool:
        """API ile checkpoint indirme (orijinal kod)"""
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
        if self.is_colab:
            return self._list_models_colab()
        else:
            return self._list_models_api()
    
    def _list_models_colab(self) -> List[Dict]:
        """Colab için model listeleme"""
        if not self.project_folder:
            return []
        
        models_dir = os.path.join(self.project_folder, 'models')
        
        if not os.path.exists(models_dir):
            return []
        
        try:
            model_files = []
            for filename in os.listdir(models_dir):
                if filename.endswith(('.pt', '.pth', '.onnx')):
                    file_path = os.path.join(models_dir, filename)
                    stat = os.stat(file_path)
                    
                    model_info = {
                        'name': filename,
                        'size': str(stat.st_size),
                        'createdTime': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modifiedTime': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'path': file_path
                    }
                    model_files.append(model_info)
            
            if model_files:
                print(f"\n📋 Drive'daki modeller ({len(model_files)} adet):")
                for i, file in enumerate(model_files, 1):
                    size_mb = int(file['size']) / (1024 * 1024)
                    created = file['createdTime'][:19].replace('T', ' ')
                    print(f"   {i}. {file['name']} ({size_mb:.1f} MB) - {created}")
            
            return model_files
            
        except Exception as e:
            print(f"❌ Model listeleme hatası: {e}")
            return []
    
    def _list_models_api(self) -> List[Dict]:
        """API ile model listeleme (orijinal kod)"""
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
    
    # Drive Manager oluştur
    drive_manager = DriveManager()
    
    if drive_manager.is_colab:
        print("🔍 Google Colab ortamı tespit edildi!")
        print("📱 Basitleştirilmiş Drive entegrasyonu kullanılacak.")
    else:
        print("🖥️ Standart Python ortamı tespit edildi!")
        print("🔐 OAuth2 kimlik doğrulama gerekli.")
        
        # Credentials dosyası kontrolü
        if not os.path.exists("credentials.json"):
            if drive_manager.is_colab:
                print("ℹ️ Colab'de credentials.json dosyasına ihtiyaç yoktur.")
                return drive_manager
            print("❌ credentials.json dosyası bulunamadı! (Colab'de credentials.json dosyasına ihtiyaç yoktur)")
            print("\n📋 Kurulum Adımları:")
            print("1. Google Cloud Console'a gidin (https://console.cloud.google.com/)")
            print("2. Yeni proje oluşturun veya mevcut projeyi seçin")
            print("3. Google Drive API'yi etkinleştirin")
            print("4. OAuth 2.0 Client ID oluşturun (Desktop Application)")
            print("5. credentials.json dosyasını indirin ve bu klasöre koyun")
            print("6. Tekrar çalıştırın")
            return None
    
    # Kimlik doğrulama
    if not drive_manager.authenticate():
        return None
    
    # Proje klasörü kurulumu (eksik olan kısım!)
    print("\n📁 Proje klasörü kurulumu...")
    if not drive_manager.setup_drive_folder():
        print("❌ Proje klasörü kurulamadı!")
        return None
    
    print("✅ Drive entegrasyonu tamamlandı!")
    return drive_manager

def activate_drive_integration(folder_path: str, project_name: Optional[str] = None) -> Optional[DriveManager]:
    """Etkileşimsiz (non-interactive) Drive entegrasyonu başlatır.

    Parametreler:
      - folder_path: Drive üzerinde kullanılacak proje klasörü yolu.
        Örnek API modu: "Tarım/SmartFarm/Models"
        Örnek Colab modu: "SmartFarm/Training/20250825_Projex"
      - project_name: İsteğe bağlı proje adı. Belirtilmezse klasör adından türetilir.

    Dönüş:
      - Başarılıysa yapılandırılmış DriveManager döner, aksi halde None.
    """
    try:
        dm = DriveManager()
        # Kimlik doğrulama
        if not dm.authenticate():
            print("❌ Drive kimlik doğrulama başarısız!")
            return None

        # Var olan (veya yoksa oluşturulacak) klasörü proje klasörü olarak seç
        ok = dm.select_existing_folder(folder_path, project_name)
        if not ok:
            print(f"❌ Proje klasörü ayarlanamadı: {folder_path}")
            return None

        print("✅ Drive entegrasyonu hazır (etkileşimsiz mod)")
        return dm
    except Exception as e:
        print(f"❌ Drive entegrasyonu başlatılamadı: {e}")
        return None



if __name__ == "__main__":
    print("Drive Manager - Google Drive entegrasyon modülü")
    
    # Test kurulumu
    dm = setup_drive_integration()
    
    if dm:
        print("\n✅ Drive entegrasyonu başarıyla kuruldu!")
        
        if dm.is_colab:
            print(f"📁 Proje klasörü: {dm.project_folder}")
            
            # İstatistikler
            if dm.project_folder and os.path.exists(dm.project_folder):
                total_size = 0
                for root, dirs, files in os.walk(dm.project_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size += os.path.getsize(file_path)
                
                print(f"📊 Toplam boyut: {total_size / (1024 * 1024):.1f} MB")
        else:
            print(f"🆔 Drive klasör ID: {dm.drive_folder_id}")
            
        # Mevcut modelleri listele
        dm.list_drive_models()
    else:
        print("❌ Drive entegrasyonu kurulamadı!")