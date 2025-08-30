#!/usr/bin/env python3
# drive_manager.py - Google Drive integration for SmartFarm model management

import os
import sys
import json
import pickle
import shutil
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

# Oturum (runtime) boyunca seçilen timestamp'i sabitlemek için global cache
_GLOBAL_SESSION_TS: Optional[str] = None

# Ortam tespiti - Geliştirilmiş
def detect_colab_environment():
    """Colab ortamını güvenli şekilde tespit et"""
    try:
        # get_ipython() fonksiyonunu kontrol et
        if 'get_ipython' in globals():
            ipython_info = str(get_ipython())
            if 'google.colab' in ipython_info:
                return True
        
        # Alternatif kontrol: sys.modules
        import sys
        if 'google.colab' in sys.modules:
            return True
            
        # Alternatif kontrol: ortam değişkenleri
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            return True
            
        return False
    except Exception as e:
        print(f"⚠️ Colab tespit hatası: {e}")
        return False

IS_COLAB = detect_colab_environment()

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
        # Zaman damgalı kök klasör (global erişim için)
        self.active_timestamp_dir: Optional[str] = None
        # Standart alt klasör yolları
        self._subdirs = {
            'models': None,
            'checkpoints': None,
            'logs': None,
            'configs': None,
        }
        
        # Colab için ek özellikler
        if self.is_colab:
            self.base_drive_path = "/content/drive/MyDrive"
            self.project_folder = None
            self.is_mounted = False
        # Tee logging durumu
        self._tee_enabled: bool = False
        self._tee_file_path: Optional[str] = None

    # Yardımcılar: standart alt klasör yolları
    def get_timestamp_dir(self) -> Optional[str]:
        return self.active_timestamp_dir or self.project_folder

    def get_subdir(self, name: str) -> Optional[str]:
        ts = self.get_timestamp_dir()
        if not ts:
            return None
        if name in self._subdirs and self._subdirs[name]:
            return self._subdirs[name]
        path = os.path.join(ts, name)
        self._subdirs[name] = path
        return path

    def get_checkpoints_dir(self) -> Optional[str]:
        return self.get_subdir('checkpoints')

    def get_models_dir(self) -> Optional[str]:
        return self.get_subdir('models')

    def get_logs_dir(self) -> Optional[str]:
        return self.get_subdir('logs')

    def get_configs_dir(self) -> Optional[str]:
        return self.get_subdir('configs')
        
    def authenticate(self) -> bool:
        """Google Drive kimlik doğrulama"""
        if self.is_colab:
            return self._authenticate_colab()
        else:
            return self._authenticate_api()
    
    def _authenticate_colab(self) -> bool:
        """Colab için Drive bağlama - Güvenli Versiyon"""
        try:
            from google.colab import drive
            print("🔄 Google Drive mount işlemi başlatılıyor...")
            
            # Önce mevcut mount durumunu kontrol et
            if os.path.exists(self.base_drive_path):
                print("ℹ️ Drive zaten mount edilmiş görünüyor, kontrol ediliyor...")
                
                # Yazma testi yap
                try:
                    test_file = os.path.join(self.base_drive_path, 'test_write.txt')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    
                    self.is_mounted = True
                    print("✅ Mevcut Drive mount'u çalışıyor!")
                    print(f"📁 Drive yolu: {self.base_drive_path}")
                    return True
                    
                except Exception:
                    print("⚠️ Mevcut mount çalışmıyor, yeniden mount ediliyor...")
            
            # Drive mount et - önce normal mount dene
            try:
                drive.mount('/content/drive')
                print("✅ Normal mount başarılı")
            except Exception as mount_error:
                print(f"⚠️ Normal mount başarısız: {mount_error}")
                
                # force_remount'u daha güvenli şekilde dene
                try:
                    print("🔄 Force remount deneniyor...")
                    # Kernel referansı sorununu önlemek için farklı yaklaşım
                    import subprocess
                    result = subprocess.run(['python', '-c', 
                        'from google.colab import drive; drive.mount("/content/drive", force_remount=True)'], 
                        capture_output=True, text=True, timeout=30)
                    
                    if result.returncode != 0:
                        # Subprocess başarısız, direkt mount dene
                        drive.mount('/content/drive')
                        
                except Exception as force_error:
                    print(f"⚠️ Force remount başarısız: {force_error}")
                    # Son çare: basit mount
                    drive.mount('/content/drive')
            
            # Mount sonrası kontrol
            if os.path.exists(self.base_drive_path):
                # İzin kontrolü
                try:
                    test_file = os.path.join(self.base_drive_path, 'test_write.txt')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    
                    self.is_mounted = True
                    print("✅ Google Drive başarıyla bağlandı ve yazma izni var!")
                    print(f"📁 Drive yolu: {self.base_drive_path}")
                    return True
                    
                except PermissionError:
                    print("❌ Drive bağlandı ama yazma izni yok!")
                    return False
                except Exception as perm_e:
                    print(f"❌ İzin testi hatası: {perm_e}")
                    return False
            else:
                print(f"❌ Drive bağlanamadı! Yol mevcut değil: {self.base_drive_path}")
                print("💡 Çözüm önerileri:")
                print("  1. Colab'de 'Files' panelinden Drive'ı manuel mount edin")
                print("  2. Google hesabınızın Drive erişim izni olduğunu kontrol edin")
                print("  3. Runtime'ı yeniden başlatıp tekrar deneyin")
                return False
                
        except ImportError:
            print("❌ Bu kod Google Colab dışında çalışıyor!")
            print(f"🔍 Tespit edilen ortam: IS_COLAB={self.is_colab}")
            return False
        except Exception as e:
            print(f"❌ Drive bağlama hatası: {e}")
            print("💡 Çözüm önerileri:")
            print("  1. Runtime > Restart runtime menüsünden yeniden başlatın")
            print("  2. Google hesabınızı yeniden doğrulayın")
            print("  3. Manuel mount: from google.colab import drive; drive.mount('/content/drive')")
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

    def select_existing_folder(self, folder_path: str, project_name: Optional[str] = None) -> bool:
        """Var olan bir klasörü proje klasörü olarak ayarla (sınıf içi sargı)."""
        try:
            # Modül düzeyindeki yardımcıları çağır
            if self.is_colab:
                return _select_existing_colab(self, folder_path, project_name)
            else:
                return _select_existing_api(self, folder_path, project_name)
        except NameError:
            # Eski sürümlerle uyum: doğrudan sınıf metotları varsa onları dene
            if self.is_colab and hasattr(self, '_select_existing_colab'):
                return self._select_existing_colab(folder_path, project_name)
            if not self.is_colab and hasattr(self, '_select_existing_api'):
                return self._select_existing_api(folder_path, project_name)
            print("❌ select_existing_folder yardımcıları bulunamadı")
            return False

class _TeeStdout:
    def __init__(self, original, file_handle):
        self.original = original
        self.file_handle = file_handle
        # Colab uyumluluğu için encoding/iwrite
        self.encoding = getattr(original, 'encoding', 'utf-8')
    def write(self, data):
        try:
            self.original.write(data)
        except Exception:
            pass
        try:
            self.file_handle.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass
        try:
            self.file_handle.flush()
        except Exception:
            pass
    def isatty(self):
        try:
            return self.original.isatty()
        except Exception:
            return False
    
    def _setup_colab_folder(self) -> bool:
        """Colab için klasör kurulumu - Otomatik ve Manuel Seçenekli"""
        if not self.is_mounted:
            print("❌ Drive bağlı değil! Önce authenticate() çalıştırın.")
            return False
        
        try:
            print("\n🔧 Google Drive Klasör Ayarları")
            
            # Otomatik kurulum seçeneği
            auto_setup = input("Otomatik klasör kurulumu kullanılsın mı? (e/h, varsayılan: e): ").lower().strip()
            if not auto_setup or auto_setup.startswith('e'):
                # Otomatik kurulum
                folder_path = "SmartFarm/colab_learn/yolo11_models"
                self.project_name = "YOLO11"
                print(f"✅ Otomatik kurulum: {folder_path}")
            else:
                # Manuel kurulum
                folder_path = input("Klasör yolu (örn: SmartFarm/colab_learn/yolo11_models): ").strip()
                if not folder_path:
                    folder_path = "SmartFarm/colab_learn/yolo11_models"
                
                self.project_name = input("Proje adı (varsayılan: YOLO11): ").strip()
                if not self.project_name:
                    self.project_name = "YOLO11"
            
            # Zaman damgası oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Kullanıcı talebine göre sadece timestamp klasör adı
            project_folder_name = f"{timestamp}"
            
            # Tam klasör yolu
            self.project_folder = os.path.join(self.base_drive_path, folder_path, project_folder_name)
            
            # Klasörleri oluştur
            os.makedirs(self.project_folder, exist_ok=True)
            
            # Alt klasörleri oluştur ve global yol referanslarını güncelle
            sub_folders = ['models', 'checkpoints', 'logs', 'configs']
            for sub_folder in sub_folders:
                os.makedirs(os.path.join(self.project_folder, sub_folder), exist_ok=True)
                self._subdirs[sub_folder] = os.path.join(self.project_folder, sub_folder)
            # Global timestamp kökü
            self.active_timestamp_dir = self.project_folder
            
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
            
            # Zaman damgalı proje klasörü oluştur (sadece timestamp)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_folder_name = f"{timestamp}"
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
            # Global timestamp kökü olarak işaretle ve alt klasör referanslarını hazırla
            self.active_timestamp_dir = self.project_folder
            for sub in ['models', 'checkpoints', 'logs', 'configs']:
                self._subdirs[sub] = os.path.join(self.project_folder, sub)
            print(f"✅ Var olan klasör kullanılacak: {self.project_folder}")
            self._save_drive_config(os.path.dirname(folder_path), os.path.basename(folder_path))
            return True
        else:
            # Klasör yoksa oluştur
            try:
                os.makedirs(full_path, exist_ok=True)
                self.project_folder = full_path
                self.project_name = project_name or os.path.basename(folder_path)
                # Global timestamp kökü olarak işaretle ve alt klasör referanslarını hazırla
                self.active_timestamp_dir = self.project_folder
                for sub in ['models', 'checkpoints', 'logs', 'configs']:
                    subp = os.path.join(self.project_folder, sub)
                    os.makedirs(subp, exist_ok=True)
                    self._subdirs[sub] = subp
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
            config['active_timestamp_dir'] = self.active_timestamp_dir or self.project_folder
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
                self.active_timestamp_dir = config.get('active_timestamp_dir', self.project_folder)
                # Alt klasör referansları
                if self.project_folder:
                    for sub in ['models', 'checkpoints', 'logs', 'configs']:
                        self._subdirs[sub] = os.path.join(self.project_folder, sub)
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

    def upload_file(self, local_path: str, rel_path: str) -> bool:
        """Belirtilen yerel dosyayı proje klasörü altındaki rel_path konumuna kopyala.

        Örnekler:
          rel_path = 'last.pt'                         -> <project_folder>/last.pt
          rel_path = 'checkpoints/weights/epoch_003.pt'-> <project_folder>/checkpoints/weights/epoch_003.pt
        """
        try:
            if not self.project_folder:
                print("❌ Proje klasörü ayarlanmamış!")
                return False
            if not os.path.exists(local_path):
                print(f"❌ Yerel dosya bulunamadı: {local_path}")
                return False
            # Her zaman ilk seçilen timestamp'i kullan
            ts_dir = getattr(self, 'active_timestamp_dir', None) or self.project_folder
            if not os.path.isdir(ts_dir):
                print(f"❌ Timestamp klasörü bulunamadı: {ts_dir}")
                return False
            target_path = os.path.join(ts_dir, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # Boyut aynıysa kopyalamayı atla
            if os.path.exists(target_path) and os.path.getsize(target_path) == os.path.getsize(local_path):
                print(f"↪️ Atlandı (değişiklik yok): {target_path}")
                return True
            shutil.copy2(local_path, target_path)
            print(f"✅ Dosya Drive'a kaydedildi: {target_path}")
            return True
        except Exception as e:
            print(f"❌ Dosya kaydetme hatası: {e}")
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
    
    def copy_directory_to_drive(self, local_dir: str, target_rel_path: str = 'checkpoints') -> bool:
        """Yerel bir klasörü Drive'daki timestamp'li proje klasörünün içine kopyala.

        - Colab modunda: dosya sistemi üstünden doğrudan kopyalar (hızlı ve güvenilir).
        - Hedef: self.project_folder/target_rel_path
        - Mevcut dosyalarda boyut aynıysa kopyalamayı atlar.
        """
        try:
            if not self.is_colab:
                print("⚠️ copy_directory_to_drive şu an Colab dışı modda uygulanmadı.")
                return False
            if not self.is_mounted or not self.project_folder:
                print("❌ Drive bağlı değil veya proje klasörü ayarlanmamış!")
                return False
            if not os.path.isdir(local_dir):
                print(f"❌ Yerel klasör bulunamadı: {local_dir}")
                return False

            # Her zaman İLK seçilen timestamp'i kullan (active_timestamp_dir > project_folder)
            ts_dir = getattr(self, 'active_timestamp_dir', None) or self.project_folder
            # Güvence: timestamp dizini mevcut olmalı
            if not os.path.isdir(ts_dir):
                print(f"❌ Timestamp klasörü bulunamadı: {ts_dir}")
                return False

            dst_root = os.path.join(ts_dir, target_rel_path)
            os.makedirs(dst_root, exist_ok=True)

            print(f"🎯 Hedef timestamp klasörü: {ts_dir}")
            print(f"📍 Kopyalama hedefi: {dst_root}")

            copied, skipped, total_size = 0, 0, 0
            t0 = time.time()
            for root, dirs, files in os.walk(local_dir):
                rel = os.path.relpath(root, local_dir)
                dst_dir = os.path.join(dst_root, rel) if rel != '.' else dst_root
                os.makedirs(dst_dir, exist_ok=True)
                for fname in files:
                    src = os.path.join(root, fname)
                    dst = os.path.join(dst_dir, fname)
                    try:
                        src_sz = os.path.getsize(src)
                        if os.path.exists(dst) and os.path.getsize(dst) == src_sz:
                            skipped += 1
                            continue
                        shutil.copy2(src, dst)
                        total_size += src_sz
                        copied += 1
                    except Exception as e:
                        print(f"⚠️ Kopyalama hatası: {src} -> {dst}: {e}")

            dt = time.time() - t0
            mb = total_size / (1024*1024)
            print(f"✅ Klasör kopyalandı → {dst_root} | 📄 {copied} kopyalandı, ⏭️ {skipped} atlandı | 📦 {mb:.1f} MB | ⏱️ {dt:.2f}s")
            return True
        except Exception as e:
            print(f"❌ Klasör kopyalama hatası: {e}")
            return False
    
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
        """Colab için checkpoint arama - colab_learn/yolo11_models yapısına uygun"""
        
        # SmartFarm colab_learn klasör yapısında ara
        base_model_dir = "/content/drive/MyDrive/SmartFarm/colab_learn/yolo11_models"
        
        # Eğer project_folder ayarlanmışsa onu da kontrol et
        search_base_dirs = [base_model_dir]
        if self.project_folder:
            search_base_dirs.append(self.project_folder)
        
        print(f"🔍 Checkpoint arama başlıyor...")
        
        for base_dir in search_base_dirs:
            if not os.path.exists(base_dir):
                print(f"⏭️ Ana klasör mevcut değil: {base_dir}")
                continue
            
            print(f"📁 Ana klasör kontrol ediliyor: {base_dir}")
            
            # Timestamp klasörlerini bul (20250821_203234 formatında)
            try:
                timestamp_dirs = []
                for item in os.listdir(base_dir):
                    item_path = os.path.join(base_dir, item)
                    if os.path.isdir(item_path) and len(item) == 15 and '_' in item:
                        timestamp_dirs.append(item_path)
                
                if timestamp_dirs:
                    # Timestamp klasörlerini en küçükten en büyüğe sırala (20250821_203234 formatı)
                    timestamp_dirs.sort(key=lambda x: os.path.basename(x))
                    print(f"📅 Bulunan timestamp klasörleri: {[os.path.basename(d) for d in timestamp_dirs]}")
                    
                    # Tüm timestamp klasörlerinde checkpoint ara (en yeniden başlayarak)
                    for timestamp_dir in reversed(timestamp_dirs):
                        print(f"📅 Kontrol ediliyor: {os.path.basename(timestamp_dir)}")
                        result = self._search_checkpoint_in_dir(timestamp_dir)
                        if result[0]:
                            return result
                
                # Timestamp klasörü yoksa doğrudan base_dir'de ara
                result = self._search_checkpoint_in_dir(base_dir)
                if result[0]:
                    return result
                    
            except Exception as e:
                print(f"⚠️ {base_dir} arama hatası: {e}")
                continue
        
        print("❌ Hiçbir klasörde checkpoint bulunamadı!")
        return None, None
    
    def _search_checkpoint_in_dir(self, search_dir):
        """Belirli bir (timestamp) klasörde checkpoints ara - iç alt klasörleri de kapsa."""
        print(f"📁 Aranıyor: {search_dir}")

        try:
            # 1) Önce bilinen alt yolları kontrol et (açık loglarla)
            candidate_dirs = [
                search_dir,
                os.path.join(search_dir, 'models'),
                os.path.join(search_dir, 'checkpoints'),  # klasör adı: 'checkpoints'
                os.path.join(search_dir, 'checkpoints', 'weights'),
            ]
            print(f"🔎 Kontrol edilecek dizinler: {candidate_dirs}")

            def try_in_dir(d):
                print(f"📂 Kontrol ediliyor: {d}")
                if not os.path.isdir(d):
                    print(f"⚠️ Erişilemedi veya yok: {d}")
                    return None
                try:
                    files = os.listdir(d)
                except Exception:
                    print(f"⚠️ Erişilemedi veya yok: {d}")
                    return None
                pt_files = [f for f in files if f.endswith('.pt')]
                # Öncelik: last.pt, sonra best.pt
                for name in ['last.pt', 'best.pt']:
                    if name in pt_files:
                        p = os.path.join(d, name)
                        try:
                            mtime = os.path.getmtime(p)
                            size_mb = os.path.getsize(p) / (1024*1024)
                            from datetime import datetime as _dt
                            date = _dt.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                            print(f"✅ Checkpoint bulundu: {p}")
                            print(f"📊 Boyut: {size_mb:.1f} MB | 📅 Tarih: {date}")
                        except Exception:
                            pass
                        return p, name
                # Epoch dosyaları
                epoch_files = [f for f in pt_files if f.startswith('epoch_') and f.endswith('.pt')]
                if epoch_files:
                    try:
                        latest = max(epoch_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
                        p = os.path.join(d, latest)
                        size_mb = os.path.getsize(p) / (1024*1024)
                        print(f"✅ Epoch checkpoint bulundu: {p} ({size_mb:.1f} MB)")
                        return p, latest
                    except Exception:
                        return None
                return None

            # Bilinen alt yolları sırayla dene
            for d in candidate_dirs:
                res = try_in_dir(d)
                if res:
                    return res

            # 2) Fallback: rekürsif tara ve last/best/epoch_*.pt bul
            latest_epoch_path = None
            latest_epoch_num = -1
            found_last = None
            found_best = None
            for root, _, files in os.walk(search_dir):
                for f in files:
                    if not f.endswith('.pt'):
                        continue
                    full = os.path.join(root, f)
                    if f == 'last.pt' and found_last is None:
                        found_last = full
                    elif f == 'best.pt' and found_best is None:
                        found_best = full
                    elif f.startswith('epoch_'):
                        try:
                            num = int(f.split('_')[1].split('.')[0])
                            if num > latest_epoch_num:
                                latest_epoch_num = num
                                latest_epoch_path = full
                        except Exception:
                            pass

            if found_last:
                print(f"✅ Rekürsif aramada last.pt bulundu: {found_last}")
                return found_last, 'last.pt'
            if found_best:
                print(f"✅ Rekürsif aramada best.pt bulundu: {found_best}")
                return found_best, 'best.pt'
            if latest_epoch_path:
                print(f"✅ Rekürsif aramada en yüksek epoch bulundu: {latest_epoch_path}")
                return latest_epoch_path, os.path.basename(latest_epoch_path)

            # Net mesaj: 'checkpoints' klasörü veya .pt dosyaları bulunamamış olabilir
            print(f"⚠️ {search_dir} içinde 'checkpoints/' altı ya da uygun .pt (last/best/epoch_*.pt) bulunamadı")
        except Exception as e:
            print(f"⚠️ {search_dir} arama hatası: {e}")
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


def debug_colab_environment():
    """Colab ortamını detaylı debug et"""
    print("\n🔍 Colab Ortam Debug Raporu")
    print("=" * 50)
    
    # 1. Ortam tespiti
    is_colab = detect_colab_environment()
    print(f"🔍 Colab tespit edildi: {is_colab}")
    
    # 2. Modül kontrolü
    import sys
    colab_modules = [m for m in sys.modules.keys() if 'colab' in m.lower()]
    print(f"📦 Colab modülleri: {colab_modules}")
    
    # 3. Ortam değişkenleri
    colab_env_vars = {k: v for k, v in os.environ.items() if 'colab' in k.lower()}
    print(f"🌍 Colab ortam değişkenleri: {colab_env_vars}")
    
    # 4. Drive mount kontrolü
    drive_paths = ['/content/drive', '/content/drive/MyDrive']
    for path in drive_paths:
        exists = os.path.exists(path)
        print(f"📁 {path}: {'✅ Mevcut' if exists else '❌ Yok'}")
        if exists:
            try:
                items = os.listdir(path)[:5]  # İlk 5 öğe
                print(f"   📋 İçerik örneği: {items}")
            except Exception as e:
                print(f"   ❌ Listeleme hatası: {e}")
    
    # 5. Google Colab kütüphanesi kontrolü
    try:
        from google.colab import drive, files
        print("✅ google.colab kütüphanesi mevcut")
    except ImportError as e:
        print(f"❌ google.colab import hatası: {e}")
    
    return is_colab

def test_drive_operations():
    """Drive işlemlerini test et"""
    print("\n🧪 Drive İşlemleri Test Raporu")
    print("=" * 50)
    
    # Drive Manager oluştur
    dm = DriveManager()
    print(f"🔍 DriveManager oluşturuldu (is_colab: {dm.is_colab})")
    
    # Kimlik doğrulama testi
    print("\n1️⃣ Kimlik Doğrulama Testi")
    auth_success = dm.authenticate()
    print(f"   Sonuç: {'✅ Başarılı' if auth_success else '❌ Başarısız'}")
    
    if not auth_success:
        return False
    
    # Klasör kurulum testi
    print("\n2️⃣ Klasör Kurulum Testi")
    # Otomatik test klasörü oluştur
    if dm.is_colab and dm.is_mounted:
        test_folder = os.path.join(dm.base_drive_path, 'SmartFarm_Test')
        try:
            os.makedirs(test_folder, exist_ok=True)
            test_file = os.path.join(test_folder, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('Test dosyası')
            print(f"   ✅ Test klasörü oluşturuldu: {test_folder}")
            
            # Temizlik
            os.remove(test_file)
            os.rmdir(test_folder)
            print("   🧹 Test dosyaları temizlendi")
            return True
        except Exception as e:
            print(f"   ❌ Test klasörü hatası: {e}")
            return False
    
    return auth_success

def manual_drive_mount():
    """Manuel Drive mount işlemi - Kernel hatası durumunda kullanın"""
    print("\n🔧 Manuel Drive Mount İşlemi")
    print("=" * 40)
    
    try:
        from google.colab import drive
        
        # Basit mount işlemi
        print("🔄 Basit mount işlemi deneniyor...")
        drive.mount('/content/drive')
        
        # Kontrol
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Manuel mount başarılı!")
            print("📁 Drive yolu: /content/drive/MyDrive")
            
            # Yazma testi
            try:
                test_file = '/content/drive/MyDrive/test_manual_mount.txt'
                with open(test_file, 'w') as f:
                    f.write('Manuel mount test')
                os.remove(test_file)
                print("✅ Yazma izni doğrulandı")
                return True
            except Exception as e:
                print(f"❌ Yazma izni hatası: {e}")
                return False
        else:
            print("❌ Manuel mount başarısız")
            return False
            
    except Exception as e:
        print(f"❌ Manuel mount hatası: {e}")
        print("\n💡 Alternatif çözümler:")
        print("1. Colab'de Files panelinden 'Mount Drive' butonuna tıklayın")
        print("2. Runtime > Restart runtime yapıp tekrar deneyin")
        print("3. Yeni bir Colab notebook açıp kodu oraya kopyalayın")
        return False

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

        # Colab için ABSOLUTE base path desteği:
        # Eğer folder_path "/content/drive/" ile başlıyorsa, bu tam yolu base_drive_path olarak kabul et
        # ve folder_path'i boş bırak. Böylece timestamp klasörü doğrudan bu klasörün altında oluşturulur.
        if dm.is_colab and folder_path:
            try:
                norm_fp = os.path.normpath(folder_path)
                if norm_fp.startswith('/content/drive/'):
                    dm.base_drive_path = norm_fp
                    folder_path = ''
                    print(f"ℹ️ Base Drive yolu absolute olarak ayarlandı: {dm.base_drive_path}")
            except Exception:
                pass

        # Var olan (veya yoksa oluşturulacak) ana klasörü seç
        ok = dm.select_existing_folder(folder_path, project_name)
        if not ok:
            print(f"❌ Proje klasörü ayarlanamadı: {folder_path}")
            return None

        # Colab modunda: timestamp alt klasörünü YENİDEN KULLAN veya yoksa oluştur
        if dm.is_colab:
            try:
                # Base path, absolute destekli olabilir
                base_path = os.path.join(dm.base_drive_path, folder_path) if folder_path else dm.base_drive_path

                # 0) Oturum kilidi (session lock): aynı runtime içinde hep aynı timestamp
                try:
                    global _GLOBAL_SESSION_TS
                    # Dosya-tabanlı oturum bilgisi
                    session_file = os.path.join(base_path, '.active_session.json')
                    session_ts = None
                    if _GLOBAL_SESSION_TS and os.path.isdir(_GLOBAL_SESSION_TS):
                        session_ts = _GLOBAL_SESSION_TS
                    elif os.path.exists(session_file):
                        with open(session_file, 'r', encoding='utf-8') as sf:
                            data = json.load(sf)
                            cand = data.get('ts_dir')
                            if cand and os.path.isdir(cand):
                                session_ts = cand
                    if session_ts and os.path.normpath(session_ts).startswith(os.path.normpath(base_path)):
                        dm.project_folder = session_ts
                        dm.active_timestamp_dir = session_ts
                        print(f"🔒 Oturum timestamp kilidi kullanılıyor: {os.path.basename(session_ts)}")
                        # Alt klasörleri garanti et
                        for sub in ['models', 'checkpoints', 'logs', 'configs']:
                            os.makedirs(os.path.join(dm.project_folder, sub), exist_ok=True)
                        # checkpoints/weights
                        os.makedirs(os.path.join(dm.project_folder, 'checkpoints', 'weights'), exist_ok=True)
                        ts_name = os.path.basename(dm.project_folder.rstrip('/'))
                        dm._save_drive_config(folder_path or os.path.relpath(dm.project_folder, dm.base_drive_path).rsplit('/', 1)[0], ts_name)
                        print(f"✅ Timestamp ve alt klasörler hazır (session lock): {dm.project_folder}")
                        print("✅ Drive entegrasyonu hazır (etkileşimsiz mod)")
                        return dm
                except Exception as sl_e:
                    print(f"⚠️ Session lock okunamadı: {sl_e}")

                # 1) Eğer config'te bir timestamp kayıtlı ve geçerliyse, HER ZAMAN onu kullan
                reused = False
                try:
                    if dm.load_drive_config():
                        ts_existing = dm.get_timestamp_dir()
                        if ts_existing and os.path.isdir(ts_existing) and os.path.dirname(ts_existing).startswith(base_path):
                            dm.project_folder = ts_existing
                            print(f"🗂️ Config'teki timestamp yeniden kullanılıyor: {os.path.basename(ts_existing)}")
                            reused = True
                except Exception:
                    pass

                # 2) Config yoksa veya geçersizse, mevcut timestamp adaylarını tara ve ILK OLUŞANINI seç
                if not reused:
                    try:
                        candidates = []
                        if os.path.isdir(base_path):
                            print(f"🔎 Timestamp taraması: base_path = {base_path}")
                            candidates = [
                                os.path.join(base_path, d)
                                for d in os.listdir(base_path)
                                if len(d) == 15 and '_' in d and d.replace('_', '').isdigit() and os.path.isdir(os.path.join(base_path, d))
                            ]
                        if candidates:
                            print(f"🔎 Bulunan timestamp adayları: {[os.path.basename(c) for c in candidates]}")
                            candidates.sort(key=lambda p: os.path.getmtime(p))
                            oldest = candidates[0]
                            print(f"🕒 En eski timestamp: {os.path.basename(oldest)}")
                            dm.project_folder = oldest
                            reused = True
                    except Exception:
                        pass

                # 3) Hiçbiri yoksa yeni timestamp oluştur
                if not dm.project_folder:
                    print("ℹ️ Config geçersiz veya aday bulunamadı; yeni timestamp oluşturulacak.")
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    proj_dir = os.path.join(base_path, ts)
                    os.makedirs(proj_dir, exist_ok=True)
                    dm.project_folder = proj_dir
                    print(f"✅ Yeni timestamp klasörü oluşturuldu: {proj_dir}")
                else:
                    print(f"✅ Mevcut timestamp klasörü kullanılıyor: {dm.project_folder}")

                # Alt klasörleri garanti et
                for sub in ['models', 'checkpoints', 'logs', 'configs']:
                    os.makedirs(os.path.join(dm.project_folder, sub), exist_ok=True)
                # checkpoints/weights alt klasörünü de garanti et
                try:
                    weights_dir = os.path.join(dm.project_folder, 'checkpoints', 'weights')
                    os.makedirs(weights_dir, exist_ok=True)
                    print(f"📁 'checkpoints/weights' hazır: {weights_dir}")
                except Exception:
                    pass

                # Global işaret ve konfigürasyon kaydı
                dm.active_timestamp_dir = dm.project_folder
                ts_name = os.path.basename(dm.project_folder.rstrip('/'))
                dm.project_name = project_name or os.path.basename(folder_path) if folder_path else dm.project_name
                # Seçilen timestamp'i config'e kaydet
                dm._save_drive_config(folder_path or os.path.relpath(dm.project_folder, dm.base_drive_path).rsplit('/', 1)[0], ts_name)
                print(f"✅ Timestamp ve alt klasörler hazır: {dm.project_folder}")

                # 4) Oturum kilidini yaz (global ve dosya)
                try:
                    global _GLOBAL_SESSION_TS
                    _GLOBAL_SESSION_TS = dm.project_folder
                    session_file = os.path.join(base_path, '.active_session.json')
                    with open(session_file, 'w', encoding='utf-8') as sf:
                        json.dump({'ts_dir': dm.project_folder, 'started_at': datetime.now().isoformat()}, sf, ensure_ascii=False, indent=2)
                    print(f"🔒 Oturum timestamp kilidi yazıldı: {session_file}")
                except Exception as slw_e:
                    print(f"⚠️ Session lock yazılamadı: {slw_e}")

                # Otomatik log yönlendirme (tee) — sadece bir kez etkinleştir
                try:
                    if not dm._tee_enabled:
                        logs_dir = os.path.join(dm.project_folder, 'logs')
                        os.makedirs(logs_dir, exist_ok=True)
                        log_file = os.path.join(logs_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                        # Python logging de dosyaya yazsın
                        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
                        fh = logging.FileHandler(log_file, encoding='utf-8')
                        fh.setLevel(logging.INFO)
                        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
                        root_logger = logging.getLogger()
                        # Aynı dosya handler'ı iki kez eklememek için kontrol
                        if not any(getattr(h, 'baseFilename', None) == fh.baseFilename for h in root_logger.handlers):
                            root_logger.addHandler(fh)
                        # Stdout'u da tee ile aynı dosyaya kopyala
                        dm._tee_file_path = log_file
                        dm._tee_enabled = True
                        dm._tee_file_handle = open(log_file, 'a', encoding='utf-8')
                        dm._tee_stdout_prev = sys.stdout
                        sys.stdout = _TeeStdout(sys.stdout, dm._tee_file_handle)
                        print(f"📝 Otomatik loglama etkin: {log_file}")
                except Exception as tee_e:
                    print(f"⚠️ Log tee kurulamadı: {tee_e}")
            except Exception as e:
                print(f"⚠️ Timestamp klasörü hazırlığı başarısız: {e}")

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