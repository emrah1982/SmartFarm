#!/usr/bin/env python3
# test_drive_integration.py - Google Drive entegrasyon testi

import sys
import os
import pickle
from pathlib import Path

# Gerekli kütüphanelerin yüklü olup olmadığını kontrol et
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    import google.auth.exceptions
    print("✅ Gerekli Google kütüphaneleri yüklü")
except ImportError as e:
    print(f"❌ Hata: {e}")
    print("Gerekli kütüphaneler yüklenmemiş. Lütfen aşağıdaki komutu çalıştırın:")
    print("pip install --upgrade google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    sys.exit(1)

def check_credentials():
    """Kimlik bilgilerini kontrol et"""
    credentials_path = Path("credentials.json")
    token_path = Path("token.pickle")
    
    print("\n🔍 Kimlik Bilgileri Kontrolü")
    print("=" * 50)
    
    # credentials.json dosyasını kontrol et
    if not credentials_path.exists():
        print("❌ 'credentials.json' dosyası bulunamadı!")
        print("\n📋 Kurulum Adımları:")
        print("1. Google Cloud Console'a gidin (https://console.cloud.google.com/)")
        print("2. Yeni proje oluşturun veya mevcut projeyi seçin")
        print("3. 'APIs & Services' > 'Library' bölümünden 'Google Drive API'yi etkinleştirin")
        print("4. 'APIs & Services' > 'Credentials' bölümüne gidin")
        print("5. 'Create Credentials' > 'OAuth client ID' seçin")
        print("6. Application type olarak 'Desktop app' seçin")
        print("7. İndirilen credentials.json dosyasını bu klasöre kopyalayın")
        return False
    
    print(f"✅ 'credentials.json' dosyası mevcut: {credentials_path.absolute()}")
    
    # Token kontrolü
    if token_path.exists():
        print(f"✅ 'token.pickle' dosyası mevcut: {token_path.absolute()}")
    else:
        print("ℹ️ 'token.pickle' dosyası bulunamadı. İlk giriş yapıldığında oluşturulacak.")
    
    return True

def test_authentication():
    """Google Drive kimlik doğrulamasını test et"""
    print("\n🔑 Kimlik Doğrulama Testi")
    print("=" * 50)
    
    creds = None
    token_path = Path("token.pickle")
    
    # Token dosyası varsa yükle
    if token_path.exists():
        try:
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
            print("✅ Token dosyası yüklendi")
        except Exception as e:
            print(f"❌ Token dosyası yüklenirken hata: {e}")
            return False
    
    # Geçerli kimlik bilgileri yoksa veya süresi dolmuşsa
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("✅ Token başarıyla yenilendi")
            except Exception as e:
                print(f"❌ Token yenilenirken hata: {e}")
                return False
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', 
                    ['https://www.googleapis.com/auth/drive.file']
                )
                creds = flow.run_local_server(port=0)
                print("✅ Başarıyla giriş yapıldı")
                
                # Token'ı kaydet
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
                print("✅ Token dosyası kaydedildi")
            except Exception as e:
                print(f"❌ Giriş hatası: {e}")
                return False
    
    try:
        # Drive servisini başlat
        service = build('drive', 'v3', credentials=creds)
        
        # Kullanıcı bilgilerini al
        about = service.about().get(fields="user").execute()
        user_email = about.get('user', {}).get('emailAddress', 'Bilinmiyor')
        print(f"\n👤 Giriş yapılan hesap: {user_email}")
        
        return True
    except Exception as e:
        print(f"❌ Drive servisine bağlanılamadı: {e}")
        return False

if __name__ == "__main__":
    print("\n🚀 Google Drive Entegrasyon Testi")
    print("=" * 50)
    
    # Kimlik bilgilerini kontrol et
    if not check_credentials():
        sys.exit(1)
    
    # Kimlik doğrulama testi
    if not test_authentication():
        print("\n❌ Google Drive entegrasyonu başarısız oldu!")
        sys.exit(1)
    
    print("\n✅ Google Drive entegrasyonu başarılı!")
    print("Artık SmartFarm uygulamasında Google Drive'ı kullanabilirsiniz.")
