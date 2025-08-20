# 🚀 SmartFarm Google Drive Entegrasyonu Kurulum Kılavuzu

Bu kılavuz, SmartFarm projesinde Google Drive ile otomatik model kaydetme ve eğitimin kaldığı yerden devam etme özelliğinin nasıl kurulacağını açıklar.

## 📋 Özellikler

### ✅ Eklenen Özellikler
- **Epoch Bazlı Kaydetme**: Belirtilen epoch aralıklarında modeli otomatik Drive'a kaydetme
- **Zaman Damgalı Klasörler**: Her eğitim için Drive'da zaman damgalı klasör oluşturma
- **Devam Etme**: Eğitimin yarıda kalması durumunda Drive'dan kaldığı yerden devam etme
- **Otomatik Klasör Yönetimi**: Drive'da "Tarım/SmartFarm" klasör yapısını otomatik oluşturma
- **Model Listeleme**: Drive'daki tüm kaydedilmiş modelleri görüntüleme

### 🎯 Kullanım Senaryoları
- **Uzun Eğitimler**: Günlerce süren eğitimlerde güvenlik için periyodik kaydetme
- **Colab Kesintileri**: Google Colab session'ının kapanması durumunda devam etme
- **Model Yedekleme**: Eğitim sırasında otomatik yedekleme
- **Takım Çalışması**: Drive üzerinden model paylaşımı

## 🔧 Kurulum Adımları

### 1. Gerekli Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

### 2. Google Cloud Console Ayarları

#### 2.1 Proje Oluştur
1. [Google Cloud Console](https://console.cloud.google.com/)'a git
2. Yeni proje oluştur veya mevcut projeyi seç
3. Proje adını not et (örn: "smartfarm-ai")

#### 2.2 Google Drive API'yi Etkinleştir
1. Sol menüden **APIs & Services** → **Library**
2. "Google Drive API" ara
3. **Enable** butonuna tıkla

#### 2.3 OAuth 2.0 Credentials Oluştur
1. **APIs & Services** → **Credentials**
2. **+ CREATE CREDENTIALS** → **OAuth client ID**
3. Application type: **Desktop application**
4. Name: "SmartFarm Drive Integration"
5. **CREATE** butonuna tıkla
6. **DOWNLOAD JSON** ile credentials dosyasını indir

#### 2.4 Credentials Dosyasını Yerleştir
```bash
# İndirilen dosyayı SmartFarm klasörüne kopyala ve yeniden adlandır
cp ~/Downloads/client_secret_*.json ./credentials.json
```

### 3. İlk Çalıştırma ve Kimlik Doğrulama

```python
python main_multi_dataset.py
```

Program çalıştığında:
1. **Google Drive entegrasyonu** sorusu gelecek → `y` yazın
2. Tarayıcı açılacak → Google hesabınızla giriş yapın
3. İzinleri onaylayın
4. Drive klasör yolu sorusu → `Tarım/SmartFarm` (varsayılan)
5. Proje adı → `SmartFarm_Training` (varsayılan)

## 📁 Drive Klasör Yapısı

Sistem otomatik olarak şu yapıyı oluşturur:

```
📁 Google Drive/
  📁 Tarım/
    📁 SmartFarm/
      📁 20250120_143022_SmartFarm_Training/
        📄 best_model_epoch_10_20250120_143500.pt
        📄 checkpoint_epoch_10_20250120_143500.pt
        📄 best_model_epoch_20_20250120_144500.pt
        📄 checkpoint_epoch_20_20250120_144500.pt
        📄 ...
```

## 🎮 Kullanım

### Yeni Eğitim Başlatma

```python
python main_multi_dataset.py
```

**Eğitim sırasında sorulacak sorular:**

1. **Google Drive kullanımı**: `y` (evet)
2. **Kaydetme aralığı**: `10` (10 epoch'ta bir kaydet)
3. **RAM temizleme**: `10` (10 epoch'ta bir temizle)

### Eğitimi Devam Ettirme

Eğitim yarıda kaldıysa:

```python
python main_multi_dataset.py
```

1. Resume seçeneğini seç
2. **Google Drive kullanımı**: `y`
3. **Devam etme kaynağı**: `2` (Google Drive'dan)

Sistem otomatik olarak:
- En son checkpoint'i bulur
- Drive'dan indirir
- Kaldığı epoch'tan devam eder

### Manuel Drive İşlemleri

```python
from drive_manager import setup_drive_integration

# Drive manager oluştur
drive_manager = setup_drive_integration()

# Drive'daki modelleri listele
models = drive_manager.list_drive_models()

# Belirli bir modeli indir
drive_manager.download_checkpoint("file_id", "local_path.pt")
```

## ⚙️ Konfigürasyon Dosyaları

### drive_config.json
Otomatik oluşturulan Drive konfigürasyonu:
```json
{
  "folder_path": "Tarım/SmartFarm",
  "project_folder_name": "20250120_143022_SmartFarm_Training",
  "drive_folder_id": "1ABC123...",
  "project_name": "SmartFarm_Training",
  "created_at": "2025-01-20T14:30:22"
}
```

### drive_uploads.json
Yükleme geçmişi:
```json
[
  {
    "filename": "checkpoint_epoch_10_20250120_143500.pt",
    "epoch": 10,
    "file_id": "1XYZ789...",
    "is_best": false,
    "uploaded_at": "2025-01-20T14:35:00"
  }
]
```

## 🔍 Sorun Giderme

### Kimlik Doğrulama Hataları

**Hata**: `credentials.json not found`
```bash
# Credentials dosyasının doğru yerde olduğunu kontrol et
ls -la credentials.json
```

**Hata**: `Access denied`
- Google Cloud Console'da Drive API'nin etkinleştirildiğini kontrol et
- OAuth consent screen'i yapılandır

### Yükleme Hataları

**Hata**: `Upload failed`
```python
# Drive bağlantısını test et
from drive_manager import DriveManager
dm = DriveManager()
dm.authenticate()
```

**Hata**: `Quota exceeded`
- Google Drive kotanızı kontrol edin
- Eski modelleri temizleyin

### Dosya Bulunamadı Hataları

**Hata**: `Checkpoint not found`
```python
# Drive'daki dosyaları listele
drive_manager.list_drive_models()

# Upload geçmişini kontrol et
cat drive_uploads.json
```

## 📊 Performans İpuçları

### Kaydetme Sıklığı
- **Hızlı eğitim** (< 2 saat): 20-50 epoch arası
- **Orta eğitim** (2-8 saat): 10-20 epoch arası  
- **Uzun eğitim** (> 8 saat): 5-10 epoch arası

### Ağ Optimizasyonu
- Stabil internet bağlantısı kullanın
- Büyük modeller için gece saatlerini tercih edin
- Paralel yükleme yapmayın (tek seferde bir model)

## 🔒 Güvenlik

### Credentials Güvenliği
```bash
# Credentials dosyasını git'e eklemeyin
echo "credentials.json" >> .gitignore
echo "token.pickle" >> .gitignore
echo "drive_config.json" >> .gitignore
echo "drive_uploads.json" >> .gitignore
```

### İzin Yönetimi
- Sadece gerekli Drive izinlerini verin
- Düzenli olarak erişim loglarını kontrol edin
- Kullanılmayan credentials'ları silin

## 📞 Destek

### Hata Raporlama
Hata durumunda şu bilgileri toplayın:
1. Hata mesajı ve stack trace
2. `drive_config.json` içeriği
3. Python ve kütüphane versiyonları
4. İşletim sistemi bilgisi

### Yararlı Komutlar
```bash
# Kütüphane versiyonlarını kontrol et
pip list | grep google

# Drive API kotasını kontrol et
# Google Cloud Console → APIs & Services → Quotas

# Log dosyalarını temizle
rm -f drive_uploads.json token.pickle
```

## 🎯 Sonuç

Bu entegrasyon ile artık:
- ✅ Eğitimleriniz güvenli şekilde Drive'a kaydediliyor
- ✅ Kesintiler durumunda kaldığınız yerden devam edebiliyorsunuz
- ✅ Model geçmişinizi takip edebiliyorsunuz
- ✅ Takım arkadaşlarınızla model paylaşabiliyorsunuz

**İyi eğitimler! 🌱🤖**
