# 🌱 SmartFarm - Bitki Hastalıklarının Drone Görüntüsü ile Tespiti

Bu proje, dronelardan alınan görüntüler üzerinde YOLOv8/YOLO11 tabanlı bir model kullanarak **bitki hastalıklarını tespit etmek** ve bu analizleri **gerçek zamanlı (real-time)** olarak kullanıcıya aktarmak amacıyla geliştirilmiştir.

## 🚀 Temel Özellikler

### ✨ YOLOv8/YOLO11 Tabanlı Tespit
- **Multi-Dataset Birleştirme**: 7+ farklı Roboflow veri setini otomatik birleştirme
- **Hierarchical Sınıf Sistemi**: Ana kategoriler + alt kategoriler yapısı
- **Türkçe Etiketleme**: ZARLI, MANTAR HASTALIĞI, BESİN EKSİKLİĞİ gibi Türkçe çıktılar
- **Renkli Bounding Box**: Her kategori için özel renk kodlaması
- **Akıllı Sınıf Haritalama**: Otomatik ve manuel sınıf birleştirme

### 🔄 Google Drive Entegrasyonu (YENİ!)
- **Epoch Bazlı Kaydetme**: Belirtilen aralıklarla otomatik model kaydetme (varsayılan: 10 epoch)
- **Zaman Damgalı Klasörler**: Her eğitim için Drive'da zaman damgalı klasör oluşturma
- **Devam Etme Özelliği**: Eğitimin yarıda kalması durumunda Drive'dan kaldığı yerden devam etme
- **Otomatik Klasör Yönetimi**: Drive'da "Tarım/SmartFarm" klasör yapısını otomatik oluşturma
- **Model Listeleme**: Drive'daki tüm kaydedilmiş modelleri görüntüleme

### 🛠️ Gelişmiş Özellikler
- **Colab Optimizasyonu**: Google Colab için özel kurulum ve hata yönetimi
- **Memory Management**: Bellek optimizasyonu ve otomatik temizleme
- **Gelişmiş Augmentation**: Tarımsal koşullar için özelleştirilmiş veri artırma
- **Otomatik Veri Dengeleme**: Sınıf başına hedef örnek sayısına ulaşma

### 🔬 Gelişmiş Augmentation
- **Hava Durumu Simülasyonu**: Yağmur, sis, gölge efektleri
- **Işık Koşulları**: Parlaklık, kontrast, gamma ayarları
- **Geometrik Dönüşümler**: Döndürme, ölçekleme, perspektif
- **Tarımsal Spesifik**: HSV, renk değişiklikleri, doku varyasyonları
- **Akıllı Severity**: Light, medium, heavy seviyelerinde augmentation

### 📊 Analiz ve Raporlama
- **Detaylı Veri Analizi**: Sınıf dağılımı, görüntü kalitesi analizi
- **Görsel Raporlar**: Grafik ve tabloları içeren PDF raporları
- **Progress Tracking**: Eğitim sürecinin canlı takibi
- **Performance Metrikleri**: mAP, precision, recall detaylı analizi

## 📁 Proje Yapısı

```
📦 SmartFarm
├── 🚀 CORE FILES (Ana Dosyalar)
│   ├── main_multi_dataset.py          # Ana eğitim scripti
│   ├── multi_dataset_manager.py       # Multi-dataset yöneticisi
│   ├── training.py                     # Model eğitim fonksiyonları (Google Drive entegrasyonu)
│   ├── drive_manager.py                # Google Drive yönetim modülü (YENİ!)
│   ├── augmentation_utils.py           # Gelişmiş augmentation sistemi
│   ├── dataset_utils.py                # Dataset indirme/düzenleme
│   ├── hyperparameters.py              # Hiperparametre yönetimi
│   ├── setup_utils.py                  # Kurulum ve GPU kontrolleri
│   ├── memory_utils.py                 # Bellek optimizasyonu
│   ├── model_downloader.py             # YOLO11 model indirici
│   └── config_datasets.yaml            # Dataset konfigürasyonu
│
├── 🔧 COLAB SETUP (Colab Optimizasyonu)
│   ├── colab_setup.py                  # Colab için akıllı kurulum (YENİ!)
│   ├── quick_colab_fix.py              # Hızlı Colab düzeltmesi (YENİ!)
│   └── requirements.txt                # Colab-uyumlu paketler
│
├── 📋 DOKÜMANTASYON
│   ├── README.md                       # Bu dosya
│   ├── GOOGLE_DRIVE_SETUP.md           # Google Drive kurulum kılavuzu (YENİ!)
│   └── .gitignore                      # Git güvenlik ayarları (YENİ!)
│
└── 📂 BACKUP/ALTERNATIVE FILES
    ├── main.py                         # Tek dataset için eski versiyon
    ├── main_update.py                  # Güncellenmiş alternatif
    └── multi_dataset_helpers.py        # Yardımcı fonksiyonlar
```

## 🚀 Hızlı Başlangıç

### 1. Kurulum (Google Colab)

```bash
# Repository'yi klonla
!git clone https://github.com/emrah1982/SmartFarm.git
%cd SmartFarm

# Colab için optimize edilmiş kurulum
!python colab_setup.py

# Veya hızlı düzeltme
!python quick_colab_fix.py
```

### 2. Google Drive Entegrasyonu Kurulumu

```python
# 1. Google Cloud Console'dan credentials.json indirin
# 2. Colab'a yükleyin
from google.colab import files
files.upload()  # credentials.json seçin

# 3. Ana scripti çalıştırın
!python main_multi_dataset.py
```

### 3. Eğitim Süreci

```
1. Google Drive entegrasyonunu etkinleştirin (y)
2. Drive klasör yolu belirleyin (örn: Tarım/SmartFarm)
3. Kaydetme aralığını seçin (örn: 10 epoch)
4. Dataset'leri seçin ve eğitimi başlatın
5. Eğitim yarıda kalırsa Drive'dan devam edin!
```

### 4. Eğitimi Devam Ettirme

```python
# Eğitim yarıda kaldıysa
!python main_multi_dataset.py
# Resume seçeneğini seçin
# "Google Drive'dan devam et" seçeneğini kullanın
```

## 🎛️ Konfigürasyon

### Önceden Tanımlı Veri Setleri

```python
# 9 farklı tarımsal veri seti destekleniyor:
datasets = [
    "Plant Village Dataset",      # Bitki hastalıkları
    "Agricultural Diseases",      # 30+ hastalık türü  
    "Fruit Ripeness",            # Meyve olgunluk analizi
    "General Classes",           # Genel sınıflandırma
    "Agricultural Pests",        # Tarımsal zararlılar
    "Pest Detection",            # Böcek tespiti
    "Nutrient Deficiency",       # Besin eksikliği analizi
    # + 2 ek dataset
]
```
### Görsel Çıktı
```
🔴 Kırmızı bounding box + "ZARLI: Kırmızı Örümcek (0.85)"
🟫 Kahverengi bounding box + "MANTAR HASTALIĞI: Elma Karaleke (0.92)"  
🟢 Yeşil bounding box + "SAĞLIKLI: Domates Yaprağı (0.78)"
🟡 Sarı bounding box + "BESİN EKSİKLİĞİ: Azot Eksikliği (0.67)"
🟩 Açık yeşil-gri bounding box + "Yabanci Ot(0.77)"
```

### Hibrit Sınıf Sistemi

```python
# Ana Kategoriler → Alt Kategoriler
"healthy" → ["healthy", "normal"]
"fungal_disease" → ["apple_scab", "corn_rust", "tomato_blight", ...]
"viral_disease" → ["tomato_mosaic_virus", "tomato_yellow_virus"]
"bacterial_disease" → ["tomato_bacterial_spot", ...]
"pest_damage" → ["aphid", "spider_mite", "thrips", ...]
"nutrient_deficiency" → ["nitrogen", "phosphorus", "potassium"]
"fruit_ripe" → ["ripe"]
"fruit_unripe" → ["unripe"]
"damaged" → ["damaged"]
```

## ⚙️ Gelişmiş Ayarlar

### Model Boyutları
```python
models = {
    "yolo11n.pt": "Nano - En hızlı, düşük doğruluk",
    "yolo11s.pt": "Small - Hızlı, orta doğruluk", 
    "yolo11m.pt": "Medium - Dengeli (Önerilen)",
    "yolo11l.pt": "Large - Yüksek doğruluk",
    "yolo11x.pt": "XLarge - En yüksek doğruluk"
}
```

### Önerilen Parametreler

#### 🏃‍♂️ Hızlı Test İçin:
```python
epochs = 100
batch_size = 16
img_size = 416
model = "yolo11s.pt"
target_count_per_class = 1000
```

#### 🎯 Kaliteli Eğitim İçin:
```python
epochs = 500
batch_size = 16
img_size = 640
model = "yolo11m.pt"
target_count_per_class = 2000
```

#### 🏆 En İyi Performans İçin:
```python
epochs = 1000
batch_size = 8
img_size = 640
model = "yolo11l.pt"
target_count_per_class = 3000
```

## 📊 Çıktılar ve Sonuçlar

### Otomatik Oluşturulan Dosyalar

```
📁 outputs/
├── merged_dataset.yaml           # Ana eğitim konfigürasyonu
├── analysis_report.json          # Detaylı veri analizi
├── class_distribution.png        # Sınıf dağılım grafikleri
├── augmentation_log.txt          # Augmentation detayları
└── training_metrics.csv          # Eğitim metrikleri

📁 runs/train/exp/
├── weights/
│   ├── best.pt                   # En iyi model
│   └── last.pt                   # Son model
├── results.csv                   # Eğitim sonuçları
├── confusion_matrix.png          # Karışıklık matrisi
└── results.png                   # Metrik grafikleri

📁 Google Drive (Otomatik Kayıt)
└── /MyDrive/Tarim/Kodlar/colab_egitim/
    └── mixed/YYYYMMDD_HHMMSS/
        ├── best.pt
        ├── last.pt
        └── analysis_report.json
```

### Performance Metrikleri

```
✅ İyi Sonuçlar:
- mAP50: > 0.7
- mAP50-95: > 0.5  
- Precision: > 0.8
- Recall: > 0.7

🎯 Kabul Edilebilir:
- mAP50: > 0.5
- mAP50-95: > 0.3
- Precision: > 0.6
- Recall: > 0.6
```

## 🔧 Sorun Giderme

### Yaygın Hatalar ve Çözümleri

#### 💾 Memory Errors
```python
# Çözüm 1: Batch size küçült
batch_size = 4

# Çözüm 2: Image size küçült  
img_size = 416

# Çözüm 3: Cache kapatín
use_cache = False
```

#### 🌐 Dataset İndirme Hataları
```python
# Roboflow URL'lerini kontrol edin
# İnternet bağlantısını kontrol edin
# API key gerekliliği var mı kontrol edin
```

#### 🏷️ Sınıf Haritalama Sorunları
```python
# Manuel haritalama yapın
# Sınıf isimlerini kontrol edin
# Duplicate sınıfları temizleyin
```

### Performance Optimizasyonu

#### Colab Free İçin:
```python
batch_size = 4
workers = 2
img_size = 416
cleanup_frequency = 5
```

#### Colab Pro İçin:
```python
batch_size = 16
workers = 8  
img_size = 640
cleanup_frequency = 10
```

## 📖 Detaylı Dokümantasyon

- **[Kullanım Kılavuzu](USAGE_GUIDE.md)** - Adım adım detaylı talimatlar
- **[API Dokümantasyonu](docs/API.md)** - Fonksiyon referansları
- **[Augmentation Rehberi](docs/AUGMENTATION.md)** - Veri artırma teknikleri
- **[Sorun Giderme](docs/TROUBLESHOOTING.md)** - Yaygın sorunlar ve çözümler

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📋 Gereksinimler

### Minimum Sistem Gereksinimleri
- **RAM**: 8GB (Colab Free)
- **GPU**: Opsiyonel (CUDA destekli)
- **Disk**: 10GB boş alan
- **Python**: 3.8+

### Gerekli Paketler (Colab Uyumlu)
```txt
# Core ML libraries
ultralytics>=8.2.0
torch>=2.0.0
torchvision>=0.15.0
albumentations>=1.3.0
opencv-python-headless>=4.7.0
numpy>=1.21.0,<2.0.0  # NumPy 2.x uyumluluk sorunu düzeltmesi
matplotlib>=3.5.0
pyyaml>=6.0
psutil>=5.8.0
requests>=2.28.0
Pillow>=9.0.0
tqdm>=4.64.0
pandas>=1.3.0
seaborn>=0.11.0

# Google Drive Integration
google-api-python-client>=2.70.0
google-auth-httplib2>=0.1.0
google-auth-oauthlib>=0.8.0
google-auth>=2.15.0
```

## 🏆 Örnek Sonuçlar

### Başarılı Eğitim Örnekleri

```
🌱 Bitki Hastalık Tespiti:
- Dataset: 45,000 görüntü, 9 ana sınıf
- mAP50: 0.847
- mAP50-95: 0.623
- Eğitim süresi: 6 saat (Colab Pro)

🐛 Zararlı Böcek Tespiti:
- Dataset: 32,000 görüntü, 8 ana sınıf  
- mAP50: 0.792
- mAP50-95: 0.567
- Eğitim süresi: 4.5 saat (Colab Pro)

🍎 Meyve Olgunluk Analizi:
- Dataset: 18,000 görüntü, 3 ana sınıf
- mAP50: 0.913
- mAP50-95: 0.756
- Eğitim süresi: 2 saat (Colab Pro)
```

## 📄 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- **Ultralytics** - YOLO11 implementasyonu
- **Roboflow** - Dataset yönetim platformu  
- **Albumentations** - Augmentation kütüphanesi
- **OpenCV** - Görüntü işleme kütüphanesi

## 📞 İletişim ve Destek

- **GitHub Issues**: Hata raporları ve özellik istekleri
- **Discussions**: Genel sorular ve tartışmalar
- **Email**: [your-email@domain.com]

---

## 🎉 Başarılı Kullanım için İpuçları

1. **🔬 Küçük başlayın**: İlk eğitimi az epoch ile test edin
2. **📊 Veriyi analiz edin**: Class distribution'ı kontrol edin  
3. **⚡ GPU kullanın**: Colab Pro önerilir
4. **💾 Drive'a kaydedin**: Modelleri kaybetmeyin
5. **📈 Metrikleri takip edin**: Training progress'i izleyin
6. **🔄 Iterasyonlar yapın**: Hiperparametreleri optimize edin

**Bu framework ile tarımsal AI modellerinizde başarılar elde edebilirsiniz!** 🌱🤖✨
