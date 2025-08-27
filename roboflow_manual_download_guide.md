# Roboflow Dataset Manuel İndirme Rehberi

## Sorun
Roboflow Universe DS linkleri (`/ds/<hash>?key=...`) Cloudflare bot koruması kullanıyor ve programatik erişimi engelliyor.

## Çözüm Yöntemleri

### 1. Gerçek API Key Kullanımı (Önerilen)
```bash
# Roboflow hesabınızdan API key alın:
# 1. https://roboflow.com → Login
# 2. Settings → API Keys → Copy
# 3. main_multi_dataset.py çalıştırırken API key girin
```

### 2. Manuel İndirme
```bash
# Her dataset için:
# 1. Tarayıcıda DS linkini açın
# 2. ZIP dosyasını indirin
# 3. datasets/ klasörüne çıkarın
# 4. Klasör adını config_datasets.yaml'deki local_path ile eşleştirin
```

### 3. Alternatif Dataset Linkleri
```yaml
# config_datasets.yaml içinde DS linklerini şu formatlarla değiştirin:

# Eski (çalışmıyor):
url: "https://universe.roboflow.com/ds/nKPr1UgofJ?key=a2sSLftQC8"

# Yeni (API endpoint):
url: "https://api.roboflow.com/dataset/workspace/project/version?api_key=YOUR_KEY&format=yolov8"

# Veya download link:
url: "https://universe.roboflow.com/workspace/project/download/version/yolov8?key=YOUR_KEY"
```

### 4. Dataset Klasör Yapısı
```
datasets/
├── plant_diseases_comprehensive/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
└── ...
```

## Test Komutu
```bash
python debug_single_link.py
```

## Eğitimi Başlatma
```bash
# Dataset'ler hazır olduktan sonra:
python main_multi_dataset.py
# → Hiyerarşik kurulum seç
# → API key gir (varsa)
# → Dataset grubunu seç
```
