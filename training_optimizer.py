#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SmartFarm Training Optimizer - Colab & Early Stopping Integration

YOLOv8/YOLO11 eğitimi için Google Colab optimize edilmiş training sistemi.
Early stopping, epoch tahmini ve optimal parametre önerileri içerir.

Özellikler:
- Colab kaynak optimizasyonu
- Otomatik early stopping
- Epoch süresi tahmini
- Optimal epoch sayısı hesaplama
- Memory management
- Progress tracking

Kullanım:
    from training_optimizer import SmartTrainingOptimizer
    
    optimizer = SmartTrainingOptimizer()
    config = optimizer.get_optimal_training_config(dataset_size=5000)
"""

import os
import psutil
import torch
import numpy as np
from datetime import datetime
import yaml
from pathlib import Path
from config_utils import get_default_batch_size, get_training_config_from_yaml
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

# SmartFarm modülleri
try:
    from early_stopping_system import EarlyStoppingManager, EarlyStoppingConfig, TrainingMetrics
    from colab_optimized_validator import ColabAugmentationValidator, get_colab_resources
    SMARTFARM_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SmartFarm modülleri import edilemedi: {e}")
    SMARTFARM_MODULES_AVAILABLE = False

# Colab tespiti
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# GPU kontrolü
def check_gpu_availability():
    """GPU durumunu kontrol et"""
    gpu_info = {
        'available': False,
        'name': 'CPU',
        'memory_gb': 0,
        'compute_capability': None
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['name'] = torch.cuda.get_device_name(0)
            gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info['compute_capability'] = torch.cuda.get_device_capability(0)
    except ImportError:
        pass
    
    return gpu_info


class SmartTrainingOptimizer:
    """Akıllı eğitim optimizasyonu sistemi"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.colab_resources = get_colab_resources() if SMARTFARM_MODULES_AVAILABLE else {}
        self.gpu_info = check_gpu_availability()
        
        print("🎯 SmartFarm Training Optimizer başlatıldı")
        print(f"🌐 Ortam: {'Google Colab' if is_colab() else 'Yerel'}")
        print(f"🎮 GPU: {self.gpu_info['name']}")
        if self.gpu_info['available']:
            print(f"💾 GPU Memory: {self.gpu_info['memory_gb']:.1f}GB")
    
    def analyze_dataset(self, images_dir: str, labels_dir: str) -> Dict[str, Any]:
        """Dataset analizi"""
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        
        # Görüntü dosyalarını say
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_path.glob(f'*{ext}')))
            image_files.extend(list(images_path.glob(f'*{ext.upper()}')))
        
        # Etiket dosyalarını say
        label_files = list(labels_path.glob('*.txt'))
        
        # Örnek görüntü boyutu analizi
        sample_sizes = []
        if image_files:
            try:
                import cv2
                for img_file in image_files[:10]:  # İlk 10 dosyayı kontrol et
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        h, w = img.shape[:2]
                        sample_sizes.append((w, h))
            except ImportError:
                pass
        
        # Ortalama görüntü boyutu
        if sample_sizes:
            avg_width = np.mean([s[0] for s in sample_sizes])
            avg_height = np.mean([s[1] for s in sample_sizes])
            avg_pixels = avg_width * avg_height
        else:
            avg_width = avg_height = avg_pixels = 0
        
        # Sınıf analizi (etiket dosyalarından)
        class_counts = {}
        total_objects = 0
        
        for label_file in label_files[:100]:  # İlk 100 dosyayı analiz et
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                            total_objects += 1
            except:
                continue
        
        return {
            'total_images': len(image_files),
            'total_labels': len(label_files),
            'matched_pairs': min(len(image_files), len(label_files)),
            'avg_image_size': {
                'width': int(avg_width) if avg_width > 0 else 640,
                'height': int(avg_height) if avg_height > 0 else 640,
                'pixels': int(avg_pixels) if avg_pixels > 0 else 640*640
            },
            'num_classes': len(class_counts),
            'total_objects': total_objects,
            'avg_objects_per_image': total_objects / max(1, len(label_files)),
            'class_distribution': class_counts
        }
    
    def get_optimal_training_config(self, 
                                  dataset_size: int,
                                  model_size: str = "yolov8n",
                                  task_complexity: str = "medium",
                                  target_accuracy: float = 0.8) -> Dict[str, Any]:
        """Optimal eğitim konfigürasyonu hesapla"""
        
        # Model boyutu kategorisi
        if 'n' in model_size.lower():
            model_category = "nano"
        elif 's' in model_size.lower():
            model_category = "small"
        elif 'm' in model_size.lower():
            model_category = "medium"
        elif 'l' in model_size.lower():
            model_category = "large"
        elif 'x' in model_size.lower():
            model_category = "xlarge"
        else:
            model_category = "medium"
        
        # Base konfigürasyon
        # Config dosyasından batch size al
        config_batch_size = get_default_batch_size()
        
        base_configs = {
            "nano": {
                "base_epochs": {"simple": 100, "medium": 200, "complex": 300},
                "batch_size": {"colab": config_batch_size, "local": config_batch_size},
                "image_size": 640,
                "patience": 30
            },
            "small": {
                "base_epochs": {"simple": 150, "medium": 250, "complex": 400},
                "batch_size": {"colab": config_batch_size, "local": config_batch_size},
                "image_size": 640,
                "patience": 40
            },
            "medium": {
                "base_epochs": {"simple": 200, "medium": 300, "complex": 500},
                "batch_size": {"colab": config_batch_size, "local": config_batch_size},
                "image_size": 640,
                "patience": 50
            },
            "large": {
                "base_epochs": {"simple": 250, "medium": 400, "complex": 600},
                "batch_size": {"colab": config_batch_size, "local": config_batch_size},
                "image_size": 640,
                "patience": 60
            },
            "xlarge": {
                "base_epochs": {"simple": 300, "medium": 500, "complex": 800},
                "batch_size": {"colab": config_batch_size, "local": config_batch_size},
                "image_size": 640,
                "patience": 70
            }
        }
        
        config = base_configs[model_category]
        environment = "colab" if is_colab() else "local"
        
        # Dataset boyutuna göre ayarlama
        if dataset_size < 500:
            size_factor = 0.5
            warning = "⚠️ Çok küçük dataset - overfitting riski çok yüksek!"
            recommendation = "Daha fazla veri toplayın veya augmentation kullanın"
        elif dataset_size < 1000:
            size_factor = 0.7
            warning = "⚠️ Küçük dataset - overfitting riski yüksek"
            recommendation = "Early stopping ve augmentation kullanın"
        elif dataset_size < 5000:
            size_factor = 1.0
            warning = None
            recommendation = "Dengeli dataset boyutu"
        elif dataset_size < 20000:
            size_factor = 1.3
            warning = None
            recommendation = "İyi dataset boyutu - uzun eğitim yapabilirsiniz"
        else:
            size_factor = 1.5
            warning = None
            recommendation = "Büyük dataset - daha uzun eğitim gerekebilir"
        
        # Hesaplanmış değerler
        base_epochs = config["base_epochs"][task_complexity]
        recommended_epochs = int(base_epochs * size_factor)
        
        # Colab için özel ayarlamalar
        if is_colab():
            # Colab session timeout (12 saat) göz önünde bulundur
            max_colab_epochs = 800  # Güvenli üst limit
            recommended_epochs = min(recommended_epochs, max_colab_epochs)
            
            # Batch size config dosyasından okunuyor
            config["batch_size"][environment] = get_default_batch_size()
        
        # Early stopping konfigürasyonu
        early_stopping_config = {
            "patience": max(20, recommended_epochs // 10),
            "min_delta": 0.001,
            "monitor_metric": "val_loss",
            "overfitting_threshold": 0.1 if dataset_size < 1000 else 0.15
        }
        
        # Epoch süresi tahmini
        estimated_epoch_duration = self._estimate_epoch_duration(
            dataset_size, config["batch_size"][environment], 
            config["image_size"], model_category
        )
        
        total_estimated_time = recommended_epochs * estimated_epoch_duration
        
        # 2000 epoch analizi
        epoch_2000_analysis = self._analyze_2000_epochs(
            dataset_size, model_category, task_complexity
        )
        
        return {
            "recommended_config": {
                "epochs": recommended_epochs,
                "min_epochs": max(50, recommended_epochs // 3),
                "max_epochs": min(1000, recommended_epochs * 2),
                "batch_size": config["batch_size"][environment],
                "image_size": config["image_size"],
                "patience": early_stopping_config["patience"],
                "learning_rate": 0.01,
                "weight_decay": 0.0005
            },
            "early_stopping": early_stopping_config,
            "time_estimates": {
                "estimated_epoch_duration_minutes": estimated_epoch_duration / 60,
                "total_estimated_hours": total_estimated_time / 3600,
                "colab_session_warning": total_estimated_time > 10 * 3600 if is_colab() else False
            },
            "dataset_analysis": {
                "size": dataset_size,
                "size_factor": size_factor,
                "category": "Küçük" if dataset_size < 1000 else "Orta" if dataset_size < 10000 else "Büyük"
            },
            "model_info": {
                "model_size": model_size,
                "category": model_category,
                "complexity": task_complexity
            },
            "warnings": [w for w in [warning] if w],
            "recommendations": [recommendation],
            "epoch_2000_analysis": epoch_2000_analysis,
            "environment": {
                "platform": "Google Colab" if is_colab() else "Local",
                "gpu_available": self.gpu_info['available'],
                "gpu_memory_gb": self.gpu_info['memory_gb']
            }
        }
    
    def _estimate_epoch_duration(self, dataset_size: int, batch_size: int, 
                               image_size: int, model_category: str) -> float:
        """Epoch süresi tahmini (saniye)"""
        
        # Base duration per image (saniye)
        base_durations = {
            "nano": 0.01,
            "small": 0.015,
            "medium": 0.025,
            "large": 0.04,
            "xlarge": 0.06
        }
        
        base_duration = base_durations.get(model_category, 0.025)
        
        # GPU faktörü
        gpu_factor = 0.3 if self.gpu_info['available'] else 1.0
        
        # Colab faktörü (shared resources)
        colab_factor = 1.2 if is_colab() else 1.0
        
        # Image size faktörü
        size_factor = (image_size / 640) ** 2
        
        # Toplam süre hesapla
        images_per_epoch = dataset_size
        batches_per_epoch = np.ceil(images_per_epoch / batch_size)
        
        epoch_duration = (batches_per_epoch * base_duration * 
                         gpu_factor * colab_factor * size_factor)
        
        return max(30, epoch_duration)  # Minimum 30 saniye
    
    def _analyze_2000_epochs(self, dataset_size: int, model_category: str, 
                           task_complexity: str) -> Dict[str, Any]:
        """2000 epoch analizi"""
        
        # 2000 epoch'un mantıklı olup olmadığını analiz et
        optimal_range = {
            "nano": {"simple": (50, 300), "medium": (100, 500), "complex": (200, 800)},
            "small": {"simple": (100, 400), "medium": (150, 600), "complex": (250, 1000)},
            "medium": {"simple": (150, 500), "medium": (200, 800), "complex": (300, 1200)},
            "large": {"simple": (200, 600), "medium": (250, 1000), "complex": (400, 1500)},
            "xlarge": {"simple": (250, 800), "medium": (300, 1200), "complex": (500, 2000)}
        }
        
        min_optimal, max_optimal = optimal_range[model_category][task_complexity]
        
        # Dataset boyutu faktörü
        if dataset_size < 1000:
            max_optimal = min(max_optimal, 500)  # Küçük dataset için limit
        
        is_reasonable = min_optimal <= 2000 <= max_optimal * 1.5
        
        if 2000 > max_optimal * 2:
            verdict = "Çok Fazla"
            reason = f"2000 epoch bu konfigürasyon için çok fazla. Önerilen maksimum: {max_optimal}"
            risk = "Yüksek overfitting riski, zaman kaybı"
        elif 2000 > max_optimal:
            verdict = "Fazla"
            reason = f"2000 epoch biraz fazla olabilir. Önerilen aralık: {min_optimal}-{max_optimal}"
            risk = "Orta overfitting riski"
        elif 2000 >= min_optimal:
            verdict = "Makul"
            reason = f"2000 epoch bu konfigürasyon için makul. Early stopping kullanın."
            risk = "Düşük risk (early stopping ile)"
        else:
            verdict = "Az"
            reason = f"2000 epoch bu model için az olabilir. Minimum önerilen: {min_optimal}"
            risk = "Underfitting riski"
        
        return {
            "verdict": verdict,
            "is_reasonable": is_reasonable,
            "reason": reason,
            "risk_assessment": risk,
            "optimal_range": f"{min_optimal}-{max_optimal}",
            "recommendation": f"Early stopping ile {min_optimal}-{max_optimal*2} aralığında başlayın"
        }
    
    def create_training_script(self, config: Dict[str, Any], 
                             output_path: str = "optimized_training.py") -> str:
        """Optimize edilmiş eğitim scripti oluştur"""
        
        script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SmartFarm Optimized Training Script
Otomatik oluşturulmuş eğitim scripti - {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""

import os
import sys
from pathlib import Path

# SmartFarm modüllerini import et
try:
    from early_stopping_system import EarlyStoppingManager, EarlyStoppingConfig, TrainingMetrics
    from training_optimizer import SmartTrainingOptimizer
    print("✅ SmartFarm modülleri yüklendi")
except ImportError as e:
    print(f"⚠️ SmartFarm modülleri yüklenemedi: {{e}}")
    print("Temel YOLO eğitimi ile devam ediliyor...")

# YOLOv8 import
try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO yüklendi")
except ImportError:
    print("❌ Ultralytics yüklenemedi! pip install ultralytics")
    sys.exit(1)

def main():
    """Ana eğitim fonksiyonu"""
    
    # Konfigürasyon
    config = {{
        "model": "{config['model_info']['model_size']}",
        "epochs": {config['recommended_config']['epochs']},
        "batch_size": {config['recommended_config']['batch_size']},
        "image_size": {config['recommended_config']['image_size']},
        "patience": {config['recommended_config']['patience']},
        "learning_rate": {config['recommended_config']['learning_rate']},
        "weight_decay": {config['recommended_config']['weight_decay']}
    }}
    
    print("🚀 SmartFarm Optimized Training başlatılıyor...")
    print(f"📊 Konfigürasyon: {{config}}")
    
    # Early stopping manager
    early_stopping_config = EarlyStoppingConfig(
        patience={config['early_stopping']['patience']},
        min_delta={config['early_stopping']['min_delta']},
        monitor_metric="{config['early_stopping']['monitor_metric']}",
        overfitting_threshold={config['early_stopping']['overfitting_threshold']}
    )
    
    early_stopping = EarlyStoppingManager(early_stopping_config)
    
    # Model yükle
    model = YOLO(config["model"])
    
    # Eğitim parametreleri
    train_args = {{
        "data": "path/to/your/dataset.yaml",  # BURAYA DATASET YAML YOLUNU GİRİN
        "epochs": config["epochs"],
        "batch": config["batch_size"],
        "imgsz": config["image_size"],
        "lr0": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "patience": config["patience"],
        "save": True,
        "save_period": 10,
        "cache": True,
        "device": "0" if "{config['environment']['gpu_available']}" == "True" else "cpu",
        "workers": 2,
        "project": "smartfarm_training",
        "name": "optimized_run"
    }}
    
    print("🎯 Eğitim başlatılıyor...")
    print(f"⏱️ Tahmini süre: {config['time_estimates']['total_estimated_hours']:.1f} saat")
    
    # Eğitimi başlat
    try:
        results = model.train(**train_args)
        print("✅ Eğitim başarıyla tamamlandı!")
        
        # Sonuçları kaydet
        early_stopping.save_training_report("final_training_report.json")
        
    except Exception as e:
        print(f"❌ Eğitim hatası: {{e}}")
        early_stopping.save_training_report("error_training_report.json")
    
    return results

if __name__ == "__main__":
    results = main()
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"📄 Optimize edilmiş eğitim scripti oluşturuldu: {output_path}")
        return output_path


# Yardımcı: main_multi_dataset.py tarafından verilen options sözlüğünü normalize et
def prepare_training_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Eğitimle ilgili parametreleri merkezi olarak normalize eder.
    - main_multi_dataset.py içindeki kullanıcı seçimlerini korur
    - Eksikse dataset önerilerinden (varsa) batch/imgsz gibi değerleri tamamlar
    - Anahtarların varlığını garanti eder
    """
    opts = dict(options) if isinstance(options, dict) else {}

    # Varsayılan anahtarlar ve güvenli değerler
    defaults = {
        'project': 'runs/train',
        'name': 'exp',
        'exist_ok': True,
        'use_hyp': True,
        'speed_mode': False,
        'workers': None,
    }
    for k, v in defaults.items():
        opts.setdefault(k, v)

    # Dataset önerilerinden batch/imgsz çek (varsa ve kullanıcı belirtmediyse)
    try:
        ds_cfg = opts.get('dataset_config') or {}
        if ds_cfg.get('type') == 'hierarchical_multi':
            rec = (ds_cfg.get('setup') or {}).get('recommendations') or {}
            if 'batch' not in opts or opts.get('batch') in (None, 0):
                if isinstance(rec.get('batch_size'), int) and rec['batch_size'] > 0:
                    opts['batch'] = rec['batch_size']
            if 'imgsz' not in opts or opts.get('imgsz') in (None, 0):
                if isinstance(rec.get('image_size'), int) and rec['image_size'] > 0:
                    opts['imgsz'] = rec['image_size']
    except Exception:
        pass

    # Zorunlu alanlar kontrolü (model, data, epochs)
    required = ['model', 'data', 'epochs']
    missing = [k for k in required if k not in opts]
    if missing:
        print(f"⚠️ prepare_training_options: Eksik alanlar: {missing}. Lütfen interaktif kurulum adımlarını tamamlayın.")

    return opts


# Kullanım örneği ve rehber
def print_epoch_recommendations():
    """Epoch sayısı rehberi yazdır"""
    
    print("""
🎯 SMARTFARM EPOCH REHBERİ
========================

❓ 2000 Epoch ile Başlamak Mantıklı mı?
---------------------------------------
KISA CEVAP: Genellikle HAYIR! 

🔍 Detaylı Analiz:
• Küçük dataset (<1000 görüntü): 100-300 epoch yeterli
• Orta dataset (1000-10000): 200-600 epoch optimal  
• Büyük dataset (>10000): 400-1000 epoch makul

⚠️ 2000 Epoch Riskleri:
• Overfitting (aşırı öğrenme)
• Zaman kaybı
• Kaynak israfı
• Colab session timeout

✅ Önerilen Yaklaşım:
1. 200-500 epoch ile başlayın
2. Early stopping kullanın (patience=50)
3. Validation loss'u izleyin
4. Gerekirse epoch sayısını artırın

🛑 Early Stopping Avantajları:
• Otomatik durdurma
• En iyi modeli koruma
• Overfitting önleme
• Zaman tasarrufu

📊 Model Boyutuna Göre Öneriler:
• YOLOv8n (nano): 100-400 epoch
• YOLOv8s (small): 150-500 epoch  
• YOLOv8m (medium): 200-600 epoch
• YOLOv8l (large): 250-800 epoch
• YOLOv8x (xlarge): 300-1000 epoch

🎯 Sonuç: Early stopping ile başlayın, 2000 epoch'u hedef değil limit olarak görün!
    """)
 
 
if __name__ == "__main__":
    # Epoch rehberini yazdır
    print_epoch_recommendations()
    
    # Örnek optimizasyon
    optimizer = SmartTrainingOptimizer()
    
    # Örnek dataset analizi
    dataset_size = 3000  # Kullanıcının dataset boyutu
    
    config = optimizer.get_optimal_training_config(
        dataset_size=dataset_size,
        model_size="yolov8m",
        task_complexity="medium"
    )
    
    print(f"\n🎯 {dataset_size} görüntülü dataset için öneriler:")
    print(f"📊 Önerilen epoch: {config['recommended_config']['epochs']}")
    print(f"⏱️ Tahmini süre: {config['time_estimates']['total_estimated_hours']:.1f} saat")
    print(f"🛑 Early stopping patience: {config['recommended_config']['patience']}")
    print(f"📈 Batch size: {config['recommended_config']['batch_size']}")
    
    print(f"\n🔍 2000 Epoch Analizi:")
    analysis = config['epoch_2000_analysis']
    print(f"Karar: {analysis['verdict']}")
    print(f"Sebep: {analysis['reason']}")
    print(f"Risk: {analysis['risk_assessment']}")
    print(f"Öneri: {analysis['recommendation']}")
    
    # Optimize edilmiş script oluştur
    optimizer.create_training_script(config, "smartfarm_optimized_training.py")
