#!/usr/bin/env python3
# main_multi_dataset.py - YOLO11 Hierarchical Multi-Dataset Training Framework

import os
import sys
from pathlib import Path
import shutil
from datetime import datetime
import json

# Import framework components
from setup_utils import check_gpu, install_required_packages
from hyperparameters import create_hyperparameters_file, load_hyperparameters
from memory_utils import show_memory_usage, clean_memory
from training import train_model, save_to_drive
from model_downloader import download_yolo11_models, download_specific_model_type
from language_manager import get_text, select_language

# Import updated multi-dataset manager
try:
    from multi_dataset_manager import YAMLBasedMultiDatasetManager
except ImportError:
    print("⚠️  YAMLBasedMultiDatasetManager not found, trying legacy import...")
    from multi_dataset_manager import MultiDatasetManager as YAMLBasedMultiDatasetManager

# Import hierarchical detection utils
try:
    from hierarchical_detection_utils import HierarchicalDetectionVisualizer
    HIERARCHICAL_DETECTION_AVAILABLE = True
except ImportError:
    print("⚠️  HierarchicalDetectionVisualizer not available")
    HIERARCHICAL_DETECTION_AVAILABLE = False

# Import augmentation systems
try:
    from tomato_disease_augmentation import TomatoDiseaseAugmentation
    from tomato_pest_augmentation import TomatoPestAugmentation
    AUGMENTATION_SYSTEMS_AVAILABLE = True
except ImportError:
    print("⚠️  Augmentation systems not available")
    AUGMENTATION_SYSTEMS_AVAILABLE = False

# Check if running in Colab
def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        print("✅ Google Colab environment detected.")
        return True
    except:
        print("💻 Running in local environment.")
        return False

def get_tarim_drive_paths():
    """Google Drive'da 'Tarım' taban klasörünü ve alt klasörlerini bul/oluştur.
    Dönüş: {
      'base': '/content/drive/MyDrive/Tarım',
      'colab_egitim': '/content/drive/MyDrive/Tarim/colab_egitim',
      'yolo11_models': '/content/drive/MyDrive/Tarim/colab_egitim/yolo11_models'
    }
    """
    if not is_colab():
        return None
    # Drive mount kontrolü
    if not os.path.exists('/content/drive'):
        if not mount_google_drive():
            return None
    mydrive = "/content/drive/MyDrive"
    # Türkçe 'Tarım' varsa onu kullan, yoksa 'Tarim' ya da oluştur
    tarim_candidates = [os.path.join(mydrive, 'Tarım'), os.path.join(mydrive, 'Tarim')]
    base = None
    for c in tarim_candidates:
        if os.path.exists(c):
            base = c
            break
    if base is None:
        # Önce Türkçe karakterli klasörü oluşturmayı dene
        base = tarim_candidates[0]
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            base = tarim_candidates[1]
            os.makedirs(base, exist_ok=True)
    colab_egitim = os.path.join(base, 'colab_egitim')
    yolo11_models = os.path.join(base, 'yolo11_models')
    os.makedirs(colab_egitim, exist_ok=True)
    os.makedirs(yolo11_models, exist_ok=True)
    return {'base': base, 'colab_egitim': colab_egitim, 'yolo11_models': yolo11_models}

def get_smartfarm_models_dir():
    """Colab için model klasörünü '/content/drive/MyDrive/SmartFarm/colab_learn/yolo11_models' olarak döndürür ve yoksa oluşturur."""
    if not is_colab():
        return None
    if not os.path.exists('/content/drive'):
        if not mount_google_drive():
            return None
    path = "/content/drive/MyDrive/SmartFarm/colab_learn/yolo11_models"
    os.makedirs(path, exist_ok=True)
    return path

def mount_google_drive():
    """Mount Google Drive"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted successfully.")
        return True
    except Exception as e:
        print(f"❌ Error mounting Google Drive: {e}")
        return False

def save_models_to_drive(drive_folder_path, best_file=True, last_file=True):
    """Save best and last model files to Google Drive"""
    if not is_colab():
        print("ℹ️  This function only works in Google Colab.")
        return False
    
    # Check if Google Drive is mounted
    if not os.path.exists('/content/drive'):
        if not mount_google_drive():
            return False
    
    # Check source directory
    source_dir = "runs/train/exp/weights"
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        return False
    
    # Create target directory
    os.makedirs(drive_folder_path, exist_ok=True)
    
    # Copy files
    copied_files = []
    
    if best_file and os.path.exists(os.path.join(source_dir, "best.pt")):
        shutil.copy2(os.path.join(source_dir, "best.pt"), os.path.join(drive_folder_path, "best.pt"))
        copied_files.append("best.pt")
    
    if last_file and os.path.exists(os.path.join(source_dir, "last.pt")):
        shutil.copy2(os.path.join(source_dir, "last.pt"), os.path.join(drive_folder_path, "last.pt"))
        copied_files.append("last.pt")
    
    # Copy additional files
    additional_files = ["merged_dataset.yaml", "unified_class_mapping.json", "analysis_report.json"]
    for file_name in additional_files:
        if os.path.exists(file_name):
            shutil.copy2(file_name, os.path.join(drive_folder_path, file_name))
            copied_files.append(file_name)
    
    if copied_files:
        print(f"✅ Files saved to Google Drive: {', '.join(copied_files)}")
        print(f"📁 Save location: {drive_folder_path}")
        print(f"📂 Dosyalar şu klasörde: {drive_folder_path}")
        print(f"🗂️  Kaydedilen dosya sayısı: {len(copied_files)}")
        return True
    else:
        print("❌ No files found to copy.")
        return False

def download_models_menu():
    """Interactive menu for downloading YOLO11 models"""
    print(f"\n{get_text('model_download_title')}")
    
    if is_colab():
        default_dir = get_smartfarm_models_dir() or "/content/colab_learn/yolo11_models"
    else:
        default_dir = "yolo11_models"
    save_dir = input(get_text('save_directory', default=default_dir)) or default_dir
    
    print(f"\n{get_text('download_options')}")
    print(get_text('single_model'))
    print(get_text('all_detection'))
    print(get_text('all_models'))
    
    choice = input(f"\n{get_text('your_choice')}")
    
    if choice == "1":
        print(f"\n{get_text('select_model_type')}")
        print(get_text('detection_default'))
        print(get_text('segmentation'))
        print(get_text('classification'))
        print(get_text('pose'))
        print(get_text('obb'))
        
        model_type_map = {
            "1": "detection",
            "2": "segmentation", 
            "3": "classification",
            "4": "pose",
            "5": "obb"
        }
        
        model_type_choice = input(get_text('enter_choice_1_5')) or "1"
        model_type = model_type_map.get(model_type_choice, "detection")
        
        print(f"\n{get_text('select_model_size')}")
        print(get_text('small'))
        print(get_text('medium_default'))
        print(get_text('large'))
        print(get_text('extra_large'))
        
        size_map = {
            "1": "s",
            "2": "m",
            "3": "l",
            "4": "x"
        }
        
        size_choice = input(get_text('enter_choice_1_4')) or "2"
        size = size_map.get(size_choice, "m")
        
        model_path = download_specific_model_type(model_type, size, save_dir)
        if model_path:
            print(f"\n✅ Model başarıyla indirildi: {model_path}")
    
    elif choice == "2":
        detection_models = ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        downloaded = download_yolo11_models(save_dir, detection_models)
        print(f"\n✅ {len(downloaded)} tespit modeli indirildi: {save_dir}")
    
    elif choice == "3":
        downloaded = download_yolo11_models(save_dir)
        print(f"\n✅ {len(downloaded)} model indirildi: {save_dir}")
    
    else:
        print("\n❌ Geçersiz seçim. Hiçbir model indirilmedi.")
        return None
    
    return save_dir

def hierarchical_dataset_setup():
    """Setup for hierarchical multi-dataset training"""
    print("\n===== Hiyerarşik Çoklu Veri Seti Kurulumu =====")
    
    # Initialize the YAML-based dataset manager
    config_file = input("Konfigürasyon dosyası yolu (varsayılan: config_datasets.yaml): ") or "config_datasets.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Konfigürasyon dosyası bulunamadı: {config_file}")
        print("Lütfen config_datasets.yaml dosyasının mevcut dizinde olduğundan emin olun")
        return None
    
    manager = YAMLBasedMultiDatasetManager(config_file=config_file)
    
    # Show system information
    print(f"\n📊 Sistem Bilgileri:")
    print(f"✅ Konfigürasyon yüklendi: {config_file}")
    print(f"📁 Mevcut gruplar: {len(manager.get_available_dataset_groups())}")
    
    # Interactive dataset selection
    selected_group = manager.interactive_dataset_selection()
    
    if not selected_group:
        print("❌ Hiçbir veri seti grubu seçilmedi")
        return None
    
    # Get recommendations
    recommendations = manager.get_training_recommendations(selected_group)
    
    print(f"\n🎯 '{selected_group}' için Eğitim Önerileri:")
    for key, value in recommendations.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    • {item}")
        else:
            print(f"  {key}: {value}")
    
    # Get global settings
    settings = manager.get_global_settings()
    
    # Ask for target count per class (global default + opsiyonel sınıf bazlı hedefler)
    default_target = settings.get('default_target_count_per_class', 5000)
    while True:
        try:
            target_count = int(input(f"\nSınıf başına hedef örnek sayısı (varsayılan: {default_target}): ") or str(default_target))
            if target_count > 0:
                break
            print("❌ Lütfen pozitif bir sayı girin.")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin.")

    # Opsiyonel: Kullanıcı sınıf bazında özel hedef sayıları girmek isterse
    per_class_targets = None
    customize = (input("\nSınıf bazında hedef sayıları özelleştirmek ister misiniz? (e/h, varsayılan: h): ") or "h").lower()
    if customize.startswith('e'):
        per_class_targets = {}
        print("\nSınıf bazlı hedefler (boş bırakılırsa genel varsayılan kullanılacak):")
        for cls in manager.hierarchical_classes.keys():
            try:
                val = input(f"  • {cls} için hedef (varsayılan {target_count}): ")
                if val.strip() == "":
                    continue
                n = int(val)
                if n > 0:
                    per_class_targets[cls] = n
            except Exception:
                pass
    
    # Output directory
    default_output = "datasets/hierarchical_merged"
    output_dir = input(f"\nBirleştirilmiş veri seti dizini (varsayılan: {default_output}): ") or default_output
    
    return {
        'manager': manager,
        'selected_group': selected_group,
        'target_count': target_count,
        'per_class_targets': per_class_targets,
        'output_dir': output_dir,
        'recommendations': recommendations,
        'settings': settings
    }

def process_hierarchical_datasets(dataset_config):
    """Process hierarchical multi-datasets"""
    print("\n===== Hiyerarşik Çoklu Veri Setleri İşleniyor =====")
    
    manager = dataset_config['manager']
    target_count = dataset_config['target_count']
    
    try:
        # 1. Download datasets
        print("\n1️⃣ Veri setleri indiriliyor...")
        download_success = manager.download_all_datasets()
        
        if not download_success:
            print("❌ Veri seti indirme başarısız!")
            return False
        
        # 2. Create unified class mapping
        print("\n2️⃣ Hiyerarşik sınıf haritalaması oluşturuluyor...")
        classes_created = manager.create_unified_class_mapping()
        
        if classes_created == 0:
            print("❌ Hiçbir sınıf haritalandırılamadı!")
            return False
        
        print(f"✅ {classes_created} ana sınıf oluşturuldu")
        
        # 3. Merge datasets with hierarchical structure
        print("\n3️⃣ Veri setleri hiyerarşik yapıyla birleştiriliyor...")
        # Fonksiyona 'setup' dict'i geçirildiği için doğrudan buradan oku
        pct = dataset_config.get('per_class_targets')
        target_arg = pct if pct else target_count
        merged_counts = manager.merge_datasets(target_count_per_class=target_arg)
        
        if not merged_counts:
            print("❌ Veri seti birleştirme başarısız!")
            return False
        
        print(f"\n✅ Hiyerarşik çoklu veri seti işleme tamamlandı!")
        print(f"📁 Birleştirilmiş veri seti: {manager.output_dir}")
        print(f"📄 YAML dosyası: merged_dataset.yaml")
        print(f"🏷️  Sınıf haritası: unified_class_mapping.json")
        
        # Display final statistics
        total_samples = sum(merged_counts.values())
        print(f"\n📊 Son Veri Seti İstatistikleri:")
        print(f"   Toplam örnek: {total_samples:,}")
        print(f"   Ana sınıflar: {len(merged_counts)}")
        print(f"   Sınıf başına örnek: {total_samples // len(merged_counts):,} (ortalama)")
        
        for class_name, count in merged_counts.items():
            print(f"   {class_name}: {count:,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Hiyerarşik veri seti işleme sırasında hata: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_training_setup():
    """Interactive training parameter setup for hierarchical model"""
    print("\n===== Hiyerarşik Model Eğitim Kurulumu =====")
    
    # Dataset type selection
    print("\nVeri seti konfigürasyonu:")
    print("1) Hiyerarşik çoklu veri seti (Önerilen)")
    print("2) Tek Roboflow veri seti (Eski)")
    
    while True:
        dataset_choice = input("\nSeçenek [1-2] (varsayılan: 1): ") or "1"
        if dataset_choice in ["1", "2"]:
            break
        print("❌ Lütfen 1 veya 2 seçin.")
    
    dataset_config = {}
    
    if dataset_choice == "1":
        # Hierarchical multi-dataset
        dataset_setup = hierarchical_dataset_setup()
        if not dataset_setup:
            return None
        
        dataset_config = {
            'type': 'hierarchical_multi',
            'setup': dataset_setup,
            'data_yaml': 'merged_dataset.yaml'
        }
    else:
        # Single dataset (legacy)
        roboflow_url = input("\nRoboflow URL: ").strip()
        if not roboflow_url:
            print("❌ URL sağlanmadı")
            return None
        
        dataset_config = {
            'type': 'single',
            'url': roboflow_url,
            'data_yaml': 'dataset.yaml'
        }
    
    # Project category
    print("\nProje kategorisi:")
    print("1) Hiyerarşik Tarımsal AI (Önerilen)")
    print("2) Hastalık Tespiti")
    print("3) Zararlı Tespiti")
    print("4) Karma Tarımsal")
    print("5) Özel")
    
    while True:
        category_choice = input("\nKategori seçin [1-5] (varsayılan: 1): ") or "1"
        category_options = {
            "1": "hierarchical_agricultural",
            "2": "diseases",
            "3": "pests", 
            "4": "mixed",
            "5": "custom"
        }
        
        if category_choice in category_options:
            category = category_options[category_choice]
            if category == "custom":
                category = input("Özel kategori adı girin: ").strip() or "custom"
            break
        print("❌ Lütfen 1-5 arası seçin.")
    
    # Get training recommendations
    if dataset_config['type'] == 'hierarchical_multi':
        recommendations = dataset_config['setup']['recommendations']
        recommended_model = recommendations.get('model', 'yolo11l.pt')
        recommended_batch = recommendations.get('batch_size', 8)
        recommended_size = recommendations.get('image_size', 640)
        estimated_time = recommendations.get('estimated_time', 'Unknown')
        
        print(f"\n🎯 Hiyerarşik model için öneriler:")
        print(f"   Model: {recommended_model}")
        print(f"   Batch boyutu: {recommended_batch}")
        print(f"   Görüntü boyutu: {recommended_size}")
        print(f"   Tahmini süre: {estimated_time}")
        
        # Show special notes if available
        special_notes = recommendations.get('special_notes', [])
        if special_notes:
            print(f"   Özel hususlar:")
            for note in special_notes:
                print(f"     • {note}")
    
    # Training parameters
    while True:
        try:
            if dataset_config['type'] == 'hierarchical_multi':
                default_epochs = 1000  # Updated default for hierarchical model
                epochs = int(input(f"\nEpoch sayısı [100-2000 önerilen] (varsayılan: {default_epochs}): ") or str(default_epochs))
            else:
                default_epochs = 1000  # Updated default for single dataset model
                epochs = int(input(f"\nEpoch sayısı [100-2000 önerilen] (varsayılan: {default_epochs}): ") or str(default_epochs))
            
            if epochs > 0:
                break
            print("❌ Lütfen pozitif bir sayı girin.")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin.")
    
    # Model size selection
    print("\nModel boyutunu seçin:")
    print("1) yolo11s.pt - Küçük (en hızlı, düşük doğruluk)")
    print("2) yolo11m.pt - Orta (dengeli)")
    print("3) yolo11l.pt - Büyük (yüksek doğruluk, yavaş) [Hiyerarşik için önerilen]")
    print("4) yolo11x.pt - Çok Büyük (en yüksek doğruluk, en yavaş)")

    while True:
        if dataset_config['type'] == 'hierarchical_multi':
            model_choice = input("\nModel seçin [1-4] (varsayılan: 3): ") or "3"
        else:
            model_choice = input("\nModel seçin [1-4] (varsayılan: 3): ") or "3"
        
        model_options = {
            "1": "yolo11s.pt",
            "2": "yolo11m.pt",
            "3": "yolo11l.pt",
            "4": "yolo11x.pt"
        }
        
        if model_choice in model_options:
            model = model_options[model_choice]
            
            # Check if model exists locally/Drive
            if is_colab():
                model_dir = get_smartfarm_models_dir() or os.path.join("/content/colab_learn", "yolo11_models")
            else:
                model_dir = "yolo11_models"
            model_path = os.path.join(model_dir, model)
            
            if not os.path.exists(model_path):
                print(f"\n⚠️  Model {model} yerel olarak bulunamadı.")
                download_now = input("Şimdi indir? (e/h, varsayılan: e): ").lower() or "e"
                
                if download_now.startswith("e"):
                    os.makedirs(model_dir, exist_ok=True)
                    download_specific_model_type("detection", model[6], model_dir)
                else:
                    print(f"ℹ️  Model eğitim sırasında otomatik olarak indirilecek.")
            break
        print("❌ Lütfen 1-4 arası seçin.")
    
    # Batch size ve image size varsayılanları (Colab için optimize)
    # Öneri: batch_size=16, img_size=512 (RAM ve hız dengesi)
    default_batch = 16
    default_img_size = 512
    
    while True:
        try:
            batch_size = int(input(f"\nBatch boyutu (varsayılan: {default_batch}, düşük RAM için küçük): ") or str(default_batch))
            if batch_size > 0:
                break
            print("❌ Lütfen pozitif bir sayı girin.")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin.")
    
    while True:
        try:
            img_size = int(input(f"\nGörüntü boyutu (varsayılan: {default_img_size}, 32'nin katı olmalı • Colab için 512 önerilir): ") or str(default_img_size))
            if img_size > 0 and img_size % 32 == 0:
                break
            print("❌ Lütfen 32'nin katı olan pozitif bir sayı girin.")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin.")

    # Speed mode (optimize epoch time)
    speed_mode_input = (input("\nHız modu (cache=ram, workers=8, plots=False) açılsın mı? (e/h, varsayılan: e): ") or "e").lower()
    speed_mode = speed_mode_input.startswith('e')
    
    # Google Drive save settings
    drive_save_path = None
    if is_colab():
        print("\nGoogle Drive kaydetme ayarları:")
        save_to_drive_opt = input("Eğitim sonuçlarını Google Drive'a kaydet? (e/h, varsayılan: e): ").lower() or "e"
        
        if save_to_drive_opt.startswith("e"):
            default_drive_path = get_smartfarm_models_dir() or "/content/drive/MyDrive/SmartFarm/colab_learn/yolo11_models"
            drive_save_path = input(f"Kaydetme dizini (varsayılan: {default_drive_path}): ") or default_drive_path
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drive_save_path = os.path.join(drive_save_path, timestamp)
            print(f"📁 Modeller şuraya kaydedilecek: {drive_save_path}")
            # Klasörleri oluştur
            os.makedirs(drive_save_path, exist_ok=True)
    
    # Hyperparameter file
    use_hyp = input("\nHiperparametre dosyası kullan (hyp.yaml)? (e/h, varsayılan: e): ").lower() or "e"
    use_hyp = use_hyp.startswith("e")
    
    # Device detection
    device = check_gpu()
    
    # Build options dictionary
    options = {
        'dataset_config': dataset_config,
        'epochs': epochs,
        'model': model,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'workers': 2,
        'data': dataset_config['data_yaml'],
        'project': 'runs/train',
        'name': 'exp',
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'exist_ok': True,
        'use_hyp': use_hyp,
        'category': category,
        'drive_save_path': drive_save_path,
        'speed_mode': speed_mode
    }
    
    # Save interval prompt (Drive kullanılıyorsa 10, değilse 50 varsayılan)
    save_interval_default = 10 if drive_save_path else 50
    try:
        if drive_save_path:
            save_interval = int(input(f"Drive'a kaç epoch'ta bir kaydedilsin? (varsayılan: {save_interval_default}): ") or str(save_interval_default))
        else:
            save_interval = int(input(f"Modele kaç epoch'ta bir yerel kaydetme yapılsın? (varsayılan: {save_interval_default}): ") or str(save_interval_default))
    except ValueError:
        save_interval = save_interval_default
    options['save_interval'] = save_interval
    
    # Display selected parameters
    print("\n===== Seçilen Eğitim Parametreleri =====")
    print(f"Veri seti tipi: {dataset_config['type']}")
    if dataset_config['type'] == 'hierarchical_multi':
        setup = dataset_config['setup']
        print(f"Veri seti grubu: {setup['selected_group']}")
        print(f"Sınıf başına hedef örnek: {setup['target_count']:,}")
        print(f"Çıktı dizini: {setup['output_dir']}")
    
    print(f"Model: {model}")
    print(f"Epoch: {epochs}")
    print(f"Batch boyutu: {batch_size}")
    print(f"Görüntü boyutu: {img_size}")
    print(f"Cihaz: {device}")
    print(f"DataLoader workers: {options['workers']} (hafıza için düşük)")
    print(f"Dataset cache varsayılanı: {'ram' if speed_mode else 'disk'}")
    print(f"cuDNN benchmark: Enabled (training.py içinde)")
    print(f"Hız modu: {'Açık' if speed_mode else 'Kapalı'}")
    print(f"Kategori: {category}")
    if dataset_config['type'] == 'hierarchical_multi':
        pct = dataset_config['setup'].get('per_class_targets')
        if pct:
            print("Sınıf bazlı hedefler: (özet)")
            shown = 0
            for k, v in pct.items():
                print(f"  • {k}: {v}")
                shown += 1
                if shown >= 10:
                    print("  • ... (daha fazla sınıf var)")
                    break
    if drive_save_path:
        print(f"Drive'a kaydetme aralığı: {options['save_interval']} epoch")
    else:
        print(f"Yerel kaydetme aralığı: {options['save_interval']} epoch")
    
    if drive_save_path:
        print(f"Drive kaydetme yolu: {drive_save_path}")
        # Kaydedilecek dosyaları net belirt
        print(f"Kaydedilecek dosyalar:")
        print(f"  • best.pt  → {os.path.join(drive_save_path, 'best.pt')}")
        print(f"  • last.pt  → {os.path.join(drive_save_path, 'last.pt')}")
    
    confirm = (input("\nBu parametrelerle devam et? (e/h, varsayılan: e): ") or "e").lower()
    if confirm != 'e' and confirm != 'evet' and confirm != 'yes':
        print("❌ Kurulum iptal edildi.")
        return None
    
    return options

def main():
    """Main function - Hierarchical Multi-Dataset Training Framework"""
    # Language selection at startup
    select_language()
    
    print("\n" + "="*70)
    print(get_text('main_title'))
    print(get_text('main_subtitle'))
    print("="*70)
    
    print(f"\n{get_text('main_menu')}")
    print(get_text('option_download'))
    print(get_text('option_training'))
    print(get_text('option_test'))
    print(get_text('option_exit'))
    
    choice = input(f"\n{get_text('select_option')}")
    
    if choice == "1":
        download_models_menu()
        if get_text('language_choice').startswith('e'):
            train_now = input("\nEğitim kurulumuna geç? (e/h, varsayılan: e): ").lower() or "e"
            if not train_now.startswith("e"):
                return
        else:
            train_now = input("\nProceed to training setup? (y/n, default: y): ").lower() or "y"
            if not train_now.startswith("y"):
                return
        choice = "2"  # Continue to training
        
    if choice == "2":
        in_colab = is_colab()
        
        # (Opsiyonel) Gerekli paketleri yükleme
        # Not: Paket kurulumlarını genellikle colab_setup.py üzerinden yönetmeniz önerilir.
        do_install = (input("\nGerekli paketleri şimdi yüklemek ister misiniz? (e/h, varsayılan: h): ") or "h").lower()
        if do_install.startswith("e"):
            print("\n📦 Gerekli paketler yükleniyor...")
            install_required_packages()
        else:
            print("\n⏭️ Paket yükleme atlandı. (colab_setup.py ile kurulumu yapabilirsiniz)")
        
        # Interactive setup - this will handle checkpoint checking
        options = interactive_training_setup()
        if options is None:
            return
        
        # Check if we're resuming from a checkpoint
        if options.get('resume'):
            print("\n" + "="*50)
            print(f"🔄 Eğitime devam ediliyor: {options['checkpoint_path']}")
            print("="*50)
            
            # Skip dataset processing when resuming
            results = train_model(options, hyp=None, epochs=options['epochs'], 
                               drive_save_interval=options.get('save_interval', 10))
        else:
            # Process dataset(s) for new training
            dataset_config = options['dataset_config']
            
            if dataset_config['type'] == 'single':
                # Single dataset processing (legacy)
                from dataset_utils import download_dataset
                
                if not download_dataset(dataset_config['url']):
                    print('❌ Veri seti indirme başarısız. Çıkılıyor...')
                    return
                    
            elif dataset_config['type'] == 'hierarchical_multi':
                # Hierarchical multi-dataset processing
                if not process_hierarchical_datasets(dataset_config['setup']):
                    print('❌ Hiyerarşik veri seti işleme başarısız. Çıkılıyor...')
                    return
            
            # Show memory status before training
            show_memory_usage("Eğitim Öncesi")
            
            # Create hyperparameter file for new training
            from hyperparameters import create_hyperparameters_file, load_hyperparameters
            hyp_path = create_hyperparameters_file()
            hyperparameters = load_hyperparameters(hyp_path)
            
            # Start new training
            print(f"\n🚀 Yeni model eğitimi başlatılıyor...")
            results = train_model(options, hyp=hyperparameters, 
                               epochs=options['epochs'], 
                               drive_save_interval=options.get('save_interval', 10))
        
        if results:
            print('✅ Eğitim başarıyla tamamlandı!')
            print(f'📊 Sonuçlar: {results}')
            
            # Initialize hierarchical detection if available
            if HIERARCHICAL_DETECTION_AVAILABLE:
                print(f"\n🎯 Hiyerarşik tespit sistemi başlatılıyor...")
                try:
                    visualizer = HierarchicalDetectionVisualizer()
                    print(f"✅ Hiyerarşik tespit sistemi hazır!")
                    print(f"🏷️  Tespit formatı: 'ZARARLI: Kırmızı Örümcek (0.85)'")
                except Exception as e:
                    print(f"⚠️  Hiyerarşik tespit başlatılamadı: {e}")
            
            # Save to Google Drive
            if in_colab and options.get('drive_save_path'):
                drive_path = options['drive_save_path']
                print(f"\n💾 Modeller Google Drive'a kaydediliyor...")
                print(f"📁 Hedef klasör: {drive_path}")
                if save_models_to_drive(drive_path):
                    print(f"✅ Modeller başarıyla kaydedildi: {drive_path}")
                    print(f"📂 Kaydedilen dosyalar şu konumda: {drive_path}")
                else:
                    print("❌ Modeller Google Drive'a kaydedilemedi.")
        else:
            print('❌ Eğitim başarısız veya kesildi.')
            
            # Save partial results if available
            if in_colab and options.get('drive_save_path'):
                save_anyway = input("\nKısmi eğitim sonuçlarını Google Drive'a kaydet? (e/h, varsayılan: e): ").lower() or "e"
                if save_anyway.startswith("e"):
                    drive_path = options['drive_save_path']
                    print(f"\n💾 Kısmi sonuçlar Google Drive'a kaydediliyor...")
                    print(f"📁 Hedef klasör: {drive_path}")
                    if save_models_to_drive(drive_path):
                        print(f"✅ Kısmi sonuçlar kaydedildi: {drive_path}")
                        print(f"📂 Kaydedilen dosyalar şu konumda: {drive_path}")
                    else:
                        print("❌ Kısmi sonuçlar kaydedilemedi.")
        
        # Clean memory
        show_memory_usage("Eğitim Sonrası")
        clean_memory()
    
    elif choice == "3":
        # Test hierarchical detection
        if not HIERARCHICAL_DETECTION_AVAILABLE:
            print("❌ Hiyerarşik tespit araçları mevcut değil.")
            return
        
        model_path = input("Eğitilmiş model yolunu girin (örn: runs/train/exp/weights/best.pt): ").strip()
        if not model_path or not os.path.exists(model_path):
            print("❌ Model dosyası bulunamadı.")
            return
        
        test_image = input("Test görüntüsü yolunu girin: ").strip()
        if not test_image or not os.path.exists(test_image):
            print("❌ Test görüntüsü bulunamadı.")
            return
        
        try:
            from ultralytics import YOLO
            import cv2
            
            # Load model and visualizer
            model = YOLO(model_path)
            visualizer = HierarchicalDetectionVisualizer()
            
            # Run detection
            print(f"🔍 Hiyerarşik tespit çalıştırılıyor...")
            image = cv2.imread(test_image)
            results = model(image)
            
            # Apply hierarchical visualization
            annotated_image = visualizer.process_yolo_results(image, results[0])
            
            # Save result
            output_path = "hierarchical_detection_result.jpg"
            cv2.imwrite(output_path, annotated_image)
            
            # Generate report
            detections = visualizer.get_detection_summary(results[0])
            report = visualizer.create_detection_report(detections)
            
            print(f"\n{report}")
            print(f"✅ Açıklamalı görüntü kaydedildi: {output_path}")
            
        except Exception as e:
            print(f"❌ Hiyerarşik tespit testi sırasında hata: {e}")
    
    elif choice == "4":
        print("👋 Çıkılıyor...")
    
    else:
        print("❌ Geçersiz seçenek. Çıkılıyor...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Kullanıcı tarafından kesildi. Çıkılıyor...")
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ İşlem tamamlandı.")
