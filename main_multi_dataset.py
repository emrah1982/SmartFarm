#!/usr/bin/env python3
# main_multi_dataset.py - Hiyerarşik çoklu dataset yönetimi ve eğitim

import os
import sys
import yaml
from pathlib import Path
import subprocess

# Roboflow API yönetimi için import
try:
    from roboflow_api_helper import get_roboflow_menu_choice, handle_roboflow_action, get_api_key_from_config
except ImportError:
    print("⚠️ roboflow_api_helper.py bulunamadı")
    def get_roboflow_menu_choice(): return {}
    def handle_roboflow_action(choice, **kwargs): return False
    def get_api_key_from_config(): return None
import shutil
from datetime import datetime
import json
import re

# Import framework components
from setup_utils import check_gpu, install_required_packages
from hyperparameters import create_hyperparameters_file, load_hyperparameters
from memory_utils import show_memory_usage, clean_memory
from training import train_model, save_to_drive
from training_optimizer import prepare_training_options
try:
    from drive_manager import DriveManager
    _DRIVE_AVAILABLE = True
except Exception:
    _DRIVE_AVAILABLE = False
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
    from augmentation import TomatoDiseaseAugmentation
    from augmentation import TomatoPestAugmentation
    AUGMENTATION_SYSTEMS_AVAILABLE = True
except ImportError:
    print("⚠️  Augmentation systems not available")
    AUGMENTATION_SYSTEMS_AVAILABLE = False

# Ek: Hedefe-tamamlama için temel augmentation pipeline (opsiyonel)
try:
    from augmentation_utils import YOLOAugmentationPipeline
    _AUG_PIPE_AVAILABLE = True
except Exception:
    _AUG_PIPE_AVAILABLE = False

# Global timestamp pattern for consistent naming
TIMESTAMP_PATTERN = re.compile(r'^\d{8}_\d{6}$')  # YYYYMMDD_HHMMSS format

# Global timestamp variable
_GLOBAL_TIMESTAMP = None

def get_global_timestamp():
    """Get or create global timestamp for consistent naming across the project"""
    global _GLOBAL_TIMESTAMP
    if _GLOBAL_TIMESTAMP is None:
        _GLOBAL_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _GLOBAL_TIMESTAMP

def save_user_preferences_config(options, dataset_config=None, augmentation_settings=None):
    """Kullanıcı tercihlerini ve ayarlarını configs klasöründe kaydet"""
    try:
        # Global timestamp ile config klasörü oluştur
        global_ts = get_global_timestamp()
        configs_dir = os.path.join('configs', global_ts)
        os.makedirs(configs_dir, exist_ok=True)
        
        # Ana konfigürasyon dosyası
        config_data = {
            'session_info': {
                'timestamp': global_ts,
                'created_at': datetime.now().isoformat(),
                'environment': 'colab' if is_colab() else 'local',
                'language': get_text('language_choice', default='tr')
            },
            'training_settings': {
                'model': options.get('model'),
                'epochs': options.get('epochs'),
                'batch_size': options.get('batch'),
                'image_size': options.get('imgsz'),
                'device': options.get('device'),
                'workers': options.get('workers'),
                'optimizer': options.get('optimizer'),
                'speed_mode': options.get('speed_mode'),
                'use_hyperparameters': options.get('use_hyp'),
                'category': options.get('category'),
                'project_path': options.get('project'),
                'experiment_name': options.get('name')
            },
            'dataset_configuration': {},
            'drive_settings': {
                'save_to_drive': bool(options.get('drive_save_path')),
                'drive_path': options.get('drive_save_path'),
                'save_interval': options.get('save_interval')
            },
            'augmentation_settings': augmentation_settings or {}
        }
        
        # Dataset konfigürasyonu detayları
        if dataset_config:
            config_data['dataset_configuration'] = {
                'type': dataset_config.get('type'),
                'data_yaml': dataset_config.get('data_yaml')
            }
            
            if dataset_config['type'] == 'hierarchical_multi':
                setup = dataset_config.get('setup', {})
                config_data['dataset_configuration'].update({
                    'selected_group': setup.get('selected_group'),
                    'target_count_per_class': setup.get('target_count'),
                    'per_class_targets': setup.get('per_class_targets'),
                    'output_directory': setup.get('output_dir'),
                    'label_mode': setup.get('label_mode'),
                    'recommendations': setup.get('recommendations')
                })
            elif dataset_config['type'] == 'single':
                config_data['dataset_configuration'].update({
                    'roboflow_url': dataset_config.get('url'),
                    'has_api_key': bool(dataset_config.get('api_key')),
                    'split_config': dataset_config.get('split_config')
                })
        
        # Ana config dosyasını kaydet
        main_config_path = os.path.join(configs_dir, 'training_session_config.json')
        with open(main_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Kullanıcı tercihleri kaydedildi: {main_config_path}")
        
        # Özet dosyası oluştur (okunabilir format)
        summary_path = os.path.join(configs_dir, 'session_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"SmartFarm Eğitim Oturumu Özeti\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Oturum Bilgileri:\n")
            f.write(f"  • Timestamp: {global_ts}\n")
            f.write(f"  • Oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  • Ortam: {'Google Colab' if is_colab() else 'Yerel'}\n")
            f.write(f"  • Dil: {config_data['session_info']['language']}\n\n")
            
            f.write(f"Eğitim Ayarları:\n")
            f.write(f"  • Model: {options.get('model')}\n")
            f.write(f"  • Epoch: {options.get('epochs')}\n")
            f.write(f"  • Batch Boyutu: {options.get('batch')}\n")
            f.write(f"  • Görüntü Boyutu: {options.get('imgsz')}\n")
            f.write(f"  • Cihaz: {options.get('device')}\n")
            f.write(f"  • Hız Modu: {'Açık' if options.get('speed_mode') else 'Kapalı'}\n")
            f.write(f"  • Kategori: {options.get('category')}\n\n")
            
            if dataset_config:
                f.write(f"Veri Seti Konfigürasyonu:\n")
                f.write(f"  • Tip: {dataset_config.get('type')}\n")
                if dataset_config['type'] == 'hierarchical_multi':
                    setup = dataset_config.get('setup', {})
                    f.write(f"  • Seçilen Grup: {setup.get('selected_group')}\n")
                    f.write(f"  • Sınıf Başına Hedef: {setup.get('target_count')}\n")
                    f.write(f"  • Etiket Modu: {setup.get('label_mode')}\n")
                f.write(f"\n")
            
            if options.get('drive_save_path'):
                f.write(f"Google Drive Ayarları:\n")
                f.write(f"  • Kaydetme: Açık\n")
                f.write(f"  • Yol: {options.get('drive_save_path')}\n")
                f.write(f"  • Kaydetme Aralığı: {options.get('save_interval', 'Varsayılan')} epoch\n\n")
            
            if augmentation_settings:
                f.write(f"Augmentation Ayarları:\n")
                for key, value in augmentation_settings.items():
                    f.write(f"  • {key}: {value}\n")
        
        print(f"📄 Oturum özeti oluşturuldu: {summary_path}")
        
        return configs_dir
        
    except Exception as e:
        print(f"⚠️ Kullanıcı tercihleri kaydedilemedi: {e}")
        return None

def set_global_timestamp(timestamp):
    """Set global timestamp (used when user chooses existing timestamp)"""
    global _GLOBAL_TIMESTAMP
    _GLOBAL_TIMESTAMP = timestamp

def save_augmentation_config(configs_dir, augmentation_settings):
    """Augmentation ayarlarını ayrı dosyada kaydet"""
    try:
        if not configs_dir or not augmentation_settings:
            return
        
        aug_config_path = os.path.join(configs_dir, 'augmentation_config.json')
        with open(aug_config_path, 'w', encoding='utf-8') as f:
            json.dump(augmentation_settings, f, indent=2, ensure_ascii=False)
        
        print(f"🎨 Augmentation ayarları kaydedildi: {aug_config_path}")
        
    except Exception as e:
        print(f"⚠️ Augmentation ayarları kaydedilemedi: {e}")

def collect_augmentation_settings(dataset_config):
    """Kullanıcının augmentation tercihlerini topla"""
    augmentation_settings = {
        'enabled': False,
        'target_completion': False,
        'settings': {}
    }
    
    try:
        # Hierarchical multi-dataset için augmentation ayarları
        if dataset_config and dataset_config.get('type') == 'hierarchical_multi':
            setup = dataset_config.get('setup', {})
            
            # Hedef tamamlama augmentation bilgileri
            if setup.get('target_count'):
                augmentation_settings.update({
                    'enabled': True,
                    'target_completion': True,
                    'settings': {
                        'target_count_per_class': setup.get('target_count'),
                        'per_class_targets': setup.get('per_class_targets'),
                        'copy_val_test': True,  # Varsayılan
                        'image_size': setup.get('settings', {}).get('default_image_size', 640)
                    }
                })
        
        # Environment'dan augmentation bilgilerini al
        env_aug_settings = os.environ.get('SMARTFARM_AUG_SETTINGS')
        if env_aug_settings:
            try:
                env_settings = json.loads(env_aug_settings)
                augmentation_settings['settings'].update(env_settings)
            except:
                pass
        
    except Exception as e:
        print(f"⚠️ Augmentation ayarları toplanırken hata: {e}")
    
    return augmentation_settings

def find_existing_timestamps(base_dir):
    """Find existing timestamp directories in base directory"""
    if not os.path.exists(base_dir):
        return []
    
    timestamps = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and TIMESTAMP_PATTERN.match(item):
            timestamps.append({
                'name': item,
                'path': item_path,
                'mtime': os.path.getmtime(item_path)
            })
    
    # Sort by creation time (oldest first)
    timestamps.sort(key=lambda x: x['mtime'])
    return timestamps

def ask_user_for_timestamp_choice(language_choice='tr'):
    """Ask user whether to use existing timestamp or create new one"""
    # Check common directories for existing timestamps
    check_dirs = []
    
    # Add Google Drive paths if in Colab
    if is_colab():
        drive_paths = get_tarim_drive_paths()
        if drive_paths:
            check_dirs.extend([
                drive_paths['yolo11_models'],
                os.path.join(drive_paths['base'], 'yolo11_models')
            ])
        
        smartfarm_dir = get_smartfarm_models_dir()
        if smartfarm_dir:
            check_dirs.append(smartfarm_dir)
    
    # Add local directories
    check_dirs.extend([
        'yolo11_models',
        'runs/train',
        'datasets',
        'checkpoints'
    ])
    
    # Find all existing timestamps
    all_timestamps = []
    for check_dir in check_dirs:
        timestamps = find_existing_timestamps(check_dir)
        for ts in timestamps:
            ts['source_dir'] = check_dir
            all_timestamps.append(ts)
    
    # Remove duplicates (same timestamp name)
    unique_timestamps = {}
    for ts in all_timestamps:
        if ts['name'] not in unique_timestamps or ts['mtime'] < unique_timestamps[ts['name']]['mtime']:
            unique_timestamps[ts['name']] = ts
    
    timestamps_list = list(unique_timestamps.values())
    timestamps_list.sort(key=lambda x: x['mtime'])  # Oldest first
    
    if not timestamps_list:
        # No existing timestamps found
        new_timestamp = get_global_timestamp()
        if language_choice.startswith('tr'):
            print(f"🕒 Yeni timestamp oluşturuldu: {new_timestamp}")
        else:
            print(f"🕒 New timestamp created: {new_timestamp}")
        return new_timestamp
    
    # Show existing timestamps to user
    if language_choice.startswith('tr'):
        print(f"\n🕒 Mevcut timestamp(ler) algılandı:")
        for i, ts in enumerate(timestamps_list, 1):
            print(f"  {i}. {ts['name']} (kaynak: {ts['source_dir']})")
        
        print(f"\nSeçenekler:")
        print(f"  e) Mevcut timestamp kullan (en eski: {timestamps_list[0]['name']})")
        print(f"  y) Yeni timestamp oluştur")
        
        choice = input(f"\nTercihiniz (e/y, varsayılan: e): ").strip().lower() or 'e'
    else:
        print(f"\n🕒 Existing timestamp(s) detected:")
        for i, ts in enumerate(timestamps_list, 1):
            print(f"  {i}. {ts['name']} (source: {ts['source_dir']})")
        
        print(f"\nOptions:")
        print(f"  e) Use existing timestamp (oldest: {timestamps_list[0]['name']})")
        print(f"  n) Create new timestamp")
        
        choice = input(f"\nYour choice (e/n, default: e): ").strip().lower() or 'e'
    
    if choice.startswith('e'):
        # Use oldest existing timestamp
        selected_timestamp = timestamps_list[0]['name']
        set_global_timestamp(selected_timestamp)
        if language_choice.startswith('tr'):
            print(f"✅ Mevcut timestamp kullanılacak: {selected_timestamp}")
        else:
            print(f"✅ Using existing timestamp: {selected_timestamp}")
        return selected_timestamp
    else:
        # Create new timestamp
        new_timestamp = get_global_timestamp()
        if language_choice.startswith('tr'):
            print(f"✅ Yeni timestamp oluşturuldu: {new_timestamp}")
        else:
            print(f"✅ New timestamp created: {new_timestamp}")
        return new_timestamp

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
        print("ℹ️  Bu fonksiyon sadece Google Colab'da çalışır.")
        return False
    
    # Check if Google Drive is mounted
    if not os.path.exists('/content/drive'):
        if not mount_google_drive():
            return False
    
    # Find the most recent training directory
    runs_dir = "runs/train"
    if not os.path.exists(runs_dir):
        print(f"❌ Eğitim dizini bulunamadı: {runs_dir}")
        return False
    
    # Get the latest experiment directory
    exp_dirs = [d for d in os.listdir(runs_dir) if d.startswith('exp')]
    if not exp_dirs:
        print(f"❌ Hiçbir eğitim denemesi bulunamadı: {runs_dir}")
        return False
    
    # Sort to get the latest (exp, exp2, exp3, etc.)
    exp_dirs.sort(key=lambda x: int(x[3:]) if x[3:].isdigit() else 0)
    latest_exp = exp_dirs[-1]
    source_dir = os.path.join(runs_dir, latest_exp, "weights")
    
    if not os.path.exists(source_dir):
        print(f"❌ Ağırlık dizini bulunamadı: {source_dir}")
        return False
    
    # Create target directory
    try:
        os.makedirs(drive_folder_path, exist_ok=True)
        print(f"📁 Hedef dizin oluşturuldu: {drive_folder_path}")
    except Exception as e:
        print(f"❌ Hedef dizin oluşturulamadı: {e}")
        return False
    
    # Copy files
    copied_files = []
    
    # Copy best.pt
    if best_file:
        best_path = os.path.join(source_dir, "best.pt")
        if os.path.exists(best_path):
            try:
                target_best = os.path.join(drive_folder_path, "best.pt")
                shutil.copy2(best_path, target_best)
                copied_files.append("best.pt")
                print(f"✅ best.pt kopyalandı: {target_best}")
            except Exception as e:
                print(f"❌ best.pt kopyalanamadı: {e}")
        else:
            print(f"⚠️  best.pt bulunamadı: {best_path}")
    
    # Copy last.pt
    if last_file:
        last_path = os.path.join(source_dir, "last.pt")
        if os.path.exists(last_path):
            try:
                target_last = os.path.join(drive_folder_path, "last.pt")
                shutil.copy2(last_path, target_last)
                copied_files.append("last.pt")
                print(f"✅ last.pt kopyalandı: {target_last}")
            except Exception as e:
                print(f"❌ last.pt kopyalanamadı: {e}")
        else:
            print(f"⚠️  last.pt bulunamadı: {last_path}")
    
    # Copy additional files from project root
    additional_files = ["merged_dataset.yaml", "unified_class_mapping.json", "analysis_report.json"]
    for file_name in additional_files:
        if os.path.exists(file_name):
            try:
                target_file = os.path.join(drive_folder_path, file_name)
                shutil.copy2(file_name, target_file)
                copied_files.append(file_name)
                print(f"✅ {file_name} kopyalandı: {target_file}")
            except Exception as e:
                print(f"❌ {file_name} kopyalanamadı: {e}")
    
    # Copy training results and plots if available
    results_dir = os.path.join(runs_dir, latest_exp)
    result_files = ["results.png", "confusion_matrix.png", "F1_curve.png", "P_curve.png", "R_curve.png"]
    for file_name in result_files:
        result_path = os.path.join(results_dir, file_name)
        if os.path.exists(result_path):
            try:
                target_file = os.path.join(drive_folder_path, file_name)
                shutil.copy2(result_path, target_file)
                copied_files.append(file_name)
                print(f"✅ {file_name} kopyalandı: {target_file}")
            except Exception as e:
                print(f"❌ {file_name} kopyalanamadı: {e}")
    
    if copied_files:
        print(f"\n✅ Google Drive'a kaydedilen dosyalar: {', '.join(copied_files)}")
        print(f"📁 Kaydetme konumu: {drive_folder_path}")
        print(f"🗂️  Toplam kaydedilen dosya: {len(copied_files)}")
        return True
    else:
        print("❌ Kopyalanacak dosya bulunamadı.")
        return False

# --- Yardımcı: master sınıf isimlerini yükleyip class_ids.json yaz ---
def _load_names_from_yaml(yaml_path: str):
    try:
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            names = []
            if isinstance(data.get('names'), list):
                names = data['names']
            elif isinstance(data.get('names'), dict):
                names = [v for k, v in sorted(((int(k), v) for k, v in data['names'].items()), key=lambda x: x[0])]
            elif isinstance(data.get('classes'), list):
                names = data['classes']
            return [str(x) for x in names] if names else None
    except Exception:
        return None
    return None

def _write_class_ids_json(configs_dir: str) -> bool:
    try:
        # Öncelik master_data.yaml, yoksa merged_dataset.yaml
        names = _load_names_from_yaml('master_data.yaml') or _load_names_from_yaml('merged_dataset.yaml')
        if not names:
            return False
        payload = {
            'generated_at': datetime.now().isoformat(),
            'names': names,
            'id_to_name': [{'id': i, 'name': n} for i, n in enumerate(names)],
        }
        os.makedirs(configs_dir, exist_ok=True)
        out_path = os.path.join(configs_dir, 'class_ids.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"✅ class_ids.json yazıldı: {out_path}")
        return True
    except Exception as e:
        print(f"⚠️ class_ids.json yazılamadı: {e}")
        return False

def download_models_menu():
    """Interactive menu for downloading YOLO11 models"""
    print(f"\n{get_text('model_download_title')}")
    
    if is_colab():
        default_dir = get_smartfarm_models_dir() or "/content/colab_learn/yolo11_models"
    else:
        default_dir = "yolo11_models"
    
    # Global timestamp ile model dizini oluştur
    base_save_dir = input(get_text('save_directory', default=default_dir)) or default_dir
    
    # Global timestamp'i kullanarak alt klasör oluştur
    global_ts = get_global_timestamp()
    save_dir = os.path.join(base_save_dir, global_ts)
    
    print(f"🌐 Modeller global timestamp klasörüne indirilecek: {save_dir}")
    
    # Klasörü oluştur
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{get_text('download_options')}")
    print(get_text('single_model'))
    print(get_text('all_detection'))
    print(get_text('all_models'))
    
    choice = input(f"\n{get_text('your_choice')}") or "2"
    
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
            print(f"📁 Global timestamp klasörü: {os.path.basename(save_dir)}")
    
    elif choice == "2":
        detection_models = ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        downloaded = download_yolo11_models(save_dir, detection_models)
        print(f"\n✅ {len(downloaded)} tespit modeli indirildi: {save_dir}")
        print(f"📁 Global timestamp klasörü: {os.path.basename(save_dir)}")
    
    elif choice == "3":
        downloaded = download_yolo11_models(save_dir)
        print(f"\n✅ {len(downloaded)} model indirildi: {save_dir}")
        print(f"📁 Global timestamp klasörü: {os.path.basename(save_dir)}")
    
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
    
    # Opsiyonel: Roboflow API key girişi (boş bırakılabilir)
    try:
        print("\n🔑 Roboflow API (opsiyonel)")
        entered_key = input("API key girin (boş geçebilirsiniz): ").strip()
        if entered_key:
            manager.api_key = entered_key
            # API key girildiyse, kullanıcıya split ayarını da soralım
            split_cfg = get_dataset_split_config(entered_key)
            if split_cfg:
                manager.split_config = split_cfg
        else:
            # Boşsa, varsa config'den otomatik kullanılacak (indirme sırasında fallback var)
            manager.api_key = None
            manager.split_config = None
    except Exception:
        # Sessiz geç
        pass

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
                val = input(f"  • {cls} için hedef (varsayılan {target_count}): ") or str(target_count)
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

        # Etiket Yeniden Eşleme Modu — İNDİRME SONRASI taşındı
        print("\nEtiket Yeniden Eşleme Modu:")
        print("1) Merge aşamasında alt-sınıf etiketleri KORUNMAZ; tüm kutular ANA sınıfa toplanır (varsayılan)")
        print("2) Merge aşamasında alt-sınıf etiketleri KORUNUR; tüm kutular ana sınıfa toplanmaz")
        while True:
            label_mode_choice = (input("Seçenek [1-2] (varsayılan: 1): ") or "1").strip()
            if label_mode_choice in ["1", "2"]:
                break
            print("❌ Lütfen 1 veya 2 giriniz.")
        dataset_config['label_mode'] = "collapse_to_main" if label_mode_choice == "1" else "preserve_subclasses"

        # 2. Create unified class mapping
        print("\n2️⃣ Hiyerarşik sınıf haritalaması oluşturuluyor...")
        classes_created = manager.create_unified_class_mapping()
        
        if classes_created == 0:
            print("❌ Hiçbir sınıf haritalandırılamadı!")
            return False
        
        print(f"✅ {classes_created} ana sınıf oluşturuldu")

        # 3. Label mode yönlendirmesi
        label_mode = dataset_config.get('label_mode') or dataset_config.get('setup', {}).get('label_mode')
        if label_mode == 'preserve_subclasses':
            print("\n⚙️ Seçenek 2: Alt-sınıf etiketleri KORUNACAK. Global remap otomatik başlatılıyor...")
            try:
                cmd = [
                    sys.executable,
                    str(Path("tools") / "remap_from_all_yaml.py"),
                    "--root", "datasets",
                    "--force-backup",
                ]
                print(f"[APPLY] Remap komutu: {' '.join(cmd)}")
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.stdout:
                    print(proc.stdout)
                if proc.returncode != 0:
                    print(f"[UYARI] Remap sırasında hata oluştu (kod={proc.returncode}). Devam ediliyor...\n{proc.stderr}")
                else:
                    print("✅ Etiket remap işlemi tamamlandı.")
            except Exception as e:
                print(f"[UYARI] Remap çağrısı başarısız: {e}. İşleme devam edilecek.")

        # 4. Merge step: only for option 1 (collapse_to_main)
        if label_mode == 'preserve_subclasses':
            # Kopyalama yapmadan çoklu kaynak YAML üret
            try:
                def _resolve_split_image_dirs(local_root: str, split: str):
                    """Split için olası görüntü klasörlerini tespit et."""
                    cand = []
                    data_yaml_path = os.path.join(local_root, 'data.yaml')
                    if os.path.exists(data_yaml_path):
                        try:
                            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                                dcfg = yaml.safe_load(f) or {}
                            base_path = dcfg.get('path') or '.'
                            entry = dcfg.get('train' if split == 'train' else ('val' if 'val' in split else split))
                            # val/valid eş anlamlı
                            if not entry and split == 'val':
                                entry = dcfg.get('valid')
                            if entry:
                                def _join(root_dir, p):
                                    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root_dir, p))
                                full = _join(local_root, _join('' if base_path == '.' else base_path, entry))
                                cand.append(full)
                                # images -> ensure it's images dir
                                if os.path.basename(full) != 'images' and os.path.isdir(full):
                                    # kullanıcı doğrudan images klasörünü değil üst klasörü vermiş olabilir
                                    cand.append(os.path.join(full, 'images'))
                        except Exception:
                            pass
                    # Yaygın dizinler
                    synonyms = [split]
                    if split == 'val':
                        synonyms.append('valid')
                    if split == 'valid':
                        synonyms.append('val')
                    for s in synonyms:
                        cand.extend([
                            os.path.join(local_root, 'images', s),
                            os.path.join(local_root, s, 'images'),
                        ])
                    # Var olanları filtrele
                    uniq = []
                    seen = set()
                    for p in cand:
                        if p and os.path.isdir(p) and p not in seen:
                            uniq.append(os.path.abspath(p))
                            seen.add(p)
                    return uniq

                # Label klasörlerini bulmak için yardımcı
                def _resolve_split_label_dirs(local_root: str, split: str):
                    cand = []
                    data_yaml_path = os.path.join(local_root, 'data.yaml')
                    if os.path.exists(data_yaml_path):
                        try:
                            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                                dcfg = yaml.safe_load(f) or {}
                            base_path = dcfg.get('path') or '.'
                            entry = dcfg.get('train' if split == 'train' else ('val' if 'val' in split else split))
                            if not entry and split == 'val':
                                entry = dcfg.get('valid')
                            if entry:
                                def _join(root_dir, p):
                                    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root_dir, p))
                                full = _join(local_root, _join('' if base_path == '.' else base_path, entry))
                                if os.path.isdir(full):
                                    # images/... kalıbından labels/... tahmini
                                    cand.append(full.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep))
                                    cand.append(os.path.join(os.path.dirname(full), 'labels'))
                        except Exception:
                            pass
                    synonyms = [split]
                    if split == 'val':
                        synonyms.append('valid')
                    if split == 'valid':
                        synonyms.append('val')
                    for s in synonyms:
                        cand.extend([
                            os.path.join(local_root, 'labels', s),
                            os.path.join(local_root, s, 'labels'),
                        ])
                    uniq = []
                    seen = set()
                    for p in cand:
                        if p and os.path.isdir(p) and p not in seen:
                            uniq.append(os.path.abspath(p))
                            seen.add(p)
                    return uniq

                # Tüm veri setlerinden train/val klasörlerini topla
                train_dirs = []
                val_dirs = []
                for ds in manager.datasets:
                    root = ds.get('local_path')
                    if not root or not os.path.isdir(root):
                        continue
                    train_dirs.extend(_resolve_split_image_dirs(root, 'train'))
                    # val bulunamazsa valid dene
                    vds = _resolve_split_image_dirs(root, 'val')
                    if not vds:
                        vds = _resolve_split_image_dirs(root, 'valid')
                    # Hiç val yoksa train'i yedek olarak kullanma — eğitim için val gerekli, ama yoksa boş bırakma yerine train'den küçük bir altküme oluşturmanızı öneririz.
                    val_dirs.extend(vds)

                # names yükle
                def _load_master_names():
                    for p in ['config/master_data.yaml', 'master_data.yaml']:
                        if os.path.exists(p):
                            try:
                                with open(p, 'r', encoding='utf-8') as f:
                                    data = yaml.safe_load(f) or {}
                                names = data.get('names') or data.get('classes')
                                if isinstance(names, list) and names:
                                    return [str(x) for x in names]
                            except Exception:
                                pass
                    # Fallback: indirilen veri setlerinden birlik
                    union = []
                    seen = set()
                    for ds in manager.datasets:
                        for c in ds.get('classes', []) or []:
                            if c not in seen:
                                union.append(c)
                                seen.add(c)
                    return union

                # İsimleri öncelikle config/class_ids.json'dan al
                names = None
                try:
                    cid_path = os.path.join('config', 'class_ids.json')
                    if os.path.exists(cid_path):
                        with open(cid_path, 'r', encoding='utf-8') as f:
                            cid = json.load(f)
                        if isinstance(cid.get('names'), list) and cid['names']:
                            names = [str(x) for x in cid['names']]
                except Exception:
                    pass
                if not names:
                    names = _load_master_names()
                if not train_dirs:
                    print("❌ Hiçbir train dizini bulunamadı. YAML oluşturulamadı.")
                    return False
                if not val_dirs:
                    print("⚠️ Val/valid dizini bulunamadı. Train dizinleri kullanılacak (değerlendirme için önerilmez).")
                    val_dirs = train_dirs[:]

                dataset_yaml = {
                    'path': '.',
                    'train': train_dirs,
                    'val': val_dirs,
                    'nc': len(names),
                    'names': names,
                }
                os.makedirs('config', exist_ok=True)
                merged_yaml_path = os.path.join('config', 'merged_class_dataset.yaml')
                with open(merged_yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)
                print("📄 config/merged_class_dataset.yaml oluşturuldu (kopyasız çoklu kaynak).")
                print(f"  • train kaynak sayısı: {len(train_dirs)}")
                print(f"  • val kaynak sayısı: {len(val_dirs)}")
            except Exception as e:
                print(f"❌ merged_dataset.yaml oluşturulurken hata: {e}")
                return False
            # Merge kopyalama adımı atlanır fakat eğitim için YAML hazır.
            print("\nℹ️ Seçenek 2 için kopyalama yapmadan eğitim YAML hazırlandı (config/merged_class_dataset.yaml).")

            # Opsiyonel: Hedefe TAMAMLAMA augmentation (yalnızca eksik kadar üret)
            try:
                if _AUG_PIPE_AVAILABLE:
                    # e/h girdisi için doğrulama döngüsü
                    while True:
                        raw = input("\nSınıf başına hedefe TAMAMLAMA için augmentation uygulansın mı? (e/h, varsayılan: h): ")
                        resp = (raw or 'h').strip().lower()
                        if resp in ('e', 'h'):
                            break
                        print("Geçersiz giriş. Lütfen 'e' veya 'h' girin.")
                    if resp == 'e':
                        # Hedef belirleme: sayısal doğrulama döngüsü
                        target_default = dataset_config.get('target_count')
                        try:
                            target_default = int(target_default)
                        except Exception:
                            target_default = 2000
                        while True:
                            raw_t = input(f"Hedef sınıf başına örnek (boş bırak: {target_default}): ")
                            if not raw_t or not raw_t.strip():
                                target_numeric = target_default
                                break
                            try:
                                target_numeric = int(raw_t.strip())
                                if target_numeric <= 0:
                                    print("Geçersiz değer. Pozitif bir tam sayı girin.")
                                    continue
                                break
                            except ValueError:
                                print("Geçersiz değer. Lütfen bir tam sayı girin.")

                        # Train label ve image dosyalarını topla
                        image_dirs = train_dirs[:]
                        label_dirs = []
                        for ds in manager.datasets:
                            root = ds.get('local_path')
                            if not root or not os.path.isdir(root):
                                continue
                            label_dirs.extend(_resolve_split_label_dirs(root, 'train'))

                        def _iter_label_files(dirs):
                            for d in dirs:
                                for root_dir, _, files in os.walk(d):
                                    for fn in files:
                                        if fn.lower().endswith('.txt'):
                                            yield os.path.join(root_dir, fn)

                        def _match_image_for_label(label_path, img_dirs):
                            base = os.path.splitext(os.path.basename(label_path))[0]
                            for d in img_dirs:
                                for ext in ['.jpg', '.jpeg', '.png']:
                                    cand = os.path.join(d, base + ext)
                                    if os.path.exists(cand):
                                        return cand
                            return None

                        all_label_paths = []
                        all_image_paths = []
                        for lp in _iter_label_files(label_dirs):
                            ip = _match_image_for_label(lp, image_dirs)
                            if ip:
                                all_label_paths.append(lp)
                                all_image_paths.append(ip)

                        if not all_label_paths:
                            print("⚠️  Augmentation için train etiket dosyası bulunamadı. Adım atlandı.")
                        else:
                            out_dir = os.path.join('datasets', 'balanced_aug')
                            os.makedirs(out_dir, exist_ok=True)
                            print(f"\n⚙️  Hedefe-tamamlama başlıyor. Hedef: {target_numeric}  Çıkış: {out_dir}")
                            pipe = YOLOAugmentationPipeline(image_size=dataset_config.get('settings', {}).get('default_image_size', 640))
                            
                            # Resolve VAL/TEST image/label directories from all datasets
                            val_img_dirs = []
                            val_lbl_dirs = []
                            test_img_dirs = []
                            test_lbl_dirs = []
                            for ds in manager.datasets:
                                root = ds.get('local_path')
                                if not root or not os.path.isdir(root):
                                    continue
                                # val/valid images
                                vids = _resolve_split_image_dirs(root, 'val')
                                if not vids:
                                    vids = _resolve_split_image_dirs(root, 'valid')
                                val_img_dirs.extend(vids)
                                # val/valid labels
                                vlds = _resolve_split_label_dirs(root, 'val')
                                if not vlds:
                                    vlds = _resolve_split_label_dirs(root, 'valid')
                                val_lbl_dirs.extend(vlds)
                                # test images/labels
                                test_img_dirs.extend(_resolve_split_image_dirs(root, 'test'))
                                test_lbl_dirs.extend(_resolve_split_label_dirs(root, 'test'))

                            # Decide automatically (no prompt): copy val/test based on config defaults
                            default_copy_val_test = None
                            try:
                                default_copy_val_test = dataset_config.get('settings', {}).get('copy_val_test_default')
                                if default_copy_val_test is not None:
                                    default_copy_val_test = bool(default_copy_val_test)
                            except Exception:
                                default_copy_val_test = None
                            if default_copy_val_test is None:
                                try:
                                    with open(os.path.join('config', 'master_data.yaml'), 'r', encoding='utf-8') as f:
                                        _md = yaml.safe_load(f) or {}
                                    default_copy_val_test = bool(_md.get('copy_val_test_default', True))
                                except Exception:
                                    default_copy_val_test = True
                            copy_val_test = bool(default_copy_val_test)

                            pipe.augment_dataset_batch(
                                all_image_paths,
                                all_label_paths,
                                out_dir,
                                target_numeric,
                                copy_val_test=copy_val_test,
                                val_image_dirs=val_img_dirs,
                                val_label_dirs=val_lbl_dirs,
                                test_image_dirs=test_img_dirs,
                                test_label_dirs=test_lbl_dirs,
                            )

                            # Create config/augmented_train.yaml to reference augmented train and original or copied val/test
                            try:
                                # Determine YAML val/test paths
                                if copy_val_test:
                                    yaml_val = os.path.join(out_dir, 'val', 'images')
                                    yaml_test = os.path.join(out_dir, 'test', 'images') if test_img_dirs or test_lbl_dirs else None
                                else:
                                    # Use original directories (list is supported)
                                    yaml_val = val_img_dirs if val_img_dirs else None
                                    yaml_test = test_img_dirs if test_img_dirs else None

                                if not names:
                                    # Try to reload names from config/class_ids.json as fallback
                                    try:
                                        with open(os.path.join('config', 'class_ids.json'), 'r', encoding='utf-8') as f:
                                            cid = json.load(f)
                                        if isinstance(cid.get('names'), list) and cid['names']:
                                            local_names = [str(x) for x in cid['names']]
                                        else:
                                            local_names = []
                                    except Exception:
                                        local_names = []
                                else:
                                    local_names = names

                                yaml_payload = {
                                    'path': '.',
                                    'train': os.path.join(out_dir, 'train', 'images'),
                                    'val': yaml_val if yaml_val is not None else [],
                                    'test': yaml_test if yaml_test is not None else [],
                                    'names': local_names,
                                }
                                os.makedirs('config', exist_ok=True)
                                aug_yaml_path = os.path.join('config', 'augmented_train.yaml')
                                with open(aug_yaml_path, 'w', encoding='utf-8') as f:
                                    yaml.dump(yaml_payload, f, sort_keys=False, allow_unicode=True)
                                print(f"\n📄 Eğitim YAML yazıldı: {aug_yaml_path}")
                                print(f"  • train: {yaml_payload['train']}")
                                print(f"  • val:   {yaml_payload['val'] if yaml_payload['val'] else '—'}")
                                print(f"  • test:  {yaml_payload['test'] if yaml_payload['test'] else '—'}")

                                # Optional: Ask user to enable time-based Drive copy for training (default: disabled)
                                try:
                                    cfg_path = 'config_datasets.yaml'
                                    def_minutes = 30
                                    if os.path.exists(cfg_path):
                                        with open(cfg_path, 'r', encoding='utf-8') as _cf:
                                            _cfg = yaml.safe_load(_cf) or {}
                                        gs = _cfg.get('global_settings', {}) if isinstance(_cfg, dict) else {}
                                        def_minutes = int(gs.get('time_based_copy_interval_minutes', 30))
                                    yn = (input("\nSüreye bağlı Drive kopyalama açılsın mı? (e/h, varsayılan: h): ") or 'h').strip().lower()
                                    use_time_based = yn.startswith('e')
                                    minutes = def_minutes
                                    if use_time_based:
                                        min_in = input(f"Kopyalama aralığı (dakika, varsayılan: {def_minutes}): ").strip()
                                        minutes = int(min_in) if min_in else def_minutes
                                    # Persist selection into config_datasets.yaml
                                    try:
                                        _cfg = {}
                                        if os.path.exists(cfg_path):
                                            with open(cfg_path, 'r', encoding='utf-8') as _cf:
                                                _cfg = yaml.safe_load(_cf) or {}
                                        if not isinstance(_cfg, dict):
                                            _cfg = {}
                                        _gs = _cfg.get('global_settings')
                                        if not isinstance(_gs, dict):
                                            _gs = {}
                                        _gs['use_time_based_copy_default'] = bool(use_time_based)
                                        _gs['time_based_copy_interval_minutes'] = int(minutes)
                                        _cfg['global_settings'] = _gs
                                        with open(cfg_path, 'w', encoding='utf-8') as _cf:
                                            yaml.dump(_cfg, _cf, sort_keys=False, allow_unicode=True)
                                        state_txt = 'AÇIK' if use_time_based else 'KAPALI'
                                        print(f"✅ Süreye bağlı kopyalama ayarı güncellendi: {state_txt}, {minutes} dk")
                                        # Ayrıca kullanıcı seçimini configs/ klasörüne de kaydet
                                        try:
                                            os.makedirs('configs', exist_ok=True)
                                            from datetime import datetime as _dt
                                            ts = _dt.now().strftime('%Y%m%d_%H%M%S')
                                            snapshot = {
                                                'timestamp': ts,
                                                'use_time_based_copy_default': bool(use_time_based),
                                                'time_based_copy_interval_minutes': int(minutes),
                                                'augmented_train_yaml': aug_yaml_path,
                                            }
                                            snap_path = os.path.join('configs', f'time_copy_selection_{ts}.yaml')
                                            with open(snap_path, 'w', encoding='utf-8') as _sf:
                                                yaml.dump(snapshot, _sf, sort_keys=False, allow_unicode=True)
                                            print(f"📝 Seçim kopyası kaydedildi: {snap_path}")
                                        except Exception as _snap_e:
                                            print(f"⚠️ Seçim kopyası kaydedilemedi: {_snap_e}")
                                    except Exception as _werr:
                                        print(f"⚠️ Süreye bağlı kopyalama ayarı kaydedilemedi: {_werr}")
                                except Exception as _tbc_err:
                                    print(f"⚠️ Süreye bağlı kopyalama ayarı sorulamadı: {_tbc_err}")
                            except Exception as _yaml_err:
                                print(f"⚠️ augmented_train.yaml oluşturulamadı: {_yaml_err}")
                            print("✅ Hedefe-tamamlama augmentation tamamlandı.")
                else:
                    print("⚠️  YOLOAugmentationPipeline mevcut değil. Hedefe-tamamlama atlandı.")
            except Exception as e:
                print(f"[UYARI] Hedefe-tamamlama sırasında hata: {e}")

            return True
        
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
        # Single dataset (legacy) - Roboflow API yönetimi ile
        roboflow_url = input("\nRoboflow URL (varsayılan: boş): ").strip() or ""
        if not roboflow_url:
            print("❌ URL sağlanmadı")
            return None
        
        # Roboflow API key yönetimi
        api_result = handle_roboflow_api_management(roboflow_url)
        
        dataset_config = {
            'type': 'single',
            'url': roboflow_url,
            'api_key': api_result['api_key'],
            'split_config': api_result['split_config'],
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
    
    # Google Drive save settings (daha erken sorulsun)
    drive_save_path = None
    if is_colab():
        print("\nGoogle Drive kaydetme ayarları:")
        save_to_drive_opt = input("Eğitim sonuçlarını Google Drive'a kaydet? (e/h, varsayılan: e): ").lower() or "e"
        
        if save_to_drive_opt.startswith("e"):
            default_drive_path = get_smartfarm_models_dir() or "/content/drive/MyDrive/SmartFarm/colab_learn/yolo11_models"
            base_input = input(f"Kaydetme dizini (varsayılan: {default_drive_path}): ") or default_drive_path

            # Global timestamp'i kullan
            timestamp_dir = None
            try:
                # 1) Global timestamp varsa onu kullan
                global_ts = get_global_timestamp()
                if global_ts:
                    timestamp_dir = os.path.join(base_input, global_ts)
                    print(f"🌐 Global timestamp kullanılıyor: {global_ts}")
                else:
                    # 2) Fallback: DriveManager'da aktif timestamp var mı?
                    if _DRIVE_AVAILABLE:
                        dm_probe = DriveManager()
                        if dm_probe.authenticate():
                            try:
                                if hasattr(dm_probe, 'load_drive_config'):
                                    dm_probe.load_drive_config()
                            except Exception:
                                pass
                            ts_existing = dm_probe.get_timestamp_dir()
                            if ts_existing and os.path.basename(os.path.dirname(ts_existing)) == 'yolo11_models':
                                timestamp_dir = ts_existing
                    # 3) Base klasörde mevcut timestamp dizinlerini tara ve ILK OLUŞANINI al (ilk timestamp kuralı)
                    if not timestamp_dir and os.path.isdir(base_input):
                        candidates = [
                            os.path.join(base_input, d)
                            for d in os.listdir(base_input)
                            if os.path.isdir(os.path.join(base_input, d)) and TIMESTAMP_PATTERN.match(d)
                        ]
                        if candidates:
                            # mtime'a göre artan sırala: ilk eleman en eski (ilk oluşturulan)
                            candidates.sort(key=lambda p: os.path.getmtime(p))
                            timestamp_dir = candidates[0]
                            print(f"🕒 İlk timestamp kuralı: mevcutlardan EN ESKİSİ kullanılacak → {os.path.basename(timestamp_dir)}")
                    # 4) Hiçbiri yoksa global timestamp ile yeni oluştur
                    if not timestamp_dir:
                        timestamp = get_global_timestamp()
                        timestamp_dir = os.path.join(base_input, timestamp)
            except Exception:
                # Son çare: yeni timestamp oluştur
                timestamp = get_global_timestamp()
                timestamp_dir = os.path.join(base_input, timestamp)
            checkpoints_dir = os.path.join(timestamp_dir, 'checkpoints')
            models_dir = os.path.join(timestamp_dir, 'models')
            logs_dir = os.path.join(timestamp_dir, 'logs')
            configs_dir = os.path.join(timestamp_dir, 'configs')

            try:
                created_any = False
                for d in [checkpoints_dir, models_dir, logs_dir, configs_dir]:
                    if not os.path.isdir(d):
                        os.makedirs(d, exist_ok=True)
                        created_any = True
                action = "kullanılıyor" if not created_any else "hazırlandı"
                print(f"✅ Drive timestamp {action}: {timestamp_dir}")
                print(f"🗂️  Kayıt hedefi (checkpoints): {checkpoints_dir}")
                # Eğitim opsiyonlarında doğrudan 'checkpoints' klasörünü hedefle
                drive_save_path = checkpoints_dir
                # Etiket modu 2 ise: sınıf ID listesini configs/ altına yaz (varsa)
                try:
                    if (dataset_config.get('type') == 'hierarchical_multi' and
                        (dataset_config.get('setup') or {}).get('label_mode') == 'preserve_subclasses'):
                        _write_class_ids_json(configs_dir)
                except Exception:
                    pass
            except Exception as e:
                print(f"❌ Drive klasörleri oluşturulamadı: {e}")
                drive_save_path = None
    
    # Eğitim parametreleri (görüntü boyutu -> batch -> epoch)
    # Model size selection (tek soru)
    print("\nModel boyutunu seçin:")
    print("1) yolo11s.pt - Küçük (en hızlı, düşük doğruluk)")
    print("2) yolo11m.pt - Orta (dengeli)")
    print("3) yolo11l.pt - Büyük (yüksek doğruluk, yavaş) [Hiyerarşik için önerilen]")
    print("4) yolo11x.pt - Çok Büyük (en yüksek doğruluk, en yavaş)")

    while True:
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

    # Önce görüntü boyutu
    while True:
        try:
            img_size = int(input(f"\nGörüntü boyutu (varsayılan: {default_img_size}, 32'nin katı olmalı • Colab için 512 önerilir): ") or str(default_img_size))
            if img_size > 0 and img_size % 32 == 0:
                break
            print("❌ Lütfen 32'nin katı olan pozitif bir sayı girin.")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin.")

    # Sonra batch boyutu
    while True:
        try:
            batch_size = int(input(f"\nBatch boyutu (varsayılan: {default_batch}, düşük RAM için küçük): ") or str(default_batch))
            if batch_size > 0:
                break
            print("❌ Lütfen pozitif bir sayı girin.")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin.")

    # En son epoch
    while True:
        try:
            default_epochs = 1000
            epochs = int(input(f"\nEpoch sayısı [100-2000 önerilen] (varsayılan: {default_epochs}): ") or str(default_epochs))
            if epochs > 0:
                break
            print("❌ Lütfen pozitif bir sayı girin.")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı girin.")

    # Speed mode (optimize epoch time)
    speed_mode_input = (input("\nHız modu (cache=ram, workers=8, plots=False) açılsın mı? (e/h, varsayılan: e): ") or "e").lower()
    speed_mode = speed_mode_input.startswith('e')
    
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
        'workers': None,
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
    # --- Veri indirme başlamadan ÖNCE: Otomatik checkpoint araması ve kullanıcıya bilgi vererek seçim alma ---
    try:
        if is_colab() and _DRIVE_AVAILABLE:
            dm = DriveManager()
            if dm.authenticate():
                # Konfigürasyon varsa yükle, yoksa yine de arama yap (projeyi bilmeden de base dizinde arıyor)
                try:
                    if hasattr(dm, 'load_drive_config'):
                        dm.load_drive_config()
                except Exception:
                    pass
                print("\n🔍 Drive'da mevcut checkpoint aranıyor (en yeni timestamp'tan geriye doğru)...")
                ckpt_path, ckpt_name = dm.find_latest_checkpoint()
                if ckpt_path:
                    print(f"✅ Bulundu: {ckpt_name}\n📄 Yol: {ckpt_path}")
                    ask = (input("Kaldığı yerden devam edilsin mi? (e/h, varsayılan: e): ") or "e").lower()
                    if ask.startswith('e'):
                        options['resume'] = True
                        options['checkpoint_path'] = ckpt_path
                        print("🔄 Eğitim, veri indirme adımı atlanarak checkpoint'ten devam edecek.")
                    else:
                        print("ℹ️ Resume iptal edildi. Yeni eğitim kurulumu ile devam edilecek.")
                else:
                    print("ℹ️ Drive'da kullanılabilir checkpoint bulunamadı. Yeni eğitim kurulumu ile devam edilecek.")
            else:
                print("⚠️ Drive mount/kimlik doğrulama başarısız. Resume araması yapılamadı.")
        else:
            if not is_colab():
                print("ℹ️ Colab ortamı değil. Drive tabanlı otomatik resume araması atlandı.")
            elif not _DRIVE_AVAILABLE:
                print("⚠️ drive_manager içe aktarılamadı. Drive tabanlı otomatik resume araması yapılamadı.")
    except Exception as pre_resume_err:
        print(f"⚠️ Otomatik resume kontrolü sırasında hata: {pre_resume_err}")

    
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
    # Kaydetme aralığı sorusu training.py içinde (menülü) yönetiliyor
    
    if drive_save_path:
        print(f"Drive kaydetme yolu: {drive_save_path}")
        # Kaydedilecek dosyaları net belirt (checkpoints altında)
        print(f"Kaydedilecek dosyalar:")
        print(f"  • best.pt  → {os.path.join(drive_save_path, 'best.pt')}")
        print(f"  • last.pt  → {os.path.join(drive_save_path, 'last.pt')}")
    
    confirm = (input("\nBu parametrelerle devam et? (e/h, varsayılan: e): ") or "e").lower()
    if confirm != 'e' and confirm != 'evet' and confirm != 'yes':
        print("❌ Kurulum iptal edildi.")
        return None
    
    return options

def get_dataset_split_config(api_key):
    """API key varsa train/test/val değerlerini al"""
    if not api_key:
        return None
    
    print("\n📊 Dataset Bölümleme Ayarları")
    print("=" * 40)
    print("API key mevcut - dataset bölümleme ayarlarını yapılandırabilirsiniz")
    
    use_custom_split = input("\nÖzel train/test/val oranı kullanmak istiyor musunuz? (e/h, varsayılan: h): ").lower() or "h"
    
    if not use_custom_split.startswith('e'):
        print("✅ Varsayılan bölümleme kullanılacak")
        return None
    
    print("\n📋 Bölümleme Oranı Girişi:")
    print("Not: Toplam 100 olmalı (train + test + val = 100)")
    
    while True:
        try:
            train_pct = int(input("Train oranı (varsayılan: 70): ") or "70")
            test_pct = int(input("Test oranı (varsayılan: 20): ") or "20")
            val_pct = int(input("Validation oranı (varsayılan: 10): ") or "10")
            
            total = train_pct + test_pct + val_pct
            if total != 100:
                print(f"❌ Toplam {total}%. Lütfen toplamı 100 yapacak şekilde girin.")
                continue
            
            if train_pct < 50:
                print("⚠️ Train oranı %50'den az. Devam etmek istiyor musunuz? (e/h): ", end="")
                if not input().lower().startswith('e'):
                    continue
            
            split_config = {
                'train': train_pct,
                'test': test_pct, 
                'val': val_pct
            }
            
            print(f"\n✅ Bölümleme ayarları: Train %{train_pct}, Test %{test_pct}, Val %{val_pct}")
            return split_config
            
        except ValueError:
            print("❌ Lütfen geçerli sayılar girin")
            continue

def handle_roboflow_api_management(url):
    """Roboflow API key yönetimini handle et"""
    print("\n🔑 Roboflow API Yönetimi")
    print("=" * 40)
    
    # Mevcut API key kontrol et
    existing_key = get_api_key_from_config()
    if existing_key:
        print(f"✅ Mevcut API key bulundu: {existing_key[:10]}...")
        use_existing = input("Mevcut API key'i kullanmak istiyor musunuz? (e/h, varsayılan: e): ").lower() or "e"
        if use_existing.startswith('e'):
            # API key varsa split config al
            split_config = get_dataset_split_config(existing_key)
            return {'api_key': existing_key, 'split_config': split_config}
    
    print("\n📋 Seçenekler:")
    print("1) API Key gir (train/test/val ayarları ile)")
    print("2) API Key olmadan devam et (public dataset)")
    
    while True:
        choice = input("\nSeçenek [1-2] (varsayılan: 2): ").strip() or "2"
        
        if choice == "2":
            print("✅ API key olmadan devam ediliyor (public dataset olarak)")
            return {'api_key': None, 'split_config': None}
        
        elif choice == "1":
            print("\n📋 API Key alma adımları:")
            print("1. https://roboflow.com adresine gidin")
            print("2. Hesabınıza giriş yapın")
            print("3. Settings > API sayfasına gidin")
            print("4. Private API Key'inizi kopyalayın")
            
            api_key = input("\n🔑 API Key'inizi girin (boş bırakabilirsiniz): ").strip()
            
            if api_key:
                # API key'i kaydet
                result = handle_roboflow_action('1', api_key=api_key)
                if result:
                    print("✅ API key başarıyla kaydedildi!")
                    # Split config al
                    split_config = get_dataset_split_config(api_key)
                    return {'api_key': api_key, 'split_config': split_config}
                else:
                    print("❌ API key kaydedilemedi, boş olarak devam ediliyor")
                    return {'api_key': None, 'split_config': None}
            else:
                print("✅ API key boş bırakıldı, public dataset olarak devam ediliyor")
                return {'api_key': None, 'split_config': None}
        
        else:
            print("❌ Geçersiz seçenek")
            continue

def main():
    """Main function - Hierarchical Multi-Dataset Training Framework"""
    # Language selection at startup
    language_choice = select_language()
    
    # --- Dil seçiminden hemen sonra: Global timestamp sabitleme sistemi ---
    try:
        # Get language choice for prompts
        lang = 'tr' if get_text('language_choice', default='tr').startswith('tr') else 'en'
        
        # Ask user for timestamp choice and set global timestamp
        global_timestamp = ask_user_for_timestamp_choice(lang)
        
        # Set environment variable for other processes
        os.environ['SMARTFARM_GLOBAL_TIMESTAMP'] = global_timestamp
        
        if lang == 'tr':
            print(f"🌐 Global timestamp oturumu: {global_timestamp}")
            print(f"📁 Bu timestamp tüm işlemlerde (Google Drive, model indirme, eğitim) tutarlı kullanılacak")
        else:
            print(f"🌐 Global timestamp session: {global_timestamp}")
            print(f"📁 This timestamp will be used consistently across all operations (Google Drive, model download, training)")
            
    except Exception as _ts_e:
        if lang == 'tr':
            print(f"⚠️ Global timestamp sabitleme atlandı: {_ts_e}")
        else:
            print(f"⚠️ Global timestamp setup skipped: {_ts_e}")
    
    # --- Eski Drive timestamp sistemi (uyumluluk için korundu) ---
    try:
        # Sadece Colab'de anlamlı; fakat kod güvenle çalışır
        from drive_manager import activate_drive_integration
        import yaml as _yaml
        # drive kökünü config'ten oku
        drive_folder = "SmartFarm/colab_learn/yolo11_models"
        try:
            _cfg_path = 'config_datasets.yaml'
            if os.path.exists(_cfg_path):
                with open(_cfg_path, 'r', encoding='utf-8') as _cf:
                    _cfg = _yaml.safe_load(_cf) or {}
                _gs = _cfg.get('global_settings', {}) if isinstance(_cfg, dict) else {}
                _p = _gs.get('drive_folder_path')
                if isinstance(_p, str) and _p.strip():
                    drive_folder = _p.strip()
        except Exception:
            pass
        dm = activate_drive_integration(folder_path=drive_folder, project_name="yolo11_models")
        if dm and getattr(dm, 'project_folder', None):
            # Global timestamp varsa onu kullan
            if _GLOBAL_TIMESTAMP:
                # DriveManager'ı global timestamp ile senkronize et
                expected_path = os.path.join(os.path.dirname(dm.project_folder), _GLOBAL_TIMESTAMP)
                if dm.project_folder != expected_path:
                    try:
                        # DriveManager'ın timestamp'ini güncelle
                        dm.project_folder = expected_path
                        os.environ['SMARTFARM_DRIVE_TS'] = expected_path
                        if lang == 'tr':
                            print(f"🔄 DriveManager global timestamp ile senkronize edildi: {_GLOBAL_TIMESTAMP}")
                        else:
                            print(f"🔄 DriveManager synchronized with global timestamp: {_GLOBAL_TIMESTAMP}")
                    except Exception:
                        pass
            else:
                # Fallback: eski sistem
                os.environ['SMARTFARM_DRIVE_TS'] = dm.project_folder
    except Exception as _sess_e:
        if lang == 'tr':
            print(f"⚠️ Drive session sabitleme atlandı: {_sess_e}")
        else:
            print(f"⚠️ Drive session setup skipped: {_sess_e}")
    
    # Drive bağlantı kontrolü (dil seçiminden sonra)
    try:
        from drive_manager import debug_colab_environment, manual_drive_mount
        
        # Colab ortamında Drive kontrolü
        is_colab = debug_colab_environment()
        if is_colab:
            print(f"\n{get_text('drive_check_title', default='🔍 Google Drive Bağlantı Kontrolü')}")
            print("="*50)
            
            # Drive mount durumu kontrol et
            import os
            if not os.path.exists('/content/drive/MyDrive'):
                print(f"{get_text('drive_not_mounted', default='❌ Google Drive mount edilmemiş!')}")
                
                mount_choice = input(f"{get_text('mount_drive_question', default='Drive\'ı şimdi mount etmek ister misiniz? (e/h, varsayılan: e)')} ").lower() or "e"
                
                if mount_choice.startswith('e'):
                    if manual_drive_mount():
                        print(f"{get_text('drive_mount_success', default='✅ Drive başarıyla mount edildi!')}")
                    else:
                        print(f"{get_text('drive_mount_failed', default='❌ Drive mount başarısız. Eğitim yerel kaydetme ile devam edecek.')}")
                else:
                    print(f"{get_text('drive_skip_info', default='ℹ️ Drive mount atlandı. Eğitim yerel kaydetme ile yapılacak.')}")
            else:
                print(f"{get_text('drive_already_mounted', default='✅ Google Drive zaten mount edilmiş!')}")
                
    except ImportError:
        pass  # Drive manager mevcut değilse sessizce devam et
    except Exception as e:
        print(f"⚠️ Drive kontrol hatası: {e}")
    
    print("\n" + "="*70)
    print(get_text('main_title'))
    print(get_text('main_subtitle'))
    print("="*70)
    
    print(f"\n{get_text('main_menu')}")
    print(get_text('option_download'))
    print(get_text('option_training'))
    print(get_text('option_test'))
    print(get_text('option_exit'))
    
    choice = input(f"\n{get_text('select_option')}") or "1"
    
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
        in_colab = is_colab
        
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
        
        # Eğitim parametrelerini merkezi olarak normalize et
        options = prepare_training_options(options)
        
        # Check if we're resuming from a checkpoint
        if options.get('resume'):
            print("\n" + "="*50)
            print(f"🔄 Eğitime devam ediliyor: {options['checkpoint_path']}")
            print("="*50)

            # Resume'da veri YAML doğrulaması: yoksa Drive'daki checkpoint klasöründen kullan
            try:
                yaml_path = options.get('data', 'merged_dataset.yaml')
                if not os.path.isabs(yaml_path):
                    local_yaml = os.path.join(os.getcwd(), yaml_path)
                else:
                    local_yaml = yaml_path

                if not os.path.exists(local_yaml):
                    ckpt_dir = os.path.dirname(options['checkpoint_path'])
                    drive_yaml = os.path.join(ckpt_dir, os.path.basename(yaml_path))
                    if os.path.exists(drive_yaml):
                        options['data'] = drive_yaml
                        print(f"ℹ️ Yerelde '{yaml_path}' bulunamadı. Drive'dan kullanılacak: {drive_yaml}")
                    else:
                        print(f"❗ Gerekli data YAML bulunamadı: '{yaml_path}'.")
                        print("   - Yerelde yok.")
                        print(f"   - Drive klasöründe de yok: {drive_yaml}")
                        # Kullanıcıya hızlı çözüm: dataset işlemi çalıştırılsın mı?
                        do_process = (input("YAML'ı üretmek için veri işleme adımını çalıştıralım mı? (e/h, varsayılan: e): ") or "e").lower()
                        if do_process.startswith('e'):
                            dc = options['dataset_config']
                            if dc['type'] == 'hierarchical_multi':
                                if not process_hierarchical_datasets(dc['setup']):
                                    print('❌ Veri seti işleme başarısız. Çıkılıyor...')
                                    return
                                # Başarılıysa yeniden yerel YAML'ı kullan
                                if os.path.exists(local_yaml):
                                    options['data'] = local_yaml
                                    print(f"✅ YAML üretildi ve kullanılacak: {local_yaml}")
                            else:
                                print("⚠️ Bu modda otomatik YAML üretimi desteklenmiyor. Lütfen 'dataset.yaml' yolunu doğru girin.")
                        else:
                            print("❌ YAML olmadan eğitime devam edilemez. Çıkılıyor...")
                            return
            except Exception as yaml_check_err:
                print(f"⚠️ Resume öncesi YAML kontrolünde hata: {yaml_check_err}")

            # Skip dataset processing when resuming (YAML doğrulaması yapıldı)
            results = train_model(options, hyp=None, epochs=options['epochs'])
        else:
            # Process dataset(s) for new training
            dataset_config = options['dataset_config']
            
            if dataset_config['type'] == 'single':
                # Single dataset processing (legacy) - API key ve split config ile
                from dataset_utils import download_dataset
                
                api_key = dataset_config.get('api_key')
                split_config = dataset_config.get('split_config')
                
                if not download_dataset(dataset_config['url'], api_key=api_key, split_config=split_config):
                    print('❌ Veri seti indirme başarısız. Çıkılıyor...')
                    return
                    
            elif dataset_config['type'] == 'hierarchical_multi':
                # Hierarchical multi-dataset processing
                if not process_hierarchical_datasets(dataset_config['setup']):
                    print('❌ Hiyerarşik veri seti işleme başarısız. Çıkılıyor...')
                    return
                # Seçenek 2 ise eğitimde config/merged_class_dataset.yaml kullanılmalı
                try:
                    lm = (dataset_config.get('setup') or {}).get('label_mode')
                    if lm == 'preserve_subclasses':
                        options['data'] = os.path.join('config', 'merged_class_dataset.yaml')
                        print(f"ℹ️ Eğitim YAML: {options['data']}")
                except Exception:
                    pass
            
            # Show memory status before training
            show_memory_usage("Eğitim Öncesi")
            
            # Create hyperparameter file for new training
            from hyperparameters import create_hyperparameters_file, load_hyperparameters
            hyp_path = create_hyperparameters_file()
            hyperparameters = load_hyperparameters(hyp_path)
            
            # Kullanıcı tercihlerini kaydet (eğitim başlamadan önce)
            print(f"\n💾 Kullanıcı tercihleri ve ayarları kaydediliyor...")
            augmentation_settings = collect_augmentation_settings(dataset_config)
            configs_dir = save_user_preferences_config(options, dataset_config, augmentation_settings)
            
            if configs_dir:
                # Augmentation ayarlarını ayrı dosyada da kaydet
                save_augmentation_config(configs_dir, augmentation_settings)
                
                # Hyperparameter dosyasını da configs klasörüne kopyala
                try:
                    if hyp_path and os.path.exists(hyp_path):
                        hyp_backup_path = os.path.join(configs_dir, 'hyperparameters_backup.yaml')
                        shutil.copy2(hyp_path, hyp_backup_path)
                        print(f"📋 Hyperparameter dosyası yedeklendi: {hyp_backup_path}")
                except Exception as e:
                    print(f"⚠️ Hyperparameter yedekleme hatası: {e}")
            
            # Start new training
            print(f"\n🚀 Yeni model eğitimi başlatılıyor...")
            # Normalize edilmiş options zaten mevcut; train_model'e aktar
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
            
            # Save to Google Drive (otomatik kaydetme - tekrar soru sorma)
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
            
            # Save partial results if available (otomatik kaydetme)
            if in_colab and options.get('drive_save_path'):
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
        
        model_path = input("Eğitilmiş model yolunu girin (varsayılan: runs/train/exp/weights/best.pt): ").strip() or "runs/train/exp/weights/best.pt"
        if not model_path or not os.path.exists(model_path):
            print("❌ Model dosyası bulunamadı.")
            return
        
        test_image = input("Test görüntüsü yolunu girin (varsayılan: test.jpg): ").strip() or "test.jpg"
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
