#!/usr/bin/env python3
# model_downloader.py - Utility for downloading YOLO11 models

import os
import requests
import sys
from pathlib import Path
import json

try:
    from drive_manager import DriveManager
    _DM_AVAILABLE = True
except Exception:
    _DM_AVAILABLE = False

def is_colab() -> bool:
    """Basit Colab tespiti."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return os.path.exists('/content')

def _append_download_log(project_folder: str, paths):
    """Drive timestamp klasöründeki logs/downloads.json'a ekleme yapar."""
    try:
        if not project_folder:
            return
        logs_dir = os.path.join(project_folder, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        logf = os.path.join(logs_dir, 'downloads.json')
        if isinstance(paths, str):
            paths = [paths]
        try:
            entries = []
            if os.path.exists(logf):
                with open(logf, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
        except Exception:
            entries = []
        now = __import__('datetime').datetime.now().isoformat()
        for p in paths or []:
            entries.append({'file': os.path.basename(p), 'dest': p, 'ts': now})
        with open(logf, 'w', encoding='utf-8') as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _prepare_drive_timestamp_folder():
    """Colab + Drive ortamında timestamp klasörünü hazırla ve models yolunu döndür.

    Returns: (models_dir, project_folder, dm) veya (None, None, None)
    """
    if not (is_colab() and _DM_AVAILABLE):
        return None, None, None
    try:
        dm = DriveManager()
        if not dm.authenticate():
            return None, None, None

        reused = False
        # Önce mevcut konfigürasyonu yüklemeyi dene (mevcut timestamp varsa onu kullan)
        if dm.load_drive_config():
            project_folder = dm.get_timestamp_dir()
            if project_folder and os.path.basename(os.path.dirname(project_folder)) == 'yolo11_models':
                reused = True
        else:
            project_folder = None

        # Konfigürasyon yoksa yeni timestamp kur
        if not project_folder:
            if not dm._setup_colab_folder():
                return None, None, None
            project_folder = dm.get_timestamp_dir()
            reused = False

        # Alt klasörleri garanti et
        for sub in ["models", "logs", "configs", "checkpoints"]:
            os.makedirs(os.path.join(project_folder, sub), exist_ok=True)
        models_dir = os.path.join(project_folder, "models")
        if reused:
            print(f"📁 Mevcut Drive timestamp klasörü kullanılıyor: {project_folder}")
        else:
            print(f"📁 Drive timestamp klasörü oluşturuldu: {project_folder}")
        return models_dir, project_folder, dm
    except Exception:
        pass
    return None, None, None

def download_yolo11_models(save_dir=None, selected_models=None):
    """
    Download YOLO11 models from GitHub releases
    
    Args:
        save_dir: Directory to save models (default: ./yolo11_models)
        selected_models: List of specific models to download (default: all)
    
    Returns:
        List of paths to downloaded models
    """
    # Use default save directory if not specified
    drive_project_folder = None
    if save_dir is None:
        # Colab + Drive otomatik yönlendirme
        models_dir, drive_project_folder, _dm = _prepare_drive_timestamp_folder()
        if models_dir:
            save_dir = models_dir
            print(f"📁 İndirme dizini Drive timestamp klasörüne ayarlandı: {save_dir}")
        else:
            save_dir = os.path.join(os.getcwd(), "yolo11_models")
    else:
        # Drive kökü veya timestamp kökü ise düzelt
        try:
            norm = os.path.normpath(save_dir)
            base = os.path.basename(norm)
            parent = os.path.basename(os.path.dirname(norm))
            if base == "yolo11_models":
                ts = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                drive_project_folder = os.path.join(norm, ts)
                for sub in ["models", "logs", "configs", "checkpoints"]:
                    os.makedirs(os.path.join(drive_project_folder, sub), exist_ok=True)
                save_dir = os.path.join(drive_project_folder, "models")
                print(f"📁 Drive timestamp klasörü oluşturuldu ve indirime yönlendirildi: {save_dir}")
            # Eğer timestamp köküne işaret ediyorsa, models altına yönlendir
            elif (len(base) == 15 and '_' in base and parent == "yolo11_models"):
                drive_project_folder = norm
                os.makedirs(os.path.join(drive_project_folder, "models"), exist_ok=True)
                save_dir = os.path.join(drive_project_folder, "models")
                print(f"📁 İndirme dizini timestamp/models olarak ayarlandı: {save_dir}")
        except Exception:
            pass
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Base URL for YOLOv11 models
    base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
    
    # List of model variants
    all_model_variants = [
        "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",  # Detection models
        "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",  # Segmentation models
        "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt",  # Classification models
        "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",  # Pose models
        "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt"  # OBB models
    ]
    
    # Determine which models to download
    model_variants = selected_models if selected_models else all_model_variants
    
    # Keep track of downloaded model paths
    downloaded_models = []
    
    # Download each model
    for model in model_variants:
        download_url = base_url + model
        save_path = os.path.join(save_dir, model)
        
        # Skip if model already exists
        if os.path.exists(save_path):
            print(f"Model {model} already exists at {save_path}")
            downloaded_models.append(save_path)
            continue
        
        print(f"Downloading {model}...")
        try:
            response = requests.get(download_url)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            with open(save_path, "wb") as f:
                f.write(response.content)
                
            print(f"Saved to {save_path}")
            downloaded_models.append(save_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {model}: {e}")
    
    print("YOLO11 models download completed!")
    # Eğer Drive timestamp klasörü kullanıldıysa indirmeleri logla
    if drive_project_folder and downloaded_models:
        _append_download_log(drive_project_folder, downloaded_models)
    return downloaded_models

def download_specific_model_type(model_type="detection", size="m", save_dir=None):
    """
    Download a specific type of YOLO11 model
    
    Args:
        model_type: Type of model to download (detection, segmentation, classification, pose, obb)
        size: Model size (s, m, l, x)
        save_dir: Directory to save models
    
    Returns:
        Path to downloaded model
    """
    # Validate model_type
    valid_types = {
        "detection": "",
        "segmentation": "-seg",
        "classification": "-cls",
        "pose": "-pose",
        "obb": "-obb"
    }
    
    if model_type not in valid_types:
        print(f"Invalid model type. Choose from: {', '.join(valid_types.keys())}")
        return None
    
    # Validate size
    valid_sizes = ["s", "m", "l", "x"]
    if size not in valid_sizes:
        print(f"Invalid model size. Choose from: {', '.join(valid_sizes)}")
        return None
    
    # Construct model name
    model_suffix = valid_types[model_type]
    model_name = f"yolo11{size}{model_suffix}.pt"
    
    # Download the model
    models = download_yolo11_models(save_dir, [model_name])
    
    # Return the path to the downloaded model
    return models[0] if models else None

if __name__ == "__main__":
    # If run directly, download all models or specific ones
    print("YOLO11 Model Downloader")
    print("======================")

    # Varsayılan dizin ve Drive timestamp klasörü (Colab ise) hazırlığı
    default_dir = os.path.join(os.getcwd(), "yolo11_models")
    dm = None
    drive_project_folder = None
    if is_colab() and _DM_AVAILABLE:
        try:
            dm = DriveManager()
            if dm.authenticate() and dm._setup_colab_folder():
                drive_project_folder = dm.project_folder  # .../SmartFarm/colab_learn/yolo11_models/<timestamp>
                # Alt klasörleri oluştur
                for sub in ["models", "logs", "configs", "checkpoints"]:
                    os.makedirs(os.path.join(drive_project_folder, sub), exist_ok=True)
                default_dir = os.path.join(drive_project_folder, "models")
                print(f"📁 İndirme hedefi Drive timestamp klasörüne yönlendirildi: {default_dir}")
        except Exception as e:
            print(f"⚠️ Drive klasör kurulumu atlandı: {e}")

    # Determine save directory (now that default_dir may point to Drive/models)
    if len(sys.argv) > 1:
        save_dir = sys.argv[1]
    else:
        save_dir = input(f"Enter save directory (default: {default_dir}): ") or default_dir
    
    # Ask whether to download all models
    download_all = input("Download all models? (y/n, default: n): ").lower() == 'y'

    if download_all:
        downloaded = download_yolo11_models(save_dir)
        if drive_project_folder and downloaded:
            _append_download_log(drive_project_folder, downloaded)
    else:
        # Ask for model type
        print("\nSelect model type:")
        print("1. Detection (default)")
        print("2. Segmentation")
        print("3. Classification")
        print("4. Pose")
        print("5. OBB (Oriented Bounding Box)")
        
        model_type_map = {
            "1": "detection",
            "2": "segmentation",
            "3": "classification",
            "4": "pose",
            "5": "obb"
        }
        
        model_type_choice = input("Enter choice (1-5, default: 1): ") or "1"
        model_type = model_type_map.get(model_type_choice, "detection")
        
        # Ask for model size
        print("\nSelect model size:")
        print("1. Small (s)")
        print("2. Medium (m) (default)")
        print("3. Large (l)")
        print("4. Extra Large (x)")
        
        size_map = {
            "1": "s",
            "2": "m",
            "3": "l",
            "4": "x"
        }
        
        size_choice = input("Enter choice (1-4, default: 2): ") or "2"
        size = size_map.get(size_choice, "m")
        
        # Download the selected model
        model_path = download_specific_model_type(model_type, size, save_dir)
        
        if model_path:
            print(f"\nModel downloaded successfully to: {model_path}")
            if drive_project_folder:
                _append_download_log(drive_project_folder, model_path)
