#!/usr/bin/env python3
# training.py - Model training functions for YOLO11

import os
import torch
import yaml
import numpy as np
from pathlib import Path
import gc
import psutil
import humanize
from typing import List, Tuple, Optional
import torchvision

from ultralytics import YOLO
from PIL import Image

def validate_bbox(bbox: List[float], image_size: Tuple[int, int]) -> bool:
    """
    Validate a bounding box.

    Args:
    - bbox (List[float]): Bounding box coordinates in the format [x1, y1, x2, y2].
    - image_size (Tuple[int, int]): Size of the image.

    Returns:
    - bool: True if the bounding box is valid, False otherwise.
    """
    x1, y1, x2, y2 = bbox
    image_width, image_height = image_size

    # Check if the bounding box is within the image boundaries
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return False

    # Check if the bounding box has a valid width and height
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return False

    return True

def process_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[float]:
    """
    Process a bounding box.

    Args:
    - bbox (List[float]): Bounding box coordinates in the format [x1, y1, x2, y2].
    - image_size (Tuple[int, int]): Size of the image.

    Returns:
    - List[float]: Processed bounding box coordinates.
    """
    x1, y1, x2, y2 = bbox
    image_width, image_height = image_size

    # Clip the bounding box to the image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return [x1, y1, x2, y2]

import os
import sys
import yaml
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

from memory_utils import show_memory_usage, clean_memory
from drive_manager import DriveManager, setup_drive_integration


def auto_profile_training(options: dict, speed_mode: bool) -> None:
    """Set sensible defaults for batch and imgsz based on hardware if not provided.
    - Only overrides when options doesn't include 'batch' or 'imgsz'.
    - Uses simple VRAM buckets.
    """
    try:
        user_specified_batch = 'batch' in options and options['batch'] is not None
        user_specified_imgsz = 'imgsz' in options and options['imgsz'] is not None

        total_vram_gb = None
        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev)
                total_vram_gb = props.total_memory / (1024 ** 3)
            except Exception:
                total_vram_gb = None

        # Determine defaults
        default_batch = None
        default_imgsz = None
        if total_vram_gb is None:
            # CPU or unknown GPU: conservative defaults
            default_batch = 8
            default_imgsz = 512
        else:
            if total_vram_gb >= 16:
                default_batch = 32 if speed_mode else 24
                default_imgsz = 1024 if speed_mode else 896
            elif total_vram_gb >= 12:
                default_batch = 24 if speed_mode else 16
                default_imgsz = 896 if speed_mode else 832
            elif total_vram_gb >= 8:
                default_batch = 16 if speed_mode else 12
                default_imgsz = 832 if speed_mode else 640
            elif total_vram_gb >= 6:
                default_batch = 8 if speed_mode else 6
                default_imgsz = 640 if speed_mode else 576
            else:
                default_batch = 4
                default_imgsz = 512

        # Apply only if user did not specify
        if not user_specified_batch and default_batch is not None:
            options['batch'] = default_batch
        if not user_specified_imgsz and default_imgsz is not None:
            options['imgsz'] = default_imgsz

        # Log decision
        info_vram = f"GPU VRAM ~{total_vram_gb:.1f}GB" if total_vram_gb is not None else "GPU yok/okunamadı"
        print(f"🧪 Auto-profile: {info_vram} -> batch={options.get('batch')} imgsz={options.get('imgsz')} (speed_mode={speed_mode})")
        print("ℹ️ Bu değerler kullanıcı tarafından geçersiz kılınabilir (options['batch'], options['imgsz']).")
    except Exception as _auto_prof_err:
        print(f"⚠️ Auto-profile başarısız: {_auto_prof_err}")

def find_latest_checkpoint(options: dict, drive_manager: Optional[DriveManager]) -> Optional[str]:
    """Find the latest checkpoint locally or on Google Drive - Colab kapanma korumalı."""
    
    # 1. Check Google Drive first if enabled - Gelişmiş arama
    if drive_manager:
        print("\n🔍 Google Drive'da checkpoint aranıyor...")
        
        # Drive'da checkpoint ara - Güncellenmiş yöntem
        try:
            checkpoint_path, filename = drive_manager.find_latest_checkpoint()
            
            if checkpoint_path and filename:
                print(f"📥 Drive'da checkpoint bulundu: {filename}")
                print(f"📁 Checkpoint yolu: {checkpoint_path}")
                
                # Dosya zaten Drive'da mevcut, doğrudan kullan
                if os.path.exists(checkpoint_path):
                    file_size = os.path.getsize(checkpoint_path) / (1024*1024)
                    print(f"✅ Checkpoint hazır: {checkpoint_path} ({file_size:.1f} MB)")
                    print("💡 Colab kapandıktan sonra eğitim devam edecek!")
                    return checkpoint_path
                else:
                    print(f"❌ Checkpoint dosyası erişilemez: {checkpoint_path}")
            else:
                print("ℹ️ Drive'da checkpoint bulunamadı")
                
        except Exception as e:
            print(f"⚠️ Drive checkpoint arama hatası: {e}")
    
    # 2. Check local runs directory - Gelişmiş yerel arama
    print("\n🔍 Yerel checkpoint aranıyor...")
    runs_dir = Path("runs/train")
    if runs_dir.exists():
        # En son experiment dizinini bul
        exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if exp_dirs:
            latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
            weights_dir = latest_exp / "weights"
            
            # Training state kontrol et
            state_file = weights_dir / "training_state.json"
            if state_file.exists():
                try:
                    import json
                    with open(state_file, 'r') as f:
                        training_state = json.load(f)
                    last_epoch = training_state.get('current_epoch', 0)
                    print(f"📊 Yerel training state bulundu - Son epoch: {last_epoch}")
                except Exception:
                    pass
            
            # last.pt dosyasını kontrol et
            last_pt = weights_dir / "last.pt"
            if last_pt.exists():
                print(f"✅ Yerel checkpoint bulundu: {last_pt}")
                return str(last_pt)
    
    print("❌ Hiçbir checkpoint bulunamadı")
    print("💡 Yeni eğitim başlatılacak")
    return None

# TensorBoard entegrasyonunu devre dışı bırak
os.environ["TENSORBOARD_BINARY"] = "False"

# Enable cuDNN autotuner for optimal performance on fixed input sizes
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# Optimize matmul precision if available (PyTorch >= 2.0)
try:
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
except Exception:
    pass

def train_model(options, hyp=None, epochs=None, drive_save_interval=10):
    """Train a YOLO model with the given options and hyperparameters."""
    print("\n" + "="*50)
    print(f"🚀 Starting training session")
    
    # Google Drive entegrasyonu (tek seferlik soru)
    print("\n🔧 Google Drive kaydetme ayarları - Colab kapanma durumu için optimize edilmiş")
    drive_default = "e"  # Varsayılan olarak Drive kullanımını öner
    use_drive = input(f"Google Drive'a otomatik kaydetme kullanılsın mı? (e/h, varsayılan: {drive_default}): ").lower() or drive_default
    use_drive = use_drive.startswith('e')
    
    drive_manager = None
    save_interval = drive_save_interval  # Fonksiyon parametresini kullan
    
    if use_drive:
        print("\n🔄 Colab Kapanma Koruması Ayarları")
        print("Colab bazen kendiliğinden kapanabilir. Bu duruma karşı:")
        print("1. Daha sık yedekleme (3-5 epoch)")
        print("2. Normal yedekleme (10 epoch)")
        print("3. Özel aralık")
        
        backup_mode = input("Yedekleme sıklığı seçin (1/2/3, varsayılan: 1): ").strip() or "1"
        
        if backup_mode == "1":
            save_interval = 3  # Colab kapanma koruması için sık yedekleme
            print("✅ Sık yedekleme modu: Her 3 epoch'ta bir kaydetme")
        elif backup_mode == "2":
            save_interval = 10  # Normal yedekleme
            print("✅ Normal yedekleme modu: Her 10 epoch'ta bir kaydetme")
        else:
            save_interval = int(input(f"Özel aralık (epoch): ") or str(drive_save_interval))
            print(f"✅ Özel yedekleme modu: Her {save_interval} epoch'ta bir kaydetme")
        
        drive_manager = setup_drive_integration()
        if not drive_manager:
            print("⚠️ Drive entegrasyonu kurulamadı, sadece yerel kaydetme yapılacak.")
            use_drive = False
            print(f"ℹ️ Yerel kaydetme aralığı: {save_interval} epoch")

    # --- Eğitim Modu Seçimi ---
    mode = input("\nEğitim modunu seçin:\n1. Yeni Eğitim Başlat\n2. Kaldığı Yerden Devam Et (Resume)\n3. Fine-tune (Önceki Ağırlıklarla Başla)\nSeçim (1/2/3): ").strip()

    model_path = options['model']
    resume_training = False
    finetune_active = False

    if mode == '2': # Resume
        print("\n🔄 Kaldığı yerden devam etme modu seçildi.")
        checkpoint_path = find_latest_checkpoint(options, drive_manager)
        if checkpoint_path:
            model_path = checkpoint_path
            resume_training = True
            print(f"Model şu checkpoint'ten devam edecek: {model_path}")
        else:
            print("❌ Devam edilecek checkpoint bulunamadı. Yeni bir eğitim başlatılıyor.")
            # Fallback to new training

    elif mode == '3': # Fine-tune
        print("\n🎯 Fine-tune modu seçildi.")
        finetune_source_path = find_latest_checkpoint(options, drive_manager) # Can also use last.pt for fine-tuning
        if finetune_source_path:
            model_path = finetune_source_path
            finetune_active = True
            print(f"Fine-tune için başlangıç ağırlığı: {model_path}")
        else:
            print(f"⚠️ Fine-tune için başlangıç ağırlığı bulunamadı. Varsayılan model '{model_path}' kullanılacak.")

    # Set up hyperparameters with safety checks
    if hyp is None and options.get('use_hyp', True):
        from hyperparameters import load_hyperparameters, create_hyperparameters_file
        
        # Ensure hyperparameters file exists
        hyp_path = create_hyperparameters_file()
        
        # Load hyperparameters
        hyp = load_hyperparameters(hyp_path)
        print(f"⚙️  Loaded hyperparameters from {hyp_path}")
    
    # Configure augmentation with safety limits
    augmentation_cfg = setup_augmentation(hyp if hyp else {})
    
    # Bounding box safety wrapper for data loading
    def safe_load_image_and_boxes(img_path, label_path):
        """Load image and validate bboxes with safety checks."""
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_width, img_height = img.size
            
            # Load and validate bboxes
            valid_boxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:  # YOLO format: class x_center y_center width height
                            cls_id, x_center, y_center, width, height = map(float, parts)
                            
                            # Convert to x1,y1,x2,y2
                            x1 = (x_center - width/2) * img_width
                            y1 = (y_center - height/2) * img_height
                            x2 = (x_center + width/2) * img_width
                            y2 = (y_center + height/2) * img_height
                            
                            # Process and validate bbox
                            bbox = [x1, y1, x2, y2]
                            if validate_bbox(bbox, (img_width, img_height)):
                                valid_boxes.append([cls_id, x_center, y_center, width, height])
                            else:
                                print(f"⚠️ Invalid bbox in {label_path}: {bbox}")
            
            return img, valid_boxes
            
        except Exception as e:
            print(f"❌ Error loading {img_path}: {str(e)}")
            return None, []

    # Kaydetme aralığı zaten yukarıda belirlendi
    print(f"\n💾 Kaydetme aralığı: Her {save_interval} epoch'ta bir")

    try:
        # Handle PyTorch model loading with compatibility settings
        print("Attempting to load model...")
        
        try:
            # First try normal loading
            model = YOLO(model_path)
            print(f"Model loaded successfully: {model_path}")
        except Exception as e1:
            print(f"Standard loading attempt failed: {e1}")
            # Try with alternative PyTorch loading options
            try:
                # For PyTorch 2.6+ with weights_only=False
                if hasattr(torch, '_C') and hasattr(torch._C, '_loading_deserializer_set_weights_only'):
                    print("Trying with weights_only=False...")
                    original_value = torch._C._loading_deserializer_set_weights_only(False)
                    try:
                        model = YOLO(model_path)
                        print(f"Model loaded successfully with weights_only=False: {model_path}")
                    finally:
                        # Reset to original value
                        torch._C._loading_deserializer_set_weights_only(original_value)
                else:
                    raise Exception("Could not set weights_only parameter")
            except Exception as e2:
                print(f"Alternative loading attempt failed: {e2}")
                # Try using standard YOLOv8 model as a fallback
                try:
                    model = YOLO('yolov8l.pt')  # Use standard YOLOv8 model as fallback
                    print("Using standard YOLOv8l model as fallback")
                    options['model'] = 'yolov8l.pt'
                except Exception as e3:
                    print(f"Fallback model loading failed: {e3}")
                    raise Exception("All model loading attempts failed")
                
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

    # --- Güvenlik: nc (sınıf sayısı) uyuşmazlığında resume devre dışı bırak ---
    try:
        # Dataset nc'yi oku
        dataset_nc = None
        data_yaml_path = options.get('data')
        if isinstance(data_yaml_path, (str, os.PathLike)) and os.path.exists(str(data_yaml_path)):
            with open(data_yaml_path, 'r') as f:
                data_cfg = yaml.safe_load(f) or {}
            if isinstance(data_cfg.get('names'), (list, tuple)):
                dataset_nc = len(data_cfg['names'])
            elif isinstance(data_cfg.get('nc'), int):
                dataset_nc = data_cfg['nc']

        # Checkpoint/model nc'yi oku
        model_nc = None
        try:
            model_nc = getattr(getattr(model, 'model', None), 'nc', None)
        except Exception:
            model_nc = None

        if resume_training and (dataset_nc is not None) and (model_nc is not None) and dataset_nc != model_nc:
            print("\n⚠️  Sınıf sayısı uyuşmazlığı tespit edildi (checkpoint nc="
                  f"{model_nc} ≠ dataset nc={dataset_nc}).")
            print("🔁 'Resume' yerine güvenli mod: fine-tune olarak devam edilecek (resume=False).")
            # Resume'ı kapat, fine-tune bayrağını aç
            resume_training = False
            finetune_active = True
            # model_path aynı kalır; Ultralytics 'data' içindeki nc'ye göre head'i ayarlar ve
            # uyumsuz katmanları transfer learning mantığıyla yeniden oluşturur.
    except Exception as _nc_err:
        print(f"⚠️ nc uyuşmazlık kontrolü başarısız: {_nc_err}")

    # Determine control flags from options/hyp
    speed_mode_flag = bool(options.get('speed_mode'))
    if hyp is not None and isinstance(hyp, dict):
        if 'speed_mode' in hyp:
            speed_mode_flag = bool(hyp.get('speed_mode'))

    # Settings for periodic memory cleanup
    cleanup_frequency = int(input("\nRAM cleanup frequency (clean every N epochs? e.g., 10): ") or "10")
    
    # Dataset caching mode
    if speed_mode_flag:
        cache_mode = 'ram'
        print("⚡ Hız modu aktif: Dataset cache 'ram' olarak ayarlandı.")
    else:
        # default to 'disk' to reduce host RAM usage in Colab
        cache_from_hyp = None
        if hyp is not None and isinstance(hyp, dict):
            cache_from_hyp = hyp.get('cache')
        if isinstance(cache_from_hyp, str):
            cache_choice = cache_from_hyp.strip().lower()
        else:
            cache_choice = (input("\nDataset cache mode [ram/disk/none] (default: disk): ") or "disk").strip().lower()
        if cache_choice in ("disk", "ram"):
            cache_mode = cache_choice
        elif cache_choice in ("n", "no", "none", "false", "0"):
            cache_mode = False
        else:
            cache_mode = "disk"

    # İsteğe bağlı: DataLoader workers sayısını kullanıcıdan al (boş bırakırsanız otomatik)
    try:
        user_workers_in = input("\nDataLoader workers sayısı (boş bırak: otomatik): ").strip()
        if user_workers_in:
            options['workers'] = int(user_workers_in)
    except Exception:
        pass

    # Hardware auto-profile: set batch/imgsz defaults if not provided
    auto_profile_training(options, speed_mode_flag)

    # DataLoader workers dinamik ayarı
    cpu_cnt = os.cpu_count() or 4
    workers_value = options.get('workers')
    try:
        workers_value = int(workers_value) if workers_value is not None else None
    except Exception:
        workers_value = None
    if speed_mode_flag:
        # Hız modunda: kullanıcı bir değer verdiyse onu koru, aksi halde otomatik yüksek değer seç
        if workers_value is None:
            workers_value = max(8, (cpu_cnt - 1) if cpu_cnt > 1 else 0)
            print(f"⚡ Hız modu (otomatik): DataLoader workers -> {workers_value} (cpu={cpu_cnt})")
        else:
            print(f"⚡ Hız modu: Kullanıcı tanımlı workers değeri korunuyor -> {workers_value}")
        print("ℹ️ Ultralytics düşük hafıza tespit ederse bu değeri runtime'da düşürebilir.")
    else:
        if workers_value is None:
            # Normal mod: mantıklı varsayılan (CPU-1, 2 ile 4 arasında)
            workers_value = max(2, min(4, cpu_cnt - 1)) if cpu_cnt > 1 else 2
        print(f"🧰 DataLoader workers seçimi: {workers_value} (cpu={cpu_cnt})")

    # Training arguments with augmentation safety
    train_args = {
        'model': model_path,
        'data': options['data'],
        'epochs': epochs if epochs is not None else options['epochs'],
        'imgsz': options.get('imgsz', 640),
        'batch': options.get('batch', 16),
        'workers': workers_value,
        'cache': cache_mode,
        'device': options.get('device', '0' if torch.cuda.is_available() else 'cpu'),
        'project': options.get('project', 'runs/train'),
        'name': options.get('name', 'exp'),
        'exist_ok': options.get('exist_ok', True),
        'resume': resume_training,
        
        # Use hyperparameters from hyp dict if available
        'hsv_h': hyp.get('hsv_h', 0.015),
        'hsv_s': hyp.get('hsv_s', 0.7),
        'hsv_v': hyp.get('hsv_v', 0.4),
        'degrees': hyp.get('degrees', 0.0),
        'translate': hyp.get('translate', 0.1),
        'scale': hyp.get('scale', 0.5),
        'shear': hyp.get('shear', 0.0),
        'perspective': hyp.get('perspective', 0.0),
        'flipud': hyp.get('flipud', 0.0),
        'fliplr': hyp.get('fliplr', 0.5),
        'mosaic': hyp.get('mosaic', 1.0),
        'mixup': hyp.get('mixup', 0.1),
        
        # Safety settings
        'rect': False,  # Disable rectangular training for better bbox safety
        'copy_paste': 0.0  # Disable copy-paste augmentation
    }

    # In speed mode, reduce overhead of plotting during training
    # Set plots flag
    plots_flag = options.get('plots', True)
    if hyp is not None and isinstance(hyp, dict) and 'plots' in hyp:
        plots_flag = bool(hyp['plots'])
    if speed_mode_flag:
        plots_flag = False
    train_args['plots'] = plots_flag

    # Add hyperparameters if available (as fixed constants, not as hyp.yaml file!)
    if hyp is not None and options.get('use_hyp', True):
        # Get patience value from hyperparameters
        if 'patience' in hyp:
            train_args['patience'] = hyp['patience']

        # Add lr0 (initial learning rate) and other supported custom parameters
        if 'lr0' in hyp:
            train_args['lr0'] = hyp['lr0']

        # Add other parameters directly supported by YOLO11 training
        supported_params = ['lrf', 'warmup_epochs', 'warmup_momentum', 'box', 'cls']
        for param in supported_params:
            if param in hyp:
                train_args[param] = hyp[param]

        print(f"Compatible settings transferred from hyperparameters")
    else:
        # Default patience value
        train_args['patience'] = 50

    # Enable AMP when CUDA is available (Ultralytics supports 'amp')
    if torch.cuda.is_available():
        train_args['amp'] = True

    # Fine-tune spesifik ayarlar: düşük lr0 ve freeze
    if finetune_active:
        # lr0 belirleme: hyp.finetune_lr0 > hyp.lr0*0.1 > 0.0005
        ft_lr = None
        if hyp is not None:
            ft_lr = hyp.get('finetune_lr0')
            if ft_lr is None and 'lr0' in hyp:
                try:
                    ft_lr = float(hyp['lr0']) * 0.1
                except Exception:
                    ft_lr = None
        if ft_lr is None:
            ft_lr = 0.0005
        train_args['lr0'] = float(ft_lr)

        # freeze parametresi
        if hyp is not None and hyp.get('freeze') is not None:
            train_args['freeze'] = hyp.get('freeze')

        print(f"🔧 Fine-tune ayarları: lr0={train_args['lr0']}, freeze={train_args.get('freeze', None)}")

    print('Training parameters:')
    for k, v in train_args.items():
        print(f'  {k}: {v}')
    if speed_mode_flag:
        print("  ⚡ speed_mode: True (cache=ram, workers>=8, plots=False)")
    if finetune_active:
        print("  🎯 finetune: True (başlangıç ağırlığı: önceki checkpoint, resume=False)")

    # Show memory status before training
    show_memory_usage("Training Start Memory Status")

    # Manuel olarak memory clean up yapan callback oluştur
    project_dir = train_args.get('project', 'runs/train')
    experiment_name = train_args.get('name', 'exp')
    # Kullanıcıdan alınan değerle eşitle (önceki hatalı kullanım: drive_save_interval)
    save_interval_epochs = save_interval
    drive_save_dir = options.get('drive_save_path')

    # Ultralytics yerel periyodik kaydetme (weights/epoch_XXX.pt) için native parametre
    try:
        train_args['save_period'] = int(save_interval_epochs)
    except Exception:
        train_args['save_period'] = save_interval_epochs

    # -------------------------------
    # Gelişmiş Drive Kaydetme Fonksiyonu - Colab Kapanma Korumalı
    # -------------------------------
    def save_models_periodically(project_dir, experiment_name, drive_manager, save_interval_epochs, current_epoch):
        """Belirlenen aralıklarda modelleri Drive'a kaydet - Colab kapanma korumalı"""
        if current_epoch % save_interval_epochs != 0:
            return  # Kaydetme zamanı değil
        
        weights_dir = Path(project_dir) / experiment_name / 'weights'
        last_pt_path = weights_dir / 'last.pt'
        best_pt_path = weights_dir / 'best.pt'
        
        print(f"\n💾 Colab Kapanma Koruması - Epoch {current_epoch} yedekleme başlıyor...")
        
        # Eğitim durumu bilgilerini kaydet
        training_state = {
            'current_epoch': current_epoch,
            'project_dir': str(project_dir),
            'experiment_name': experiment_name,
            'timestamp': time.time(),
            'save_interval': save_interval_epochs
        }
        
        try:
            # Eğitim durumu dosyasını kaydet
            state_file = weights_dir / 'training_state.json'
            with open(state_file, 'w') as f:
                import json
                json.dump(training_state, f, indent=2)
            
            # last.pt kaydet (en önemli - devam etmek için gerekli)
            if last_pt_path.exists():
                ok1 = drive_manager.upload_model(str(last_pt_path), f'epoch_{current_epoch:03d}.pt')
                ok2 = drive_manager.upload_model(str(last_pt_path), 'last.pt')
                
                # Eğitim durumu da kaydet
                ok3 = drive_manager.upload_model(str(state_file), 'training_state.json')
                
                if ok1 and ok2 and ok3:
                    print(f"✅ Checkpoint kaydedildi: epoch_{current_epoch:03d}.pt, last.pt, training_state.json")
                else:
                    print(f"⚠️ Kısmi yedekleme (epoch {current_epoch})")
            
            # best.pt kaydet (varsa)
            if best_pt_path.exists():
                okb = drive_manager.upload_model(str(best_pt_path), 'best.pt')
                if okb:
                    print(f"✅ best.pt yüklendi (epoch {current_epoch})")
                else:
                    print(f"⚠️ best.pt yükleme başarısız (epoch {current_epoch})")
            
            # Eski checkpoint'leri temizle (disk alanı için)
            cleanup_old_checkpoints(weights_dir, current_epoch, keep_last=3)
                    
        except Exception as e:
            print(f"❌ Yedekleme hatası (epoch {current_epoch}): {e}")
            print("💡 Colab kapanırsa son kaydedilen checkpoint'ten devam edebilirsiniz")
    
    def cleanup_old_checkpoints(weights_dir, current_epoch, keep_last=3):
        """Eski checkpoint'leri temizle - disk alanı tasarrufu"""
        try:
            pattern = weights_dir / 'epoch_*.pt'
            checkpoint_files = list(weights_dir.glob('epoch_*.pt'))
            
            if len(checkpoint_files) > keep_last:
                # Epoch numarasına göre sırala
                checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[1]))
                
                # Eski dosyaları sil (son N tanesini koru)
                for old_file in checkpoint_files[:-keep_last]:
                    try:
                        old_file.unlink()
                        print(f"🧹 Eski checkpoint temizlendi: {old_file.name}")
                    except Exception:
                        pass  # Sessizce devam et
                        
        except Exception:
            pass  # Temizlik hatası kritik değil

    try:
        # Manage model training with periodic memory cleanup
        print("\n--- Training Model ---")
        
        # TensorBoard'i ağır buluyorsanız yalnızca plotting kapatıldı (plots=False). Callback'ler korunur.

        # -------------------------------
        # Drive kaydetme ayarları
        # -------------------------------
        print(f"💾 Drive kaydetme aralığı: Her {save_interval_epochs} epoch'ta bir")
        
        # Eğitim sırasında periyodik kaydetme için thread başlat
        import threading
        import time
        
        def periodic_save_thread():
            """Arka planda periyodik kaydetme (dosya izleme ile gerçek epoch)"""
            # Tek satırda güncellenen durum yazıcısı
            def _make_status_printer(prefix=""):
                last_len = 0
                def _printer(msg: str):
                    nonlocal last_len
                    line = f"{prefix}{msg}"
                    pad = max(0, last_len - len(line))
                    sys.stdout.write("\r" + line + (" " * pad))
                    sys.stdout.flush()
                    last_len = len(line)
                return _printer

            status_print = _make_status_printer()

            seen_epochs = set()
            last_report = 0
            while True:
                time.sleep(10)  # 10 sn'de bir kontrol
                try:
                    weights_dir = Path(project_dir) / experiment_name / 'weights'
                    if not weights_dir.exists():
                        continue

                    # epoch_*.pt dosyalarını tara
                    epoch_files = list(weights_dir.glob('epoch_*.pt'))
                    for p in epoch_files:
                        try:
                            stem = p.stem  # epoch_XXX
                            ep = int(stem.split('_')[1])
                        except Exception:
                            continue

                        if ep in seen_epochs:
                            continue

                        seen_epochs.add(ep)
                        if ep % int(save_interval_epochs) == 0:
                            save_models_periodically(project_dir, experiment_name, drive_manager, int(save_interval_epochs), ep)

                    # Bilgi mesajını tek satırda güncelle
                    import time as _t
                    now = _t.time()
                    if now - last_report > 5:  # 5 sn'de bir satırı güncelle
                        last_seen = max(seen_epochs) if seen_epochs else 0
                        status_print(f"📡 Dosya izleme aktif - son görülen epoch: {last_seen}")
                        last_report = now
                except Exception as e:
                    # Yeni satıra geç ve hatayı yaz
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    print(f"⚠️ Periyodik kaydetme thread hatası: {e}")
                    break
            # Thread biterken yeni satıra geç
            sys.stdout.write("\n")
            sys.stdout.flush()
        
        # Thread'i başlat (daemon olarak)
        if use_drive and drive_manager:
            save_thread = threading.Thread(target=periodic_save_thread, daemon=True)
            save_thread.start()
            print("🔄 Periyodik kaydetme thread'i başlatıldı")
            
        # Resume modunda: tespit edilen checkpoint'in klasöründeki last.pt/best.pt dosyalarını
        # yeni deneyin weights klasörüne kopyalayarak başlangıç dosyalarını hazırla
        try:
            if resume_training and isinstance(model_path, str):
                src_dir = Path(model_path).parent
                dst_weights = Path(project_dir) / experiment_name / 'weights'
                dst_weights.mkdir(parents=True, exist_ok=True)
                copied_any = False
                for fname in ['last.pt', 'best.pt']:
                    src = src_dir / fname
                    if src.exists():
                        import shutil as _shutil
                        _shutil.copy2(src, dst_weights / fname)
                        print(f"📄 Başlangıç dosyası kopyalandı → {dst_weights / fname}")
                        copied_any = True
                if not copied_any:
                    print("ℹ️ Resume için kopyalanacak last.pt/best.pt bulunamadı (devam ediliyor).")
        except Exception as prep_e:
            print(f"⚠️ Resume başlangıç dosyaları kopyalanırken hata: {prep_e}")

        # Model eğitimini başlat
        results = model.train(**train_args)
        
        # Eğitim tamamlandıktan sonra final kaydetme işlemleri
        print(f"\n🎯 Eğitim tamamlandı! Belirlenen aralık: {save_interval_epochs} epoch")
        
        # Eğitim sonunda final model kaydetme (Drive API ve dosya sistemi)
        if results is not None:
            print("\n🎯 Final modeller kaydediliyor...")

            save_dir = os.path.join(project_dir, experiment_name)
            best_path = os.path.join(save_dir, "weights", "best.pt")
            last_path = os.path.join(save_dir, "weights", "last.pt")
            final_epoch = train_args.get('epochs', 100)

            # 1) Drive API ile yükleme (eğer etkinse)
            if use_drive and drive_manager:
                # Doğru imza: upload_model(local_path, drive_filename)
                if os.path.exists(best_path):
                    ok_best_name = drive_manager.upload_model(best_path, 'best.pt')
                    ok_best_epoch = drive_manager.upload_model(best_path, f'epoch_{final_epoch:03d}_best.pt')
                    if ok_best_name and ok_best_epoch:
                        print("✅ Final best.pt yüklendi (best.pt ve epoch_*_best.pt)")
                    else:
                        print("❌ Final best.pt yükleme başarısız. Ayrıntılar yukarıdaki loglarda.")
                if os.path.exists(last_path):
                    ok_last_name = drive_manager.upload_model(last_path, 'last.pt')
                    ok_last_epoch = drive_manager.upload_model(last_path, f'epoch_{final_epoch:03d}.pt')
                    if ok_last_name and ok_last_epoch:
                        print("✅ Final last.pt yüklendi (last.pt ve epoch_*.pt)")
                    else:
                        print("❌ Final last.pt yükleme başarısız. Ayrıntılar yukarıdaki loglarda.")
                print("\n📋 Drive'daki tüm modeller:")
                drive_manager.list_drive_models()

            # 2) Dosya sistemine kopyalama (Drive mount edilen dizine)
            if drive_save_dir:
                try:
                    os.makedirs(drive_save_dir, exist_ok=True)
                    if os.path.exists(best_path):
                        shutil.copy(best_path, os.path.join(drive_save_dir, "best.pt"))
                        print(f"💾 Final best.pt kopyalandı → {os.path.join(drive_save_dir, 'best.pt')}")
                    if os.path.exists(last_path):
                        shutil.copy(last_path, os.path.join(drive_save_dir, "last.pt"))
                        print(f"💾 Final last.pt kopyalandı → {os.path.join(drive_save_dir, 'last.pt')}")
                except Exception as copy_e:
                    print(f"Final kopyalama hatası: {copy_e}")
        
        # Show memory status after training
        show_memory_usage("After Training")
        
        # Clean memory after training
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Geçici checkpoint dosyasını temizle
        temp_files = [f for f in os.listdir('.') if f.startswith('temp_checkpoint_epoch_')]
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"🗑️ Geçici dosya temizlendi: {temp_file}")
            except:
                pass
        
        return results
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean memory after error
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return None

def save_to_drive(options, results=None):
    """Save trained model to Google Drive (for Colab)"""
    try:
        # Connect to Google Drive
        from google.colab import drive
        print("\nConnecting to Google Drive...")
        drive.mount('/content/drive')

        # Determine model file paths
        project_dir = options.get('project', 'runs/train')
        exp_name = options.get('name', 'exp')
        best_model_path = f"{project_dir}/{exp_name}/weights/best.pt"
        last_model_path = f"{project_dir}/{exp_name}/weights/last.pt"

        # Create folder in Drive
        drive_folder = "/content/drive/MyDrive/Tarim/YapayZeka_model/Model/YOLO11_Egitim"
        os.makedirs(drive_folder, exist_ok=True)

        # Copy models
        import shutil
        if os.path.exists(best_model_path):
            drive_best_path = f"{drive_folder}/best_model.pt"
            shutil.copy(best_model_path, drive_best_path)
            print(f"Best model saved to Drive: {drive_best_path}")

        if os.path.exists(last_model_path):
            drive_last_path = f"{drive_folder}/last_model.pt"
            shutil.copy(last_model_path, drive_last_path)
            print(f"Last model saved to Drive: {drive_last_path}")

        # Also copy training results
        results_path = f"{project_dir}/{exp_name}/results.csv"
        if os.path.exists(results_path):
            drive_results_path = f"{drive_folder}/results.csv"
            shutil.copy(results_path, drive_results_path)
            print(f"Training results saved to Drive: {drive_results_path}")

        print(f"All files saved to {drive_folder}.")
        return True
    except Exception as e:
        print(f"Error saving to Google Drive: {e}")
        print("Don't forget to save your model files manually!")
        return False

def validate_model(model_path, data_yaml, batch_size=16, img_size=640):
    """Validate a trained model on test/validation data"""
    try:
        print(f"\n===== Model Validation =====")
        print(f"Loading model: {model_path}")
        
        # Load the model with compatibility settings
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"Standard model loading failed: {e}")
            
            # Try with alternative loading method
            if hasattr(torch, '_C') and hasattr(torch._C, '_loading_deserializer_set_weights_only'):
                print("Trying with weights_only=False...")
                original_value = torch._C._loading_deserializer_set_weights_only(False)
                try:
                    model = YOLO(model_path)
                finally:
                    torch._C._loading_deserializer_set_weights_only(original_value)
            else:
                raise Exception("Could not load model with alternate method")
        
        # Run validation
        print(f"Running validation on: {data_yaml}")
        results = model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=img_size,
            verbose=True
        )
        
        # Report results
        print("\nValidation Results:")
        metrics = ["precision", "recall", "mAP50", "mAP50-95"]
        for metric in metrics:
            if hasattr(results, metric):
                value = getattr(results, metric)
                print(f"  {metric}: {value:.4f}")
        
        return results
    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_model(model_path, format='onnx', img_size=640, simplify=True):
    """Export YOLO model to different formats"""
    try:
        print(f"\n===== Model Export =====")
        print(f"Loading model: {model_path}")
        
        # Load the model with compatibility settings
        try:
            model = YOLO(model_path)
        except Exception as e:
            print(f"Standard model loading failed: {e}")
            
            # Try with alternative loading method
            if hasattr(torch, '_C') and hasattr(torch._C, '_loading_deserializer_set_weights_only'):
                print("Trying with weights_only=False...")
                original_value = torch._C._loading_deserializer_set_weights_only(False)
                try:
                    model = YOLO(model_path)
                finally:
                    torch._C._loading_deserializer_set_weights_only(original_value)
            else:
                raise Exception("Could not load model with alternate method")
        
        # Available formats
        formats = ['torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 
                  'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle', 'ncnn']
        
        if format.lower() not in formats:
            print(f"Unsupported format: {format}")
            print(f"Supported formats: {', '.join(formats)}")
            return None
        
        # Export the model
        print(f"Exporting to {format} format...")
        model.export(
            format=format, 
            imgsz=img_size,
            simplify=simplify
        )
        
        print(f"Model exported successfully!")
        
        # Find exported file
        model_dir = os.path.dirname(model_path)
        if not model_dir:
            model_dir = '.'
            
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        exported_file = None
        
        # Map of format to extension
        format_extensions = {
            'torchscript': '.torchscript',
            'onnx': '.onnx',
            'openvino': '_openvino_model',
            'engine': '.engine',
            'coreml': '.mlmodel',
            'saved_model': '_saved_model',
            'pb': '.pb',
            'tflite': '.tflite',
            'edgetpu': '_edgetpu.tflite',
            'tfjs': '_web_model',
            'paddle': '_paddle_model',
            'ncnn': '_ncnn_model'
        }
        
        # Look for exported file
        if format in format_extensions:
            extension = format_extensions[format]
            potential_file = os.path.join(model_dir, base_name + extension)
            if os.path.exists(potential_file):
                exported_file = potential_file
            elif os.path.isdir(potential_file):
                exported_file = potential_file
                
        if exported_file:
            print(f"Exported file: {exported_file}")
            
        return exported_file
    except Exception as e:
        print(f"Export error: {e}")
        import traceback
        traceback.print_exc()
        return None

def setup_augmentation(hyp: dict) -> dict:
    """
    Augmentation parametrelerini hyp içinden alır, yoksa güvenli varsayılanlar kullanır
    ve değerleri mantıklı aralıklara sıkıştırır.
    """
    def clamp(val, lo, hi):
        try:
            v = float(val)
        except Exception:
            v = lo
        return max(lo, min(hi, v))

    cfg = {
        'hsv_h': clamp(hyp.get('hsv_h', 0.015), 0.0, 0.1),
        'hsv_s': clamp(hyp.get('hsv_s', 0.7),   0.0, 0.99),
        'hsv_v': clamp(hyp.get('hsv_v', 0.4),   0.0, 0.99),
        'degrees': clamp(hyp.get('degrees', 0.0), -45.0, 45.0),
        'translate': clamp(hyp.get('translate', 0.1), 0.0, 0.5),
        'scale': clamp(hyp.get('scale', 0.5), 0.0, 1.0),
        'shear': clamp(hyp.get('shear', 0.0), -10.0, 10.0),
        'perspective': clamp(hyp.get('perspective', 0.0), 0.0, 0.001),
        'flipud': clamp(hyp.get('flipud', 0.0), 0.0, 1.0),
        'fliplr': clamp(hyp.get('fliplr', 0.5), 0.0, 1.0),
        'mosaic': clamp(hyp.get('mosaic', 1.0), 0.0, 1.0),
        'mixup': clamp(hyp.get('mixup', 0.0), 0.0, 1.0),
    }

    # Güvenlik için ek sınırlamalar
    if cfg['mosaic'] > 0:
        cfg['mixup'] = clamp(cfg['mixup'], 0.0, 0.2)
    if cfg['translate'] > 0.4:
        cfg['translate'] = 0.4
    if cfg['scale'] > 0.9:
        cfg['scale'] = 0.9

    return cfg

if __name__ == "__main__":
    print("This module provides model training functions.")
    print("It is not meant to be run directly.")
