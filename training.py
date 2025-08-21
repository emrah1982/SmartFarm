#!/usr/bin/env python3
# training.py - Model training functions for YOLO11

import os
import sys
import yaml
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

from memory_utils import show_memory_usage, clean_memory
from drive_manager import DriveManager, setup_drive_integration

# TensorBoard entegrasyonunu devre dışı bırak
os.environ["TENSORBOARD_BINARY"] = "False"

# Enable cuDNN autotuner for optimal performance on fixed input sizes
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

def train_model(options, hyp=None, resume=False, epochs=None, drive_save_interval=10):
    """Train YOLO11 model"""
    # Model selection - Check if it's a full path or just a filename
    model_path = options['model']
    
    # If it's just a filename and not a full path, check in the yolo11_models directory
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        yolo_models_dir = os.path.join("/content/colab_learn", "yolo11_models")
        full_model_path = os.path.join(yolo_models_dir, os.path.basename(model_path))
        
        if os.path.exists(full_model_path):
            print(f"Found model at: {full_model_path}")
            model_path = full_model_path
    
    print(f"Loading model: {model_path}")
    # Check model path - YOLO will download automatically if file doesn't exist
    if not os.path.exists(model_path) and model_path.startswith('yolo11') and model_path.endswith('.pt'):
        print(f"Model file not found. YOLO will automatically download the '{model_path}' model...")

    # Google Drive entegrasyonu kurulumu (resume'dan önce)
    print("\n🔧 Google Drive Entegrasyonu")
    use_drive = input("Google Drive'a otomatik kaydetme kullanılsın mı? (y/n): ").lower().startswith('y')
    
    drive_manager = None
    if use_drive:
        drive_manager = setup_drive_integration()
        if not drive_manager:
            print("⚠️ Drive entegrasyonu kurulamadı, sadece yerel kaydetme yapılacak.")
            use_drive = False

    # Load pre-trained model or create a new one
    if resume:
        # Drive'dan devam etme seçeneği
        resume_from_drive = False
        if use_drive and drive_manager:
            choice = input("\nEğitimi nereden devam ettirmek istiyorsunuz?\n1. Yerel dosyalardan\n2. Google Drive'dan\nSeçim (1/2): ")
            if choice == '2':
                resume_from_drive = True
        
        if resume_from_drive:
            # Drive'dan en son checkpoint'i bul ve indir
            print("\n🔍 Drive'da en son checkpoint aranıyor...")
            file_id, latest_epoch = drive_manager.find_latest_checkpoint()
            
            if file_id and latest_epoch > 0:
                print(f"📥 En son checkpoint bulundu: Epoch {latest_epoch}")
                
                # Geçici checkpoint dosyası oluştur
                temp_checkpoint = f"temp_checkpoint_epoch_{latest_epoch}.pt"
                
                if drive_manager.download_checkpoint(file_id, temp_checkpoint):
                    model_path = temp_checkpoint
                    print(f'✅ Drive\'dan devam ediliyor: Epoch {latest_epoch}')
                else:
                    print('❌ Drive\'dan checkpoint indirilemedi, yerel aramaya geçiliyor')
                    resume_from_drive = False
            else:
                print('❌ Drive\'da checkpoint bulunamadı, yerel aramaya geçiliyor')
                resume_from_drive = False
        
        if not resume_from_drive:
            # Yerel checkpoint arama (mevcut kod)
            runs_dir = Path(options.get('save_dir', '') or options.get('project', 'runs/train'))
            exp_name = options.get('name', 'exp')
            weights_dir = runs_dir / exp_name / 'weights'

            if weights_dir.exists():
                last_pt = weights_dir / 'last.pt'
                if last_pt.exists():
                    model_path = str(last_pt)
                    print(f'Yerel dosyadan devam ediliyor: {model_path}')
                else:
                    print('Yerel checkpoint bulunamadı, sıfırdan başlanıyor')
                    resume = False
            else:
                print('Önceki eğitim bulunamadı, sıfırdan başlanıyor')
                resume = False
    
    # Set epoch save interval
    if use_drive:
        save_interval = int(input(f"\nKaç epoch'ta bir Drive'a kaydetme yapılsın? (varsayılan: {drive_save_interval}): ") or str(drive_save_interval))
    else:
        save_interval = int(input("\nHow often to save the model (epochs)? (default: 50): ") or "50")

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
                print("Attempting to use standard YOLOv8 model instead...")
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

    # Settings for periodic memory cleanup
    cleanup_frequency = int(input("\nRAM cleanup frequency (clean every N epochs? e.g., 10): ") or "10")
    # Dataset caching mode: default to 'disk' to reduce host RAM usage in Colab
    cache_choice = (input("\nDataset cache mode [ram/disk/none] (default: disk): ") or "disk").strip().lower()
    if cache_choice in ("disk", "ram"):
        cache_mode = cache_choice
    elif cache_choice in ("n", "no", "none", "false", "0"):
        cache_mode = False
    else:
        cache_mode = "disk"

    # Set training parameters - nolog parametresini kaldırdık
    train_args = {
        'data': options['data'],
        'epochs': epochs if epochs is not None else options['epochs'],
        'imgsz': options['imgsz'],
        'batch': options['batch'],
        'project': options.get('project', 'runs/train'),
        'name': options.get('name', 'exp'),
        'device': '0' if torch.cuda.is_available() else 'cpu',  # Use GPU 0 if available or CPU
        'workers': options.get('workers', 2),
        'exist_ok': options.get('exist_ok', False),
        'pretrained': options.get('pretrained', True),
        'optimizer': options.get('optimizer', 'auto'),
        'verbose': options.get('verbose', True),
        'seed': options.get('seed', 0),
        'cache': cache_mode,  # Use 'disk' by default to limit RAM usage
        'resume': resume,  # Resume from checkpoint
        # DataLoader memory-lean settings
        'pin_memory': False,
        'persistent_workers': False,
        # 'nolog' parametresini kaldırdık - bu parametre desteklenmiyor
    }

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

    print('Training parameters:')
    for k, v in train_args.items():
        print(f'  {k}: {v}')

    # Show memory status before training
    show_memory_usage("Training Start Memory Status")

    # Manuel olarak memory clean up yapan callback oluştur
    project_dir = train_args.get('project', 'runs/train')
    experiment_name = train_args.get('name', 'exp')
    save_interval_epochs = save_interval
    drive_save_dir = options.get('drive_save_path')

    try:
        # Manage model training with periodic memory cleanup
        print("\n--- Training Model ---")
        
        # TensorBoard ve callbacks ayarları - model çağrılmadan önce ayarlama
        if hasattr(model, 'callbacks') and model.callbacks is not None:
            # Eğer model callbacks'e sahipse
            try:
                # TensorBoard callback'ini devre dışı bırak
                model._callbacks = []  # Tüm callbacks'leri temizle ve yeniden ekle
                # Sadece gerekli callback'leri ekle (TensorBoard hariç)
            except Exception as cb_err:
                print(f"Callback devre dışı bırakma hatası: {cb_err}")
        
        # Periyodik temizleme ve Drive kaydetme için manuel callback sınıfı
        class MemoryCleanupAndDriveCallback:
            def __init__(self, cleanup_frequency, save_interval, drive_manager, use_drive, project_dir, experiment_name, drive_save_dir=None):
                self.cleanup_frequency = cleanup_frequency
                self.save_interval = save_interval
                self.drive_manager = drive_manager
                self.use_drive = use_drive
                self.project_dir = project_dir
                self.experiment_name = experiment_name
                self.last_saved_epoch = 0
                self.drive_save_dir = drive_save_dir
            
            def __call__(self, trainer):
                if hasattr(trainer, 'epoch'):
                    current_epoch = trainer.epoch + 1
                    
                    # Memory cleanup
                    if current_epoch % self.cleanup_frequency == 0:
                        print(f"\n--- Memory cleanup at epoch {current_epoch} ---")
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Kaydetme işlemleri (Drive API ve/veya dosya sistemi)
                    if current_epoch % self.save_interval == 0 and current_epoch > self.last_saved_epoch:
                        print(f"\n--- Kaydetme: Epoch {current_epoch} ---")

                        # Yollar
                        best_path = os.path.join(self.project_dir, self.experiment_name, "weights", "best.pt")
                        last_path = os.path.join(self.project_dir, self.experiment_name, "weights", "last.pt")

                        # 1) Drive API ile yükleme (opsiyonel)
                        if self.use_drive and self.drive_manager:
                            if os.path.exists(last_path):
                                success = self.drive_manager.upload_model(last_path, current_epoch, is_best=False)
                                if success:
                                    print(f"✅ Checkpoint epoch {current_epoch} Drive API ile yüklendi")
                            if os.path.exists(best_path) and (os.path.getmtime(best_path) > os.path.getmtime(last_path) if os.path.exists(last_path) else True):
                                success = self.drive_manager.upload_model(best_path, current_epoch, is_best=True)
                                if success:
                                    print(f"✅ Best model epoch {current_epoch} Drive API ile yüklendi")

                        # 2) Dosya sistemine kopyalama (Drive mount edilen dizine)
                        if self.drive_save_dir:
                            try:
                                os.makedirs(self.drive_save_dir, exist_ok=True)
                                if os.path.exists(last_path):
                                    shutil.copy(last_path, os.path.join(self.drive_save_dir, "last.pt"))
                                    print(f"💾 last.pt kopyalandı → {os.path.join(self.drive_save_dir, 'last.pt')}")
                                if os.path.exists(best_path):
                                    shutil.copy(best_path, os.path.join(self.drive_save_dir, "best.pt"))
                                    print(f"💾 best.pt kopyalandı → {os.path.join(self.drive_save_dir, 'best.pt')}")
                            except Exception as copy_e:
                                print(f"Dosya sistemi kopyalama hatası: {copy_e}")

                        self.last_saved_epoch = current_epoch
        
        # Callback'i model nesnesine ekle (eğer destekleniyorsa)
        try:
            if hasattr(model, 'add_callback'):
                callback = MemoryCleanupAndDriveCallback(
                    cleanup_frequency, save_interval, drive_manager, use_drive, 
                    project_dir, experiment_name, drive_save_dir
                )
                model.add_callback("on_train_epoch_end", callback)
                print(f"✅ Drive kaydetme callback'i eklendi (her {save_interval} epoch'ta bir)")
        except Exception as add_cb_err:
            print(f"Callback ekleme hatası: {add_cb_err}")
            
        # Model eğitimini başlat
        results = model.train(**train_args)
        
        # Periyodik olarak manuel kaydet (ultralytics callback'ini kullanamıyorsak)
        if results is not None:
            try:
                save_dir = os.path.join(project_dir, experiment_name)
                best_path = os.path.join(save_dir, "weights", "best.pt")
                if os.path.exists(best_path):
                    # Periyodik olarak en iyi modeli kopyala
                    for i in range(save_interval_epochs, int(train_args['epochs']), save_interval_epochs):
                        save_path = os.path.join(save_dir, "weights", f"epoch_{i}.pt")
                        if not os.path.exists(save_path) and os.path.exists(best_path):
                            shutil.copy(best_path, save_path)
                            print(f"\n--- Model saved for epoch {i}: {save_path} ---")
            except Exception as save_e:
                print(f"Error saving periodic model snapshots: {save_e}")
        
        # Eğitim sonunda final model kaydetme (Drive API ve dosya sistemi)
        if results is not None:
            print("\n🎯 Eğitim tamamlandı! Final modeller kaydediliyor...")

            save_dir = os.path.join(project_dir, experiment_name)
            best_path = os.path.join(save_dir, "weights", "best.pt")
            last_path = os.path.join(save_dir, "weights", "last.pt")
            final_epoch = train_args.get('epochs', 100)

            # 1) Drive API ile yükleme (eğer etkinse)
            if use_drive and drive_manager:
                if os.path.exists(best_path):
                    success = drive_manager.upload_model(best_path, final_epoch, is_best=True)
                    if success:
                        print("✅ Final best model Drive API ile yüklendi")
                if os.path.exists(last_path):
                    success = drive_manager.upload_model(last_path, final_epoch, is_best=False)
                    if success:
                        print("✅ Final checkpoint Drive API ile yüklendi")
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

if __name__ == "__main__":
    print("This module provides model training functions.")
    print("It is not meant to be run directly.")
