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
        return True
    else:
        print("❌ No files found to copy.")
        return False

def download_models_menu():
    """Interactive menu for downloading YOLO11 models"""
    print("\n===== YOLO11 Model Download =====")
    
    default_dir = os.path.join("/content/colab_learn", "yolo11_models") if is_colab() else "yolo11_models"
    save_dir = input(f"\nSave directory (default: {default_dir}): ") or default_dir
    
    print("\nDownload options:")
    print("1. Download single model")
    print("2. Download all detection models")
    print("3. Download all models (all types)")
    
    choice = input("\nYour choice (1-3): ")
    
    if choice == "1":
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
        
        model_path = download_specific_model_type(model_type, size, save_dir)
        if model_path:
            print(f"\n✅ Model downloaded successfully to: {model_path}")
    
    elif choice == "2":
        detection_models = ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        downloaded = download_yolo11_models(save_dir, detection_models)
        print(f"\n✅ Downloaded {len(downloaded)} detection models to {save_dir}")
    
    elif choice == "3":
        downloaded = download_yolo11_models(save_dir)
        print(f"\n✅ Downloaded {len(downloaded)} models to {save_dir}")
    
    else:
        print("\n❌ Invalid choice. No models downloaded.")
        return None
    
    return save_dir

def hierarchical_dataset_setup():
    """Setup for hierarchical multi-dataset training"""
    print("\n===== Hierarchical Multi-Dataset Setup =====")
    
    # Initialize the YAML-based dataset manager
    config_file = input("Config file path (default: config_datasets.yaml): ") or "config_datasets.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        print("Please ensure config_datasets.yaml exists in the current directory")
        return None
    
    manager = YAMLBasedMultiDatasetManager(config_file=config_file)
    
    # Show system information
    print(f"\n📊 System Information:")
    print(f"✅ Config loaded: {config_file}")
    print(f"📁 Available groups: {len(manager.get_available_dataset_groups())}")
    
    # Interactive dataset selection
    selected_group = manager.interactive_dataset_selection()
    
    if not selected_group:
        print("❌ No dataset group selected")
        return None
    
    # Get recommendations
    recommendations = manager.get_training_recommendations(selected_group)
    
    print(f"\n🎯 Training Recommendations for '{selected_group}':")
    for key, value in recommendations.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    • {item}")
        else:
            print(f"  {key}: {value}")
    
    # Get global settings
    settings = manager.get_global_settings()
    
    # Ask for target count per class
    default_target = settings.get('default_target_count_per_class', 2000)
    while True:
        try:
            target_count = int(input(f"\nTarget samples per class (default: {default_target}): ") or str(default_target))
            if target_count > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Output directory
    default_output = "datasets/hierarchical_merged"
    output_dir = input(f"\nMerged dataset directory (default: {default_output}): ") or default_output
    
    return {
        'manager': manager,
        'selected_group': selected_group,
        'target_count': target_count,
        'output_dir': output_dir,
        'recommendations': recommendations,
        'settings': settings
    }

def process_hierarchical_datasets(dataset_config):
    """Process hierarchical multi-datasets"""
    print("\n===== Processing Hierarchical Multi-Datasets =====")
    
    manager = dataset_config['manager']
    target_count = dataset_config['target_count']
    
    try:
        # 1. Download datasets
        print("\n1️⃣ Downloading datasets...")
        download_success = manager.download_all_datasets()
        
        if not download_success:
            print("❌ Dataset download failed!")
            return False
        
        # 2. Create unified class mapping
        print("\n2️⃣ Creating hierarchical class mapping...")
        classes_created = manager.create_unified_class_mapping()
        
        if classes_created == 0:
            print("❌ No classes could be mapped!")
            return False
        
        print(f"✅ Created {classes_created} main classes")
        
        # 3. Merge datasets with hierarchical structure
        print("\n3️⃣ Merging datasets with hierarchical structure...")
        merged_counts = manager.merge_datasets(target_count_per_class=target_count)
        
        if not merged_counts:
            print("❌ Dataset merging failed!")
            return False
        
        print(f"\n✅ Hierarchical multi-dataset processing completed!")
        print(f"📁 Merged dataset: {manager.output_dir}")
        print(f"📄 YAML file: merged_dataset.yaml")
        print(f"🏷️  Class mapping: unified_class_mapping.json")
        
        # Display final statistics
        total_samples = sum(merged_counts.values())
        print(f"\n📊 Final Dataset Statistics:")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Main classes: {len(merged_counts)}")
        print(f"   Samples per class: {total_samples // len(merged_counts):,} (average)")
        
        for class_name, count in merged_counts.items():
            print(f"   {class_name}: {count:,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during hierarchical dataset processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_training_setup():
    """Interactive training parameter setup for hierarchical model"""
    print("\n===== Hierarchical Model Training Setup =====")
    
    # Dataset type selection
    print("\nDataset configuration:")
    print("1) Hierarchical multi-dataset (Recommended)")
    print("2) Single Roboflow dataset (Legacy)")
    
    while True:
        dataset_choice = input("\nSelect option [1-2] (default: 1): ") or "1"
        if dataset_choice in ["1", "2"]:
            break
        print("❌ Please select 1 or 2.")
    
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
            print("❌ No URL provided")
            return None
        
        dataset_config = {
            'type': 'single',
            'url': roboflow_url,
            'data_yaml': 'dataset.yaml'
        }
    
    # Project category
    print("\nProject category:")
    print("1) Hierarchical Agricultural AI (Recommended)")
    print("2) Disease Detection")
    print("3) Pest Detection")
    print("4) Mixed Agricultural")
    print("5) Custom")
    
    while True:
        category_choice = input("\nSelect category [1-5] (default: 1): ") or "1"
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
                category = input("Enter custom category name: ").strip() or "custom"
            break
        print("❌ Please select 1-5.")
    
    # Get training recommendations
    if dataset_config['type'] == 'hierarchical_multi':
        recommendations = dataset_config['setup']['recommendations']
        recommended_model = recommendations.get('model', 'yolo11l.pt')
        recommended_batch = recommendations.get('batch_size', 8)
        recommended_size = recommendations.get('image_size', 640)
        estimated_time = recommendations.get('estimated_time', 'Unknown')
        
        print(f"\n🎯 Recommendations for hierarchical model:")
        print(f"   Model: {recommended_model}")
        print(f"   Batch size: {recommended_batch}")
        print(f"   Image size: {recommended_size}")
        print(f"   Estimated time: {estimated_time}")
        
        # Show special notes if available
        special_notes = recommendations.get('special_notes', [])
        if special_notes:
            print(f"   Special considerations:")
            for note in special_notes:
                print(f"     • {note}")
    
    # Resume training options
    resume_training = False
    checkpoint_path = None
    
    if is_colab():
        has_previous = input("\nResume training from checkpoint? (y/n, default: n): ").lower() or "n"
        
        if has_previous.startswith("y"):
            resume_training = True
            resume_from_drive = input("Load checkpoint from Google Drive? (y/n, default: y): ").lower() or "y"
            
            if resume_from_drive.startswith("y"):
                if not os.path.exists('/content/drive'):
                    mount_google_drive()
                
                base_folder = f"/content/drive/MyDrive/Tarim/Kodlar/colab_egitim/{category}"
                print(f"\nExpected model directory: {base_folder}")
                
                custom_path = input(f"Confirm or enter new path (default: {base_folder}): ") or base_folder
                
                model_type = input("\nWhich model file to use? (best/last, default: best): ").lower() or "best"
                if model_type not in ["best", "last"]:
                    model_type = "best"
                
                checkpoint_path = os.path.join(custom_path, f"{model_type}.pt")
                
                if os.path.exists(checkpoint_path):
                    print(f"✅ Model file found: {checkpoint_path}")
                    os.makedirs("runs/train/exp/weights", exist_ok=True)
                    shutil.copy2(checkpoint_path, f"runs/train/exp/weights/{model_type}.pt")
                    print(f"✅ Model file copied to training directory.")
                else:
                    print(f"⚠️  WARNING: Model file not found: {checkpoint_path}")
                    print("Training will start from scratch.")
                    resume_training = False
    
    # Training parameters
    while True:
        try:
            if dataset_config['type'] == 'hierarchical_multi':
                default_epochs = 300  # More epochs for hierarchical model
                epochs = int(input(f"\nEpochs [100-2000 recommended] (default: {default_epochs}): ") or str(default_epochs))
            else:
                epochs = int(input(f"\nEpochs [100-1000 recommended] (default: 300): ") or "300")
            
            if epochs > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Model size selection
    print("\nSelect model size:")
    print("1) yolo11s.pt - Small (fastest, lower accuracy)")
    print("2) yolo11m.pt - Medium (balanced)")
    print("3) yolo11l.pt - Large (high accuracy, slower) [Recommended for hierarchical]")
    print("4) yolo11x.pt - Extra Large (highest accuracy, slowest)")

    while True:
        if dataset_config['type'] == 'hierarchical_multi':
            model_choice = input("\nSelect model [1-4] (default: 3): ") or "3"
        else:
            model_choice = input("\nSelect model [1-4] (default: 2): ") or "2"
        
        model_options = {
            "1": "yolo11s.pt",
            "2": "yolo11m.pt",
            "3": "yolo11l.pt",
            "4": "yolo11x.pt"
        }
        
        if model_choice in model_options:
            model = model_options[model_choice]
            
            # Check if model exists locally
            model_dir = os.path.join("/content/colab_learn", "yolo11_models") if is_colab() else "yolo11_models"
            model_path = os.path.join(model_dir, model)
            
            if not os.path.exists(model_path):
                print(f"\n⚠️  Model {model} not found locally.")
                download_now = input("Download now? (y/n, default: y): ").lower() or "y"
                
                if download_now.startswith("y"):
                    os.makedirs(model_dir, exist_ok=True)
                    download_specific_model_type("detection", model[6], model_dir)
                else:
                    print(f"ℹ️  Model will be downloaded automatically during training.")
            break
        print("❌ Please select 1-4.")
    
    # Batch size and image size
    if dataset_config['type'] == 'hierarchical_multi':
        default_batch = recommended_batch
        default_img_size = recommended_size
    else:
        default_batch = 16
        default_img_size = 640
    
    while True:
        try:
            batch_size = int(input(f"\nBatch size (default: {default_batch}, smaller for low RAM): ") or str(default_batch))
            if batch_size > 0:
                break
            print("❌ Please enter a positive number.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    while True:
        try:
            img_size = int(input(f"\nImage size (default: {default_img_size}, must be multiple of 32): ") or str(default_img_size))
            if img_size > 0 and img_size % 32 == 0:
                break
            print("❌ Please enter a positive number that's a multiple of 32.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Google Drive save settings
    drive_save_path = None
    if is_colab():
        print("\nGoogle Drive save settings:")
        save_to_drive_opt = input("Save training results to Google Drive? (y/n, default: y): ").lower() or "y"
        
        if save_to_drive_opt.startswith("y"):
            if not os.path.exists('/content/drive'):
                mount_google_drive()
            
            default_drive_path = f"/content/drive/MyDrive/Tarim/Kodlar/colab_egitim/{category}"
            drive_save_path = input(f"Save directory (default: {default_drive_path}): ") or default_drive_path
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drive_save_path = os.path.join(drive_save_path, timestamp)
            print(f"📁 Models will be saved to: {drive_save_path}")
    
    # Hyperparameter file
    use_hyp = input("\nUse hyperparameter file (hyp.yaml)? (y/n, default: y): ").lower() or "y"
    use_hyp = use_hyp.startswith("y")
    
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
        'data': dataset_config['data_yaml'],
        'project': 'runs/train',
        'name': 'exp',
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'exist_ok': True,
        'resume': resume_training,
        'use_hyp': use_hyp,
        'category': category,
        'drive_save_path': drive_save_path,
        'checkpoint_path': checkpoint_path
    }
    
    # Display selected parameters
    print("\n===== Selected Training Parameters =====")
    print(f"Dataset type: {dataset_config['type']}")
    if dataset_config['type'] == 'hierarchical_multi':
        setup = dataset_config['setup']
        print(f"Dataset group: {setup['selected_group']}")
        print(f"Target samples per class: {setup['target_count']:,}")
        print(f"Output directory: {setup['output_dir']}")
    
    print(f"Model: {model}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"Device: {device}")
    print(f"Category: {category}")
    
    if drive_save_path:
        print(f"Drive save path: {drive_save_path}")
    
    confirm = input("\nProceed with these parameters? (y/n): ").lower()
    if confirm != 'y' and confirm != 'yes' and confirm != 'evet':
        print("❌ Setup cancelled.")
        return None
    
    return options

def main():
    """Main function - Hierarchical Multi-Dataset Training Framework"""
    print("\n" + "="*70)
    print("🌱 YOLO11 Hierarchical Multi-Dataset Training Framework")
    print("🎯 Advanced Agricultural AI with Hierarchical Detection")
    print("="*70)
    
    print("\n🚀 Main Menu:")
    print("1. 📥 Download YOLO11 models")
    print("2. 🎛️  Training setup and execution")
    print("3. 🔍 Test hierarchical detection (requires trained model)")
    print("4. ❌ Exit")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == "1":
        download_models_menu()
        train_now = input("\nProceed to training setup? (y/n, default: y): ").lower() or "y"
        if not train_now.startswith("y"):
            return
        choice = "2"  # Continue to training
        
    if choice == "2":
        in_colab = is_colab()
        
        # Install required packages
        print("\n📦 Installing required packages...")
        install_required_packages()
        
        # Create hyperparameter file
        hyp_path = create_hyperparameters_file()
        hyperparameters = load_hyperparameters(hyp_path)
        
        # Interactive setup
        options = interactive_training_setup()
        if options is None:
            return
        
        # Process dataset(s)
        dataset_config = options['dataset_config']
        
        if dataset_config['type'] == 'single':
            # Single dataset processing (legacy)
            from dataset_utils import download_dataset
            
            if not download_dataset(dataset_config['url']):
                print('❌ Dataset download failed. Exiting...')
                return
                
        elif dataset_config['type'] == 'hierarchical_multi':
            # Hierarchical multi-dataset processing
            if not process_hierarchical_datasets(dataset_config['setup']):
                print('❌ Hierarchical dataset processing failed. Exiting...')
                return
        
        # Show memory status before training
        show_memory_usage("Before Training")
        
        # Train the model
        print(f"\n🚀 Starting hierarchical model training...")
        results = train_model(options, hyp=hyperparameters, resume=options.get('resume', False), epochs=options['epochs'])
        
        if results:
            print('✅ Training completed successfully!')
            print(f'📊 Results: {results}')
            
            # Initialize hierarchical detection if available
            if HIERARCHICAL_DETECTION_AVAILABLE:
                print(f"\n🎯 Initializing hierarchical detection system...")
                try:
                    visualizer = HierarchicalDetectionVisualizer()
                    print(f"✅ Hierarchical detection system ready!")
                    print(f"🏷️  Detection format: 'ZARLI: Kırmızı Örümcek (0.85)'")
                except Exception as e:
                    print(f"⚠️  Could not initialize hierarchical detection: {e}")
            
            # Save to Google Drive
            if in_colab and options.get('drive_save_path'):
                print("\n💾 Saving models to Google Drive...")
                if save_models_to_drive(options['drive_save_path']):
                    print("✅ Models saved to Google Drive successfully.")
                else:
                    print("❌ Failed to save models to Google Drive.")
        else:
            print('❌ Training failed or was interrupted.')
            
            # Save partial results if available
            if in_colab and options.get('drive_save_path'):
                save_anyway = input("\nSave partial training results to Google Drive? (y/n, default: y): ").lower() or "y"
                if save_anyway.startswith("y"):
                    print("\n💾 Saving partial results to Google Drive...")
                    if save_models_to_drive(options['drive_save_path']):
                        print("✅ Partial results saved to Google Drive.")
                    else:
                        print("❌ Failed to save partial results.")
        
        # Clean memory
        show_memory_usage("After Training")
        clean_memory()
    
    elif choice == "3":
        # Test hierarchical detection
        if not HIERARCHICAL_DETECTION_AVAILABLE:
            print("❌ Hierarchical detection utils not available.")
            return
        
        model_path = input("Enter trained model path (e.g., runs/train/exp/weights/best.pt): ").strip()
        if not model_path or not os.path.exists(model_path):
            print("❌ Model file not found.")
            return
        
        test_image = input("Enter test image path: ").strip()
        if not test_image or not os.path.exists(test_image):
            print("❌ Test image not found.")
            return
        
        try:
            from ultralytics import YOLO
            import cv2
            
            # Load model and visualizer
            model = YOLO(model_path)
            visualizer = HierarchicalDetectionVisualizer()
            
            # Run detection
            print(f"🔍 Running hierarchical detection...")
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
            print(f"✅ Annotated image saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error during hierarchical detection test: {e}")
    
    elif choice == "4":
        print("👋 Exiting...")
    
    else:
        print("❌ Invalid option. Exiting...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ Process completed.")
