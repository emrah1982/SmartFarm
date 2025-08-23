#!/usr/bin/env python3
# language_manager.py - Multi-language support for SmartFarm

import json
import os

class LanguageManager:
    """Multi-language support manager for SmartFarm"""
    
    def __init__(self, default_language='tr'):
        self.current_language = default_language
        self.translations = self._load_translations()
    
    def _load_translations(self):
        """Load translation dictionaries"""
        return {
            'tr': {
                # Main Menu
                'main_title': '🌱 YOLO11 Hiyerarşik Çoklu Veri Seti Eğitim Çerçevesi',
                'main_subtitle': '🎯 Hiyerarşik Tespit ile Gelişmiş Tarımsal AI',
                'main_menu': '🚀 Ana Menü:',
                'option_download': '📥 YOLO11 modellerini indir',
                'option_training': '🎛️  Eğitim kurulumu ve yürütme',
                'option_test': '🔍 Hiyerarşik tespiti test et (eğitilmiş model gerekli)',
                'option_exit': '❌ Çıkış',
                'select_option': 'Seçenek (1-4): ',
                
                # Model Download
                'model_download_title': '===== YOLO11 Model İndirme =====',
                'save_directory': 'Kaydetme dizini (varsayılan: {default}): ',
                'download_options': 'İndirme seçenekleri:',
                'single_model': '1. Tek model indir',
                'all_detection': '2. Tüm tespit modellerini indir',
                'all_models': '3. Tüm modelleri indir (tüm tipler)',
                'your_choice': 'Seçiminiz (1-3): ',
                
                # Model Types
                'select_model_type': 'Model tipini seçin:',
                'detection_default': '1. Tespit (varsayılan)',
                'segmentation': '2. Segmentasyon',
                'classification': '3. Sınıflandırma',
                'pose': '4. Poz',
                'obb': '5. OBB (Yönlendirilmiş Sınırlayıcı Kutu)',
                'enter_choice_1_5': 'Seçiminizi girin (1-5, varsayılan: 1): ',
                
                # Model Sizes
                'select_model_size': 'Model boyutunu seçin:',
                'small': '1. Küçük (s)',
                'medium_default': '2. Orta (m) (varsayılan)',
                'large': '3. Büyük (l)',
                'extra_large': '4. Çok Büyük (x)',
                'enter_choice_1_4': 'Seçiminizi girin (1-4, varsayılan: 2): ',
                
                # Dataset Setup
                'hierarchical_setup_title': '===== Hiyerarşik Çoklu Veri Seti Kurulumu =====',
                'config_file_path': 'Konfigürasyon dosyası yolu (varsayılan: config_datasets.yaml): ',
                'config_not_found': '❌ Konfigürasyon dosyası bulunamadı: {file}',
                'ensure_config_exists': 'Lütfen config_datasets.yaml dosyasının mevcut dizinde olduğundan emin olun',
                'system_info': '📊 Sistem Bilgileri:',
                'config_loaded': '✅ Konfigürasyon yüklendi: {file}',
                'available_groups': '📁 Mevcut gruplar: {count}',
                
                # Training Setup
                'training_setup_title': '===== Hiyerarşik Model Eğitim Kurulumu =====',
                'dataset_config': 'Veri seti konfigürasyonu:',
                'hierarchical_multi_recommended': '1) Hiyerarşik çoklu veri seti (Önerilen)',
                'single_roboflow_legacy': '2) Tek Roboflow veri seti (Eski)',
                'select_option_1_2': 'Seçenek [1-2] (varsayılan: 1): ',
                'please_select_1_2': '❌ Lütfen 1 veya 2 seçin.',
                
                # Project Categories
                'project_category': 'Proje kategorisi:',
                'hierarchical_agricultural_recommended': '1) Hiyerarşik Tarımsal AI (Önerilen)',
                'disease_detection': '2) Hastalık Tespiti',
                'pest_detection': '3) Zararlı Tespiti',
                'mixed_agricultural': '4) Karma Tarımsal',
                'custom': '5) Özel',
                'select_category_1_5': 'Kategori seçin [1-5] (varsayılan: 1): ',
                'please_select_1_5': '❌ Lütfen 1-5 arası seçin.',
                'enter_custom_category': 'Özel kategori adı girin: ',
                
                # Training Parameters
                'epoch_count_recommended': 'Epoch sayısı [100-2000 önerilen] (varsayılan: {default}): ',
                'epoch_count_legacy': 'Epoch sayısı [100-1000 önerilen] (varsayılan: 300): ',
                'positive_number': '❌ Lütfen pozitif bir sayı girin.',
                'valid_number': '❌ Lütfen geçerli bir sayı girin.',
                
                # Model Selection
                'select_model_size_training': 'Model boyutunu seçin:',
                'yolo11s_small': '1) yolo11s.pt - Küçük (en hızlı, düşük doğruluk)',
                'yolo11m_medium': '2) yolo11m.pt - Orta (dengeli)',
                'yolo11l_large': '3) yolo11l.pt - Büyük (yüksek doğruluk, yavaş) [Hiyerarşik için önerilen]',
                'yolo11x_extra': '4) yolo11x.pt - Çok Büyük (en yüksek doğruluk, en yavaş)',
                'select_model_1_4_default_3': 'Model seçin [1-4] (varsayılan: 3): ',
                'select_model_1_4_default_2': 'Model seçin [1-4] (varsayılan: 2): ',
                
                # Batch and Image Size
                'batch_size': 'Batch boyutu (varsayılan: {default}, düşük RAM için küçük): ',
                'image_size': 'Görüntü boyutu (varsayılan: {default}, 32\'nin katı olmalı): ',
                'multiple_of_32': '❌ Lütfen 32\'nin katı olan pozitif bir sayı girin.',
                
                # Google Drive Settings
                'drive_save_settings': 'Google Drive kaydetme ayarları:',
                'save_results_to_drive': 'Eğitim sonuçlarını Google Drive\'a kaydet? (e/h, varsayılan: e): ',
                'save_directory_drive': 'Kaydetme dizini (varsayılan: {default}): ',
                'models_will_be_saved': '📁 Modeller şuraya kaydedilecek: {path}',
                
                # Hyperparameters
                'use_hyperparameter_file': 'Hiperparametre dosyası kullan (hyp.yaml)? (e/h, varsayılan: e): ',
                
                # Training Parameters Summary
                'selected_training_params': '===== Seçilen Eğitim Parametreleri =====',
                'dataset_type': 'Veri seti tipi: {type}',
                'dataset_group': 'Veri seti grubu: {group}',
                'target_samples_per_class': 'Sınıf başına hedef örnek: {count:,}',
                'output_directory': 'Çıktı dizini: {dir}',
                'model': 'Model: {model}',
                'epochs': 'Epoch: {epochs}',
                'batch_size_display': 'Batch boyutu: {size}',
                'image_size_display': 'Görüntü boyutu: {size}',
                'device': 'Cihaz: {device}',
                'category': 'Kategori: {category}',
                'drive_save_path': 'Drive kaydetme yolu: {path}',
                'proceed_with_params': 'Bu parametrelerle devam et? (e/h): ',
                'setup_cancelled': '❌ Kurulum iptal edildi.',
                
                # Training Process
                'installing_packages': '📦 Gerekli paketler yüklüyor...',
                'processing_datasets': '===== Hiyerarşik Çoklu Veri Setleri İşleniyor =====',
                'downloading_datasets': '1️⃣ Veri setleri indiriliyor...',
                'dataset_download_failed': '❌ Veri seti indirme başarısız!',
                'creating_class_mapping': '2️⃣ Hiyerarşik sınıf haritalaması oluşturuluyor...',
                'no_classes_mapped': '❌ Hiçbir sınıf haritalandırılamadı!',
                'main_classes_created': '✅ {count} ana sınıf oluşturuldu',
                'merging_datasets': '3️⃣ Veri setleri hiyerarşik yapıyla birleştiriliyor...',
                'dataset_merging_failed': '❌ Veri seti birleştirme başarısız!',
                'hierarchical_processing_completed': '✅ Hiyerarşik çoklu veri seti işleme tamamlandı!',
                'merged_dataset': '📁 Birleştirilmiş veri seti: {dir}',
                'yaml_file': '📄 YAML dosyası: merged_dataset.yaml',
                'class_mapping': '🏷️  Sınıf haritası: unified_class_mapping.json',
                'final_dataset_stats': '📊 Son Veri Seti İstatistikleri:',
                'total_samples': '   Toplam örnek: {count:,}',
                'main_classes': '   Ana sınıflar: {count}',
                'samples_per_class': '   Sınıf başına örnek: {count:,} (ortalama)',
                
                # Training Execution
                'memory_before_training': 'Eğitim Öncesi',
                'memory_after_training': 'Eğitim Sonrası',
                'starting_hierarchical_training': '🚀 Hiyerarşik model eğitimi başlatılıyor...',
                'training_completed': '✅ Eğitim başarıyla tamamlandı!',
                'results': '📊 Sonuçlar: {results}',
                'training_failed': '❌ Eğitim başarısız veya kesildi.',
                
                # Google Drive Save
                'saving_to_drive': '💾 Modeller Google Drive\'a kaydediliyor...',
                'models_saved_successfully': '✅ Modeller Google Drive\'a başarıyla kaydedildi.',
                'models_save_failed': '❌ Modeller Google Drive\'a kaydedilemedi.',
                'save_partial_results': 'Kısmi eğitim sonuçlarını Google Drive\'a kaydet? (e/h, varsayılan: e): ',
                'saving_partial_results': '💾 Kısmi sonuçlar Google Drive\'a kaydediliyor...',
                'partial_results_saved': '✅ Kısmi sonuçlar Google Drive\'a kaydedildi.',
                'partial_results_failed': '❌ Kısmi sonuçlar kaydedilemedi.',
                
                # Testing
                'hierarchical_detection_unavailable': '❌ Hiyerarşik tespit araçları mevcut değil.',
                'enter_model_path': 'Eğitilmiş model yolunu girin (örn: runs/train/exp/weights/best.pt): ',
                'model_file_not_found': '❌ Model dosyası bulunamadı.',
                'enter_test_image_path': 'Test görüntüsü yolunu girin: ',
                'test_image_not_found': '❌ Test görüntüsü bulunamadı.',
                'running_hierarchical_detection': '🔍 Hiyerarşik tespit çalıştırılıyor...',
                'annotated_image_saved': '✅ Açıklamalı görüntü kaydedildi: {path}',
                'hierarchical_detection_error': '❌ Hiyerarşik tespit testi sırasında hata: {error}',
                
                # General
                'exiting': '👋 Çıkılıyor...',
                'invalid_option_exiting': '❌ Geçersiz seçenek. Çıkılıyor...',
                'interrupted_by_user': '⚠️  Kullanıcı tarafından kesildi. Çıkılıyor...',
                'error_occurred': '❌ Bir hata oluştu: {error}',
                'process_completed': '✅ İşlem tamamlandı.',
                
                # Language Selection
                'language_selection': '🌍 Dil Seçimi / Language Selection',
                'select_language': 'Lütfen dilinizi seçin / Please select your language:',
                'turkish': '1) Türkçe',
                'english': '2) English',
                'language_choice': 'Seçiminiz / Your choice (1-2): ',
                'invalid_language': '❌ Geçersiz seçim / Invalid choice. Türkçe kullanılacak / Using Turkish.',
            },
            
            'en': {
                # Main Menu
                'main_title': '🌱 YOLO11 Hierarchical Multi-Dataset Training Framework',
                'main_subtitle': '🎯 Advanced Agricultural AI with Hierarchical Detection',
                'main_menu': '🚀 Main Menu:',
                'option_download': '📥 Download YOLO11 models',
                'option_training': '🎛️  Training setup and execution',
                'option_test': '🔍 Test hierarchical detection (requires trained model)',
                'option_exit': '❌ Exit',
                'select_option': 'Select option (1-4): ',
                
                # Model Download
                'model_download_title': '===== YOLO11 Model Download =====',
                'save_directory': 'Save directory (default: {default}): ',
                'download_options': 'Download options:',
                'single_model': '1. Download single model',
                'all_detection': '2. Download all detection models',
                'all_models': '3. Download all models (all types)',
                'your_choice': 'Your choice (1-3): ',
                
                # Model Types
                'select_model_type': 'Select model type:',
                'detection_default': '1. Detection (default)',
                'segmentation': '2. Segmentation',
                'classification': '3. Classification',
                'pose': '4. Pose',
                'obb': '5. OBB (Oriented Bounding Box)',
                'enter_choice_1_5': 'Enter choice (1-5, default: 1): ',
                
                # Model Sizes
                'select_model_size': 'Select model size:',
                'small': '1. Small (s)',
                'medium_default': '2. Medium (m) (default)',
                'large': '3. Large (l)',
                'extra_large': '4. Extra Large (x)',
                'enter_choice_1_4': 'Enter choice (1-4, default: 2): ',
                
                # Dataset Setup
                'hierarchical_setup_title': '===== Hierarchical Multi-Dataset Setup =====',
                'config_file_path': 'Config file path (default: config_datasets.yaml): ',
                'config_not_found': '❌ Config file not found: {file}',
                'ensure_config_exists': 'Please ensure config_datasets.yaml exists in the current directory',
                'system_info': '📊 System Information:',
                'config_loaded': '✅ Config loaded: {file}',
                'available_groups': '📁 Available groups: {count}',
                
                # Training Setup
                'training_setup_title': '===== Hierarchical Model Training Setup =====',
                'dataset_config': 'Dataset configuration:',
                'hierarchical_multi_recommended': '1) Hierarchical multi-dataset (Recommended)',
                'single_roboflow_legacy': '2) Single Roboflow dataset (Legacy)',
                'select_option_1_2': 'Select option [1-2] (default: 1): ',
                'please_select_1_2': '❌ Please select 1 or 2.',
                
                # Project Categories
                'project_category': 'Project category:',
                'hierarchical_agricultural_recommended': '1) Hierarchical Agricultural AI (Recommended)',
                'disease_detection': '2) Disease Detection',
                'pest_detection': '3) Pest Detection',
                'mixed_agricultural': '4) Mixed Agricultural',
                'custom': '5) Custom',
                'select_category_1_5': 'Select category [1-5] (default: 1): ',
                'please_select_1_5': '❌ Please select 1-5.',
                'enter_custom_category': 'Enter custom category name: ',
                
                # Training Parameters
                'epoch_count_recommended': 'Epochs [100-2000 recommended] (default: {default}): ',
                'epoch_count_legacy': 'Epochs [100-1000 recommended] (default: 300): ',
                'positive_number': '❌ Please enter a positive number.',
                'valid_number': '❌ Please enter a valid number.',
                
                # Model Selection
                'select_model_size_training': 'Select model size:',
                'yolo11s_small': '1) yolo11s.pt - Small (fastest, lower accuracy)',
                'yolo11m_medium': '2) yolo11m.pt - Medium (balanced)',
                'yolo11l_large': '3) yolo11l.pt - Large (high accuracy, slower) [Recommended for hierarchical]',
                'yolo11x_extra': '4) yolo11x.pt - Extra Large (highest accuracy, slowest)',
                'select_model_1_4_default_3': 'Select model [1-4] (default: 3): ',
                'select_model_1_4_default_2': 'Select model [1-4] (default: 2): ',
                
                # Batch and Image Size
                'batch_size': 'Batch size (default: {default}, smaller for low RAM): ',
                'image_size': 'Image size (default: {default}, must be multiple of 32): ',
                'multiple_of_32': '❌ Please enter a positive number that\'s a multiple of 32.',
                
                # Google Drive Settings
                'drive_save_settings': 'Google Drive save settings:',
                'save_results_to_drive': 'Save training results to Google Drive? (y/n, default: y): ',
                'save_directory_drive': 'Save directory (default: {default}): ',
                'models_will_be_saved': '📁 Models will be saved to: {path}',
                
                # Hyperparameters
                'use_hyperparameter_file': 'Use hyperparameter file (hyp.yaml)? (y/n, default: y): ',
                
                # Training Parameters Summary
                'selected_training_params': '===== Selected Training Parameters =====',
                'dataset_type': 'Dataset type: {type}',
                'dataset_group': 'Dataset group: {group}',
                'target_samples_per_class': 'Target samples per class: {count:,}',
                'output_directory': 'Output directory: {dir}',
                'model': 'Model: {model}',
                'epochs': 'Epochs: {epochs}',
                'batch_size_display': 'Batch size: {size}',
                'image_size_display': 'Image size: {size}',
                'device': 'Device: {device}',
                'category': 'Category: {category}',
                'drive_save_path': 'Drive save path: {path}',
                'proceed_with_params': 'Proceed with these parameters? (y/n): ',
                'setup_cancelled': '❌ Setup cancelled.',
                
                # Training Process
                'installing_packages': '📦 Installing required packages...',
                'processing_datasets': '===== Processing Hierarchical Multi-Datasets =====',
                'downloading_datasets': '1️⃣ Downloading datasets...',
                'dataset_download_failed': '❌ Dataset download failed!',
                'creating_class_mapping': '2️⃣ Creating hierarchical class mapping...',
                'no_classes_mapped': '❌ No classes could be mapped!',
                'main_classes_created': '✅ Created {count} main classes',
                'merging_datasets': '3️⃣ Merging datasets with hierarchical structure...',
                'dataset_merging_failed': '❌ Dataset merging failed!',
                'hierarchical_processing_completed': '✅ Hierarchical multi-dataset processing completed!',
                'merged_dataset': '📁 Merged dataset: {dir}',
                'yaml_file': '📄 YAML file: merged_dataset.yaml',
                'class_mapping': '🏷️  Class mapping: unified_class_mapping.json',
                'final_dataset_stats': '📊 Final Dataset Statistics:',
                'total_samples': '   Total samples: {count:,}',
                'main_classes': '   Main classes: {count}',
                'samples_per_class': '   Samples per class: {count:,} (average)',
                
                # Training Execution
                'memory_before_training': 'Before Training',
                'memory_after_training': 'After Training',
                'starting_hierarchical_training': '🚀 Starting hierarchical model training...',
                'training_completed': '✅ Training completed successfully!',
                'results': '📊 Results: {results}',
                'training_failed': '❌ Training failed or was interrupted.',
                
                # Google Drive Save
                'saving_to_drive': '💾 Saving models to Google Drive...',
                'models_saved_successfully': '✅ Models saved to Google Drive successfully.',
                'models_save_failed': '❌ Failed to save models to Google Drive.',
                'save_partial_results': 'Save partial training results to Google Drive? (y/n, default: y): ',
                'saving_partial_results': '💾 Saving partial results to Google Drive...',
                'partial_results_saved': '✅ Partial results saved to Google Drive.',
                'partial_results_failed': '❌ Failed to save partial results.',
                
                # Testing
                'hierarchical_detection_unavailable': '❌ Hierarchical detection utils not available.',
                'enter_model_path': 'Enter trained model path (e.g., runs/train/exp/weights/best.pt): ',
                'model_file_not_found': '❌ Model file not found.',
                'enter_test_image_path': 'Enter test image path: ',
                'test_image_not_found': '❌ Test image not found.',
                'running_hierarchical_detection': '🔍 Running hierarchical detection...',
                'annotated_image_saved': '✅ Annotated image saved to: {path}',
                'hierarchical_detection_error': '❌ Error during hierarchical detection test: {error}',
                
                # General
                'exiting': '👋 Exiting...',
                'invalid_option_exiting': '❌ Invalid option. Exiting...',
                'interrupted_by_user': '⚠️  Interrupted by user. Exiting...',
                'error_occurred': '❌ An error occurred: {error}',
                'process_completed': '✅ Process completed.',
                
                # Language Selection
                'language_selection': '🌍 Dil Seçimi / Language Selection',
                'select_language': 'Lütfen dilinizi seçin / Please select your language:',
                'turkish': '1) Türkçe',
                'english': '2) English',
                'language_choice': 'Seçiminiz / Your choice (1-2): ',
                'invalid_language': '❌ Geçersiz seçim / Invalid choice. Using English.',
            }
        }
    
    def set_language(self, language_code):
        """Set the current language"""
        if language_code in self.translations:
            self.current_language = language_code
            return True
        return False
    
    def get_text(self, key, **kwargs):
        """Get translated text for the current language"""
        try:
            text = self.translations[self.current_language].get(key, key)
            if kwargs:
                return text.format(**kwargs)
            return text
        except (KeyError, AttributeError):
            return key
    
    def interactive_language_selection(self):
        """Interactive language selection at startup"""
        print("\n" + "="*70)
        print("🌍 Dil Seçimi / Language Selection")
        print("="*70)
        
        # Auto-select Turkish (1) as default
        self.set_language('tr')
        print("✅ Varsayılan olarak Türkçe seçildi.")
        return self.current_language

# Global language manager instance
language_manager = LanguageManager()

def get_text(key, **kwargs):
    """Global function to get translated text"""
    return language_manager.get_text(key, **kwargs)

def set_language(language_code):
    """Global function to set language"""
    return language_manager.set_language(language_code)

def select_language():
    """Global function for interactive language selection"""
    return language_manager.interactive_language_selection()
