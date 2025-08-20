#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛑 SmartFarm Early Stopping ve Epoch Management Sistemi

YOLOv8/YOLO11 eğitimi için gelişmiş early stopping, epoch tahmini ve 
aşırı öğrenme (overfitting) önleme sistemi.

Özellikler:
- Adaptive early stopping
- Validation loss tracking
- Learning rate scheduling
- Epoch duration estimation
- Overfitting detection
- Model checkpoint management
- Training progress analytics

Kullanım:
    from early_stopping_system import EarlyStoppingManager
    
    manager = EarlyStoppingManager()
    should_stop = manager.check_early_stopping(val_loss, epoch)
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch bulunamadı. Bazı özellikler çalışmayabilir.")


@dataclass
class TrainingMetrics:
    """Eğitim metrikleri"""
    epoch: int
    train_loss: float
    val_loss: float
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    learning_rate: float = 0.0
    epoch_duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EarlyStoppingConfig:
    """Early stopping konfigürasyonu"""
    patience: int = 50  # Kaç epoch iyileşme beklenecek
    min_delta: float = 0.001  # Minimum iyileşme miktarı
    monitor_metric: str = 'val_loss'  # İzlenecek metrik
    mode: str = 'min'  # 'min' veya 'max'
    restore_best_weights: bool = True
    verbose: bool = True
    
    # Overfitting detection
    overfitting_threshold: float = 0.1  # Train-val loss farkı eşiği
    overfitting_patience: int = 20  # Overfitting için patience
    
    # Learning rate scheduling
    lr_reduction_factor: float = 0.5  # LR azaltma faktörü
    lr_patience: int = 20  # LR azaltma için patience
    min_lr: float = 1e-7  # Minimum learning rate


class EpochEstimator:
    """Epoch süresi ve tahmini hesaplayıcı"""
    
    def __init__(self):
        self.epoch_durations = []
        self.start_times = {}
        self.logger = logging.getLogger(__name__)
    
    def start_epoch(self, epoch: int):
        """Epoch başlangıcını kaydet"""
        self.start_times[epoch] = time.time()
    
    def end_epoch(self, epoch: int) -> float:
        """Epoch bitişini kaydet ve süreyi döndür"""
        if epoch in self.start_times:
            duration = time.time() - self.start_times[epoch]
            self.epoch_durations.append(duration)
            del self.start_times[epoch]
            return duration
        return 0.0
    
    def get_average_epoch_duration(self) -> float:
        """Ortalama epoch süresini hesapla"""
        if not self.epoch_durations:
            return 0.0
        
        # Son 10 epoch'un ortalamasını al (daha doğru tahmin için)
        recent_durations = self.epoch_durations[-10:]
        return np.mean(recent_durations)
    
    def estimate_remaining_time(self, current_epoch: int, total_epochs: int) -> Dict[str, Any]:
        """Kalan süreyi tahmin et"""
        if current_epoch >= total_epochs:
            return {
                'remaining_epochs': 0,
                'estimated_time_seconds': 0,
                'estimated_time_str': "Tamamlandı",
                'completion_time': datetime.now()
            }
        
        remaining_epochs = total_epochs - current_epoch
        avg_duration = self.get_average_epoch_duration()
        
        if avg_duration == 0:
            return {
                'remaining_epochs': remaining_epochs,
                'estimated_time_seconds': 0,
                'estimated_time_str': "Hesaplanıyor...",
                'completion_time': None
            }
        
        estimated_seconds = remaining_epochs * avg_duration
        estimated_time_str = str(timedelta(seconds=int(estimated_seconds)))
        completion_time = datetime.now() + timedelta(seconds=estimated_seconds)
        
        return {
            'remaining_epochs': remaining_epochs,
            'estimated_time_seconds': estimated_seconds,
            'estimated_time_str': estimated_time_str,
            'completion_time': completion_time,
            'avg_epoch_duration': avg_duration
        }
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Eğitim istatistiklerini al"""
        if not self.epoch_durations:
            return {}
        
        durations = np.array(self.epoch_durations)
        
        return {
            'total_epochs_completed': len(durations),
            'total_training_time': np.sum(durations),
            'avg_epoch_duration': np.mean(durations),
            'min_epoch_duration': np.min(durations),
            'max_epoch_duration': np.max(durations),
            'std_epoch_duration': np.std(durations),
            'epochs_per_hour': 3600 / np.mean(durations) if np.mean(durations) > 0 else 0
        }


class OverfittingDetector:
    """Aşırı öğrenme (overfitting) tespit sistemi"""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.overfitting_warnings = 0
        self.logger = logging.getLogger(__name__)
    
    def add_metrics(self, train_loss: float, val_loss: float):
        """Metrik ekle"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def detect_overfitting(self) -> Dict[str, Any]:
        """Overfitting tespit et"""
        if len(self.train_losses) < 10:  # En az 10 epoch gerekli
            return {
                'is_overfitting': False,
                'overfitting_score': 0.0,
                'recommendation': 'Daha fazla epoch gerekli'
            }
        
        # Son N epoch'un ortalamasını al
        window_size = min(10, len(self.train_losses))
        recent_train = np.mean(self.train_losses[-window_size:])
        recent_val = np.mean(self.val_losses[-window_size:])
        
        # Overfitting skoru hesapla
        if recent_train > 0:
            overfitting_score = (recent_val - recent_train) / recent_train
        else:
            overfitting_score = 0.0
        
        is_overfitting = overfitting_score > self.config.overfitting_threshold
        
        if is_overfitting:
            self.overfitting_warnings += 1
        
        # Öneri oluştur
        if overfitting_score > 0.2:
            recommendation = "Güçlü overfitting! Early stopping önerilir."
        elif overfitting_score > 0.1:
            recommendation = "Orta seviye overfitting. Learning rate azaltılabilir."
        elif overfitting_score > 0.05:
            recommendation = "Hafif overfitting belirtisi. İzlemeye devam edin."
        else:
            recommendation = "Overfitting yok. Eğitime devam edebilirsiniz."
        
        return {
            'is_overfitting': is_overfitting,
            'overfitting_score': float(overfitting_score),
            'overfitting_warnings': self.overfitting_warnings,
            'train_loss_avg': float(recent_train),
            'val_loss_avg': float(recent_val),
            'recommendation': recommendation
        }
    
    def plot_loss_curves(self, save_path: str = "loss_curves.png"):
        """Loss eğrilerini çiz"""
        if len(self.train_losses) < 2:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', alpha=0.8)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overfitting score
        plt.subplot(1, 2, 2)
        if len(self.train_losses) >= 10:
            overfitting_scores = []
            for i in range(10, len(self.train_losses) + 1):
                train_avg = np.mean(self.train_losses[i-10:i])
                val_avg = np.mean(self.val_losses[i-10:i])
                if train_avg > 0:
                    score = (val_avg - train_avg) / train_avg
                else:
                    score = 0.0
                overfitting_scores.append(score)
            
            score_epochs = range(10, len(self.train_losses) + 1)
            plt.plot(score_epochs, overfitting_scores, 'g-', label='Overfitting Score')
            plt.axhline(y=self.config.overfitting_threshold, color='r', linestyle='--', 
                       label=f'Threshold ({self.config.overfitting_threshold})')
            plt.xlabel('Epoch')
            plt.ylabel('Overfitting Score')
            plt.title('Overfitting Detection')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Loss curves kaydedildi: {save_path}")


class EarlyStoppingManager:
    """Kapsamlı early stopping ve epoch management sistemi"""
    
    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        self.config = config or EarlyStoppingConfig()
        self.metrics_history: List[TrainingMetrics] = []
        self.best_metric = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        
        # Alt sistemler
        self.epoch_estimator = EpochEstimator()
        self.overfitting_detector = OverfittingDetector(self.config)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Checkpoint yönetimi
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"🛑 Early Stopping Manager başlatıldı:")
        print(f"   📊 Monitör metrik: {self.config.monitor_metric}")
        print(f"   ⏱️ Patience: {self.config.patience} epoch")
        print(f"   📉 Min delta: {self.config.min_delta}")
        print(f"   🎯 Overfitting threshold: {self.config.overfitting_threshold}")
    
    def add_epoch_metrics(self, metrics: TrainingMetrics) -> Dict[str, Any]:
        """Epoch metriklerini ekle ve analiz et"""
        
        # Epoch süresini kaydet
        if metrics.epoch_duration > 0:
            self.epoch_estimator.epoch_durations.append(metrics.epoch_duration)
        
        # Metrikleri kaydet
        self.metrics_history.append(metrics)
        
        # Overfitting detection için loss'ları ekle
        self.overfitting_detector.add_metrics(metrics.train_loss, metrics.val_loss)
        
        # Early stopping kontrolü
        current_metric = getattr(metrics, self.config.monitor_metric.replace('val_', ''))
        
        if self.config.mode == 'min':
            is_improvement = (self.best_metric is None or 
                            current_metric < self.best_metric - self.config.min_delta)
        else:
            is_improvement = (self.best_metric is None or 
                            current_metric > self.best_metric + self.config.min_delta)
        
        if is_improvement:
            self.best_metric = current_metric
            self.best_epoch = metrics.epoch
            self.wait = 0
            
            # En iyi modeli kaydet
            self._save_best_checkpoint(metrics)
            
            if self.config.verbose:
                self.logger.info(f"🎯 Yeni en iyi metrik: {current_metric:.6f} (epoch {metrics.epoch})")
        else:
            self.wait += 1
            
            if self.config.verbose and self.wait % 10 == 0:
                self.logger.info(f"⏳ {self.wait}/{self.config.patience} epoch iyileşme yok")
        
        # Early stopping kontrolü
        if self.wait >= self.config.patience:
            self.should_stop = True
            self.stopped_epoch = metrics.epoch
            
            if self.config.verbose:
                self.logger.info(f"🛑 Early stopping triggered at epoch {metrics.epoch}")
                self.logger.info(f"🏆 En iyi metrik: {self.best_metric:.6f} (epoch {self.best_epoch})")
        
        # Overfitting analizi
        overfitting_info = self.overfitting_detector.detect_overfitting()
        
        # Sonuç özeti
        analysis = {
            'should_stop': self.should_stop,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'wait': self.wait,
            'patience_remaining': max(0, self.config.patience - self.wait),
            'current_metric': current_metric,
            'improvement': is_improvement,
            'overfitting': overfitting_info
        }
        
        return analysis
    
    def estimate_training_completion(self, target_epochs: int) -> Dict[str, Any]:
        """Eğitim tamamlanma tahmini"""
        if not self.metrics_history:
            return {'error': 'Henüz metrik yok'}
        
        current_epoch = self.metrics_history[-1].epoch
        
        # Epoch estimator'dan tahmin al
        time_estimate = self.epoch_estimator.estimate_remaining_time(current_epoch, target_epochs)
        
        # Early stopping tahmini
        if self.should_stop:
            early_stop_estimate = {
                'will_early_stop': True,
                'estimated_stop_epoch': current_epoch,
                'reason': 'Early stopping triggered'
            }
        else:
            # Mevcut trend'e göre early stopping tahmini
            if self.wait > self.config.patience * 0.7:
                estimated_stop_epoch = current_epoch + (self.config.patience - self.wait)
                early_stop_estimate = {
                    'will_early_stop': True,
                    'estimated_stop_epoch': min(estimated_stop_epoch, target_epochs),
                    'reason': f'Patience trend (current wait: {self.wait}/{self.config.patience})'
                }
            else:
                early_stop_estimate = {
                    'will_early_stop': False,
                    'estimated_stop_epoch': target_epochs,
                    'reason': 'Normal completion expected'
                }
        
        return {
            'current_epoch': current_epoch,
            'target_epochs': target_epochs,
            'time_estimate': time_estimate,
            'early_stopping': early_stop_estimate,
            'training_stats': self.epoch_estimator.get_training_statistics()
        }
    
    def get_training_recommendations(self) -> List[str]:
        """Eğitim önerileri"""
        recommendations = []
        
        if not self.metrics_history:
            return ["Henüz yeterli veri yok"]
        
        # Overfitting kontrolü
        overfitting_info = self.overfitting_detector.detect_overfitting()
        if overfitting_info['is_overfitting']:
            recommendations.append(f"⚠️ {overfitting_info['recommendation']}")
        
        # Patience kontrolü
        patience_ratio = self.wait / self.config.patience
        if patience_ratio > 0.8:
            recommendations.append("🛑 Early stopping yaklaşıyor. Model performansını kontrol edin.")
        elif patience_ratio > 0.5:
            recommendations.append("⏳ Uzun süredir iyileşme yok. Learning rate azaltmayı düşünün.")
        
        # Epoch süresi analizi
        stats = self.epoch_estimator.get_training_statistics()
        if stats and stats.get('std_epoch_duration', 0) > stats.get('avg_epoch_duration', 0) * 0.3:
            recommendations.append("📊 Epoch süreleri değişken. Sistem kaynaklarını kontrol edin.")
        
        # Genel öneriler
        if len(self.metrics_history) < 50:
            recommendations.append("📈 Daha kararlı sonuçlar için en az 50-100 epoch eğitim önerilir.")
        
        if not recommendations:
            recommendations.append("✅ Eğitim normal seyrinde devam ediyor.")
        
        return recommendations
    
    def _save_best_checkpoint(self, metrics: TrainingMetrics):
        """En iyi checkpoint'i kaydet"""
        checkpoint_info = {
            'epoch': metrics.epoch,
            'metrics': {
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'mAP50': metrics.mAP50,
                'mAP50_95': metrics.mAP50_95,
                'precision': metrics.precision,
                'recall': metrics.recall
            },
            'best_metric': self.best_metric,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / "best_checkpoint_info.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
    
    def save_training_report(self, save_path: str = "training_report.json"):
        """Eğitim raporunu kaydet"""
        if not self.metrics_history:
            return
        
        # Loss curves çiz
        self.overfitting_detector.plot_loss_curves("training_loss_curves.png")
        
        # Rapor oluştur
        report = {
            'training_summary': {
                'total_epochs': len(self.metrics_history),
                'best_epoch': self.best_epoch,
                'best_metric': self.best_metric,
                'early_stopped': self.should_stop,
                'stopped_epoch': self.stopped_epoch if self.should_stop else None
            },
            'config': {
                'patience': self.config.patience,
                'min_delta': self.config.min_delta,
                'monitor_metric': self.config.monitor_metric,
                'overfitting_threshold': self.config.overfitting_threshold
            },
            'training_statistics': self.epoch_estimator.get_training_statistics(),
            'overfitting_analysis': self.overfitting_detector.detect_overfitting(),
            'recommendations': self.get_training_recommendations(),
            'metrics_history': [
                {
                    'epoch': m.epoch,
                    'train_loss': m.train_loss,
                    'val_loss': m.val_loss,
                    'mAP50': m.mAP50,
                    'mAP50_95': m.mAP50_95,
                    'precision': m.precision,
                    'recall': m.recall,
                    'epoch_duration': m.epoch_duration
                } for m in self.metrics_history
            ]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Eğitim raporu kaydedildi: {save_path}")


def get_optimal_epoch_recommendations(dataset_size: int, 
                                    model_size: str = "medium",
                                    task_complexity: str = "medium") -> Dict[str, Any]:
    """Optimal epoch sayısı önerisi"""
    
    # Base epoch sayıları
    base_epochs = {
        "small": {"simple": 100, "medium": 200, "complex": 300},
        "medium": {"simple": 150, "medium": 300, "complex": 500},
        "large": {"simple": 200, "medium": 400, "complex": 800}
    }
    
    # Dataset boyutuna göre ayarlama
    if dataset_size < 1000:
        size_factor = 0.7
        warning = "Küçük dataset - overfitting riski yüksek"
    elif dataset_size < 5000:
        size_factor = 1.0
        warning = None
    elif dataset_size < 20000:
        size_factor = 1.3
        warning = None
    else:
        size_factor = 1.5
        warning = "Büyük dataset - daha uzun eğitim gerekebilir"
    
    base_epoch = base_epochs[model_size][task_complexity]
    recommended_epochs = int(base_epoch * size_factor)
    
    # Early stopping önerileri
    patience = max(20, recommended_epochs // 10)
    
    return {
        'recommended_epochs': recommended_epochs,
        'min_epochs': recommended_epochs // 2,
        'max_epochs': recommended_epochs * 2,
        'early_stopping_patience': patience,
        'dataset_size': dataset_size,
        'size_factor': size_factor,
        'warning': warning,
        'explanation': f"""
Önerilen epoch sayısı: {recommended_epochs}
- Model boyutu: {model_size}
- Görev karmaşıklığı: {task_complexity}
- Dataset boyutu: {dataset_size}
- Early stopping patience: {patience}

2000 epoch ile başlamak genellikle çok fazladır. 
Önerilen aralık: {recommended_epochs//2}-{recommended_epochs*2} epoch
Early stopping ile otomatik durdurma kullanın.
        """
    }


# Kullanım örneği
if __name__ == "__main__":
    # Early stopping manager oluştur
    config = EarlyStoppingConfig(
        patience=50,
        min_delta=0.001,
        monitor_metric='val_loss',
        overfitting_threshold=0.1
    )
    
    manager = EarlyStoppingManager(config)
    
    # Örnek epoch sayısı önerisi
    dataset_size = 5000  # Kullanıcının dataset boyutu
    recommendations = get_optimal_epoch_recommendations(
        dataset_size=dataset_size,
        model_size="medium",
        task_complexity="medium"
    )
    
    print("🎯 Epoch Önerileri:")
    print(recommendations['explanation'])
    
    # Simüle edilmiş eğitim
    print("\n🧪 Örnek eğitim simülasyonu:")
    for epoch in range(1, 101):
        # Simüle edilmiş metrikler
        train_loss = 1.0 * np.exp(-epoch/50) + np.random.normal(0, 0.05)
        val_loss = 1.1 * np.exp(-epoch/60) + np.random.normal(0, 0.08)
        
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=max(0.01, train_loss),
            val_loss=max(0.01, val_loss),
            mAP50=min(0.95, 0.3 + epoch/150),
            epoch_duration=np.random.uniform(30, 45)
        )
        
        analysis = manager.add_epoch_metrics(metrics)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, "
                  f"Wait = {analysis['wait']}/{config.patience}")
        
        if analysis['should_stop']:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            break
    
    # Rapor kaydet
    manager.save_training_report("example_training_report.json")
    
    # Tamamlanma tahmini
    completion_estimate = manager.estimate_training_completion(300)
    print(f"\n📊 Eğitim Tahmini:")
    print(f"Mevcut epoch: {completion_estimate['current_epoch']}")
    print(f"Tahmini bitiş: {completion_estimate['early_stopping']['estimated_stop_epoch']}")
