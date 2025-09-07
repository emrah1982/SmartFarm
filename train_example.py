#!/usr/bin/env python3
# train_example.py - Example training script with safe Drive integration

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from config_utils import get_default_batch_size

# Import our training utilities
from training_utils import train

# Basit bir model tanımla
class SimpleModel(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, num_classes=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def create_sample_data(num_samples=1000, input_size=20, num_classes=2):
    """Örnek veri kümesi oluştur"""
    # Rastgele veri oluştur
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)
    
    # Eğitim ve test olarak ayır
    split = int(0.8 * num_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # PyTorch veri yükleyicilerine dönüştür
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    batch_size = get_default_batch_size()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def main():
    print("🚀 Eğitim Örneği Başlatılıyor...\n")
    
    # Parametreler
    input_size = 20
    hidden_size = 50
    num_classes = 2
    num_epochs = 20
    learning_rate = 0.001
    
    # Modeli oluştur
    model = SimpleModel(input_size, hidden_size, num_classes)
    
    # Kayıp fonksiyonu ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Örnek veri oluştur
    train_loader, val_loader = create_sample_data()
    
    print(f"📊 Eğitim verisi: {len(train_loader.dataset)} örnek")
    print(f"📊 Doğrulama verisi: {len(val_loader.dataset)} örnek\n")
    
    # Eğitimi başlat
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        model_dir='runs/simple_model/weights',
        model_name='simple_classifier',
        use_drive=True,  # Drive entegrasyonunu etkinleştir
        save_interval=1  # Her epoch'ta kaydet
    )

if __name__ == "__main__":
    main()
