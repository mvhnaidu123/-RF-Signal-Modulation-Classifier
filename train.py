"""
RF Signal Modulation Classifier
================================
Classifies radio frequency modulation types from raw IQ samples
using a 1D CNN trained on the RadioML 2016.10a dataset.

Dataset: https://www.deepsig.ai/datasets
Paper reference: O'Shea & Corgan, 2016 - "Convolutional Radio Modulation Recognition Networks"

Author: Your Name
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
import os
import json

# ─── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "data_path": "data/RML2016.10a_dict.pkl",
    "snr_threshold": 0,          # Only train on SNRs >= this value (dB)
    "train_split": 0.8,
    "batch_size": 256,
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "seed": 42,
    "save_path": "checkpoints/best_model.pth",
    "results_path": "results/"
}

MODULATIONS = [
    'AM-DSB', 'AM-SSB', 'WBFM',
    'BPSK', 'QPSK', '8PSK',
    'QAM16', 'QAM64',
    'CPFSK', 'GFSK', 'PAM4'
]

# ─── Dataset ──────────────────────────────────────────────────────────────────

class RadioMLDataset(Dataset):
    """
    Loads RadioML 2016.10a pickle file.
    Each sample: (2, 128) array — row 0 = I channel, row 1 = Q channel.
    Label: modulation class index.
    """

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_radioml(path, snr_threshold=0):
    """
    Parse RadioML pickle into numpy arrays.
    Returns X: (N, 2, 128), y: (N,) integer labels, snrs: (N,)
    """
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    X, y, snrs = [], [], []

    for (mod, snr), samples in data.items():
        if mod not in MODULATIONS:
            continue
        if snr < snr_threshold:
            continue
        label = MODULATIONS.index(mod)
        for sample in samples:
            X.append(sample)          # shape (2, 128)
            y.append(label)
            snrs.append(snr)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    snrs = np.array(snrs)

    print(f"Loaded {len(y):,} samples | {len(MODULATIONS)} classes | SNR >= {snr_threshold} dB")
    return X, y, snrs


def split_data(X, y, train_split=0.8, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(y))
    split = int(len(y) * train_split)
    train_idx, val_idx = idx[:split], idx[split:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ─── Model ────────────────────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    """
    1D Residual block for temporal RF signal features.
    Skip connection handles dimension mismatch with a 1x1 conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class RFModulationClassifier(nn.Module):
    """
    1D ResNet for RF modulation classification.

    Input:  (batch, 2, 128)  — I/Q channels × 128 time steps
    Output: (batch, num_classes)

    Architecture choice: 1D convolutions because modulation features
    (phase transitions, amplitude variations) are temporal patterns
    in the IQ stream, not spatial patterns. ResNet skip connections
    prevent gradient vanishing on deeper feature extraction.
    """

    def __init__(self, num_classes=11):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock1D(64, 64)
        self.layer2 = ResidualBlock1D(64, 128)
        self.layer3 = ResidualBlock1D(128, 128)
        self.layer4 = ResidualBlock1D(128, 256)

        self.pool = nn.AdaptiveAvgPool1d(1)   # Global average pooling → (batch, 256, 1)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── Evaluation Plots ─────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks(range(len(MODULATIONS)))
    ax.set_yticks(range(len(MODULATIONS)))
    ax.set_xticklabels(MODULATIONS, rotation=45, ha='right')
    ax.set_yticklabels(MODULATIONS)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix — RF Modulation Classifier')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_training_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(CONFIG['results_path'], exist_ok=True)

    # Load data
    X, y, snrs = load_radioml(CONFIG['data_path'], CONFIG['snr_threshold'])
    X_train, y_train, X_val, y_val = split_data(X, y, CONFIG['train_split'], CONFIG['seed'])

    train_ds = RadioMLDataset(X_train, y_train)
    val_ds   = RadioMLDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)

    # Model
    model = RFModulationClassifier(num_classes=len(MODULATIONS)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    # Train
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG['save_path'])
            print(f"  ✓ Best model saved (val_acc={val_acc:.3f})")

    # Final evaluation
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=MODULATIONS))

    # Save results
    plot_confusion_matrix(val_labels, val_preds, os.path.join(CONFIG['results_path'], 'confusion_matrix.png'))
    plot_training_curves(history, os.path.join(CONFIG['results_path'], 'training_curves.png'))

    with open(os.path.join(CONFIG['results_path'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\nDone. Results saved to results/")


if __name__ == '__main__':
    main()
