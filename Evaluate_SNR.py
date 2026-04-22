"""
evaluate_snr.py
===============
Evaluate model accuracy broken down by SNR level.
This is the standard evaluation methodology for RadioML papers
and the most meaningful metric for RF classification — accuracy
at low SNR is what separates good models from great ones.

Produces: results/snr_accuracy.png + snr_accuracy.json
"""

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict
from torch.utils.data import DataLoader
from train import RFModulationClassifier, RadioMLDataset, MODULATIONS, CONFIG


def evaluate_by_snr(model, data_dict, device):
    """
    Evaluate model accuracy for each SNR level independently.
    Returns dict: {snr_value: accuracy}
    """
    model.eval()
    snr_results = defaultdict(lambda: {'correct': 0, 'total': 0})

    for (mod, snr), samples in data_dict.items():
        if mod not in MODULATIONS:
            continue
        label = MODULATIONS.index(mod)
        X = torch.tensor(np.array(samples), dtype=torch.float32).to(device)
        y = torch.tensor([label] * len(samples), dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(X)
            preds  = logits.argmax(1)

        snr_results[snr]['correct'] += (preds == y).sum().item()
        snr_results[snr]['total']   += len(y)

    snr_acc = {}
    for snr, counts in sorted(snr_results.items()):
        snr_acc[snr] = counts['correct'] / counts['total']

    return snr_acc


def plot_snr_accuracy(snr_acc, save_path):
    snrs = sorted(snr_acc.keys())
    accs = [snr_acc[s] for s in snrs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(snrs, accs, marker='o', linewidth=2, color='steelblue', markersize=6)
    ax.axhline(y=1/len(MODULATIONS), color='red', linestyle='--', alpha=0.7, label=f'Random baseline ({1/len(MODULATIONS):.2f})')
    ax.fill_between(snrs, accs, alpha=0.15, color='steelblue')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('RF Modulation Classification Accuracy vs SNR', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"SNR accuracy plot saved → {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CONFIG['results_path'], exist_ok=True)

    # Load model
    model = RFModulationClassifier(num_classes=len(MODULATIONS)).to(device)
    model.load_state_dict(torch.load(CONFIG['save_path'], map_location=device))
    print("Model loaded.")

    # Load raw data dict (all SNRs)
    with open(CONFIG['data_path'], 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')

    # Evaluate
    snr_acc = evaluate_by_snr(model, data_dict, device)

    print("\nSNR vs Accuracy:")
    print(f"{'SNR (dB)':<12} {'Accuracy':<10}")
    print("-" * 22)
    for snr, acc in sorted(snr_acc.items()):
        print(f"{snr:<12} {acc:.4f}")

    # Save
    plot_snr_accuracy(snr_acc, os.path.join(CONFIG['results_path'], 'snr_accuracy.png'))
    with open(os.path.join(CONFIG['results_path'], 'snr_accuracy.json'), 'w') as f:
        json.dump({str(k): v for k, v in snr_acc.items()}, f, indent=2)


if __name__ == '__main__':
    main()
