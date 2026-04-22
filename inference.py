"""
inference.py
============
Load trained model and classify a single IQ sample or batch.
Useful for demonstrating the model in interviews or demos.

Usage:
    python inference.py --sample_path sample.npy
    python inference.py --demo   # runs on synthetic signal
"""

import numpy as np
import torch
import argparse
import json
from train import RFModulationClassifier, MODULATIONS


def load_model(checkpoint_path, num_classes=11, device='cpu'):
    model = RFModulationClassifier(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model.to(device)


def predict(model, iq_sample, device='cpu'):
    """
    iq_sample: numpy array of shape (2, 128) — I/Q channels
    Returns predicted class name and confidence scores.
    """
    x = torch.tensor(iq_sample, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 2, 128)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = probs.argmax()
    return MODULATIONS[pred_idx], probs


def generate_synthetic_bpsk(num_samples=128, snr_db=10):
    """
    Generate a synthetic BPSK signal for demo purposes.
    BPSK: symbols are +1 or -1 on the I channel, Q channel is noise only.
    """
    symbols = np.random.choice([-1, 1], size=num_samples).astype(np.float32)
    signal_power = np.mean(symbols ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=num_samples).astype(np.float32)

    I = symbols + noise
    Q = np.random.normal(0, np.sqrt(noise_power), size=num_samples).astype(np.float32)
    return np.stack([I, Q])  # (2, 128)


def main():
    parser = argparse.ArgumentParser(description='RF Modulation Classifier Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--sample_path', type=str, default=None, help='Path to .npy IQ sample (2, 128)')
    parser.add_argument('--demo', action='store_true', help='Run on synthetic BPSK signal')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(args.checkpoint, device=device)

    if args.demo:
        print("Running demo on synthetic BPSK signal (SNR=10dB)...")
        iq = generate_synthetic_bpsk(snr_db=10)
    elif args.sample_path:
        iq = np.load(args.sample_path)
        assert iq.shape == (2, 128), f"Expected shape (2, 128), got {iq.shape}"
    else:
        print("No input provided. Use --demo or --sample_path. Running demo by default.")
        iq = generate_synthetic_bpsk(snr_db=10)

    pred_class, probs = predict(model, iq, device)

    print(f"\nPredicted Modulation: {pred_class}")
    print("\nConfidence Scores:")
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx[:5]:
        bar = '█' * int(probs[i] * 40)
        print(f"  {MODULATIONS[i]:<10} {probs[i]:.3f}  {bar}")


if __name__ == '__main__':
    main()
