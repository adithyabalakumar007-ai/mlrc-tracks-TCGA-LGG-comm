"""
Post-training quantization (FP16 and INT8) of the lightweight U-Net.

Usage (run after training lightweight_unet.py):
    python extensions/quantize_model.py \
        --model_path models/unet_slim_best.pth \
        --variant    slim \
        --datapath   data/raw/kaggle_3m

Outputs:
    models/unet_slim_fp16.pth
    models/unet_slim_int8.pth
    deploy/unet_slim.onnx
    deploy/unet_slim_int8.onnx
    results/quantization_results.json
    figures/quantization_comparison.png
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightweight_unet import UNet, LGGDataset, collect_pairs, dice_score, CHANNEL_CONFIGS

os.makedirs('models',  exist_ok=True)
os.makedirs('deploy',  exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='models/unet_slim_best.pth', type=str)
parser.add_argument('--variant',    default='slim', choices=['full', 'slim', 'micro'])
parser.add_argument('--datapath',   default='data/raw/kaggle_3m', type=str)
args = parser.parse_args()

DEVICE = torch.device('cpu')

def model_file_size(path):
    return os.path.getsize(path) / 1024 / 1024

def measure_latency(model, n=50, half=False):
    model.eval()
    dummy = torch.randn(1, 1, 256, 256)
    if half: dummy = dummy.half()
    with torch.no_grad():
        for _ in range(5): model(dummy)
        t = time.time()
        for _ in range(n): model(dummy)
    return (time.time() - t) / n * 1000

# ── Load model and test data ──────────────────────────────────────────────────
if not os.path.exists(args.model_path):
    print(f"Model not found: {args.model_path}. Run lightweight_unet.py --mode train first.")
    exit(1)

print("Loading model...")
model_fp32 = UNet(CHANNEL_CONFIGS[args.variant])
model_fp32.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=False))
model_fp32.eval()

pairs = collect_pairs(args.datapath)
np.random.seed(42)
val_pairs = pairs[int(0.85*len(pairs)):]
val_ds = LGGDataset(val_pairs)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

def evaluate(model, loader, half=False):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for imgs, masks in loader:
            if half: imgs = imgs.half()
            preds = model(imgs)
            if half: preds = preds.float()
            total_dice += dice_score(preds, masks)
    return total_dice / len(loader)

torch.save(model_fp32.state_dict(), f'models/unet_{args.variant}_fp32.pth')
fp32_size   = model_file_size(f'models/unet_{args.variant}_fp32.pth')
fp32_lat    = measure_latency(model_fp32)
fp32_dice   = evaluate(model_fp32, val_loader)
print(f"FP32: Dice={fp32_dice:.4f}  Size={fp32_size:.2f} MB  Latency={fp32_lat:.0f} ms")

# ── FP16 ──────────────────────────────────────────────────────────────────────
model_fp16 = UNet(CHANNEL_CONFIGS[args.variant]).half()
model_fp16.load_state_dict(torch.load(args.model_path, map_location='cpu', weights_only=False))
model_fp16 = model_fp16.half().eval()
torch.save(model_fp16.state_dict(), f'models/unet_{args.variant}_fp16.pth')
fp16_size   = model_file_size(f'models/unet_{args.variant}_fp16.pth')
fp16_lat    = measure_latency(model_fp16, half=True)
fp16_dice   = evaluate(model_fp16, val_loader, half=True)
print(f"FP16: Dice={fp16_dice:.4f}  Size={fp16_size:.2f} MB  Latency={fp16_lat:.0f} ms")

# ── INT8 Dynamic Quantization ─────────────────────────────────────────────────
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
)
torch.save(model_int8.state_dict(), f'models/unet_{args.variant}_int8.pth')
int8_size   = model_file_size(f'models/unet_{args.variant}_int8.pth')
int8_lat    = measure_latency(model_int8)
int8_dice   = evaluate(model_int8, val_loader)
print(f"INT8: Dice={int8_dice:.4f}  Size={int8_size:.2f} MB  Latency={int8_lat:.0f} ms")

# ── ONNX Export ───────────────────────────────────────────────────────────────
dummy = torch.randn(1, 1, 256, 256)
onnx_path = f'deploy/unet_{args.variant}.onnx'
torch.onnx.export(model_fp32, dummy, onnx_path,
                  input_names=['image'], output_names=['logits'],
                  opset_version=17, dynamo=False)
onnx_size = model_file_size(onnx_path)
print(f"ONNX: Size={onnx_size:.2f} MB  (Android deployment)")

# ── Results ───────────────────────────────────────────────────────────────────
results = {
    'FP32':  {'dice': fp32_dice, 'size_mb': fp32_size, 'latency_ms': fp32_lat},
    'FP16':  {'dice': fp16_dice, 'size_mb': fp16_size, 'latency_ms': fp16_lat},
    'INT8':  {'dice': int8_dice, 'size_mb': int8_size, 'latency_ms': int8_lat},
    'ONNX':  {'size_mb': onnx_size},
}
with open('results/quantization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── Figure ────────────────────────────────────────────────────────────────────
labels = ['FP32', 'FP16', 'INT8']
dices  = [fp32_dice, fp16_dice, int8_dice]
sizes  = [fp32_size, fp16_size, int8_size]
lats   = [fp32_lat,  fp16_lat,  int8_lat]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
colors = ['#2196F3', '#E91E63', '#FF9800']
for ax, vals, title, ylabel in zip(axes,
        [dices, sizes, lats],
        ['Dice Score', 'Model Size (MB)', 'CPU Latency (ms)'],
        ['Dice', 'MB', 'ms']):
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.02,
                f'{v:.3f}' if ylabel == 'Dice' else f'{v:.1f}',
                ha='center', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
axes[0].axhline(0.82, color='red', linestyle=':', lw=1, label='Target 0.82')
axes[0].legend(fontsize=8)
plt.suptitle(f'Quantization Results: {args.variant.capitalize()} U-Net', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/quantization_comparison.png', dpi=150)
plt.close()
print("Saved figures/quantization_comparison.png")
print("Saved results/quantization_results.json")
