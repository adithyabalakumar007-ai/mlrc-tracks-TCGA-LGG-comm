"""
Exploratory analysis of the TCGA-LGG dataset before nnU-Net conversion.

Usage:
    python analyze_dataset.py --datapath data/raw/kaggle_3m

Outputs:
    figures/dataset_overview.png
    figures/institution_breakdown.png
    figures/tumor_examples.png
    results/dataset_stats.json
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='data/raw/kaggle_3m', type=str)
args = parser.parse_args()

# ── Scan all patients ─────────────────────────────────────────────────────────
patient_dirs = sorted([
    d for d in os.listdir(args.datapath)
    if os.path.isdir(os.path.join(args.datapath, d))
])

records = []
print("Scanning dataset...")
for patient in tqdm(patient_dirs):
    ppath = os.path.join(args.datapath, patient)
    flair_files = sorted([f for f in os.listdir(ppath)
                          if f.endswith('.tif') and '_mask' not in f])
    institution = patient.split('_')[1]  # CS, DU, FG, HT, TM

    for fname in flair_files:
        mask_path = os.path.join(ppath, fname.replace('.tif', '_mask.tif'))
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path).convert('L'))
        flair = np.array(Image.open(os.path.join(ppath, fname)).convert('L'))
        tumor_pixels = int((mask > 0).sum())
        records.append({
            'patient':        patient,
            'institution':    institution,
            'slice':          fname,
            'has_tumor':      tumor_pixels > 0,
            'tumor_pixels':   tumor_pixels,
            'tumor_fraction': tumor_pixels / mask.size,
            'img_h':          flair.shape[0],
            'img_w':          flair.shape[1],
            'flair_mean':     float(flair.mean()),
            'flair_std':      float(flair.std()),
        })

df = pd.DataFrame(records)

# ── Summary stats ─────────────────────────────────────────────────────────────
stats = {
    'total_patients':      len(patient_dirs),
    'total_slices':        len(df),
    'slices_with_tumor':   int(df['has_tumor'].sum()),
    'slices_without_tumor': int((~df['has_tumor']).sum()),
    'institutions':        df['institution'].unique().tolist(),
    'per_institution': {}
}
for inst, grp in df.groupby('institution'):
    stats['per_institution'][inst] = {
        'patients':          int(grp['patient'].nunique()),
        'slices':            int(len(grp)),
        'tumor_slices':      int(grp['has_tumor'].sum()),
        'tumor_fraction_mean': float(grp['tumor_fraction'].mean()),
    }

with open('results/dataset_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("\n--- Dataset Summary ---")
print(f"Patients:          {stats['total_patients']}")
print(f"Total slices:      {stats['total_slices']}")
print(f"  With tumor:      {stats['slices_with_tumor']}")
print(f"  Without tumor:   {stats['slices_without_tumor']}")
print(f"Institutions:      {stats['institutions']}")
print("\nPer institution:")
for inst, s in stats['per_institution'].items():
    print(f"  TCGA_{inst}: {s['patients']} patients, {s['slices']} slices, "
          f"{s['tumor_slices']} tumor ({100*s['tumor_fraction_mean']:.1f}% avg coverage)")

# ── Figure 1: Dataset overview bar chart ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

inst_counts = df.groupby('institution')['patient'].nunique().sort_values(ascending=False)
axes[0].bar(inst_counts.index, inst_counts.values, color='#2196F3', alpha=0.85)
axes[0].set_xlabel('Institution (TCGA_X)', fontsize=11)
axes[0].set_ylabel('Number of Patients', fontsize=11)
axes[0].set_title('Patients per Institution', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

tumor_counts = df['has_tumor'].value_counts()
axes[1].bar(['No Tumor', 'Tumor'], [tumor_counts.get(False, 0), tumor_counts.get(True, 0)],
            color=['#4CAF50', '#F44336'], alpha=0.85)
axes[1].set_ylabel('Number of Slices', fontsize=11)
axes[1].set_title('Slice-Level Class Balance', fontsize=12, fontweight='bold')
for i, v in enumerate([tumor_counts.get(False, 0), tumor_counts.get(True, 0)]):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

tumor_df = df[df['has_tumor']]
axes[2].hist(tumor_df['tumor_fraction'] * 100, bins=40, color='#FF9800', alpha=0.85, edgecolor='white')
axes[2].set_xlabel('Tumor Area (% of slice)', fontsize=11)
axes[2].set_ylabel('Number of Slices', fontsize=11)
axes[2].set_title('Tumor Size Distribution', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.suptitle('TCGA-LGG Dataset Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/dataset_overview.png', dpi=150)
plt.close()
print("\nSaved figures/dataset_overview.png")

# ── Figure 2: Cross-institution equity radar ──────────────────────────────────
inst_df = df.groupby('institution').agg(
    patients=('patient', 'nunique'),
    slices=('slice', 'count'),
    tumor_slices=('has_tumor', 'sum'),
    mean_tumor_frac=('tumor_fraction', 'mean')
).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(inst_df))
ax.bar([i - 0.2 for i in x], inst_df['patients'], width=0.4,
       label='Patients', color='#2196F3', alpha=0.85)
ax2 = ax.twinx()
ax2.bar([i + 0.2 for i in x], inst_df['tumor_slices'] / inst_df['slices'] * 100, width=0.4,
        label='Tumor slice %', color='#F44336', alpha=0.65)
ax.set_xticks(x)
ax.set_xticklabels([f"TCGA_{i}" for i in inst_df['institution']], fontsize=10)
ax.set_ylabel('Number of Patients', fontsize=11, color='#2196F3')
ax2.set_ylabel('Tumor Slice Prevalence (%)', fontsize=11, color='#F44336')
ax.set_title('Cross-Institution Representation\n(proxy for healthcare resource equity)',
             fontsize=12, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/institution_breakdown.png', dpi=150)
plt.close()
print("Saved figures/institution_breakdown.png")

# ── Figure 3: Sample FLAIR slices with masks ──────────────────────────────────
tumor_samples = df[df['has_tumor']].sample(6, random_state=42)
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
for col, (_, row) in enumerate(tumor_samples.iterrows()):
    ppath = os.path.join(args.datapath, row['patient'])
    flair = np.array(Image.open(os.path.join(ppath, row['slice'])).convert('L'))
    mask  = np.array(Image.open(os.path.join(ppath, row['slice'].replace('.tif', '_mask.tif'))).convert('L'))
    axes[0, col].imshow(flair, cmap='gray')
    axes[0, col].set_title(f"TCGA_{row['institution']}", fontsize=8)
    axes[0, col].axis('off')
    overlay = np.zeros((*flair.shape, 3), dtype=np.uint8)
    overlay[..., 0] = flair
    overlay[..., 1] = flair
    overlay[..., 2] = flair
    overlay[mask > 0, 0] = 255
    overlay[mask > 0, 1] = 0
    overlay[mask > 0, 2] = 0
    axes[1, col].imshow(overlay)
    axes[1, col].set_title(f"{row['tumor_fraction']*100:.1f}% tumor", fontsize=8)
    axes[1, col].axis('off')
axes[0, 0].set_ylabel('FLAIR', fontsize=10)
axes[1, 0].set_ylabel('+ Mask', fontsize=10)
fig.suptitle('TCGA-LGG: Sample FLAIR Slices with Tumor Masks (red overlay)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/tumor_examples.png', dpi=150)
plt.close()
print("Saved figures/tumor_examples.png")
print("\nAll outputs saved. See results/dataset_stats.json for full statistics.")
