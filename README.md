# Reproducing and Extending nnU-Net on TCGA-LGG: Lightweight Models for Brain Tumor Segmentation in Low-Resource African Healthcare Settings

> Reproducibility study + responsible AI extensions for the MLRC (Machine Learning Reproducibility Challenge)

---

## Project Summary

We reproduce the nnU-Net 2D full-resolution baseline on the TCGA-LGG MRI Segmentation Dataset (110 lower-grade glioma patients, 5 institutions, sourced via The Cancer Imaging Archive / Kaggle). We then introduce extensions targeting low-resource clinical deployment: model compression, robustness to degraded scans, cross-institution fairness analysis, and responsible AI discussion.

---

## Deadlines

| Phase | Target | Status |
|---|---|---|
| Phase 0+1: Setup & Data | May 23 | Done |
| Phase 2: Reproduction | May 27 | Training on Kaggle |
| Phase 3: Extensions | June 2 | In progress |
| Phase 4: Paper | June 4 | In progress |

---

## Dataset

**TCGA-LGG MRI Segmentation** (mateuszbuda/lgg-mri-segmentation on Kaggle)
- 110 patients, 5 institutions (TCGA_CS, TCGA_DU, TCGA_FG, TCGA_HT, TCGA_TM)
- 3,929 MRI slices total: 1,373 with tumor mask, 2,556 without
- Sequences: FLAIR, T1, T1Gd (we use FLAIR only for baseline)
- Binary segmentation masks

Download from Kaggle:
```
kaggle datasets download mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d data/raw/
```

---

## Environment Setup

```bash
conda create -n lgg_seg python=3.10
conda activate lgg_seg
pip install -r requirements.txt
```

For nnU-Net, set environment variables:
```bash
export nnUNet_raw="$(pwd)/nnunet_data"
export nnUNet_preprocessed="$(pwd)/nnunet_preprocessed"
export nnUNet_results="$(pwd)/nnunet_results"
```

On Windows (PowerShell):
```powershell
$env:nnUNet_raw = "$PWD\nnunet_data"
$env:nnUNet_preprocessed = "$PWD\nnunet_preprocessed"
$env:nnUNet_results = "$PWD\nnunet_results"
```

---

## Phase 0+1: Data Preparation

### Step 1: Extract and explore
```bash
python prepare_dataset.py --datapath data/raw/kaggle_3m --outputpath nnunet_data/Dataset001_TCGALGG
```

This script:
- Reads all patient folders from `kaggle_3m/`
- Extracts FLAIR slices and corresponding masks
- Converts PNG/TIF to NIfTI (.nii.gz)
- Writes `imagesTr/`, `labelsTr/`, and `dataset.json` in nnU-Net format

### Step 2: Analyze dataset
```bash
python analyze_dataset.py --datapath data/raw/kaggle_3m
```

Outputs institution breakdown, tumor prevalence, slice counts.

### Step 3: Fingerprint and preprocess
```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

---

## Phase 2: Reproduction (Run on Kaggle GPU)

Three team members each train one fold:
```bash
# Person A
nnUNetv2_train Dataset001_TCGALGG 2d 0

# Person B
nnUNetv2_train Dataset001_TCGALGG 2d 1

# Person C
nnUNetv2_train Dataset001_TCGALGG 2d 2
```

Target metrics (from prior literature):
- Dice coefficient: 0.82-0.92
- IoU: 0.70-0.85
- Hausdorff Distance: < 10 mm

After all 3 folds, collect results:
```bash
nnUNetv2_find_best_configuration Dataset001_TCGALGG -c 2d
python aggregate_results.py --results_dir nnunet_results/Dataset001_TCGALGG/nnUNetTrainer__nnUNetPlans__2d
```

---

## Phase 3: Extensions

### Lightweight Models
```bash
python extensions/quantize_model.py          # FP16 and INT8 quantization
python extensions/distill.py                 # Knowledge distillation to smaller U-Net
python extensions/lightweight_unet.py        # Train reduced-channel U-Net from scratch
```

### Robustness Testing
```bash
python extensions/robustness_test.py         # Add noise, blur, downsample, missing slices
```

### Fairness & Cross-Institution Analysis
```bash
python extensions/fairness_analysis.py       # Per-institution and per-subgroup Dice
```

### Visualisation
```bash
python visualize_results.py                  # Generate all paper figures
```

---

## Repository Structure

```
mlrc-tracks-TCGA-LGG-comm/
├── prepare_dataset.py          # Kaggle -> nnU-Net format conversion
├── analyze_dataset.py          # EDA and institution breakdown
├── aggregate_results.py        # Average metrics across 3 folds
├── visualize_results.py        # Publication figures
├── requirements.txt
├── extensions/
│   ├── quantize_model.py       # FP16 / INT8 compression
│   ├── distill.py              # Knowledge distillation
│   ├── lightweight_unet.py     # Smaller U-Net baseline
│   ├── robustness_test.py      # Scan degradation simulation
│   └── fairness_analysis.py    # Cross-institution equity analysis
├── nnunet_data/
│   └── Dataset001_TCGALGG/
│       ├── imagesTr/           # FLAIR NIfTI inputs
│       ├── labelsTr/           # Binary mask NIfTIs
│       └── dataset.json
├── figures/                    # Generated plots
├── results/                    # Fold metrics JSONs
└── paper.md                    # Full paper write-up
```

---

## Key Results (filled after training)

| Model | Dice | IoU | HD95 | Size | CPU-runnable |
|---|---|---|---|---|---|
| nnU-Net 2D (baseline) | - | - | - | ~200 MB | No |
| nnU-Net FP16 | - | - | - | ~100 MB | Slow |
| nnU-Net INT8 | - | - | - | ~50 MB | Yes |
| Distilled small U-Net | - | - | - | ~5 MB | Yes |

---

## Citation

```
@misc{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm},
  author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  journal={Computers in Biology and Medicine},
  year={2019}
}
```
