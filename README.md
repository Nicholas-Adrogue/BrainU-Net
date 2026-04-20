# Hemorrhage Segmentation with U-Net + Spectral Clustering + Adaptive ROI (v3)

## Overview

This pipeline trains a modified U-Net to perform pixel-level segmentation of intracranial hemorrhages in CT head scans. It adds two preprocessing stages designed to help the model locate small hemorrhage regions:

1. **Spectral clustering** groups similar pixels into clusters, providing the model with pre-computed region similarity information as a second input channel.
2. **Adaptive ROI extraction** identifies candidate hemorrhage regions from high-intensity clusters and runs a focused second prediction pass on 64×64 patches, where the hemorrhage fills a much larger proportion of the image.

## What Changed from v2

| | v2 (in-memory) | v3 (clustering + ROI) |
|---|---|---|
| **Input channels** | 1 (grayscale) | 2 (grayscale + cluster map) |
| **Preprocessing** | Load & grayscale only | Load → spectral clustering → 2-channel stack |
| **Inference** | Single full-image pass | Two-pass: full image + ROI patches, stitched |
| **ROI handling** | None | Adaptive 64×64 patches from high-intensity clusters |
| **Steps** | 4 | 5 |
| **New hyperparameters** | — | N_CLUSTERS, CLUSTER_DS, ROI_INTENSITY_PERCENTILE, ROI_PAD, PATCH_SIZE |

## Pipeline Steps

### Step 1 — Load, Cluster, Build Arrays
For each scan:
1. Load 4 CT window renders from `Renders/`, collapse to 3-channel BGR, convert to grayscale
2. Downsample to 64×64, run spectral clustering (6 clusters by default), upsample labels back to 256×256
3. Normalise cluster labels to [0, 1]
4. Stack as 2-channel input: channel 0 = grayscale scan, channel 1 = cluster map
5. Parse expert polygon annotations into binary segmentation masks (nearest-neighbor resize)

Flagged and corrupt images are automatically excluded.

### Step 2 — Train / Val / Test Split + Augmentation
Splits data 70/10/20 with a fixed random seed. Upsamples under-represented hemorrhage classes using random augmentations (rotation, shift, shear, zoom, flip).

### Step 3 — Train U-Net
Builds and trains a U-Net on full 256×256 × 2-channel images:
- **Encoder:** 64 → 128 → 256 → 512 filters with BatchNorm + ReLU + MaxPool
- **Bottleneck:** 1024 filters at 16×16
- **Decoder:** Symmetric with transposed convolutions and skip connections
- **Loss:** Dice + weighted BCE (both weights configurable; set DICE_WEIGHT=0 and BCE_FG_WEIGHT=1 for plain cross-entropy)
- **Checkpointing:** Saves model with highest validation MeanIoU

### Step 4 — Two-Pass Inference (at prediction time)
For each test image, two prediction passes are run using the same trained model:

**Pass A — Full image:** Run the model on the complete 256×256 × 2-channel input. Produces a full-resolution prediction mask.

**Pass B — ROI patches:** Analyze the cluster map to find clusters whose mean scan intensity exceeds the 75th percentile (configurable). For each candidate cluster:
- Extract a 64×64 bounding box centered on the cluster region
- Run the same U-Net on this patch (hemorrhage fills a much larger fraction of the patch)
- Place the patch prediction back into a full-size mask at the original coordinates
- Multiple ROIs per image are supported; overlapping patches take the max prediction

**Final mask = pixel-wise max(Pass A, Pass B).** If either pass predicts hemorrhage at a pixel, it's kept.

### Step 5 — Evaluation & Visualisation
Saves training curves and 4-panel prediction images using the two-pass predictions:
- Panel 1: Input scan (grayscale)
- Panel 2: Ground truth hemorrhage overlay (red)
- Panel 3: Two-pass predicted overlay (green)
- Panel 4: Overlap (yellow = correct, red = missed, green = false positive)

## Directory Structure

```
Project/
├── Renders/                          ← raw CT window renders (input)
│   ├── epidural/
│   │   ├── brain_bone_window/
│   │   ├── brain_window/
│   │   ├── max_contrast_window/
│   │   └── subdural_window/
│   ├── intraparenchymal/             (same sub-dirs)
│   ├── subarachnoid/
│   ├── intraventricular/
│   ├── multi/
│   ├── subdural/
│   └── normal/
├── HemorrhageLabels/
│   ├── hemorrhage-labels.csv
│   ├── flagged.txt
│   └── Results_*.csv                 (per-class annotation files)
├── great_results/                    ← all outputs
├── grid_results/                     ← grid search outputs
├── hemorrhage_segmentation_unet.py   ← main pipeline
├── grid_search.py                    ← hyperparameter search
└── evaluate_test.py                  ← detailed test evaluation
```

## Requirements

```
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn tqdm
```

scikit-learn provides `SpectralClustering` — it's already a dependency from earlier versions.

## How to Run

### Full pipeline:
```bash
cd /path/to/Project
python hemorrhage_segmentation_unet.py
```

### Evaluation only (reload model, two-pass predictions):
```bash
python hemorrhage_segmentation_unet.py --step 5
```

### Hyperparameter grid search:
```bash
python grid_search.py
```

### Detailed test evaluation with confusion matrix:
```bash
python evaluate_test.py
```

## Configurable Hyperparameters

Edit at the top of `hemorrhage_segmentation_unet.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 256 | Full image dimensions |
| `PATCH_SIZE` | 64 | ROI patch dimensions |
| `CHANNELS` | 2 | Input channels (grayscale + cluster map) |
| `OUTPUT_CLASSES` | 2 | Background + hemorrhage |
| `BATCH_SIZE` | 4 | Training batch size |
| `EPOCHS` | 30 | Maximum training epochs |
| `PATIENCE` | 20 | Early stopping patience (monitors val MeanIoU) |
| `DICE_WEIGHT` | 3.0 | Dice loss multiplier (set to 0 to disable) |
| `BCE_FG_WEIGHT` | 10.0 | Foreground pixel weight in BCE (set to 1 to disable) |
| `RANDOM_SEED` | 42 | Seed for all random processes |
| `NORMAL_SAMPLE` | 1000 | Number of normal images to sample |
| `N_CLUSTERS` | 6 | Number of spectral clusters per image |
| `CLUSTER_DS` | 64 | Downsample resolution for clustering |
| `ROI_INTENSITY_PERCENTILE` | 75 | Clusters above this intensity percentile become ROI candidates |
| `ROI_PAD` | 8 | Padding (pixels) around ROI bounding boxes |

### Disabling Components

To run with plain unweighted cross-entropy (no Dice, no foreground weighting):
```python
DICE_WEIGHT   = 0.0
BCE_FG_WEIGHT = 1.0
```

The spectral clustering and ROI extraction are always active during Step 1 and Step 4, but the cluster channel's influence on the model depends on what the U-Net learns — if the clusters aren't helpful, the model will learn to ignore channel 1 and rely on channel 0 alone.

## Outputs

### From `hemorrhage_segmentation_unet.py` (saved to `great_results/`):
- `hemorrhage_unet.h5` — best model weights
- `training_curves.png` — loss / accuracy / MeanIoU plots
- `prediction_samples/` — 4-panel overlay images (using two-pass prediction)

### From `evaluate_test.py` (saved to `great_results/`):
- `test_results.txt` — human-readable metrics summary
- `test_metrics.csv` — loss, accuracy, MeanIoU, sensitivity, specificity, precision, F1/Dice
- `per_sample_iou.csv` — per-image metrics with TP/FP/FN/TN counts
- `confusion_matrix.png` — pixel-level confusion matrix (RdYlGn colormap)

### From `grid_search.py` (saved to `grid_results/`):
- `summary.csv` — all configurations ranked by test MeanIoU
- Per-configuration subfolders with models, curves, and prediction images

## How the Spectral Clustering Works

For each image:
1. The 256×256 grayscale scan is downsampled to 64×64 (4,096 pixels)
2. `SpectralClustering` from scikit-learn builds a nearest-neighbor affinity graph (k=10)
3. The graph Laplacian's eigenvectors are computed, and k-means is run in the spectral embedding space to assign 6 cluster labels
4. Labels are upsampled back to 256×256 using nearest-neighbor interpolation
5. Labels are normalised to [0, 1] and stacked as the second input channel

If spectral clustering fails for a particular image (e.g., disconnected graph), it falls back to standard KMeans.

The clustering groups pixels with similar intensity into regions. Hemorrhages tend to appear as bright, dense clusters distinct from the surrounding brain tissue. By providing this pre-computed grouping as an input channel, the U-Net gets a head start on identifying which pixel regions belong together.

## How the Adaptive ROI Works

At inference time (not during training):
1. The cluster map is analyzed to find which clusters have high mean intensity in the original scan
2. Clusters whose mean intensity exceeds the 75th percentile are flagged as candidates
3. For each candidate, a 64×64 bounding box is extracted centered on the cluster's spatial extent
4. Overlapping boxes are deduplicated (keeps unique centers at least 32px apart)
5. Each patch is run through the same U-Net — because the hemorrhage fills a much larger fraction of the 64×64 patch than the 256×256 full image, the model has an easier time detecting it
6. Patch predictions are placed back into the full 256×256 mask
7. The final mask is the pixel-wise maximum of the full-image prediction and all patch predictions

## Reproducibility

All random processes use `RANDOM_SEED` (default 42):
- Python `random`, NumPy, TensorFlow, `PYTHONHASHSEED`
- `train_test_split`, `SpectralClustering`, KMeans fallback
- Change in one place at the top of `hemorrhage_segmentation_unet.py`; all three scripts follow

## Suppressing Warnings

Spectral clustering may emit `UserWarning: Graph is not fully connected` for some images. This is harmless. To suppress:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
```
Add after the imports in `hemorrhage_segmentation_unet.py`.
