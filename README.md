# Hemorrhage Segmentation with U-Net

## Overview

This project trains a modified U-Net to perform pixel-level segmentation of intracranial hemorrhages in CT head scans. The pipeline loads brain window CT renders directly from disk, generates binary segmentation masks from expert polygon annotations, balances the training set via median-based bidirectional resampling, and trains a U-Net with a combined Dice + weighted binary cross-entropy loss function. The model is checkpointed on validation Mean Intersection over Union (MeanIoU) and evaluated with sensitivity, specificity, precision, F1/Dice, per-sample IoU, and a pixel-level confusion matrix.

The project went through several iterations:

1. **v1 (disk-based)** — 6-step pipeline with intermediate files saved to disk (window stacking → skull stripping → mask generation → split → train → evaluate)
2. **v2 (in-memory)** — Consolidated to 4 steps by loading directly from Renders/ with no disk intermediates. Added exclusion lists for flagged/corrupt images. Fixed a mask interpolation bug (bilinear → nearest-neighbor) and a grayscale conversion bug (BGR weights applied to non-BGR channels → direct brain window loading).
3. **v3 (clustering + ROI)** — Added spectral clustering as a second input channel and adaptive ROI two-pass inference. Removed after evaluation — results coincided with a preprocessing bug that corrupted input data, making effectiveness uninterpretable. Documented separately.

The final model uses the **v2 in-memory pipeline** with the brain window fix and median-based class balancing.

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
├── HemorrhageLabels/                 ← annotation CSVs + labels
│   ├── hemorrhage-labels.csv
│   ├── flagged.txt                   ← image IDs to exclude (one per line)
│   ├── Results_Epidural Hemorrhage Detection_*.csv
│   ├── Results_Intraparenchymal Hemorrhage Detection_*.csv
│   ├── Results_Subarachnoid Hemorrhage Detection_*.csv
│   ├── Results_Subdural Hemorrhage Detection_*.csv
│   ├── Results_Multiple Hemorrhage Detection_*.csv
│   └── Results_Brain Hemorrhage Tracing_*.csv
├── great_results/                    ← all outputs (auto-created)
│   ├── hemorrhage_unet.h5           ← best model weights
│   ├── training_curves.png          ← loss / accuracy / MeanIoU plots
│   ├── prediction_samples/          ← 4-panel overlay images
│   ├── cached_arrays/               ← numpy arrays for fast reruns
│   ├── test_results.txt             ← human-readable test summary
│   ├── test_metrics.csv             ← overall test metrics
│   ├── per_sample_iou.csv           ← per-image metrics
│   └── confusion_matrix.png         ← pixel-level confusion matrix
├── grid_results/                     ← hyperparameter search outputs
│   ├── summary.csv                  ← comparison of all configurations
│   └── dice<W>_bce<W>/             ← per-configuration subfolder
│       ├── model.h5
│       ├── training_curves.png
│       ├── history.csv
│       └── prediction_samples/
├── hemorrhage_segmentation_unet.py   ← main pipeline
├── grid_search.py                    ← hyperparameter grid search
├── evaluate_test.py                  ← detailed test evaluation
└── spectral_clustering_roi_writeup.md ← documentation of tested approaches
```

## Requirements

```
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn tqdm
```

## Pipeline Steps

### Step 1 — Load from Renders and build arrays

For each scan across all 7 classes (6 hemorrhage types + normal):

1. Read the brain window JPEG directly as grayscale using `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` — no color conversion needed
2. Resize to 256×256 with bilinear interpolation, normalise to [0, 1]
3. For hemorrhage scans: parse expert polygon annotations from the CSV files, rasterise them onto a 3-channel copy of the scan using `cv2.drawContours`, convert to single-channel grayscale, resize to 256×256 with nearest-neighbor interpolation to preserve crisp binary edges
4. For normal scans: create an all-zero mask (no annotation parsing)
5. Skip any images in the exclusion list (flagged IDs from `flagged.txt` + the known corrupt file `ID_6431af929.jpg`)
6. Cache all arrays to `great_results/cached_arrays/` for fast reruns

### Step 2 — Train / validation / test split + augmentation

1. Split data 70% train / 10% validation / 20% test using `train_test_split` with a fixed random seed
2. Compute the median positive sample count across all hemorrhage classes
3. Classes above the median are randomly downsampled
4. Classes below the median are augmented to reach the target using random rotations (±40°), width/height shifts (±20%), shear (±20%), zoom (±20%), and horizontal flips
5. Result: a balanced training set at roughly half the size of full upsampling

### Step 3 — Build and train U-Net

Architecture:
- **Encoder:** 4 blocks with filters [64, 128, 256, 512], each block = Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU → MaxPool 2×2
- **Bottleneck:** 1024 filters at 16×16 spatial resolution
- **Decoder:** Symmetric to encoder with transposed convolutions (stride 2) for upsampling, skip connections concatenate encoder features at each level
- **Output head:** Conv3×3 (64) → Conv3×3 (64) → Conv3×3 (2 classes)
- **Input:** 256×256×1 grayscale, **Output:** 256×256×2 logits (background, hemorrhage)

Loss function: `DICE_WEIGHT × dice_loss + weighted_cross_entropy`
- **Dice loss:** `1 - (2 × |intersection| + 1) / (|predicted| + |true| + 1)` — measures overlap between predicted and true hemorrhage regions
- **Weighted cross-entropy:** per-pixel cross-entropy where each hemorrhage pixel is weighted `BCE_FG_WEIGHT` times more than background pixels
- Best configuration from grid search: `DICE_WEIGHT = 6`, `BCE_FG_WEIGHT = 5`

Training:
- Optimizer: Adam (learning rate 1e-4)
- Model checkpointed on highest validation MeanIoU (not lowest loss)
- Early stopping: halts if MeanIoU hasn't improved by 0.005 for 20 epochs
- Maximum 30 epochs

### Step 4 — Evaluate and visualise

Saves training curves (loss, accuracy, MeanIoU) and 10 prediction sample images. Each prediction image has 4 panels:
1. Input scan (grayscale, bone colormap)
2. Ground truth hemorrhage overlay (red on scan)
3. Predicted hemorrhage overlay (green on scan)
4. Overlap comparison with solid opaque colors: yellow = correct (true positive), red = missed (false negative), green = false positive

## How to Run

### Full pipeline:
```bash
cd /path/to/Project
python hemorrhage_segmentation_unet.py
```

### Skip data loading (use cached arrays from previous run):
```bash
python hemorrhage_segmentation_unet.py --step 2
```

### Evaluation only (reload saved model):
```bash
python hemorrhage_segmentation_unet.py --step 4
```

### Hyperparameter grid search:
Tests multiple Dice weight / BCE foreground weight combinations. Edit the `GRID` list at the top of `grid_search.py` to change configurations.
```bash
python grid_search.py
```
Results saved to `grid_results/summary.csv` with per-configuration subfolders containing models, training curves, prediction images, and epoch-by-epoch history.

### Detailed test evaluation:
Loads the trained model and produces comprehensive test metrics including per-sample IoU, sensitivity, specificity, precision, F1/Dice, and a pixel-level confusion matrix.
```bash
python evaluate_test.py
```

## Configurable Hyperparameters

Edit at the top of `hemorrhage_segmentation_unet.py`:

| Parameter | Default | Best (grid search) | Description |
|-----------|---------|-------------------|-------------|
| `IMG_SIZE` | 256 | — | Input image dimensions (256×256) |
| `CHANNELS` | 1 | — | Input channels (grayscale brain window) |
| `OUTPUT_CLASSES` | 2 | — | Background + hemorrhage |
| `BATCH_SIZE` | 4 | — | Training batch size |
| `EPOCHS` | 30 | — | Maximum training epochs |
| `PATIENCE` | 20 | — | Early stopping patience (monitors val MeanIoU) |
| `MIN_DELTA` | 0.005 | — | Minimum MeanIoU improvement to count as progress |
| `DICE_WEIGHT` | 3.0 | 6.0 | Multiplier on Dice loss component |
| `BCE_FG_WEIGHT` | 10.0 | 5.0 | Per-pixel weight for hemorrhage in cross-entropy |
| `RANDOM_SEED` | 42 | — | Seed for all random processes |
| `NORMAL_SAMPLE` | 1000 | — | Number of normal images to include |

To disable the Dice component entirely, set `DICE_WEIGHT = 0.0`. To disable foreground weighting, set `BCE_FG_WEIGHT = 1.0`. This gives plain unweighted cross-entropy for comparison.

## Evaluation Metrics

The test evaluation script (`evaluate_test.py`) produces:

- **Pixel accuracy** — fraction of correctly classified pixels (misleadingly high due to class imbalance)
- **MeanIoU** — intersection over union averaged across both classes; the primary metric for segmentation quality
- **Sensitivity (recall)** — TP / (TP + FN); what fraction of hemorrhage pixels the model found
- **Specificity** — TN / (TN + FP); what fraction of background pixels were correctly identified
- **Precision** — TP / (TP + FP); of pixels predicted as hemorrhage, how many actually were
- **F1 / Dice coefficient** — harmonic mean of precision and sensitivity; mathematically identical to the Dice similarity coefficient
- **Per-sample IoU** — individual IoU for every test image, useful for identifying which scans the model struggles with
- **Confusion matrix** — pixel-level TP/FP/FN/TN counts with RdYlGn colormap

## Exclusion List

Images can be excluded from training by adding their filenames (one per line) to `HemorrhageLabels/flagged.txt`. The corrupt file `ID_6431af929.jpg` is always excluded. The count of excluded images is printed at the start of Step 1.

## Reproducibility

All random processes use `RANDOM_SEED` (default 42):
- Python `random.seed()` — normal image sampling
- NumPy `np.random.seed()` — augmentation sample selection, downsampling
- TensorFlow `tf.random.set_seed()` — weight initialisation, data shuffling, dropout
- `PYTHONHASHSEED` environment variable — Python hash randomisation
- `train_test_split(random_state=RANDOM_SEED)` — all data splits

Change the seed in one place at the top of `hemorrhage_segmentation_unet.py` and all three scripts follow.

## Experimental Approaches (Tested and Removed)

Two additional preprocessing techniques were implemented and tested but not included in the final model. Full documentation is in `spectral_clustering_roi_writeup.md`.

### Spectral clustering

Downsampled each scan to 64×64, ran scikit-learn's `SpectralClustering` (6 clusters, nearest-neighbor affinity graph), upsampled labels back to 256×256, and stacked as a second input channel alongside the grayscale scan. The U-Net received 2-channel input. Removed because it coincided with a grayscale conversion bug that corrupted the input data, making the results uninterpretable.

### Adaptive ROI two-pass inference

At inference time, ran the same U-Net in two passes: Pass A on the full 256×256 image, and Pass B on 64×64 patches extracted around high-intensity cluster regions (above the 75th percentile). Multiple patches per image were supported. The final mask was the pixel-wise maximum of both passes. Removed alongside spectral clustering due to the same preprocessing bug.

## Key Bugs Found and Fixed

1. **Mask interpolation** — `tf.image.resize` default bilinear interpolation blurred binary masks (0/255) to intermediate values. Fixed with `method='nearest'`.

2. **BGR grayscale conversion** — When switching from the disk pipeline to in-memory loading, `img_4ch[:, :, :3]` gave channels [subdural, brain, contrast], but `cv2.cvtColor(COLOR_BGR2GRAY)` applied BGR weights (0.114, 0.587, 0.299) to non-BGR channels, producing a meaningless weighted average. Fixed by loading the brain window directly as grayscale: `cv2.imread(brain_path, cv2.IMREAD_GRAYSCALE)`.

3. **Prediction overlay colors** — Blending mask colors with the scan caused bright areas to appear as false yellow overlap. Fixed with solid opaque colors using explicit set logic (overlap/true_only/pred_only).

## References

See `references.bib` for BibTeX citations covering:
- Shi & Malik (2000) — Normalized Cuts / spectral clustering
- Ng, Jordan & Weiss (2001) — Spectral clustering framework
- Zhang et al. (2021) — Fast spectral clustering for large images
- Shirokikh et al. (2021) — Adaptive ROI for medical image segmentation
- Ronneberger et al. (2015) — U-Net architecture
- Dice (1945) — Dice coefficient
- Sørensen (1948) — Sørensen coefficient
