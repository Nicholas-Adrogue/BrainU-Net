# Intracranial Hemorrhage Detection and Segmentation
**Multi-Modal Deep Learning with U-Net, Hint Maps, and CLIP**

---

## 1. Overview

This repository implements a full end-to-end system for **intracranial hemorrhage analysis from CT scans**, combining:

- Pixel-level **segmentation** via U-Net
- Multi-label **classification** across 6 hemorrhage types
- Structured **spatial reasoning** via engineered hint maps
- Global **semantic understanding** via CLIP embeddings
- Systematic **ablation studies** for rigorous validation

The project evolved from a pure segmentation pipeline into a multi-branch classification system where segmentation outputs are reused as structured spatial features.

### Key Insight

Segmentation serves a dual role in this architecture:

1. **Standalone task** — detecting and localizing hemorrhage regions
2. **Feature generator** — feeding spatial priors into the downstream classifier

---

## 2. System Architecture

The final model is a **three-branch fusion architecture**.

### 2.1 CT Backbone (Visual Branch)

| Property | Value |
|---|---|
| Model | DenseNet121 |
| Input | 4-channel CT tensor (bone, brain, max contrast, subdural) |
| Initialization | RadImageNet (falls back to ImageNet) |

### 2.2 Hint Branch (Spatial Guidance)

A 3-channel spatial tensor composed of:

- **Segmentation mask** — from U-Net prediction or ground truth
- **Symmetry map** — left-right deviation highlighting asymmetry
- **Arrow map** — directional attention toward hemorrhage centers

Processed via convolutional layers → global pooling → dense projection.

### 2.3 CLIP Branch (Semantic Context)

| Property | Value |
|---|---|
| Model | `openai/clip-vit-base-patch32` |
| Output | 512-dimensional embedding |
| Purpose | Global image semantics and structure |

### 2.4 Fusion

```
[CT features] + [Hint features] + [CLIP embedding]
                        ↓
                  Dense layers
                        ↓
            Multi-label output (6 classes)
```

---

## 3. Pipeline Structure

### 3.1 Segmentation Pipeline

```
CT → preprocessing → annotations → U-Net → segmentation masks
```

### 3.2 Classification Pipeline

```
CT + (segmentation + symmetry + arrows) + CLIP → classifier
```

### 3.3 Integration

Segmentation outputs feed into classification as:
- **Direct input** — mask channel
- **Indirect input** — arrow maps derived from predicted contours

---

## 4. Data Pipeline

### 4.1 Multi-Window CT Stacking

Each scan is converted into a 4-channel tensor:

```python
channels = [bone, brain, max_contrast, subdural]
image = np.stack(channels, axis=-1)
```

### 4.2 Annotation Parsing

- Polygon annotations loaded from CSV
- Converted to binary masks via OpenCV contour drawing
- Supports nested annotation formats

### 4.3 Quality Filtering

Low-quality slices are removed based on:
- Brain content ratio
- Intensity variance

### 4.4 Dataset Splitting

Train / Validation / Test split with a deterministic fixed seed.

---

## 5. Segmentation Module

### 5.1 Architecture

| Property | Value |
|---|---|
| Encoder | DenseNet121 (RadImageNet) |
| Decoder | Skip connections + attention gates + spatial dropout |
| Input shape | `(256, 256, 4)` |
| Output shape | `(256, 256, 1)` |

### 5.2 Loss Function

Compound loss combining three terms:

```
Loss = Focal + Dice + 0.5 × IoU
```

- Adaptive ROI focal loss
- Dice loss
- IoU loss

### 5.3 Training

- Full-resolution training at 256×256
- Online augmentation: flips, brightness jitter
- Early stopping with learning rate scheduling

---

## 6. Hint Generation

### 6.1 Segmentation Hint

Sourced from U-Net predictions or ground truth masks. Represents hemorrhage location directly.

### 6.2 Symmetry Map

```python
symmetry = abs(brain - flipped_brain)
```

Weighted composite:
- Brain window: **70%**
- Bone window: **30%**

Highlights left-right asymmetry, a primary hemorrhage signal.

### 6.3 Arrow Maps

Generated pipeline:

```
CT → U-Net → predicted mask → contours → centroids → arrows
```

- Arrows originate from image edges and point toward hemorrhage centers
- Saved to `./output/arrow_hints/<id>.npy`

---

## 7. CLIP Embeddings

### 7.1 Purpose

- Replaces slower LLM-based semantic approaches
- Provides robust global image understanding

### 7.2 Properties

- 512-dimensional vectors computed from raw CT images
- Captures global structure and context

### 7.3 Storage

Saved to `./output/clip_embeddings.npz`, containing:
- `stems` — scan identifiers
- `embeddings` — corresponding CLIP vectors

---

## 8. Classification Model

### 8.1 Inputs

| Branch | Input |
|---|---|
| CT backbone | 4-channel CT image |
| Hint branch | 3-channel spatial maps |
| CLIP branch | 512-d embedding vector |

### 8.2 Loss Function

```
Loss = BCE + λ × SoftF1
```

- **Primary**: Weighted Binary Cross-Entropy
- **Optional regularization**: Soft Macro F1 Loss

### 8.3 Training Strategy

**Phase 1 — Head training**
- Backbone frozen
- Classifier head trained to convergence

**Phase 2 — Fine-tuning**
- Top backbone layers unfrozen
- End-to-end fine-tuning at reduced learning rate

### 8.4 Class Imbalance Handling

- Per-class loss weighting
- Oversampling of minority classes
- Epidural-specific augmentation
- Soft F1 regularization

---

## 9. Ablation Study Framework

### 9.1 Purpose

Systematically evaluate the contribution of:
- Hint types (segmentation, symmetry, arrows)
- Augmentation strategies
- Loss function combinations
- Learning rate schedules

### 9.2 Configurations

| ID | Description |
|---|---|
| `A_baseline` | Full system with all branches |
| `B_symm` | Symmetry-only hints |
| `N1_no_hints` | Hint branch removed |
| `Z*` | Optimized combinations |

### 9.3 Evaluation Outputs

- Macro F1 (primary metric)
- Hamming loss
- Per-class F1 scores
- Optimal classification thresholds

---

## 10. Visualization

The visualization tool displays side-by-side:
- Raw CT scan
- Ground truth mask
- U-Net prediction
- Symmetry map
- Arrow map
- Combined overlays

---

## 11. How to Run

### Step 1 — Generate CLIP embeddings

```bash
python clip_hints.py
```

### Step 2 — Generate arrow hint maps

```bash
python arrow_hints.py
```

### Step 3 — Train segmentation and classifier

```bash
python pretrain.py
```

### Step 4 — Run ablation study

```bash
# Quick mode
python ablation.py

# Full training
python ablation.py --full
```

### Step 5 — Visualize results

```bash
python visualize_hints.py
```

---

## 12. Directory Structure

```
dcms/
├── segmentation/
├── renders/
└── 02_Contour/

output/
├── models/
├── figures/
├── arrow_hints/
├── arrow_viz/
└── clip_embeddings.npz
```

---

## 13. Key Design Decisions

| Decision | Rationale |
|---|---|
| RadImageNet initialization | Domain-specific pretraining on medical images |
| CLIP over LLM-based semantics | Faster inference, more robust global encoding |
| Hint maps for spatial reasoning | Injects hemorrhage location priors into classifier |
| Multi-branch fusion | Richer representations than single-stream CNN |
| Segmentation as dual-purpose | Used for both prediction and feature extraction |

---

## 14. Evaluation Metrics

### Classification

| Metric | Role |
|---|---|
| Macro F1 | Primary metric |
| Hamming loss | Multi-label error rate |
| Precision / Recall | Per-class diagnostics |
| AUC | Threshold-independent performance |
| Per-class F1 | Fine-grained breakdown |

### Segmentation

| Metric | Description |
|---|---|
| Dice coefficient | Overlap between prediction and ground truth |
| IoU | Intersection over union |
| Pixel accuracy | Overall pixel-level correctness |

---

## 15. System Summary

This system represents a deliberate progression across three stages:

```
U-Net segmentation
        ↓
Spatial feature extraction
        ↓
Multi-modal classification
```

Combining four complementary signals:

- **Segmentation** — spatial grounding
- **Hint maps** — structured attention
- **CLIP** — semantic context
- **DenseNet** — deep visual features

The result is a unified, extensible framework for intracranial hemorrhage detection and analysis, validated through systematic ablation.