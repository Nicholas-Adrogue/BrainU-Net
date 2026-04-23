Intracranial Hemorrhage Detection & Segmentation
Multi-Modal Deep Learning with U-Net, Hint Maps, and CLIP
Overview
This project implements a comprehensive deep learning system for intracranial hemorrhage analysis from CT scans, combining:
Pixel-level segmentation (U-Net)
Multi-label classification (6 hemorrhage types)
Spatial reasoning via engineered hint maps
Semantic understanding via CLIP embeddings
Systematic ablation studies for model validation
Originally developed as a segmentation pipeline, the system has evolved into a multi-branch architecture where segmentation outputs are reused as structured features for classification.
System Architecture
The final classifier uses a three-branch fusion model:
1. CT Backbone (Visual Features)
Model: DenseNet121
Input: 4 CT window channels:
Bone
Brain
Max contrast
Subdural
Initialization:
RadImageNet pretrained weights (radiology-specific)
2. Hint Branch (Spatial Guidance)
Three spatial maps are stacked as input:
Segmentation mask
From U-Net or radiologist annotations
Symmetry map
Highlights left-right asymmetry in brain
Arrow map
Points toward predicted hemorrhage centroids
Processed via a small CNN → pooled → dense representation
3. CLIP Branch (Semantic Context)
Model: openai/clip-vit-base-patch32
Output: 512-dimensional embedding per image
Captures global semantic structure of CT scans
Fusion
All three branches are concatenated and passed through an MLP:
[CT features] + [Hint features] + [CLIP embedding]
                ↓
           Dense layers
                ↓
        Multi-label output (6 classes)
Relationship Between Segmentation and Classification
This repository contains two tightly coupled pipelines:
Pipeline A — Segmentation (Original Core)
CT → annotations → U-Net → hemorrhage masks
Pipeline B — Classification (Extended System)
CT + (segmentation + symmetry + arrows) + CLIP → classifier
Key Insight:
The segmentation model is both:
a standalone task
AND a feature generator for classification
Segmentation Pipeline (Original System)
Overview
A modified U-Net performs pixel-level segmentation of hemorrhage regions from CT scans.
Data Pipeline
Multi-Window Channel Stacking
Each scan is loaded as a 4-channel image:
channels = [bone, brain, max_contrast, subdural]
merged = np.stack(channels, axis=-1)
Annotation Parsing
Polygon annotations parsed from CSV files
Rasterized into binary masks using cv2.drawContours
Masks resized with nearest-neighbor interpolation (critical fix)
Skull Stripping (Contour-Based)
Thresholding + contour detection
Largest contour used as brain region
Background removed
Data Cleaning
Excludes:
Known corrupt files
User-provided flagged.txt
Ensures training stability
Train / Val / Test Split
70% / 10% / 20%
Fixed random seed for reproducibility
Class Balancing
Median-based balancing:
Downsample majority classes
Augment minority classes
Augmentations:
Rotation (±40°)
Shifts (±20%)
Shear, zoom, flips
U-Net Architecture
Encoder: DenseNet121 (RadImageNet pretrained)
Decoder:
Skip connections
Attention gates
Input: 256×256×4
Output: 256×256×1
Segmentation Loss
Combined loss:
Adaptive ROI Focal + Dice + IoU
Focal loss → handles class imbalance
Dice loss → enforces overlap
IoU loss → penalizes false positives
Segmentation Metrics
Dice coefficient (primary)
IoU
Precision / Recall
Pixel confusion matrix
Hint Generation System
Hint maps provide structured spatial priors to the classifier.
1. Segmentation Hint
From:
U-Net predictions
or ground truth masks
Represents hemorrhage location directly
2. Symmetry Hint
Computed via left-right difference:
abs(brain - flipped_brain)
Weighted combination:
Brain window (70%)
Bone window (30%)
Highlights abnormal asymmetry
3. Arrow Hint
Generated via:
CT → U-Net → predicted mask → contours → centroids → arrows
Each arrow:
Starts from image edge
Points toward hemorrhage center
Saved as:
./output/arrow_hints/<id>.npy
CLIP Embeddings
CLIP replaces earlier LLM-based approaches:
Fast (~15 minutes)
Produces 512-d embeddings
Encodes global semantic structure
Saved as:
./output/clip_embeddings.npz
Classification Model
Input
CT image (4 channels)
Hint maps (3 channels)
CLIP embedding (512-d)
Loss Functions
Weighted Binary Cross-Entropy
Optional:
Soft Macro F1 Loss
Loss = BCE + λ * SoftF1
Training Strategy
Phase 1
Backbone frozen
Train classifier head
Phase 2
Unfreeze top layers
Fine-tune
Class Imbalance Handling
Per-class weighting
Oversampling
Epidural-specific augmentation
Soft F1 optimization
Ablation Study Framework
ablation.py enables controlled experiments.
What It Tests
Hint types:
bbox vs symmetry vs none
Augmentation strategies
Loss functions (BCE vs BCE+F1)
Learning rates
Training schedules
Data filtering
Example Configs
A_baseline — original strong model
B_symm — symmetry hints only
N1_no_hints — remove hint branch
Z* — best combined configs
Outputs
Macro F1
Hamming loss
Per-class F1
Optimal thresholds
Evaluation Metrics
Classification
Macro F1 (primary)
Hamming loss
Precision / Recall / AUC
Per-class F1
Segmentation
Dice coefficient
IoU
Pixel accuracy
Confusion matrix
How to Run
1. Generate CLIP embeddings
python clip_hints.py
2. Generate arrow hints
python arrow_hints.py
3. Train models
python pretrain.py
4. Run ablation study
python ablation.py
Quick mode:
python ablation.py
Full training:
python ablation.py --full
🔍 Visualization
python visualize_hints.py
Displays:
CT scan
Ground truth
U-Net prediction
Symmetry map
Arrow map
Composite overlays
Key Bugs Found and Fixed
Mask interpolation bug
Bilinear → nearest-neighbor
Incorrect grayscale conversion
Fixed incorrect BGR weighting
Overlay visualization artifacts
Fixed color blending issues
Experimental Approaches (Removed)
Spectral Clustering
Clustered pixels by intensity
Failed due to lack of spatial context
Adaptive ROI
Two-pass inference
Unreliable due to preprocessing issues
Key Insights
Hint maps significantly improve performance
Symmetry is a strong signal for hemorrhage
CLIP adds global context missing from CNNs
Aggressive augmentation helps small datasets
Soft F1 stabilizes class imbalance
Summary
This project evolved from a U-Net segmentation model into a multi-modal medical vision system:
Segmentation → spatial grounding
Hint maps → structured attention
CLIP → semantic understanding
DenseNet → feature extraction
Ablation → scientific validation
All combined into a unified, extensible framework for intracranial hemorrhage analysis.
