#!/usr/bin/env python3
"""
Hemorrhage Segmentation with U-Net + Spectral Clustering + Adaptive ROI
========================================================================
1 – Load scans from Renders/, run spectral clustering, build 2-channel
    input arrays [grayscale, cluster_map] and segmentation masks.
2 – Train / val / test split + augmentation.
3 – Train a single U-Net on full 256×256 images (2 channels).
4 – At inference, run a two-pass prediction:
      Pass A — full-image prediction (256×256)
      Pass B — extract adaptive ROI patches (64×64) from high-intensity
               cluster regions, predict on each, stitch back into full mask
      Final mask = pixel-wise max(pass_A, pass_B)
5 – Evaluate & save results.

Usage:
    python hemorrhage_segmentation_unet.py           # full pipeline
    python hemorrhage_segmentation_unet.py --step 5  # reload model, eval only
"""

import os
import ast
import argparse
from random import sample

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR  = os.path.join(BASE_DIR, 'Renders')
LABEL_DIR   = os.path.join(BASE_DIR, 'HemorrhageLabels')
RESULTS_DIR = os.path.join(BASE_DIR, 'great_results')

CSV_FILES = {
    'epidural':         'Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148.csv',
    'intraparenchymal': 'Results_Intraparenchymal Hemorrhage Detection_2020-11-16_21.39.31.268.csv',
    'subarachnoid':     'Results_Subarachnoid Hemorrhage Detection_2020-11-16_21.36.18.668.csv',
    'subdural':         'Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040.csv',
    'multi':            'Results_Multiple Hemorrhage Detection_2020-11-16_21.36.24.018.csv',
    'intraventricular': 'Results_Brain Hemorrhage Tracing_2020-09-28_15.21.52.597.csv',
}
LABEL_CSV = os.path.join(LABEL_DIR, 'hemorrhage-labels.csv')

CLS_EPIDURAL         = 'epidural'
CLS_INTRAPARENCHYMAL = 'intraparenchymal'
CLS_SUBARACHNOID     = 'subarachnoid'
CLS_INTRAVENTRICULAR = 'intraventricular'
CLS_MULTI            = 'multi'
CLS_SUBDURAL         = 'subdural'
CLS_NORMAL           = 'normal'

ALL_HEMORRHAGE_CLASSES = [
    CLS_EPIDURAL, CLS_INTRAPARENCHYMAL, CLS_SUBARACHNOID,
    CLS_INTRAVENTRICULAR, CLS_MULTI, CLS_SUBDURAL,
]
ALL_CLASSES = ALL_HEMORRHAGE_CLASSES + [CLS_NORMAL]

BONE_WIN     = 'brain_bone_window'
BRAIN_WIN    = 'brain_window'
CONTRAST_WIN = 'max_contrast_window'
SUB_WIN      = 'subdural_window'

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMG_SIZE       = 256
PATCH_SIZE     = 64           # ROI patch size
CHANNELS       = 2            # [grayscale_scan, cluster_map]
OUTPUT_CLASSES = 2
BATCH_SIZE     = 4
BUFFER_SIZE    = 1000
EPOCHS         = 30
PATIENCE       = 20
MIN_DELTA      = 0.005
NORMAL_SAMPLE  = 1000
DICE_WEIGHT    = 3.0          # set to 0 to disable Dice component
BCE_FG_WEIGHT  = 10.0         # set to 1.0 to disable foreground weighting
RANDOM_SEED    = 42
N_CLUSTERS     = 6            # number of spectral clusters
CLUSTER_DS     = 64           # downsample resolution for clustering
ROI_PAD        = 8            # padding around ROI bounding boxes
ROI_INTENSITY_PERCENTILE = 75 # clusters above this intensity percentile are candidates

# ── Exclusion lists ───────────────────────────────────────────────────────────
CORRUPT_IDS = {'ID_6431af929.jpg'}

def _load_flagged_ids(path):
    try:
        with open(path) as fh:
            return {line.strip() for line in fh if line.strip()}
    except FileNotFoundError:
        return set()

FLAGGED_IDS  = _load_flagged_ids(os.path.join(LABEL_DIR, 'flagged.txt'))
EXCLUDED_IDS = FLAGGED_IDS | CORRUPT_IDS

# ── Set global seeds ──────────────────────────────────────────────────────────
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  HELPER FUNCTIONS                                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def read_class_csv(cls_name):
    return pd.read_csv(os.path.join(LABEL_DIR, CSV_FILES[cls_name]))


def load_four_channel_image(class_render_dir, fname):
    ch_bone     = cv2.imread(os.path.join(class_render_dir, BONE_WIN,     fname), cv2.IMREAD_GRAYSCALE)
    ch_brain    = cv2.imread(os.path.join(class_render_dir, BRAIN_WIN,    fname), cv2.IMREAD_GRAYSCALE)
    ch_contrast = cv2.imread(os.path.join(class_render_dir, CONTRAST_WIN, fname), cv2.IMREAD_GRAYSCALE)
    ch_sub      = cv2.imread(os.path.join(class_render_dir, SUB_WIN,      fname), cv2.IMREAD_GRAYSCALE)
    if any(ch is None for ch in [ch_bone, ch_brain, ch_contrast, ch_sub]):
        return None
    return np.stack([ch_sub, ch_brain, ch_contrast, ch_bone], axis=-1)


def build_file_lists(excluded=None):
    if excluded is None:
        excluded = EXCLUDED_IDS
    file_lists, label_list = [], []
    for cls in ALL_CLASSES:
        probe = os.path.join(RENDER_DIR, cls, BRAIN_WIN)
        if not os.path.isdir(probe):
            file_lists.append([])
        else:
            files = sorted(
                f for f in os.listdir(probe)
                if not f.startswith('.') and f not in excluded
            )
            file_lists.append(files)
        label_list.append(cls)
    return file_lists, label_list


# ── Spectral clustering ──────────────────────────────────────────────────────

def compute_cluster_map(gray_img, n_clusters=N_CLUSTERS, ds_size=CLUSTER_DS):
    """Downsample to ds_size×ds_size, run spectral clustering, upsample
    labels back to original resolution. Returns normalised cluster map [0, 1]."""
    h, w = gray_img.shape[:2]
    # Downsample
    small = cv2.resize(gray_img.squeeze(), (ds_size, ds_size), interpolation=cv2.INTER_AREA)
    flat  = small.reshape(-1, 1).astype(np.float64)

    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=10,
            assign_labels='kmeans',
            random_state=RANDOM_SEED,
        )
        labels_small = sc.fit_predict(flat).reshape(ds_size, ds_size)
    except Exception:
        # Fallback: if spectral clustering fails, use simple k-means
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=3)
        labels_small = km.fit_predict(flat).reshape(ds_size, ds_size)

    # Upsample labels back to original size using nearest-neighbor
    labels_full = cv2.resize(labels_small.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_NEAREST)
    # Normalise to [0, 1]
    if labels_full.max() > 0:
        labels_full = labels_full / labels_full.max()
    return labels_full.reshape(h, w, 1)


# ── Adaptive ROI extraction ──────────────────────────────────────────────────

def extract_roi_boxes(gray_img, cluster_map, patch_size=PATCH_SIZE,
                      pad=ROI_PAD, intensity_pctl=ROI_INTENSITY_PERCENTILE):
    """Identify candidate hemorrhage regions from the cluster map.

    For each cluster whose mean intensity in the original scan exceeds the
    given percentile, extract a bounding box. Returns a list of
    (y_start, y_end, x_start, x_end) tuples for 64×64 patches.
    Multiple ROIs can be returned per image."""
    h, w = gray_img.shape[:2]
    cluster_ids = np.unique(cluster_map)
    scan_flat = gray_img.squeeze()
    clust_flat = cluster_map.squeeze()

    # Compute mean intensity per cluster
    cluster_intensities = {}
    for cid in cluster_ids:
        mask = (clust_flat == cid)
        if mask.sum() > 0:
            cluster_intensities[cid] = scan_flat[mask].mean()

    if not cluster_intensities:
        return []

    threshold = np.percentile(list(cluster_intensities.values()), intensity_pctl)

    boxes = []
    for cid, mean_int in cluster_intensities.items():
        if mean_int < threshold:
            continue

        mask = (clust_flat == cid)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        # Bounding box with padding
        y_min = max(0, ys.min() - pad)
        y_max = min(h, ys.max() + pad)
        x_min = max(0, xs.min() - pad)
        x_max = min(w, xs.max() + pad)

        # Center a patch_size×patch_size box on this region
        cy = (y_min + y_max) // 2
        cx = (x_min + x_max) // 2

        y1 = max(0, cy - patch_size // 2)
        x1 = max(0, cx - patch_size // 2)
        y2 = min(h, y1 + patch_size)
        x2 = min(w, x1 + patch_size)

        # Adjust if we hit an edge
        if y2 - y1 < patch_size:
            y1 = max(0, y2 - patch_size)
        if x2 - x1 < patch_size:
            x1 = max(0, x2 - patch_size)

        boxes.append((y1, y2, x1, x2))

    # Deduplicate overlapping boxes (keep unique centers)
    if len(boxes) > 1:
        unique = [boxes[0]]
        for b in boxes[1:]:
            cy, cx = (b[0]+b[1])//2, (b[2]+b[3])//2
            is_dup = False
            for u in unique:
                uy, ux = (u[0]+u[1])//2, (u[2]+u[3])//2
                if abs(cy - uy) < patch_size//2 and abs(cx - ux) < patch_size//2:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(b)
        boxes = unique

    return boxes


# ── Annotation parsing ────────────────────────────────────────────────────────

def _extract_point_list(region):
    pts = []
    for pt in region:
        if not isinstance(pt, str):
            pts.append(list(pt.values()))
    return pts

def _append_region_coords(regions, region):
    pts = _extract_point_list(region)
    if pts:
        regions.append(pts)
    return regions

def _unwrap_nested_label(raw_label):
    parsed = ast.literal_eval(raw_label)
    inner  = np.array(parsed, dtype=object)[0]
    return inner.replace('[]', '')

def parse_annotation_regions(raw_label, is_nested):
    try:
        if is_nested:
            raw_label = _unwrap_nested_label(raw_label)
            if not raw_label:
                return False
        parsed  = ast.literal_eval(raw_label)
        regions = np.array(parsed, dtype=object)
        if regions.size == 0:
            return False
        coords = []
        if regions.ndim > 1:
            for region in regions:
                coords = _append_region_coords(coords, region)
        else:
            if isinstance(regions[0], list):
                for region in regions:
                    coords = _append_region_coords(coords, region)
            else:
                coords = _append_region_coords(coords, regions)
        return coords
    except Exception:
        return False

def resolve_annotation(main_label, fallback_label):
    coords = parse_annotation_regions(main_label, is_nested=False)
    if coords is False:
        coords = parse_annotation_regions(fallback_label, is_nested=True)
        if coords is False:
            coords = []
    return coords

def draw_annotation_mask(region_coords, scan_img):
    filled_mask = np.zeros_like(scan_img, dtype=np.uint8)
    border_img  = np.zeros_like(scan_img, dtype=np.uint8)
    for pts in region_coords:
        if not pts:
            continue
        scaled = (np.array(pts) * np.array(scan_img.shape[:2])).astype(np.int32)
        scaled = scaled.reshape((-1, 1, 2))
        cv2.drawContours(filled_mask, [scaled], 0, (255, 255, 255), thickness=cv2.FILLED)
        cv2.polylines(border_img, [scaled], isClosed=True, color=(255, 255, 255), thickness=3)
    inv_border = cv2.bitwise_not(border_img)
    annotated  = cv2.bitwise_and(inv_border, scan_img)
    return annotated, filled_mask


def resize_and_normalise(raw_img, annotated_img, target_size, seg_mask):
    sz = [target_size, target_size]
    resized_raw       = tf.image.resize(raw_img,       sz).numpy()
    resized_annotated = tf.image.resize(annotated_img, sz).numpy()
    resized_mask      = tf.image.resize(seg_mask,      sz, method='nearest').numpy()
    norm_img       = resized_raw / 255.0
    gray_scan      = cv2.cvtColor(norm_img.astype(np.float32), cv2.COLOR_BGR2GRAY)
    gray_scan      = gray_scan.reshape(target_size, target_size, 1)
    gray_annotated = cv2.cvtColor(resized_annotated.astype(np.float32), cv2.COLOR_BGR2GRAY)
    gray_annotated = gray_annotated.reshape(target_size, target_size, 1)
    gray_mask      = cv2.cvtColor(resized_mask.astype(np.float32), cv2.COLOR_BGR2GRAY)
    gray_mask      = gray_mask.reshape(target_size, target_size, 1)
    return norm_img, gray_scan, gray_annotated, gray_mask


def get_one_hot_label(img_name, class_cols, label_df):
    stem = img_name.replace('.jpg', '')
    if stem not in label_df.index:
        return None
    vec     = label_df.loc[stem][class_cols].to_numpy().copy().astype(np.float32)
    any_idx = class_cols.index('any')
    vec[any_idx] = (vec[any_idx] + 1) % 2
    return vec


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — LOAD, CLUSTER, BUILD 2-CHANNEL ARRAYS                             ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def step1_build_arrays():
    """Load images from Renders/, compute spectral cluster maps, build
    2-channel input arrays [grayscale, cluster_map] and binary masks."""
    print('\n══ Step 1: Loading, clustering, & building arrays ══')
    print(f'  Excluding {len(EXCLUDED_IDS)} flagged/corrupt images')
    print(f'  Spectral clustering: {N_CLUSTERS} clusters at {CLUSTER_DS}×{CLUSTER_DS}')

    scan_lists, scan_folders = build_file_lists()
    label_df = pd.read_csv(LABEL_CSV, index_col='Image')

    display_labels = [lbl for lbl in scan_folders if lbl != CLS_MULTI]
    class_cols     = [lbl.replace(CLS_NORMAL, 'any') for lbl in display_labels]

    for cls, files in zip(scan_folders, scan_lists):
        print(f'    {cls:25s}: {len(files)} images')

    two_ch_scans = []  # will hold (256, 256, 2) arrays
    seg_masks    = []
    targets      = []

    for img_names, cls in zip(scan_lists, scan_folders):
        if not img_names:
            continue

        ann_df = None
        if cls == CLS_INTRAVENTRICULAR:
            ann_df = read_class_csv(cls)
            ann_df = ann_df[['Origin', 'ROI', 'All Annotations']]
        elif cls not in (CLS_NORMAL,):
            ann_df = read_class_csv(cls)
            ann_df = ann_df[['Origin', 'Majority Label', 'Correct Label', 'All Labels']]

        cls_render_dir = os.path.join(RENDER_DIR, cls)

        for img_name in tqdm(img_names, desc=cls, unit='img'):
            if img_name in EXCLUDED_IDS:
                continue

            img_4ch = load_four_channel_image(cls_render_dir, img_name)
            if img_4ch is None:
                continue
            scan = img_4ch[:, :, :3]  # 3-channel BGR for mask generation

            # Resize and get grayscale scan + mask
            if cls == CLS_NORMAL:
                empty_mask = np.zeros_like(scan, dtype=np.uint8)
                _, gray_scan, _, proc_mask = resize_and_normalise(
                    scan, scan, IMG_SIZE, empty_mask
                )
                label_vec = get_one_hot_label(img_name, class_cols, label_df)
                if label_vec is None:
                    continue
            else:
                row_matches = ann_df[ann_df.Origin == img_name]
                if cls == CLS_INTRAVENTRICULAR:
                    row_iter = zip(
                        row_matches['ROI'],
                        row_matches['All Annotations'],
                        row_matches['Origin'],
                    )
                else:
                    row_iter = zip(
                        row_matches['Correct Label'],
                        row_matches['Majority Label'],
                        row_matches['All Labels'],
                    )
                combined_regions = []
                for cl_val, ml_val, al_val in row_iter:
                    if isinstance(cl_val, str):
                        region_coords = parse_annotation_regions(cl_val, is_nested=False)
                        if not region_coords:
                            region_coords = resolve_annotation(ml_val, al_val)
                    elif isinstance(ml_val, str):
                        region_coords = resolve_annotation(ml_val, al_val)
                    else:
                        region_coords = []
                    combined_regions += region_coords

                if not combined_regions:
                    continue

                overlaid, mask = draw_annotation_mask(combined_regions, scan)
                _, gray_scan, _, proc_mask = resize_and_normalise(
                    scan, overlaid, IMG_SIZE, mask
                )
                label_vec = get_one_hot_label(img_name, class_cols, label_df)
                if label_vec is None:
                    continue

            # Compute spectral cluster map on the grayscale scan
            cluster_map = compute_cluster_map(gray_scan)

            # Stack: channel 0 = grayscale, channel 1 = cluster map
            two_ch = np.concatenate([gray_scan, cluster_map], axis=-1)

            two_ch_scans.append(two_ch)
            seg_masks.append(proc_mask)
            targets.append(label_vec)

    two_ch_scans = np.array(two_ch_scans)
    seg_masks    = np.array(seg_masks)
    targets      = np.array(targets)

    print(f'  Scans  : {two_ch_scans.shape}  (channels: grayscale + cluster)')
    print(f'  Masks  : {seg_masks.shape}')
    print(f'  Labels : {targets.shape}')

    # Cache arrays to disk so --step 2 can skip clustering
    cache_dir = os.path.join(RESULTS_DIR, 'cached_arrays')
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, 'scans.npy'),  two_ch_scans)
    np.save(os.path.join(cache_dir, 'masks.npy'),   seg_masks)
    np.save(os.path.join(cache_dir, 'targets.npy'), targets)
    # Save display_labels as text
    with open(os.path.join(cache_dir, 'display_labels.txt'), 'w') as f:
        f.write('\n'.join(display_labels))
    print(f'  Cached arrays to {cache_dir}/')

    return two_ch_scans, seg_masks, targets, display_labels


def load_cached_arrays():
    """Load arrays saved by step1_build_arrays, skipping clustering."""
    cache_dir = os.path.join(RESULTS_DIR, 'cached_arrays')
    scans_path = os.path.join(cache_dir, 'scans.npy')

    if not os.path.exists(scans_path):
        print(f'  ✗ No cached arrays found at {cache_dir}/')
        print('    Run without --step first to generate them.')
        return None, None, None, None

    print(f'\n══ Loading cached arrays from {cache_dir}/ ══')
    scans   = np.load(os.path.join(cache_dir, 'scans.npy'))
    masks   = np.load(os.path.join(cache_dir, 'masks.npy'))
    targets = np.load(os.path.join(cache_dir, 'targets.npy'))
    with open(os.path.join(cache_dir, 'display_labels.txt')) as f:
        display_labels = [line.strip() for line in f if line.strip()]

    print(f'  Scans  : {scans.shape}')
    print(f'  Masks  : {masks.shape}')
    print(f'  Labels : {targets.shape}')
    return scans, masks, targets, display_labels


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — TRAIN / VAL / TEST SPLIT + AUGMENTATION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def step2_split_and_augment(scans, masks, labels, display_labels):
    print('\n══ Step 2: Splitting & augmenting ══')

    X_tr, X_test, M_tr, M_test, y_tr, y_test = train_test_split(
        scans, masks, labels, test_size=0.2, random_state=RANDOM_SEED
    )
    X_train, X_val, M_train, M_val, y_train, y_val = train_test_split(
        X_tr, M_tr, y_tr, test_size=0.125, random_state=RANDOM_SEED
    )
    print(f'  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}')

    aug_gen = ImageDataGenerator(
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
    )
    pos_counts   = [np.sum(y_train[:, i] == 1) for i in range(y_train.shape[1])]
    target_count = max(pos_counts)
    aug_images, aug_masks, aug_targets = [], [], []

    for cls_idx in range(y_train.shape[1]):
        pos_idx  = np.where(y_train[:, cls_idx] == 1)[0]
        n_needed = target_count - len(pos_idx)
        if n_needed <= 0:
            continue
        chosen = np.random.choice(pos_idx, n_needed, replace=True)
        for sample_idx in chosen:
            patch = X_train[sample_idx].reshape((1,) + X_train[sample_idx].shape)
            for aug_batch, _ in aug_gen.flow(patch, np.zeros(1), batch_size=1):
                aug_images.append(aug_batch[0])
                aug_masks.append(M_train[sample_idx])
                aug_targets.append(y_train[sample_idx])
                break

    if aug_images:
        X_train = np.concatenate([X_train, np.array(aug_images)],  axis=0)
        M_train = np.concatenate([M_train, np.array(aug_masks)],   axis=0)
        y_train = np.concatenate([y_train, np.array(aug_targets)], axis=0)

    print(f'  After augmentation — Train: {X_train.shape[0]}')
    return X_train, M_train, y_train, X_val, M_val, y_val, X_test, M_test, y_test


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — BUILD & TRAIN U-NET (accepts variable input size)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def build_unet(input_shape, num_classes):
    """U-Net with batch-normalised encoder/decoder.
    Input shape uses None for spatial dims so the same model can handle
    both 256×256 full images and 64×64 ROI patches."""
    inp = layers.Input(shape=input_shape)
    x   = layers.BatchNormalization()(inp)
    x   = layers.Activation('relu')(x)

    skips   = []
    filters = [64, 128, 256, 512]
    for n_f in filters:
        x = layers.Conv2D(n_f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(n_f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        skips.append(x)
        x = layers.MaxPooling2D((2, 2))(x)

    bn = filters[-1] * 2
    x = layers.Conv2D(bn, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(bn, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    skips.reverse()
    filters.reverse()
    for n_f, skip in zip(filters, skips):
        x = layers.Conv2DTranspose(n_f, 3, strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.concatenate([x, skip])
        x = layers.Conv2D(n_f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(n_f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.Conv2D(num_classes, 3, padding='same')(x)
    return keras.Model(inputs=inp, outputs=x)


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_mask = tf.argmax(y_pred, axis=-1)[..., tf.newaxis]
        return super().update_state(tf.argmax(y_true, axis=-1), pred_mask, sample_weight)


def make_tf_datasets(X_train, M_train, X_val, M_val, X_test, M_test):
    def _quantise(m):
        return (m > 127).astype(np.int32)
    ds_train = tf.data.Dataset.from_tensor_slices(
        (X_train.astype(np.float32), _quantise(M_train))
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices(
        (X_val.astype(np.float32), _quantise(M_val))
    ).batch(BATCH_SIZE)
    ds_test = tf.data.Dataset.from_tensor_slices(
        (X_test.astype(np.float32), _quantise(M_test))
    ).batch(BATCH_SIZE)
    return ds_train, ds_val, ds_test


def dice_bce_loss(y_true, y_pred):
    """Combined Dice + weighted BCE. Set DICE_WEIGHT=0 and BCE_FG_WEIGHT=1
    to get plain unweighted cross-entropy."""
    probs   = tf.nn.softmax(y_pred, axis=-1)
    pred_fg = probs[..., 1]
    true_fg = tf.cast(tf.squeeze(y_true, axis=-1), tf.float32)

    smooth = 1.0
    intersection = tf.reduce_sum(pred_fg * true_fg)
    dice = (2.0 * intersection + smooth) / (tf.reduce_sum(pred_fg) + tf.reduce_sum(true_fg) + smooth)
    dice_loss = 1.0 - dice

    sample_weights = tf.where(true_fg > 0.5, BCE_FG_WEIGHT, 1.0)
    ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
    weighted_ce = tf.reduce_mean(ce * sample_weights)

    return DICE_WEIGHT * dice_loss + weighted_ce


def step3_train(X_train, M_train, y_train,
                X_val, M_val, y_val,
                X_test, M_test, y_test):
    print('\n══ Step 3: Building & training U-Net ══')

    # Train on full 256×256 images with 2 channels
    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)
    keras.backend.clear_session()
    model = build_unet(input_shape, OUTPUT_CLASSES)
    model.summary()

    ds_train, ds_val, ds_test = make_tf_datasets(
        X_train, M_train, X_val, M_val, X_test, M_test
    )

    miou = MyMeanIOU(num_classes=OUTPUT_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_bce_loss,
        metrics=['accuracy', miou],
    )

    steps_per_epoch = max(1, X_train.shape[0] // BATCH_SIZE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(RESULTS_DIR, 'hemorrhage_unet.h5')

    history = model.fit(
        ds_train, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        validation_data=ds_val,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, save_best_only=True,
                monitor='val_my_mean_iou', mode='max'),
            keras.callbacks.EarlyStopping(
                monitor='val_my_mean_iou', mode='max',
                min_delta=MIN_DELTA, patience=PATIENCE),
        ],
    )

    model.load_weights(checkpoint_path)
    print('\n── Test-set evaluation (full image only) ──')
    model.evaluate(ds_test)

    return model, history, ds_train, ds_val, ds_test


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — TWO-PASS INFERENCE (full image + ROI patches → stitched mask)     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def predict_two_pass(model, image_2ch):
    """Run two-pass prediction on a single 2-channel image (256×256×2).

    Pass A: predict on the full 256×256 image.
    Pass B: extract ROI patches from high-intensity clusters, predict on
            each 64×64 patch, place predictions back into full mask.
    Final: pixel-wise max of pass A and pass B.

    Returns the stitched binary mask (256×256)."""
    h, w = image_2ch.shape[:2]

    # ── Pass A: full image ────────────────────────────────────────────────────
    full_input = image_2ch[np.newaxis, ...].astype(np.float32)
    full_logits = model.predict(full_input, verbose=0)
    full_pred = tf.argmax(full_logits, axis=-1).numpy().squeeze()  # (256, 256)

    # ── Pass B: ROI patches ───────────────────────────────────────────────────
    gray_ch    = image_2ch[:, :, 0:1]  # grayscale channel
    cluster_ch = image_2ch[:, :, 1:2]  # cluster channel

    roi_boxes = extract_roi_boxes(gray_ch, cluster_ch)

    roi_pred = np.zeros((h, w), dtype=np.float32)

    for (y1, y2, x1, x2) in roi_boxes:
        patch = image_2ch[y1:y2, x1:x2, :]

        # Pad if patch is smaller than PATCH_SIZE
        ph, pw = patch.shape[:2]
        if ph < PATCH_SIZE or pw < PATCH_SIZE:
            padded = np.zeros((PATCH_SIZE, PATCH_SIZE, CHANNELS), dtype=patch.dtype)
            padded[:ph, :pw, :] = patch
            patch = padded

        patch_input = patch[np.newaxis, ...].astype(np.float32)
        patch_logits = model.predict(patch_input, verbose=0)
        patch_pred = tf.argmax(patch_logits, axis=-1).numpy().squeeze()

        # Place back (only the valid region, not padding)
        roi_pred[y1:y2, x1:x2] = np.maximum(
            roi_pred[y1:y2, x1:x2],
            patch_pred[:y2-y1, :x2-x1].astype(np.float32)
        )

    # ── Merge: pixel-wise max ─────────────────────────────────────────────────
    final_mask = np.maximum(full_pred.astype(np.float32), roi_pred)
    return (final_mask > 0.5).astype(np.int32)


def create_mask(pred_logits):
    """Convert raw logits to class mask (for single-pass display)."""
    pred_mask = tf.argmax(pred_logits, axis=-1)
    return pred_mask[0][..., tf.newaxis]


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 — EVALUATION & VISUALISATION                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def show_predictions(model, dataset, num=5, save_dir=None):
    """Save 4-panel comparison using two-pass prediction."""
    if save_dir is None:
        save_dir = os.path.join(RESULTS_DIR, 'prediction_samples')
    os.makedirs(save_dir, exist_ok=True)

    for idx, (image, mask) in enumerate(dataset.take(num)):
        # Two-pass prediction on first sample in batch
        image_2ch = image[0].numpy()
        pred_np = predict_two_pass(model, image_2ch)

        # Grayscale for display (channel 0)
        scan_np = image_2ch[:, :, 0]
        true_np = mask[0].numpy().squeeze()

        scan_rgb = np.stack([scan_np, scan_np, scan_np], axis=-1)
        if scan_rgb.max() > 1.0:
            scan_rgb = scan_rgb / 255.0
        scan_rgb = np.clip(scan_rgb, 0, 1)

        true_overlay = scan_rgb.copy()
        true_binary = (true_np > 0.5)
        true_overlay[true_binary, 0] = np.clip(true_overlay[true_binary, 0] + 0.5, 0, 1)
        true_overlay[true_binary, 1] *= 0.4
        true_overlay[true_binary, 2] *= 0.4

        pred_overlay = scan_rgb.copy()
        pred_binary = (pred_np > 0.5)
        pred_overlay[pred_binary, 0] *= 0.4
        pred_overlay[pred_binary, 1] = np.clip(pred_overlay[pred_binary, 1] + 0.5, 0, 1)
        pred_overlay[pred_binary, 2] *= 0.4

        fig, axes = plt.subplots(1, 4, figsize=(20, 6), constrained_layout=True)
        axes[0].imshow(scan_np, cmap='bone')
        axes[0].set_title('Input scan')
        axes[0].axis('off')

        axes[1].imshow(true_overlay)
        axes[1].set_title('Ground truth (red)')
        axes[1].axis('off')

        axes[2].imshow(pred_overlay)
        axes[2].set_title('Prediction (green)')
        axes[2].axis('off')

        combined = scan_rgb.copy()
        overlap   = true_binary & pred_binary
        true_only = true_binary & ~pred_binary
        pred_only = pred_binary & ~true_binary
        combined[overlap, 0] = 1.0;   combined[overlap, 1] = 1.0;   combined[overlap, 2] = 0.0
        combined[true_only, 0] = 1.0; combined[true_only, 1] = 0.0; combined[true_only, 2] = 0.0
        combined[pred_only, 0] = 0.0; combined[pred_only, 1] = 1.0; combined[pred_only, 2] = 0.0

        axes[3].imshow(combined)
        axes[3].set_title('Overlap (red=missed, green=false pos, yellow=correct)')
        axes[3].axis('off')

        save_path = os.path.join(save_dir, f'prediction_{idx+1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'  Saved {num} prediction images to {save_dir}/')


def step5_evaluate(model, history, ds_test):
    print('\n══ Step 5: Evaluation & visualisation ══')
    h = history.history

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot(h['loss'], label='Train'); ax1.plot(h['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(h['accuracy'], label='Train'); ax2.plot(h['val_accuracy'], label='Val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()

    miou_key     = [k for k in h if 'mean_iou' in k and 'val' not in k][0]
    val_miou_key = 'val_' + miou_key
    ax3.plot(h[miou_key], label='Train'); ax3.plot(h[val_miou_key], label='Val')
    ax3.set_title('Mean IoU'); ax3.set_xlabel('Epoch'); ax3.legend()

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    curves_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f'  Training curves saved to {curves_path}')

    print('\n── Saving sample predictions (two-pass) ──')
    show_predictions(model, ds_test, num=10)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def step5_only():
    """Reload saved model and generate two-pass predictions only."""
    print('\n══ Step 5 (standalone): Loading model & two-pass predictions ══')

    scans, masks, labels, display_labels = step1_build_arrays()
    X_tr, X_test, M_tr, M_test, y_tr, y_test = train_test_split(
        scans, masks, labels, test_size=0.2, random_state=RANDOM_SEED
    )

    def _quantise(m):
        return (m > 127).astype(np.int32)

    ds_test = tf.data.Dataset.from_tensor_slices(
        (X_test.astype(np.float32), _quantise(M_test))
    ).batch(BATCH_SIZE)

    checkpoint_path = os.path.join(RESULTS_DIR, 'hemorrhage_unet.h5')
    if not os.path.exists(checkpoint_path):
        print(f'  ✗ Model not found: {checkpoint_path}')
        return

    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)
    keras.backend.clear_session()
    model = build_unet(input_shape, OUTPUT_CLASSES)
    model.load_weights(checkpoint_path)
    print(f'  Loaded weights from {checkpoint_path}')

    miou = MyMeanIOU(num_classes=OUTPUT_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_bce_loss, metrics=['accuracy', miou],
    )
    print('\n── Test-set evaluation ──')
    model.evaluate(ds_test)

    print('\n── Saving two-pass predictions ──')
    show_predictions(model, ds_test, num=10)
    print('\n✓ Done.')


def main():
    parser = argparse.ArgumentParser(
        description='Hemorrhage segmentation: spectral clustering + adaptive ROI + U-Net')
    parser.add_argument('--step', type=int, default=1,
                        help='Step to start from (1-5). Use 2 to skip clustering, 5 for eval only.')
    args = parser.parse_args()

    if args.step == 5:
        step5_only()
        return

    # Step 1: build arrays (or load from cache)
    if args.step <= 1:
        scans, masks, labels, display_labels = step1_build_arrays()
    else:
        scans, masks, labels, display_labels = load_cached_arrays()
        if scans is None:
            return

    (X_train, M_train, y_train,
     X_val, M_val, y_val,
     X_test, M_test, y_test) = step2_split_and_augment(scans, masks, labels, display_labels)

    model, history, ds_train, ds_val, ds_test = step3_train(
        X_train, M_train, y_train,
        X_val, M_val, y_val,
        X_test, M_test, y_test,
    )

    step5_evaluate(model, history, ds_test)
    print(f'\n✓ Pipeline complete. All results saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
