#!/usr/bin/env python3
"""
Hemorrhage Segmentation with U-Net
===================================
Loads CT scans directly from Renders/, generates segmentation masks from
expert annotations, and trains a modified U-Net for hemorrhage segmentation.

Run from the Project directory:
    cd /Users/Nico/Desktop/MATH7243_ML1/Project
    python hemorrhage_segmentation_unet.py

Steps:
  1 – Load images from Renders/ in memory & build segmentation masks
  2 – Train / val / test split + augmentation
  3 – Build & train U-Net
  4 – Evaluate & plot metrics

Use --step 2 to load cached arrays (skip reloading).
Use --step 4 to reload a saved model and regenerate predictions only.
"""

import os
import ast
import argparse
from random import sample
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

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
PATCH_SIZE     = 256
CHANNELS       = 1            # grayscale (brain window)
OUTPUT_CLASSES = 2
BATCH_SIZE     = 4
BUFFER_SIZE    = 1000
EPOCHS         = 30
PATIENCE       = 20
MIN_DELTA      = 0.005
NORMAL_SAMPLE  = 1000
DICE_WEIGHT    = 6.0
BCE_FG_WEIGHT  = 5.0
RANDOM_SEED    = 42

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
    """Read the four CT window PNGs and return as (H, W, 4).
    Channel order: [subdural, brain, contrast, bone]."""
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
    """Rasterise polygon annotations. scan_img must be 3-channel."""
    filled_mask = np.zeros_like(scan_img, dtype=np.uint8)
    border_img  = np.zeros_like(scan_img, dtype=np.uint8)
    for pts in region_coords:
        if not pts:
            continue
        scaled = (np.array(pts) * np.array(scan_img.shape[:2])).astype(np.int32)
        scaled = scaled.reshape((-1, 1, 2))
        cv2.drawContours(filled_mask, [scaled], 0, (255, 255, 255), thickness=cv2.FILLED)
        cv2.polylines(border_img, [scaled], isClosed=True, color=(255, 255, 255), thickness=3)
    return filled_mask

def get_one_hot_label(img_name, class_cols, label_df):
    stem = img_name.replace('.jpg', '')
    if stem not in label_df.index:
        return None
    vec     = label_df.loc[stem][class_cols].to_numpy().copy().astype(np.float32)
    any_idx = class_cols.index('any')
    vec[any_idx] = (vec[any_idx] + 1) % 2
    return vec


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 — LOAD FROM RENDERS & BUILD ARRAYS                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def step1_build_arrays():
    """Load brain window directly from Renders/, parse annotations into masks,
    resize to IMG_SIZE, return (scans, masks, labels, display_labels)."""
    print('\n══ Step 1: Loading from Renders/ & building arrays ══')
    print(f'  Excluding {len(EXCLUDED_IDS)} flagged/corrupt images')

    scan_lists, scan_folders = build_file_lists()
    label_df = pd.read_csv(LABEL_CSV, index_col='Image')

    display_labels = [lbl for lbl in scan_folders if lbl != CLS_MULTI]
    class_cols     = [lbl.replace(CLS_NORMAL, 'any') for lbl in display_labels]

    for cls, files in zip(scan_folders, scan_lists):
        print(f'    {cls:25s}: {len(files)} images')

    grey_scans = []
    seg_masks  = []
    targets    = []

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

            # Load brain window directly as grayscale input
            brain_path = os.path.join(cls_render_dir, BRAIN_WIN, img_name)
            brain_gray = cv2.imread(brain_path, cv2.IMREAD_GRAYSCALE)
            if brain_gray is None:
                continue

            # Resize and normalise to [0, 1]
            gray_scan = cv2.resize(brain_gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            gray_scan = gray_scan.reshape(IMG_SIZE, IMG_SIZE, 1)

            # ── Normal scans: empty mask ──────────────────────────────────────
            if cls == CLS_NORMAL:
                proc_mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
                label_vec = get_one_hot_label(img_name, class_cols, label_df)
                if label_vec is None:
                    continue

            # ── Hemorrhage scans: parse annotations ───────────────────────────
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

                # Need 3-channel image for drawContours
                scan_3ch = cv2.imread(brain_path)
                if scan_3ch is None:
                    continue
                mask_3ch = draw_annotation_mask(combined_regions, scan_3ch)
                mask_gray = cv2.cvtColor(mask_3ch, cv2.COLOR_BGR2GRAY)
                mask_resized = cv2.resize(mask_gray, (IMG_SIZE, IMG_SIZE),
                                          interpolation=cv2.INTER_NEAREST)
                proc_mask = mask_resized.reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

                label_vec = get_one_hot_label(img_name, class_cols, label_df)
                if label_vec is None:
                    continue

            grey_scans.append(gray_scan)
            seg_masks.append(proc_mask)
            targets.append(label_vec)

    grey_scans = np.array(grey_scans)
    seg_masks  = np.array(seg_masks)
    targets    = np.array(targets)

    print(f'  Scans : {grey_scans.shape}')
    print(f'  Masks : {seg_masks.shape}')
    print(f'  Labels: {targets.shape}')

    # Cache to disk for --step 2
    cache_dir = os.path.join(RESULTS_DIR, 'cached_arrays')
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, 'scans.npy'),  grey_scans)
    np.save(os.path.join(cache_dir, 'masks.npy'),   seg_masks)
    np.save(os.path.join(cache_dir, 'targets.npy'), targets)
    with open(os.path.join(cache_dir, 'display_labels.txt'), 'w') as f:
        f.write('\n'.join(display_labels))
    print(f'  Cached arrays to {cache_dir}/')

    return grey_scans, seg_masks, targets, display_labels


def load_cached_arrays():
    cache_dir = os.path.join(RESULTS_DIR, 'cached_arrays')
    scans_path = os.path.join(cache_dir, 'scans.npy')
    if not os.path.exists(scans_path):
        print(f'  ✗ No cached arrays at {cache_dir}/')
        print('    Run without --step first.')
        return None, None, None, None
    print(f'\n══ Loading cached arrays from {cache_dir}/ ══')
    scans   = np.load(os.path.join(cache_dir, 'scans.npy'))
    masks   = np.load(os.path.join(cache_dir, 'masks.npy'))
    targets = np.load(os.path.join(cache_dir, 'targets.npy'))
    with open(os.path.join(cache_dir, 'display_labels.txt')) as f:
        display_labels = [line.strip() for line in f if line.strip()]
    print(f'  Scans : {scans.shape}')
    print(f'  Masks : {masks.shape}')
    print(f'  Labels: {targets.shape}')
    return scans, masks, targets, display_labels


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 — TRAIN / VAL / TEST SPLIT + AUGMENTATION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def step2_split_and_augment(scans, masks, labels, display_labels):
    print('\n══ Step 2: Splitting & augmenting ══')
    X_tr, X_test, M_tr, M_test, y_tr, y_test = train_test_split(
        scans, masks, labels, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_val, M_train, M_val, y_train, y_val = train_test_split(
        X_tr, M_tr, y_tr, test_size=0.125, random_state=RANDOM_SEED)
    print(f'  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}')

    # Count positives per class
    pos_counts = [np.sum(y_train[:, i] == 1) for i in range(y_train.shape[1])]
    target_count = int(np.median(pos_counts))
    print(f'  Per-class counts: {dict(zip(display_labels, pos_counts))}')
    print(f'  Balancing to target: {target_count} per class (median)')

    aug_gen = ImageDataGenerator(
        rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Collect indices to keep (downsampled) and augmented samples (upsampled)
    keep_indices = set()
    aug_images, aug_masks, aug_targets = [], [], []

    for cls_idx in range(y_train.shape[1]):
        pos_idx = np.where(y_train[:, cls_idx] == 1)[0]
        n_pos = len(pos_idx)

        if n_pos > target_count:
            # Downsample: randomly keep only target_count samples
            chosen = np.random.choice(pos_idx, target_count, replace=False)
            keep_indices.update(chosen)
        elif n_pos < target_count:
            # Keep all originals + augment the shortfall
            keep_indices.update(pos_idx)
            n_needed = target_count - n_pos
            chosen = np.random.choice(pos_idx, n_needed, replace=True)
            for sample_idx in chosen:
                patch = X_train[sample_idx].reshape((1,) + X_train[sample_idx].shape)
                for aug_batch, _ in aug_gen.flow(patch, np.zeros(1), batch_size=1):
                    aug_images.append(aug_batch[0])
                    aug_masks.append(M_train[sample_idx])
                    aug_targets.append(y_train[sample_idx])
                    break
        else:
            keep_indices.update(pos_idx)

    # Also keep samples that aren't positive for any class (shouldn't happen, but safety)
    for i in range(len(y_train)):
        if y_train[i].sum() == 0:
            keep_indices.add(i)

    # Build the balanced training set
    keep_idx = sorted(keep_indices)
    X_bal = X_train[keep_idx]
    M_bal = M_train[keep_idx]
    y_bal = y_train[keep_idx]

    if aug_images:
        X_bal = np.concatenate([X_bal, np.array(aug_images)],  axis=0)
        M_bal = np.concatenate([M_bal, np.array(aug_masks)],   axis=0)
        y_bal = np.concatenate([y_bal, np.array(aug_targets)], axis=0)

    print(f'  After balancing — Train: {X_bal.shape[0]} (was {X_train.shape[0]})')
    return X_bal, M_bal, y_bal, X_val, M_val, y_val, X_test, M_test, y_test


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 — BUILD & TRAIN U-NET                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def build_unet(input_shape, num_classes):
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
    skips.reverse(); filters.reverse()
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
    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)
    keras.backend.clear_session()
    model = build_unet(input_shape, OUTPUT_CLASSES)
    model.summary()
    ds_train, ds_val, ds_test = make_tf_datasets(
        X_train, M_train, X_val, M_val, X_test, M_test)
    miou = MyMeanIOU(num_classes=OUTPUT_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_bce_loss, metrics=['accuracy', miou])
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
        ])
    model.load_weights(checkpoint_path)
    print('\n── Test-set evaluation ──')
    model.evaluate(ds_test)
    return model, history, ds_train, ds_val, ds_test


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 — EVALUATION & VISUALISATION                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    return pred_mask[0][..., tf.newaxis]


def show_predictions(model, dataset, num=5, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(RESULTS_DIR, 'prediction_samples')
    os.makedirs(save_dir, exist_ok=True)
    for idx, (image, mask) in enumerate(dataset.take(num)):
        pred = model.predict(image, verbose=0)
        pred_mask = create_mask(pred)
        scan_np = image[0].numpy().squeeze()
        true_np = mask[0].numpy().squeeze()
        pred_np = pred_mask.numpy().squeeze()

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
        axes[0].set_title('Input scan'); axes[0].axis('off')
        axes[1].imshow(true_overlay)
        axes[1].set_title('Ground truth (red)'); axes[1].axis('off')
        axes[2].imshow(pred_overlay)
        axes[2].set_title('Prediction (green)'); axes[2].axis('off')

        combined = scan_rgb.copy()
        overlap   = true_binary & pred_binary
        true_only = true_binary & ~pred_binary
        pred_only = pred_binary & ~true_binary
        combined[overlap, 0] = 1.0;   combined[overlap, 1] = 1.0;   combined[overlap, 2] = 0.0
        combined[true_only, 0] = 1.0; combined[true_only, 1] = 0.0; combined[true_only, 2] = 0.0
        combined[pred_only, 0] = 0.0; combined[pred_only, 1] = 1.0; combined[pred_only, 2] = 0.0
        axes[3].imshow(combined)
        axes[3].set_title('Overlap (red=missed, green=false pos, yellow=correct)'); axes[3].axis('off')

        save_path = os.path.join(save_dir, f'prediction_{idx+1:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'  Saved {num} prediction images to {save_dir}/')


def step4_evaluate(model, history, ds_test):
    print('\n══ Step 4: Evaluation & visualisation ══')
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
    print('\n── Saving sample predictions (test set) ──')
    show_predictions(model, ds_test, num=10)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

def step4_only():
    print('\n══ Step 4 (standalone): Loading model & generating predictions ══')
    scans, masks, labels, display_labels = load_cached_arrays()
    if scans is None:
        scans, masks, labels, display_labels = step1_build_arrays()
    X_tr, X_test, M_tr, M_test, y_tr, y_test = train_test_split(
        scans, masks, labels, test_size=0.2, random_state=RANDOM_SEED)
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss=dice_bce_loss, metrics=['accuracy', miou])
    print('\n── Test-set evaluation ──')
    model.evaluate(ds_test)
    print('\n── Saving sample predictions ──')
    show_predictions(model, ds_test, num=10)
    print('\n✓ Done.')


def main():
    parser = argparse.ArgumentParser(description='Hemorrhage segmentation U-Net pipeline')
    parser.add_argument('--step', type=int, default=1,
                        help='1=full, 2=skip loading (use cache), 4=eval only')
    args = parser.parse_args()

    if args.step == 4:
        step4_only()
        return

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
        X_test, M_test, y_test)

    step4_evaluate(model, history, ds_test)
    print(f'\n✓ Pipeline complete. All results saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
