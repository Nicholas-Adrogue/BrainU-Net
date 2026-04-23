#!/usr/bin/env python3
"""
ablation.py — Systematic comparison of training configurations.

Loads data once, then trains multiple configurations varying one factor at a time
against the 0.62 baseline. Prints a comparison table at the end.

Usage:
  python ablation.py                    # Quick screening (15+20 epochs per config)
  python ablation.py --full             # Full training (P1+P2 per config settings)
  python ablation.py --configs A B E    # Run specific configs only
  python ablation.py --list             # Show all available configs
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, ast, warnings, argparse, json, time
warnings.filterwarnings("ignore")

try:
    import atexit, multiprocessing.resource_tracker as _rt
    atexit.register(lambda: _rt._resource_tracker._stop())
except (ImportError, AttributeError):
    pass

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    pass

from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, hamming_loss, precision_score, recall_score,
    classification_report, precision_recall_curve
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)

# ── Paths & constants (mirrored from pret.py) ────────────────────────────────
ROOT_DIR    = './dcms/'
seg_dir     = ROOT_DIR + 'segmentation/'
render_dir  = ROOT_DIR + 'renders/'
cnt_dir     = ROOT_DIR + '02_Contour/'

LABEL_FILE  = 'hemorrhage-labels'
EXT_CSV     = '.csv'

CSV_EPIDURAL         = 'Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148'
CSV_INTRAPARENCHYMAL = 'Results_Intraparenchymal Hemorrhage Detection_2020-11-16_21.39.31.268'
CSV_SUBARACHNOID     = 'Results_Subarachnoid Hemorrhage Detection_2020-11-16_21.36.18.668'
CSV_MULTI            = 'Results_Multiple Hemorrhage Detection_2020-11-16_21.36.24.018'
CSV_SUBDURAL_A       = 'Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040'
CSV_SUBDURAL_B       = 'Results_Subdural Hemorrhage Detection_2020-11-16_21.37.19.745'

CLS_EPIDURAL         = 'epidural'
CLS_INTRAPARENCHYMAL = 'intraparenchymal'
CLS_SUBARACHNOID     = 'subarachnoid'
CLS_INTRAVENTRICULAR = 'intraventricular'
CLS_MULTI            = 'multiple'
CLS_SUBDURAL         = 'subdural'
CLS_NORMAL           = 'normal'

BONE_WIN     = '/brain_bone_window/'
BRAIN_WIN    = '/brain_window/'
CONTRAST_WIN = '/max_contrast_window/'
SUB_WIN      = '/subdural_window/'

class_cols     = ['any', 'epidural', 'intraparenchymal', 'intraventricular',
                  'subarachnoid', 'subdural']
display_labels = ['normal', 'epidural', 'intraparenchymal', 'intraventricular',
                  'subarachnoid', 'subdural']
CONTOUR_CLASSES = [CLS_EPIDURAL, CLS_INTRAPARENCHYMAL, CLS_SUBARACHNOID,
                   CLS_MULTI, CLS_SUBDURAL]

N_CLS      = len(display_labels)
IMG_SIZE   = 256
BATCH_SIZE = 32
CLIP_EMB_DIM = 512
SEED       = 42

MODEL_DIR   = './output/models/';   os.makedirs(MODEL_DIR,   exist_ok=True)
FIG_DIR     = './output/figures/';  os.makedirs(FIG_DIR,     exist_ok=True)
ARROW_DIR   = './output/arrow_hints/'
CLIP_FILE   = './output/clip_embeddings.npz'
RADIMAGENET_WEIGHTS = './weights/RadImageNet-DenseNet121_notop.h5'
CACHE_FILE  = './output/ablation_data_cache.npz'

FLAGGED_FILE = './flagged.txt'
CORRUPT_IDS  = {'ID_6431af929'}

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING (runs once, cached to disk)
# ═════════════════════════════════════════════════════════════════════════════

# ── Helpers (from pret.py) ───────────────────────────────────────────────────
_flagged_stems = set(CORRUPT_IDS)
if os.path.exists(FLAGGED_FILE):
    with open(FLAGGED_FILE) as f:
        for line in f:
            s = line.strip()
            if s:
                _flagged_stems.add(s)

def is_flagged(name):
    return Path(name).stem in _flagged_stems

def stack_window_channels(class_dir, fname):
    results = []
    for win in [BONE_WIN, BRAIN_WIN, CONTRAST_WIN, SUB_WIN]:
        img = cv2.imread(class_dir + win + fname, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            results.append(img)
    return results

def _extract_point_list(region):
    return [list(pt.values()) for pt in region if not isinstance(pt, str)]

def _append_region_coords(acc, region):
    pts = _extract_point_list(region)
    if pts:
        acc.append(pts)
    return acc

def _unwrap_nested(raw):
    parsed = ast.literal_eval(raw)
    inner  = np.array(parsed, dtype=object)[0]
    return inner.replace('[]', '')

def parse_annotation_regions(raw_label, is_nested):
    if is_nested:
        raw_label = _unwrap_nested(raw_label)
        if not raw_label:
            return False
    parsed  = ast.literal_eval(raw_label)
    regions = np.array(parsed, dtype=object)
    if regions.size == 0:
        return False
    coords = []
    if regions.ndim > 1:
        for r in regions:
            coords = _append_region_coords(coords, r)
    else:
        if isinstance(regions[0], list):
            for r in regions:
                coords = _append_region_coords(coords, r)
        else:
            coords = _append_region_coords(coords, regions)
    return coords

def resolve_annotation(main, fallback):
    rc = parse_annotation_regions(main, is_nested=False)
    if rc is False:
        rc = parse_annotation_regions(fallback, is_nested=True)
        if rc is False:
            rc = []
    return rc

def draw_annotation_mask(region_coords, scan_img):
    filled = np.zeros_like(scan_img, dtype=np.uint8)
    border = np.zeros_like(scan_img, dtype=np.uint8)
    for pts in region_coords:
        if not pts:
            continue
        scaled = (np.array(pts) * np.array(scan_img.shape[:2])).astype(np.int32).reshape((-1, 1, 2))
        cv2.drawContours(filled, [scaled], 0, (255, 255, 255), thickness=cv2.FILLED)
        cv2.polylines(border, [scaled], isClosed=True, color=(255, 255, 255), thickness=3)
    annotated = cv2.bitwise_and(cv2.bitwise_not(border), scan_img)
    return annotated, filled

def resize_and_normalise(raw_img, annotated_img, target_size, seg_mask):
    sz    = [target_size, target_size]
    r_raw = tf.image.resize(raw_img,  sz).numpy()
    r_msk = tf.image.resize(seg_mask, sz).numpy()
    norm  = r_raw / 255.0
    gray  = cv2.cvtColor(norm,  cv2.COLOR_BGR2GRAY).reshape(target_size, target_size, 1)
    g_msk = cv2.cvtColor(r_msk, cv2.COLOR_BGR2GRAY).reshape(target_size, target_size, 1)
    return norm, gray, g_msk

def get_one_hot_label(img_name, label_df):
    stem = Path(img_name).stem
    vec  = label_df.loc[stem][class_cols].to_numpy().copy()
    any_idx = class_cols.index('any')
    vec[any_idx] = (vec[any_idx] + 1) % 2
    return vec.astype(np.float32)

def get_clip_emb(img_name):
    stem = Path(img_name).stem
    return CLIP_LOOKUP.get(stem, np.zeros(CLIP_EMB_DIM, dtype=np.float32))

def get_arrow_hint(img_name):
    stem = Path(img_name).stem
    p    = ARROW_DIR + stem + '.npy'
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

# ── Both hint types ──────────────────────────────────────────────────────────
def bbox_hint_from_gray(gray_1ch):
    g    = (gray_1ch[:, :, 0] * 255).astype(np.uint8)
    hi   = np.percentile(g, 92)
    mask = (g >= hi).astype(np.uint8)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    attn = np.zeros_like(g, dtype=np.float32)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 50:
            continue
        x, y, w, h = cv2.boundingRect(c)
        attn[y:y+h, x:x+w] = 1.0
    attn = cv2.GaussianBlur(attn, (21, 21), 0)
    mx   = attn.max()
    return ((attn / mx) if mx > 0 else attn).reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

def symmetry_hint(img_4ch):
    brain = img_4ch[:, :, 1]
    bone  = img_4ch[:, :, 0]
    brain_diff = np.abs(brain - np.fliplr(brain))
    bone_diff  = np.abs(bone - np.fliplr(bone))
    combined = 0.7 * brain_diff + 0.3 * bone_diff
    border = 12
    combined[:border, :] = 0; combined[-border:, :] = 0
    combined[:, :border] = 0; combined[:, -border:] = 0
    combined = cv2.GaussianBlur(combined, (15, 15), 0)
    mx = combined.max()
    if mx > 0:
        combined = combined / mx
    return combined.reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)


def load_all_data():
    """Load all data, compute both hint types. Returns dict of numpy arrays."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}...")
        d = np.load(CACHE_FILE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print("Loading data from scratch (will cache for next run)...")

    global CLIP_LOOKUP
    _clip_data  = np.load(CLIP_FILE, allow_pickle=True)
    CLIP_LOOKUP = dict(zip(_clip_data['stems'].tolist(),
                           _clip_data['embeddings'].astype(np.float32)))

    label_df = pd.read_csv(seg_dir + LABEL_FILE + EXT_CSV, index_col='Image')

    all_3ch, all_seg, all_bbox, all_symm = [], [], [], []
    all_arrow, all_clip, all_y = [], [], []

    for cls in CONTOUR_CLASSES:
        cls_img_dir = cnt_dir + cls + '/'
        if not os.path.isdir(cls_img_dir):
            continue
        if cls == CLS_SUBDURAL:
            ann_df = pd.concat([
                pd.read_csv(seg_dir + CSV_SUBDURAL_A + EXT_CSV),
                pd.read_csv(seg_dir + CSV_SUBDURAL_B + EXT_CSV)
            ], ignore_index=True)[['Origin', 'Majority Label', 'Correct Label', 'All Labels']]
        elif cls == CLS_EPIDURAL:
            ann_df = pd.read_csv(seg_dir + CSV_EPIDURAL + EXT_CSV)[['Origin', 'Majority Label', 'Correct Label', 'All Labels']]
        elif cls == CLS_INTRAPARENCHYMAL:
            ann_df = pd.read_csv(seg_dir + CSV_INTRAPARENCHYMAL + EXT_CSV)[['Origin', 'Majority Label', 'Correct Label', 'All Labels']]
        elif cls == CLS_SUBARACHNOID:
            ann_df = pd.read_csv(seg_dir + CSV_SUBARACHNOID + EXT_CSV)[['Origin', 'Majority Label', 'Correct Label', 'All Labels']]
        elif cls == CLS_MULTI:
            ann_df = pd.read_csv(seg_dir + CSV_MULTI + EXT_CSV)[['Origin', 'Majority Label', 'Correct Label', 'All Labels']]
        else:
            continue

        render_cls_dir = render_dir + ('subdural/subdural' if cls == CLS_SUBDURAL else cls)
        img_names = [f for f in os.listdir(cls_img_dir) if f != '.DS_Store']

        for img_name in tqdm(img_names, desc=cls, unit='image'):
            if is_flagged(img_name):
                continue
            scan = cv2.imread(cls_img_dir + img_name)
            if scan is None:
                continue
            row_matches = ann_df[ann_df.Origin == img_name]
            combined_regions = []
            for cl_val, ml_val, al_val in zip(
                    row_matches['Correct Label'],
                    row_matches['Majority Label'],
                    row_matches['All Labels']):
                if isinstance(cl_val, str):
                    rc = parse_annotation_regions(cl_val, is_nested=False)
                    if not rc:
                        rc = resolve_annotation(ml_val, al_val)
                elif isinstance(ml_val, str):
                    rc = resolve_annotation(ml_val, al_val)
                else:
                    rc = []
                if rc:
                    combined_regions += rc
            if not combined_regions or cls == CLS_MULTI:
                continue

            overlaid, seg_mask = draw_annotation_mask(combined_regions, scan)
            _, gray_scan, gray_mask = resize_and_normalise(scan, overlaid, IMG_SIZE, seg_mask)
            gray_mask = (gray_mask / 255.0).astype(np.float32)
            channels = stack_window_channels(render_cls_dir, img_name)
            if len(channels) < 4:
                continue
            img_4ch = np.stack([cv2.resize(c, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                                for c in channels[:4]], axis=-1)
            try:
                lbl = get_one_hot_label(img_name, label_df)
            except KeyError:
                continue

            all_3ch.append(img_4ch)
            all_seg.append(gray_mask)
            all_bbox.append(bbox_hint_from_gray(gray_scan))
            all_symm.append(symmetry_hint(img_4ch))
            all_arrow.append(get_arrow_hint(img_name))
            all_clip.append(get_clip_emb(img_name))
            all_y.append(lbl)

    # Load renders
    def _load_renders(cls_name, cap, all_arrays):
        rdir = render_dir + cls_name
        bdir = rdir + BRAIN_WIN
        if not os.path.isdir(bdir):
            return
        files = [f for f in os.listdir(bdir) if f != '.DS_Store' and not is_flagged(f)][:cap]
        for img_name in tqdm(files, desc=f"render/{cls_name}", unit='image'):
            channels = stack_window_channels(rdir, img_name)
            if len(channels) < 4:
                continue
            img_4ch = np.stack([cv2.resize(c, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                                for c in channels[:4]], axis=-1)
            gray_scan = np.mean(img_4ch, axis=-1, keepdims=True).astype(np.float32)
            try:
                lbl = get_one_hot_label(img_name, label_df)
            except KeyError:
                continue
            all_3ch.append(img_4ch)
            all_seg.append(np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32))
            all_bbox.append(bbox_hint_from_gray(gray_scan))
            all_symm.append(symmetry_hint(img_4ch))
            all_arrow.append(get_arrow_hint(img_name))
            all_clip.append(get_clip_emb(img_name))
            all_y.append(lbl)

    _load_renders(CLS_NORMAL, 500, None)
    _load_renders(CLS_INTRAVENTRICULAR, 500, None)
    _load_renders(CLS_EPIDURAL, 300, None)

    data = {
        'all_3ch':   np.array(all_3ch,   dtype=np.float32),
        'all_seg':   np.array(all_seg,   dtype=np.float32),
        'all_bbox':  np.array(all_bbox,  dtype=np.float32),
        'all_symm':  np.array(all_symm,  dtype=np.float32),
        'all_arrow': np.array(all_arrow, dtype=np.float32),
        'all_clip':  np.array(all_clip,  dtype=np.float32),
        'all_y':     np.array(all_y,     dtype=np.float32),
    }

    print(f"Saving cache to {CACHE_FILE} ({sum(v.nbytes for v in data.values()) / 1e9:.1f} GB)...")
    np.savez_compressed(CACHE_FILE, **data)
    return data


# ═════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ═════════════════════════════════════════════════════════════════════════════

def _load_densenet121_4ch(input_shape, weights_path=None):
    import glob
    h, w, _ = input_shape
    model_3ch = keras.applications.DenseNet121(
        input_shape=(h, w, 3), include_top=False, weights=None)
    if weights_path and os.path.exists(weights_path):
        model_3ch.load_weights(weights_path, by_name=True, skip_mismatch=True)
    else:
        model_3ch_img = keras.applications.DenseNet121(
            input_shape=(h, w, 3), include_top=False, weights="imagenet")
        model_3ch.set_weights(model_3ch_img.get_weights())
        del model_3ch_img
    first_conv = model_3ch.get_layer("conv1_conv")
    w3 = first_conv.get_weights()[0]
    w4_new = np.mean(w3, axis=2, keepdims=True)
    w4 = np.concatenate([w3, w4_new], axis=2)
    model_4ch = keras.applications.DenseNet121(
        input_shape=(h, w, 4), include_top=False, weights=None)
    for layer_3, layer_4 in zip(model_3ch.layers, model_4ch.layers):
        if layer_4.name == "conv1_conv":
            old_weights = layer_3.get_weights()
            layer_4.set_weights([w4] + old_weights[1:])
        else:
            try:
                layer_4.set_weights(layer_3.get_weights())
            except (ValueError, Exception):
                pass
    del model_3ch
    return model_4ch


def build_model(loss_fn, lr=1e-3):
    import glob
    img_in  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 4), name="ct_4ch")
    hint_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="hints_3ch")
    clip_in = keras.Input(shape=(CLIP_EMB_DIM,),         name="clip_emb")

    rad_path = RADIMAGENET_WEIGHTS
    if not os.path.exists(rad_path):
        candidates = glob.glob('./weights/*[Dd]ense[Nn]et*121*.h5')
        if candidates:
            rad_path = candidates[0]

    base = _load_densenet121_4ch(
        (IMG_SIZE, IMG_SIZE, 4),
        weights_path=rad_path if os.path.exists(rad_path) else None)
    base.trainable = False

    x = base(img_in, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    h = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(hint_in)
    h = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.GlobalAveragePooling2D()(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.3)(h)

    c = layers.Dense(256, activation="relu")(clip_in)
    c = layers.BatchNormalization()(c)
    c = layers.Dense(128, activation="relu")(c)
    c = layers.Dropout(0.3)(c)

    fused = layers.Concatenate()([x, h, c])
    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.3)(fused)
    out   = layers.Dense(N_CLS, activation="sigmoid", name="output")(fused)

    model = keras.Model(inputs=[img_in, hint_in, clip_in], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=loss_fn,
        metrics=[
            keras.metrics.BinaryAccuracy(name="bin_acc"),
            keras.metrics.AUC(name="auc", multi_label=True),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )
    return model, base


# ═════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

# These get created per-config since CLS_W depends on the config
def make_weighted_bce(cls_w, label_smooth=0.0):
    def weighted_bce(y_true, y_pred):
        if label_smooth > 0:
            y_true = y_true * (1.0 - label_smooth) + 0.5 * label_smooth
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weights = y_true * cls_w + (1.0 - y_true) * 1.0
        return tf.reduce_mean(bce * weights)
    weighted_bce.__name__ = "weighted_bce"
    return weighted_bce

def make_soft_f1_loss():
    def soft_macro_f1_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=0)
        soft_f1 = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
        return 1.0 - tf.reduce_mean(soft_f1)
    soft_macro_f1_loss.__name__ = "soft_macro_f1_loss"
    return soft_macro_f1_loss

def make_combined_loss(cls_w, f1_coeff=0.2, label_smooth=0.0):
    bce_fn = make_weighted_bce(cls_w, label_smooth)
    f1_fn  = make_soft_f1_loss()
    def combined(y_true, y_pred):
        return bce_fn(y_true, y_pred) + f1_coeff * f1_fn(y_true, y_pred)
    combined.__name__ = "combined_clf_loss"
    return combined


# ═════════════════════════════════════════════════════════════════════════════
# AUGMENTATION STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════

def apply_oversampling(X_3ch, X_seg, X_mid, X_arrow, X_clip, y, target='max', aug_style='original'):
    """Offline oversampling with augmentation. target='max'|'median'|'1.5x'|None.
    aug_style='original' uses the 0.62 params, 'mild' uses toned-down params."""
    if target is None:
        return X_3ch, X_seg, X_mid, X_arrow, X_clip, y

    if aug_style == 'original':
        # The params that produced 0.62 — aggressive but effective with small datasets
        aug_gen = ImageDataGenerator(
            rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    else:
        aug_gen = ImageDataGenerator(
            rotation_range=12, width_shift_range=0.15, height_shift_range=0.15,
            shear_range=0.05, zoom_range=0.15, horizontal_flip=True, fill_mode='nearest')

    train_df = pd.DataFrame(y, columns=display_labels)
    counts = [int(train_df[c].sum()) for c in display_labels]

    if target == 'max':
        target_n = max(counts)
    elif target == 'median':
        target_n = int(np.median(counts))
    elif target == '1.5x':
        target_n = int(np.median(counts) * 1.5)
    else:
        target_n = int(target)

    aug_arrs = [[] for _ in range(6)]
    for cls_idx in range(len(display_labels)):
        pos_idx = np.where(y[:, cls_idx] == 1)[0]
        n_needed = target_n - len(pos_idx)
        if n_needed <= 0 or len(pos_idx) == 0:
            continue
        chosen = np.random.choice(pos_idx, n_needed, replace=True)
        for si in chosen:
            sv = np.random.randint(0, 9999)
            def _aug(ch, sv=sv):
                for b, _ in aug_gen.flow(ch[None], np.zeros(1), batch_size=1, seed=sv):
                    return b[0]
            aug_arrs[0].append(_aug(X_3ch[si]))
            aug_arrs[1].append(_aug(X_seg[si]))
            aug_arrs[2].append(_aug(X_mid[si]))
            aug_arrs[3].append(_aug(X_arrow[si]))
            aug_arrs[4].append(X_clip[si])
            aug_arrs[5].append(y[si])

    if aug_arrs[0]:
        X_3ch   = np.concatenate([X_3ch,   np.array(aug_arrs[0])], axis=0)
        X_seg   = np.concatenate([X_seg,   np.array(aug_arrs[1])], axis=0)
        X_mid   = np.concatenate([X_mid,   np.array(aug_arrs[2])], axis=0)
        X_arrow = np.concatenate([X_arrow, np.array(aug_arrs[3])], axis=0)
        X_clip  = np.concatenate([X_clip,  np.array(aug_arrs[4])], axis=0)
        y       = np.concatenate([y,       np.array(aug_arrs[5])], axis=0)

    return X_3ch, X_seg, X_mid, X_arrow, X_clip, y


@tf.function
def clf_online_aug(inputs, labels):
    ct   = inputs["ct_4ch"]
    hint = inputs["hints_3ch"]
    clip = inputs["clip_emb"]
    if tf.random.uniform(()) > 0.5:
        ct   = tf.image.flip_left_right(ct)
        hint = tf.image.flip_left_right(hint)
    if tf.random.uniform(()) > 0.5:
        ct   = tf.image.flip_up_down(ct)
        hint = tf.image.flip_up_down(hint)
    ct = tf.image.random_brightness(ct, 0.1)
    ct = tf.clip_by_value(ct, 0.0, 1.0)
    return {"ct_4ch": ct, "hints_3ch": hint, "clip_emb": clip}, labels


def make_dataset(c3, hints, clip, y, shuffle=True, online_aug=False):
    ds = tf.data.Dataset.from_tensor_slices((
        {"ct_4ch": c3, "hints_3ch": hints, "clip_emb": clip}, y))
    if online_aug:
        ds = ds.cache().shuffle(len(y), reshuffle_each_iteration=True)
        ds = ds.map(clf_online_aug, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(y), seed=SEED)
    return ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

CONFIGS = {
    # ── A: TRUE Baseline (the 0.62 config — original aug + epi boost) ───
    'A_baseline': dict(
        desc='TRUE 0.62: oversample-to-max, ORIGINAL aug (rot40), bbox, bce, val_auc, epi_boost',
        hint='bbox', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
    ),
    # ── Single-factor changes vs baseline ─────────────────────────────────
    'B_symm': dict(
        desc='Symmetry hint instead of bbox (only change)',
        hint='symm', aug='max', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
    ),
    'C_no_aug': dict(
        desc='No offline augmentation, online flips only',
        hint='bbox', aug=None, loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=True, epi_boost=False,
    ),
    'D_mild_aug': dict(
        desc='Mild augmentation (1.5x median, mild transforms)',
        hint='bbox', aug='1.5x', aug_style='mild', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
    ),
    'E_f1_02': dict(
        desc='Add soft F1 loss at 0.2 coefficient',
        hint='bbox', aug='max', loss='bce+f1_0.2', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
    ),
    'F_capped_w': dict(
        desc='CLS_W capped at 5.0',
        hint='bbox', aug='max', loss='bce', cls_w_cap=5.0,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
    ),
    'G_short_p1': dict(
        desc='Phase 1=25, Phase 2=100 (get to fine-tuning faster)',
        hint='bbox', aug='max', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=25, p2_epochs=100, online_aug=False,
    ),
    'H_val_loss': dict(
        desc='Monitor val_loss instead of val_auc',
        hint='bbox', aug='max', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_loss', monitor_mode='min',
        p1_epochs=80, p2_epochs=80, online_aug=False,
    ),
    'I_label_smooth': dict(
        desc='Label smoothing eps=0.05',
        hint='bbox', aug='max', loss='bce', cls_w_cap=None,
        label_smooth=0.05, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
    ),
    # ── Learning rate variants ────────────────────────────────────────────
    'J_hi_lr': dict(
        desc='Higher LR: P1=3e-3, P2=5e-5',
        hint='bbox', aug='max', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
        p1_lr=3e-3, p2_lr=5e-5,
    ),
    'K_hi_lr_p2': dict(
        desc='Higher P2 LR only: P1=1e-3, P2=3e-4',
        hint='bbox', aug='max', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
        p1_lr=1e-3, p2_lr=3e-4,
    ),
    # ── Higher soft F1 coefficient ────────────────────────────────────────
    'L_f1_05': dict(
        desc='Soft F1 loss at 0.5 coefficient',
        hint='bbox', aug='max', loss='bce+f1_0.5', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
    ),
    'M_f1_10': dict(
        desc='Soft F1 loss at 1.0 coefficient (equal to BCE)',
        hint='bbox', aug='max', loss='bce+f1_1.0', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False,
    ),
    # ── Hint channel diagnostics (based on TRUE baseline) ───────────────
    'N1_no_hints': dict(
        desc='TRUE baseline + all hints zeroed — hint branch value?',
        hint='zeros', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
    ),
    'N2_seg_only': dict(
        desc='TRUE baseline + only U-Net seg mask — isolate seg contribution',
        hint='seg_only', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
    ),
    'N3_symm_only': dict(
        desc='TRUE baseline + only symmetry hint — is symmetry alone useful?',
        hint='symm_only', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
    ),
    'N4_no_unet': dict(
        desc='TRUE baseline + bbox only, no U-Net channels',
        hint='no_unet', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
    ),
    'N5_hem_unet': dict(
        desc='TRUE baseline + hemorrhage_unet.h5 for seg+arrow (symm mid)',
        hint='symm', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
        unet_path= MODEL_DIR + '/hemorrhage_unet.h5',
    ),
    'N6_model_h5_unet': dict(
        desc='TRUE baseline + model.h5 for seg+arrow (symm mid)',
        hint='symm', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
        unet_path='./model.h5',
    ),
    # ── Pretrained model evaluation (no training) ─────────────────────────
    'P_model_h5': dict(
        desc='Evaluate pretrained model.h5 as classifier (no training)',
        hint='bbox', aug='max', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=0, p2_epochs=0, online_aug=False,
        model_path='./model.h5',
    ),
    # ── Best-guess combos (all use ORIGINAL aug style now) ──────────────
    'Z1_combo_conservative': dict(
        desc='Symm + original aug + epi boost + bce only + short_p1 + val_loss',
        hint='symm', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_loss', monitor_mode='min',
        p1_epochs=25, p2_epochs=100, online_aug=False, epi_boost=True,
    ),
    'Z2_combo_moderate': dict(
        desc='Symm + original aug + epi boost + f1_0.2 + capped_w5 + short_p1',
        hint='symm', aug='max', aug_style='original', loss='bce+f1_0.2', cls_w_cap=5.0,
        label_smooth=0.0, monitor='val_loss', monitor_mode='min',
        p1_epochs=25, p2_epochs=100, online_aug=False, epi_boost=True,
    ),
    'Z3_combo_quality': dict(
        desc='Z2 + quality filter p10',
        hint='symm', aug='max', aug_style='original', loss='bce+f1_0.2', cls_w_cap=5.0,
        label_smooth=0.0, monitor='val_loss', monitor_mode='min',
        p1_epochs=25, p2_epochs=100, online_aug=False, epi_boost=True,
        quality_filter=10,
    ),
    'Z4_baseline_plus_symm': dict(
        desc='TRUE baseline + only swap bbox→symm (isolate symm value)',
        hint='symm', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
    ),
    'Z5_baseline_nohints': dict(
        desc='TRUE baseline + hints zeroed (is hint branch helping at all?)',
        hint='zeros', aug='max', aug_style='original', loss='bce', cls_w_cap=None,
        label_smooth=0.0, monitor='val_auc', monitor_mode='max',
        p1_epochs=80, p2_epochs=80, online_aug=False, epi_boost=True,
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def collect_preds(ds, mdl):
    probs, labels = [], []
    for batch_x, lbl_b in ds:
        probs.append(mdl.predict(batch_x, verbose=0))
        labels.append(lbl_b.numpy())
    return np.vstack(probs), np.vstack(labels)


def run_config(name, cfg, data, idx_train, idx_val, idx_test, quick=False, verbose=2):
    """Train one configuration, return results dict."""
    print(f"\n{'='*70}")
    print(f"CONFIG: {name}")
    print(f"  {cfg['desc']}")
    print(f"{'='*70}")
    t0 = time.time()

    # ── Optional quality filtering ───────────────────────────────────────
    q_pct = cfg.get('quality_filter', 0)
    if q_pct > 0:
        brain_ch = data['all_3ch'][:, :, :, 1]  # brain window
        content = np.array([np.mean(brain_ch[i] > 0.05) for i in range(len(brain_ch))])
        variance = np.array([np.var(brain_ch[i]) for i in range(len(brain_ch))])
        q_scores = content * variance
        q_thresh = np.percentile(q_scores, q_pct)
        q_mask   = q_scores >= q_thresh
        # Filter train/val/test indices
        idx_train = np.array([i for i in idx_train if q_mask[i]])
        idx_val   = np.array([i for i in idx_val   if q_mask[i]])
        idx_test  = np.array([i for i in idx_test  if q_mask[i]])
        n_removed = int((~q_mask).sum())
        print(f"  Quality filter (p{q_pct}): removed {n_removed} images, "
              f"train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    # ── Select hint channels ────────────────────────────────────────────
    # hint options: 'bbox', 'symm', 'zeros', 'seg_only', 'symm_only', 'no_unet'
    mid_key = 'all_bbox' if cfg['hint'] in ('bbox', 'seg_only', 'zeros', 'no_unet') else 'all_symm'
    n_samples = len(data['all_y'])
    zeros_ch = np.zeros((n_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    if cfg['hint'] == 'zeros':
        # All three hint channels zeroed — measures hint branch contribution
        seg_all   = zeros_ch
        mid_all   = zeros_ch
        arrow_all = zeros_ch
    elif cfg['hint'] == 'seg_only':
        # Only U-Net seg mask, zero the rest
        seg_all   = data['all_seg']
        mid_all   = zeros_ch
        arrow_all = zeros_ch
    elif cfg['hint'] == 'symm_only':
        # Only symmetry, zero seg and arrow
        seg_all   = zeros_ch
        mid_all   = data['all_symm']
        arrow_all = zeros_ch
    elif cfg['hint'] == 'no_unet':
        # Bbox + zero seg + zero arrow — isolate non-UNet hints
        seg_all   = zeros_ch
        mid_all   = data['all_bbox']
        arrow_all = zeros_ch
    else:
        # Normal: seg + mid(bbox/symm) + arrow
        seg_all   = data['all_seg']
        mid_all   = data[mid_key]
        arrow_all = data['all_arrow']

    # ── Optional: regenerate seg+arrow from a different U-Net ────────────
    unet_path = cfg.get('unet_path', None)
    if unet_path:
        if not os.path.exists(unet_path):
            print(f"  ⚠ U-Net not found: {unet_path} — using cached hints")
        else:
            print(f"  Loading U-Net for hint generation: {unet_path}")
            try:
                unet = keras.models.load_model(unet_path, compile=False)
                unet_input_size = unet.input_shape[1]  # e.g. 128 or 256
                print(f"  Running U-Net inference ({unet_input_size}px) on {n_samples} images...")

                fresh_seg = np.zeros((n_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
                # Batch inference for speed
                batch_sz = 16
                for start in range(0, n_samples, batch_sz):
                    end = min(start + batch_sz, n_samples)
                    batch = data['all_3ch'][start:end]
                    # Resize to U-Net input size if needed
                    if unet_input_size != IMG_SIZE:
                        batch = tf.image.resize(batch, [unet_input_size, unet_input_size]).numpy()
                    # Handle channel mismatch (U-Net might expect 4ch or different)
                    expected_ch = unet.input_shape[-1]
                    if expected_ch != batch.shape[-1]:
                        if expected_ch == 3 and batch.shape[-1] == 4:
                            batch = batch[:, :, :, :3]
                        elif expected_ch == 1 and batch.shape[-1] == 4:
                            batch = np.mean(batch, axis=-1, keepdims=True)
                    preds = unet.predict(batch, verbose=0)
                    if isinstance(preds, (list, tuple)):
                        preds = preds[0]
                    # Resize back to IMG_SIZE
                    if preds.shape[1] != IMG_SIZE:
                        preds = tf.image.resize(preds, [IMG_SIZE, IMG_SIZE]).numpy()
                    # Ensure single channel
                    if preds.shape[-1] != 1:
                        preds = preds[:, :, :, :1]
                    fresh_seg[start:end] = preds.astype(np.float32)

                mean_activation = fresh_seg.mean()
                nonzero_frac = (fresh_seg > 0.1).mean()
                print(f"  U-Net predictions: mean={mean_activation:.4f}, "
                      f"{nonzero_frac*100:.1f}% pixels > 0.1")

                # Replace seg channel with fresh predictions
                seg_all = fresh_seg
                # Also use soft predictions as arrow channel (replaces stale .npy arrows)
                arrow_all = fresh_seg.copy()
                print(f"  Replaced seg + arrow channels with fresh U-Net predictions")

                del unet
                keras.backend.clear_session()
            except Exception as e:
                print(f"  ⚠ U-Net inference failed: {e} — using cached hints")
                import traceback; traceback.print_exc()

    # ── Split ────────────────────────────────────────────────────────────
    X_tr  = data['all_3ch'][idx_train].copy()
    S_tr  = seg_all[idx_train].copy()
    M_tr  = mid_all[idx_train].copy()
    A_tr  = arrow_all[idx_train].copy()
    C_tr  = data['all_clip'][idx_train].copy()
    y_tr  = data['all_y'][idx_train].copy()

    # ── Augmentation ─────────────────────────────────────────────────────
    if cfg['aug'] is not None:
        aug_style = cfg.get('aug_style', 'original')
        print(f"  Oversampling: {cfg['aug']} (style={aug_style})")
        X_tr, S_tr, M_tr, A_tr, C_tr, y_tr = apply_oversampling(
            X_tr, S_tr, M_tr, A_tr, C_tr, y_tr,
            target=cfg['aug'], aug_style=aug_style)

        # Extra epidural boost (was in original 0.62 config)
        if cfg.get('epi_boost', True):
            from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
            epi_gen = IDG(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                          shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
            epi_col = display_labels.index('epidural')
            epi_idx = np.where(y_tr[:, epi_col] == 1)[0]
            chosen  = np.random.choice(epi_idx, len(epi_idx), replace=True)
            ex = [[] for _ in range(6)]
            for si in chosen:
                sv = np.random.randint(0, 9999)
                def _a(ch, sv=sv):
                    for b, _ in epi_gen.flow(ch[None], np.zeros(1), batch_size=1, seed=sv):
                        return b[0]
                ex[0].append(_a(X_tr[si])); ex[1].append(_a(S_tr[si]))
                ex[2].append(_a(M_tr[si])); ex[3].append(_a(A_tr[si]))
                ex[4].append(C_tr[si]);     ex[5].append(y_tr[si])
            X_tr = np.concatenate([X_tr, np.array(ex[0])], axis=0)
            S_tr = np.concatenate([S_tr, np.array(ex[1])], axis=0)
            M_tr = np.concatenate([M_tr, np.array(ex[2])], axis=0)
            A_tr = np.concatenate([A_tr, np.array(ex[3])], axis=0)
            C_tr = np.concatenate([C_tr, np.array(ex[4])], axis=0)
            y_tr = np.concatenate([y_tr, np.array(ex[5])], axis=0)
            print(f"  Epidural boost: +{len(chosen)} (total {len(y_tr)})")

    print(f"  Train samples: {len(y_tr)}")

    # ── Build hints ──────────────────────────────────────────────────────
    tr_h = np.concatenate([S_tr, M_tr, A_tr], axis=-1).astype(np.float32)
    v_h  = np.concatenate([seg_all[idx_val],  mid_all[idx_val],
                           arrow_all[idx_val]], axis=-1).astype(np.float32)
    t_h  = np.concatenate([seg_all[idx_test], mid_all[idx_test],
                           arrow_all[idx_test]], axis=-1).astype(np.float32)

    y_val  = data['all_y'][idx_val]
    y_test = data['all_y'][idx_test]

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = make_dataset(X_tr, tr_h, C_tr, y_tr, True, cfg['online_aug'])
    val_ds   = make_dataset(data['all_3ch'][idx_val], v_h,
                            data['all_clip'][idx_val], y_val, False)
    test_ds  = make_dataset(data['all_3ch'][idx_test], t_h,
                            data['all_clip'][idx_test], y_test, False)

    # ── Loss function ────────────────────────────────────────────────────
    _pos = y_tr.sum(axis=0)
    _neg = len(y_tr) - _pos
    raw_w = (_neg / (_pos + 1e-6)).astype(np.float32)
    if cfg['cls_w_cap']:
        raw_w = np.clip(raw_w, 1.0, cfg['cls_w_cap'])
    cls_w = tf.constant(raw_w)

    if cfg['loss'] == 'bce':
        loss_fn = make_weighted_bce(cls_w, cfg['label_smooth'])
    elif cfg['loss'].startswith('bce+f1_'):
        coeff = float(cfg['loss'].split('_')[1])
        loss_fn = make_combined_loss(cls_w, coeff, cfg['label_smooth'])
    else:
        loss_fn = make_weighted_bce(cls_w, cfg['label_smooth'])

    # ── Epoch counts ─────────────────────────────────────────────────────
    p1_ep = min(cfg['p1_epochs'], 15) if quick else cfg['p1_epochs']
    p2_ep = min(cfg['p2_epochs'], 20) if quick else cfg['p2_epochs']
    patience = 8 if quick else 12

    p1_lr = cfg.get('p1_lr', 1e-3)
    p2_lr = cfg.get('p2_lr', 1e-5)

    # ── Build or load model ──────────────────────────────────────────────
    model_path = cfg.get('model_path', None)
    if model_path and not os.path.exists(model_path):
        print(f"  ⚠ Model not found: {model_path} — SKIPPING config")
        return {
            'name': name, 'desc': cfg['desc'],
            'macro_f1': 0.0, 'hamming_loss': 1.0,
            'per_class_f1': {c: 0.0 for c in display_labels},
            'thresholds': {c: 0.5 for c in display_labels},
            'elapsed_min': 0.0, 'error': f'Model not found: {model_path}',
        }

    if model_path and os.path.exists(model_path):
        print(f"  Loading pretrained model: {model_path}")
        custom_objs = {
            "weighted_bce": loss_fn,
            "soft_macro_f1_loss": make_soft_f1_loss(),
            "combined_clf_loss": loss_fn,
        }
        try:
            model = keras.models.load_model(model_path, custom_objects=custom_objs)
        except Exception:
            # Try compile=False if custom objects don't match
            model = keras.models.load_model(model_path, compile=False)
            model.compile(loss=loss_fn, metrics=[
                keras.metrics.BinaryAccuracy(name="bin_acc"),
                keras.metrics.AUC(name="auc", multi_label=True),
            ])
        backbone = None
        print(f"  Skipping training — evaluate only")
    else:
        model, backbone = build_model(loss_fn, lr=p1_lr)

        mon     = cfg['monitor']
        mon_m   = cfg['monitor_mode']

        # ── Phase 1 ─────────────────────────────────────────────────────
        print(f"  Phase 1: {p1_ep} epochs, lr={p1_lr} (frozen backbone)")
        model.fit(train_ds, validation_data=val_ds, epochs=p1_ep, verbose=verbose,
                  callbacks=[
                      keras.callbacks.EarlyStopping(
                          monitor=mon, mode=mon_m, patience=patience,
                          restore_best_weights=True),
                  ])

        # ── Phase 2 ─────────────────────────────────────────────────────
        if backbone is not None:
            for layer in backbone.layers[-40:]:
                layer.trainable = True
            model.compile(
                optimizer=keras.optimizers.Adam(p2_lr),
                loss=loss_fn,
                metrics=[
                    keras.metrics.BinaryAccuracy(name="bin_acc"),
                    keras.metrics.AUC(name="auc", multi_label=True),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall"),
                ])
            print(f"  Phase 2: {p2_ep} epochs, lr={p2_lr} (fine-tuning top-40)")
            model.fit(train_ds, validation_data=val_ds, epochs=p2_ep, verbose=verbose,
                      callbacks=[
                          keras.callbacks.EarlyStopping(
                              monitor=mon, mode=mon_m, patience=patience,
                              restore_best_weights=True),
                          keras.callbacks.ReduceLROnPlateau(
                              monitor=mon, mode=mon_m,
                              factor=0.5, patience=4, min_lr=1e-7),
                      ])

    # ── Threshold optimization + evaluation ──────────────────────────────
    val_probs, val_true = collect_preds(val_ds, model)
    best_thresh = np.zeros(N_CLS)
    for i in range(N_CLS):
        prec, rec, thresh = precision_recall_curve(val_true[:, i], val_probs[:, i])
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-8), 0)
        idx = np.argmax(f1)
        best_thresh[i] = float(thresh[idx]) if idx < len(thresh) else 0.5

    test_probs, test_true = collect_preds(test_ds, model)
    test_preds = (test_probs >= best_thresh).astype(int)

    hl  = hamming_loss(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average="macro", zero_division=0)

    per_class_f1 = {}
    for i, cls in enumerate(display_labels):
        per_class_f1[cls] = f1_score(test_true[:, i], test_preds[:, i], zero_division=0)

    elapsed = time.time() - t0
    print(f"  → Macro F1: {mf1:.4f}  Hamming: {hl:.4f}  ({elapsed/60:.1f} min)")

    # Clean up
    del model, backbone, train_ds, val_ds, test_ds
    del X_tr, S_tr, M_tr, A_tr, C_tr, y_tr, tr_h
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    return {
        'name': name, 'desc': cfg['desc'],
        'macro_f1': mf1, 'hamming_loss': hl,
        'per_class_f1': per_class_f1,
        'thresholds': {c: float(t) for c, t in zip(display_labels, best_thresh)},
        'elapsed_min': elapsed / 60,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Ablation study for ICH classifier')
    parser.add_argument('--full', action='store_true',
                        help='Full training (default: quick screening)')
    parser.add_argument('--verbose', type=int, default=2, choices=[0, 1, 2],
                        help='Training verbosity: 0=silent, 1=progress bar, 2=one line/epoch (default: 2)')
    parser.add_argument('--configs', nargs='+', default=None,
                        help='Specific configs to run (e.g. A_baseline B_symm)')
    parser.add_argument('--list', action='store_true',
                        help='List all configs and exit')
    parser.add_argument('--eval', nargs='+', default=None,
                        help='Evaluate pretrained model files (e.g. --eval model.h5 best_clf.keras)')
    parser.add_argument('--arrow-dir', type=str, default=None,
                        help='Alternate arrow hint directory (e.g. from partner U-Net)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force reload data (ignore cache)')
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Config':<24} Description")
        print("-" * 90)
        for k, v in CONFIGS.items():
            print(f"{k:<24} {v['desc']}")
        return

    # Add any --eval model paths as configs
    if args.eval:
        for model_file in args.eval:
            tag = Path(model_file).stem.replace('.', '_').replace('-', '_')
            config_name = f"P_{tag}"
            CONFIGS[config_name] = dict(
                desc=f'Evaluate pretrained: {model_file}',
                hint='bbox', aug='max', loss='bce', cls_w_cap=None,
                label_smooth=0.0, monitor='val_auc', monitor_mode='max',
                p1_epochs=0, p2_epochs=0, online_aug=False,
                model_path=model_file,
            )
            if args.configs is None:
                args.configs = []
            args.configs.append(config_name)

    quick = not args.full
    mode_str = "QUICK SCREENING (15+20 epochs)" if quick else "FULL TRAINING"
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY — {mode_str}")
    print(f"{'='*70}")

    if args.no_cache and os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    # Override arrow directory if specified (e.g. partner's U-Net)
    global ARROW_DIR
    if args.arrow_dir:
        ARROW_DIR = args.arrow_dir.rstrip('/') + '/'
        print(f"  Using alternate arrow dir: {ARROW_DIR}")
        # Must regenerate cache since arrows are baked in
        if os.path.exists(CACHE_FILE):
            print(f"  Deleting cache (arrows changed)")
            os.remove(CACHE_FILE)

    # Need CLIP_LOOKUP in global scope for data loading
    global CLIP_LOOKUP
    if os.path.exists(CACHE_FILE):
        CLIP_LOOKUP = {}  # not needed when loading from cache
    else:
        _clip_data  = np.load(CLIP_FILE, allow_pickle=True)
        CLIP_LOOKUP = dict(zip(_clip_data['stems'].tolist(),
                               _clip_data['embeddings'].astype(np.float32)))

    data = load_all_data()
    print(f"\nTotal images: {len(data['all_y'])}")
    for i, c in enumerate(display_labels):
        print(f"  {c:<22}: {int(data['all_y'][:, i].sum()):>5}")

    # Consistent split
    idx_tv, idx_test   = train_test_split(np.arange(len(data['all_y'])),
                                           test_size=0.2, random_state=10)
    idx_train, idx_val = train_test_split(idx_tv, test_size=0.2, random_state=10)
    print(f"Train:{len(idx_train)}  Val:{len(idx_val)}  Test:{len(idx_test)}")

    # Select configs
    config_names = args.configs if args.configs else list(CONFIGS.keys())
    configs_to_run = [(k, CONFIGS[k]) for k in config_names if k in CONFIGS]
    print(f"\nRunning {len(configs_to_run)} configs: {[c[0] for c in configs_to_run]}")

    # Run
    results = []
    for name, cfg in configs_to_run:
        try:
            r = run_config(name, cfg, data, idx_train, idx_val, idx_test,
                          quick, args.verbose)
            results.append(r)
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            import traceback; traceback.print_exc()

    # ── Results table ────────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("ABLATION RESULTS")
    print(f"{'='*90}")

    # Sort by macro F1
    results.sort(key=lambda r: r['macro_f1'], reverse=True)

    lin_f1 = 0.6503
    print(f"\n{'Config':<24} {'Macro F1':>9} {'Hamming':>9} {'ΔvsLin':>8} {'Time':>7}  Description")
    print("-" * 110)
    for r in results:
        delta = r['macro_f1'] - lin_f1
        beat  = "✅" if r['macro_f1'] > lin_f1 else "  "
        print(f"{r['name']:<24} {r['macro_f1']:>9.4f} {r['hamming_loss']:>9.4f} "
              f"{delta:>+8.4f}{beat} {r['elapsed_min']:>5.1f}m  {r['desc'][:40]}")

    # Per-class breakdown
    print(f"\n{'Config':<24}", end="")
    for c in display_labels:
        print(f" {c[:6]:>8}", end="")
    print()
    print("-" * (24 + 8 * len(display_labels)))
    for r in results:
        print(f"{r['name']:<24}", end="")
        for c in display_labels:
            print(f" {r['per_class_f1'][c]:>8.3f}", end="")
        print()

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = FIG_DIR + f"ablation_{ts}.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Best config
    best = results[0]
    print(f"\n🏆 Best: {best['name']} — Macro F1 {best['macro_f1']:.4f}, Hamming {best['hamming_loss']:.4f}")
    if best['macro_f1'] > lin_f1:
        print("   Beats Lin & Peng!")
    else:
        gap = lin_f1 - best['macro_f1']
        print(f"   Gap to Lin & Peng: {gap:.4f}")


if __name__ == '__main__':
    main()
