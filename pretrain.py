import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, ast, warnings
warnings.filterwarnings("ignore")

# Suppress harmless semaphore-leak warning from tf prefetch threads (if present)
try:
    import atexit, multiprocessing.resource_tracker as _rt
    atexit.register(lambda: _rt._resource_tracker._stop())
except (ImportError, AttributeError):
    pass

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Force CPU: Metal plugin causes abort on Epoch 1 and is slower anyway ──────
try:
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    pass  # already initialised

from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from pathlib import Path

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, hamming_loss, precision_score, recall_score,
    classification_report, precision_recall_curve, confusion_matrix
)
# ImageDataGenerator removed — replaced by light online augmentation via tf.data

# Use all available CPU threads
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)

print(f"TF  : {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')} (Metal disabled — CPU is faster for this workload)")

# ── Run control ───────────────────────────────────────────────────────────────
RETRAIN_CLASSIFIER   = True
RETRAIN_SEGMENTATION = False
SKIP_CLASSIFIER      = os.environ.get('SKIP_CLASSIFIER', '0') == '1'
USE_ARROW_HINTS      = True
CLIP_EMB_DIM         = 512

# ── Paths ─────────────────────────────────────────────────────────────────────
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
SEG_SIZE   = 256      # Full-res — thin hemorrhage masks (epidural crescents, SAH) need spatial detail
BATCH_SIZE = 32
SEG_BATCH  = 16       # Smaller batch for U-Net at full-res to keep CPU memory in check
EPOCHS     = 80
SEED       = 42

MODEL_DIR   = './output/models/';   os.makedirs(MODEL_DIR,   exist_ok=True)
FIG_DIR     = './output/figures/';  os.makedirs(FIG_DIR,     exist_ok=True)
ARROW_DIR   = './output/arrow_hints/'
CLIP_FILE   = './output/clip_embeddings.npz'

# ── Run tagging — every checkpoint includes timestamp + metric ────────────────
from datetime import datetime
import re, glob as _glob

RUN_TAG = datetime.now().strftime('%Y%m%d_%H%M%S')
print(f"Run tag: {RUN_TAG}")

def _find_best_ckpt(prefix, metric_key, mode="max"):
    """Find the best checkpoint matching `prefix*{metric_key}*.keras`.
    Parses the metric value from the filename (e.g. 'loss0.8712').
    Returns (path, metric_value) or (None, None)."""
    pattern = MODEL_DIR + prefix + f"*{metric_key}*.keras"
    hits = _glob.glob(pattern)
    if not hits:
        return None, None
    fallback = -1.0 if mode == "max" else float('inf')
    def _parse_metric(path):
        m = re.search(rf'{metric_key}(\d+\.\d+)', os.path.basename(path))
        return float(m.group(1)) if m else fallback
    best = max(hits, key=_parse_metric) if mode == "max" else min(hits, key=_parse_metric)
    val = _parse_metric(best)
    if val == fallback:
        return None, None
    return best, val

# RadImageNet — pretrained on 1.35M radiology images (CT/MRI/US)
# Download from: https://www.kaggle.com/datasets/ipythonx/notop-wg-radimagenet
RADIMAGENET_WEIGHTS = './weights/RadImageNet-DenseNet121_notop.h5'

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Load CLIP embeddings ──────────────────────────────────────────────────────
print("Loading CLIP embeddings...")
_clip_data  = np.load(CLIP_FILE, allow_pickle=True)
_clip_stems = _clip_data['stems'].tolist()
_clip_embs  = _clip_data['embeddings'].astype(np.float32)
CLIP_LOOKUP = dict(zip(_clip_stems, _clip_embs))
print(f"  Loaded {len(CLIP_LOOKUP)} CLIP embeddings ({CLIP_EMB_DIM}-d)")

def get_clip_emb(img_name):
    stem = Path(img_name).stem
    return CLIP_LOOKUP.get(stem, np.zeros(CLIP_EMB_DIM, dtype=np.float32))

def get_arrow_hint(img_name):
    stem = Path(img_name).stem
    p    = ARROW_DIR + stem + '.npy'
    if USE_ARROW_HINTS and os.path.exists(p):
        return np.load(p).astype(np.float32)
    return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

# ── Preprocessing helpers ─────────────────────────────────────────────────────
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
    r_raw = tf.image.resize(raw_img,       sz).numpy()
    r_msk = tf.image.resize(seg_mask,      sz).numpy()
    norm  = r_raw / 255.0
    gray  = cv2.cvtColor(norm,  cv2.COLOR_BGR2GRAY).reshape(target_size, target_size, 1)
    g_msk = cv2.cvtColor(r_msk, cv2.COLOR_BGR2GRAY).reshape(target_size, target_size, 1)
    return norm, gray, g_msk

def get_one_hot_label(img_name, label_df):
    stem    = Path(img_name).stem
    vec     = label_df.loc[stem][class_cols].to_numpy().copy()
    any_idx = class_cols.index('any')
    vec[any_idx] = (vec[any_idx] + 1) % 2
    return vec.astype(np.float32)

def symmetry_hint(img_4ch):
    """Bilateral symmetry deviation — highlights asymmetric regions in the brain.
    Healthy brain tissue is roughly left-right symmetric; hemorrhage breaks this.
    Uses the brain window (ch1) and bone window (ch0) for complementary signals.

    Returns (IMG_SIZE, IMG_SIZE, 1) float32 in [0, 1].
    """
    brain = img_4ch[:, :, 1]  # brain window — best for parenchymal hemorrhage
    bone  = img_4ch[:, :, 0]  # bone window — good for epidural (near skull)

    # Flip horizontally and compute absolute difference
    brain_diff = np.abs(brain - np.fliplr(brain))
    bone_diff  = np.abs(bone - np.fliplr(bone))

    # Combine: brain window weighted higher (most hemorrhage is parenchymal)
    combined = 0.7 * brain_diff + 0.3 * bone_diff

    # Suppress the skull edge (always asymmetric due to slight tilt/positioning)
    # Mask out a thin border where registration artifacts live
    border = 12
    combined[:border, :]  = 0
    combined[-border:, :] = 0
    combined[:, :border]  = 0
    combined[:, -border:] = 0

    # Smooth to reduce noise and create a soft attention map
    combined = cv2.GaussianBlur(combined, (15, 15), 0)

    # Normalize to [0, 1]
    mx = combined.max()
    if mx > 0:
        combined = combined / mx

    return combined.reshape(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

# ── Data loading ──────────────────────────────────────────────────────────────
label_df = pd.read_csv(seg_dir + LABEL_FILE + EXT_CSV, index_col='Image')

# ── Flagged / corrupt image exclusion (from Kelvin's data cleaning v2) ────────
FLAGGED_FILE = './flagged.txt'
CORRUPT_IDS  = {'ID_6431af929'}  # known corrupt file — do not attempt to read
_flagged_stems = set(CORRUPT_IDS)
if os.path.exists(FLAGGED_FILE):
    with open(FLAGGED_FILE) as f:
        for line in f:
            stem = line.strip()
            if stem:
                _flagged_stems.add(stem)
    print(f"Loaded {len(_flagged_stems)} flagged/corrupt cases from {FLAGGED_FILE}")
else:
    print(f"No {FLAGGED_FILE} found — skipping only known corrupt IDs ({len(CORRUPT_IDS)})")

def is_flagged(img_name):
    """Check if an image is flagged or corrupt and should be skipped."""
    stem = Path(img_name).stem
    return stem in _flagged_stems

all_3ch   = []   # NOTE: actually 4-channel (bone/brain/contrast/subdural windows) — legacy name
all_seg   = []
all_symm  = []
all_arrow = []
all_clip  = []
all_y     = []

for cls in CONTOUR_CLASSES:
    cls_img_dir = cnt_dir + cls + '/'
    if not os.path.isdir(cls_img_dir):
        print(f"  SKIPPING {cls} (no contour dir)")
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
    img_names      = [f for f in os.listdir(cls_img_dir) if f != '.DS_Store']
    n_loaded       = 0

    for img_name in tqdm(img_names, desc=cls, unit='image'):
        if is_flagged(img_name):
            continue
        scan = cv2.imread(cls_img_dir + img_name)
        if scan is None:
            continue
        row_matches      = ann_df[ann_df.Origin == img_name]
        combined_regions = []
        for cl_val, ml_val, al_val in zip(row_matches['Correct Label'],
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
        channels  = stack_window_channels(render_cls_dir, img_name)
        if len(channels) < 4:
            continue
        img_3ch = np.stack([cv2.resize(c, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                             for c in channels[:4]], axis=-1)
        try:
            lbl = get_one_hot_label(img_name, label_df)
        except KeyError:
            continue

        all_3ch.append(img_3ch);         all_seg.append(gray_mask)
        all_symm.append(symmetry_hint(img_3ch))
        all_arrow.append(get_arrow_hint(img_name))
        all_clip.append(get_clip_emb(img_name))
        all_y.append(lbl)
        n_loaded += 1

    print(f"  {cls}: {n_loaded} / {len(img_names)} loaded")

def load_renders(cls_name, cap=1000):
    rdir = render_dir + cls_name
    bdir = rdir + BRAIN_WIN
    if not os.path.isdir(bdir):
        print(f"  SKIPPING {cls_name} renders (not found)")
        return
    files    = [f for f in os.listdir(bdir) if f != '.DS_Store' and not is_flagged(f)][:cap]
    n_loaded = 0
    for img_name in tqdm(files, desc=cls_name, unit='image'):
        channels = stack_window_channels(rdir, img_name)
        if len(channels) < 4:
            continue
        img_3ch   = np.stack([cv2.resize(c, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                               for c in channels[:4]], axis=-1)
        try:
            lbl = get_one_hot_label(img_name, label_df)
        except KeyError:
            continue
        all_3ch.append(img_3ch)
        all_seg.append(np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32))
        all_symm.append(symmetry_hint(img_3ch))
        all_arrow.append(get_arrow_hint(img_name))
        all_clip.append(get_clip_emb(img_name))
        all_y.append(lbl)
        n_loaded += 1
    print(f"  {cls_name}: {n_loaded} loaded from renders")

load_renders(CLS_NORMAL, cap=500)
load_renders(CLS_INTRAVENTRICULAR, cap=500)
# Epidural: 241 contour + 300 renders ≈ 541 total (was 241 — 2.2× more)
load_renders(CLS_EPIDURAL, cap=300)
# Subarachnoid already at 500 from contour — no extra renders needed

all_3ch   = np.array(all_3ch,   dtype=np.float32)
all_seg   = np.array(all_seg,   dtype=np.float32)
all_symm  = np.array(all_symm,  dtype=np.float32)
all_arrow = np.array(all_arrow, dtype=np.float32)
all_clip  = np.array(all_clip,  dtype=np.float32)
all_y     = np.array(all_y,     dtype=np.float32)

# ── Automated quality filtering (inspired by Ying et al. VGG16 cleaning) ─────
# Instead of training a separate quality classifier, we use unsupervised metrics
# on the brain-window channel to reject uninformative slices (edge slices, scout
# views, low-content images). Ying et al. showed +8% F1 from quality filtering.
def compute_quality_scores(images_4ch):
    """Score each image by brain content + information density.
    Uses the brain window (channel 1) as the primary signal."""
    scores = np.empty(len(images_4ch), dtype=np.float32)
    for i in range(len(images_4ch)):
        brain_ch = images_4ch[i, :, :, 1]   # brain window channel
        # Content ratio: fraction of pixels above noise floor
        content_ratio = np.mean(brain_ch > 0.05)
        # Variance: information density (low = uninformative)
        variance = np.var(brain_ch)
        # Combined score — both matter, multiplicative so both must be decent
        scores[i] = content_ratio * variance
    return scores

QUALITY_PERCENTILE = 10  # filter bottom 10% — adjust if too aggressive/lenient
q_scores  = compute_quality_scores(all_3ch)
q_thresh  = np.percentile(q_scores, QUALITY_PERCENTILE)
keep_mask = q_scores >= q_thresh
n_removed = int((~keep_mask).sum())

print(f"\n── Quality filtering ──")
print(f"  Score range: [{q_scores.min():.6f}, {q_scores.max():.6f}]")
print(f"  Threshold (p{QUALITY_PERCENTILE}): {q_thresh:.6f}")
print(f"  Removing {n_removed} / {len(all_y)} low-quality images ({100*n_removed/len(all_y):.1f}%)")

# Show per-class impact
for i, col in enumerate(display_labels):
    cls_mask = all_y[:, i] == 1
    cls_removed = int((cls_mask & ~keep_mask).sum())
    cls_total   = int(cls_mask.sum())
    if cls_total > 0:
        print(f"    {col:<22}: {cls_removed:>4} removed / {cls_total:>5} ({100*cls_removed/cls_total:.1f}%)")

all_3ch   = all_3ch[keep_mask]
all_seg   = all_seg[keep_mask]
all_symm  = all_symm[keep_mask]
all_arrow = all_arrow[keep_mask]
all_clip  = all_clip[keep_mask]
all_y     = all_y[keep_mask]

n_arrow = int((all_arrow.max(axis=(1, 2, 3)) > 0).sum())
n_clip  = int((all_clip.sum(axis=1) != 0).sum())
print(f"\nTotal images   : {len(all_y)}")
print(f"Arrow hints    : {n_arrow} / {len(all_y)} non-zero")
print(f"CLIP embeddings: {n_clip} / {len(all_y)} non-zero")
print("Class distribution:")
for i, col in enumerate(display_labels):
    n = int(all_y[:, i].sum())
    print(f"  {col:<22}: {n:>5}  ({100*n/max(len(all_y),1):.1f}%)")

# ── Split ─────────────────────────────────────────────────────────────────────
idx_tv, idx_test   = train_test_split(np.arange(len(all_y)), test_size=0.2, random_state=10)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.2, random_state=10)

X_tr_3ch   = all_3ch[idx_train];    X_v_3ch   = all_3ch[idx_val];    X_te_3ch   = all_3ch[idx_test]
X_tr_seg   = all_seg[idx_train];    X_v_seg   = all_seg[idx_val];    X_te_seg   = all_seg[idx_test]
X_tr_symm  = all_symm[idx_train];   X_v_symm  = all_symm[idx_val];   X_te_symm  = all_symm[idx_test]
X_tr_arrow = all_arrow[idx_train];  X_v_arrow = all_arrow[idx_val];  X_te_arrow = all_arrow[idx_test]
X_tr_clip  = all_clip[idx_train];   X_v_clip  = all_clip[idx_val];   X_te_clip  = all_clip[idx_test]
y_train    = all_y[idx_train];      y_val     = all_y[idx_val];      y_test     = all_y[idx_test]
print(f"Train:{len(idx_train)}  Val:{len(idx_val)}  Test:{len(idx_test)}")

# ── No synthetic augmentation or downsampling ─────────────────────────────────
# Class imbalance is handled by three mechanisms in the loss:
#   1. CLS_W in weighted_bce (upweights rare classes per-sample)
#   2. soft_macro_f1_loss (equalises per-class F1 globally)
#   3. Label smoothing (reduces overconfidence on noisy labels)
# Online augmentation (flips + brightness) in tf.data provides regularisation.
print(f"\n── Training class distribution (no downsampling) ──")
train_df = pd.DataFrame(y_train, columns=display_labels)
for c in display_labels:
    print(f"    {c:<22}: {int(train_df[c].sum()):>5}")
print(f"  Total training samples: {len(y_train)}")

# ── Build hint array: seg + symmetry + arrow ─────────────────────────────────
def make_hints(seg, symm, arrow):
    return np.concatenate([seg, symm, arrow], axis=-1).astype(np.float32)

tr_hints = np.concatenate([X_tr_seg, X_tr_symm, X_tr_arrow], axis=-1).astype(np.float32)
v_hints  = np.concatenate([X_v_seg,  X_v_symm,  X_v_arrow],  axis=-1).astype(np.float32)
te_hints = np.concatenate([X_te_seg, X_te_symm, X_te_arrow], axis=-1).astype(np.float32)

# ── TF Datasets — from_tensor_slices avoids generator abort issues ────────────
@tf.function
def clf_augment(inputs, labels):
    """Light online augmentation for classifier — flips + brightness on CT and hints.
    Replaces heavy offline augmentation. CLIP embeddings are not spatial so left alone."""
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

def make_tf_dataset(c3, hints, clip, y, shuffle=True, augment=False):
    """
    Uses from_tensor_slices instead of from_generator.
    For training: cache raw → shuffle → augment → batch (fresh augmentations each epoch).
    For val/test: batch → cache (deterministic).
    """
    ds = tf.data.Dataset.from_tensor_slices((
        {"ct_4ch": c3, "hints_3ch": hints, "clip_emb": clip},
        y
    ))
    if augment:
        # Cache raw data, then shuffle + augment fresh each epoch
        ds = ds.cache()
        ds = ds.shuffle(len(y), reshuffle_each_iteration=True)
        ds = ds.map(clf_augment, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    else:
        if shuffle:
            ds = ds.shuffle(len(y), seed=SEED)
        return ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

train_ds = make_tf_dataset(X_tr_3ch, tr_hints, X_tr_clip, y_train, True, augment=True)
val_ds   = make_tf_dataset(X_v_3ch,  v_hints,  X_v_clip,  y_val,   False)
test_ds  = make_tf_dataset(X_te_3ch, te_hints, X_te_clip, y_test,  False)

# ── Per-class weighted BCE (replaces broken class_weight for multi-label) ────
_pos = y_train.sum(axis=0)
_neg = len(y_train) - _pos
_raw_w = (_neg / (_pos + 1e-6)).astype(np.float32)
MAX_CLS_W = 5.0  # Cap — uncapped weights (10-20×) push model to always predict positive
_capped_w = np.clip(_raw_w, 1.0, MAX_CLS_W)
CLS_W = tf.constant(_capped_w)
print("Per-class pos-weight (raw → capped):")
for c, rw, cw in zip(display_labels, _raw_w, _capped_w):
    clipped = " ← capped" if rw > MAX_CLS_W else ""
    print(f"  {c:<22}: {rw:>6.1f} → {cw:>4.1f}{clipped}")

def weighted_bce(y_true, y_pred):
    # Label smoothing — softens hard 0/1 targets to reduce overconfidence
    # on residual label noise (post-flagged.txt filtering)
    eps    = 0.05
    y_sm   = y_true * (1.0 - eps) + 0.5 * eps
    bce     = tf.keras.backend.binary_crossentropy(y_sm, y_pred)   # (B, C)
    weights = y_true * CLS_W + (1.0 - y_true) * 1.0
    return tf.reduce_mean(bce * weights)

def soft_macro_f1_loss(y_true, y_pred):
    """Differentiable macro F1 loss — directly optimises the evaluation metric.
    Uses soft counts (sigmoid probs as fractional TP/FP/FN) instead of hard
    thresholds, so gradients flow. Computed per-class, then averaged (macro).
    Returns 1 - macro_F1 so minimising the loss maximises F1."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    # Soft counts per class (summed across batch)
    tp = tf.reduce_sum(y_true * y_pred, axis=0)           # (C,)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=0)   # (C,)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=0)   # (C,)

    # Per-class soft F1
    soft_f1 = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)  # (C,)

    # Macro average
    return 1.0 - tf.reduce_mean(soft_f1)

def combined_clf_loss(y_true, y_pred):
    """Weighted BCE (per-sample stability) + Soft Macro F1 (global class balance).
    BCE keeps gradients flowing early; F1 steers toward the actual eval metric.
    F1 at 0.2× — gentle steering, not compounding with CLS_W recall pressure."""
    return weighted_bce(y_true, y_pred) + 0.2 * soft_macro_f1_loss(y_true, y_pred)

# ── 4-channel DenseNet121 with smart weight initialization ────────────────────
def _load_densenet121_4ch(input_shape, weights_path=None):
    """
    Build DenseNet121 that accepts 4 CT window channels.
    Strategy: load pretrained 3ch weights, expand the first conv kernel from
    (7,7,3,64) → (7,7,4,64) by initializing the 4th channel as the mean of
    the existing 3. This preserves pretrained features instead of random init.
    """
    h, w, _ = input_shape  # ignore channel dim, we force 4

    # Step 1: build 3ch model, load weights
    model_3ch = keras.applications.DenseNet121(
        input_shape=(h, w, 3), include_top=False, weights=None)
    if weights_path and os.path.exists(weights_path):
        model_3ch.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print(f"    Loaded RadImageNet weights from {weights_path}")
    else:
        # Fall back to ImageNet
        model_3ch_img = keras.applications.DenseNet121(
            input_shape=(h, w, 3), include_top=False, weights="imagenet")
        model_3ch.set_weights(model_3ch_img.get_weights())
        del model_3ch_img
        print("    Using ImageNet weights (RadImageNet not found)")

    # Step 2: get first conv weights — shape (7, 7, 3, 64)
    first_conv = model_3ch.get_layer("conv1_conv")
    w3 = first_conv.get_weights()[0]  # (7, 7, 3, 64)

    # Step 3: expand to (7, 7, 4, 64) — 4th channel = mean of existing 3
    w4_new = np.mean(w3, axis=2, keepdims=True)  # (7, 7, 1, 64)
    w4 = np.concatenate([w3, w4_new], axis=2)     # (7, 7, 4, 64)

    # Step 4: build 4ch model
    model_4ch = keras.applications.DenseNet121(
        input_shape=(h, w, 4), include_top=False, weights=None)

    # Step 5: copy all weights, replacing first conv
    for layer_3, layer_4 in zip(model_3ch.layers, model_4ch.layers):
        if layer_4.name == "conv1_conv":
            old_weights = layer_3.get_weights()
            layer_4.set_weights([w4] + old_weights[1:])  # keep bias if any
        else:
            try:
                layer_4.set_weights(layer_3.get_weights())
            except (ValueError, Exception):
                pass  # skip mismatched layers

    del model_3ch
    return model_4ch

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    img_in  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 4), name="ct_4ch")
    hint_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="hints_3ch")
    clip_in = keras.Input(shape=(CLIP_EMB_DIM,),         name="clip_emb")

    # CT backbone — DenseNet121 + RadImageNet, native 4ch input
    import glob
    rad_path = RADIMAGENET_WEIGHTS
    if not os.path.exists(rad_path):
        candidates = glob.glob('./weights/*[Dd]ense[Nn]et*121*.h5')
        if candidates:
            rad_path = candidates[0]
    print("  Building classifier with 4-channel CT input...")
    base = _load_densenet121_4ch(
        (IMG_SIZE, IMG_SIZE, 4),
        weights_path=rad_path if os.path.exists(rad_path) else None)
    base.trainable = False
    x = base(img_in, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    # Hint branch
    h = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(hint_in)
    h = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(h)
    h = layers.GlobalAveragePooling2D()(h)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.3)(h)

    # CLIP branch
    c = layers.Dense(256, activation="relu")(clip_in)
    c = layers.BatchNormalization()(c)
    c = layers.Dense(128, activation="relu")(c)
    c = layers.Dropout(0.3)(c)

    fused = layers.Concatenate()([x, h, c])
    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.3)(fused)
    out   = layers.Dense(N_CLS, activation="sigmoid", name="output")(fused)

    model = keras.Model(inputs=[img_in, hint_in, clip_in], outputs=out,
                        name="HintDenseNet121_CLIP")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=combined_clf_loss,
        metrics=[
            keras.metrics.BinaryAccuracy(name="bin_acc"),
            keras.metrics.AUC(name="auc", multi_label=True),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )
    return model, base

# ── Collect predictions ───────────────────────────────────────────────────────
def collect_preds(ds, mdl):
    probs, labels = [], []
    for batch_x, lbl_b in ds:
        probs.append(mdl.predict(batch_x, verbose=0))
        labels.append(lbl_b.numpy())
    return np.vstack(probs), np.vstack(labels)

if not SKIP_CLASSIFIER:
    # ── Training ──────────────────────────────────────────────────────────────────
    backbone = None
    if not RETRAIN_CLASSIFIER:
        # Try P2 first (fine-tuned), then P1 (frozen backbone), then legacy names
        # New checkpoints use val_loss (lower=better); old ones used val_auc (higher=better)
        clf_path, clf_metric = _find_best_ckpt("clf_p2_", "loss", mode="min")
        if clf_path is None:
            clf_path, clf_metric = _find_best_ckpt("clf_p1_", "loss", mode="min")
        if clf_path is None:
            # Fall back to auc-based checkpoints from older runs
            clf_path, clf_metric = _find_best_ckpt("clf_p2_", "auc", mode="max")
        if clf_path is None:
            clf_path, clf_metric = _find_best_ckpt("clf_p1_", "auc", mode="max")
        if clf_path is None:
            for legacy in ["best_clf_p2.keras", "best_clf_p1.keras"]:
                if os.path.exists(MODEL_DIR + legacy):
                    clf_path = MODEL_DIR + legacy; break
        if clf_path:
            try:
                metric_str = ""
                if clf_metric and clf_metric > 0:
                    metric_str = f"  (loss={clf_metric:.4f})" if "loss" in os.path.basename(clf_path) else f"  (auc={clf_metric:.4f})"
                print(f"\nLoading saved classifier: {os.path.basename(clf_path)}{metric_str}")
                model      = keras.models.load_model(
                    clf_path, custom_objects={
                        "weighted_bce": weighted_bce,
                        "soft_macro_f1_loss": soft_macro_f1_loss,
                        "combined_clf_loss": combined_clf_loss,
                    })
                expected_ch = model.input_shape[0][-1] if isinstance(model.input_shape, list) else model.input_shape[-1]
                if expected_ch != all_3ch.shape[-1]:
                    print(f"  ⚠ Shape mismatch ({expected_ch}ch vs {all_3ch.shape[-1]}ch) — retraining.")
                    RETRAIN_CLASSIFIER = True
                else:
                    history_p1 = None
            except Exception as e:
                print(f"  ⚠ Failed to load: {e} — retraining.")
                RETRAIN_CLASSIFIER = True
        else:
            RETRAIN_CLASSIFIER = True

    if RETRAIN_CLASSIFIER:
        model, backbone = build_model()
        model.summary()

        P1_EPOCHS = 25  # Short Phase 1 — frozen backbone just warms up the head

        def cosine_lr_with_warmup(epoch, lr, warmup=3, base_lr=1e-3, min_lr=1e-5):
            if epoch < warmup:
                return base_lr * (epoch + 1) / warmup
            progress = (epoch - warmup) / max(P1_EPOCHS - warmup, 1)
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))

        callbacks_p1 = [
            keras.callbacks.ModelCheckpoint(
                MODEL_DIR + f"clf_p1_{RUN_TAG}_loss{{val_loss:.4f}}.keras",
                monitor="val_loss", mode="min", save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=12, restore_best_weights=True),
            keras.callbacks.LearningRateScheduler(cosine_lr_with_warmup, verbose=0),
            keras.callbacks.CSVLogger(MODEL_DIR + f"log_p1_{RUN_TAG}.csv"),
        ]

        print("\n── Phase 1: frozen backbone (short — just warmup head layers) ──")
        history_p1 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=P1_EPOCHS, callbacks=callbacks_p1,
        )

        # ── Phase 2: fine-tune top layers of DenseNet121 ─────────────────────────
        if backbone is not None:
            print("\n── Phase 2: unfreezing top backbone layers ──")
            for layer in backbone.layers[-40:]:
                layer.trainable = True

            model.compile(
                optimizer=keras.optimizers.Adam(1e-5),
                loss=combined_clf_loss,
                metrics=[
                    keras.metrics.BinaryAccuracy(name="bin_acc"),
                    keras.metrics.AUC(name="auc", multi_label=True),
                    keras.metrics.Precision(name="precision"),
                    keras.metrics.Recall(name="recall"),
                ]
            )

            callbacks_p2 = [
                keras.callbacks.ModelCheckpoint(
                    MODEL_DIR + f"clf_p2_{RUN_TAG}_loss{{val_loss:.4f}}.keras",
                    monitor="val_loss", mode="min", save_best_only=True, verbose=1),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", mode="min", patience=15, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-7),
                keras.callbacks.CSVLogger(MODEL_DIR + f"log_p2_{RUN_TAG}.csv"),
            ]

            history_p2 = model.fit(
                train_ds, validation_data=val_ds,
                epochs=100, callbacks=callbacks_p2,
            )

    # ── Threshold optimisation ────────────────────────────────────────────────────
    val_probs, val_true = collect_preds(val_ds, model)

    def find_best_thresholds(y_true, y_prob):
        thresholds, f1s = [], []
        print(f"\n{'Class':<22} {'Thresh':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print("-" * 56)
        for i, cls in enumerate(display_labels):
            prec, rec, thresh = precision_recall_curve(y_true[:, i], y_prob[:, i])
            f1  = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-8), 0)
            idx = np.argmax(f1)
            t   = float(thresh[idx]) if idx < len(thresh) else 0.5
            thresholds.append(t); f1s.append(f1[idx])
            print(f"{cls:<22} {t:>8.4f} {prec[idx]:>8.4f} {rec[idx]:>8.4f} {f1[idx]:>8.4f}")
        return np.array(thresholds), np.array(f1s)

    best_thresh, best_f1s = find_best_thresholds(val_true, val_probs)
    np.save(MODEL_DIR + f"thresholds_{RUN_TAG}.npy", best_thresh)

    # ── Test evaluation ───────────────────────────────────────────────────────────
    test_probs, test_true = collect_preds(test_ds, model)
    test_preds = (test_probs >= best_thresh).astype(int)

    print("\n" + "=" * 55)
    print("TEST SET RESULTS")
    print("=" * 55)
    print(classification_report(test_true, test_preds,
                                 target_names=display_labels, zero_division=0))
    hl  = hamming_loss(test_true, test_preds)
    mf1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    print(f"Hamming Loss : {hl:.4f}")
    print(f"Macro F1     : {mf1:.4f}")

    # Build a results tag for figures: e.g. "20260418_143022_f1-0.6432_hl-0.1049"
    RESULTS_TAG = f"{RUN_TAG}_f1-{mf1:.4f}_hl-{hl:.4f}"

    # ── Plots ─────────────────────────────────────────────────────────────────────
    def save_fig(fname):
        plt.tight_layout()
        plt.savefig(FIG_DIR + fname, dpi=120)
        plt.close()
        print(f"Saved {FIG_DIR}{fname}")

    if history_p1:
        h1  = history_p1.history
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for i, (key, title) in enumerate(
                zip(["loss", "auc", "bin_acc"], ["BCE Loss", "AUC", "Binary Acc"])):
            p1_tr  = h1[key]
            p1_val = h1[f"val_{key}"]
            ax[i].plot(p1_tr,  label="P1 train")
            ax[i].plot(p1_val, label="P1 val")
            if 'history_p2' in dir() and history_p2 is not None:
                h2     = history_p2.history
                offset = len(p1_tr)
                epochs2 = range(offset, offset + len(h2[key]))
                ax[i].plot(epochs2, h2[key],          label="P2 train", linestyle="--")
                ax[i].plot(epochs2, h2[f"val_{key}"], label="P2 val",   linestyle="--")
                ax[i].axvline(x=offset, color='gray', linestyle=':', alpha=0.5)
            ax[i].set_title(title); ax[i].legend(fontsize=7); ax[i].grid(alpha=0.3)
        plt.suptitle("Training — HintDenseNet121 + CLIP (Phase 1 + 2)")
        save_fig(f"training_curves_{RESULTS_TAG}.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, cls, i in zip(axes.flat, display_labels, range(N_CLS)):
        cm = confusion_matrix(test_true[:, i], test_preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        f1 = f1_score(test_true[:, i], test_preds[:, i], zero_division=0)
        ax.set_title(f"{cls}\nF1={f1:.3f}")
        ax.set_ylabel("True"); ax.set_xlabel("Pred")
    plt.suptitle("Per-Class Confusion Matrices")
    save_fig(f"confusion_matrices_{RESULTS_TAG}.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(display_labels):
        prec, rec, _ = precision_recall_curve(test_true[:, i], test_probs[:, i])
        f1 = (2 * prec * rec / (prec + rec + 1e-8)).max()
        ax.plot(rec, prec, label=f"{cls} (F1={f1:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_fig(f"pr_curves_{RESULTS_TAG}.png")

    # ── Comparison ────────────────────────────────────────────────────────────────
    ying_acc = {"normal": None, "epidural": 0.9833, "intraparenchymal": 0.9409,
                "intraventricular": 0.9534, "subarachnoid": 0.8969, "subdural": 0.9267}
    lin_pr   = {"normal": (0.617, 0.699), "epidural": (0.598, 0.549),
                "intraparenchymal": (0.853, 0.616), "subarachnoid": (0.564, 0.428),
                "intraventricular": (0.757, 0.832), "subdural": (0.439, 0.506)}

    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Class':<22} {'Lin F1':>8} {'Ying Acc':>10} {'Ours F1':>9} {'Δ vs Lin':>10}")
    print("-" * 70)
    for i, cls in enumerate(display_labels):
        f1     = f1_score(test_true[:, i], test_preds[:, i], zero_division=0)
        lp, lr = lin_pr[cls]
        lf1    = 2 * lp * lr / (lp + lr + 1e-8)
        ya     = ying_acc[cls]
        print(f"{cls:<22} {lf1:>8.3f} {str(round(ya,4)) if ya else 'N/A':>10} "
              f"{f1:>9.3f} {f1-lf1:>+10.3f}")

    print(f"\n{'Metric':<25} {'Lin & Peng':>12} {'Ying et al':>12} {'Ours':>10}")
    print("-" * 60)
    print(f"{'Macro F1':<25} {0.6503:>12.4f} {0.8947:>12.4f} {mf1:>10.4f}")
    print(f"{'Hamming Loss':<25} {0.1279:>12.4f} {'N/A':>12} {hl:>10.4f}")
    print(f"\nBeat Lin & Peng?   {'✅' if mf1 > 0.6503 else '❌'}")
    print(f"Beat Ying et al.?  {'✅' if mf1 > 0.8947 else '❌  target: F1 > 0.89'}")

    # ── Save final model snapshot with test metrics in filename ───────────────
    final_path = MODEL_DIR + f"clf_final_{RESULTS_TAG}.keras"
    model.save(final_path)
    print(f"\nFinal classifier saved: {os.path.basename(final_path)}")


# ── Segmentation ──────────────────────────────────────────────────────────────
def dice_coef(y_t, y_p, s=1e-6):
    y_t = tf.reshape(tf.cast(y_t > 0.5, tf.float32), [-1])
    y_p = tf.reshape(tf.cast(y_p > 0.5, tf.float32), [-1])
    return float((2 * tf.reduce_sum(y_t * y_p) + s) /
                 (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + s))

def soft_dice_loss(y_true, y_pred):
    """Differentiable dice loss — works on soft predictions (no threshold)."""
    y_t = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_p = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_t * y_p, axis=1)
    return 1.0 - tf.reduce_mean(
        (2.0 * intersection + 1.0) /
        (tf.reduce_sum(y_t, axis=1) + tf.reduce_sum(y_p, axis=1) + 1.0))

def soft_iou_loss(y_true, y_pred):
    """Differentiable Jaccard/IoU loss — penalises FPs more aggressively than Dice."""
    y_t = tf.reshape(tf.cast(y_true, tf.float32), [tf.shape(y_true)[0], -1])
    y_p = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_t * y_p, axis=1)
    union = tf.reduce_sum(y_t, axis=1) + tf.reduce_sum(y_p, axis=1) - intersection
    return 1.0 - tf.reduce_mean((intersection + 1.0) / (union + 1.0))

def focal_loss(y_true, y_pred, alpha=0.75, gamma=2.0):
    """Focal loss — down-weights easy background pixels, emphasises hard positives."""
    y_t = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_p = tf.reshape(tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7), [-1])
    bce  = -(y_t * tf.math.log(y_p) + (1 - y_t) * tf.math.log(1 - y_p))
    p_t  = y_t * y_p + (1 - y_t) * (1 - y_p)
    a_t  = y_t * alpha + (1 - y_t) * (1 - alpha)
    return tf.reduce_mean(a_t * tf.pow(1 - p_t, gamma) * bce)

def adaptive_roi_focal(y_true, y_pred, alpha=0.75, gamma=2.0,
                        roi_weight=3.0, boundary_weight=5.0):
    """Focal loss with adaptive ROI weighting — upweights hemorrhage region
    and boundary pixels so the model focuses on clinically relevant areas."""
    y_t = tf.cast(y_true, tf.float32)
    y_p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    # ── Derive spatial weight map from ground truth ──────────────────────
    # Dilate mask to define ROI neighbourhood
    y_4d = tf.reshape(y_t, [-1, tf.shape(y_t)[1], tf.shape(y_t)[2], 1])
    dilated = tf.nn.max_pool2d(y_4d, ksize=11, strides=1, padding='SAME')
    # Erode to find interior
    eroded  = -tf.nn.max_pool2d(-y_4d, ksize=5, strides=1, padding='SAME')
    # Boundary = dilated ring minus eroded interior
    boundary = dilated - eroded
    # Weight map: background=1.0, ROI region=roi_weight, boundary=boundary_weight
    w_map = 1.0 + (roi_weight - 1.0) * dilated + (boundary_weight - roi_weight) * boundary
    w_map = tf.reshape(w_map, tf.shape(y_t))

    # ── Per-pixel focal BCE ──────────────────────────────────────────────
    bce = -(y_t * tf.math.log(y_p) + (1.0 - y_t) * tf.math.log(1.0 - y_p))
    p_t = y_t * y_p + (1.0 - y_t) * (1.0 - y_p)
    a_t = y_t * alpha + (1.0 - y_t) * (1.0 - alpha)
    focal = a_t * tf.pow(1.0 - p_t, gamma) * bce

    return tf.reduce_mean(w_map * focal)

def focal_dice_loss(y_true, y_pred):
    """Focal (handles imbalance) + Dice (forces spatial overlap). No raw BCE."""
    return focal_loss(y_true, y_pred) + soft_dice_loss(y_true, y_pred)

def compound_seg_loss(y_true, y_pred):
    """Adaptive-ROI focal + Dice + IoU — boundary-aware, overlap-driven compound loss.
    - Adaptive ROI focal: pixel-level, spatially weighted toward hemorrhage regions
    - Dice: forces global spatial overlap
    - IoU: complements Dice — penalises false positives more aggressively
    """
    return (adaptive_roi_focal(y_true, y_pred)
            + soft_dice_loss(y_true, y_pred)
            + 0.5 * soft_iou_loss(y_true, y_pred))

class DiceMetric(keras.metrics.Metric):
    """Track mean dice per epoch for checkpointing (thresholded at 0.5)."""
    def __init__(self, name="dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", shape=(), initializer="zeros")
        self.count = self.add_weight(name="count", shape=(), initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_t = tf.cast(tf.reshape(y_true, [tf.shape(y_true)[0], -1]) > 0.5, tf.float32)
        y_p = tf.cast(tf.reshape(y_pred, [tf.shape(y_pred)[0], -1]) > 0.5, tf.float32)
        inter = tf.reduce_sum(y_t * y_p, axis=1)
        dice  = (2.0 * inter + 1e-6) / (
            tf.reduce_sum(y_t, axis=1) + tf.reduce_sum(y_p, axis=1) + 1e-6)
        self.total.assign_add(tf.reduce_sum(dice))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    def result(self):
        return self.total / (self.count + 1e-6)
    def reset_state(self):
        self.total.assign(0.0); self.count.assign(0.0)

def conv_block(x, f, name):
    x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu")(x)
    return x

def attention_gate(skip, gate, filters, name):
    """Attention gate (Oktay et al.) — suppresses irrelevant encoder features
    using coarser decoder features as the gating signal.
    skip:  encoder feature map (high-res, may contain noise)
    gate:  decoder feature map (low-res, semantically stronger)
    """
    w_skip = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                           name=f"{name}_ws")(skip)
    w_gate = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                           name=f"{name}_wg")(gate)
    psi = layers.Activation("relu")(layers.Add(name=f"{name}_add")([w_skip, w_gate]))
    psi = layers.Conv2D(1, 1, padding="same", activation="sigmoid",
                        name=f"{name}_psi")(psi)
    return layers.Multiply(name=f"{name}_mul")([skip, psi])

def build_unet():
    inp = keras.Input(shape=(SEG_SIZE, SEG_SIZE, 4), name="ct_4ch")

    # ── Encoder: DenseNet121 with RadImageNet, native 4ch ─────────────────────
    import glob
    rad_path = RADIMAGENET_WEIGHTS
    if not os.path.exists(rad_path):
        candidates = glob.glob('./weights/*[Dd]ense[Nn]et*121*.h5')
        if candidates:
            rad_path = candidates[0]

    print("  Building U-Net with 4-channel CT input...")
    enc = _load_densenet121_4ch(
        (SEG_SIZE, SEG_SIZE, 4),
        weights_path=rad_path if os.path.exists(rad_path) else None)

    enc_model = keras.Model(inputs=enc.input, outputs=[
        enc.get_layer("conv1_relu").output,
        enc.get_layer("pool2_bn").output,
        enc.get_layer("pool3_bn").output,
        enc.get_layer("pool4_bn").output,
        enc.get_layer("relu").output,
    ], name="densenet121_encoder")

    # Full encoder unfrozen — RadImageNet gives good init, but at 256px the model
    # needs to adapt all layers to learn thin hemorrhage boundaries.
    # Early conv1 block stays frozen (generic edge/texture filters are fine).
    for layer in enc_model.layers:
        layer.trainable = True
    for layer in enc_model.layers[:4]:   # just input + conv1_conv + conv1_bn
        layer.trainable = False

    s1, s2, s3, s4, bridge = enc_model(inp)

    # ── Decoder with attention gates + dropout ──────────────────────────
    def up(x, skip, f, name):
        x = layers.UpSampling2D(2, name=f"{name}_up")(x)
        # Attention gate: let decoder features suppress irrelevant encoder features
        skip = attention_gate(skip, x, f // 4, name=f"{name}_attn")
        x = layers.Concatenate(name=f"{name}_cat")([x, skip])
        x = conv_block(x, f, name)
        x = layers.SpatialDropout2D(0.15, name=f"{name}_drop")(x)
        return x

    d4  = up(bridge, s4, 256, "d4")
    d3  = up(d4,     s3, 128, "d3")
    d2  = up(d3,     s2,  64, "d2")
    d1  = up(d2,     s1,  32, "d1")
    d0  = conv_block(layers.UpSampling2D(2)(d1), 16, "d0")
    out = layers.Conv2D(1, 1, activation="sigmoid", name="seg_out")(d0)

    m = keras.Model(inputs=inp, outputs=out, name="AttentionUNet_DenseNet121")
    m.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss=compound_seg_loss, metrics=[DiceMetric()])
    return m

normal_col = display_labels.index('normal')
hem_idx    = np.where(all_y[:, normal_col] == 0)[0]

if len(hem_idx) > 100:
    # U-Net trains at full IMG_SIZE (SEG_SIZE=256) for spatial detail
    def _resize_batch(arr, sz):
        return tf.image.resize(arr, [sz, sz]).numpy()

    if not RETRAIN_SEGMENTATION:
        unet_path, unet_metric = _find_best_ckpt("unet_", "dice", mode="max")
        if unet_path is None:
            # Fall back to legacy filename
            unet_path = MODEL_DIR + "best_unet.keras" if os.path.exists(MODEL_DIR + "best_unet.keras") else None
        if unet_path:
            try:
                print(f"\nLoading saved U-Net: {os.path.basename(unet_path)}"
                      + (f"  (dice={unet_metric:.4f})" if unet_metric and unet_metric > 0 else ""))
                unet = keras.models.load_model(unet_path, compile=False)
                expected_ch = unet.input_shape[-1]
                actual_ch   = all_3ch.shape[-1]
                if expected_ch != actual_ch:
                    print(f"  ⚠ U-Net expects {expected_ch}ch but data is {actual_ch}ch — retraining.")
                    RETRAIN_SEGMENTATION = True
            except Exception as e:
                print(f"  ⚠ Failed to load U-Net: {e} — retraining.")
                RETRAIN_SEGMENTATION = True
        else:
            RETRAIN_SEGMENTATION = True

    if RETRAIN_SEGMENTATION:
        hem_train = idx_train[np.isin(idx_train, hem_idx)]
        hem_val   = idx_val[np.isin(idx_val, hem_idx)]   # val on hemorrhage only
        print(f"  Seg train: {len(hem_train)} hemorrhage images")
        print(f"  Seg val:   {len(hem_val)} hemorrhage images (normals excluded)")

        seg_tr_x  = _resize_batch(all_3ch[hem_train], SEG_SIZE)
        seg_tr_y  = _resize_batch(all_seg[hem_train], SEG_SIZE)
        seg_vl_x  = _resize_batch(all_3ch[hem_val],   SEG_SIZE)
        seg_vl_y  = _resize_batch(all_seg[hem_val],   SEG_SIZE)

        # ── Online augmentation via tf.data.map (no dataset blowup) ──────────
        @tf.function
        def seg_augment(img, mask):
            if tf.random.uniform(()) > 0.5:
                img  = tf.image.flip_left_right(img)
                mask = tf.image.flip_left_right(mask)
            if tf.random.uniform(()) > 0.5:
                img  = tf.image.flip_up_down(img)
                mask = tf.image.flip_up_down(mask)
            img = tf.image.random_brightness(img, 0.15)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, mask

        seg_train_ds = tf.data.Dataset.from_tensor_slices(
            (seg_tr_x, seg_tr_y)
        ).shuffle(len(seg_tr_x), seed=SEED).map(
            seg_augment, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(SEG_BATCH).prefetch(tf.data.AUTOTUNE)
        print(f"  Steps/epoch: {len(seg_tr_x) // SEG_BATCH} (online augmentation)")

        seg_val_ds = tf.data.Dataset.from_tensor_slices(
            (seg_vl_x, seg_vl_y)
        ).batch(SEG_BATCH).cache().prefetch(tf.data.AUTOTUNE)

        unet = build_unet()

        # Warmup for 3 epochs — ramp from 1e-6 to 1e-4, then let ReduceLROnPlateau manage
        def unet_warmup(epoch, lr, warmup=3, target_lr=1e-4):
            if epoch < warmup:
                return 1e-6 + (target_lr - 1e-6) * (epoch / warmup)
            return lr  # pass through — ReduceLROnPlateau handles decay

        unet.fit(
            seg_train_ds, validation_data=seg_val_ds, epochs=80,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    MODEL_DIR + f"unet_{RUN_TAG}_dice{{val_dice:.4f}}.keras",
                    monitor="val_dice", mode="max", save_best_only=True),
                keras.callbacks.EarlyStopping(
                    monitor="val_dice", mode="max",
                    patience=12, restore_best_weights=True),
                keras.callbacks.LearningRateScheduler(unet_warmup, verbose=0),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_dice", mode="max",
                    factor=0.5, patience=5, min_lr=1e-7),
            ]
        )
        del seg_tr_x, seg_tr_y, seg_vl_x, seg_vl_y  # free RAM

    # Evaluate — hemorrhage cases only
    hem_eval = idx_val[np.isin(idx_val, hem_idx)]
    seg_eval_x = _resize_batch(all_3ch[hem_eval], SEG_SIZE)
    seg_eval_y = _resize_batch(all_seg[hem_eval], SEG_SIZE)
    seg_val_ds = tf.data.Dataset.from_tensor_slices(
        (seg_eval_x, seg_eval_y)
    ).batch(SEG_BATCH).prefetch(tf.data.AUTOTUNE)

    scores = []
    for imgs, msks in seg_val_ds:
        preds = unet.predict(imgs, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        for gt, pr in zip(msks.numpy(), preds):
            scores.append(dice_coef(gt, pr))
    print(f"\nDice (hemorrhage-only val): {np.mean(scores):.4f}  (Ying et al. baseline: 0.59)")
    del seg_eval_x, seg_eval_y
else:
    print("Not enough hemorrhage samples for segmentation.")
