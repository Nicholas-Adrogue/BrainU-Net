#!/usr/bin/env python3
"""
visualize_hints.py — Show all hint channels + radiologist masks + U-Net predictions.

For each sampled contour image, displays:
  1. Brain window (channel 1 of the 4ch CT input)
  2. Radiologist mask (ground truth from annotation CSVs)
  3. U-Net prediction (if model found — live inference at SEG_SIZE)
  4. Bbox hint (bright-region bounding boxes, Gaussian-blurred)
  5. Arrow hint (pre-saved U-Net predictions from .npy files)
  6. GT vs Pred overlay (cyan = GT, red = predicted, white = overlap)
  7. Full composite (all layers on brain window)

Usage:
  python visualize_hints.py                              # 3 per class, saves PNG
  python visualize_hints.py --per-class 5                # 5 per class
  python visualize_hints.py --classes epidural subdural   # specific classes
  python visualize_hints.py --no-unet                    # skip U-Net inference
  python visualize_hints.py --show                       # interactive display
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, ast, argparse, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ── TF (CPU only, matching pret.py) ──────────────────────────────────────────
import tensorflow as tf
try:
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    pass

from tensorflow import keras

# ── Paths (mirrored from pret.py) ────────────────────────────────────────────
ROOT_DIR    = './dcms/'
seg_dir     = ROOT_DIR + 'segmentation/'
render_dir  = ROOT_DIR + 'renders/'
cnt_dir     = ROOT_DIR + '02_Contour/'

LABEL_FILE  = 'hemorrhage-labels'
EXT_CSV     = '.csv'

CSV_MAP = {
    'epidural':         'Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148',
    'intraparenchymal': 'Results_Intraparenchymal Hemorrhage Detection_2020-11-16_21.39.31.268',
    'subarachnoid':     'Results_Subarachnoid Hemorrhage Detection_2020-11-16_21.36.18.668',
    'multiple':         'Results_Multiple Hemorrhage Detection_2020-11-16_21.36.24.018',
    'subdural_a':       'Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040',
    'subdural_b':       'Results_Subdural Hemorrhage Detection_2020-11-16_21.37.19.745',
}

BONE_WIN     = '/brain_bone_window/'
BRAIN_WIN    = '/brain_window/'
CONTRAST_WIN = '/max_contrast_window/'
SUB_WIN      = '/subdural_window/'

ARROW_DIR   = './output/arrow_hints/'
MODEL_DIR   = './output/models/'
FIG_DIR     = './output/figures/'
os.makedirs(FIG_DIR, exist_ok=True)

IMG_SIZE = 256
SEG_SIZE = 256  # must match pret.py

CONTOUR_CLASSES = ['epidural', 'intraparenchymal', 'subarachnoid', 'subdural']

# ── Flagged / corrupt exclusion ──────────────────────────────────────────────
FLAGGED_FILE = './flagged.txt'
CORRUPT_IDS  = {'ID_6431af929'}
_flagged_stems = set(CORRUPT_IDS)
if os.path.exists(FLAGGED_FILE):
    with open(FLAGGED_FILE) as f:
        for line in f:
            s = line.strip()
            if s:
                _flagged_stems.add(s)

def is_flagged(name):
    return Path(name).stem in _flagged_stems


# ── Annotation parsing (from pret.py — kept standalone) ──────────────────────
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


# ── Hint helpers ─────────────────────────────────────────────────────────────
def stack_window_channels(class_dir, fname):
    results = []
    for win in [BONE_WIN, BRAIN_WIN, CONTRAST_WIN, SUB_WIN]:
        img = cv2.imread(class_dir + win + fname, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            results.append(img)
    return results

def symmetry_hint(img_4ch):
    """Bilateral symmetry deviation — highlights asymmetric regions in the brain."""
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

def get_arrow_hint(img_name):
    stem = Path(img_name).stem
    p    = ARROW_DIR + stem + '.npy'
    if os.path.exists(p):
        return np.load(p).astype(np.float32)
    return np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)


# ── U-Net loading ────────────────────────────────────────────────────────────
def find_best_unet():
    """Find the best U-Net checkpoint (tagged or legacy)."""
    import glob as _glob, re
    pattern = MODEL_DIR + "unet_*dice*.keras"
    hits = _glob.glob(pattern)
    if hits:
        def _parse(p):
            m = re.search(r'dice(\d+\.\d+)', os.path.basename(p))
            return float(m.group(1)) if m else -1.0
        best = max(hits, key=_parse)
        return best, _parse(best)
    legacy = MODEL_DIR + "best_unet.keras"
    if os.path.exists(legacy):
        return legacy, None
    return None, None

def load_unet():
    path, dice = find_best_unet()
    if path is None:
        print("  No U-Net checkpoint found — skipping live predictions.")
        return None
    tag = f"  (dice={dice:.4f})" if dice and dice > 0 else ""
    print(f"  Loading U-Net: {os.path.basename(path)}{tag}")
    try:
        model = keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        print(f"  ⚠ Failed to load U-Net: {e}")
        return None

def predict_seg(unet, img_4ch):
    """Run U-Net on a single 4ch image, return (H, W, 1) prediction at IMG_SIZE."""
    inp = tf.image.resize(img_4ch, [SEG_SIZE, SEG_SIZE])[np.newaxis]
    pred = unet.predict(inp, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = pred[0]
    if pred.shape[0] != IMG_SIZE or pred.shape[1] != IMG_SIZE:
        pred = tf.image.resize(pred, [IMG_SIZE, IMG_SIZE]).numpy()
    return pred.astype(np.float32)


# ── Annotation CSV loading ───────────────────────────────────────────────────
def load_annotation_df(cls_name):
    if cls_name == 'subdural':
        return pd.concat([
            pd.read_csv(seg_dir + CSV_MAP['subdural_a'] + EXT_CSV),
            pd.read_csv(seg_dir + CSV_MAP['subdural_b'] + EXT_CSV)
        ], ignore_index=True)[['Origin', 'Majority Label', 'Correct Label', 'All Labels']]
    elif cls_name in CSV_MAP:
        return pd.read_csv(seg_dir + CSV_MAP[cls_name] + EXT_CSV)[
            ['Origin', 'Majority Label', 'Correct Label', 'All Labels']]
    return None


# ── Sample loading ───────────────────────────────────────────────────────────
def load_contour_samples(cls_name, n=3, unet=None):
    """Load contour images with radiologist masks, hints, and optional U-Net preds."""
    cls_img_dir = cnt_dir + cls_name + '/'
    if not os.path.isdir(cls_img_dir):
        print(f"  ⚠ Contour dir not found: {cls_img_dir}")
        return []

    render_cls_dir = render_dir + ('subdural/subdural' if cls_name == 'subdural' else cls_name)
    ann_df = load_annotation_df(cls_name)
    if ann_df is None:
        print(f"  ⚠ No annotation CSV for {cls_name}")
        return []

    files = [f for f in os.listdir(cls_img_dir) if f != '.DS_Store' and not is_flagged(f)]
    np.random.shuffle(files)

    samples = []
    for fname in files:
        if len(samples) >= n:
            break

        scan = cv2.imread(cls_img_dir + fname)
        if scan is None:
            continue

        # Parse radiologist annotations → ground truth mask
        row_matches = ann_df[ann_df.Origin == fname]
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

        if not combined_regions:
            continue

        _, seg_mask = draw_annotation_mask(combined_regions, scan)

        # Resize
        scan_resized = tf.image.resize(scan, [IMG_SIZE, IMG_SIZE]).numpy()
        mask_resized = tf.image.resize(seg_mask, [IMG_SIZE, IMG_SIZE]).numpy()
        scan_norm    = scan_resized / 255.0
        gt_gray      = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY).reshape(IMG_SIZE, IMG_SIZE, 1)
        gt_gray      = (gt_gray / 255.0).astype(np.float32)
        brain_gray   = cv2.cvtColor(scan_norm, cv2.COLOR_BGR2GRAY).reshape(IMG_SIZE, IMG_SIZE, 1)

        # 4ch render windows
        channels = stack_window_channels(render_cls_dir, fname)
        if len(channels) < 4:
            gray_u8 = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
            gray_r  = cv2.resize(gray_u8, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
            img_4ch = np.stack([gray_r] * 4, axis=-1)
        else:
            img_4ch = np.stack([cv2.resize(c, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                                for c in channels[:4]], axis=-1)

        symm  = symmetry_hint(img_4ch)
        arrow = get_arrow_hint(fname)
        unet_pred = predict_seg(unet, img_4ch) if unet is not None else None

        samples.append({
            'fname':     fname,
            'cls':       cls_name,
            'brain':     img_4ch[:, :, 1],
            'bone':      img_4ch[:, :, 0],
            'gt_mask':   gt_gray[:, :, 0],
            'unet_pred': unet_pred[:, :, 0] if unet_pred is not None else None,
            'symm':      symm[:, :, 0],
            'arrow':     arrow[:, :, 0],
        })

    return samples


# ── Plotting ─────────────────────────────────────────────────────────────────
def compute_dice(gt, pred, thresh=0.5):
    g = (gt > thresh).astype(np.float32).flatten()
    p = (pred > thresh).astype(np.float32).flatten()
    inter = (g * p).sum()
    return float((2 * inter + 1e-6) / (g.sum() + p.sum() + 1e-6))

def plot_all(samples_by_class, has_unet=True, out_path=None, show=False):
    all_samples = []
    for cls_name, samples in samples_by_class:
        for s in samples:
            all_samples.append(s)

    n_rows = len(all_samples)
    if n_rows == 0:
        print("No samples to plot.")
        return

    n_cols = 7 if has_unet else 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    if has_unet:
        col_titles = [
            'Brain Window',
            'Radiologist Mask\n(Ground Truth)',
            'U-Net Prediction',
            'Symmetry Hint',
            'Arrow Hint\n(saved .npy)',
            'GT vs Pred\ncyan=GT  red=pred',
            'Full Composite',
        ]
    else:
        col_titles = [
            'Brain Window',
            'Radiologist Mask\n(Ground Truth)',
            'Symmetry Hint',
            'Arrow Hint\n(saved .npy)',
            'GT Overlay',
            'Full Composite',
        ]

    for row, s in enumerate(all_samples):
        brain    = s['brain']
        gt       = s['gt_mask']
        unet_p   = s['unet_pred']
        symm_h   = s['symm']
        arrow_h  = s['arrow']
        has_arrow = arrow_h.max() > 0.01

        # GT vs Pred comparison
        if unet_p is not None:
            dice = compute_dice(gt, unet_p)
            # cyan = GT only, red = pred only, white = overlap
            overlap = np.minimum(gt, (unet_p > 0.5).astype(np.float32))
            gt_only = gt - overlap
            pred_only = (unet_p > 0.5).astype(np.float32) - overlap
            gt_vs_pred = np.stack([
                np.clip(brain + pred_only * 0.8 + overlap * 0.7, 0, 1),   # R
                np.clip(brain + gt_only * 0.8 + overlap * 0.7, 0, 1),     # G
                np.clip(brain + gt_only * 0.8 + overlap * 0.7, 0, 1),     # B
            ], axis=-1)
        else:
            dice = None
            gt_vs_pred = np.stack([
                brain,
                np.clip(brain + gt * 0.5, 0, 1),
                np.clip(brain + gt * 0.5, 0, 1),
            ], axis=-1)

        # Full composite: GT=cyan, pred=red, symm=green, arrow=magenta
        comp = np.stack([brain] * 3, axis=-1).copy()
        comp[:, :, 1] = np.clip(comp[:, :, 1] + gt * 0.4, 0, 1)
        comp[:, :, 2] = np.clip(comp[:, :, 2] + gt * 0.4, 0, 1)
        comp[:, :, 1] = np.clip(comp[:, :, 1] + symm_h * 0.4, 0, 1)  # symmetry in green
        comp[:, :, 0] = np.clip(comp[:, :, 0] + arrow_h * 0.4, 0, 1)
        comp[:, :, 2] = np.clip(comp[:, :, 2] + arrow_h * 0.3, 0, 1)
        if unet_p is not None:
            comp[:, :, 0] = np.clip(comp[:, :, 0] + unet_p * 0.5, 0, 1)

        # Assemble columns
        if has_unet:
            panels = [
                (brain,      'gray',    f"{s['cls']}\n{s['fname'][:22]}"),
                (gt,         'inferno', f"area={gt.mean():.4f}"),
                (unet_p,     'inferno', f"dice={dice:.3f}" if dice else ""),
                (symm_h,     'inferno', f"max={symm_h.max():.2f}"),
                (arrow_h,    'inferno', f"max={arrow_h.max():.2f}" + ("" if has_arrow else " (empty)")),
                (gt_vs_pred, None,      f"dice={dice:.3f}  white=overlap" if dice else ""),
                (comp,       None,      "cyan=GT grn=symm mag=arrow red=pred"),
            ]
        else:
            panels = [
                (brain,      'gray',    f"{s['cls']}\n{s['fname'][:22]}"),
                (gt,         'inferno', f"area={gt.mean():.4f}"),
                (symm_h,     'inferno', f"max={symm_h.max():.2f}"),
                (arrow_h,    'inferno', f"max={arrow_h.max():.2f}" + ("" if has_arrow else " (empty)")),
                (gt_vs_pred, None,      "cyan=GT"),
                (comp,       None,      "cyan=GT grn=symm mag=arrow"),
            ]

        for col, (img, cmap, subtitle) in enumerate(panels):
            ax = axes[row, col]
            if img is None:
                ax.set_visible(False)
                continue
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight='bold')
            if subtitle:
                ax.set_xlabel(subtitle, fontsize=7)

    plt.suptitle('Hint & Segmentation Visualization — Contour Images',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {out_path}")
    if show:
        plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Visualize hints + segmentation masks')
    parser.add_argument('--per-class', type=int, default=3,
                        help='Images per class (default: 3)')
    parser.add_argument('--classes', nargs='+', default=CONTOUR_CLASSES,
                        help=f'Classes to show (default: {CONTOUR_CLASSES})')
    parser.add_argument('--no-unet', action='store_true',
                        help='Skip U-Net loading and live predictions')
    parser.add_argument('--show', action='store_true',
                        help='Interactive display instead of saving')
    parser.add_argument('--out', type=str, default=None,
                        help='Output path (default: auto)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load U-Net
    unet = None
    if not args.no_unet:
        print("Looking for U-Net checkpoint...")
        unet = load_unet()

    # Load samples
    samples_by_class = []
    for cls_name in args.classes:
        if cls_name in ('normal', 'intraventricular', 'multiple'):
            print(f"  Skipping {cls_name} — no contour annotations for this class")
            continue
        print(f"Loading {args.per_class} contour samples for {cls_name}...")
        samples = load_contour_samples(cls_name, n=args.per_class, unet=unet)
        if samples:
            n_arrow = sum(1 for s in samples if s['arrow'].max() > 0.01)
            n_pred  = sum(1 for s in samples if s['unet_pred'] is not None)
            print(f"  Got {len(samples)} images ({n_arrow} with arrow hints, {n_pred} with U-Net pred)")
            samples_by_class.append((cls_name, samples))
        else:
            print(f"  No contour images with annotations found")

    out_path = args.out or (FIG_DIR + 'hint_seg_visualization.png')
    plot_all(samples_by_class, has_unet=(unet is not None),
             out_path=None if args.show else out_path, show=args.show)

    # Summary stats
    n_arrows = sum(1 for f in os.listdir(ARROW_DIR) if f.endswith('.npy')) if os.path.isdir(ARROW_DIR) else 0
    print(f"Arrow hint files in {ARROW_DIR}: {n_arrows}")

    if unet is not None:
        dices = [compute_dice(s['gt_mask'], s['unet_pred'])
                 for _, samps in samples_by_class for s in samps
                 if s['unet_pred'] is not None]
        if dices:
            print(f"Mean Dice on {len(dices)} sampled images: {np.mean(dices):.4f}")


if __name__ == '__main__':
    main()