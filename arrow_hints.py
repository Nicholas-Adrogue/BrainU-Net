"""
arrow_hints.py
==============
Generates arrow hint maps for every CT image using the trained U-Net.

Pipeline:
  CT image → EfficientNetB0 U-Net → predicted mask → contour centroids → arrow map

The arrow map is a (256,256,1) float32 image where arrows point FROM the
image periphery TOWARD each predicted hemorrhage centroid.

Output:
  ./output/arrow_hints/<stem>.npy       — one per image
  ./output/arrow_viz/<stem>_viz.png     — visualization (optional)

Usage:
    python3 arrow_hints.py
"""
import os, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = './dcms/'
cnt_dir     = ROOT_DIR + '02_Contour/'
render_dir  = ROOT_DIR + 'renders/'
MODEL_DIR   = './output/models/'
ARROW_DIR   = './output/arrow_hints/';  os.makedirs(ARROW_DIR, exist_ok=True)
VIZ_DIR     = './output/arrow_viz/';   os.makedirs(VIZ_DIR,   exist_ok=True)

CONTOUR_CLASSES = ['epidural', 'intraparenchymal', 'subarachnoid', 'multiple', 'subdural']
RENDER_CLASSES  = ['normal', 'intraventricular', 'epidural', 'subarachnoid']
BRAIN_WIN       = '/brain_bone_window/'
BRAIN_WIN_2     = '/brain_window/'
CONTRAST_WIN    = '/max_contrast_window/'
SUB_WIN         = '/subdural_window/'

IMG_SIZE        = 256       # output arrow map size (matches pret.py classifier input)
MASK_THRESHOLD  = 0.5
MIN_REGION_AREA = 30        # at SEG_SIZE=128, regions are ~4x smaller in pixels
SAVE_VIZ        = True      # set False to skip PNG visualizations
FORCE_REGEN     = True      # regenerate ALL hints (not just missing ones)

# ── Load U-Net ────────────────────────────────────────────────────────────────
print("Loading U-Net...")
unet = keras.models.load_model(MODEL_DIR + "best_unet.keras", compile=False)
SEG_SIZE = unet.input_shape[1]  # auto-detect from model (e.g. 128)
print(f"U-Net loaded — input {unet.input_shape}, predicting at {SEG_SIZE}×{SEG_SIZE}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def stack_window_channels(class_dir, fname):
    """Load up to 4 CT window renders, skip missing ones."""
    results = []
    for win in [BRAIN_WIN, BRAIN_WIN_2, CONTRAST_WIN, SUB_WIN]:
        img = cv2.imread(class_dir + win + fname, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            results.append(img)
    return results


def load_3ch(img_path):
    """
    Load a 4-channel CT image (bone, brain, contrast, subdural) for the given path.
    Tries render directories first, falls back to the contour image itself.
    """
    path   = Path(img_path)
    fname  = path.name

    # Try render dirs
    for cls in CONTOUR_CLASSES + RENDER_CLASSES:
        if cls not in str(path):
            continue
        rdir = render_dir + ('subdural/subdural' if cls == 'subdural' else cls)
        channels = stack_window_channels(rdir, fname)
        if len(channels) >= 4:
            return np.stack([
                cv2.resize(c, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
                for c in channels[:4]
            ], axis=-1)

    # Fallback: read the image directly and replicate to 4ch
    raw = cv2.imread(str(img_path))
    if raw is not None:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        return np.stack([gray, gray, gray, gray], axis=-1)

    return None


def predict_mask(img_3ch):
    """Run U-Net on a (H,W,3) image, return (IMG_SIZE, IMG_SIZE) uint8 binary mask."""
    # Resize to U-Net's expected input size
    if img_3ch.shape[0] != SEG_SIZE:
        inp = cv2.resize(img_3ch, (SEG_SIZE, SEG_SIZE))[None].astype(np.float32)
    else:
        inp = img_3ch[None].astype(np.float32)
    out = unet.predict(inp, verbose=0)
    if isinstance(out, (list, tuple)):
        out = out[0]
    pred = out[0]
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    # Resize prediction back to IMG_SIZE for arrow drawing
    if pred.shape[0] != IMG_SIZE:
        pred = cv2.resize(pred, (IMG_SIZE, IMG_SIZE),
                          interpolation=cv2.INTER_LINEAR)
    return (pred >= MASK_THRESHOLD).astype(np.uint8)


def find_hemorrhage_regions(binary_mask):
    """
    Find contours of predicted hemorrhage regions.
    Returns list of (centroid_x, centroid_y, contour, area).
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_REGION_AREA:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        regions.append((cx, cy, cnt, area))
    return regions


def arrow_origin(cx, cy, img_size=IMG_SIZE, margin=20):
    """
    Compute arrow start point at the nearest image edge toward the centroid.
    """
    mid = img_size // 2
    dx  = cx - mid
    dy  = cy - mid
    if abs(dx) >= abs(dy):
        sx = margin if dx >= 0 else img_size - margin
        sy = cy
    else:
        sx = cx
        sy = margin if dy >= 0 else img_size - margin
    return sx, sy


def draw_arrow_map(regions, img_size=IMG_SIZE):
    """
    Draw soft arrows on a blank canvas pointing to each hemorrhage centroid.
    Returns (H,W,1) float32 map in [0,1].
    """
    canvas = np.zeros((img_size, img_size), dtype=np.float32)
    for cx, cy, cnt, area in regions:
        sx, sy = arrow_origin(cx, cy, img_size)
        cv2.arrowedLine(canvas, (sx, sy), (cx, cy),
                        color=1.0, thickness=2, tipLength=0.25)
        radius = max(5, int(np.sqrt(area / np.pi) * 0.3))
        cv2.circle(canvas, (cx, cy), radius, 1.0, thickness=2)
    canvas = cv2.GaussianBlur(canvas, (11, 11), 0)
    mx = canvas.max()
    if mx > 0:
        canvas /= mx
    return canvas.reshape(img_size, img_size, 1).astype(np.float32)


def save_visualization(img_3ch, binary_mask, arrow_map, regions, out_path):
    """4-panel PNG: CT | predicted mask | arrow map | overlay."""
    ct    = (img_3ch[:, :, 0] * 255).astype(np.uint8)
    mask  = (binary_mask * 255).astype(np.uint8)
    arrow = (arrow_map[:, :, 0] * 255).astype(np.uint8)

    overlay = cv2.cvtColor(ct, cv2.COLOR_GRAY2BGR)
    for cx, cy, cnt, _ in regions:
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), 1)
        sx, sy = arrow_origin(cx, cy)
        cv2.arrowedLine(overlay, (sx, sy), (cx, cy),
                        (0, 0, 255), 2, tipLength=0.25)
        cv2.circle(overlay, (cx, cy), 5, (255, 0, 0), -1)

    panel = np.hstack([
        cv2.cvtColor(ct,    cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask,  cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(arrow, cv2.COLOR_GRAY2BGR),
        overlay
    ])
    cv2.imwrite(out_path, panel)


# ── Collect all image paths ───────────────────────────────────────────────────
all_paths = []

for cls in CONTOUR_CLASSES:
    d = cnt_dir + cls + '/'
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f != '.DS_Store':
            all_paths.append(Path(d + f))

for cls in RENDER_CLASSES:
    d = render_dir + cls + BRAIN_WIN_2
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f != '.DS_Store':
            all_paths.append(Path(d + f))

if FORCE_REGEN:
    pending = all_paths
else:
    pending = [p for p in all_paths
               if not os.path.exists(ARROW_DIR + p.stem + '.npy')]

print(f"Total images : {len(all_paths)}")
print(f"Already done : {len(all_paths) - len(pending)}")
print(f"To process   : {len(pending)}\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
n_with_regions = 0
n_empty        = 0

for img_path in tqdm(pending, unit='image'):
    img_3ch = load_3ch(img_path)
    if img_3ch is None:
        continue

    binary_mask = predict_mask(img_3ch)
    regions     = find_hemorrhage_regions(binary_mask)
    arrow_map   = draw_arrow_map(regions)

    np.save(ARROW_DIR + img_path.stem + '.npy', arrow_map)

    if regions:
        n_with_regions += 1
        if SAVE_VIZ:
            save_visualization(img_3ch, binary_mask, arrow_map, regions,
                               VIZ_DIR + img_path.stem + '_viz.png')
    else:
        n_empty += 1

print(f"\nDone.")
print(f"  Images with predicted hemorrhage : {n_with_regions}")
print(f"  Images with no prediction (empty): {n_empty}")
print(f"  Arrow maps saved to: {ARROW_DIR}")
if SAVE_VIZ:
    print(f"  Visualizations saved to: {VIZ_DIR}")
print(f"\nNext: run pret.py — arrow hints will be loaded automatically.")