"""
clip_hints.py
=============
Generates CLIP image embeddings for all CT images.
Replaces LLaVA — runs in ~15 minutes instead of 36 hours.

CLIP encodes the visual content of each image into a 512-d embedding
that captures semantic structure without any text generation.

Install:
    pip install transformers torch torchvision

Usage:
    python3 clip_hints.py
    → saves ./output/clip_embeddings.npz
      keys: 'stems' (N,) string array, 'embeddings' (N, 512) float32
"""
import os, cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = './dcms/'
cnt_dir     = ROOT_DIR + '02_Contour/'
render_dir  = ROOT_DIR + 'renders/'
OUT_FILE    = './output/clip_embeddings.npz'
BATCH_SIZE  = 32   # process images in batches for speed
BRAIN_WIN   = '/brain_window/'

CONTOUR_CLASSES = ['epidural', 'intraparenchymal', 'subarachnoid', 'multiple', 'subdural']
RENDER_CLASSES  = ['normal', 'intraventricular', 'epidural', 'subarachnoid']

# ── Load CLIP ─────────────────────────────────────────────────────────────────
print("Loading CLIP model (openai/clip-vit-base-patch32)...")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device    = "mps" if torch.backends.mps.is_available() else "cpu"
model     = model.to(device)
model.eval()
print(f"CLIP on {device}")

# ── Collect image paths ───────────────────────────────────────────────────────
all_paths = []

for cls in CONTOUR_CLASSES:
    d = cnt_dir + cls + '/'
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f != '.DS_Store':
            all_paths.append(Path(d + f))

for cls in RENDER_CLASSES:
    d = render_dir + cls + BRAIN_WIN
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f != '.DS_Store':
            all_paths.append(Path(d + f))

print(f"Images to embed: {len(all_paths)}")

# ── Check for existing partial results ───────────────────────────────────────
done_stems = set()
if os.path.exists(OUT_FILE):
    existing = np.load(OUT_FILE, allow_pickle=True)
    done_stems = set(existing['stems'].tolist())
    print(f"Already embedded: {len(done_stems)} — resuming...")

pending = [p for p in all_paths if p.stem not in done_stems]
print(f"Remaining: {len(pending)}")

# ── Embed in batches ──────────────────────────────────────────────────────────
stems      = list(done_stems)
embeddings = []

if done_stems:
    embeddings = list(existing['embeddings'])

def load_pil(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return Image.fromarray(img)

batch_imgs, batch_stems = [], []

def flush_batch():
    global batch_imgs, batch_stems
    if not batch_imgs:
        return
    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats, dim=-1)
    embeddings.extend(feats.cpu().numpy())
    stems.extend(batch_stems)
    batch_imgs, batch_stems = [], []

for path in tqdm(pending, unit='image'):
    pil = load_pil(path)
    if pil is None:
        continue
    batch_imgs.append(pil)
    batch_stems.append(path.stem)
    if len(batch_imgs) >= BATCH_SIZE:
        flush_batch()

flush_batch()  # remainder

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez_compressed(
    OUT_FILE,
    stems=np.array(stems),
    embeddings=np.array(embeddings, dtype=np.float32)
)
print(f"\nSaved {len(stems)} embeddings → {OUT_FILE}")
print(f"Embedding shape: {np.array(embeddings).shape}")
print(f"\nNext: set USE_CLIP=True in pret.py to use these as the caption branch input.")