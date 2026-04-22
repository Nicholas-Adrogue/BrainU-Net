#!/usr/bin/env python3
"""
Test Set Evaluation
===================
Loads the trained model from great_results/ and runs a full evaluation
on the held-out test set. Saves:

    great_results/
    ├── test_results.txt         ← printed metrics summary
    ├── test_metrics.csv         ← loss, accuracy, MeanIoU
    ├── per_sample_iou.csv       ← IoU for every individual test image
    └── prediction_samples/      ← 4-panel overlay images (regenerated)

Usage:
    cd /Users/Nico/Desktop/MATH7243_ML1/Project
    python evaluate_test.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from hemorrhage_segmentation_unet import (
    step1_build_arrays, build_unet, MyMeanIOU,
    create_mask, show_predictions, dice_bce_loss,
    IMG_SIZE, CHANNELS, OUTPUT_CLASSES, BATCH_SIZE,
    RANDOM_SEED, RESULTS_DIR,
)

# Set seeds
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def compute_per_sample_iou(model, images, masks_q):
    """Compute IoU for each individual test image."""
    results = []
    for i in range(len(images)):
        img = images[i:i+1]
        true = masks_q[i].squeeze()

        pred_logits = model.predict(img, verbose=0)
        pred = tf.argmax(pred_logits, axis=-1).numpy().squeeze()

        # IoU for hemorrhage class (class 1)
        intersection = np.sum((pred == 1) & (true == 1))
        union = np.sum((pred == 1) | (true == 1))
        iou_fg = intersection / union if union > 0 else float('nan')

        # IoU for background class (class 0)
        intersection_bg = np.sum((pred == 0) & (true == 0))
        union_bg = np.sum((pred == 0) | (true == 0))
        iou_bg = intersection_bg / union_bg if union_bg > 0 else float('nan')

        # Mean IoU
        if np.isnan(iou_fg):
            mean_iou = iou_bg
        else:
            mean_iou = (iou_bg + iou_fg) / 2.0

        # Pixel counts
        true_pos = np.sum((pred == 1) & (true == 1))
        false_pos = np.sum((pred == 1) & (true == 0))
        false_neg = np.sum((pred == 0) & (true == 1))
        true_neg = np.sum((pred == 0) & (true == 0))
        has_hemorrhage = np.any(true == 1)

        results.append({
            'sample_idx': i,
            'has_hemorrhage': has_hemorrhage,
            'iou_hemorrhage': round(iou_fg, 4) if not np.isnan(iou_fg) else 'N/A',
            'iou_background': round(iou_bg, 4),
            'mean_iou': round(mean_iou, 4),
            'true_positives': int(true_pos),
            'false_positives': int(false_pos),
            'false_negatives': int(false_neg),
            'true_negatives': int(true_neg),
            'sensitivity': round(true_pos / (true_pos + false_neg), 4) if (true_pos + false_neg) > 0 else 'N/A',
            'specificity': round(true_neg / (true_neg + false_pos), 4) if (true_neg + false_pos) > 0 else 'N/A',
            'precision': round(true_pos / (true_pos + false_pos), 4) if (true_pos + false_pos) > 0 else 'N/A',
            'recall': round(true_pos / (true_pos + false_neg), 4) if (true_pos + false_neg) > 0 else 'N/A',
            'dice': round(2 * true_pos / (2 * true_pos + false_pos + false_neg), 4) if (2 * true_pos + false_pos + false_neg) > 0 else 'N/A',
        })
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data & rebuild test split ────────────────────────────────────────
    print('Loading data...')
    scans, masks, labels, display_labels = step1_build_arrays()

    X_tr, X_test, M_tr, M_test, y_tr, y_test = train_test_split(
        scans, masks, labels, test_size=0.2, random_state=RANDOM_SEED
    )

    def _quantise(m):
        return (m > 127).astype(np.int32)

    M_test_q = _quantise(M_test)

    ds_test = tf.data.Dataset.from_tensor_slices(
        (X_test.astype(np.float32), M_test_q)
    ).batch(BATCH_SIZE)

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(RESULTS_DIR, 'hemorrhage_unet.h5')
    if not os.path.exists(checkpoint_path):
        print(f'✗ Model not found: {checkpoint_path}')
        print('  Run hemorrhage_segmentation_unet.py first to train a model.')
        return

    input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)
    keras.backend.clear_session()
    model = build_unet(input_shape, OUTPUT_CLASSES)
    model.load_weights(checkpoint_path)
    print(f'✓ Loaded model from {checkpoint_path}')

    miou = MyMeanIOU(num_classes=OUTPUT_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_bce_loss,
        metrics=['accuracy', miou],
    )

    # ── Overall test metrics ──────────────────────────────────────────────────
    print('\n══ Test Set Evaluation ══')
    results = model.evaluate(ds_test, return_dict=True)

    test_loss = results.get('loss', float('nan'))
    test_acc = results.get('accuracy', float('nan'))
    miou_key = [k for k in results if 'mean_iou' in k][0] if any('mean_iou' in k for k in results) else None
    test_miou = results.get(miou_key, float('nan')) if miou_key else float('nan')

    # ── Per-sample IoU ────────────────────────────────────────────────────────
    print('\nComputing per-sample metrics...')
    per_sample = compute_per_sample_iou(model, X_test.astype(np.float32), M_test_q)
    per_sample_df = pd.DataFrame(per_sample)
    per_sample_path = os.path.join(RESULTS_DIR, 'per_sample_iou.csv')
    per_sample_df.to_csv(per_sample_path, index=False)
    print(f'  Saved {len(per_sample)} samples to {per_sample_path}')

    # ── Summary statistics ────────────────────────────────────────────────────
    hemorrhage_samples = per_sample_df[per_sample_df['has_hemorrhage'] == True]
    normal_samples = per_sample_df[per_sample_df['has_hemorrhage'] == False]

    # Filter numeric IoU values for hemorrhage samples
    hem_ious = hemorrhage_samples['iou_hemorrhage']
    hem_ious_numeric = pd.to_numeric(hem_ious, errors='coerce').dropna()

    hem_precisions = pd.to_numeric(hemorrhage_samples['precision'], errors='coerce').dropna()
    hem_recalls = pd.to_numeric(hemorrhage_samples['recall'], errors='coerce').dropna()

    # ── Aggregate pixel-level sensitivity & specificity across ALL test images ─
    total_tp = per_sample_df['true_positives'].astype(int).sum()
    total_fp = per_sample_df['false_positives'].astype(int).sum()
    total_fn = per_sample_df['false_negatives'].astype(int).sum()
    total_tn = per_sample_df['true_negatives'].astype(int).sum()

    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else float('nan')
    precision   = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
    f1_score    = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else float('nan')
    dice_coeff  = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else float('nan')

    # Update the metrics CSV with sensitivity/specificity
    metrics_df = pd.DataFrame([{
        'test_loss': round(test_loss, 6),
        'test_accuracy': round(test_acc, 6),
        'test_mean_iou': round(test_miou, 6),
        'test_sensitivity': round(sensitivity, 6),
        'test_specificity': round(specificity, 6),
        'test_precision': round(precision, 6),
        'test_f1_score': round(f1_score, 6),
        'test_dice': round(dice_coeff, 6),
        'total_tp': int(total_tp),
        'total_fp': int(total_fp),
        'total_fn': int(total_fn),
        'total_tn': int(total_tn),
        'test_samples': len(X_test),
    }])
    metrics_path = os.path.join(RESULTS_DIR, 'test_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)

    summary_lines = [
        '══════════════════════════════════════════════════════',
        '  TEST SET RESULTS',
        '══════════════════════════════════════════════════════',
        '',
        f'  Total test samples:     {len(X_test)}',
        f'  With hemorrhage:        {len(hemorrhage_samples)}',
        f'  Without hemorrhage:     {len(normal_samples)}',
        '',
        '── Overall metrics ──',
        f'  Loss:                   {test_loss:.4f}',
        f'  Pixel accuracy:         {test_acc:.4f}  ({test_acc*100:.2f}%)',
        f'  Mean IoU:               {test_miou:.4f}',
        '',
        '── Pixel-level metrics (aggregated across all test images) ──',
        f'  Sensitivity (recall):   {sensitivity:.4f}  (TP / (TP + FN) — how much hemorrhage was found)',
        f'  Specificity:            {specificity:.4f}  (TN / (TN + FP) — how much background was correct)',
        f'  Precision:              {precision:.4f}  (TP / (TP + FP) — how much of predicted is real)',
        f'  F1 / Dice:              {f1_score:.4f}  (harmonic mean of precision and sensitivity)',
        f'  True positives:         {total_tp:,}',
        f'  False positives:        {total_fp:,}',
        f'  False negatives:        {total_fn:,}',
        f'  True negatives:         {total_tn:,}',
        '',
        '── Hemorrhage samples only ──',
        f'  Mean hemorrhage IoU:    {hem_ious_numeric.mean():.4f}' if len(hem_ious_numeric) > 0 else '  Mean hemorrhage IoU:    N/A',
        f'  Median hemorrhage IoU:  {hem_ious_numeric.median():.4f}' if len(hem_ious_numeric) > 0 else '  Median hemorrhage IoU:  N/A',
        f'  Mean precision:         {hem_precisions.mean():.4f}' if len(hem_precisions) > 0 else '  Mean precision:         N/A',
        f'  Mean recall:            {hem_recalls.mean():.4f}' if len(hem_recalls) > 0 else '  Mean recall:            N/A',
        '',
        '── False positive check (normal samples) ──',
        f'  Normal samples with false positives: {normal_samples["false_positives"].astype(int).gt(0).sum()} / {len(normal_samples)}',
        '',
        '══════════════════════════════════════════════════════',
    ]

    # Print and save
    summary_text = '\n'.join(summary_lines)
    print('\n' + summary_text)

    results_path = os.path.join(RESULTS_DIR, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(summary_text + '\n')
    print(f'\n  Saved to {results_path}')

    # ── Confusion matrix ───────────────────────────────────────────────────────
    print('\n── Saving confusion matrix ──')
    cm = np.array([[total_tn, total_fp],
                   [total_fn, total_tp]])

    total_pixels = total_tn + total_fp + total_fn + total_tp
    annot_labels = np.array([
        [f'True negative\n{total_tn:,}\n({total_tn/total_pixels*100:.1f}%)',
         f'False positive\n{total_fp:,}\n({total_fp/total_pixels*100:.1f}%)'],
        [f'False negative\n{total_fn:,}\n({total_fn/total_pixels*100:.1f}%)',
         f'True positive\n{total_tp:,}\n({total_tp/total_pixels*100:.1f}%)'],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='RdYlGn',
                xticklabels=['Background', 'Hemorrhage'],
                yticklabels=['Background', 'Hemorrhage'],
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                linewidths=1, linecolor='white',
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
    ax.set_title('Pixel-level confusion matrix', fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved to {cm_path}')

    # ── Regenerate prediction images ──────────────────────────────────────────
    print('\n── Saving prediction images ──')
    show_predictions(model, ds_test, num=10)

    print(f'\n✓ All test results saved to {RESULTS_DIR}/')


if __name__ == '__main__':
    main()
