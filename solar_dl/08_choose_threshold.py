"""
Step 08 — Choose the decision threshold on the VALIDATION set (never on test).

The model outputs a probability per pixel; 'panel' means probability > threshold. 0.5 is
not sacred: with rare positives and a Dice term in the loss, another value can give a
better balance between false alarms (precision) and missed panels (recall).

This step builds the pixel-level precision-recall curve on the validation set (lecture
'ROC and PR curves': positives are rare here, so PR/AP is the right curve, and AP only
means something next to the prevalence π, which is printed too), and picks the threshold
with the highest F1. Note: IoU = F1 / (2 - F1) is an increasing function of F1, so the
F1-optimal threshold is also the IoU-optimal one.

  python 08_choose_threshold.py --run unet_base

Requires: a finished run
Output:   runs/<run>/threshold.json, plots/pr_curve_<run>.png
Next:     09_evaluate_test.py (only when ALL decisions are made)
"""

import argparse
import json

import numpy as np
import torch

from common import PLOTS_DIR, RUNS_DIR, ensure_dirs, get_device, iterate_probs, load_run, make_datasets

BINS = 1000


def pr_from_histograms(h_pos, h_neg):
    """Precision/recall/F1/IoU for thresholds at every bin edge, from probability histograms."""
    tp = np.cumsum(h_pos[::-1])[::-1].astype(np.float64)  # pixels with prob >= edge that are panel
    fp = np.cumsum(h_neg[::-1])[::-1].astype(np.float64)
    positives = h_pos.sum()
    fn = positives - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(positives, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    iou = tp / np.maximum(tp + fp + fn, 1)
    return precision, recall, f1, iou


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-images", type=int, default=None, help="limit for a quick CPU check")
    a = p.parse_args()
    ensure_dirs()
    device = get_device(a.device)
    model, config, _ = load_run(a.run, device)
    _, val_ds, _ = make_datasets(split_file=config.get("split_file", "split.csv"))
    if a.max_images:
        val_ds = torch.utils.data.Subset(val_ds, range(min(a.max_images, len(val_ds))))

    h_pos, h_neg = np.zeros(BINS, np.int64), np.zeros(BINS, np.int64)
    for _, masks, probs in iterate_probs(model, val_ds, device):
        idx = np.clip((probs.numpy() * BINS).astype(int), 0, BINS - 1)
        truth = masks.numpy() > 0.5
        h_pos += np.bincount(idx[truth], minlength=BINS)
        h_neg += np.bincount(idx[~truth], minlength=BINS)

    edges = np.arange(BINS) / BINS
    precision, recall, f1, iou = pr_from_histograms(h_pos, h_neg)
    best = int(np.argmax(f1))
    at05 = BINS // 2
    # average precision = area under the PR curve (step-wise, as sklearn does)
    order = np.argsort(recall)
    ap = float(np.sum(np.diff(np.concatenate([[0], recall[order]])) * precision[order]))
    prevalence = h_pos.sum() / (h_pos.sum() + h_neg.sum())
    print(f"Validation pixels: {h_pos.sum() + h_neg.sum():,}, prevalence π = {prevalence:.4f} "
          f"(an AP of {prevalence:.3f} = random guessing)")
    print(f"Average precision (AP): {ap:.3f}")
    print(f"threshold 0.50: P {precision[at05]:.3f} R {recall[at05]:.3f} F1 {f1[at05]:.3f} IoU {iou[at05]:.3f}")
    print(f"threshold {edges[best]:.2f}: P {precision[best]:.3f} R {recall[best]:.3f} F1 {f1[best]:.3f} "
          f"IoU {iou[best]:.3f}   <- chosen (max F1)")

    out = RUNS_DIR / a.run / "threshold.json"
    out.write_text(json.dumps({"threshold": float(edges[best]), "val_f1": float(f1[best]),
                               "val_iou": float(iou[best]), "val_precision": float(precision[best]),
                               "val_recall": float(recall[best]), "val_ap": ap, "prevalence": float(prevalence)},
                              indent=2))
    print(f"Saved {out}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, label=f"{a.run} (AP {ap:.3f})")
    ax.plot(recall[at05], precision[at05], "ks", label="threshold 0.50")
    ax.plot(recall[best], precision[best], "r*", ms=12, label=f"chosen {edges[best]:.2f}")
    ax.axhline(prevalence, color="grey", ls=":", label=f"random (π = {prevalence:.3f})")
    ax.set_xlabel("recall (share of panel pixels found)")
    ax.set_ylabel("precision (share of predicted pixels that are panel)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Pixel precision-recall, validation set")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"pr_curve_{a.run}.png", dpi=150)
    plt.close(fig)
    print(f"Saved {PLOTS_DIR / f'pr_curve_{a.run}.png'}")


if __name__ == "__main__":
    main()
