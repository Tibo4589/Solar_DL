"""
Step 09 — Evaluate the FINAL model(s) on the BDAPPV test set. Once.

Like step 16 of lab 1: this is the only script that touches the test set. You made many
decisions by looking at validation scores (loss, learning rate, model, threshold...), so
the validation score is optimistic by construction. The test set (other départements)
gives the honest in-domain number: 'how well does it work on FRENCH images it never saw?'

  python 09_evaluate_test.py --runs unet_final segformer_final

Reports per run, with the threshold of step 08: IoU, F1, precision, recall, and the area
ratio (predicted panel area / true panel area: 1.00 = the total AREA is right even if
individual pixels are wrong, which matters for our m² estimates).
Also makes an error picture: green = correct, yellow = false alarm, cyan = missed.

Requires: 08_choose_threshold.py for each run
Output:   outputs/test_metrics.csv, plots/test_examples_<run>.png, plots/test_iou_per_image.png
Next:     phase 5, Flanders (10_prepare_flanders.py ...)
"""

import argparse

import numpy as np
import pandas as pd
import torch

from common import (OUTPUTS_DIR, PLOTS_DIR, Counts, ensure_dirs, error_image, get_device, iterate_probs, load_run,
                    make_datasets, predict_probs, run_threshold, to_uint8_rgb)


def evaluate(model, dataset, device, threshold):
    """Pooled pixel metrics + the IoU of every single image (nan: no panel present nor predicted)."""
    counts, per_image = Counts(), []
    for images, masks, probs in iterate_probs(model, dataset, device):
        pred, truth = probs > threshold, masks > 0.5
        counts.add(pred, truth)
        for i in range(len(images)):
            inter, union = int((truth[i] & pred[i]).sum()), int((truth[i] | pred[i]).sum())
            per_image.append(inter / union if union else np.nan)
    return counts.metrics(), np.array(per_image)


def example_figure(model, dataset, device, threshold, per_image, run, out_name, n=4):
    """Worst, median and best images (among images with panels): only these are predicted again,
    so we never keep the whole test set in memory."""
    import matplotlib.pyplot as plt
    has_panel = [i for i in range(len(dataset)) if not np.isnan(per_image[i]) and dataset[i][1].any()]
    if not has_panel:
        return None
    order = sorted(has_panel, key=lambda i: per_image[i])
    k = len(order)
    picks = {"worst": order[:n], "median": order[max(0, k // 2 - n // 2): k // 2 - n // 2 + n], "best": order[-n:]}
    fig, axes = plt.subplots(3, n, figsize=(2.8 * n, 9))
    for row, (label, idx) in enumerate(picks.items()):
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(idx):
                img, mask = dataset[idx[col]]
                prob = predict_probs(model, img[None], device)[0, 0]
                ax.imshow(error_image(to_uint8_rgb(img), mask[0].numpy() > 0.5, (prob > threshold).numpy()))
                ax.set_title(f"{label}: IoU {per_image[idx[col]]:.2f}", fontsize=8)
    fig.suptitle(f"{run} on the test set: green correct, yellow false alarm, cyan missed", fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / out_name, dpi=130)
    plt.close(fig)
    return PLOTS_DIR / out_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-images", type=int, default=None)
    a = p.parse_args()
    ensure_dirs()
    device = get_device(a.device)
    out_csv = OUTPUTS_DIR / "test_metrics.csv"
    old = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["run"])
    again = set(a.runs) & set(old.run)
    if again:
        print(f"WARNING: {sorted(again)} were already evaluated on test. Re-evaluating is fine, but if you CHANGED "
              f"the model because of the test result, the test score is no longer an honest estimate.")

    rows, ious = [], {}
    for run in a.runs:
        model, config, _ = load_run(run, device)
        _, _, test_ds = make_datasets(split_file=config.get("split_file", "split.csv"))
        if a.max_images:
            test_ds = torch.utils.data.Subset(test_ds, range(min(a.max_images, len(test_ds))))
        thr = run_threshold(run)
        m, per_image = evaluate(model, test_ds, device, thr)
        ious[run] = per_image[~np.isnan(per_image)]
        rows.append({"run": run, "model": config["model"], "threshold": thr, "n_images": len(test_ds),
                     **{k: round(v, 4) for k, v in m.items()}})
        print(f"{run}: IoU {m['iou']:.3f} F1 {m['f1']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"area ratio {m['area_ratio']:.2f} (threshold {thr:.2f}, {len(test_ds)} images)")
        print(f"  saved {example_figure(model, test_ds, device, thr, per_image, run, f'test_examples_{run}.png')}")

    new = pd.concat([old[~old.run.isin(a.runs)], pd.DataFrame(rows)], ignore_index=True)
    new.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(list(ious.values()))
    ax.set_xticks(range(1, len(ious) + 1), list(ious.keys()), rotation=15)
    ax.set_ylabel("IoU per test image (images with panels)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_iou_per_image.png", dpi=150)
    plt.close(fig)
    print(f"Saved {PLOTS_DIR / 'test_iou_per_image.png'}")


if __name__ == "__main__":
    main()
