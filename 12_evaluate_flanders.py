"""
Step 12 — How accurate is the model on FLEMISH images? (the first half of the research question)

Evaluates the final run(s) on your hand-labelled Flemish chips, per city and pooled, with
the threshold chosen on the French validation set (step 08). Do NOT re-tune the threshold
on these chips: they are your Flemish TEST set.

Compare with the French test set (step 09): the difference is the DOMAIN GAP
(other country, other camera and flight, winter instead of summer, other roofs).

Also checks AREA, which is what we report on the map: per chip, true panel area (m²) vs
predicted panel area. The pooled 'area ratio' tells you whether the model systematically
over- or underestimates m² in Flanders; you can use it to put an error bar on step 14.

  python 12_evaluate_flanders.py --runs unet_final segformer_final

Requires: 11_rasterize_labels.py, 08_choose_threshold.py
Output:   outputs/flanders_metrics.csv, plots/flanders_examples_<run>.png, plots/flanders_area_<run>.png
Next:     13_predict_area.py
"""

import argparse

import numpy as np
import pandas as pd
import torch

from common import (GSD, LABELS_DIR, OUTPUTS_DIR, PLOTS_DIR, Counts, SegDataset, ensure_dirs, error_image,
                    get_device, load_run, predict_probs, run_threshold, to_uint8_rgb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    ensure_dirs()
    device = get_device(a.device)
    chips = pd.read_csv(LABELS_DIR / "flanders_eval.csv")
    ds = SegDataset(chips.img_path, chips.mask_path, augment=False)
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    masks = torch.stack([ds[i][1] for i in range(len(ds))]) > 0.5
    print(f"{len(chips)} labelled Flemish chips: " + ", ".join(f"{c} {n}" for c, n in chips.city.value_counts().items()))

    french = pd.read_csv(OUTPUTS_DIR / "test_metrics.csv").set_index("run") \
        if (OUTPUTS_DIR / "test_metrics.csv").exists() else None
    rows = []
    import matplotlib.pyplot as plt
    for run in a.runs:
        model, config, _ = load_run(run, device)
        thr = run_threshold(run)
        pred = predict_probs(model, images, device) > thr
        for city in sorted(chips.city.unique()) + ["all"]:
            sel = np.ones(len(chips), bool) if city == "all" else (chips.city == city).to_numpy(copy=True)
            sel_t = torch.from_numpy(sel)
            c = Counts()
            c.add(pred[sel_t], masks[sel_t])
            m = c.metrics()
            rows.append({"run": run, "city": city, "chips": int(sel.sum()), "threshold": thr,
                         **{k: round(v, 4) for k, v in m.items()}})
            print(f"{run} | {city:8s}: IoU {m['iou']:.3f} F1 {m['f1']:.3f} P {m['precision']:.3f} "
                  f"R {m['recall']:.3f} area ratio {m['area_ratio']:.2f}")
        if french is not None and run in french.index:
            f = french.loc[run]
            print(f"{run} | France test: IoU {f.iou:.3f} F1 {f.f1:.3f}  -> domain gap in IoU: "
                  f"{f.iou - rows[-1]['iou']:+.3f}")

        # area per chip: true vs predicted (m²)
        true_m2 = masks.flatten(1).sum(1).numpy() * GSD ** 2
        pred_m2 = pred.flatten(1).sum(1).numpy() * GSD ** 2
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        for city in sorted(chips.city.unique()):
            s = (chips.city == city).values
            ax.scatter(true_m2[s], pred_m2[s], label=city, alpha=0.7)
        lim = max(true_m2.max(), pred_m2.max(), 1) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
        ax.set_xlabel("true panel area per chip (m²)")
        ax.set_ylabel("predicted panel area per chip (m²)")
        ax.set_title(f"{run}: area on Flemish chips")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"flanders_area_{run}.png", dpi=150)
        plt.close(fig)

        # error pictures: the 6 chips with the most panel pixels + the 6 with the most false alarms
        fp = (pred & ~masks).flatten(1).sum(1).numpy()
        pick = list(np.argsort(-true_m2)[:6]) + [i for i in np.argsort(-fp) if true_m2[i] == 0][:6]
        n = len(pick)
        ncols = 6
        fig, axes = plt.subplots(int(np.ceil(n / ncols)), ncols, figsize=(2.8 * ncols, 3 * int(np.ceil(n / ncols))))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, i in zip(np.atleast_1d(axes).ravel(), pick):
            ax.imshow(error_image(to_uint8_rgb(images[i]), masks[i, 0].numpy(), pred[i, 0].numpy()))
            ax.set_title(f"{chips.chip_id[i]}: true {true_m2[i]:.0f} m², pred {pred_m2[i]:.0f} m²", fontsize=7)
        fig.suptitle(f"{run} on Flemish chips: green correct, yellow false alarm, cyan missed "
                     f"(top: most panels, bottom: most false alarms)", fontsize=9)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"flanders_examples_{run}.png", dpi=130)
        plt.close(fig)
        print(f"  saved plots/flanders_area_{run}.png and plots/flanders_examples_{run}.png")

    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "flanders_metrics.csv", index=False)
    print(f"Saved {OUTPUTS_DIR / 'flanders_metrics.csv'}")


if __name__ == "__main__":
    main()
