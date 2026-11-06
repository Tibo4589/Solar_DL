"""
Step 06 — Learning curve: do we have TOO LITTLE data? (decides whether we need the fallback)

Your TA suggested extra data (a Chinese dataset to pre-train, BDAPPV to fine-tune) in case
BDAPPV is too small. This step turns 'too small' into something you can MEASURE and defend.

Train the SAME configuration on growing fractions of the training set:
  python 05_train.py --model unet --frac 0.10 --run-name lc_unet_f0.10
  python 05_train.py --model unet --frac 0.25 --run-name lc_unet_f0.25
  python 05_train.py --model unet --frac 0.50 --run-name lc_unet_f0.50
  (the full run, frac 1.0, is your baseline: name it lc_unet_f1.00 or pass --full-run NAME)
and plot the best validation IoU against the number of training images (log scale).

How to read it:
  - The curve is FLAT at the end (doubling the data from 50% to 100% adds < ~0.01 IoU):
    more data of the SAME kind will not help much -> no fallback needed.
  - The curve still CLIMBS steeply at 100%: more labelled data would help -> the fallback
    (extra dataset) is justified. Say so on the poster with this plot.
  - Careful: this tells you about French validation data. If later (step 12) Flanders is
    much worse than the French test set, that is a DOMAIN gap, which more French or
    Chinese data does not necessarily fix. Label a few Flemish tiles for fine-tuning instead.

Requires: the lc_* runs (05_train.py)
Output:   plots/learning_curve.png
"""

import argparse

import numpy as np
import pandas as pd

from common import PLOTS_DIR, RESULTS_CSV, ensure_dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="lc_", help="runs whose name starts with this are learning-curve runs")
    p.add_argument("--full-run", default=None, help="name of the frac=1.0 run if it is not called lc_...")
    a = p.parse_args()
    ensure_dirs()
    res = pd.read_csv(RESULTS_CSV)
    lc = res[res.run.str.startswith(a.prefix) | (res.run == a.full_run)].drop_duplicates("run", keep="last")
    if len(lc) < 2:
        raise SystemExit(f"Need at least 2 learning-curve runs in {RESULTS_CSV}; found {len(lc)}.")
    lc = lc.sort_values("n_train")
    print(lc[["run", "frac", "n_train", "val_iou", "val_f1", "best_epoch"]].to_string(index=False))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lc.n_train, lc.val_iou, "o-")
    for r in lc.itertuples():
        ax.annotate(f"{r.frac:g}", (r.n_train, r.val_iou), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training images (log scale)")
    ax.set_ylabel("best validation IoU")
    ax.set_title("Learning curve: does more data still help?")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {PLOTS_DIR / 'learning_curve.png'}")

    last, prev = lc.iloc[-1], lc.iloc[-2]
    gain_per_doubling = (last.val_iou - prev.val_iou) / max(np.log2(last.n_train / prev.n_train), 1e-9)
    print(f"Gain per doubling of the data at the end of the curve: {gain_per_doubling:+.3f} IoU")
    if gain_per_doubling > 0.02:
        print("-> still climbing: extra training data is likely to help (fallback justified).")
    else:
        print("-> (nearly) flat: more data of this kind will not help much; no need for the fallback.")
    print("   These thresholds are a rule of thumb: discuss the plot, not only the number.")


if __name__ == "__main__":
    main()
