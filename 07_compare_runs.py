"""
Step 07 — Compare training runs side by side (lab 1, steps 09-15 style).

  python 07_compare_runs.py unet_bce unet_dice unet_bce_dice --title "Loss function" --out loss.png

Plots training loss, validation loss and validation IoU per epoch for each run (dot = best
epoch) and prints their lines from runs/results.csv.

How to read (same rules as lab 1):
  - Compare at the dots (best epoch), not at the last epoch.
  - Use the training curves to DIAGNOSE: high train AND val loss = underfitting;
    low train loss but high val loss = overfitting. They need opposite fixes.
  - Differences of ~0.005 IoU between runs are run-to-run noise. Then take the simpler or
    cheaper option. (If you have time: repeat the best two runs with another seed.)

Output: plots/<out>
"""

import argparse
import json

import pandas as pd

from common import RESULTS_CSV, RUNS_DIR, plot_histories


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--title", default="Comparison")
    p.add_argument("--out", default="comparison.png")
    a = p.parse_args()
    histories = {r: json.loads((RUNS_DIR / r / "history.json").read_text()) for r in a.runs}
    print(f"Saved {plot_histories(histories, a.title, a.out)}")
    if RESULTS_CSV.exists():
        res = pd.read_csv(RESULTS_CSV)
        cols = ["run", "model", "loss", "lr", "frac", "freeze_encoder", "pretrained", "best_epoch",
                "val_iou", "val_f1", "val_precision", "val_recall", "params_M", "train_min"]
        print(res[res.run.isin(a.runs)].drop_duplicates("run", keep="last")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
