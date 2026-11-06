"""
Step 05 — Train one segmentation model. Every experiment of the project is a call of this script.

Like labs 2-3, ALWAYS do the quick check first:
  python 05_train.py --model unet --quick --device cpu        (tiny subset, 2 epochs: does the code run?)
then the real run on a GPU node (slurm/gpu.slurm), e.g.:
  python 05_train.py --model unet --run-name unet_base

Useful variations (each is one run; give each a clear --run-name):
  --loss bce | dice | bce_dice        which loss (step 05 experiments)
  --lr 3e-4                           learning rate
  --freeze-encoder                    train only the decoder (lab 2's frozen ResNet) vs fine-tune all
  --no-pretrained                     random initialisation: how much does ImageNet help?
  --frac 0.25                         only 25% of the training images (learning curve, step 06)
  --model segformer                   the transformer model (phase 4)
  --split-file split_random.csv       the leaky random split (optional leakage experiment)
FALLBACK ONLY (if step 06 says you have too little data, see the guide):
  --data extra --extra-dir DIR --extra-gsd 0.10      pre-train on another dataset (validated on BDAPPV val)
  --data both  --extra-dir DIR --extra-gsd 0.10      train on BDAPPV + the extra dataset together
  --init-from RUN                                   start from the weights of an earlier run (fine-tuning)
  The extra dataset must have DIR/img/<name>.png and DIR/mask/<name>.png; it is resampled to 0.20 m/px.

Output: runs/<run-name>/{best.pt, config.json, history.json}, plots/curves_<run-name>.png,
        and one line in runs/results.csv
Next:   07_compare_runs.py to compare runs, 08_choose_threshold.py for the final model(s)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from common import (BATCH_SIZE, EPOCHS, GSD, LOSS, LR, RUNS_DIR, SEED, WEIGHT_DECAY, SegDataset, append_result,
                    count_parameters, ensure_dirs, get_device, make_datasets, make_loader, plot_histories,
                    save_run, set_seed, train_model)
from models import build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="unet", choices=["unet", "segformer", "segformer_hf"])
    p.add_argument("--encoder", default=None, help="default: resnet34 (unet), mit_b2 (segformer)")
    p.add_argument("--loss", default=LOSS, choices=["bce", "dice", "bce_dice"])
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--frac", type=float, default=1.0, help="fraction of the training images to use")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--init-from", default=None, help="run name whose best.pt initialises this model")
    p.add_argument("--data", default="bdappv", choices=["bdappv", "extra", "both"])
    p.add_argument("--extra-dir", default=None)
    p.add_argument("--extra-gsd", type=float, default=None, help="m/pixel of the extra dataset")
    p.add_argument("--split-file", default="split.csv")
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--quick", action="store_true", help="tiny subset, 2 epochs: only checks that the code runs")
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def extra_datasets(folder: str, gsd: float, seed: int = SEED):
    """img/ + mask/ of another PV dataset, resampled to our GSD, split 85/15 (only its train part is used)."""
    folder = Path(folder)
    stems = sorted(p.stem for p in (folder / "img").iterdir())
    masks = {p.stem: p for p in (folder / "mask").iterdir()}
    imgs = {p.stem: p for p in (folder / "img").iterdir()}
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)
    scale = gsd / GSD  # e.g. 0.10 / 0.20 = 0.5: halve the size so a panel has as many pixels as in IGN
    print(f"Extra dataset {folder}: {len(stems):,} images at {gsd} m/px -> resampled by x{scale:.2f}")
    return SegDataset([imgs[s] for s in stems], [masks.get(s) for s in stems], augment=True, scale=scale)


def main():
    a = parse_args()
    ensure_dirs()
    set_seed()
    device = get_device(a.device)
    name = a.run_name or f"{a.model}_{a.loss}_lr{a.lr:g}" + (f"_frac{a.frac:g}" if a.frac < 1 else "")
    if a.quick:
        name = "quick_" + name
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run '{name}' on {device.type.upper()}" + (" (QUICK CHECK: tiny subset, 2 epochs)" if a.quick else ""))

    train_ds, val_ds, _ = make_datasets(frac=a.frac, split_file=a.split_file)
    if a.data in ("extra", "both"):
        if not (a.extra_dir and a.extra_gsd):
            raise SystemExit("--data extra/both needs --extra-dir and --extra-gsd")
        extra = extra_datasets(a.extra_dir, a.extra_gsd)
        train_ds = extra if a.data == "extra" else ConcatDataset([train_ds, extra])
    epochs, workers = a.epochs, a.num_workers
    if a.quick:
        train_ds = Subset(train_ds, range(min(64, len(train_ds))))
        val_ds = Subset(val_ds, range(min(32, len(val_ds))))
        epochs, workers = 2, 0 if workers is None else workers
    workers = 4 if workers is None else workers
    train_loader = make_loader(train_ds, a.batch_size, shuffle=True, num_workers=workers)
    val_loader = make_loader(val_ds, a.batch_size, shuffle=False, num_workers=workers)
    print(f"Training images: {len(train_ds):,} | validation images: {len(val_ds):,} | "
          f"{len(train_loader)} updates per epoch")

    model = build_model(a.model, a.encoder, pretrained=not a.no_pretrained, freeze_encoder=a.freeze_encoder)
    if a.init_from:
        state = torch.load(RUNS_DIR / a.init_from / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"Initialised from run {a.init_from}")
    print(f"Parameters: {count_parameters(model, False) / 1e6:.1f} M total, {count_parameters(model) / 1e6:.1f} M trainable")
    model = model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                                  weight_decay=WEIGHT_DECAY)

    start = time.time()
    history = train_model(model, train_loader, val_loader, optimizer, device, run_dir,
                          epochs=epochs, loss_kind=a.loss, desc=name)
    minutes = (time.time() - start) / 60
    best = history["best_val"]
    print(f"\nTraining took {minutes:.1f} min. Best epoch {history['best_epoch']}: val IoU {best['iou']:.3f}, "
          f"F1 {best['f1']:.3f}, precision {best['precision']:.3f}, recall {best['recall']:.3f}")

    config = {"model": a.model, "encoder": a.encoder, "loss": a.loss, "lr": a.lr, "frac": a.frac,
              "freeze_encoder": a.freeze_encoder, "pretrained": not a.no_pretrained, "init_from": a.init_from,
              "data": a.data, "extra_dir": a.extra_dir, "extra_gsd": a.extra_gsd, "split_file": a.split_file,
              "batch_size": a.batch_size, "epochs": epochs, "quick": a.quick}
    save_run(run_dir, config, history)
    print(f"Saved {run_dir}")
    print(plot_histories({name: history}, name, f"curves_{name}.png"))
    if not a.quick:
        append_result({"run": name, "model": a.model, "encoder": a.encoder or "default", "loss": a.loss, "lr": a.lr,
                       "frac": a.frac, "freeze_encoder": a.freeze_encoder, "pretrained": not a.no_pretrained,
                       "init_from": a.init_from or "", "data": a.data, "n_train": len(train_ds),
                       "epochs_run": len(history["train_loss"]), "best_epoch": history["best_epoch"],
                       "val_iou": round(best["iou"], 4), "val_f1": round(best["f1"], 4),
                       "val_precision": round(best["precision"], 4), "val_recall": round(best["recall"], 4),
                       "params_M": round(count_parameters(model, False) / 1e6, 2), "train_min": round(minutes, 1)})
    else:
        print('\nQuick check done. No errors and the numbers move? Then submit the real run on a GPU '
              '(slurm/gpu.slurm). Quick runs are NOT written to runs/results.csv.')


if __name__ == "__main__":
    main()
