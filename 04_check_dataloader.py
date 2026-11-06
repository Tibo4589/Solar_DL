"""
Step 04 — Check the Dataset, the augmentation and the model BEFORE spending GPU time.

  1. shapes and value ranges of one batch (image (B, 3, 384, 384) in [0, 1], mask (B, 1, 384, 384) in {0, 1})
  2. how many mini-batches (= weight updates) one epoch has (lab 1, step 05)
  3. a picture of augmented training samples: are the masks still ON the panels after
     rotating/flipping/cropping? (a bug here silently ruins every later result)
  4. the trivial baseline 'predict no panel anywhere' on the validation set: its accuracy
     looks great and its IoU is 0. Remember this when someone asks 'why not accuracy?'
  5. one forward pass of the U-Net on the CPU (no pretrained download needed)

Requires: 03_make_split.py
Output:   plots/augmented_batch.png
Next:     05_train.py (first with --quick on the CPU)
"""

import torch

from common import (BATCH_SIZE, PLOTS_DIR, Counts, count_parameters, ensure_dirs, make_datasets, make_loader,
                    overlay, set_seed, to_uint8_rgb)
from models import build_model


def main():
    ensure_dirs()
    set_seed()
    train_ds, val_ds, test_ds = make_datasets()
    print(f"Images: train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,}")
    loader = make_loader(train_ds, shuffle=True, num_workers=0)
    images, masks = next(iter(loader))
    print(f"One training batch: images {tuple(images.shape)} {images.dtype} [{images.min():.2f}, {images.max():.2f}], "
          f"masks {tuple(masks.shape)} unique {torch.unique(masks).tolist()}")
    print(f"Mini-batches per epoch at batch size {BATCH_SIZE}: {len(loader):,} "
          f"(= weight updates per epoch; the last incomplete batch is dropped)")

    import matplotlib.pyplot as plt
    n = min(8, len(images))
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5))
    for i in range(n):
        img = to_uint8_rgb(images[i])
        axes[0, i].imshow(img)
        axes[1, i].imshow(overlay(img, masks[i, 0].numpy()))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("augmented input", fontsize=8, loc="left")
    axes[1, 0].set_title("with its mask", fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "augmented_batch.png", dpi=130)
    plt.close(fig)
    print(f"Saved {PLOTS_DIR / 'augmented_batch.png'}  <- check that every red area sits exactly on a panel")

    counts = Counts()
    for i in range(len(val_ds)):
        _, m = val_ds[i]
        counts.add(torch.zeros_like(m, dtype=torch.bool), m > 0.5)
    m = counts.metrics()
    print(f"\n'No panel anywhere' on validation: accuracy {m['accuracy'] * 100:.1f}%  IoU {m['iou']:.3f}  "
          f"recall {m['recall']:.3f}   <- why we never report accuracy")

    model = build_model("unet", pretrained=False).eval()
    with torch.no_grad():
        out = model(images[:2])
    print(f"\nU-Net forward pass: {tuple(images[:2].shape)} -> {tuple(out.shape)} logits "
          f"(one per pixel; sigmoid(logit) = probability of 'panel'), {count_parameters(model) / 1e6:.1f} M parameters")


if __name__ == "__main__":
    main()
