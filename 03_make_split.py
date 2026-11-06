"""
Step 03 — Split BDAPPV into train / validation / test, GEOGRAPHICALLY.

Lab 2 (ECG) showed that splitting beats at random puts beats of the same patient in train
AND validation, which overestimates performance on new patients. The same happens here:
a 400x400 thumbnail covers 80 m x 80 m, so neighbouring installations show up in each
other's thumbnails, and houses in one town look alike (same roof tiles, same building
style). A random split would test on near-copies of the training images.

So we split per GROUP = département (from metadata.csv). All images of one département go
to the same set. Images without metadata go to 'train' only (we cannot place them).
Test is used ONCE, in step 09. Validation is for every decision before that.

SPLIT_MODE = "random" makes the leaky split instead, for an optional leakage experiment
like lab 2's 03_train_random_split.py.

Requires: 01_check_data.py (and the INCLUDE_UNMASKED decision from step 02 in common.py)
Output:   processed/split.csv (stem, img_path, mask_path, group, split)
Next:     04_check_dataloader.py
"""

import numpy as np
import pandas as pd

from common import INCLUDE_UNMASKED, METADATA_CSV, PROCESSED_DIR, SEED, ensure_dirs

SPLIT_MODE = "group"            # "group" (by département) or "random" (leaky, only for the experiment)
GROUP_COLUMN = "departement"    # or "city": smaller groups, more of them
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign_groups(sizes: pd.Series, rng) -> dict:
    """Shuffle the groups, then fill test and val up to their share of IMAGES; the rest is train."""
    order = rng.permutation(sizes.index.values)
    total, assigned, filled = sizes.sum(), {}, {"test": 0, "val": 0}
    for g in order:
        for split in ("test", "val"):
            if filled[split] < FRACTIONS[split] * total:
                assigned[g] = split
                filled[split] += sizes[g]
                break
        else:
            assigned[g] = "train"
    return assigned


def main():
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    inv = pd.read_csv(PROCESSED_DIR / "inventory.csv", dtype={"stem": str}).fillna("")
    if not INCLUDE_UNMASKED:
        inv = inv[inv.mask_path != ""].copy()
        print(f"Using only the {len(inv):,} images WITH a mask (INCLUDE_UNMASKED = False)")
    else:
        print(f"Using {len(inv):,} images; {(inv.mask_path == '').sum():,} without mask are treated as EMPTY")

    inv["group"] = "unknown"
    if METADATA_CSV.exists():
        meta = pd.read_csv(METADATA_CSV, dtype=str)
        if "identifiant" in meta.columns and GROUP_COLUMN in meta.columns:
            meta["stem"] = meta["identifiant"].astype(str).str.replace(r"\.(png|jpg)$", "", regex=True)
            lookup = meta.dropna(subset=[GROUP_COLUMN]).drop_duplicates("stem").set_index("stem")[GROUP_COLUMN]
            inv["group"] = inv.stem.map(lookup).fillna("unknown").astype(str)
    known = inv[inv.group != "unknown"]
    print(f"Images with a known {GROUP_COLUMN}: {len(known):,} of {len(inv):,} "
          f"({known.group.nunique()} different groups)")

    if SPLIT_MODE == "random" or len(known) == 0:
        if SPLIT_MODE != "random":
            print("WARNING: no group information -> random split. Mention this leakage risk in your report!")
        u = rng.random(len(inv))
        inv["split"] = np.where(u < FRACTIONS["test"], "test",
                                np.where(u < FRACTIONS["test"] + FRACTIONS["val"], "val", "train"))
    else:
        assigned = assign_groups(known.groupby("group").size(), rng)
        inv["split"] = inv.group.map(assigned).fillna("train")

    out = PROCESSED_DIR / ("split.csv" if SPLIT_MODE == "group" else "split_random.csv")
    inv.to_csv(out, index=False)
    print()
    for split in ("train", "val", "test"):
        part = inv[inv.split == split]
        print(f"  {split:5s}: {len(part):6,} images from {part.group.nunique():3d} groups "
              f"({(part.mask_path != '').mean() * 100:.0f}% with panels)")
    overlap = set(inv[inv.split == "train"].group) & set(inv[inv.split == "test"].group) - {"unknown"}
    print(f"  groups in both train and test: {len(overlap)} (must be 0 for the group split)")
    print(f"\nSaved {out}")
    if SPLIT_MODE == "random":
        print("Train on it with: python 05_train.py --split-file split_random.csv --run-name unet_randomsplit ...")
        print("Then set SPLIT_MODE back to \"group\".")


if __name__ == "__main__":
    main()
