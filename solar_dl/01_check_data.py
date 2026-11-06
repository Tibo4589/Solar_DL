"""
Step 01 — Check that BDAPPV (IGN campaign) is where we expect it and what it contains.

BDAPPV (Kasmi et al., Scientific Data 2023) ships, per campaign, an img/ folder with all
thumbnails shown to the crowd and a mask/ folder with a mask for the images in which
annotators drew panels. For IGN: 17,325 images, 7,686 masks, 400x400 px at 0.20 m/px.
metadata.csv links installations to images through the column 'identifiant'.

This script does not decide anything. It counts, pairs images with masks by file name,
checks the pixel values, and tells you how many masks can be linked to metadata (which
we need for the geographic split in step 03 and the area factors in step 02).

Requires: BDAPPV unzipped in $SOLAR_ROOT/data/bdappv (see the guide, phase 1)
Output:   processed/inventory.csv  (one row per image: stem, img_path, mask_path)
Next:     02_explore_data.py
"""

import numpy as np
import pandas as pd
from PIL import Image

from common import IMG_DIR, MASK_DIR, METADATA_CSV, PROCESSED_DIR, ensure_dirs

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder):
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS}


def main():
    ensure_dirs()
    for d in (IMG_DIR, MASK_DIR):
        if not d.is_dir():
            raise FileNotFoundError(f"{d} not found. Unzip BDAPPV so that bdappv/ign/img and bdappv/ign/mask exist, "
                                    f"and check that SOLAR_ROOT points to the shared project folder.")
    imgs, masks = list_images(IMG_DIR), list_images(MASK_DIR)
    paired = sorted(set(imgs) & set(masks))
    mask_only = sorted(set(masks) - set(imgs))
    print(f"Images: {len(imgs):,}   masks: {len(masks):,}   pairs (same file name): {len(paired):,}")
    print(f"Images WITHOUT a mask: {len(imgs) - len(paired):,}   masks without an image: {len(mask_only):,}")

    # --- what does one pair look like? -------------------------------------
    if paired:
        s = paired[0]
        img, m = np.asarray(Image.open(imgs[s])), np.asarray(Image.open(masks[s]))
        print(f"\nExample {s}: image {img.shape} {img.dtype}, values {img.min()}..{img.max()}")
        print(f"             mask  {m.shape} {m.dtype}, unique values {np.unique(m)[:10]}")
        print("  -> common.read_mask() treats every value > 0 as 'panel'. Check that this is right for these masks.")

    # --- can the masks be linked to the metadata? --------------------------
    if METADATA_CSV.exists():
        meta = pd.read_csv(METADATA_CSV, dtype=str)
        print(f"\nmetadata.csv: {len(meta):,} rows, columns: {list(meta.columns)}")
        if "identifiant" in meta.columns:
            ids = set(meta["identifiant"].astype(str).str.replace(r"\.(png|jpg)$", "", regex=True))
            linked = [s for s in paired if s in ids]
            print(f"Masks whose file name matches an 'identifiant': {len(linked):,} of {len(paired):,}")
            if len(linked) < 0.3 * max(len(paired), 1):
                print("  -> LOW match rate. Look at a few identifiant values and file names and fix the matching "
                      "in 03_make_split.py before you rely on the geographic split.")
        for col in ("departement", "city", "surface", "kWp", "tilt"):
            if col in meta.columns:
                print(f"  {col:12s} non-empty in {meta[col].notna().sum():,} rows")
    else:
        print(f"\n{METADATA_CSV} not found: the split in step 03 will fall back to a RANDOM split (leakage risk!).")

    rows = [{"stem": s, "img_path": str(p), "mask_path": str(masks[s]) if s in masks else ""}
            for s, p in imgs.items()]
    out = PROCESSED_DIR / "inventory.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
