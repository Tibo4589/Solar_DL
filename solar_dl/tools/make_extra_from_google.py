"""
FALLBACK ONLY (if step 06 says more data would help): extra training data from BDAPPV's
OWN Google campaign (13,303 masks at 0.10 m/px).

Many Google thumbnails show the SAME installations as the IGN ones. If a Google image of a
test-département installation ends up in training, the test set is no longer 'unseen'
(leakage, as in lab 2). So this script copies ONLY the Google images whose name is in the
TRAIN part of processed/split.csv, into $SOLAR_ROOT/data/extra_google/{img,mask}.

  python tools/make_extra_from_google.py
  python 05_train.py --data both --extra-dir $SOLAR_ROOT/data/extra_google --extra-gsd 0.10 --run-name unet_plus_google

Requires: BDAPPV's google/ folder unzipped next to ign/ (unzip bdappv.zip 'bdappv/google/*'), 03_make_split.py
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BDAPPV_DIR, DATA_DIR, PROCESSED_DIR  # noqa: E402


def main():
    src_img, src_mask = BDAPPV_DIR / "google" / "img", BDAPPV_DIR / "google" / "mask"
    if not src_mask.is_dir():
        raise SystemExit(f"{src_mask} not found: unzip the google folder of BDAPPV first.")
    split = pd.read_csv(PROCESSED_DIR / "split.csv", dtype={"stem": str})
    train, other = set(split.stem[split.split == "train"]), set(split.stem[split.split != "train"])
    masks = {p.stem: p for p in src_mask.iterdir()}
    imgs = {p.stem: p for p in src_img.iterdir()}
    keep = sorted(s for s in masks if s in train and s in imgs)
    dropped = sum(1 for s in masks if s in other)
    unknown = sum(1 for s in masks if s not in train and s not in other)
    out = DATA_DIR / "extra_google"
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    for s in keep:
        shutil.copy(imgs[s], out / "img" / imgs[s].name)
        shutil.copy(masks[s], out / "mask" / masks[s].name)
    print(f"Google masks: {len(masks):,} | copied (train départements): {len(keep):,} | "
          f"left out, in val/test: {dropped:,} | left out, name not in the IGN split: {unknown:,}")
    if unknown > 0.5 * len(masks):
        print("Most Google names do not match IGN names: check how the files are named before using this.")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
