"""
Step 11 — Turn your QGIS panel polygons into masks for the labelled chips.

How you label (details in the guide, step 5.3):
  - Each of you draws in YOUR OWN GeoPackage: data/flanders/labels/labels_<name>.gpkg
    (polygon layer, EPSG:31370). Separate files = no editing conflicts between you two.
  - Draw EVERY panel area on the chip (also solar water heaters? decide once, write it in
    your labelling protocol, apply it everywhere).
  - When a chip is finished, add a line to data/flanders/labels/labelled.csv:
        chip_id,annotator
        leuven_000,axel
        leuven_001,tibo
    A chip that is NOT in labelled.csv is ignored: 'not labelled yet' must never be
    confused with 'no panels' (an empty but labelled chip is a valid, important example).
  - Label ~10 chips TWICE (once each, both lines in labelled.csv). On those chips this
    script measures how much you two agree (inter-annotator IoU): the best IoU you can
    expect from any model, given how ambiguous panel borders are at 20 cm.

  python 11_rasterize_labels.py

Requires: 10_prepare_flanders.py, labels_<name>.gpkg files, labelled.csv
Output:   data/flanders/labels/masks/<chip_id>.png, data/flanders/labels/flanders_eval.csv
Next:     12_evaluate_flanders.py
"""

import argparse

import numpy as np
import pandas as pd

from common import LABELS_DIR, ensure_dirs
from geo import load_polygons


def rasterize_chip(polys, tif_path):
    import rasterio
    from rasterio.features import rasterize
    with rasterio.open(tif_path) as src:
        shape, transform = (src.height, src.width), src.transform
        from shapely.geometry import box
        chip = box(*src.bounds)
    geoms = [g for g in polys.geometry if g is not None and g.intersects(chip)]
    if not geoms:
        return np.zeros(shape, dtype=np.uint8)
    # a pixel is 'panel' if its CENTRE lies inside a polygon (all_touched=False)
    return rasterize(geoms, out_shape=shape, transform=transform, fill=0, default_value=1, dtype="uint8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labelled", default=str(LABELS_DIR / "labelled.csv"))
    a = p.parse_args()
    ensure_dirs()
    from PIL import Image
    chips = pd.read_csv(LABELS_DIR / "chips.csv").set_index("chip_id")
    done = pd.read_csv(a.labelled, dtype=str)
    annotators = sorted(done.annotator.unique())
    layers = {}
    for name in annotators:
        path = LABELS_DIR / f"labels_{name}.gpkg"
        if not path.exists():
            raise SystemExit(f"{path} not found (annotator {name!r} in {a.labelled})")
        layers[name] = load_polygons(path)
        print(f"{path.name}: {len(layers[name])} polygons")

    masks_dir = LABELS_DIR / "masks"
    masks_dir.mkdir(exist_ok=True)
    rows, agreement = [], []
    for cid, group in done.groupby("chip_id", sort=True):
        if cid not in chips.index:
            print(f"  {cid} is in labelled.csv but not in chips.csv: skipped")
            continue
        chip = chips.loc[cid]
        masks = {r.annotator: rasterize_chip(layers[r.annotator], chip.tif) for r in group.itertuples()}
        first = group.annotator.iloc[0]  # the first annotator listed is the 'truth' for double-labelled chips
        Image.fromarray(masks[first] * 255).save(masks_dir / f"{cid}.png")
        rows.append({"chip_id": cid, "city": chip.city, "img_path": chip.png,
                     "mask_path": str(masks_dir / f"{cid}.png"), "annotator": first,
                     "panel_pixels": int(masks[first].sum())})
        if len(masks) == 2:
            m1, m2 = list(masks.values())
            inter, union = int((m1 & m2).sum()), int((m1 | m2).sum())
            agreement.append((cid, inter, union))

    df = pd.DataFrame(rows)
    df.to_csv(LABELS_DIR / "flanders_eval.csv", index=False)
    print(f"\nLabelled chips: {len(df)} ({(df.panel_pixels > 0).sum()} with panels, "
          f"{(df.panel_pixels == 0).sum()} without)")
    print(df.groupby("city").agg(chips=("chip_id", "count"), with_panels=("panel_pixels", lambda s: (s > 0).sum())))
    if agreement:
        inter = sum(x[1] for x in agreement)
        union = sum(x[2] for x in agreement)
        print(f"\nInter-annotator agreement on {len(agreement)} double-labelled chips: IoU {inter / max(union, 1):.3f}")
        print("  -> no model can be expected to agree with 'the truth' much better than you agree with each other.")
    print(f"Saved {LABELS_DIR / 'flanders_eval.csv'} and masks in {masks_dir}")


if __name__ == "__main__":
    main()
