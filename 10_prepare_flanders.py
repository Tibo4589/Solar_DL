"""
Step 10 — Inspect the Flemish orthophotos and cut CHIPS for hand-labelling.

There are no labelled Flemish images, but the research question asks 'how ACCURATELY'.
So you label a small, random sample yourselves (step 11) and measure IoU/F1 on it (step 12).
This step picks that sample.

  python 10_prepare_flanders.py --city leuven --ortho data/flanders/ortho/leuven/*.tif \
      --boundary data/flanders/boundaries/gemeenten.gpkg --boundary-field NAAM --boundary-value Leuven \
      --buildings data/flanders/grb/leuven_gbg.gpkg --n-chips 50

What it does:
  1. prints CRS, pixel size, size and area of every orthophoto file (check: EPSG:31370, 0.15 m)
  2. lays a grid of TILE x TILE chips (384 px at 0.20 m = 76.8 m x 76.8 m) over the photo,
     keeps chips that lie inside the municipality and (if given) touch at least one building,
  3. draws a RANDOM sample of n chips (seeded) and writes each as a georeferenced GeoTIFF
     (open these in QGIS to draw the panels) and a PNG.

Why random (and not 'chips where the model finds panels'): choosing chips with the model
would hide exactly the panels the model MISSES, and the measured recall would be too high.
Why 'touching a building': panels are almost always on roofs; this avoids labelling 50
chips of fields, but keeps the sample random within the built-up area. Mention this.

Requires: the orthophotos (see the guide, phase 5)
Output:   data/flanders/labels/chips/<city>_<i>.tif/.png, data/flanders/labels/chips.csv,
          outputs/flanders/<city>/ortho_files.json
Next:     label the chips in QGIS, then 11_rasterize_labels.py
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd

from common import GSD, LABELS_DIR, OUTPUTS_DIR, SEED, TILE, ensure_dirs
from geo import LAMBERT72, describe, load_polygons, mostly_empty, open_resampled, read_padded, window_bounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True)
    p.add_argument("--ortho", nargs="+", required=True, help="orthophoto file(s) of this city (GeoTIFF/JPEG2000)")
    p.add_argument("--boundary", default=None)
    p.add_argument("--boundary-field", default=None)
    p.add_argument("--boundary-value", default=None)
    p.add_argument("--buildings", default=None)
    p.add_argument("--n-chips", type=int, default=50)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    ensure_dirs()
    from shapely.geometry import box

    files = sorted({f for pattern in a.ortho for f in glob.glob(pattern)})
    if not files:
        raise SystemExit(f"No files match {a.ortho}")
    for f in files:
        d = describe(f)
        print(f"{f}\n  CRS {d['crs']} | pixel {d['res_m']} m | {d['width']:,} x {d['height']:,} px | "
              f"{d['bands']} bands {d['dtype']} | {d['area_km2']} km²")
        if "31370" not in d["crs"]:
            print(f"  WARNING: not Lambert 72 ({LAMBERT72}). Fine if your vector data is reprojected (geo.py does), "
                  "but check it.")
    out_dir = OUTPUTS_DIR / "flanders" / a.city
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ortho_files.json").write_text(json.dumps(files, indent=2))

    area = None
    if a.boundary:
        area = load_polygons(a.boundary, field=a.boundary_field, value=a.boundary_value).union_all()
        print(f"Municipality boundary: {area.area / 1e6:.1f} km²")
    buildings = load_polygons(a.buildings) if a.buildings else None
    if buildings is not None:
        print(f"Buildings: {len(buildings):,} polygons")

    candidates = []
    for fi, f in enumerate(files):
        with open_resampled(f) as vrt:
            for row in range(0, vrt.height - TILE + 1, TILE):
                for col in range(0, vrt.width - TILE + 1, TILE):
                    chip = box(*window_bounds(vrt, col, row))
                    if area is not None and not area.contains(chip):
                        continue
                    candidates.append((fi, col, row, chip))
    if buildings is not None:
        sindex = buildings.sindex
        candidates = [c for c in candidates if len(sindex.query(c[3], predicate="intersects"))]
    print(f"Candidate chips ({TILE} px = {TILE * GSD:.1f} m): {len(candidates):,}")

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(candidates))
    chips_dir = LABELS_DIR / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    import rasterio
    from PIL import Image
    rows, i = [], 0
    for k in order:
        if i >= a.n_chips:
            break
        fi, col, row, chip = candidates[k]
        with open_resampled(files[fi]) as vrt:
            rgb = read_padded(vrt, col, row)
            transform = vrt.window_transform(rasterio.windows.Window(col, row, TILE, TILE))
            crs = vrt.crs
        if mostly_empty(rgb):
            continue
        cid = f"{a.city}_{i:03d}"
        with rasterio.open(chips_dir / f"{cid}.tif", "w", driver="GTiff", width=TILE, height=TILE, count=3,
                           dtype="uint8", crs=crs, transform=transform) as dst:
            dst.write(rgb.transpose(2, 0, 1))
        Image.fromarray(rgb).save(chips_dir / f"{cid}.png")
        rows.append({"chip_id": cid, "city": a.city, "file": files[fi], "col": col, "row": row,
                     "left": chip.bounds[0], "bottom": chip.bounds[1], "right": chip.bounds[2], "top": chip.bounds[3],
                     "tif": str(chips_dir / f"{cid}.tif"), "png": str(chips_dir / f"{cid}.png")})
        i += 1

    chips_csv = LABELS_DIR / "chips.csv"
    old = pd.read_csv(chips_csv) if chips_csv.exists() else pd.DataFrame(columns=["city"])
    pd.concat([old[old.city != a.city], pd.DataFrame(rows)], ignore_index=True).to_csv(chips_csv, index=False)
    print(f"Wrote {len(rows)} chips to {chips_dir} and {chips_csv}")
    print("Next: label them in QGIS (guide, step 5.3), list the finished ones in labels/labelled.csv, "
          "then run 11_rasterize_labels.py")


if __name__ == "__main__":
    main()
