"""
Make a tiny FAKE version of every input, to try the whole pipeline (steps 01-16) before
the real data is downloaded, or after you change code. Nothing here is real.

  export SOLAR_ROOT=/tmp/solar_fake          # NEVER point this at your real project folder
  python tools/make_fake_data.py             # fake BDAPPV + fake Flemish inputs
  ... run steps 01-10 ...
  python tools/make_fake_data.py --labels    # after step 10: fake 'hand labels' for the chips

Fake 'panels' are dark blue rectangles with a grid; fake roofs are red/grey rectangles.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BDAPPV_DIR, FLANDERS_DIR, LABELS_DIR, SOLAR_ROOT  # noqa: E402

RNG = np.random.default_rng(0)


def draw_scene(w, h, px_m, n_houses):
    """RGB image + list of house boxes and panel boxes (pixel coordinates)."""
    img = Image.new("RGB", (w, h), (90, 120, 70))
    d = ImageDraw.Draw(img)
    arr_noise = RNG.integers(-12, 12, (h, w, 3))
    houses, panels = [], []
    for _ in range(n_houses):
        hw, hh = int(RNG.uniform(8, 16) / px_m), int(RNG.uniform(8, 14) / px_m)
        x, y = int(RNG.uniform(0, w - hw)), int(RNG.uniform(0, h - hh))
        roof = tuple(int(v) for v in RNG.choice([[150, 70, 60], [120, 120, 125], [100, 60, 50]]))
        d.rectangle([x, y, x + hw, y + hh], fill=roof)
        houses.append((x, y, x + hw, y + hh))
        if RNG.random() < 0.5:
            pw, ph = int(hw * RNG.uniform(0.3, 0.6)), int(hh * RNG.uniform(0.25, 0.4))
            px, py = x + int(RNG.uniform(1, hw - pw - 1)), y + int(RNG.uniform(1, hh - ph - 1))
            d.rectangle([px, py, px + pw, py + ph], fill=(30, 40, 85))
            step = max(2, int(1.0 / px_m))
            for gx in range(px, px + pw, step):
                d.line([gx, py, gx, py + ph], fill=(60, 70, 110))
            panels.append((px, py, px + pw, py + ph))
    arr = np.clip(np.asarray(img).astype(int) + arr_noise, 0, 255).astype(np.uint8)
    return arr, houses, panels


def fake_bdappv(n=160):
    img_dir, mask_dir = BDAPPV_DIR / "ign" / "img", BDAPPV_DIR / "ign" / "mask"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        arr, _, panels = draw_scene(400, 400, 0.2, n_houses=6)
        stem = f"FAKE{i:05d}"
        Image.fromarray(arr).save(img_dir / f"{stem}.png")
        if panels and i < int(n * 0.75):
            m = Image.new("L", (400, 400), 0)
            for b in panels:
                ImageDraw.Draw(m).rectangle(b, fill=255)
            m.save(mask_dir / f"{stem}.png")
            area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in panels) * 0.04
            kwp = area / 7.0
            rows.append({"idInstallation": i, "identifiant": stem, "surface": round(area * 1.2, 1),
                         "kWp": round(kwp, 2), "tilt": int(RNG.uniform(20, 45)),
                         "departement": f"{int(RNG.integers(1, 13)):02d}", "city": f"town{i % 30}"})
    pd.DataFrame(rows).to_csv(BDAPPV_DIR / "metadata.csv", index=False)
    print(f"fake BDAPPV: {n} images in {img_dir}")


def fake_city(name, x0, y0, size_px=2600, px_m=0.15):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box
    arr, houses, panels = draw_scene(size_px, size_px, px_m, n_houses=int(size_px * size_px * px_m ** 2 / 600))
    ortho_dir = FLANDERS_DIR / "ortho" / name
    ortho_dir.mkdir(parents=True, exist_ok=True)
    top = y0 + size_px * px_m
    t = from_origin(x0, top, px_m, px_m)
    with rasterio.open(ortho_dir / f"{name}.tif", "w", driver="GTiff", width=size_px, height=size_px, count=3,
                       dtype="uint8", crs="EPSG:31370", transform=t) as dst:
        dst.write(arr.transpose(2, 0, 1))
    to_geo = lambda b: box(x0 + b[0] * px_m, top - b[3] * px_m, x0 + b[2] * px_m, top - b[1] * px_m)
    gpd.GeoDataFrame({"suit": RNG.choice(["ideaal", "bruikbaar", "beperkt"], len(houses))},
                     geometry=[to_geo(b) for b in houses], crs="EPSG:31370").to_file(
        FLANDERS_DIR / f"buildings_{name}.gpkg")
    gpd.GeoDataFrame(geometry=[to_geo(b) for b in panels], crs="EPSG:31370").to_file(
        FLANDERS_DIR / f"truth_panels_{name}.gpkg")
    margin = 30
    return box(x0 + margin, y0 + margin, x0 + size_px * px_m - margin, top - margin), len(panels)


def fake_flanders():
    import geopandas as gpd
    (FLANDERS_DIR).mkdir(parents=True, exist_ok=True)
    b1, n1 = fake_city("leuven", 173000, 174000)
    b2, n2 = fake_city("gent", 104000, 193000)
    gpd.GeoDataFrame({"NAAM": ["Leuven", "Gent"]}, geometry=[b1, b2], crs="EPSG:31370").to_file(
        FLANDERS_DIR / "gemeenten.gpkg")
    pd.DataFrame({"Hoofdgemeente": ["Leuven", "Gent", "Leuven"],
                  "Technologie": ["Zonne-energie", "Zonne-energie", "Wind"],
                  "Aantal installaties": [n1, n2, 1],
                  "Vermogen (kVA)": [f"{n1 * 4.5:.1f}".replace(".", ","), f"{n2 * 4.5:.1f}".replace(".", ","), "2000"]}
                 ).to_csv(FLANDERS_DIR / "installations.csv", sep=";", index=False)
    print(f"fake Flanders data in {FLANDERS_DIR}")


def fake_labels():
    """Pretend 'axel' labelled every chip with the true panels and 'tibo' double-labelled 4 chips, a bit sloppier."""
    import geopandas as gpd
    chips = pd.read_csv(LABELS_DIR / "chips.csv")
    truth = pd.concat([gpd.read_file(FLANDERS_DIR / f"truth_panels_{c}.gpkg") for c in chips.city.unique()])
    truth = gpd.GeoDataFrame(truth, crs="EPSG:31370")
    truth.to_file(LABELS_DIR / "labels_axel.gpkg")
    sloppy = truth.copy()
    sloppy["geometry"] = sloppy.buffer(0.3, join_style=2).translate(0.2, -0.2)
    sloppy.to_file(LABELS_DIR / "labels_tibo.gpkg")
    rows = [{"chip_id": c, "annotator": "axel"} for c in chips.chip_id]
    rows += [{"chip_id": c, "annotator": "tibo"} for c in chips.chip_id[:4]]
    pd.DataFrame(rows).to_csv(LABELS_DIR / "labelled.csv", index=False)
    print(f"fake labels: {len(chips)} chips, labels_axel.gpkg, labels_tibo.gpkg, labelled.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", action="store_true")
    a = p.parse_args()
    if "SOLAR_ROOT" not in os.environ or "fake" not in os.environ["SOLAR_ROOT"]:
        raise SystemExit("Set SOLAR_ROOT to a folder with 'fake' in its name first, e.g. export SOLAR_ROOT=/tmp/solar_fake")
    print(f"SOLAR_ROOT = {SOLAR_ROOT}")
    if a.labels:
        fake_labels()
    else:
        fake_bdappv()
        fake_flanders()
