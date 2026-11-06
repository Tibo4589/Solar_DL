"""
Step 14 — From probabilities to panel POLYGONS and AREA (m²) for one city.

Your TA's point: segmentation measures AREA, it does not count panels. Two panels next
to each other are one blob; one installation can be several blobs. So the main number
here is m² of panel, not 'number of panels'. (We still print the number of blobs, but
call it 'panel areas', not 'installations'.)

  python 14_area_statistics.py --run unet_final --city leuven \
      --boundary data/flanders/boundaries/gemeenten.gpkg --boundary-field NAAM --boundary-value Leuven \
      --buildings data/flanders/grb/leuven_gbg.gpkg

What it does:
  1. threshold the probability map (threshold of step 08), block by block
  2. polygonise the panel pixels, merge pieces cut by block borders, drop blobs smaller
     than MIN_PANEL_AREA_M2 (noise: one module is already ~1.7-2 m²)
  3. keep only what lies inside the municipality (the download has a 500 m buffer!)
  4. if buildings are given: mark blobs ON a building (rooftop) vs elsewhere (ground-mounted,
     greenhouses, false alarms on fields...). The project is about ROOFTOP panels.
  5. areas:
       projected area   = what the photo shows (horizontal projection), pixel count x 0.04 m²
       real area (est.) = projected / cos(typical tilt)   (factor from step 02, BDAPPV metadata)
       bias-corrected   = divided by the Flemish area ratio of step 12 (if available):
                          if the model over-predicts area by 10% on labelled chips, we correct for it.
     Report all three; the spread between them is part of your uncertainty.

Requires: 13_predict_area.py, (02 for area_factors.json, 12 for the Flemish area ratio)
Output:   outputs/flanders/<city>/panels_<run>.gpkg, outputs/flanders/<city>/summary_<run>.json
Next:     15_compare_installations.py
"""

import argparse
import json

import numpy as np
import pandas as pd

from common import GSD, MIN_PANEL_AREA_M2, OUTPUTS_DIR, PROCESSED_DIR, ensure_dirs, run_threshold
from geo import LAMBERT72, load_polygons

BLOCK = 4096


def polygonise(prob_path, threshold):
    """Panel polygons of one probability GeoTIFF, read in BLOCK x BLOCK pieces."""
    import rasterio
    from rasterio.features import shapes
    from rasterio.windows import Window
    from shapely.geometry import shape
    polys = []
    with rasterio.open(prob_path) as src:
        crs = src.crs
        for r in range(0, src.height, BLOCK):
            for c in range(0, src.width, BLOCK):
                win = Window(c, r, min(BLOCK, src.width - c), min(BLOCK, src.height - r))
                mask = (src.read(1, window=win) >= threshold * 255).astype(np.uint8)
                if not mask.any():
                    continue
                for geom, val in shapes(mask, mask=mask.astype(bool), transform=src.window_transform(win)):
                    polys.append(shape(geom))
    return polys, crs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--boundary", default=None)
    p.add_argument("--boundary-field", default=None)
    p.add_argument("--boundary-value", default=None)
    p.add_argument("--buildings", default=None)
    a = p.parse_args()
    ensure_dirs()
    import geopandas as gpd
    city_dir = OUTPUTS_DIR / "flanders" / a.city
    probs = sorted(city_dir.glob(f"prob_{a.run}_*.tif"))
    if not probs:
        raise SystemExit(f"No prob_{a.run}_*.tif in {city_dir}. Run 13_predict_area.py first.")
    thr = run_threshold(a.run)

    polys, crs = [], None
    for f in probs:
        new, crs = polygonise(f, thr)
        polys += new
        print(f"{f.name}: {len(new):,} raw polygons")
    if not polys:
        raise SystemExit("No panel pixels at all above the threshold.")
    gdf = gpd.GeoDataFrame(geometry=polys, crs=crs).to_crs(LAMBERT72)
    gdf = gdf.dissolve().explode(index_parts=False).reset_index(drop=True)  # merge pieces cut by block borders
    gdf["area_m2"] = gdf.area
    n_raw = len(gdf)
    gdf = gdf[gdf.area_m2 >= MIN_PANEL_AREA_M2].reset_index(drop=True)
    print(f"Panel areas: {n_raw:,} after merging, {len(gdf):,} after dropping blobs < {MIN_PANEL_AREA_M2} m²")

    muni_km2 = None
    if a.boundary:
        b = load_polygons(a.boundary, field=a.boundary_field, value=a.boundary_value)
        gdf = gpd.clip(gdf, b).explode(index_parts=False).reset_index(drop=True)
        gdf = gdf[gdf.geom_type == "Polygon"].reset_index(drop=True)  # clipping can leave lines/points on the border
        gdf["area_m2"] = gdf.area
        muni_km2 = float(b.area.sum() / 1e6)
        print(f"Inside the municipality ({muni_km2:.1f} km²): {len(gdf):,} panel areas")
    gdf["on_building"] = True
    if a.buildings:
        bld = load_polygons(a.buildings)
        hits = gpd.sjoin(gdf, bld[["geometry"]], predicate="intersects", how="left")
        on = hits.groupby(level=0)["index_right"].apply(lambda s: s.notna().any())
        gdf["on_building"] = on.reindex(gdf.index).fillna(False).values
        print(f"On a building: {gdf.on_building.sum():,} ({gdf[gdf.on_building].area_m2.sum():,.0f} m²), "
              f"elsewhere: {(~gdf.on_building).sum():,} ({gdf[~gdf.on_building].area_m2.sum():,.0f} m²)")

    factors = json.loads((PROCESSED_DIR / "area_factors.json").read_text()) \
        if (PROCESSED_DIR / "area_factors.json").exists() else {}
    inv_cos = factors.get("inv_cos_median_tilt", 1.0)
    flm = OUTPUTS_DIR / "flanders_metrics.csv"
    area_ratio = None
    if flm.exists():
        m = pd.read_csv(flm)
        m = m[(m.run == a.run) & (m.city == "all")]
        if len(m):
            area_ratio = float(m.area_ratio.iloc[0])

    roof = gdf[gdf.on_building]
    projected = float(roof.area_m2.sum())
    summary = {"run": a.run, "city": a.city, "threshold": thr, "n_panel_areas_roof": int(len(roof)),
               "projected_m2_roof": projected, "projected_m2_all": float(gdf.area_m2.sum()),
               "real_m2_est": projected * inv_cos, "inv_cos_tilt": inv_cos,
               "flemish_area_ratio": area_ratio,
               "bias_corrected_m2": projected / area_ratio if area_ratio else None,
               "municipality_km2": muni_km2,
               "projected_m2_per_km2": projected / muni_km2 if muni_km2 else None,
               "pixel_m2": GSD ** 2}
    out = city_dir / f"panels_{a.run}.gpkg"
    gdf.to_file(out, driver="GPKG")
    (city_dir / f"summary_{a.run}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRooftop panel area in {a.city}: projected {projected:,.0f} m² | "
          f"real (÷cos tilt) {projected * inv_cos:,.0f} m²"
          + (f" | bias-corrected {projected / area_ratio:,.0f} m²" if area_ratio else ""))
    print(f"Saved {out} (open in QGIS on top of the orthophoto) and summary_{a.run}.json")


if __name__ == "__main__":
    main()
