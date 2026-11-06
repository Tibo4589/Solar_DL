"""
Step 16 (optional, lowest priority) — Suitable roofs WITHOUT panels: the remaining potential.

The Zonnekaart (VEKA, based on 2013-2015 laser measurements) scores every roof for solar
suitability, but it does NOT know which roofs already have panels. Our map does. Combined:
'suitable roof + no panel detected' = where new panels could go.

Input: a polygon layer of roofs (or roof parts) with a suitability class, e.g. Zonnekaart
data if you can get it as GIS data, in any format geopandas reads. If you can only get the
Zonnekaart as a web map (WMS), make this map in QGIS instead (guide, step 6.4); and if you
have no suitability data at all, you can pass building footprints (GRB) without
--class-field: then every roof counts as a candidate and you get 'roofs without panels'.

  python 16_zonnekaart_overlay.py --run unet_final --city leuven --roofs zonnekaart_leuven.gpkg \
      --class-field geschiktheid --suitable ideaal bruikbaar \
      --boundary data/flanders/boundaries/gemeenten.gpkg --boundary-field NAAM --boundary-value Leuven

Output: outputs/flanders/<city>/potential_<run>.gpkg (every roof with: suitable, has_panel),
        plots/potential_<city>.png (map of one neighbourhood), and the potential in m².
"""

import argparse

import numpy as np

from common import OUTPUTS_DIR, PLOTS_DIR, ensure_dirs
from geo import load_polygons, open_resampled

MIN_OVERLAP_M2 = 2.0   # a roof 'has a panel' if at least this much detected panel area lies on it


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--roofs", required=True)
    p.add_argument("--class-field", default=None)
    p.add_argument("--suitable", nargs="+", default=None, help="class values that count as suitable")
    p.add_argument("--boundary", default=None, help="same municipality boundary as in step 14")
    p.add_argument("--boundary-field", default=None)
    p.add_argument("--boundary-value", default=None)
    p.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="area (Lambert 72 m) for the map figure; default: 400 m around the most panels")
    a = p.parse_args()
    ensure_dirs()
    import json

    import geopandas as gpd
    city_dir = OUTPUTS_DIR / "flanders" / a.city
    panels = gpd.read_file(city_dir / f"panels_{a.run}.gpkg")
    roofs = load_polygons(a.roofs).reset_index(drop=True)
    if a.boundary:  # panels were clipped to the municipality in step 14: do the same with the roofs,
        b = load_polygons(a.boundary, field=a.boundary_field, value=a.boundary_value)  # or roofs outside it
        inside = roofs.representative_point().within(b.geometry.union_all())    # would all look 'free'
        roofs = roofs[inside.values].reset_index(drop=True)
    roofs["roof_m2"] = roofs.area
    if a.class_field:
        roofs["suitable"] = roofs[a.class_field].astype(str).str.lower().isin([s.lower() for s in a.suitable])
    else:
        roofs["suitable"] = True

    inter = gpd.overlay(roofs[["geometry"]].reset_index(), panels[["geometry"]], how="intersection")
    covered = inter.assign(m2=inter.area).groupby("index").m2.sum()
    roofs["panel_m2"] = covered.reindex(roofs.index).fillna(0).values
    roofs["has_panel"] = roofs.panel_m2 >= MIN_OVERLAP_M2
    roofs.to_file(city_dir / f"potential_{a.run}.gpkg", driver="GPKG")

    s = roofs[roofs.suitable]
    free = s[~s.has_panel]
    print(f"Roofs: {len(roofs):,}; suitable: {len(s):,} ({s.roof_m2.sum():,.0f} m²)")
    print(f"Suitable roofs WITH detected panels: {s.has_panel.sum():,} ({s.has_panel.mean() * 100:.1f}%)")
    print(f"Suitable roofs WITHOUT panels      : {len(free):,} ({free.roof_m2.sum():,.0f} m² of roof) "
          f"<- the remaining potential")
    print("Caveats: Zonnekaart roofs are from 2013-2015 (newer buildings missing); a whole 'suitable' roof is "
          "not all usable (windows, edges); a missed panel (false negative) looks like 'potential'.")
    (city_dir / f"potential_{a.run}.json").write_text(json.dumps({
        "roofs": int(len(roofs)), "suitable": int(len(s)), "suitable_with_panel": int(s.has_panel.sum()),
        "suitable_free": int(len(free)), "suitable_free_roof_m2": float(free.roof_m2.sum())}, indent=2))

    # --- map of one neighbourhood ------------------------------------------
    if a.bbox:
        x0, y0, x1, y1 = a.bbox
    else:
        c = panels.geometry.centroid
        x, y = np.median(c.x), np.median(c.y)
        x0, y0, x1, y1 = x - 200, y - 200, x + 200, y + 200
    import matplotlib.pyplot as plt
    from rasterio.windows import from_bounds
    files = json.loads((city_dir / "ortho_files.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 8))
    for f in files:
        with open_resampled(f) as vrt:
            l, b, r, t = vrt.bounds
            if r <= x0 or l >= x1 or t <= y0 or b >= y1:
                continue
            win = from_bounds(max(x0, l), max(y0, b), min(x1, r), min(y1, t), vrt.transform).round_offsets().round_lengths()
            img = vrt.read([1, 2, 3], window=win).transpose(1, 2, 0)
            wb = vrt.window_bounds(win)
            ax.imshow(img, extent=(wb[0], wb[2], wb[1], wb[3]))
    clip = lambda g: gpd.clip(g, (x0, y0, x1, y1))
    # colours chosen to stay visible on red roof tiles and green gardens
    clip(roofs[roofs.suitable & ~roofs.has_panel]).plot(ax=ax, facecolor="yellow", alpha=0.35, edgecolor="orange")
    clip(roofs[roofs.suitable & roofs.has_panel]).plot(ax=ax, facecolor="none", edgecolor="cyan", lw=1)
    clip(panels).plot(ax=ax, facecolor="magenta", alpha=0.7, edgecolor="none")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(f"{a.city}: detected panels (magenta), suitable roofs with panels (cyan outline),\n"
                 f"suitable roofs without panels = potential (yellow)", fontsize=9)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"potential_{a.city}.png", dpi=170)
    plt.close(fig)
    print(f"Saved {PLOTS_DIR / f'potential_{a.city}.png'} and {city_dir / f'potential_{a.run}.gpkg'}")


if __name__ == "__main__":
    main()
