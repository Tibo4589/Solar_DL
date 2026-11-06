"""
Step 02 — Explore BDAPPV before training anything (like step 02 of lab 1).

Questions this step answers (write the answers down, you will be asked about them):
  1. Class balance: what fraction of all pixels is 'panel'? -> why accuracy is useless,
     and why we add a Dice term to the loss.
  2. How big is a panel in pixels and in m²? (at 0.20 m/px one pixel = 0.04 m²)
  3. Are the images WITHOUT a mask really empty? Look at plots/bdappv_unmasked_examples.png.
     If many of them DO show panels, they are unlabelled, not negatives -> keep
     INCLUDE_UNMASKED = False in common.py.
  4. Projected vs real area: the photo shows a tilted roof from above, so the mask area is
     the HORIZONTAL projection of the panel. metadata.csv has the real panel 'surface'.
     Their ratio tells you how much a mask underestimates real panel area (~cos(tilt)).
  5. m² per kWp: needed in step 14 to turn installed POWER (what Fluvius publishes)
     into an expected panel AREA (what we measure).

Requires: 01_check_data.py
Output:   plots/bdappv_examples.png, plots/bdappv_unmasked_examples.png,
          plots/bdappv_statistics.png, processed/area_factors.json
Next:     03_make_split.py
"""

import json

import numpy as np
import pandas as pd
from scipy import ndimage

from common import (GSD, METADATA_CSV, PLOTS_DIR, PROCESSED_DIR, SEED, ensure_dirs, overlay,
                    read_mask, read_rgb)

PIXEL_M2 = GSD * GSD
N_EXAMPLES = 12


def grid(images, titles, out_name, ncols=6):
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im, t in zip(np.atleast_1d(axes).ravel(), images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / out_name, dpi=130)
    plt.close(fig)
    return PLOTS_DIR / out_name


def main():
    ensure_dirs()
    rng = np.random.default_rng(SEED)
    inv = pd.read_csv(PROCESSED_DIR / "inventory.csv", dtype={"stem": str}).fillna("")
    masked = inv[inv.mask_path != ""].reset_index(drop=True)
    unmasked = inv[inv.mask_path == ""].reset_index(drop=True)

    # --- 1-2: class balance and panel sizes --------------------------------
    fractions, n_components, comp_areas, proj_area = [], [], [], {}
    for r in masked.itertuples():
        m = read_mask(r.mask_path)
        fractions.append(m.mean())
        lab, n = ndimage.label(m)
        n_components.append(n)
        if n:
            comp_areas.extend(np.bincount(lab.ravel())[1:] * PIXEL_M2)
        if n == 1:  # one connected panel area: safe to compare with ONE installation's metadata
            proj_area[r.stem] = m.sum() * PIXEL_M2
    fractions = np.array(fractions)
    print(f"Masked images: {len(masked):,}")
    print(f"  panel pixels per masked image: median {np.median(fractions) * 100:.2f}%, mean {fractions.mean() * 100:.2f}%")
    print(f"  -> over ALL masked images, {fractions.mean() * 100:.2f}% of the pixels are panel.")
    print(f"     A model that predicts 'no panel' everywhere is {100 - fractions.mean() * 100:.1f}% 'accurate'"
          " and completely useless: we report IoU, precision, recall and F1 instead.")
    print(f"  connected panel areas per image: median {np.median(n_components):.0f}, max {max(n_components)}")
    print(f"  area of one connected panel area: median {np.median(comp_areas):.1f} m² "
          f"(= {np.median(comp_areas) / PIXEL_M2:.0f} pixels)")
    print(f"  masks with zero panel pixels: {(fractions == 0).sum()}")

    # --- examples ----------------------------------------------------------
    pick = rng.choice(len(masked), size=min(N_EXAMPLES, len(masked)), replace=False)
    ims = [overlay(read_rgb(masked.img_path[i]), read_mask(masked.mask_path[i])) for i in pick]
    print(f"\nSaved {grid(ims, [masked.stem[i] for i in pick], 'bdappv_examples.png')}  (red = mask)")
    if len(unmasked):
        pick = rng.choice(len(unmasked), size=min(N_EXAMPLES, len(unmasked)), replace=False)
        ims = [read_rgb(unmasked.img_path[i]) for i in pick]
        print(f"Saved {grid(ims, [unmasked.stem[i] for i in pick], 'bdappv_unmasked_examples.png')}")
        print("  -> Do these images contain panels? Count them. That decides INCLUDE_UNMASKED in common.py.")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    axes[0].hist(fractions * 100, bins=50)
    axes[0].set_xlabel("panel pixels in the image (%)")
    axes[1].hist(n_components, bins=range(0, max(n_components) + 2))
    axes[1].set_xlabel("connected panel areas per image")
    axes[2].hist(np.clip(comp_areas, 0, 150), bins=60)
    axes[2].set_xlabel("area of one panel area (m², clipped at 150)")
    for ax in axes:
        ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bdappv_statistics.png", dpi=150)
    plt.close(fig)
    print(f"Saved {PLOTS_DIR / 'bdappv_statistics.png'}")

    # --- 4-5: area factors from the metadata --------------------------------
    factors = {"pixel_m2": PIXEL_M2}
    if METADATA_CSV.exists():
        meta = pd.read_csv(METADATA_CSV, dtype=str)
        meta["stem"] = meta.get("identifiant", pd.Series(dtype=str)).astype(str).str.replace(
            r"\.(png|jpg)$", "", regex=True)
        for col in ("surface", "kWp", "tilt"):
            meta[col] = pd.to_numeric(meta.get(col), errors="coerce")
        ok = meta[(meta.surface > 0) & (meta.kWp > 0)]
        if len(ok):
            ratio = ok.surface / ok.kWp
            factors["m2_per_kwp_bdappv_median"] = float(ratio.median())
            factors["m2_per_kwp_bdappv_iqr"] = [float(ratio.quantile(0.25)), float(ratio.quantile(0.75))]
            print(f"\nm² of panel per kWp (BDAPPV metadata, n={len(ok):,}): median {ratio.median():.2f}, "
                  f"IQR {ratio.quantile(0.25):.2f}-{ratio.quantile(0.75):.2f}")
            print("  (French installations, many of them older: older panels need more m² per kWp than new ones)")
        tilt = meta.tilt[(meta.tilt >= 0) & (meta.tilt < 80)]
        if len(tilt):
            factors["median_tilt_deg"] = float(tilt.median())
            factors["inv_cos_median_tilt"] = float(1 / np.cos(np.radians(tilt.median())))
            print(f"Median tilt {tilt.median():.0f}° -> real area ≈ projected area x {factors['inv_cos_median_tilt']:.2f}")
        linked = meta[meta.stem.isin(proj_area) & (meta.surface > 0)]
        if len(linked) > 10:
            r = np.array([proj_area[s] for s in linked.stem]) / linked.surface.values
            factors["projected_over_surface_median"] = float(np.median(r))
            print(f"Mask area / metadata surface (single-area masks, n={len(linked):,}): median {np.median(r):.2f}")
            print("  < 1 because of the tilt (and annotation/registration error); you can quote this on the poster.")
    out = PROCESSED_DIR / "area_factors.json"
    out.write_text(json.dumps(factors, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
