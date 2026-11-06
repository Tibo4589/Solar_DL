"""
Step 13 — Map a whole city: sliding-window prediction over the full orthophoto(s).

  python 13_predict_area.py --run unet_final --city leuven      (on a GPU node: slurm/predict_gpu.slurm)

For every orthophoto file listed by step 10 (outputs/flanders/<city>/ortho_files.json):
  - view it at 0.20 m/pixel (geo.open_resampled),
  - cut it in TILE x TILE windows that OVERLAP by 2 x MARGIN pixels,
  - predict each window, but keep only its CENTRE (step x step pixels).
Why the overlap: near the border of a window the network sees only half of the context
(a panel cut in two, zero-padding). Its predictions there are worse. Keeping only the
centre of every window means every pixel is predicted with enough context around it.

The output is a GeoTIFF with the same georeference as the photo, holding the panel
probability x 255 (uint8, to keep files small). Open it in QGIS on top of the photo.

Requires: 10_prepare_flanders.py (for ortho_files.json), a final run (+ step 08)
Output:   outputs/flanders/<city>/prob_<run>_<i>.tif
Next:     14_area_statistics.py
"""

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from common import OUTPUTS_DIR, TILE, ensure_dirs, get_device, load_run
from geo import mostly_empty, open_resampled, read_padded

MARGIN = 48             # pixels of context thrown away on each side of a window
STEP = TILE - 2 * MARGIN
BATCH = 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--city", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    ensure_dirs()
    import rasterio
    from rasterio.windows import Window
    device = get_device(a.device)
    model, _, _ = load_run(a.run, device)
    city_dir = OUTPUTS_DIR / "flanders" / a.city
    files = json.loads((city_dir / "ortho_files.json").read_text())

    for fi, f in enumerate(files):
        out_path = city_dir / f"prob_{a.run}_{fi}.tif"
        with open_resampled(f) as vrt:
            profile = {"driver": "GTiff", "width": vrt.width, "height": vrt.height, "count": 1, "dtype": "uint8",
                       "crs": vrt.crs, "transform": vrt.transform, "tiled": True, "blockxsize": 256,
                       "blockysize": 256, "compress": "deflate", "BIGTIFF": "IF_SAFER"}
            origins = [(c, r) for r in range(0, vrt.height, STEP) for c in range(0, vrt.width, STEP)]
            print(f"{f}: {vrt.width:,} x {vrt.height:,} px at 0.20 m -> {len(origins):,} windows")
            with rasterio.open(out_path, "w", **profile) as dst:
                for b in tqdm(range(0, len(origins), BATCH), desc=f"{a.city} file {fi}", unit="batch"):
                    batch = origins[b:b + BATCH]
                    tiles = [read_padded(vrt, c - MARGIN, r - MARGIN) for c, r in batch]
                    keep = [i for i, t in enumerate(tiles) if not mostly_empty(t, 0.95)]
                    probs = np.zeros((len(batch), TILE, TILE), np.float32)
                    if keep:
                        x = torch.from_numpy(np.stack([tiles[i] for i in keep])).permute(0, 3, 1, 2).float() / 255
                        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                             enabled=device.type == "cuda"):
                            pr = torch.sigmoid(model(x.to(device)).float())[:, 0].cpu().numpy()
                        probs[keep] = pr
                    for (c, r), pr in zip(batch, probs):
                        w, h = min(STEP, vrt.width - c), min(STEP, vrt.height - r)
                        centre = pr[MARGIN:MARGIN + h, MARGIN:MARGIN + w]
                        dst.write((centre * 255).round().astype(np.uint8)[None], window=Window(c, r, w, h))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
