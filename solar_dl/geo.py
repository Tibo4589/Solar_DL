"""
Geodata helpers for the Flemish part (steps 10-15). Nothing to fill in.

Three ideas you need to be able to explain:

1. Coordinate reference system (CRS). Flemish data is in Belgian Lambert 72 (EPSG:31370):
   x and y in METRES. One pixel of a georeferenced image (GeoTIFF/JPEG2000) is linked to
   map coordinates by an affine 'transform' (top-left corner + pixel size). That is why
   areas can be computed in m², and why your predictions can be overlaid on any map.

2. Ground sampling distance (GSD). The model was trained on 0.20 m pixels (IGN). The
   Flemish orthophotos are 0.15 m (from 2022) or 0.25 m (up to 2021). A CNN is NOT
   scale-invariant: a panel that is 8 pixels wide in training should be ~8 pixels wide at
   test time. So every Flemish image is resampled to 0.20 m before it goes into the model.
   We do that on the fly with a WarpedVRT ('virtual' resampled view of the file), so we
   never write a resampled copy of a multi-GB orthophoto.

3. Windows. A whole municipality at 0.20 m is billions of pixels: far too big for memory
   or a GPU. We read and predict it in windows of TILE x TILE pixels.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from common import GSD, TILE

LAMBERT72 = "EPSG:31370"


@contextmanager
def open_resampled(path, gsd: float = GSD):
    """Open a georeferenced image as a virtual raster with gsd-metre pixels.

    'average' resampling: when going from 0.15 m to 0.20 m, each new pixel is the average of
    the old pixels it covers (anti-aliasing), rather than an interpolation between two of them."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT

    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        width, height = int((right - left) // gsd), int((top - bottom) // gsd)
        transform = from_origin(left, top, gsd, gsd)
        with WarpedVRT(src, crs=src.crs, transform=transform, width=width, height=height,
                       resampling=Resampling.average) as vrt:
            yield vrt


def describe(path) -> dict:
    import rasterio
    with rasterio.open(path) as src:
        return {"file": str(path), "crs": str(src.crs), "res_m": [round(r, 4) for r in src.res],
                "width": src.width, "height": src.height, "bands": src.count, "dtype": src.dtypes[0],
                "bounds": [round(b, 1) for b in src.bounds],
                "area_km2": round((src.bounds.right - src.bounds.left) * (src.bounds.top - src.bounds.bottom) / 1e6, 2)}


def read_padded(vrt, col_off: int, row_off: int, size: int = TILE) -> np.ndarray:
    """(size, size, 3) uint8 window starting at (col_off, row_off); parts outside the raster are 0."""
    from rasterio.windows import Window
    out = np.zeros((3, size, size), dtype=np.uint8)
    c0, r0 = max(col_off, 0), max(row_off, 0)
    c1, r1 = min(col_off + size, vrt.width), min(row_off + size, vrt.height)
    if c1 > c0 and r1 > r0:
        data = vrt.read([1, 2, 3], window=Window(c0, r0, c1 - c0, r1 - r0))
        if data.dtype != np.uint8:  # e.g. 16-bit imagery: scale to 0-255
            data = np.clip(data / max(float(data.max()), 1) * 255, 0, 255).astype(np.uint8)
        out[:, r0 - row_off:r1 - row_off, c0 - col_off:c1 - col_off] = data
    return out.transpose(1, 2, 0)


def mostly_empty(rgb: np.ndarray, limit: float = 0.5) -> bool:
    """True if more than `limit` of the pixels are pure black or pure white (outside the photo / nodata)."""
    black = (rgb == 0).all(-1)
    white = (rgb == 255).all(-1)
    return (black | white).mean() > limit


def load_polygons(path, crs: str = LAMBERT72, field: str | None = None, value: str | None = None):
    """GeoDataFrame from any vector file (GeoPackage, shapefile, GeoJSON), reprojected to crs.
    field/value keep only the rows where field == value (case-insensitive), e.g. one municipality."""
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        print(f"  WARNING: {path} has no CRS; assuming {crs}")
        gdf = gdf.set_crs(crs)
    gdf = gdf.to_crs(crs)
    if field:
        if field not in gdf.columns:
            raise SystemExit(f"{path} has no column {field!r}. Columns: {list(gdf.columns)}")
        gdf = gdf[gdf[field].astype(str).str.lower() == str(value).lower()]
        if gdf.empty:
            raise SystemExit(f"No row with {field} == {value!r} in {path}")
    return gdf


def window_bounds(vrt, col_off: int, row_off: int, size: int = TILE):
    """(left, bottom, right, top) in map coordinates of a window of the resampled raster."""
    t = vrt.transform
    left, top = t * (col_off, row_off)
    right, bottom = t * (col_off + size, row_off + size)
    return left, bottom, right, top
