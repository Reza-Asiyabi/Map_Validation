"""
Shared I/O utilities for loading per-site forest attribute maps.

A "site bundle" is a dict-of-arrays for one site, with keys like
    {('PG-CBM','Height'), ('StruMPL','Height'), 'GEDI_RH98', 'Lang_Height',
     'lon', 'lat', ...}
All arrays are 2-D numpy float32, NaN for nodata, same shape per site
(we assume pre-aligned grids — verified here, hard-fail otherwise).
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform

from config import (
    ROOT_DIR, MODELS, MODEL_LAYOUT, YEARS, DEFAULT_YEAR,
    EXTERNAL_PARENT_DIR, EXTERNAL_DIR, EXT_BANDS, ATTRIBUTES,
    COVER_SCALE, COVER_KEYS, COVER_UNITS,
)


def _find_single_tif(folder: Path) -> Path:
    """Return the single .tif inside a folder; raise if none / multiple."""
    tifs = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    if len(tifs) == 0:
        raise FileNotFoundError(f"No .tif found in {folder}")
    if len(tifs) > 1:
        raise RuntimeError(
            f"Expected exactly one .tif in {folder}, found {len(tifs)}:\n"
            + "\n".join(f"  {t.name}" for t in tifs)
        )
    return tifs[0]


def _find_tif_by_token(folder: Path, token: str) -> Path:
    """
    Return the single .tif in `folder` whose filename contains `token`.
    Match is case-sensitive — adjust if your filenames are mixed-case.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    candidates = [p for p in (sorted(folder.glob("*.tif"))
                              + sorted(folder.glob("*.tiff")))
                  if token in p.name]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No .tif containing '{token}' found in {folder}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple .tif containing '{token}' in {folder}:\n"
            + "\n".join(f"  {t.name}" for t in candidates)
        )
    return candidates[0]


def _read_band(path: Path, band: int = 1) -> Tuple[np.ndarray, dict]:
    """Read a single band as float32 with nodata → NaN. Returns (array, profile)."""
    with rasterio.open(path) as src:
        if band > src.count:
            raise ValueError(
                f"{path.name} has {src.count} bands; requested band {band}"
            )
        arr = src.read(band, masked=True).astype(np.float32).filled(np.nan)
        profile = src.profile.copy()
        profile["transform"] = src.transform
        profile["height"], profile["width"] = src.height, src.width
        profile["crs"] = src.crs
    return arr, profile


def _verify_shape(arr: np.ndarray, ref_shape: tuple, name: str):
    if arr.shape != ref_shape:
        raise RuntimeError(
            f"Shape mismatch for {name}: got {arr.shape}, expected {ref_shape}.\n"
            f"This module assumes maps are pre-aligned to a common grid per site."
        )


def _rescale_cover(arrays: Dict[str, np.ndarray], site: str, year: int) -> None:
    """
    Multiply every cover array by COVER_SCALE, in place (NaN preserved).
    Inputs are expected as fractions in [0, 1]; a maximum well above 1 means
    the data is probably already in percent, so we warn instead of silently
    scaling it twice.
    """
    for key in COVER_KEYS:
        arr = arrays.get(key)
        if arr is None:
            continue
        if np.isfinite(arr).any():
            mx = float(np.nanmax(arr))
            if mx > 1.5:
                warnings.warn(
                    f"{site} {year} {key}: max={mx:.3g} > 1.5, but cover is "
                    f"expected as a fraction in [0, 1]. Input may already be "
                    f"in percent (COVER_UNITS={COVER_UNITS!r}).",
                    stacklevel=3,
                )
        if COVER_SCALE != 1.0:
            arrays[key] = arr * np.float32(COVER_SCALE)


def load_site_year(site: str, year: int,
                   root_dir: Path = ROOT_DIR) -> Dict:
    """
    Load every layer for one site and one year.

    Returns a dict (the 'bundle'):
        bundle['arrays']     : dict[str, np.ndarray]   2-D float32, NaN for nodata
        bundle['profile']    : rasterio profile (from the first raster loaded)
        bundle['lon']        : 2-D float32 longitude per pixel (true degrees)
        bundle['lat']        : 2-D float32 latitude  per pixel (true degrees)
        bundle['x_native']   : 2-D float32 native CRS x (e.g. metres)
        bundle['y_native']   : 2-D float32 native CRS y
        bundle['crs']        : the raster CRS object
        bundle['site']       : site name
        bundle['year']       : year
    Array keys:
        f"{model}_{attr}"    for model in MODELS, attr in 4 attributes
        "GEDI_RH98", "GEDI_Cover"
        "Lang_Height", "Hansen_Cover", "CCI_AGB", "GEDI_L4B_AGB"
    """
    site_dir = Path(root_dir) / site
    if not site_dir.is_dir():
        raise FileNotFoundError(f"Site folder not found: {site_dir}")

    arrays: Dict[str, np.ndarray] = {}
    profile: dict | None = None
    ref_shape: tuple | None = None

    # ---- Model outputs (per-year folder) -------------------------------------
    for model in MODELS:
        spec = MODEL_LAYOUT[model]
        parent = spec.get("parent_dir", "").format(year=year)
        per_year_dir = spec["dir"].format(year=year)
        mdir = site_dir / parent / per_year_dir if parent else site_dir / per_year_dir
        if not mdir.is_dir():
            raise FileNotFoundError(
                f"Model folder not found for {model} year {year}: {mdir}"
            )

        if spec["layout"] == "subfolder_per_attribute":
            attr_iter = spec["attr_subfolders"].items()
            for attr, subname in attr_iter:
                tif = _find_single_tif(mdir / subname)
                arr, prof = _read_band(tif, band=1)
                key = f"{model}_{attr}"
                if profile is None:
                    profile, ref_shape = prof, arr.shape
                else:
                    _verify_shape(arr, ref_shape, key)
                arrays[key] = arr

        elif spec["layout"] == "flat_with_filename_pattern":
            attr_iter = spec["attr_tokens"].items()
            for attr, token in attr_iter:
                tif = _find_tif_by_token(mdir, token)
                arr, prof = _read_band(tif, band=1)
                key = f"{model}_{attr}"
                if profile is None:
                    profile, ref_shape = prof, arr.shape
                else:
                    _verify_shape(arr, ref_shape, key)
                arrays[key] = arr

        else:
            raise ValueError(
                f"Unknown layout '{spec['layout']}' for model '{model}'. "
                f"Expected 'subfolder_per_attribute' or "
                f"'flat_with_filename_pattern'."
            )

    # ---- External reference stack for this YEAR (incl. GEDI) -----------------
    ext_dir = site_dir / EXTERNAL_PARENT_DIR / EXTERNAL_DIR.format(year=year)
    if not ext_dir.is_dir():
        # Fallback: maybe External_Ref is flat with year-suffixed tifs, try
        # the parent without the per-year subfolder.
        ext_dir = site_dir / EXTERNAL_PARENT_DIR
        if not ext_dir.is_dir():
            raise FileNotFoundError(
                f"External reference folder not found for year {year}: "
                f"{site_dir / EXTERNAL_PARENT_DIR / EXTERNAL_DIR.format(year=year)}"
            )
    ext_tif = _find_single_tif(ext_dir)
    for name, band in EXT_BANDS.items():
        arr, _ = _read_band(ext_tif, band=band)
        _verify_shape(arr, ref_shape, name)
        arrays[name] = arr

    # ---- Canopy cover units ---------------------------------------------------
    # Inputs are fractions in [0, 1]; optionally convert to percent here so
    # every downstream script sees the configured unit (config.COVER_UNITS).
    _rescale_cover(arrays, site, year)

    # ---- lon/lat per pixel (2-D, same shape as rasters) ----------------------
    h, w = ref_shape
    transform = profile["transform"]
    # Pixel-centre column/row index grids
    col_idx, row_idx = np.meshgrid(np.arange(w), np.arange(h))   # both (h, w)
    # Affine maps (col, row) -> (x, y) at pixel CORNER; add 0.5 for centre.
    # These are NATIVE coordinates in the raster's own CRS (could be metres).
    x_native = (transform.c + transform.a * (col_idx + 0.5)
                + transform.b * (row_idx + 0.5)).astype(np.float64)
    y_native = (transform.f + transform.d * (col_idx + 0.5)
                + transform.e * (row_idx + 0.5)).astype(np.float64)

    # Convert to true longitude/latitude (EPSG:4326). If the raster is already
    # geographic this is a near-identity transform; if it's projected (e.g. a
    # UTM grid in metres) this turns eastings/northings into real lon/lat so
    # the profile axes and degree-based labels are correct.
    crs = profile.get("crs")
    if crs is not None and not crs.is_geographic:
        lon_flat, lat_flat = warp_transform(
            crs, "EPSG:4326",
            x_native.ravel().tolist(), y_native.ravel().tolist()
        )
        lon = np.asarray(lon_flat, dtype=np.float32).reshape(h, w)
        lat = np.asarray(lat_flat, dtype=np.float32).reshape(h, w)
    else:
        lon = x_native.astype(np.float32)
        lat = y_native.astype(np.float32)

    return {
        "site":       site,
        "year":       year,
        "arrays":     arrays,
        "profile":    profile,
        "lon":        lon,            # true longitude (degrees)
        "lat":        lat,            # true latitude (degrees)
        "x_native":   x_native.astype(np.float32),   # native CRS x (e.g. metres)
        "y_native":   y_native.astype(np.float32),   # native CRS y
        "crs":        crs,
    }


def load_site(site: str, root_dir: Path = ROOT_DIR) -> Dict:
    """
    Backward-compatible single-year loader: loads the configured DEFAULT_YEAR.
    Existing scripts (01-09) keep working unchanged.
    """
    return load_site_year(site, DEFAULT_YEAR, root_dir=root_dir)


# -----------------------------------------------------------------------------
# Mask helpers
# -----------------------------------------------------------------------------
def gedi_mask(bundle: Dict, attribute: str) -> np.ndarray:
    """
    Boolean 2-D mask of pixels where the GEDI reference for `attribute` is valid.
    Returns None if no GEDI reference exists for that attribute.
    """
    ref = ATTRIBUTES[attribute]["ref"]
    if ref is None:
        return None
    return np.isfinite(bundle["arrays"][ref])


def joint_valid_mask(bundle: Dict, keys: list[str]) -> np.ndarray:
    """Boolean mask: True where ALL listed arrays are finite."""
    m = np.ones_like(bundle["arrays"][keys[0]], dtype=bool)
    for k in keys:
        m &= np.isfinite(bundle["arrays"][k])
    return m


def stack_valid(bundle: Dict, keys: list[str], extra_mask: np.ndarray | None = None
                ) -> np.ndarray:
    """
    Return a (N, len(keys)) array of pixel values where every key is finite
    (and extra_mask is True if provided). Useful for pair plots / metrics.
    """
    m = joint_valid_mask(bundle, keys)
    if extra_mask is not None:
        m &= extra_mask
    return np.column_stack([bundle["arrays"][k][m] for k in keys])


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def density_scatter(ax, x, y, bins: int = 200, cmap: str = "viridis",
                    s: float = 3, sort: bool = True):
    """
    Density-coloured scatter that keeps EVERY point (no subsampling).

    Each point is coloured by the local point density, estimated from a 2-D
    histogram (O(N), scales to millions of points — unlike a Gaussian KDE).
    Dense regions stand out instead of saturating into a solid blob.

    Parameters
    ----------
    ax    : matplotlib Axes to draw on.
    x, y  : 1-D arrays (NaNs are dropped together).
    bins  : 2-D histogram resolution per axis. Higher = finer density detail.
    cmap  : perceptually-uniform colormap ('viridis' light=dense by default).
    s     : point size.
    sort  : draw densest points last so they sit on top (recommended).

    Returns the matplotlib PathCollection (or None if no finite data), so the
    caller can attach a colourbar if desired.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return None

    # Histogram the points; then look up the bin each point falls in.
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Bin index per point (clip to valid range; right-most edge is inclusive).
    ix = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, bins - 1)
    iy = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, bins - 1)
    z = counts[ix, iy]

    # Optionally draw densest last so high-density points are visible on top.
    if sort:
        order = np.argsort(z)
        x, y, z = x[order], y[order], z[order]

    return ax.scatter(x, y, c=z, s=s, cmap=cmap, edgecolor="none",
                      rasterized=True)