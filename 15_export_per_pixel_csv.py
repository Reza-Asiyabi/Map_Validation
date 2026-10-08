"""
15 — Per-pixel wide CSV export, one CSV per site.

For each test site, build a single wide-format CSV where every row is one
pixel and every column carries that pixel's value from one of the input
GeoTIFFs across years and sources.

Column schema (in order):
    site_id, row, col,
    GEDI_Height_<year>     (one per year, from External_Ref bands)
    GEDI_Cover_<year>      (one per year)
    GEDI_AGBD_<year>        (one per year; this is the GEDI_L4B_AGBD layer)
    Lang_Height_<year>     (one per year)
    Hansen_Cover_<year>    (one per year)
    CCI_AGBD_<year>         (one per year)
    PG-CBM_<Attribute>_<year>    (Attribute in {AGBD,Height,Cover,Stem,WoodDensity},
                                  one per year)
    StruMPL_<Attribute>_<year>   (same 5 × 4 years)

With 4 years × 5 attributes × 2 models + 4 years × 6 per-year external/GEDI
sources + 3 identifier columns ≈ 67 columns per pixel.

Pixel filtering:
    Drop pixels where GEDI_Height AND GEDI_Cover are NaN across ALL years
    (i.e. that pixel is never sampled by GEDI in any of the four years).
    This is the most permissive drop rule — it keeps any pixel that GEDI
    touched at least once, while removing the vast majority of empty pixels
    that GEDI never observed.

Notes:
- Pixel location is encoded as integer (row, col) indices within the raster
  grid. They are consistent across all years for a given site because every
  layer is on the same per-site grid.
- The script can run for one site (--site SITE) or all configured sites.
- Use --gzip to write CSVs gzip-compressed (typically 5-10x smaller).

Usage:
    python 15_export_per_pixel_csv.py                     # all sites in config.SITES
    python 15_export_per_pixel_csv.py --site Ang          # just one site
    python 15_export_per_pixel_csv.py --gzip              # gzip output
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (SITES, YEARS, MODELS, ATTRIBUTES, OUTPUT_DIR)
from io_utils import load_site_year


OUT = OUTPUT_DIR / "15_per_pixel_csv"
OUT.mkdir(parents=True, exist_ok=True)


# All attributes we want exported per model.
ATTRIBUTES_TO_EXPORT = ["AGBD", "Height", "Cover", "Stem"]

# Per-year external/GEDI sources we include. Each maps from its bundle key
# (left) to the CSV column prefix (right). The year suffix is appended later.
PER_YEAR_EXTERNAL_SOURCES = {
    "GEDI_RH98":     "GEDI_Height",
    "GEDI_Cover":    "GEDI_Cover",
    "GEDI_L4B_AGBD":  "GEDI_AGBD",
    "Lang_Height":   "Lang_Height",
    "Hansen_Cover":  "Hansen_Cover",
    "CCI_AGBD":       "CCI_AGBD",
}


def _build_site_dataframe(site: str) -> pd.DataFrame:
    """Load every year's bundle for one site, assemble into a wide DataFrame.

    Returns:
        DataFrame with one row per kept pixel and the full column schema.
        Pixels where GEDI_Height and GEDI_Cover are NaN in EVERY year are
        dropped before returning.
    """
    # ---- Load all year bundles up front -----------------------------------
    bundles: dict[int, dict] = {}
    ref_shape: tuple | None = None
    for year in YEARS:
        try:
            b = load_site_year(site, year)
        except FileNotFoundError as e:
            print(f"  [WARN] {site} {year}: {e}  — skipping this year")
            continue
        if ref_shape is None:
            ref_shape = next(iter(b["arrays"].values())).shape
        bundles[year] = b
    if not bundles:
        raise RuntimeError(f"No year loaded for site {site}")

    H, W = ref_shape
    n_pixels_total = H * W
    print(f"  {site}: grid {H}x{W} = {n_pixels_total:,} pixels, "
          f"{len(bundles)} year(s) loaded")

    # ---- Assemble columns column-by-column (avoids row-wise appending) ----
    # Each column is a flat 1-D array of length H*W. NaN where the source is
    # absent for that year. We'll keep them as float32 to halve memory vs
    # default float64; pandas handles NaN in float32 fine.
    columns: dict[str, np.ndarray] = {}

    # Identifier columns
    row_idx, col_idx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    columns["site_id"] = np.full(n_pixels_total, site, dtype=object)
    columns["row"]     = row_idx.ravel().astype(np.int32)
    columns["col"]     = col_idx.ravel().astype(np.int32)

    # Per-year external/GEDI source columns
    for src_key, col_prefix in PER_YEAR_EXTERNAL_SOURCES.items():
        for year in YEARS:
            col_name = f"{col_prefix}_{year}"
            if year in bundles and src_key in bundles[year]["arrays"]:
                columns[col_name] = (bundles[year]["arrays"][src_key]
                                     .ravel().astype(np.float32))
            else:
                columns[col_name] = np.full(n_pixels_total, np.nan,
                                            dtype=np.float32)

    # Per-year, per-model attribute columns
    for model in MODELS:
        for attr in ATTRIBUTES_TO_EXPORT:
            for year in YEARS:
                col_name = f"{model}_{attr}_{year}"
                bundle_key = f"{model}_{attr}"
                if year in bundles and bundle_key in bundles[year]["arrays"]:
                    columns[col_name] = (bundles[year]["arrays"][bundle_key]
                                         .ravel().astype(np.float32))
                else:
                    columns[col_name] = np.full(n_pixels_total, np.nan,
                                                dtype=np.float32)

    df = pd.DataFrame(columns)

    # ---- Pixel filter: drop rows where GEDI_Height AND GEDI_Cover are NaN
    # across ALL years (i.e. GEDI never observed this pixel) -----------------
    gedi_height_cols = [c for c in df.columns if c.startswith("GEDI_Height_")]
    gedi_cover_cols  = [c for c in df.columns if c.startswith("GEDI_Cover_")]
    # 'any year valid' across each layer
    height_any = df[gedi_height_cols].notna().any(axis=1)
    cover_any  = df[gedi_cover_cols].notna().any(axis=1)
    keep = height_any | cover_any
    n_before = len(df)
    df = df[keep].reset_index(drop=True)
    n_after = len(df)
    pct_kept = 100.0 * n_after / n_before if n_before else 0
    print(f"  {site}: kept {n_after:,}/{n_before:,} pixels "
          f"({pct_kept:.1f}%) — dropped pixels with no GEDI in any year")

    return df


def export_site(site: str, gzip: bool = False) -> Path:
    """Build and write the per-pixel CSV for one site. Returns the output path."""
    df = _build_site_dataframe(site)
    ext = ".csv.gz" if gzip else ".csv"
    out_path = OUT / f"{site}_per_pixel{ext}"
    df.to_csv(out_path, index=False,
              compression="gzip" if gzip else None,
              float_format="%.6g")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  {site}: wrote {out_path.name}  "
          f"({len(df):,} rows × {len(df.columns)} cols, {size_mb:.1f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default=None,
                        help="Export only this site. Default: every site in config.SITES.")
    parser.add_argument("--gzip", action="store_true",
                        help="Write CSVs as .csv.gz (typically 5-10x smaller).")
    args = parser.parse_args()

    sites = [args.site] if args.site else SITES
    if args.site and args.site not in SITES:
        print(f"[15] WARNING: '{args.site}' is not in config.SITES; "
              f"attempting anyway")

    print(f"[15] exporting {len(sites)} site(s) → {OUT}")
    for site in sites:
        print(f"[15] {site} ...")
        try:
            # export_site(site, gzip=args.gzip)
            export_site(site, gzip=True)
        except Exception as e:
            print(f"  [ERROR] {site}: {type(e).__name__}: {e}")
    print("[15] done.")


if __name__ == "__main__":
    main()
