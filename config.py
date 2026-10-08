"""
Central configuration for forest-attribute map validation.

Edit ROOT_DIR and SITES to point at your data. Everything else flows from here.

Folder layout assumed (per site):
    <ROOT_DIR>/<site>/
        GeoTiffs_2020/*.tif         # multi-band EO, band 30 = GEDI Cover, band 31 = GEDI RH98
        PG-CBM_055095_2020/AGB/*.tif, Height/*.tif, Cover/*.tif, Stem/*.tif
        StruMPL_055095_2020/   (same four sub-folders)
        External_Rf/*.tif           # 4 bands: 1=Lang H, 2=Hansen C, 3=CCI AGB, 4=GEDI L4B AGB
"""

from __future__ import annotations
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths — EDIT THESE
# -----------------------------------------------------------------------------
ROOT_DIR = Path(r"K:/Reza/Africa_Dry/Test_Sites")          # parent containing one folder per site
OUTPUT_DIR = Path(r"K:/Reza/Africa_Dry/Test_Sites/_validation_outputs")

# List of site folder names under ROOT_DIR. Order is preserved in outputs.
SITES: list[str] = [
    "Ang",
    "Congo",
    "Ghana",
    "Mada",
    "Nam",
    "Niger",
    "Sally",
    "Tanz",
    "Ugan",
    "Zim",
]

MODELS = ["StruMPL"] #, "PG-CBM"

# Years included in temporal analyses. The first entry is also the DEFAULT
# year used by the single-year scripts (01-09), so changing it switches their
# reference year. Order matters for year-over-year change analyses.
YEARS = [2019, 2020, 2021, 2022]
DEFAULT_YEAR = 2020

# Per-model layout. Two supported layouts:
#
#   layout = "subfolder_per_attribute"
#       <site>/<dir>/<attr_subfolder>/*.tif   — exactly one .tif per subfolder
#       'attr_subfolders' maps attribute -> sub-folder name
#
#   layout = "flat_with_filename_pattern"
#       <site>/<dir>/*<token>*.tif            — all attribute tifs in one folder,
#       distinguished by a substring 'token' in the filename
#       'attr_tokens' maps attribute -> substring to look for (case-sensitive)
#
# Multi-year support: the 'dir' string may contain '{year}' which is filled in
# at load time. The PARENT folder name (without year) also uses '{year}'-free
# templating via 'parent_dir' if you nest per-year folders inside a parent.
#
# Folder structure expected (per site):
#   <site>/PG-CBM_055095/PG-CBM_055095_2019/<attr>/*.tif
#   <site>/PG-CBM_055095/PG-CBM_055095_2020/<attr>/*.tif
#   <site>/StruMPL_055095/StruMPL_055095_2019/*_<token>_*.tif
#   ...
MODEL_LAYOUT = {
    "PG-CBM": {
        "parent_dir": "PG-CBM_055095",          # holds per-year sub-folders
        "dir":        "PG-CBM_055095_{year}",   # the per-year folder name
        "layout": "subfolder_per_attribute",
        "attr_subfolders": {
            "AGB":    "AGB",
            "Height": "Height",
            "Cover":  "Cover",
            "Stem":   "Stem",
        },
    },
    "StruMPL": {
        "parent_dir": "StruMPL_055095",
        "dir":        "StruMPL_055095_{year}",
        "layout": "flat_with_filename_pattern",
        # tokens that appear in the .tif filenames, e.g.
        #   LonLat_ALOS_HLSL_GEDI_img_Ang_2020-01-01_Biomass_pred.tif
        "attr_tokens": {
            "AGB":    "Biomass",
            "Height": "Height",
            "Cover":  "Cover",
            "Stem":   "StemDensity",
        },
    },
}


# -----------------------------------------------------------------------------
# External reference stack  (now also holds the GEDI ground-truth bands)
# -----------------------------------------------------------------------------
# Per-year layout:
#   <site>/External_Ref/External_Ref_<YEAR>/*.tif
# Use '{year}' placeholder in both the parent and per-year names if your folder
# convention differs.
EXTERNAL_PARENT_DIR = "External_Ref"
EXTERNAL_DIR        = "External_Ref_{year}"
EXT_BANDS = {           # 1-based band indices within the per-year External tif
    "Lang_Height":   1,
    "Hansen_Cover":  2,
    "CCI_AGB":       3,
    "GEDI_L4B_AGB":  4,
    "GEDI_Cover":    5,   # GEDI canopy cover (ground-truth reference)
    "GEDI_RH98":     6,   # GEDI RH98 canopy height (ground-truth reference)
}

# -----------------------------------------------------------------------------
# Canopy-cover units
# -----------------------------------------------------------------------------
# The input rasters store cover as a FRACTION in [0, 1] and are never modified.
# COVER_UNITS controls how cover is represented in everything the pipelines
# compute and write (metrics, plots, CSVs, residual rasters):
#     "fraction"  -> [0, 1]    (cover arrays used exactly as stored)
#     "percent"   -> [0, 100]  (cover arrays multiplied by 100 at load time)
# The scaling itself is applied once in io_utils.load_site_year, to every array
# named in COVER_KEYS. Re-run the whole pipeline after changing this setting:
# scripts 09 and the aggregate_* scripts read earlier outputs, so mixing
# results from different settings would give inconsistent units.
COVER_UNITS = "percent"          # "fraction" or "percent"

if COVER_UNITS not in ("fraction", "percent"):
    raise ValueError(f"COVER_UNITS must be 'fraction' or 'percent', got {COVER_UNITS!r}")

COVER_SCALE = 100.0 if COVER_UNITS == "percent" else 1.0
COVER_UNIT_LABEL = "%" if COVER_UNITS == "percent" else "fraction"

# Bundle array keys that hold canopy cover (scaled by COVER_SCALE on load).
COVER_KEYS = ["GEDI_Cover", "Hansen_Cover"] + [f"{m}_Cover" for m in MODEL_LAYOUT]

# -----------------------------------------------------------------------------
# Attribute metadata
# -----------------------------------------------------------------------------
ATTRIBUTES = {
    "Height": {
        "unit":      "m",
        "vmin":      0,
        "vmax":      25,
        "ref":       "GEDI_RH98",      # the ground-truth label
        "externals": ["Lang_Height"],
    },
    "Cover": {
        "unit":      COVER_UNIT_LABEL,   # "%" or "fraction" (see COVER_UNITS)
        "vmin":      0,
        "vmax":      1 * COVER_SCALE,
        "ref":       "GEDI_Cover",
        "externals": ["Hansen_Cover"],
    },
    "AGB": {
        "unit":      "Mg/ha",
        "vmin":      0,
        "vmax":      150,
        "ref":       None,             # no ground truth
        "externals": ["CCI_AGB", "GEDI_L4B_AGB"],
    },
    "Stem": {
        "unit":      "stems/ha",
        "vmin":      0,
        "vmax":      1000,
        "ref":       None,
        "externals": [],
    },
}

# -----------------------------------------------------------------------------
# Plot styling — consistent colours across all scripts
# -----------------------------------------------------------------------------
COLOURS = {
    "PG-CBM":       "#1b7837",   # green
    "StruMPL":      "#762a83",   # purple
    "GEDI_RH98":    "#000000",   # black (ground truth)
    "GEDI_Cover":   "#000000",
    "Lang_Height":  "#2166ac",   # blue
    "Hansen_Cover": "#2166ac",
    "CCI_AGB":      "#b2182b",   # red
    "GEDI_L4B_AGB": "#d6604d",   # light red
}

# Sub-sampling for scatter / pair plots (memory & rendering)
MAX_PIXELS_FOR_PLOTS = 50_000
RANDOM_SEED = 42

# Bootstrap iterations for metric CIs
N_BOOTSTRAP = 1_000

# Profile method binning.
# We bin each site's coordinate range into a FIXED NUMBER of bins rather than a
# fixed degree width, so the profile adapts to each site's spatial extent
# (sites differ in size, and a fixed degree width gives too few/too many bins).
PROFILE_N_BINS = 100 #60               # number of bins across the site's lon (or lat) range
# Bins with fewer valid pixels are dropped. With sparse GEDI footprints
# (hundreds-low thousands of valid pixels per site) and ~60 bins, an average
# bin will hold ~5-30 pixels. Set conservatively low so most bins survive;
# the line just gets noisier where pixels are sparse.
PROFILE_MIN_PIX_PER_BIN = 5 #20