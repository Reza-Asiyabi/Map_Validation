# Forest attribute map validation (PG-CBM & StruMPL)

Python scripts to validate two model outputs (PG-CBM, StruMPL) for four forest
attributes (AGBD, Height, Cover, Stem Density) across 10 African sites,
against GEDI ground truth and external reference products (Lang Height,
Hansen Cover, CCI AGBD, GEDI L4B AGBD).

## Quick start

1. **Edit `config.py`** — set `ROOT_DIR`, `OUTPUT_DIR`, and the list of `SITES`.
2. **Install deps**:
   ```
   pip install numpy pandas matplotlib rasterio scipy
   ```
3. **Run scripts in numbered order** (they can run independently, but `09`
   reads `01`'s output):
   ```
   python 01_quantitative_metrics.py
   python 02_longitude_profiles.py
   python 03_attribute_pairs.py
   python 04_source_pairs.py
   python 05_distributions.py
   python 06_spatial_residuals.py
   python 07_stratified_metrics.py
   python 08_allometric_consistency.py
   python 09_summary_tables.py
   ```

Each script writes to `OUTPUT_DIR/<NN_method_name>/`.

## What each script does

| # | Script | Question it answers |
|---|---|---|
| 01 | quantitative_metrics | Per-site & pooled bias/MAE/RMSE/rRMSE/R²/r vs GEDI for Height & Cover, with cross-site bootstrap CIs |
| 02 | longitude_profiles | Does each model's spatial mean track GEDI's along lon and lat (the GEE profile workflow, in Python)? |
| 03 | attribute_pairs | Within each model, do the four attributes correlate sensibly (the ggpairs equivalent)? |
| 04 | source_pairs | For each attribute, how do PG-CBM / StruMPL / external sources compare pixel-by-pixel? |
| 05 | distributions | Are the value distributions consistent? Q-Q plots reveal saturation; KS gives a number |
| 06 | spatial_residuals | Are residuals spatially clustered? (Moran's I) — and what do the residual maps look like? |
| 07 | stratified_metrics | Where does each model fail? RMSE/bias binned by reference value (saturation diagnostic) and cover class |
| 08 | allometric_consistency | Is each model internally consistent? AGBD vs H/C/S response curves + OLS fits |
| 09 | summary_tables | Compact tables for paper / head-to-head PG-CBM vs StruMPL with paired Wilcoxon |

## Outputs

```
OUTPUT_DIR/
├── 01_quantitative_metrics/
│   ├── per_site_metrics.csv
│   ├── pooled_metrics.csv
│   ├── site_bootstrap_ci.csv
│   └── summary_Height.png, summary_Cover.png
├── 02_profiles/
│   ├── <site>_<attr>_lon.png, _lat.png
│   ├── <site>_<attr>_<axis>_profile.csv
│   └── profile_summary_metrics.csv
├── 03_attribute_pairs/
│   ├── <site>_<model>_pairs.png
│   └── all_sites_<model>_corr_summary.csv
├── 04_source_pairs/...
├── 05_distributions/
│   ├── <site>_<attr>_hist_qq.png
│   └── ks_test_results.csv
├── 06_spatial_residuals/
│   ├── <site>_<attr>_<model>_residual.tif & .png
│   └── morans_i.csv
├── 07_stratified/
│   ├── stratified_by_refbin_<attr>.csv
│   ├── stratified_by_cover_<attr>.csv
│   └── saturation_<attr>.png
├── 08_allometric/
│   ├── allometric_curves_<predictor>.png
│   ├── allometric_ols_per_site.csv
│   └── allometric_curves_per_site/<site>.png
└── 09_summary/
    ├── summary_<attr>.csv
    ├── model_comparison.csv
    └── overview.txt
```

## Design notes

- **Cover units are switchable.** Input rasters hold canopy cover as a fraction
  [0, 1] and are never modified. `COVER_UNITS` in `config.py` (`"fraction"` or
  `"percent"`) controls how cover appears in every output (metrics, plots,
  CSVs, residual rasters). Scaling is applied once in `io_utils.load_site_year`;
  cover bins/classes in scripts 07, 08 and 14 follow `COVER_SCALE`. Unit-free
  metrics (r, R², rRMSE) are unaffected. Re-run the full pipeline after
  changing it, since scripts 09 and `aggregate_*` read earlier outputs.

- **No reprojection or resampling.** All maps per site are assumed already on
  a common grid. The I/O layer asserts shape equality and fails loudly otherwise.
- **GEDI pixels not footprints.** The user confirmed GEDI footprints are already
  rasterised onto the maps' pixel grid, so GEDI is treated like any other raster
  with a sparse valid mask. The mask is recomputed per attribute (`gedi_mask`).
- **Bootstrap CIs are across sites**, not across pixels. Pixel-level bootstrap
  ignores spatial autocorrelation and dramatically understates uncertainty.
  With 10 sites you get usable but wide CIs — that's an honest picture.
- **For AGBD**, no ground truth exists. Metrics against CCI / GEDI L4B should be
  read as *inter-product agreement*, not error. The summary tables label these
  appropriately.
- **For Stem density**, no external reference. Validation relies on
  distribution shape, attribute pairs (script 03), allometric consistency
  (script 08), and PG-CBM ↔ StruMPL agreement (script 04).
- **Ecoregion stratification** is not implemented because you don't have
  ecoregion rasters in the site folders. If you add them later (RESOLVE
  Ecoregions, same as the GEE app), `07_stratified_metrics.py` is the right
  place to wire them in.
