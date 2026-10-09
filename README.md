# Forest structure map validation

Validation pipelines for forest-structure maps from two models:

- **StruMPL** — <https://arxiv.org/abs/2605.19931>
- **PG-CBM** — <https://arxiv.org/abs/2601.10562>

The suite checks four attributes (**AGBD**, **Height**, **Cover**, **Stem**
density) over 10 African sites and four years (2019–2022). Height and Cover
are validated against GEDI (RH98 and canopy cover). AGBD is compared with
external products (CCI, GEDI L4B), which gives agreement rather than accuracy.
Stem density is checked for consistency only. Each script reads GeoTIFFs,
computes metrics, and writes figures and CSVs. The inputs are never modified.

The code is written for these two models. Other models or sites need config
edits (see the guide).

## Quick start

```bash
pip install numpy pandas matplotlib rasterio scipy
pip install statsmodels     # optional, only for the mixed models in script 13
```

1. Edit `config.py`: `ROOT_DIR`, `OUTPUT_DIR`, `SITES`, `YEARS`, `DEFAULT_YEAR`.
   Set `MODELS = ["PG-CBM", "StruMPL"]` to run the full suite.
2. Run any script: `python 01_quantitative_metrics.py`. Each writes to
   `OUTPUT_DIR/<NN_name>/`. Script 09 needs the output of 01.

Expected layout, per site:
`PG-CBM_055095/PG-CBM_055095_<year>/<attr>/*.tif`,
`StruMPL_055095/StruMPL_055095_<year>/*<token>*.tif`, and
`External_Ref/External_Ref_<year>/*.tif` (6 bands: Lang height, Hansen cover,
CCI AGBD, GEDI L4B AGBD, GEDI cover, GEDI RH98). All layers of a site must
already share one grid. Nothing is reprojected or resampled.

## Switches in `config.py`

- **`COVER_UNITS`** is `"percent"` (default, cover 0–100) or `"fraction"`
  (0–1). The input rasters hold fractions; cover is converted once, when it is
  loaded. Re-run the whole pipeline after changing it.
- **`DEFAULT_YEAR`** is the year used by single-year scripts 01–09.
- **`MODELS`**, **`YEARS`** and **`SITES`** control which models, years and
  sites are processed.

## Scripts

| Script | Purpose |
|---|---|
| `01_quantitative_metrics` | Bias, MAE, RMSE, rRMSE, R², r vs GEDI (Height, Cover), with cross-site bootstrap CIs |
| `02_longitude_profiles` | Longitude and latitude profiles of each model vs GEDI |
| `03_attribute_pairs` | Pair plots of the attributes within each model |
| `04_source_pairs` | Pair plots of the sources (models, externals, GEDI) for each attribute |
| `05_distributions` | CDFs, Q-Q plots and KS tests (saturation check) |
| `06_spatial_residuals` | Residual maps and rasters, Moran's I |
| `07_stratified_metrics` | Errors by reference-value bin and by cover class |
| `08_allometric_consistency` | AGBD response to Height, Cover and Stem, with OLS fits |
| `09_summary_tables` | Summary tables and a paired Wilcoxon model comparison |
| `10_temporal_residual_profiles` | Residual profiles and RMSE/bias trajectories by year |
| `11_year_over_year_change_agreement` | Does Δmodel track ΔGEDI? |
| `12_temporal_consistency_spatial` | Per-year Moran's I, residual-map similarity, z-anomaly maps |
| `13_mixed_model_and_allometry` | Mixed models (`model*year + (1\|site)`), allometric stability |
| `14_height_cover_allometry` | Height–Cover coupling, model vs GEDI |
| `15_export_per_pixel_csv` | Wide per-pixel CSV per site (`--site`, `--gzip`) |
| `aggregate_01…04_*` | Cross-year tables and pooled figures (see the guide) |

Shared code: `config.py` (all settings), `io_utils.py` (loading, masks,
cover scaling), `metrics.py` (error metrics, bootstrap).

## Design notes

- **Sites are the unit of replication.** Bootstrap CIs resample sites, not
  pixels, because pixels are spatially autocorrelated.
- **GEDI is a sparse raster** on the same pixel grid as the maps, handled as a
  validity mask.
- **AGBD and Stem have no ground truth.** Treat their metrics as agreement or
  plausibility, not accuracy.
- **Naming:** biomass density is `AGBD` in all code and outputs. The PG-CBM
  input folder on disk is still `AGB`, and the StruMPL files are matched by
  `Biomass`. Both are mapped in `config.py`.

## Documentation

[`VALIDATION_GUIDE.md`](VALIDATION_GUIDE.md) is the full reference. It covers
every script's logic and formulas, the output files, how to interpret each
result, the multi-year aggregation strategy, cover units, how to extend the
suite, and troubleshooting.
