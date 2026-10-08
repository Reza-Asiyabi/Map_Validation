# Forest Attribute Map Validation — Code & Interpretation Guide

This document explains every script in the validation suite (`01`–`15`), the
shared modules they rely on, the formulas and logic behind each method, and —
most importantly — **how to read the outputs**. For each method you'll find:
what it computes, the maths, what files it writes, and a "how to interpret"
section describing what a good versus a problematic result looks like.

The suite validates two models (**PG-CBM** and **StruMPL**) for five forest
attributes — **AGBD** (above-ground biomass), **Height** (canopy height),
**Cover** (canopy cover), **Stem Density**, and **Wood Density** — across 10
African sites and 4 years (**2019–2022**), against GEDI ground truth and
external reference products.

---

## Table of contents

1. [Mental model: what "validation" means here](#mental-model)
2. [The data you're feeding in](#the-data)
3. [Multi-year workflow and aggregation strategies](#multi-year)
4. [Shared modules: config, io_utils, metrics](#shared-modules)
5. [Script 01 — Quantitative metrics vs GEDI](#script-01)
6. [Script 02 — Longitude & latitude profiles](#script-02)
7. [Script 03 — Within-model attribute pairs](#script-03)
8. [Script 04 — Cross-source attribute pairs](#script-04)
9. [Script 05 — Distributions, Q-Q, KS](#script-05)
10. [Script 06 — Spatial residuals & Moran's I](#script-06)
11. [Script 07 — Stratified metrics](#script-07)
12. [Script 08 — Allometric consistency](#script-08)
13. [Script 09 — Cross-site summary tables](#script-09)
14. [Script 10 — Temporal residual profiles & trajectories](#script-10)
15. [Script 11 — Year-over-year change agreement](#script-11)
16. [Script 12 — Spatial-pattern temporal consistency](#script-12)
17. [Script 13 — Mixed-effects models & allometric stability](#script-13)
18. [Script 14 — Height ↔ Cover allometry vs GEDI](#script-14)
19. [Script 15 — Per-pixel wide CSV export](#script-15)
20. [Cross-cutting cautions](#cautions)
21. [Suggested reading order](#reading-order)

---

<a name="mental-model"></a>
## 1. Mental model: what "validation" means here

It's worth being precise about what each comparison can actually tell you,
because the five attributes sit at different rungs of evidential strength:

| Attribute | Ground truth? | What a comparison establishes |
|---|---|---|
| **Height** | GEDI RH98 (sparse) | **Accuracy** — how close the map is to truth |
| **Cover** | GEDI Cover (sparse) | **Accuracy** |
| **AGBD** | None | **Agreement** — consistency with other products, not accuracy |
| **Stem** | None, no external map | **Plausibility & internal consistency** only |
| **Wood Density** | None, no external map | **Plausibility & internal consistency** only |

This distinction runs through the whole suite. When you compare Height against
GEDI you can say "PG-CBM has an RMSE of X metres." When you compare AGBD against
CCI you can only say "PG-CBM and CCI agree to within X Mg/ha" — if they disagree,
the data alone cannot tell you which one is wrong. The scripts compute the same
arithmetic in both cases, but the **interpretation** must respect this ceiling.
Throughout this guide, "reference" means GEDI (true reference) for Height/Cover
and a "comparator" (CCI, GEDI L4B) for AGBD.

GEDI itself is a *reference*, not perfect truth: RH98 carries noise from beam
sensitivity, geolocation, and terrain slope. It's the best available, but a
non-zero RMSE against GEDI is partly GEDI's own error, not solely your model's.

---

<a name="the-data"></a>
## 2. The data you're feeding in

Per site, the loader expects this folder layout (configured in `config.py`):

```
<site>/
├── PG-CBM_055095/
│   ├── PG-CBM_055095_2019/
│   │   ├── AGB/*.tif   (input folder name, unchanged)
│   │   ├── Height/*.tif
│   │   ├── Cover/*.tif
│   │   ├── Stem/*.tif
│   │   └── WoodDensity/*.tif
│   ├── PG-CBM_055095_2020/  ...
│   ├── PG-CBM_055095_2021/  ...
│   └── PG-CBM_055095_2022/  ...
├── StruMPL_055095/
│   ├── StruMPL_055095_2019/
│   │   └── *_Biomass_*.tif, *_Height_*.tif, *_Cover_*.tif,
│   │     *_StemDensity_*.tif, *_WoodDensity_*.tif  (flat layout, tokens in filename)
│   ├── StruMPL_055095_2020/  ...
│   ├── StruMPL_055095_2021/  ...
│   └── StruMPL_055095_2022/  ...
└── External_Ref/
    ├── External_Ref_2019/*.tif   # 6 bands per year
    ├── External_Ref_2020/*.tif
    ├── External_Ref_2021/*.tif
    └── External_Ref_2022/*.tif
```

The external stack has bands:  1=Lang_Height, 2=Hansen_Cover, 3=CCI_AGBD,
4=GEDI_L4B_AGBD, 5=GEDI_Cover, 6=GEDI_RH98.

**Critical assumption: all maps within a site are already on a common grid**
(same CRS, resolution, extent). The loader verifies this and raises an error if
any layer's shape differs. No reprojection happens in the suite. GEDI is treated
as an ordinary raster with a sparse valid mask — you confirmed footprints are
already rasterised onto the model grid during GEE export, so there is no
footprint-geometry handling.

**Coordinate handling.** If a raster is in a projected CRS (e.g. UTM in metres),
the loader reprojects pixel centres to true lon/lat (EPSG:4326) so profile
plots and coordinate-dependent code always work in degrees. Native CRS
coordinates are still available in the bundle as `x_native` / `y_native`.

---

<a name="multi-year"></a>
## 3. Multi-year workflow and aggregation strategies

You now have 4 years of maps (2019–2022) per site. This section is the single
answer to the question "how do I aggregate single-year methods when I have four
years of data" — it's referenced by the individual script sections rather than
repeated in each.

### 3.1 Two workflow styles, both supported

There are two ways to use the suite in a multi-year context:

**A) Single-year mode (scripts 01–09).** Every script from 01 to 09 was
designed for one year. They still work exactly as before — the shared loader
now defaults to `DEFAULT_YEAR` (2020 by default), so calling `load_site(site)`
returns that year's data. To get single-year outputs for a *different* year,
change `DEFAULT_YEAR` in `config.py` and re-run. To get single-year outputs
for *all* years, run each script four times, changing `DEFAULT_YEAR` each
time, and write outputs to separate folders. This is the "repeat the
single-year analysis per year" approach.

**B) Multi-year mode (scripts 10–14).** These scripts iterate over all
`YEARS` internally in a single run and produce cross-year aggregated
outputs. They are the recommended way to answer temporal questions.

### 3.2 How to aggregate the single-year methods across years

If you take approach (A) and generate single-year outputs for each of the
four years, you'll want to combine them into paper-ready aggregations. The
right strategy depends on the metric type:

**Scalar metrics (bias, RMSE, R², KS-D, Moran's I, correlation).** These are
aggregable straightforwardly:

- *Cross-year mean ± SD per (site, source, attribute)*: one row per site, mean
  of the metric across 4 years, SD as a measure of temporal stability.
  Small SD → the model behaves consistently year-to-year at that site.
- *Cross-site + cross-year mean ± SD per (source, attribute)*: two levels of
  aggregation. Report as the headline number for a paper.
- *Bootstrap CI across sites × years*: with 10 sites × 4 years = 40
  observations, cluster-bootstrap by site (not by year, since years within
  a site are correlated) to get a proper CI.

**Curves (longitude profiles, saturation curves, allometric curves).** These
cannot be pixel-wise averaged across years because pixel identity changes
(GEDI samples different footprints each year). Aggregate at the *binned*
level:

- *Pool GEDI-masked pixels across years*, then rebin. This is what script 14
  does for the H↔C curve, and it's the general pattern that works: because
  the underlying physical relationship is stable year-to-year, pooling the
  pixels just gives you a denser and cleaner reference curve.
- *Alternatively*, keep four curves per source (one per year) and overlay
  them, letting the visual spread convey the year-to-year stability. This
  is what script 10 does for the residual profiles.

**Distributions (CDFs, histograms).** Aggregate by concatenating pixels
across years, then re-fit. Same principle as curves: the H distribution's
shape doesn't change year-to-year (only which pixels GEDI samples), so
pooling gives a denser, better-conditioned distribution.

**Maps (residual maps, z-anomaly rasters).** For paper figures, a
representative year is the pragmatic choice. Show the map for one year in
the main text and put the other three in supplementary. For the *analysis*
itself, script 12 already computes cross-year metrics of these maps
(residual-map similarity between year pairs; per-pixel max|z|).

### 3.3 The unit of replication changes with 4 years

With 10 sites × 4 years, you have 40 site-year observations, but they are
**not 40 independent observations**. Sites are the unit of replication (a
sample from a population of African dryland sites), and years within a site
are correlated (the same underlying landscape). This has three practical
consequences:

- **Bootstrap CIs should cluster on site**, not on year or on pixel. When you
  resample for a CI, resample sites with all their years together.
- **Statistical tests** comparing models should use paired designs (paired
  Wilcoxon, mixed models with site as random effect) — script 13 does this
  properly.
- **N=40 does not mean N=40 for statistical power**. The "effective sample
  size" for a cross-site comparison is closer to N=10, adjusted upward
  slightly if years contribute genuinely new information.

### 3.4 When to run each script

| Question | Script | Multi-year handling |
|---|---|---|
| How accurate is each model in a single year? | 01–09 | Run per year, aggregate scalar metrics with mean ± SD |
| Does model accuracy change across years? | 10 | Native multi-year — RMSE / bias trajectories |
| Does the model track real change (Δmodel vs ΔGEDI)? | 11 | Native multi-year — Δ-agreement per year-pair |
| Is the spatial error structure stable across years? | 12 | Native multi-year — Moran's I per year + map similarity |
| Is one model reliably better across years and sites? | 13 | Native multi-year — mixed model |
| Does the H↔C physical coupling hold across years? | 14 | Native multi-year — one line per (model, year) |
| I need one CSV per site with all years for external analysis | 15 | Native multi-year — wide CSV per site |

---

<a name="shared-modules"></a>
## 4. Shared modules

These four files are imported by the numbered scripts; they hold all the logic
that would otherwise be duplicated.

### `config.py` — single source of truth

Everything tunable lives here so you never edit a method script to change a
path or a threshold. Key entries:

- `ROOT_DIR`, `OUTPUT_DIR`, `SITES` — where data is and where results go.
- `YEARS` — list of years to include in multi-year analyses (default
  `[2019, 2020, 2021, 2022]`).
- `DEFAULT_YEAR` — the year used by single-year scripts (01–09). Change this
  and re-run 01–09 to get single-year outputs for a different year.
- `MODEL_LAYOUT` — per-model file-discovery rules. `dir` and `parent_dir`
  contain `{year}` placeholders that are filled in at load time. PG-CBM uses
  `subfolder_per_attribute`; StruMPL uses `flat_with_filename_pattern` with
  `attr_tokens` mapping each attribute to a substring in the filename
  (`AGBD→"Biomass"`, `Stem→"StemDensity"`, `WoodDensity→"WoodDensity"`, etc.).
  The token match is **case-sensitive** — if filenames vary in case across
  sites this is the first place to look when a file isn't found.
- `EXTERNAL_PARENT_DIR = "External_Ref"`, `EXTERNAL_DIR = "External_Ref_{year}"`
  — per-year external reference sub-folder pattern.
- `EXT_BANDS` — 1-based band indices in each year's external stack. GEDI Cover
  (band 5) and GEDI RH98 (band 6) live here alongside Lang/Hansen/CCI/L4B.
- `ATTRIBUTES` — per-attribute metadata: units, plotting range, the `ref`
  (ground-truth key, or `None`), and the list of `externals`. This dict is what
  tells every script which comparisons are even possible for a given attribute.
- `COLOURS` — consistent colour per source across all figures (PG-CBM green,
  StruMPL purple, GEDI black, external maps blue/red).
- `MAX_PIXELS_FOR_PLOTS = 50_000` — legacy scatter subsampling threshold.
  Scripts 03 and 04 now use density-coloured scatters that render all points,
  so this value is only a fallback for pathological cases.
- `N_BOOTSTRAP = 1_000`, `RANDOM_SEED = 42` — reproducibility and CI resolution.
- `PROFILE_N_BINS = 60`, `PROFILE_MIN_PIX_PER_BIN = 5` — profile binning. The
  fixed bin *count* (not degree width) adapts to each site's spatial extent
  regardless of size or CRS.

### `io_utils.py` — loading a site

The core loader is `load_site_year(site, year)` which returns a "bundle" dict.
`load_site(site)` is a backward-compatible wrapper that uses `DEFAULT_YEAR`, so
existing single-year scripts (01–09) work without modification.

Bundle contents:

- `bundle["arrays"]` — dict of 2-D float32 arrays, nodata as `NaN`, keyed by
  `f"{model}_{attr}"` (e.g. `"PG-CBM_Height"`), plus `"GEDI_RH98"`,
  `"GEDI_Cover"`, `"Lang_Height"`, `"Hansen_Cover"`, `"CCI_AGBD"`,
  `"GEDI_L4B_AGBD"`.
- `bundle["lon"]`, `bundle["lat"]` — 2-D coordinate arrays (pixel centres,
  reprojected to EPSG:4326 if needed), same shape as the rasters.
- `bundle["x_native"]`, `bundle["y_native"]` — native CRS coordinates, in
  case you need them.
- `bundle["profile"]` — the rasterio profile (CRS, transform) for writing
  GeoTIFFs back out.
- `bundle["site"]`, `bundle["year"]` — bookkeeping.

Helper functions:

- `gedi_mask(bundle, attribute)` — boolean array of pixels where the GEDI
  reference for that attribute is finite; returns `None` for AGBD/Stem/WoodDensity
  (no GEDI reference). **This is the masking backbone**: it's how every
  GEDI-referenced comparison restricts itself to the sparse footprints.
- `joint_valid_mask(bundle, keys)` — True only where *all* listed arrays are
  finite, so comparisons use an identical pixel set across every source.
- `stack_valid(...)` — returns an `(N, k)` array of co-valid pixel values for
  pair plots and metrics.
- `density_scatter(ax, x, y, ...)` — density-coloured scatter that keeps every
  point (used by scripts 03, 04, 11).

### `metrics.py` — the error/agreement formulas

`error_metrics(pred, ref)` drops any pair where either value is NaN, then
computes (with `resid = pred − ref`):

| Metric | Formula | Meaning |
|---|---|---|
| **bias** | mean(resid) | systematic over/under-estimation; sign matters |
| **MAE** | mean(\|resid\|) | typical absolute error, robust to outliers |
| **RMSE** | sqrt(mean(resid²)) | error magnitude, penalises large errors heavily |
| **rRMSE** | RMSE / mean(ref) | RMSE as a fraction of the mean — comparable across attributes |
| **R²** | 1 − SS_res/SS_tot | fraction of reference variance explained |
| **r** | Pearson correlation | linear association |
| **rho** | Spearman correlation | rank (monotonic) association, robust to non-linearity |

where `SS_res = Σ(pred−ref)²` and `SS_tot = Σ(ref − mean(ref))²`.

Two subtleties you should keep in mind when reading R² in particular:

- **R² here is the coefficient of determination against the 1:1 line**, not the
  squared Pearson correlation. A map can have `r = 0.95` (great correlation) but
  `R² < 0` if it's correlated yet badly biased or mis-scaled — because `R²`
  penalises departure from `y = x`, while `r` does not. Reading both together
  tells you whether a problem is *scatter* (low r) or *systematic offset/scaling*
  (good r, poor R²).
- **rRMSE divides by mean(ref)**, so it's only meaningful for strictly positive
  quantities (all five here qualify) and becomes unstable when mean(ref) is near
  zero (very sparse/low-cover sites).

`bootstrap_metrics(...)` resamples pixels with replacement to put confidence
intervals on each metric. **iid pixel bootstrap understates uncertainty** for
spatially autocorrelated data (neighbouring pixels aren't independent), so an
optional `block_size` enables a moving-block bootstrap as a coarse correction.
Script 01 sidesteps this issue entirely by bootstrapping **across sites** rather
than pixels (see below) — that's the statistically honest choice with 10 sites.

---

<a name="script-01"></a>
## 5. Script 01 — Quantitative metrics vs GEDI

**File:** `01_quantitative_metrics.py`
**Applies to:** Height, Cover (the attributes with a GEDI reference).
**Compares:** `PG-CBM`, `StruMPL`, and the external map (Lang / Hansen) each
against GEDI.
**Year:** Runs on `DEFAULT_YEAR` — see §3.2 for aggregating across years.

### Logic

For each site and attribute, build the GEDI mask, extract the reference and each
candidate at those sparse pixels, and run `error_metrics`. Then:

1. **Per-site metrics** — one row per (site × attribute × source). Saved to
   `per_site_metrics.csv`.
2. **Pooled metrics** — concatenate every site's pixels and compute metrics on
   the combined set. Saved to `pooled_metrics.csv`. This weights sites by how
   many GEDI pixels they contribute.
3. **Cross-site bootstrap CIs** — resample the *10 per-site metric values* with
   replacement 1,000 times and take the 2.5/97.5 percentiles of the mean. Saved
   to `site_bootstrap_ci.csv`. This treats each site as the unit of replication,
   which respects the fact that pixels within a site are not independent.

### Why bootstrap across sites, not pixels

If you bootstrap pixels you'd get absurdly tight CIs (millions of "independent"
samples that are actually highly correlated), making PG-CBM look reliably better
than StruMPL when the difference is within noise. Bootstrapping the 10 site-level
numbers gives wider, honest intervals: if PG-CBM's and StruMPL's RMSE CIs overlap
heavily, you cannot claim one is better.

### Outputs

- `per_site_metrics.csv` — the raw material; every other summary derives from it.
- `pooled_metrics.csv` — single number per source per attribute.
- `site_bootstrap_ci.csv` — `<metric>_mean`, `<metric>_lo`, `<metric>_hi` per
  source.
- `summary_Height.png`, `summary_Cover.png` — bar charts of RMSE and bias with
  the cross-site 95% CI as error bars.

### Multi-year aggregation

Run this script four times, changing `DEFAULT_YEAR` each time and writing to
distinct output folders. Then merge the four `per_site_metrics.csv` files (add
a `year` column), and produce a cross-site × cross-year summary as described in
§3.2. Alternatively use script 10 which does exactly this natively.

### How to interpret

- **Bias** is the first thing to read. A large positive bias means the map
  systematically overestimates; negative means underestimate. A model can have a
  small RMSE but a worrying bias if errors are consistent in one direction.
- **Compare models via the bar charts.** If the CI error bars overlap, treat the
  two models as statistically indistinguishable on that metric — do *not*
  over-claim from the point estimate.
- **rRMSE lets you compare across attributes**: an RMSE of 4 m for Height and
  0.1 for Cover aren't directly comparable, but rRMSE (e.g. 0.30 vs 0.20) is.
- **Lang/Hansen as a yardstick**: these are mature published products. If your
  model's RMSE against GEDI is comparable to Lang's/Hansen's RMSE against GEDI,
  that's a strong result — you're matching the state of the art. If yours is far
  worse, that gap is the headline finding to explain.

---

<a name="script-02"></a>
## 6. Script 02 — Longitude & latitude profiles

**File:** `02_longitude_profiles.py`
**Applies to:** Height, Cover.
**Year:** Runs on `DEFAULT_YEAR`. Use script 10 for temporal residual profiles.

### Logic

The idea: collapse the 2-D map into a 1-D spatial profile and see whose curve
hugs GEDI's. For a chosen axis (longitude or latitude):

1. Restrict to GEDI-valid pixels (the same mask for *every* source, so all
   curves average over an identical pixel set — this is essential for fairness).
2. Bin the chosen coordinate into `PROFILE_N_BINS` = 60 fixed-count bins that
   span the site's spatial extent. Fixed bin *count* adapts to each site's
   size regardless of CRS — a fixed degree width would collapse UTM-projected
   sites to a couple of bins.
3. For each bin, compute the **mean** of each source over the pixels in that bin.
4. Drop bins with fewer than `PROFILE_MIN_PIX_PER_BIN` = 5 pixels so sparse
   edges don't produce noisy spikes.

The reference (GEDI) defines the set of bins; candidate sources are snapped onto
exactly those bins so the curves are directly comparable point-for-point.

The script runs **both** longitude and latitude because, with sites scattered
across Africa, the dominant ecological gradient at a given site may lie on either
axis.

### Profile-similarity metrics

Beyond the visual, the script computes `error_metrics` between each candidate's
**binned profile** and GEDI's binned profile (RMSE/R²/etc. over the bin means,
not over pixels). Saved to `profile_summary_metrics.csv`. This gives a number for
"how well does the profile track" to complement the eye test.

### Cross-site aggregation

The script also computes a cross-site aggregated view: one row per
`(source, axis)` combination with mean ± SD across sites for every metric,
plus a `mean(|bias|)` column that captures typical bias magnitude regardless
of direction. Saved to `profile_summary_aggregated.csv`. The distinction
between `bias_mean` and `abs_bias_mean` matters: `bias_mean` can be near zero
even when a source is systematically biased in opposite directions at
different sites, while `abs_bias_mean` catches that.

### Summary figures

Two per-attribute summary figures visualise the aggregation:

- `summary_Height.png` — 3-panel figure (bias, |bias|, RMSE) for the three
  Height sources with cross-site SD as error bars.
- `summary_Cover.png` — same layout for Cover sources.

Split by attribute because Cover (fraction 0–1 or percent 0–100, per `COVER_UNITS`) and Height (metres, 0–25)
have very different dynamic ranges that would compress on a shared axis. The
`lon`/`lat` distinction is encoded by hatching (solid = lon, hatched = lat)
so source colour stays constant across both bars.

### Outputs

- `<site>_<attr>_lon.png`, `<site>_<attr>_lat.png` — the profile plots.
- `<site>_<attr>_<axis>_profile.csv` — the binned values, so you can re-plot in
  your own style.
- `profile_summary_metrics.csv` — profile-level RMSE/R² per (site, source, axis).
- `profile_summary_aggregated.csv` — cross-site mean ± SD per (source, axis)
  for every metric, including `abs_bias`.
- `summary_Height.png`, `summary_Cover.png` — visual summary of the
  aggregation across sites.

### How to interpret

- **Shape tracking matters more than absolute level here.** A model whose curve
  is parallel to GEDI but shifted up has a bias (better seen in Script 01); a
  model whose curve has the *wrong shape* (peaks where GEDI dips) has a deeper
  structural problem.
- **Watch the `n_pixels` column.** A bin with a handful of pixels is far noisier
  than one with thousands; treat low-count bins as suggestive, not definitive.
- **Profiles are a smoothing device** — they average out per-pixel scatter, so a
  model can have a beautiful profile and still poor per-pixel RMSE. Always read
  this alongside Script 01. The profile answers "does the map get the *spatial
  trend* right," not "is each pixel right."
- **On the summary bars**: read `bias_mean` and `abs_bias_mean` together. A
  small bar for `bias_mean` beside a large bar for `abs_bias_mean` means the
  source is biased at each site but the direction of the bias varies across
  sites — an important finding easily missed if you only look at mean bias.

---

<a name="script-03"></a>
## 7. Script 03 — Within-model attribute pairs

**File:** `03_attribute_pairs.py`
**This is the Python equivalent of your R `ggpairs`/GGally workflow.**

### Logic

For each model and site, build a 4×4 matrix comparing that model's four core
attributes (AGBD, Height, Cover, Stem) against each other:

- **Diagonal** — histogram (density) of each attribute.
- **Lower triangle** — density-coloured scatter of every attribute pair (every
  pixel is rendered — the density colouring reveals concentration without
  saturating), with a binned-mean curve overlaid.
- **Upper triangle** — Pearson `r` and Spearman `ρ` printed for each pair.

Correlations are computed on the **full** pixel set. The density-scatter
helper handles millions of points via a 2-D histogram lookup, so no
subsampling is needed for the plots either.

Wood Density is not included in the 4-attribute pair matrix by default; if
you want it included, change `ATTRS_ORDER` at the top of the script to a
5-element list.

### Outputs

- `<site>_<model>_pairs.png` — the pair-plot matrix.
- `<site>_<model>_corr.csv` — full Pearson and Spearman correlation matrices.
- `all_sites_<model>_corr_summary.csv` — mean ± sd of each attribute-pair
  correlation across the 10 sites.

### How to interpret

This is a **consistency** check, not an accuracy check — there's no external
truth involved. You're asking: *do this model's attributes relate to each other
the way forest ecology says they should?*

- **Expected couplings** (should be positive): AGBD↔Height, AGBD↔Cover, AGBD↔Stem,
  Height↔Cover. Taller, denser, more-stemmed forest should carry more biomass.
- **A near-perfect correlation (|r| > 0.97) between two attributes is a red
  flag**, not a success: it suggests the model isn't predicting them
  independently — e.g. if Stem is essentially a rescaling of Cover, the model
  hasn't learned a distinct stem signal. You want strong-but-not-degenerate
  couplings.
- **Cross-site stability** (the summary CSV) tells you whether these
  relationships hold everywhere or only at some sites. Large sd across sites
  means the model's internal logic shifts by region — worth investigating.
- **Compare the two models' pair plots side by side.** If PG-CBM shows
  ecologically sensible couplings and StruMPL shows degenerate ones (or vice
  versa), that's evidence about which model is learning real structure.

---

<a name="script-04"></a>
## 8. Script 04 — Cross-source attribute pairs

**File:** `04_source_pairs.py`
**The attribute-centric counterpart of Script 03.**

### Logic

Where Script 03 fixes a model and compares its attributes, Script 04 fixes an
**attribute** and compares its **sources**. For each attribute it builds a pair
plot across all available sources:

- Height: PG-CBM, StruMPL, Lang, GEDI_RH98
- Cover: PG-CBM, StruMPL, Hansen, GEDI_Cover
- AGBD: PG-CBM, StruMPL, CCI, GEDI_L4B
- Stem: PG-CBM, StruMPL (no externals)

**Masking:** if a GEDI reference exists for the attribute, all sources are
restricted to the GEDI mask (so every source is compared on the same support);
otherwise the joint-valid mask across all sources is used. The lower-triangle
scatters use the density-scatter helper (all pixels rendered, colour = local
density) and include a dashed **1:1 line** so you can see at a glance whether
two sources agree in level, not just in correlation.

### Outputs

- `<site>_<attr>_source_pairs.png` — pair-plot matrix across sources.
- `<site>_<attr>_source_corr.csv` — Pearson + Spearman matrices.
- `all_sites_<attr>_source_corr_summary.csv` — cross-site mean ± sd of each
  source-pair correlation.

### How to interpret

- **Points hugging the 1:1 line** = the two sources agree in both correlation
  and magnitude. **Points correlated but parallel-shifted off the 1:1 line** =
  they agree on pattern but disagree on level (one is biased relative to the
  other).
- **For AGBD this is your main quantitative tool** since there's no truth. If
  PG-CBM, StruMPL, CCI, and GEDI L4B all cluster tightly, you have a "converging
  evidence" argument. Where they fan out, none can be declared right — report it
  as genuine inter-product uncertainty.
- **The model-vs-model panel (PG-CBM vs StruMPL) is informative for every
  attribute, including Stem.** High agreement between two independently built
  models is weak evidence of correctness (they could share a bias), but strong
  *disagreement* definitely flags that at least one is unreliable in that range.

---

<a name="script-05"></a>
## 9. Script 05 — Distributions, Q-Q, KS

**File:** `05_distributions.py`
**Catches the single most common regression failure mode: saturation / tail
compression.**

### Logic

A model can have a decent RMSE yet systematically squash the extremes —
predicting too few very-tall or very-short pixels. Mean-based metrics miss this;
distribution comparison catches it. For each attribute:

- Pick an **anchor**: the GEDI reference where it exists, otherwise the first
  available source (PG-CBM) for AGBD/Stem.
- Plot **empirical CDFs** of the anchor and each candidate on shared axes.
- Plot an **empirical Q-Q**: the 1st–99th percentiles of each candidate against
  the same percentiles of the anchor, with a 1:1 reference line.
- Run a **two-sample Kolmogorov–Smirnov test** (`ks_2samp`) between anchor and
  each candidate, reporting the D-statistic (max gap between the two CDFs) and
  p-value.

### Outputs

- `<site>_<attr>_hist_qq.png` — two panels: overlaid CDFs (left), Q-Q (right).
- `ks_test_results.csv` — D-statistic, p-value, and sample sizes per
  (site, attribute, source).

### How to interpret

- **The Q-Q plot is the workhorse.** If a candidate's Q-Q curve lies *along the
  1:1 line*, its distribution matches the anchor. Two classic departures:
  - **S-shape flattening at the top** (candidate's high quantiles fall below the
    1:1 line) = **saturation**: the model can't reach the tallest/highest values.
    Very common for canopy height (plateau ~20–25 m) and AGBD (plateau
    ~150–200 Mg/ha). This is often the headline limitation of a deep-learning
    structure map.
  - **Compression toward the middle** (steep in the centre, flat at both ends) =
    the model is regressing toward the mean, hedging away from extremes.
- **KS caveat:** with hundreds of thousands of pixels the KS test will almost
  always return p < 0.001 — distributions are "significantly different" even when
  the difference is trivial. **Read the D-statistic (effect size), not the
  p-value.** A small D (≈0.05) means the distributions are practically
  identical regardless of significance; a large D (>0.2) means a real shape
  mismatch.
- For AGBD/Stem the anchor is just a reference point, not truth — the comparison
  shows whether sources have the *same shape*, not which shape is correct.

---

<a name="script-06"></a>
## 10. Script 06 — Spatial residuals & Moran's I

**File:** `06_spatial_residuals.py`
**Tests whether errors are random noise (good) or spatially structured (bad).**
**Multi-year note:** for Moran's I trajectories across years, see script 12.

### Logic

A well-behaved model's residuals (`pred − reference`) should look like spatial
white noise. If residuals cluster — whole regions over- or under-predicted — the
model is missing something systematic (a covariate, a regional calibration
issue). For Height and Cover (vs GEDI) and AGBD (vs CCI, as a comparator):

1. Compute per-pixel residuals over the valid mask.
2. Write a residual **GeoTIFF** (so you can overlay it in QGIS/GEE) and a
   coloured residual **map** (diverging red–blue, symmetric about zero, clipped
   at the 98th percentile of |residual| so outliers don't wash out the scale).
3. Compute **global Moran's I** on the residuals.

### Moran's I — the formula and how it's done

Moran's I measures spatial autocorrelation:

```
        N      Σ_i Σ_j w_ij (z_i)(z_j)
  I =  ---  ·  -----------------------
        W              Σ_i z_i²
```

where `z_i = residual_i − mean(residual)` and `w_ij` are spatial weights. The
implementation:

- Builds **k-nearest-neighbour weights** (k = 8) via a KD-tree on pixel
  coordinates, **row-standardised** (each pixel's 8 neighbours weighted 1/8).
- Sub-samples to 5,000 pixels for tractability (Moran's I on millions of pixels
  is both slow and dominated by trivial near-duplicates).
- Computes a **permutation p-value**: shuffle the residual values across
  locations 199 times, recompute I each time, and see how extreme the observed I
  is against that null distribution.

Interpretation of the statistic itself:
- **I ≈ 0** → residuals are spatially random (the desired outcome).
- **I > 0** → positive autocorrelation: nearby residuals are similar — errors
  come in spatial clumps. This is the warning sign.
- **I < 0** → negative autocorrelation (checkerboard); rare for this kind of map.

(The implementation was unit-tested on synthetic fields: ≈0 for random noise,
≈+0.98 for a smooth gradient, ≈−0.96 for a checkerboard.)

### Outputs

- `<site>_<attr>_<model>_residual.tif` — full-resolution residual raster.
- `<site>_<attr>_<model>_residual.png` — coloured residual map.
- `morans_i.csv` — I, permutation p-value, and pixel counts per
  (site, attribute, model).

### How to interpret

- **Open the residual maps first.** Random salt-and-pepper red/blue = good.
  Coherent red blobs or a red/blue split across the scene = structured error.
  The *spatial pattern* often points straight at the cause — e.g. residuals
  tracking a river, a slope aspect, or a land-cover boundary.
- **Then read Moran's I.** A significant positive I (p < 0.05) confirms
  statistically what the map shows. Compare I between PG-CBM and StruMPL: the
  model with I closer to zero has more random (better-behaved) errors.
- **Caveat for AGBD:** residuals here are `model − CCI`, so a structured pattern
  could be the model's fault *or* CCI's. Don't attribute the structure to your
  model without corroboration.
- **Slope/terrain reminder:** GEDI has known slope-related bias, so on steep
  sites some residual structure may originate in the reference, not the map.

---

<a name="script-07"></a>
## 11. Script 07 — Stratified metrics

**File:** `07_stratified_metrics.py`
**Pooled metrics hide *where* a model fails; this reveals it.**

### Logic

Two stratifications:

**(A) By reference-value bin.** Split the reference into bins and compute
metrics within each bin:
- Height bins: `[0,3,6,9,12,15,20,30]` m
- Cover bins: `[0,0.1,0.2,0.3,0.5,0.7,1.0]`
- AGBD bins (against CCI as comparator): `[0,10,25,50,100,200,400]` Mg/ha

Bins with fewer than 30 pixels are skipped. This directly exposes **saturation**:
if bias goes increasingly negative as the reference value rises, the model
under-predicts tall/dense/high-biomass pixels.

**(B) By cover class.** Bin pixels into Sparse (0–0.10), Open (0.10–0.40), and
Closed (0.40+) forest using **Hansen Cover** (chosen over GEDI here because it's
a continuous wall-to-wall map, whereas GEDI is too sparse to populate classes).
Then compute metrics for every attribute within each cover class. This is a
proxy for ecological zone when you don't have an ecoregion raster — it answers
"does the model work as well in open woodland as in closed forest?"

### Outputs

- `stratified_by_refbin_<attr>.csv` — metrics per reference-value bin, per
  source, per site.
- `stratified_by_cover_<attr>.csv` — metrics per cover class, per source,
  per site.
- `saturation_<attr>.png` — **the key diagnostic**: bias (mean ± sd across
  sites) plotted against reference-value-bin centre, one line per source.

### How to interpret

- **The saturation plot is the most diagnostic single figure in the suite.**
  Read the slope of the bias line:
  - **Flat line near zero** across all bins = unbiased across the whole range —
    ideal.
  - **Downward slope** (bias increasingly negative at high reference values) =
    classic saturation; the model compresses the top of the range. Expect this
    for Height and AGBD; the question is *how severe* and *which model is worse*.
  - **Upward slope** = over-prediction at high values (less common).
- **Cover-class table:** look for a class where RMSE balloons. A model that's
  fine in closed forest but poor in sparse woodland (or vice versa) has an
  ecological blind spot — important for African dryland sites where open
  woodland dominates.
- **AGBD stratification is against CCI**, so again it's agreement, not accuracy —
  divergence at high biomass may reflect CCI's own known saturation rather than
  your model's.

> **Extending to true ecoregions:** ecoregion stratification is not implemented
> because the site folders don't contain ecoregion rasters. If you add a RESOLVE
> Ecoregions raster per site, this script is where you'd wire in a third
> stratification — the structure mirrors the cover-class block.

---

<a name="script-08"></a>
## 12. Script 08 — Allometric consistency

**File:** `08_allometric_consistency.py`
**Internal-consistency check via the physical couplings between attributes.**
**Multi-year note:** for allometric stability across years, see script 13.
**For the Height ↔ Cover coupling specifically (with GEDI reference), see script 14.**

### Logic

Forest structure obeys allometry: biomass rises monotonically with height,
cover, and stem density. A self-consistent model should reproduce these. For
each model and site:

1. **Response curves.** Bin AGBD by each predictor (Height, Cover, Stem) and plot
   the mean AGBD per bin (±sd). A flat or *decreasing* curve is a red flag — it
   means the model predicts no (or inverted) biomass response to a driver that
   physically must increase it.
2. **OLS fit.** Fit `AGBD ≈ a·Height + b·Cover + c·Stem + d` per site via least
   squares, reporting the coefficients, their signs, and the R². **All three
   coefficients should be positive**; a negative one signals the model has
   learned a physically implausible relationship. R² here measures how much of
   the model's *own* AGBD is explained by its *own* H/C/S — high R² means a
   tightly self-consistent model.
3. **Model overlay.** Plot PG-CBM's and StruMPL's response curves on the same
   axes, per site and as a cross-site mean (interpolated onto a common predictor
   grid, mean ± sd across sites).

### Outputs

- `allometric_ols_per_site.csv` — coefficients, intercept, R², pixel count per
  (site, model).
- `allometric_curves_<predictor>.png` — cross-site mean AGBD-vs-predictor curve,
  PG-CBM vs StruMPL.
- `allometric_curves_per_site/<site>.png` — the three response curves per site.

### How to interpret

- **Coefficient signs are a hard plausibility gate.** Any negative coefficient
  on Height/Cover/Stem means that model, at that site, predicts that *more* of a
  structural driver yields *less* biomass — physically wrong, and a concrete flaw
  to report.
- **Curve shape:** a monotonic rising AGBD-vs-Height curve that eventually
  flattens is expected (large trees saturate). A curve that turns *down* at high
  predictor values is the warning sign.
- **This is not accuracy.** A model can be perfectly self-consistent (clean
  allometry) and still wrong in absolute terms. Use this to catch *internally
  incoherent* predictions, and pair it with Scripts 01/04 for the accuracy/
  agreement side.
- **Model comparison:** if one model produces tight, monotonic, ecologically
  sensible curves and the other produces noisy or non-monotonic ones, that's a
  qualitative point in favour of the former — independent of the metric tables.

---

<a name="script-09"></a>
## 13. Script 09 — Cross-site summary tables

**File:** `09_summary_tables.py`
**Aggregates Script 01's per-site metrics into paper-ready tables and a
head-to-head model verdict. Must be run after Script 01.**

### Logic

Reads `01_quantitative_metrics/per_site_metrics.csv` and produces:

1. **Per-attribute summary** — for each (attribute, source): n_sites, and the
   mean / sd / median / min / max of every metric across sites. Sorted by mean
   RMSE so the best source is on top.
2. **Head-to-head model comparison** — for Height and Cover, pairs PG-CBM and
   StruMPL by site and runs a **paired Wilcoxon signed-rank test** on the
   per-site metric differences. The Wilcoxon is the non-parametric paired test:
   it asks whether one model is *consistently* better across sites without
   assuming the differences are normally distributed — appropriate for n = 10.
3. **`overview.txt`** — a human-readable digest of both, including a "winner" per
   metric (written UTF-8 for the `→`, `²`, `±` characters).

### Outputs

- `summary_<attr>.csv` — the cross-site metric summary table.
- `model_comparison.csv` — PG-CBM vs StruMPL means, mean difference, Wilcoxon W
  and p, per (attribute, metric).
- `overview.txt` — plain-text summary.

### How to interpret

- **The Wilcoxon p-value is the formal answer to "is one model better?"** With
  only 10 sites the test has limited power, so:
  - **p < 0.05** → one model is consistently better on that metric across sites —
    a defensible claim.
  - **p ≥ 0.05** → the per-site winner varies; report the models as comparable
    on that metric even if one has a nicer mean. **Resist reading a "winner" off
    the mean alone when p is non-significant.**
- **mean ± sd vs median:** if mean and median diverge a lot for a source, one or
  two sites are dominating — check the min/max columns and look at those sites
  individually (Scripts 02/06 maps will usually explain why).
- This script only covers Height and Cover head-to-head, because those are the
  only attributes with a true reference. For AGBD, the model "comparison" is
  agreement-based and lives in Script 04, not here.

---

<a name="script-10"></a>
## 14. Script 10 — Temporal residual profiles & trajectories

**File:** `10_temporal_residual_profiles.py`
**Applies to:** Height, Cover.
**Native multi-year script — iterates over all years in a single run.**

### Logic

Two complementary temporal views:

**(A) Residual longitude/latitude profiles per site.** For each site, plot
residual = (model − GEDI) binned by longitude (and latitude). One line per
(model × year) — 8 lines total for 2 models × 4 years, encoded so earlier
years are lighter. A model that is *temporally consistent* shows years stacked
on top of each other; a model that drifts shows years fanning out.

**(B) RMSE & bias trajectories across years.** Per-site RMSE and bias for
each (model × year), aggregated across sites with bootstrap CIs. Shows
whether either model is degrading or improving over time.

### Outputs

- `residual_profiles/<site>_<attr>_lon.png`, `_lat.png` — the 8-line residual
  profile figures.
- `residual_profiles/<site>_<attr>_<axis>.csv` — the binned data for re-plot.
- `trajectories/<attr>_rmse.png`, `<attr>_bias.png` — cross-site trajectory
  figures with 95% CIs.
- `per_year_metrics.csv` — site × year × model × attribute metric table.
- `trajectory_bootstrap_ci.csv` — per (year, model, attribute) mean + 95% CI.

### How to interpret

- **On the residual profiles**: if all 4 years for a model overlap tightly the
  model is temporally stable. If they fan out (drift), the model's error
  structure is changing year-to-year — worrying for anyone using it for
  change detection.
- **On the trajectories**: a flat line = temporally stable model. A rising
  line = degrading. A dip at a particular year suggests the model was tuned
  for that year (e.g. 2020 as a reference year in training) and doesn't
  generalise well.
- **Compare models via CI overlap** on the trajectory plot: if PG-CBM's and
  StruMPL's CIs overlap at every year, you can't claim one is temporally
  more stable than the other from the trajectory alone.

---

<a name="script-11"></a>
## 15. Script 11 — Year-over-year change agreement

**File:** `11_year_over_year_change_agreement.py`
**Applies to:** Height, Cover.
**Answers: does the model track REAL change, not just static level?**

### Logic

A model that gets the level right in every year isn't necessarily good at
detecting *change* between years. This script computes:

    Δmodel = model_{y+1} − model_{y}
    ΔGEDI  = GEDI_{y+1}  − GEDI_{y}

then correlates them per (site, attribute, model, year-pair). A high positive
Pearson r means "when GEDI says a pixel got taller, so did the model." A near-
zero r means the model's inter-annual differences are unrelated to GEDI's — i.e.
the model may be tracking level but not tracking change.

**Masking:** only pixels where GEDI has valid observations in *both* years
contribute. With GEDI's sparse coverage this is typically a smaller set than
either year alone.

### Outputs

- `delta_metrics.csv` — per (site, attribute, model, year_pair): correlation,
  RMSE, bias, mean Δ.
- `delta_summary.csv` — cross-site mean ± sd of each metric per year-pair.
- `delta_<attr>_corr.png` — timeline of correlations across year-pairs, one
  line per model with cross-site SD as error bars.
- `scatter/<site>_<attr>_<model>_<y1>to<y2>.png` — density-coloured Δ scatter
  per site/year-pair.

### How to interpret

- **The correlation timeline is the headline.** If r stays close to +1 across
  all year-pairs, the model tracks change well. Values near 0 mean it does
  not — even if per-year RMSE was decent, the model isn't a change-detection
  tool for that attribute.
- **CAVEAT specific to GEDI:** GEDI in 2019 and 2020 samples *different*
  physical footprints, so ΔGEDI is intrinsically noisy — some of the low r
  you'll see comes from GEDI, not your model. The script reports per-site
  `n_pixels` so you can see how strong the Δ signal actually was.
- Use this together with script 10's trajectory: script 10 says "is the
  model stable in level year-to-year"; this script says "if there IS change,
  does the model track it."

---

<a name="script-12"></a>
## 16. Script 12 — Spatial-pattern temporal consistency

**File:** `12_temporal_consistency_spatial.py`
**Multi-year extension of the spatial diagnostics in script 06.**

### Logic

Three complementary diagnostics asking whether spatial error structure is
stable across years:

**(A) Moran's I per (site, year, model, attribute).** Same statistic as
script 06 but computed per year. A model whose Moran's I is stable across
years has stable spatial error structure; one that jumps around has an
unstable error pattern.

**(B) Residual-map similarity across years.** For each (site, attribute,
model), the Pearson correlation between every pair of years' residual maps
over their common valid mask. Rendered as a 4×4 heatmap. Values near +1 mean
"the model's spatial residual pattern in 2019 looks like its pattern in
2020" — stable error geography.

*(Technical note: this is mathematically equivalent to a normalised
RV-coefficient for 1-D flattened arrays, which is the standard inter-map
similarity measure in geosciences.)*

**(C) Per-pixel temporal z-anomaly.** For each pixel, `z = (value_y −
mean_over_years) / sd_over_years`. Then `max_over_years |z|` highlights
pixels where the model jumps around unphysically over time. Saved as a
GeoTIFF and a coloured map per (site, attribute, model).

### Outputs

- `morans_i_per_year.csv` — I + p-value per (site, year, model, attribute).
- `map_similarity_<site>_<attr>_<model>.png` — 4×4 correlation heatmap.
- `map_similarity_summary.csv` — pairwise correlations in long format.
- `z_anomaly/<site>_<attr>_<model>_maxabs_z.tif` — per-pixel max|z| raster.
- `z_anomaly/<site>_<attr>_<model>_maxabs_z.png` — coloured map.

### How to interpret

- **Read the residual-map similarity heatmap first.** All cells close to +1 =
  the model's spatial error geography is stable — it makes the same mistakes
  in the same places every year. Cells near 0 or mixed = the error pattern
  is different year-to-year, which for a static forest is a red flag.
- **A stable-but-non-random error pattern is still an error pattern.** High
  similarity + high Moran's I means "consistently structured error." That's
  worse than low similarity + low Moran's I (unstructured but noisy).
- **The z-anomaly map is a fishing tool** for finding specific pixels or
  regions where the model behaves erratically over time. Bright spots on the
  z-anomaly map are your candidates for closer inspection or exclusion.

---

<a name="script-13"></a>
## 17. Script 13 — Mixed-effects models & allometric stability

**File:** `13_mixed_model_and_allometry.py`
**The formal statistical answer to "is one model reliably better?" over sites
and years, plus a check on allometric temporal stability.**

### Logic

**(A) Mixed-effects models on site-year-model cell statistics.**

Aggregate residuals to one row per (site, year, model, attribute) — cell mean
and cell RMSE as separate responses — and fit:

    bias ~ model * year + (1 | site)
    RMSE ~ model * year + (1 | site)

- **Why aggregated, not pixel-level?** Pixels are heavily autocorrelated; a
  pixel-level model would give absurdly tight p-values that mean nothing.
  Aggregating to cell stats matches the actual unit of replication.
- **Why year as a fixed factor, not random?** With only 4 years the variance
  component for a random year effect is poorly estimated. Fixed effects let
  us test year-on-year differences directly.
- **The model × year interaction is the scientifically interesting term:** it
  asks whether the gap BETWEEN models changes across years. If it's not
  significant, we fall back to the additive form as the more parsimonious
  fit.

The script also produces an ANOVA-style **variance decomposition** — a
descriptive breakdown of how much of the residual variance is attributable to
model, year, site, and interaction terms. This tells you the *magnitudes* of
each effect, complementing the mixed-model p-values.

**(B) Allometric stability across years.**

For each (model, year, site), fit `AGBD ~ a·H + b·C + c·S + d` (same as
script 08). Then measure how much the coefficients (a, b, c) drift across
years at a given site. A physically consistent model has coefficients that
barely move year-to-year; a model whose coefficients drift substantially has
learned different allometric relationships in different years.

### Outputs

- `cell_stats.csv` — one row per (site, year, model, attribute).
- `mixed_model_bias_<attr>.txt`, `mixed_model_rmse_<attr>.txt` — statsmodels
  MixedLM output summaries (interactive + additive).
- `variance_decomposition.csv` — % of variance attributable to each factor.
- `allometric_year_fits.csv` — OLS coefficients per (model, year, site).
- `allometric_stability.csv` — per (site, model) mean + SD of each coefficient
  across years.
- `allometric_stability.png` — bar chart of coefficient stability.

### How to interpret

- **Mixed model p-values are the formal statement.** For `model * year`:
  - p < 0.05 → the model gap varies with year (interaction real).
  - p ≥ 0.05 → the additive form is fine; report the additive p-values for
    model and year separately.
- **Variance decomposition tells you the effect sizes.** "Model" explaining
  20% of variance is a much bigger deal than "model" being significant with
  p = 0.02 but 2% of variance.
- **Allometric coefficient drift**: a coefficient with SD ≪ 0.01 across years
  is rock-solid; a coefficient with SD > 0.1 means the model's allometry
  changed between years — either it's overfitting per-year data or the input
  imagery drove the change.
- **A model with tight allometric stability AND poor per-year RMSE is
  worse than one with unstable allometry but good RMSE.** The first has
  learnt a wrong-but-consistent story; the second is at least tracking the
  data even if noisily.

---

<a name="script-14"></a>
## 18. Script 14 — Height ↔ Cover allometry vs GEDI

**File:** `14_height_cover_allometry.py`
**Answers a question script 08 couldn't: does the model reproduce the physical
Height ↔ Cover coupling that GEDI directly observes?**

### Logic

Unlike script 08 (which fits within-model AGBD allometry with no external
reference), the H↔C relationship CAN be checked against GEDI: GEDI measures
both RH98 and Cover at the same footprint, so we have a true reference curve.

Two views per site:

**(A) GEDI-masked (headline figure).** For each (model, year), restrict to
pixels where BOTH GEDI Cover and GEDI RH98 are valid, then compute the
model's H-vs-C and C-vs-H curves on those exact pixels. Compare against
GEDI's own curve (pooled across all available years at that site). This is
the apples-to-apples accuracy check.

**(B) Wall-to-wall (supplementary check).** Same curves computed on all
model pixels (not just GEDI footprints). Tells us whether the model's H-C
coupling is the same in regions GEDI doesn't sample. A divergence between
GEDI-masked and wall-to-wall curves is a sampling-bias warning: the model
learnt one relationship at GEDI-covered pixels and another elsewhere.

Both directions are computed and shown side-by-side per site:
- C-binned-by-H: at each height, what cover does this source predict?
- H-binned-by-C: at each cover, what height does this source predict?

Scalar metrics per (site, year, model, mask): Pearson r, Spearman ρ, OLS
slope on the joint pixels. Uses `scipy.stats.linregress` with a variance-
spread gate so slopes are only reported when the fit is well-conditioned.

**GEDI reference pooling.** The GEDI reference curve is built by pooling
GEDI pixels across all available years per site (rather than picking one
year). The physical H↔C relationship doesn't change year-to-year; pooling
just gives a denser, cleaner reference curve, and naturally handles sites
where GEDI is missing in some years (e.g. Niger in 2019 in your data).

### Outputs

- `gedi_masked/<site>_HC_gedi.png` — 2-panel headline figure per site.
- `wallcheck/<site>_HC_wallcheck.png` — 4-panel supplementary check.
- `hc_metrics.csv` — r, ρ, slope, slope_stderr per (site, year, model, mask).
- `hc_curves.csv` — binned curve points in long format for re-plotting.
- `summary_C_vs_H.png`, `summary_H_vs_C.png` — cross-site mean curve, model vs GEDI.

### How to interpret

- **In the headline figure**, PG-CBM's curves should sit close to the black
  GEDI curve if the model has learnt the physical H-C coupling. StruMPL
  sitting well away from the GEDI curve means it hasn't. The two most
  common failure modes are (i) wrong shape (e.g. linear where truth is
  saturating), and (ii) year-to-year drift (multiple lightness-encoded lines
  for one model diverging).
- **On the wallcheck figure**, if the solid (GEDI-masked) and dashed
  (wall-to-wall) lines lie on top of each other, the model's H-C coupling
  is uniform across space — good sign. If they diverge, the model behaves
  differently at GEDI footprints than elsewhere — a sampling-bias warning.
- **Reading Pearson r together with the curve shape.** A model with a
  perfectly linear (wrong) H-C relationship will show a higher r than GEDI's
  own saturating (correct) relationship because the linear fit has less
  scatter. Don't rank models by r alone — always look at the curve shape.
- **The `slope_stderr` column** in `hc_metrics.csv` tells you how reliable
  the OLS slope is. A slope of 0.05 ± 0.001 is meaningful; a slope of 0.05
  ± 0.04 is not.

### GEDI reference band

The shaded band around the black GEDI line is the *within-bin standard
deviation of GEDI pixel values*, not a confidence interval and not
year-to-year variation. It shows how much natural variability in Cover
exists among GEDI pixels at a given Height (or vice versa). Because
individual pixels vary substantially even at the same reference height, this
band is often wider than the difference between model curves — read it as
"this is the spread of the underlying data," not "this is the uncertainty
in the mean."

---

<a name="script-15"></a>
## 19. Script 15 — Per-pixel wide CSV export

**File:** `15_export_per_pixel_csv.py`
**Aggregation for external analysis — one CSV per site with every year and
source as columns.**

### Logic

Each row is one pixel within a site. Columns cover every year × source
combination:

- `site_id`, `row`, `col` (integer pixel coordinates)
- 24 per-year external columns: `GEDI_Height_<year>`, `GEDI_Cover_<year>`,
  `GEDI_AGBD_<year>` (= GEDI L4B AGBD), `Lang_Height_<year>`,
  `Hansen_Cover_<year>`, `CCI_AGBD_<year>`, one per year.
- 40 per-year model columns: `PG-CBM_<Attribute>_<year>` and
  `StruMPL_<Attribute>_<year>` for the 5 attributes (AGBD, Height, Cover,
  Stem, WoodDensity) × 4 years.

Total: **67 columns per row**. Values are float32; NaN where a source has no
observation at that pixel/year.

**Pixel filtering.** Drop rows where GEDI Height AND GEDI Cover are NaN in
**every** year — i.e. GEDI never observed that pixel in any of the four
years. This is the most permissive drop rule; it keeps any pixel GEDI
touched at least once and removes the vast majority of pixels that GEDI
never observed.

### Usage

```bash
python 15_export_per_pixel_csv.py                 # every site in config.SITES
python 15_export_per_pixel_csv.py --site Niger    # one site only
python 15_export_per_pixel_csv.py --gzip          # gzip output (~10-15× smaller)
```

### Outputs

- `<site>_per_pixel.csv` (or `.csv.gz` with `--gzip`) — one wide file per site.

### When to use this

This isn't a validation figure — it's the pivot into whatever downstream
analysis you want to do outside the pipeline. Typical uses:

- Load in R or Python for custom mixed models, spatial statistics, or
  machine-learning experiments.
- Feed into a pixel-level uncertainty-quantification analysis.
- Cross-tabulate model predictions against covariates not in the pipeline
  (e.g. soil, climate) that you can join by pixel coordinates.

For a paper, `.csv.gz` is a compact and reader-friendly way to publish the
per-pixel data as supplementary material.

### Notes on filename conventions

The script reads WoodDensity from PG-CBM subfolder `WoodDensity/` and from
StruMPL filenames containing the substring `WoodDensity`. If your actual
filenames use a different token, edit the two entries in
`MODEL_LAYOUT["PG-CBM"]["attr_subfolders"]["WoodDensity"]` and
`MODEL_LAYOUT["StruMPL"]["attr_tokens"]["WoodDensity"]` in `config.py`. The
script fails loudly with a "no .tif" error if it can't locate the file.

---

<a name="cautions"></a>
## 20. Cross-cutting cautions

A few principles that apply across the whole suite — worth keeping in mind when
you write up results:

1. **Accuracy vs agreement.** Only Height and Cover have ground truth. Every AGBD,
   Stem, and Wood Density "metric" is agreement or plausibility. State this
   explicitly in any write-up; it's the most common over-claim in map validation.

2. **GEDI is a noisy reference.** Part of every Height/Cover RMSE is GEDI's own
   error (sensitivity, geolocation, slope). A small non-zero RMSE may be near the
   floor set by the reference itself, not a model deficiency.

3. **Significance ≠ importance with big pixel counts.** KS tests (Script 05) and
   any pixel-level test will read "significant" on hundreds of thousands of
   pixels. Lead with **effect sizes** (D-statistic, RMSE, bias magnitude), not
   p-values, except where the unit of replication is sites (Scripts 01, 09, 13).

4. **Sites are the unit of replication, not pixels.** With 10 sites × 4 years,
   claims about "PG-CBM is better" rest on ~10 independent observations, not
   millions of correlated pixels. The cross-site bootstrap (01), paired Wilcoxon
   (09), and mixed model (13) honour this; pixel-level confidence would be
   spuriously narrow.

5. **Years within a site are correlated.** Don't treat 40 site-year observations
   as 40 independent draws. Cluster bootstrap by site, use mixed models with
   site random effects, or paired comparisons per site.

6. **Profiles and curves smooth away scatter.** Scripts 02, 08, 10, 14 can look
   clean while per-pixel accuracy (01) is poor. They answer "right trend / right
   physics," not "right pixel." Always read them in tandem with the metric
   tables.

7. **Read maps before statistics.** The residual maps (06, 12) and saturation
   plots (07) frequently reveal the *mechanism* of an error that a single summary
   number only flags. The number tells you something's wrong; the picture tells
   you what.

8. **Change detection needs its own validation.** A model with excellent per-year
   RMSE (script 01) can have near-zero Δ-agreement (script 11). If your paper
   claims the model tracks change, run script 11 explicitly and report those
   numbers — don't rely on static-year metrics.

9. **WoodDensity has no external reference at all.** Every WoodDensity number in
   the outputs is model-internal. Do not compare WoodDensity to GEDI or CCI
   because there is nothing to compare to. Use it for internal-consistency
   checks (script 08, 13) only.

### Suggested reading order for results

1. `09_summary/overview.txt` — the headline single-year numbers and model verdict.
2. `01` bar charts — accuracy + CIs for Height/Cover in a representative year.
3. `10` trajectory plots — how the numbers move across years.
4. `13` mixed-model summaries — is one model reliably better across sites *and*
   years.
5. `07` saturation plots — where in the value range each model fails.
6. `06` residual maps — the spatial mechanism of those failures.
7. `12` residual-map similarity heatmaps — is the spatial error stable in time.
8. `14` H↔C figure — does the model reproduce the GEDI-observed physical coupling.
9. `05` Q-Q plots — confirm/quantify saturation in distribution terms.
10. `04` source pairs (esp. AGBD) — inter-product agreement where there's no truth.
11. `11` Δ-correlation timeline — does the model track REAL change.
12. `08` allometric curves — internal physical consistency.
13. `02`, `03` — supporting spatial-trend and internal-correlation evidence.
14. `15` per-pixel CSV — the substrate for any custom downstream analysis.