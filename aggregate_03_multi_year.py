"""
aggregate_03_multi_year.py — Cross-site pooled attribute pair plots per year.

Script 03 produces one 4x4 attribute pair matrix per (site, model). This
aggregator produces one pooled matrix per (model, year) — pixels from all
sites concatenated, density-coloured so all points are visible, with two
correlation numbers annotated per cell:

    pooled r        Pearson correlation on the pooled pixels (one number
                    across all 10 sites' worth of pixels)
    per-site r      mean ± SD of the 10 within-site Pearson correlations

The distinction matters because pooled r mixes within-site and between-site
variation (Simpson's-paradox territory), while per-site r captures the
consistency of the relationship WITHIN each site independently. A paper
should report per-site r as the honest model-consistency number and use the
pooled scatter as the visual.

Design notes:
- Sites are loaded via load_site_year(site, year), so the aggregator needs
  access to the raster data — the per-site corr CSVs written by script 03
  are not enough to build the pooled scatter (only the numbers).
- We render one figure per (model, year), so with 2 models × 4 years you
  get 8 figures. Change the MODELS / YEARS lists in this file if you want
  a subset.
- The subsampling cap is more aggressive than in script 03's per-site
  version, because we're pooling 10 sites' worth of pixels — many millions
  in typical data. Density scatter handles it fine but rendering time
  grows.

Usage: this aggregator uses load_site_year() directly and does not read
script 03's per-year output CSVs, so it can be run independently of any
prior script 03 runs. Just call it after your rasters are in place.

Outputs (under OUTPUT_DIR/03_aggregate_multi_year/):
    <year>_<model>_pooled_pairs.png     the pooled pair matrix
    pooled_correlations.csv             pooled r and per-site r mean ± SD
                                        per (year, model, attr_x, attr_y)
"""

from __future__ import annotations
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import (SITES, YEARS, MODELS, ATTRIBUTES, OUTPUT_DIR, COLOURS,
                    RANDOM_SEED)
from io_utils import load_site_year, density_scatter


OUT = OUTPUT_DIR / "03_aggregate_multi_year"
OUT.mkdir(parents=True, exist_ok=True)


# Attributes for the pair grid. Matches script 03's default (4-attribute pair
# plot). If you want to include WoodDensity, change this to a 5-item list.
ATTRS_ORDER = ["AGB", "Height", "Cover", "Stem"]

# Density-scatter memory cap. Pooled data from 10 sites can be very large;
# subsample only if it exceeds this. Density colouring still gives faithful
# results at 10M points; higher is fine but slower to render.
POOLED_SUBSAMPLE_CAP = 10_000_000


def _load_pooled_pixels(year: int, model: str) -> pd.DataFrame:
    """Load all sites' pixels for one (year, model) and return a long
    DataFrame with an added 'site' column. Only rows where every attribute
    is finite are kept."""
    frames = []
    for site in SITES:
        try:
            bundle = load_site_year(site, year)
        except FileNotFoundError as e:
            print(f"  [WARN] {site} {year}: {e}")
            continue
        cols = {a: bundle["arrays"][f"{model}_{a}"].ravel()
                for a in ATTRS_ORDER}
        df = pd.DataFrame(cols).dropna()
        if df.empty:
            continue
        df["site"] = site
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No site data loaded for {year}/{model}")
    return pd.concat(frames, ignore_index=True)


def _per_site_correlations(pooled: pd.DataFrame) -> dict[tuple[str, str],
                                                          dict[str, float]]:
    """Compute per-site Pearson r for each attribute pair, then return
    mean ± SD across sites. Returns dict keyed by (attr_x, attr_y).

    Guards against degenerate sites: skips a site's contribution to a pair
    if it has <30 finite pixels or if either attribute has zero variance
    (would give a Pearson divide-by-zero warning).
    """
    result = {}
    for a, b in combinations(ATTRS_ORDER, 2):
        per_site = []
        for site, g in pooled.groupby("site"):
            if len(g) < 30 or g[a].std() < 1e-9 or g[b].std() < 1e-9:
                continue
            per_site.append(float(stats.pearsonr(g[a], g[b])[0]))
        per_site = np.asarray(per_site)
        per_site = per_site[np.isfinite(per_site)]
        if per_site.size == 0:
            result[(a, b)] = dict(mean=np.nan, sd=np.nan, n_sites=0)
        else:
            result[(a, b)] = dict(mean=float(per_site.mean()),
                                  sd=float(per_site.std(ddof=1))
                                     if per_site.size > 1 else 0.0,
                                  n_sites=int(per_site.size))
    return result


def _pooled_correlations(pooled: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Pearson r computed on the full pooled pixels (one number per pair)."""
    result = {}
    for a, b in combinations(ATTRS_ORDER, 2):
        ok = np.isfinite(pooled[a]) & np.isfinite(pooled[b])
        if ok.sum() < 30:
            result[(a, b)] = np.nan
            continue
        result[(a, b)] = float(stats.pearsonr(pooled[a][ok], pooled[b][ok])[0])
    return result


def _pair_plot(pooled: pd.DataFrame, pooled_r: dict, per_site_r: dict,
               title: str, model: str, out_path: Path):
    """4x4 pair matrix:
        diagonal:      density histogram of the pooled distribution
        lower tri:     density-coloured scatter (all pooled pixels)
        upper tri:     pooled r + per-site r mean ± SD
    """
    cols = ATTRS_ORDER
    k = len(cols)
    fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k))

    # Optional cap to keep rendering tractable — pooled 10-site data can
    # easily exceed 20M pixels
    plot_df = pooled
    if len(plot_df) > POOLED_SUBSAMPLE_CAP:
        plot_df = plot_df.sample(POOLED_SUBSAMPLE_CAP,
                                 random_state=RANDOM_SEED)

    fill_colour = COLOURS.get(model, "#1b7837")

    for i, yname in enumerate(cols):
        for j, xname in enumerate(cols):
            ax = axes[i, j]
            x = plot_df[xname].to_numpy()
            y = plot_df[yname].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]

            if i == j:
                # Diagonal: histogram of the pooled distribution
                if x.size > 5:
                    ax.hist(x, bins=60, color=fill_colour, alpha=0.6,
                            edgecolor="none", density=True)
                ax.set_yticks([])
            elif i > j:
                # Lower triangle: pooled density scatter
                if x.size > 0:
                    density_scatter(ax, x, y, bins=200, cmap="viridis", s=2)
            else:
                # Upper triangle: correlation stats
                key = (xname, yname) if (xname, yname) in pooled_r \
                       else (yname, xname)
                pr = pooled_r.get(key, np.nan)
                ps = per_site_r.get(key, dict(mean=np.nan, sd=np.nan,
                                              n_sites=0))
                pooled_txt = (f"pooled r = {pr:+.2f}"
                              if np.isfinite(pr) else "pooled r = —")
                if np.isfinite(ps["mean"]):
                    site_txt = (f"per-site r = "
                                f"{ps['mean']:+.2f} ± {ps['sd']:.2f}\n"
                                f"(n = {ps['n_sites']} sites)")
                else:
                    site_txt = "per-site r: n/a"
                ax.text(0.5, 0.66, pooled_txt, ha="center", va="center",
                        fontsize=10, transform=ax.transAxes)
                ax.text(0.5, 0.32, site_txt, ha="center", va="center",
                        fontsize=8, color="gray", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if i == k - 1:
                ax.set_xlabel(f"{xname}\n[{ATTRIBUTES[xname]['unit']}]",
                              fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"{yname}\n[{ATTRIBUTES[yname]['unit']}]",
                              fontsize=9)
            else:
                ax.set_yticklabels([])

    fig.suptitle(title, y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    corr_rows = []
    print(f"[agg03] pooling across {len(SITES)} sites × "
          f"{len(YEARS)} years × {len(MODELS)} models")
    for year in YEARS:
        for model in MODELS:
            print(f"[agg03] {year} / {model} — loading and pooling pixels ...")
            try:
                pooled = _load_pooled_pixels(year, model)
            except RuntimeError as e:
                print(f"  [WARN] {e}")
                continue
            print(f"          {len(pooled):,} pooled pixels from "
                  f"{pooled['site'].nunique()} sites")

            pooled_r = _pooled_correlations(pooled)
            per_site_r = _per_site_correlations(pooled)

            for (a, b), pr in pooled_r.items():
                ps = per_site_r[(a, b)]
                corr_rows.append({
                    "year": year, "model": model,
                    "attr_x": a, "attr_y": b,
                    "pooled_r": pr,
                    "per_site_r_mean": ps["mean"],
                    "per_site_r_sd":   ps["sd"],
                    "n_sites":         ps["n_sites"],
                    "n_pixels_pooled": int(len(pooled)),
                })

            title = f"Cross-site pooled — {model} — {year}"
            out_png = OUT / f"{year}_{model}_pooled_pairs.png"
            _pair_plot(pooled, pooled_r, per_site_r, title, model, out_png)
            print(f"          wrote {out_png.name}")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "pooled_correlations.csv", index=False)
    print(f"[agg03] wrote pooled_correlations.csv ({len(corr_df)} rows)")
    print("[agg03] done.")


if __name__ == "__main__":
    main()
