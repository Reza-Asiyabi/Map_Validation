"""
aggregate_01_multi_year.py — Cross-year aggregation of script 01 outputs.

Reads the per_site_metrics.csv from four years of script 01 runs and
produces:
  (a) an aggregated CSV with cross-site bootstrap CIs per (attribute,
      source, year), and
  (b) one summary figure per attribute using the "Option B" layout:
        - StruMPL and PG-CBM as multi-year bars (year on x-axis)
        - Lang/Hansen as horizontal reference lines with shaded CI bands
      This shows temporal stability of the multi-year models AND their
      performance relative to the fixed single-year reference products
      in a single figure.

Design notes on why this layout:
- Sources with multi-year data (StruMPL, PG-CBM) get bars because year-to-
  year variation is meaningful; a bar per year lets the reader see it.
- Sources with genuinely single-year data (Lang 2020, Hansen ~annual but
  we treat as a fixed reference) get horizontal lines because their number
  is fixed — drawing them as bars in each year would falsely imply they
  changed. A shaded band shows the CI so their line's uncertainty is
  visible.
- The CIs on all bars and lines are cross-site bootstraps (n = 10 sites,
  N_BOOTSTRAP resamples). Sites are the unit of replication — pixels
  within a site are spatially correlated and would give spuriously tight
  intervals if used directly.

Usage: point INPUT_DIRS at the four per-year output folders of script 01
and run. If your folder scheme is different, edit INPUT_DIRS.

Outputs (under OUTPUT_DIR/01_aggregate_multi_year/):
    aggregated_metrics.csv    site × year × attr × source metrics + bootstrap CIs
    summary_<attr>_multi_year.png    one per attribute
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (OUTPUT_DIR, COLOURS, MODELS, ATTRIBUTES,
                    N_BOOTSTRAP, RANDOM_SEED)


# -----------------------------------------------------------------------------
# Where to find the four per-year script-01 outputs. Two ways to spell this:
#
#   (a) explicit mapping — most robust
#   (b) if you followed the convention of naming output folders like
#       01_quantitative_metrics_<year>, uncomment the auto-discovery block
#       below and this file will pick them up automatically.
# -----------------------------------------------------------------------------
INPUT_DIRS: dict[int, Path] = {
    2019: OUTPUT_DIR / "01_quantitative_metrics_2019",
    2020: OUTPUT_DIR / "01_quantitative_metrics_2020",
    2021: OUTPUT_DIR / "01_quantitative_metrics_2021",
    2022: OUTPUT_DIR / "01_quantitative_metrics_2022",
}

# Which sources are multi-year (drawn as bars) and which are single-year
# reference products (drawn as horizontal lines with a shaded CI band).
# The single-year sources are read from ONE canonical year — the year they
# are published for or that we treat them as representing.
MULTI_YEAR_SOURCES = {
    "Height": ["StruMPL_Height", "PG-CBM_Height"],
    "Cover":  ["StruMPL_Cover",  "PG-CBM_Cover"],
}
SINGLE_YEAR_SOURCES = {
    "Height": {"Lang_Height":  2020},
    "Cover":  {"Hansen_Cover": 2020},
}

OUT = OUTPUT_DIR / "01_aggregate_multi_year"
OUT.mkdir(parents=True, exist_ok=True)


def _bootstrap_cis(per_site_df: pd.DataFrame,
                   metrics: list[str],
                   n_boot: int,
                   seed: int) -> dict[str, dict[str, float]]:
    """Cross-site bootstrap: resample sites with replacement, compute the mean
    of each metric per resample, return {metric: {mean, lo, hi}}.
    Same logic as script 01's inline bootstrap.
    """
    rng = np.random.default_rng(seed)
    n = len(per_site_df)
    if n == 0:
        return {m: {"mean": np.nan, "lo": np.nan, "hi": np.nan} for m in metrics}
    arr = {m: per_site_df[m].to_numpy(dtype=np.float64) for m in metrics}
    boot: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for m in metrics:
            vals = arr[m][idx]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                boot[m].append(float(vals.mean()))
    out = {}
    for m in metrics:
        v = np.asarray(boot[m])
        if v.size == 0:
            out[m] = {"mean": np.nan, "lo": np.nan, "hi": np.nan}
        else:
            out[m] = {
                "mean": float(v.mean()),
                "lo":   float(np.percentile(v,  2.5)),
                "hi":   float(np.percentile(v, 97.5)),
            }
    return out


def _colour_for(src: str) -> str:
    """Look up the project colour for a source. Falls back to the model
    prefix (PG-CBM / StruMPL) if the full source name isn't a key."""
    if src in COLOURS:
        return COLOURS[src]
    prefix = src.split("_")[0]
    return COLOURS.get(prefix, "#666")


def _load_all_years() -> pd.DataFrame:
    """Read the four per-site CSVs and concatenate with a `year` column.
    Returns a long-format DataFrame: (year, site, attribute, source, metrics...).
    """
    frames = []
    for year, folder in INPUT_DIRS.items():
        f = folder / "per_site_metrics.csv"
        if not f.is_file():
            print(f"  [WARN] {f} not found — year {year} will be missing")
            continue
        df = pd.read_csv(f)
        df["year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No per_site_metrics.csv found under the configured INPUT_DIRS. "
            f"Checked: {[str(p) for p in INPUT_DIRS.values()]}"
        )
    return pd.concat(frames, ignore_index=True)


def _compute_aggregated(long_df: pd.DataFrame) -> pd.DataFrame:
    """For every (attribute, source, year) present, compute cross-site bootstrap
    CIs on the metrics we plot. Returns one row per group."""
    METRICS = ["bias", "MAE", "RMSE", "rRMSE", "R2", "r"]
    rows = []
    seed = RANDOM_SEED
    for (attr, src, year), grp in long_df.groupby(["attribute", "source", "year"]):
        cis = _bootstrap_cis(grp, METRICS, N_BOOTSTRAP, seed)
        seed += 1
        row = {"attribute": attr, "source": src, "year": int(year),
               "n_sites": len(grp)}
        for m in METRICS:
            row[f"{m}_mean"] = cis[m]["mean"]
            row[f"{m}_lo"]   = cis[m]["lo"]
            row[f"{m}_hi"]   = cis[m]["hi"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["attribute", "source", "year"])


def _plot_attribute(attr: str, agg: pd.DataFrame):
    """Build the Option B multi-year summary figure for one attribute:
      - two panels: RMSE (left), bias (right)
      - multi-year sources as grouped bars, one bar per year
      - single-year reference sources as horizontal lines with shaded CI bands
    """
    sub = agg[agg["attribute"] == attr]
    if sub.empty:
        return None

    years_present = sorted(sub["year"].unique().tolist())
    if not years_present:
        return None

    multi_srcs  = MULTI_YEAR_SOURCES.get(attr, [])
    single_srcs = SINGLE_YEAR_SOURCES.get(attr, {})
    unit = ATTRIBUTES[attr]["unit"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric in zip(axes, ["RMSE", "bias"]):
        # Grouped bars for multi-year sources
        x = np.arange(len(years_present))
        n_multi = len(multi_srcs)
        # widths sum to about 0.8 so groups have breathing room
        bar_w = 0.8 / max(n_multi, 1)

        for i, src in enumerate(multi_srcs):
            src_rows = sub[sub["source"] == src].set_index("year")
            means = np.array([src_rows.loc[y, f"{metric}_mean"] if y in src_rows.index else np.nan
                              for y in years_present])
            los   = np.array([src_rows.loc[y, f"{metric}_lo"]   if y in src_rows.index else np.nan
                              for y in years_present])
            his   = np.array([src_rows.loc[y, f"{metric}_hi"]   if y in src_rows.index else np.nan
                              for y in years_present])
            # yerr for asymmetric CIs
            err_lo = np.clip(means - los, 0, np.inf)
            err_hi = np.clip(his - means, 0, np.inf)
            err = np.vstack([err_lo, err_hi])

            offset = (i - (n_multi - 1) / 2) * bar_w
            colour = _colour_for(src)
            ax.bar(x + offset, means, width=bar_w,
                   yerr=err, capsize=3, color=colour, alpha=0.55,
                   edgecolor="black", linewidth=0.4,
                   label=src.split("_")[0])

        # Horizontal reference lines for single-year sources
        for src, ref_year in single_srcs.items():
            src_row = sub[(sub["source"] == src) & (sub["year"] == ref_year)]
            if src_row.empty:
                continue
            mean = float(src_row[f"{metric}_mean"].iloc[0])
            lo   = float(src_row[f"{metric}_lo"].iloc[0])
            hi   = float(src_row[f"{metric}_hi"].iloc[0])
            colour = _colour_for(src)
            # Draw a solid line at the mean and a shaded CI band across the plot
            ax.axhline(mean, color=colour, lw=1.8, linestyle="--",
                       label=f"{src.split('_')[0]} ({ref_year})", zorder=1)
            ax.axhspan(lo, hi, color=colour, alpha=0.12, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels(years_present)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{metric} [{unit}]")
        ax.set_title(f"{attr} — {metric}")
        ax.grid(alpha=0.3, axis="y")
        if metric == "bias":
            ax.axhline(0, color="k", lw=0.5)

    # Single legend on the right panel so we don't duplicate
    axes[1].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"{attr}: cross-site RMSE and bias, 2019–2022\n"
        "(bars = multi-year models with 95% CI; dashed lines = single-year "
        "reference products with shaded CI)",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    out_png = OUT / f"summary_{attr}_multi_year.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _compute_paper_table_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the numbers that fill the paper's multi-year cross-site table AND
    the cross-year-vs-cross-site SD comparison sentence.

    For each (attribute, source), returns:
        <metric>_mean:              mean across the 40 site-year cells
        <metric>_sd:                SD across the 40 site-year cells (this is
                                    the "cross-site x cross-year SD" that
                                    goes into the table as `mean ± SD`)
        <metric>_sd_across_years:   for each site, SD across years, then mean
                                    across sites (temporal-only variation)
        <metric>_sd_across_sites:   for each year, SD across sites, then mean
                                    across years (spatial-only variation)
        <metric>_pct_years_over_sites:  100 * sd_across_years / sd_across_sites
                                    (the "X% of between-site variation"
                                    ratio in the paper's paragraph)
        n_site_years:               number of site-year cells (10 x 4 = 40
                                    if the data are complete)

    This function does not compute bootstrap CIs — the CIs are per-year and
    already live in aggregated_metrics.csv; the paper table wants the simpler
    mean-and-SD summary across the full site-year grid instead.
    """
    METRICS = ["bias", "MAE", "RMSE", "rRMSE", "R2", "r"]
    rows = []
    for (attr, src), grp in long_df.groupby(["attribute", "source"]):
        row = {"attribute": attr, "source": src, "n_site_years": len(grp)}
        for m in METRICS:
            vals = grp[m].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_sd"]   = np.nan
                row[f"{m}_sd_across_years"] = np.nan
                row[f"{m}_sd_across_sites"] = np.nan
                row[f"{m}_pct_years_over_sites"] = np.nan
                continue

            # Mean and SD across the full site-year grid — the table numbers.
            # ddof=1 gives the unbiased sample SD, consistent with what most
            # papers report.
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_sd"]   = float(vals.std(ddof=1)) if vals.size > 1 \
                                else float("nan")

            # Cross-year variation: SD across years within each site, then
            # averaged across sites. Sites with only one year of data
            # contribute nothing to this.
            per_site_sds = []
            for _site, site_grp in grp.groupby("site"):
                v = site_grp[m].to_numpy(dtype=np.float64)
                v = v[np.isfinite(v)]
                if v.size > 1:
                    per_site_sds.append(float(v.std(ddof=1)))
            sd_across_years = float(np.mean(per_site_sds)) \
                              if per_site_sds else float("nan")

            # Cross-site variation: SD across sites within each year, then
            # averaged across years. Symmetric to the above.
            per_year_sds = []
            for _yr, year_grp in grp.groupby("year"):
                v = year_grp[m].to_numpy(dtype=np.float64)
                v = v[np.isfinite(v)]
                if v.size > 1:
                    per_year_sds.append(float(v.std(ddof=1)))
            sd_across_sites = float(np.mean(per_year_sds)) \
                              if per_year_sds else float("nan")

            row[f"{m}_sd_across_years"] = sd_across_years
            row[f"{m}_sd_across_sites"] = sd_across_sites
            row[f"{m}_pct_years_over_sites"] = (
                100.0 * sd_across_years / sd_across_sites
                if (np.isfinite(sd_across_years)
                    and np.isfinite(sd_across_sites)
                    and sd_across_sites > 0)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["attribute", "source"])


def main():
    print(f"[agg01] reading per_site_metrics.csv from "
          f"{len(INPUT_DIRS)} year folders")
    long_df = _load_all_years()
    print(f"[agg01] loaded {len(long_df)} per-site rows total "
          f"({long_df['year'].nunique()} years, "
          f"{long_df['site'].nunique()} sites, "
          f"{long_df['source'].nunique()} sources)")

    agg = _compute_aggregated(long_df)
    agg_path = OUT / "aggregated_metrics.csv"
    agg.to_csv(agg_path, index=False)
    print(f"[agg01] wrote {agg_path}  ({len(agg)} rows)")

    # Second aggregation: cross-site x cross-year mean and SD per (attribute,
    # source), for the paper's multi-year table AND the cross-year-vs-cross-
    # site SD ratio in the surrounding paragraph. See docstring of
    # _compute_paper_table_stats for exactly which columns come from where.
    paper_stats = _compute_paper_table_stats(long_df)
    paper_stats_path = OUT / "paper_table_stats.csv"
    paper_stats.to_csv(paper_stats_path, index=False)
    print(f"[agg01] wrote {paper_stats_path}  ({len(paper_stats)} rows)")

    for attr in sorted(long_df["attribute"].unique()):
        png = _plot_attribute(attr, agg)
        if png:
            print(f"[agg01] wrote {png}")
    print("[agg01] done.")


if __name__ == "__main__":
    main()