"""
aggregate_02_multi_year.py — Cross-year aggregation of script 02 outputs.

Reads the per-year script-02 output folders and produces two complementary
deliverables:

  (A) Cross-site × cross-year METRIC summary (paper-ready numbers).
      - aggregated_metrics.csv: one row per (attribute, source, axis, year)
        with cross-site mean ± SD of bias, |bias|, MAE, RMSE, R², r, plus
        n_sites. Contains ALL sources including single-year references
        (Lang, Hansen), for anyone who wants to slice the numbers manually.
      - summary_grand.csv: one row per (attribute, source, axis) collapsed
        across ALL sites and years — the single-number-per-source view for
        a paper table. Also contains all sources.
      - summary_<attr>.png: 3-panel bar chart per attribute (bias, |bias|,
        RMSE) with YEAR on the x-axis and SOURCE as bar colour. lon/lat
        distinguished by hatching within each cluster. RESTRICTED TO
        MULTI-YEAR SOURCES (StruMPL, PG-CBM) — Lang and Hansen are fixed
        single-year products; comparing the same Lang 2020 map against
        four different GEDI year-samples is a sampling artefact, not a
        model property, so plotting them as year-varying bars would be
        misleading. Their 2020 numbers are in the CSVs above.

  (B) Temporal-stability profile FIGURES per site.
      Overlays four years of GEDI + four years of StruMPL profiles at each
      site, one figure per (site, attr, axis). GEDI shown in black shades
      (dashed), StruMPL in green shades (solid), both encoded so earlier
      years are lighter. This is the "does the model drift more than the
      GEDI sampling variability" visualisation.

Why this design:
- The metric summary answers "how well does the profile track GEDI's over
  years and sites" as one paper table.
- The bar chart puts year on the x-axis and makes source the colour — the
  layout you specified for the summary.
- The temporal profile figures per site answer a different question: does
  the model's profile SHAPE drift year to year? A model where StruMPL's
  four lines fan wider than GEDI's four lines is drifting more than the
  reference sampling would predict.

Usage: point INPUT_DIRS at the four per-year output folders of script 02.
If your folder scheme is different, edit INPUT_DIRS below.

Outputs land under OUTPUT_DIR/02_aggregate_multi_year/.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, COLOURS, ATTRIBUTES, MODELS


# -----------------------------------------------------------------------------
# Where to find the four per-year script-02 outputs.
# -----------------------------------------------------------------------------
INPUT_DIRS: dict[int, Path] = {
    2019: OUTPUT_DIR / "02_profiles_2019",
    2020: OUTPUT_DIR / "02_profiles_2020",
    2021: OUTPUT_DIR / "02_profiles_2021",
    2022: OUTPUT_DIR / "02_profiles_2022",
}

# For the temporal profile figures (deliverable B): which sources to overlay
# across years. GEDI is the reference — its year-to-year variation reflects
# sampling of different footprints, not physical change. StruMPL's variation
# reflects model drift on top of that.
TEMPORAL_PROFILE_SOURCES = {
    "Height": {"reference": "GEDI_RH98",  "model": "StruMPL_Height"},
    "Cover":  {"reference": "GEDI_Cover", "model": "StruMPL_Cover"},
}

# Axes to include in the temporal profile figures.
AXES = ["lon", "lat"]

OUT = OUTPUT_DIR / "02_aggregate_multi_year"
OUT_PROFILES = OUT / "temporal_profiles"
OUT.mkdir(parents=True, exist_ok=True)
OUT_PROFILES.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# (A) Metric summary aggregation
# -----------------------------------------------------------------------------
def _load_all_year_metrics() -> pd.DataFrame:
    """Read profile_summary_metrics.csv from each year folder, tag with year.
    The per-year CSV has one row per (site, attribute, axis, source) — the
    profile-similarity metrics between that source's binned profile and GEDI's.
    """
    frames = []
    for year, folder in INPUT_DIRS.items():
        f = folder / "profile_summary_metrics.csv"
        if not f.is_file():
            print(f"  [WARN] {f} not found — year {year} will be missing")
            continue
        df = pd.read_csv(f)
        df["year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No profile_summary_metrics.csv under {list(INPUT_DIRS.values())}"
        )
    return pd.concat(frames, ignore_index=True)


def _aggregate_metrics(long_df: pd.DataFrame):
    """Return two DataFrames:
        agg_year: mean ± SD across SITES per (attribute, source, axis, year)
        agg_grand: mean ± SD across SITES × YEARS per (attribute, source, axis)
                   — the single-number-per-source table.
    """
    d = long_df.copy()
    d["abs_bias"] = d["bias"].abs()
    METRICS = ["bias", "abs_bias", "MAE", "RMSE", "rRMSE", "R2", "r"]
    agg_spec = {m: ["mean", "std"] for m in METRICS}

    def _flatten(agg_df):
        agg_df.columns = [
            c[0] if c[1] == "" else f"{c[0]}_{'sd' if c[1]=='std' else c[1]}"
            for c in agg_df.columns
        ]
        return agg_df

    # Per-year cross-site aggregation
    agg_year = (d.groupby(["attribute", "source", "axis", "year"],
                          as_index=False).agg(agg_spec))
    agg_year = _flatten(agg_year)
    site_counts_y = (d.groupby(["attribute", "source", "axis", "year"])["site"]
                     .nunique().reset_index().rename(columns={"site": "n_sites"}))
    agg_year = agg_year.merge(site_counts_y,
                              on=["attribute", "source", "axis", "year"])

    # Grand aggregation across sites AND years — the paper-headline table
    agg_grand = (d.groupby(["attribute", "source", "axis"],
                           as_index=False).agg(agg_spec))
    agg_grand = _flatten(agg_grand)
    n_grand = (d.groupby(["attribute", "source", "axis"])
                .agg(n_sites=("site", "nunique"),
                     n_years=("year", "nunique"),
                     n_obs=("year", "count"))
                .reset_index())
    agg_grand = agg_grand.merge(n_grand, on=["attribute", "source", "axis"])

    return agg_year, agg_grand


def _resolve_colour(src: str) -> str:
    if src in COLOURS:
        return COLOURS[src]
    prefix = src.split("_")[0]
    return COLOURS.get(prefix, "#666")


def _src_order_key(src: str):
    for i, mk in enumerate(["PG-CBM", "StruMPL"]):
        if src.startswith(mk):
            return (0, i, src)
    return (1, 0, src)


def _is_multi_year_source(src: str) -> bool:
    """A source is 'multi-year' if it comes from one of the models listed in
    config.MODELS (currently PG-CBM and StruMPL), which produce independent
    predictions each year. Fixed reference products like Lang (published
    2020) and Hansen (a single fixed vintage) appear in every per-year CSV
    with the same values compared against different GEDI samples, so
    plotting them as year-varying bars would be misleading. Excluding them
    here confines the multi-year plot to sources whose year-to-year
    variation is a real property of the source, not a sampling artefact.
    Their per-site metrics are still preserved in the CSV outputs for
    anyone who wants them (e.g. for a 2020-only comparison).
    """
    return any(src.startswith(m) for m in MODELS)


def _plot_metric_summary(agg_year: pd.DataFrame, attr: str):
    """3-panel bar chart per attribute (bias, |bias|, RMSE).
    x-axis = year, groups within each year = source, hatching = lon/lat.
    Error bars = across-site SD.

    Only multi-year sources (from config.MODELS) are drawn — see
    _is_multi_year_source() for the rationale.
    """
    sub = agg_year[agg_year["attribute"] == attr]
    if sub.empty:
        return None

    # Restrict to sources whose year-to-year variation is meaningful.
    sub = sub[sub["source"].apply(_is_multi_year_source)]
    if sub.empty:
        print(f"  [WARN] no multi-year sources found for {attr} — skipping figure")
        return None

    years = sorted(sub["year"].unique().tolist())
    sources = sorted(sub["source"].unique().tolist(), key=_src_order_key)
    unit = ATTRIBUTES[attr]["unit"]

    PLOT_METRICS = [
        ("bias",     f"Bias [{unit}]",     True),
        ("abs_bias", f"|Bias| [{unit}]",   False),
        ("RMSE",     f"RMSE [{unit}]",     False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    n_src = len(sources)
    # Within each year, we have n_src * 2 bars (source × axis). The 'lon' vs
    # 'lat' distinction is hatching, not colour, so effective group width uses
    # only n_src. Each source occupies (n_axes = 2) slots side-by-side.
    n_axes = 2
    total_group_width = 0.85
    sub_bar_w = total_group_width / (n_src * n_axes)

    for ax, (metric, ylabel, signed) in zip(axes, PLOT_METRICS):
        for si, src in enumerate(sources):
            colour = _resolve_colour(src)
            for ai, ax_name in enumerate(["lon", "lat"]):
                means, sds = [], []
                for y in years:
                    row = sub[(sub["source"] == src) & (sub["axis"] == ax_name)
                              & (sub["year"] == y)]
                    if row.empty:
                        means.append(np.nan); sds.append(np.nan)
                    else:
                        means.append(float(row[f"{metric}_mean"].iloc[0]))
                        sds.append(float(row[f"{metric}_sd"].iloc[0]))
                hatch = None if ax_name == "lon" else "//"
                x_positions = np.arange(len(years))
                slot_index = si * n_axes + ai
                offset = ((slot_index + 0.5) / (n_src * n_axes) - 0.5) * total_group_width
                ax.bar(x_positions + offset, means, width=sub_bar_w,
                       yerr=sds, capsize=2, color=colour, alpha=0.55,
                       edgecolor="black", linewidth=0.4, hatch=hatch,
                       error_kw=dict(lw=0.8, alpha=0.7))
        ax.set_xticks(np.arange(len(years)))
        ax.set_xticklabels(years)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{attr} — {metric.replace('_', ' ')}")
        ax.grid(alpha=0.3, axis="y")
        if signed:
            ax.axhline(0, color="k", lw=0.5)

    # Legend: source colours + axis hatching
    handles = []
    for src in sources:
        handles.append(Patch(facecolor=_resolve_colour(src),
                             edgecolor="black", label=src))
    handles.append(Patch(facecolor="#bbbbbb", edgecolor="black",
                         label="axis = lon"))
    handles.append(Patch(facecolor="#bbbbbb", edgecolor="black", hatch="//",
                         label="axis = lat"))
    axes[-1].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{attr}: profile-vs-GEDI metrics across sites and years",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    out_png = OUT / f"summary_{attr}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_png


# -----------------------------------------------------------------------------
# (B) Temporal-stability profile figures
# -----------------------------------------------------------------------------
def _year_lightness(year: int, years_present: list[int]) -> float:
    """Return alpha in [0.4, 1.0]; earlier years lighter, later darker."""
    idx = years_present.index(year)
    if len(years_present) == 1:
        return 1.0
    return 0.4 + 0.6 * (idx / (len(years_present) - 1))


def _plot_temporal_profile(site: str, attr: str, axis: str,
                            year_curves: dict[int, pd.DataFrame]):
    """One figure per (site, attr, axis) with GEDI (dashed black shades) and
    StruMPL (solid green shades) — 4 lines each, year encoded by lightness.
    """
    ref_key  = TEMPORAL_PROFILE_SOURCES[attr]["reference"]
    model_key = TEMPORAL_PROFILE_SOURCES[attr]["model"]
    unit = ATTRIBUTES[attr]["unit"]

    years_present = sorted(year_curves.keys())
    if not years_present:
        return None

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for year in years_present:
        df = year_curves[year]
        alpha = _year_lightness(year, years_present)
        # GEDI: black with alpha, dashed
        if ref_key in df.columns:
            ax.plot(df["bin_centre"], df[ref_key],
                    color="black", alpha=alpha, lw=1.6, linestyle="--",
                    label=f"GEDI {year}")
        # StruMPL: green (COLOURS['StruMPL']) with alpha, solid
        if model_key in df.columns:
            ax.plot(df["bin_centre"], df[model_key],
                    color=_resolve_colour(model_key), alpha=alpha, lw=1.8,
                    linestyle="-",
                    label=f"StruMPL {year}")

    ax.set_xlabel("Longitude (°)" if axis == "lon" else "Latitude (°)")
    ax.set_ylabel(f"{attr} [{unit}]")
    ax.set_title(
        f"{site} — {attr} profile by {axis}: temporal stability\n"
        "(GEDI dashed = reference sampling variability; "
        "StruMPL solid = model + reference variability)"
    )
    ax.grid(alpha=0.3)
    # 8-line legend — put outside plot to keep the curves readable
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_png = OUT_PROFILES / f"{site}_{attr}_{axis}.png"
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _build_temporal_profiles():
    """For each (site, attr, axis), collect the per-year binned-profile CSVs
    and draw the 8-line temporal figure. Skip a site if fewer than 2 years
    are available (need at least 2 to show 'stability')."""
    n_figs = 0
    for attr in TEMPORAL_PROFILE_SOURCES:
        for axis in AXES:
            # Discover sites present in AT LEAST ONE year
            sites: set[str] = set()
            per_year: dict[int, dict[str, Path]] = {}
            for year, folder in INPUT_DIRS.items():
                per_year[year] = {}
                if not folder.is_dir():
                    continue
                for csv in folder.glob(f"*_{attr}_{axis}_profile.csv"):
                    # Filename pattern: <site>_<attr>_<axis>_profile.csv
                    # Extract site by stripping suffix
                    suffix = f"_{attr}_{axis}_profile.csv"
                    site = csv.name[:-len(suffix)]
                    sites.add(site)
                    per_year[year][site] = csv

            for site in sorted(sites):
                # Load whichever years have a CSV for this site
                year_curves = {}
                for year in sorted(per_year.keys()):
                    if site in per_year[year]:
                        try:
                            year_curves[year] = pd.read_csv(per_year[year][site])
                        except Exception as e:
                            print(f"  [WARN] could not read "
                                  f"{per_year[year][site].name}: {e}")
                if len(year_curves) < 2:
                    continue
                png = _plot_temporal_profile(site, attr, axis, year_curves)
                if png:
                    n_figs += 1
    return n_figs


def main():
    # ---- (A) metric summary aggregation --------------------------------------
    print(f"[agg02] reading profile_summary_metrics.csv from "
          f"{len(INPUT_DIRS)} year folders")
    long_df = _load_all_year_metrics()
    print(f"[agg02] loaded {len(long_df)} per-site rows total "
          f"({long_df['year'].nunique()} years, "
          f"{long_df['site'].nunique()} sites, "
          f"{long_df['source'].nunique()} sources)")

    agg_year, agg_grand = _aggregate_metrics(long_df)
    agg_year.to_csv(OUT / "aggregated_metrics.csv", index=False)
    agg_grand.to_csv(OUT / "summary_grand.csv", index=False)
    print(f"[agg02] wrote aggregated_metrics.csv "
          f"({len(agg_year)} rows) and summary_grand.csv ({len(agg_grand)} rows)")

    for attr in sorted(agg_year["attribute"].unique()):
        png = _plot_metric_summary(agg_year, attr)
        if png:
            print(f"[agg02] wrote {png}")

    # ---- (B) temporal profile figures ---------------------------------------
    print("[agg02] building temporal-stability profile figures ...")
    n_figs = _build_temporal_profiles()
    print(f"[agg02] wrote {n_figs} temporal profile figures "
          f"under {OUT_PROFILES}")
    print("[agg02] done.")


if __name__ == "__main__":
    main()