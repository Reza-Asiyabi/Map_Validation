"""
02 — Longitude & latitude profiles against GEDI.

For each site and for Height & Cover (the attributes with a GEDI reference),
mask all candidate maps with GEDI's valid-pixel mask, then bin by longitude
(and separately by latitude) and plot the mean of each source per bin. The
candidate whose curve hugs the GEDI curve most closely 'wins' the visual test.

Bins with too few valid pixels (< PROFILE_MIN_PIX_PER_BIN) are dropped to
avoid noisy edges.

Outputs (under OUTPUT_DIR/02_profiles/):
    <site>_<attr>_lon.png
    <site>_<attr>_lat.png
    <site>_<attr>_profile.csv          (raw binned values for re-plotting)
    profile_summary_metrics.csv        Per-site detail: one row per (site, attr, axis, source)
    profile_summary_aggregated.csv     Across-site aggregation: one row per (source, axis)
                                       with mean ± SD of every metric and mean of |bias|
    summary_<attr>.png                 One 3-panel figure per attribute (Height, Cover, ...)
                                       showing bias, |bias|, and RMSE across sites
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (SITES, ATTRIBUTES, OUTPUT_DIR, COLOURS,
                    PROFILE_N_BINS, PROFILE_MIN_PIX_PER_BIN)
from io_utils import load_site, gedi_mask
from metrics import error_metrics


OUT = OUTPUT_DIR / "02_profiles"
OUT.mkdir(parents=True, exist_ok=True)


TARGETS = {
    "Height": {"reference": "GEDI_RH98",
               "sources":   ["PG-CBM_Height", "StruMPL_Height", "Lang_Height"]},
    "Cover":  {"reference": "GEDI_Cover",
               "sources":   ["PG-CBM_Cover", "StruMPL_Cover", "Hansen_Cover"]},
}


def _make_edges(coords: np.ndarray, n_bins: int):
    """
    Build n_bins+1 edges spanning the finite coordinate range.
    Returns None if the coordinates have no spread (degenerate).
    """
    c = coords[np.isfinite(coords)]
    if c.size == 0:
        return None
    lo, hi = float(c.min()), float(c.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    # pad the top edge a hair so the max value falls inside the last bin
    return np.linspace(lo, hi, n_bins + 1)


def _bin_on_edges(values: np.ndarray, coords: np.ndarray, edges: np.ndarray):
    """
    Bin `values` by `coords` onto fixed `edges`.
    Returns (mean_per_bin, count_per_bin) arrays of length len(edges)-1.
    Bins with no pixels get NaN mean / 0 count.
    """
    finite = np.isfinite(values) & np.isfinite(coords)
    nb = edges.size - 1
    if finite.sum() == 0:
        return np.full(nb, np.nan), np.zeros(nb, dtype=int)
    v = values[finite]
    c = coords[finite]
    idx = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, nb - 1)
    sums   = np.bincount(idx, weights=v, minlength=nb)
    counts = np.bincount(idx,            minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)
    return means, counts


def _profile_for_axis(bundle, attr, cfg, axis: str):
    """axis ∈ {'lon','lat'}. Returns a DataFrame, one row per kept bin.

    All sources are binned onto a SINGLE shared set of edges derived from the
    GEDI-masked coordinate range, so every column is directly comparable. Bins
    where the reference has fewer than PROFILE_MIN_PIX_PER_BIN valid pixels are
    dropped (these are the noisy edges).
    """
    coord_arr = bundle[axis]
    mask = gedi_mask(bundle, attr)
    if mask is None or mask.sum() == 0:
        return pd.DataFrame()
    coords = coord_arr[mask]

    edges = _make_edges(coords, PROFILE_N_BINS)
    if edges is None:
        return pd.DataFrame()
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Reference defines which bins are "well-populated" enough to keep
    ref = bundle["arrays"][cfg["reference"]][mask]
    ref_mean, ref_count = _bin_on_edges(ref, coords, edges)
    keep = ref_count >= PROFILE_MIN_PIX_PER_BIN
    if keep.sum() < 2:
        return pd.DataFrame()

    df = pd.DataFrame({
        "bin_centre":      centres[keep],
        "n_pixels":        ref_count[keep],
        cfg["reference"]:  ref_mean[keep],
    })
    # Every candidate binned onto the SAME edges, sliced to the same kept bins
    for src in cfg["sources"]:
        src_mean, _ = _bin_on_edges(bundle["arrays"][src][mask], coords, edges)
        df[src] = src_mean[keep]
    return df


def _plot_profile(df, site, attr, cfg, axis, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ref_key = cfg["reference"]
    ax.plot(df["bin_centre"], df[ref_key], color="black", lw=1.0,
            label=f"{ref_key} (reference)", zorder=10, alpha=0.6)
    for src in cfg["sources"]:
        c = COLOURS.get(src.split("_")[0] if src.startswith(("PG-CBM", "StruMPL"))
                        else src, "#888")
        ax.plot(df["bin_centre"], df[src], color=c, lw=1.0,
                label=src, alpha=0.6)
    ax.set_xlabel("Longitude (°)" if axis == "lon" else "Latitude (°)")
    ax.set_ylabel(f"Mean {attr} [{ATTRIBUTES[attr]['unit']}]")
    ax.set_title(f"{site} — {attr} profile by {axis}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    summary_rows = []
    for site in SITES:
        print(f"[02] loading {site} ...")
        bundle = load_site(site)
        for attr, cfg in TARGETS.items():
            for axis in ("lon", "lat"):
                df = _profile_for_axis(bundle, attr, cfg, axis)
                if df.empty:
                    continue
                csv_path = OUT / f"{site}_{attr}_{axis}_profile.csv"
                df.to_csv(csv_path, index=False)
                png_path = OUT / f"{site}_{attr}_{axis}.png"
                _plot_profile(df, site, attr, cfg, axis, png_path)

                # Profile-similarity metrics: each candidate vs reference,
                # computed over the bin means (not pixels).
                ref_vals = df[cfg["reference"]].to_numpy()
                for src in cfg["sources"]:
                    m = error_metrics(df[src].to_numpy(), ref_vals)
                    summary_rows.append({
                        "site": site, "attribute": attr, "axis": axis,
                        "source": src, **m,
                    })
                print(f"  {attr}/{axis} → {png_path.name}")

    detail = pd.DataFrame(summary_rows)
    detail.to_csv(OUT / "profile_summary_metrics.csv", index=False)
    print(f"[02] wrote {OUT / 'profile_summary_metrics.csv'}")

    # ---- Cross-site aggregated summary: one row per (source, axis) ----------
    # Mean of |bias| is reported separately from mean of bias, because they
    # answer different questions:
    #   mean(bias)  can be near zero if a source over-predicts at some sites
    #               and under-predicts at others (errors cancel)
    #   mean(|bias|) captures typical bias MAGNITUDE regardless of direction
    # Reporting both makes both stories visible.
    if not detail.empty:
        # |bias| as a derived column for aggregation
        d = detail.copy()
        d["abs_bias"] = d["bias"].abs()

        METRICS = ["bias", "abs_bias", "MAE", "RMSE", "rRMSE", "R2", "r", "rho"]
        # Build aggregation spec: mean + SD for each metric, plus n_sites count
        agg_spec = {col: ["mean", "std"] for col in METRICS}
        # Use 'n' (the per-site sample count) only for total pixel count if needed
        agg = d.groupby(["source", "axis"], as_index=False).agg(agg_spec)
        # Flatten multi-index columns: 'bias_mean', 'bias_sd', ...
        agg.columns = [
            c[0] if c[1] == "" else f"{c[0]}_{'sd' if c[1]=='std' else c[1]}"
            for c in agg.columns
        ]
        # Add n_sites = number of sites contributing to each (source, axis) row
        site_counts = (d.groupby(["source", "axis"])["site"]
                        .nunique().reset_index().rename(columns={"site":"n_sites"}))
        agg = agg.merge(site_counts, on=["source", "axis"], how="left")

        # Reorder columns: identifiers first, then n_sites, then metric pairs
        ordered = ["source", "axis", "n_sites"]
        for m in METRICS:
            ordered += [f"{m}_mean", f"{m}_sd"]
        agg = agg[ordered]

        agg.to_csv(OUT / "profile_summary_aggregated.csv", index=False)
        print(f"[02] wrote {OUT / 'profile_summary_aggregated.csv'}  "
              f"({len(agg)} source × axis rows)")

        # ---- Visual summary figures ----------------------------------------
        # One 3-panel figure PER ATTRIBUTE (Height, Cover, ...). Cover and
        # Height live on very different dynamic ranges (Cover ∈ [0,1] or [0,100] vs
        # Height ∈ [0,25] m), so plotting them on the same axes would squash
        # the smaller one into a flat line — splitting by attribute is the
        # right fix.
        #
        # Within each per-attribute figure: three panels (bias, |bias|, RMSE),
        # one bar per (source, axis), error bars = across-site SD. Bias is
        # signed (bars cross zero line); |bias| and RMSE are non-negative.
        PLOT_METRICS = [
            ("bias",     "Bias (mean across sites)",      True),
            ("abs_bias", "|Bias| (mean across sites)",    False),
            ("RMSE",     "RMSE (mean across sites)",      False),
        ]
        AXES_ORDER = ["lon", "lat"]

        def _resolve_colour(src):
            """Project colour for a source: model sources use the model's
            colour, external sources use their own key."""
            if src in COLOURS:
                return COLOURS[src]
            prefix = src.split("_")[0]
            return COLOURS.get(prefix, "#666")

        # Stable source ordering: PG-CBM first, then StruMPL, then externals
        def _src_order_key(src):
            for i, mk in enumerate(["PG-CBM", "StruMPL"]):
                if src.startswith(mk):
                    return (0, i, src)
            return (1, 0, src)

        # Per-attribute aggregation built directly from `detail` so the
        # attribute column survives. This is the same arithmetic as `agg`
        # above but grouped by (attribute, source, axis) instead of just
        # (source, axis).
        d_attr = detail.copy()
        d_attr["abs_bias"] = d_attr["bias"].abs()

        for attr in sorted(d_attr["attribute"].unique()):
            sub = d_attr[d_attr["attribute"] == attr]
            unit = ATTRIBUTES[attr]["unit"]
            agg_attr_spec = {m: ["mean", "std"]
                             for m in ["bias", "abs_bias", "RMSE"]}
            agg_attr = sub.groupby(["source", "axis"], as_index=False).agg(agg_attr_spec)
            agg_attr.columns = [
                c[0] if c[1] == "" else f"{c[0]}_{'sd' if c[1]=='std' else c[1]}"
                for c in agg_attr.columns
            ]
            sources_attr = sorted(agg_attr["source"].unique(), key=_src_order_key)

            def _draw_metric(ax, metric, title, signed):
                x = np.arange(len(sources_attr))
                width = 0.38
                for i, ax_name in enumerate(AXES_ORDER):
                    means, sds, colours = [], [], []
                    for src in sources_attr:
                        row = agg_attr[(agg_attr["source"] == src)
                                       & (agg_attr["axis"] == ax_name)]
                        if row.empty:
                            means.append(np.nan); sds.append(np.nan)
                            colours.append("#cccccc")
                            continue
                        means.append(float(row[f"{metric}_mean"].iloc[0]))
                        sds.append(float(row[f"{metric}_sd"].iloc[0]))
                        colours.append(_resolve_colour(src))
                    offset = (i - 0.5) * width
                    # lon = solid, lat = hatched — encodes axis without
                    # competing with source colour
                    hatch = None if ax_name == "lon" else "//"
                    ax.bar(x + offset, means, width, yerr=sds, capsize=3,
                           color=colours, edgecolor="black", linewidth=0.4,
                           hatch=hatch,
                           error_kw=dict(lw=0.8, alpha=0.7))
                from matplotlib.patches import Patch
                legend_handles = [
                    Patch(facecolor="#bbbbbb", edgecolor="black",
                          label="axis = lon"),
                    Patch(facecolor="#bbbbbb", edgecolor="black", hatch="//",
                          label="axis = lat"),
                ]
                ax.set_xticks(x)
                ax.set_xticklabels(sources_attr, rotation=30, ha="right",
                                   fontsize=8)
                ax.set_ylabel(f"{title} [{unit}]")
                ax.set_title(title)
                ax.grid(alpha=0.3, axis="y")
                if signed:
                    ax.axhline(0, color="k", lw=0.6)
                return legend_handles

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            handles = None
            for ax, (metric, title, signed) in zip(axes, PLOT_METRICS):
                handles = _draw_metric(ax, metric, title, signed)
            axes[-1].legend(handles=handles, loc="best", fontsize=8,
                            framealpha=0.85)
            fig.suptitle(f"{attr}: profile-vs-GEDI similarity — summary across sites",
                         y=1.02, fontsize=12)
            fig.tight_layout()
            out_png = OUT / f"profile_summary_{attr}.png"
            fig.savefig(out_png, dpi=170, bbox_inches="tight")
            plt.close(fig)
            print(f"[02] wrote {out_png}")

    print("[02] done.")


if __name__ == "__main__":
    main()