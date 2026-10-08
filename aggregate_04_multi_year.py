"""
aggregate_04_multi_year.py — Cross-site pooled cross-source pair plots per year.

Script 04 produces one pair matrix per (site, attribute) comparing sources
(PG-CBM, StruMPL, Lang/Hansen, GEDI reference where available). This
aggregator produces one pooled matrix per (attribute, year) — pixels from
all sites concatenated within the same mask used by script 04:

    - GEDI-masked pixels if the attribute has a GEDI reference (Height, Cover)
    - joint-valid across sources otherwise (AGBD, Stem, WoodDensity)

Each cell's upper-triangle annotation carries two correlation numbers:

    pooled r        Pearson r on the pooled pixels (one number across sites)
    per-site r      mean ± SD of the 10 within-site Pearson correlations

Pooled r captures the modal cross-source agreement across the whole study
region. Per-site r captures how consistently the sources agree WITHIN each
site independently. Both are useful; both should be reported in a paper.

Sources per attribute (matches script 04):
    Height: PG-CBM, StruMPL, Lang, GEDI_RH98
    Cover:  PG-CBM, StruMPL, Hansen, GEDI_Cover
    AGBD:    PG-CBM, StruMPL, CCI, GEDI_L4B
    Stem:   PG-CBM, StruMPL
    WoodDensity: PG-CBM, StruMPL

Attributes with fewer than 2 sources are skipped.

Outputs (under OUTPUT_DIR/04_aggregate_multi_year/):
    <year>_<attr>_pooled_source_pairs.png   pooled pair matrix
    pooled_source_correlations.csv          pooled r + per-site r stats per
                                             (year, attribute, src_x, src_y)
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
from config import (SITES, YEARS, ATTRIBUTES, OUTPUT_DIR, RANDOM_SEED)
from io_utils import (load_site_year, gedi_mask, joint_valid_mask,
                       density_scatter)


OUT = OUTPUT_DIR / "04_aggregate_multi_year"
OUT.mkdir(parents=True, exist_ok=True)


# Density-scatter memory cap. Same rationale as aggregate_03.
POOLED_SUBSAMPLE_CAP = 10_000_000


def _sources_for(attr: str) -> list[str]:
    """Same rule as script 04's _sources_for()."""
    cfg = ATTRIBUTES[attr]
    srcs = [f"PG-CBM_{attr}", f"StruMPL_{attr}"]
    srcs += cfg["externals"]
    if cfg["ref"] is not None:
        srcs.append(cfg["ref"])
    return srcs


def _load_pooled_pixels(year: int, attr: str,
                         srcs: list[str]) -> pd.DataFrame:
    """Load all sites for a (year, attribute) and pool the co-valid pixel
    values across the listed sources.

    Uses the same mask rule as script 04: GEDI mask if the attribute has a
    GEDI reference; otherwise joint-valid across all sources.
    """
    frames = []
    for site in SITES:
        try:
            bundle = load_site_year(site, year)
        except FileNotFoundError as e:
            print(f"  [WARN] {site} {year}: {e}")
            continue

        m = gedi_mask(bundle, attr)
        if m is None:
            m = joint_valid_mask(bundle, srcs)
        else:
            m = m & joint_valid_mask(bundle, srcs)
        if m.sum() == 0:
            continue

        data = {s: bundle["arrays"][s][m] for s in srcs}
        df = pd.DataFrame(data)
        df["site"] = site
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No site data pooled for {year}/{attr}")
    return pd.concat(frames, ignore_index=True)


def _per_site_correlations(pooled: pd.DataFrame,
                            srcs: list[str]) -> dict[tuple[str, str],
                                                     dict[str, float]]:
    """Per-source-pair Pearson r within each site, then mean ± SD across
    sites."""
    result = {}
    for a, b in combinations(srcs, 2):
        per_site = []
        for _site, g in pooled.groupby("site"):
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


def _pooled_correlations(pooled: pd.DataFrame,
                          srcs: list[str]) -> dict[tuple[str, str], float]:
    """Pearson r on the full pooled data, one number per source pair."""
    result = {}
    for a, b in combinations(srcs, 2):
        ok = np.isfinite(pooled[a]) & np.isfinite(pooled[b])
        if ok.sum() < 30:
            result[(a, b)] = np.nan
            continue
        result[(a, b)] = float(stats.pearsonr(pooled[a][ok], pooled[b][ok])[0])
    return result


def _pair_plot(pooled: pd.DataFrame, srcs: list[str],
               pooled_r: dict, per_site_r: dict,
               attr: str, title: str, out_path: Path):
    """Cross-source pair matrix — same layout as agg03 but with source
    column names instead of attribute names.

    Lower triangle: pooled density scatter + 1:1 line so bias-vs-scatter is
    visible.
    """
    k = len(srcs)
    fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k))
    unit = ATTRIBUTES[attr]["unit"]

    plot_df = pooled
    if len(plot_df) > POOLED_SUBSAMPLE_CAP:
        plot_df = plot_df.sample(POOLED_SUBSAMPLE_CAP,
                                 random_state=RANDOM_SEED)

    for i, yname in enumerate(srcs):
        for j, xname in enumerate(srcs):
            ax = axes[i, j] if k > 1 else axes
            x = plot_df[xname].to_numpy()
            y = plot_df[yname].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]

            if i == j:
                if x.size > 5:
                    ax.hist(x, bins=60, color="#444", alpha=0.55,
                            edgecolor="none", density=True)
                ax.set_yticks([])
            elif i > j:
                if x.size > 0:
                    density_scatter(ax, x, y, bins=200, cmap="viridis", s=2)
                    # 1:1 line — makes bias vs scatter separable at a glance
                    lim_lo = min(x.min(), y.min())
                    lim_hi = max(x.max(), y.max())
                    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                            "k--", lw=0.7, alpha=0.5)
            else:
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
                ax.set_xlabel(xname, fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yname, fontsize=9)
            else:
                ax.set_yticklabels([])

    fig.suptitle(f"{title}  [{unit}]", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    corr_rows = []
    attrs_to_plot = list(ATTRIBUTES.keys())
    print(f"[agg04] pooling across {len(SITES)} sites × "
          f"{len(YEARS)} years × {len(attrs_to_plot)} attributes")

    for year in YEARS:
        for attr in attrs_to_plot:
            srcs = _sources_for(attr)
            if len(srcs) < 2:
                print(f"[agg04] {year} / {attr}: only {len(srcs)} sources — "
                      "skipping")
                continue

            print(f"[agg04] {year} / {attr} — pooling across sites ...")
            try:
                pooled = _load_pooled_pixels(year, attr, srcs)
            except RuntimeError as e:
                print(f"  [WARN] {e}")
                continue
            print(f"          {len(pooled):,} pooled pixels from "
                  f"{pooled['site'].nunique()} sites, {len(srcs)} sources")

            pooled_r   = _pooled_correlations(pooled, srcs)
            per_site_r = _per_site_correlations(pooled, srcs)

            for (a, b), pr in pooled_r.items():
                ps = per_site_r[(a, b)]
                corr_rows.append({
                    "year": year, "attribute": attr,
                    "src_x": a, "src_y": b,
                    "pooled_r":        pr,
                    "per_site_r_mean": ps["mean"],
                    "per_site_r_sd":   ps["sd"],
                    "n_sites":         ps["n_sites"],
                    "n_pixels_pooled": int(len(pooled)),
                })

            title = f"Cross-site pooled — {attr} — {year}"
            out_png = OUT / f"{year}_{attr}_pooled_source_pairs.png"
            _pair_plot(pooled, srcs, pooled_r, per_site_r,
                       attr, title, out_png)
            print(f"          wrote {out_png.name}")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "pooled_source_correlations.csv", index=False)
    print(f"[agg04] wrote pooled_source_correlations.csv ({len(corr_df)} rows)")
    print("[agg04] done.")


if __name__ == "__main__":
    main()
