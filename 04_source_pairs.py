"""
04 — Cross-source pair plots & correlations per attribute.

For each attribute that has more than one source available, make a pair plot
across {PG-CBM, StruMPL, external maps, reference if any}. This is the
attribute-centric counterpart of script 03:
    Height: PG-CBM, StruMPL, Lang, GEDI_RH98(*)
    Cover:  PG-CBM, StruMPL, Hansen, GEDI_Cover(*)
    AGB:    PG-CBM, StruMPL, CCI, GEDI_L4B
    Stem:   PG-CBM, StruMPL  (no externals)

(*) When a GEDI reference exists, we restrict pixels to the GEDI mask so
every source is compared on the same support. Otherwise we use joint-valid
pixels across all sources.

Outputs (under OUTPUT_DIR/04_source_pairs/):
    <site>_<attr>_source_pairs.png
    <site>_<attr>_source_corr.csv
    all_sites_<attr>_source_corr_summary.csv
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
from config import (SITES, ATTRIBUTES, OUTPUT_DIR, COLOURS,
                    MAX_PIXELS_FOR_PLOTS, RANDOM_SEED)
from io_utils import load_site, gedi_mask, joint_valid_mask, density_scatter


OUT = OUTPUT_DIR / "04_source_pairs"
OUT.mkdir(parents=True, exist_ok=True)


def _sources_for(attr: str) -> list[str]:
    cfg = ATTRIBUTES[attr]
    srcs = [f"PG-CBM_{attr}", f"StruMPL_{attr}"]
    srcs += cfg["externals"]
    if cfg["ref"] is not None:
        srcs.append(cfg["ref"])
    return srcs


def _pair_plot(df: pd.DataFrame, title: str, out_path: Path, attr: str):
    cols = list(df.columns)
    k = len(cols)
    fig, axes = plt.subplots(k, k, figsize=(2.5 * k, 2.5 * k))
    unit = ATTRIBUTES[attr]["unit"]

    for i, yname in enumerate(cols):
        for j, xname in enumerate(cols):
            ax = axes[i, j] if k > 1 else axes
            x = df[xname].to_numpy(); y = df[yname].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if i == j:
                if x.size > 5:
                    ax.hist(x, bins=40, color="#444", alpha=0.6,
                            edgecolor="none", density=True)
                ax.set_yticks([])
            elif i > j:
                if x.size > 0:
                    density_scatter(ax, x, y, bins=200, cmap="viridis", s=3)
                if x.size > 1:
                    lim_lo = min(x.min(), y.min())
                    lim_hi = max(x.max(), y.max())
                    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                            "k--", lw=0.7, alpha=0.5)   # 1:1 line
            else:
                if x.size >= 3:
                    r, _ = stats.pearsonr(x, y)
                    rho, _ = stats.spearmanr(x, y)
                    ax.text(0.5, 0.6, f"r = {r:+.2f}", ha="center",
                            va="center", fontsize=11, transform=ax.transAxes)
                    ax.text(0.5, 0.35, f"ρ = {rho:+.2f}", ha="center",
                            va="center", fontsize=9, color="gray",
                            transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            if i == k - 1:
                ax.set_xlabel(xname, fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yname, fontsize=9)
            else:
                ax.set_yticklabels([])
    fig.suptitle(f"{title}  [{unit}]", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = {attr: [] for attr in ATTRIBUTES}

    for site in SITES:
        print(f"[04] loading {site} ...")
        bundle = load_site(site)

        for attr in ATTRIBUTES:
            srcs = _sources_for(attr)
            if len(srcs) < 2:
                continue

            # Choose mask: GEDI mask if available, else joint-valid
            m = gedi_mask(bundle, attr)
            if m is None:
                m = joint_valid_mask(bundle, srcs)
            else:
                m = m & joint_valid_mask(bundle, srcs)
            if m.sum() == 0:
                continue

            data = {s: bundle["arrays"][s][m] for s in srcs}
            df_full = pd.DataFrame(data)

            # Density-coloured scatter keeps ALL points — no subsampling needed.
            # High safety cap only guards against pathological memory use.
            df_plot = (df_full.sample(5_000_000, random_state=RANDOM_SEED)
                       if len(df_full) > 5_000_000 else df_full)

            pear  = df_full.corr(method="pearson")
            spear = df_full.corr(method="spearman")
            pd.concat({"pearson": pear, "spearman": spear}).to_csv(
                OUT / f"{site}_{attr}_source_corr.csv")

            for a, b in combinations(srcs, 2):
                summary[attr].append({
                    "site": site, "src_x": a, "src_y": b,
                    "pearson":  pear.loc[a, b],
                    "spearman": spear.loc[a, b],
                    "n_pixels": int(m.sum()),
                })

            out_png = OUT / f"{site}_{attr}_source_pairs.png"
            _pair_plot(df_plot, f"{site} — {attr} across sources",
                       out_png, attr)
            print(f"  {attr} → {out_png.name}")

    for attr, recs in summary.items():
        if not recs:
            continue
        d = pd.DataFrame(recs)
        agg = (d.groupby(["src_x", "src_y"])
                 .agg(pearson_mean=("pearson", "mean"),
                      pearson_sd  =("pearson", "std"),
                      spearman_mean=("spearman", "mean"),
                      spearman_sd  =("spearman", "std"),
                      n_sites=("site", "nunique"))
                 .reset_index())
        agg.to_csv(OUT / f"all_sites_{attr}_source_corr_summary.csv",
                   index=False)
    print("[04] done.")


if __name__ == "__main__":
    main()