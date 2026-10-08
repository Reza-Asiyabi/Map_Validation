"""
03 — Within-model attribute pair scatterplots & correlations.

For each model (PG-CBM, StruMPL) and each site, create a 4x4 pair-plot of the
four attributes (AGB, Height, Cover, Stem). Diagonal = density; lower triangle
= scatter + LOESS; upper triangle = Pearson r (and Spearman ρ).

This is the Python equivalent of your R ggpairs/GGally workflow. We use seaborn
PairGrid for layout and fall back to a custom regplot/loess line.

Outputs (under OUTPUT_DIR/03_attribute_pairs/):
    <site>_<model>_pairs.png
    <site>_<model>_corr.csv              full Pearson + Spearman matrices
    all_sites_<model>_corr_summary.csv   mean ± sd correlation across sites
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
from config import (SITES, MODELS, ATTRIBUTES, OUTPUT_DIR, COLOURS,
                    MAX_PIXELS_FOR_PLOTS, RANDOM_SEED)
from io_utils import load_site, density_scatter


OUT = OUTPUT_DIR / "03_attribute_pairs"
OUT.mkdir(parents=True, exist_ok=True)

ATTRS_ORDER = ["AGB", "Height", "Cover", "Stem"]


def _pair_plot(df: pd.DataFrame, title: str, color: str, out_path: Path):
    """Custom pair plot — diagonal KDE, lower scatter+loess, upper correlation."""
    cols = list(df.columns)
    k = len(cols)
    fig, axes = plt.subplots(k, k, figsize=(2.4 * k, 2.4 * k))

    for i, yname in enumerate(cols):
        for j, xname in enumerate(cols):
            ax = axes[i, j]
            x = df[xname].to_numpy()
            y = df[yname].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]

            if i == j:
                # Diagonal: density
                if x.size > 5:
                    ax.hist(x, bins=40, color=color, alpha=0.6,
                            edgecolor="none", density=True)
                ax.set_yticks([])
            elif i > j:
                # Lower triangle: density-coloured scatter (keeps all points) + LOESS
                if x.size > 0:
                    density_scatter(ax, x, y, bins=200, cmap="viridis", s=3)
                # LOESS — use a binned mean as a robust, dependency-free proxy
                if x.size > 50:
                    order = np.argsort(x)
                    xs, ys = x[order], y[order]
                    nb = 50
                    edges = np.linspace(xs.min(), xs.max(), nb + 1)
                    idx = np.clip(np.searchsorted(edges, xs) - 1, 0, nb - 1)
                    mx = np.bincount(idx, weights=xs,
                                     minlength=nb) / np.maximum(
                                         np.bincount(idx, minlength=nb), 1)
                    my = np.bincount(idx, weights=ys,
                                     minlength=nb) / np.maximum(
                                         np.bincount(idx, minlength=nb), 1)
                    keep = np.bincount(idx, minlength=nb) > 5
                    ax.plot(mx[keep], my[keep], color="red", lw=1)
            else:
                # Upper triangle: correlation values
                if x.size >= 3:
                    r, _   = stats.pearsonr(x, y)
                    rho, _ = stats.spearmanr(x, y)
                    ax.text(0.5, 0.6, f"r = {r:+.2f}", ha="center",
                            va="center", fontsize=11, transform=ax.transAxes)
                    ax.text(0.5, 0.35, f"ρ = {rho:+.2f}", ha="center",
                            va="center", fontsize=9, transform=ax.transAxes,
                            color="gray")
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

    fig.suptitle(title, y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    summary = {m: {pair: [] for pair in combinations(ATTRS_ORDER, 2)}
               for m in MODELS}

    for site in SITES:
        print(f"[03] loading {site} ...")
        bundle = load_site(site)

        for model in MODELS:
            keys = [f"{model}_{a}" for a in ATTRS_ORDER]
            cols = {a: bundle["arrays"][f"{model}_{a}"].ravel()
                    for a in ATTRS_ORDER}
            df = pd.DataFrame(cols).dropna()
            # Density-coloured scatter keeps ALL points — no subsampling needed.
            # (A high safety cap only guards against pathological memory use.)
            if len(df) > 5_000_000:
                df = df.sample(5_000_000, random_state=RANDOM_SEED)

            # Correlation matrix (full data, not just sample)
            full = pd.DataFrame({a: bundle["arrays"][f"{model}_{a}"].ravel()
                                 for a in ATTRS_ORDER}).dropna()
            pear = full.corr(method="pearson")
            spear = full.corr(method="spearman")
            corr = pd.concat({"pearson": pear, "spearman": spear})
            corr.to_csv(OUT / f"{site}_{model}_corr.csv")

            for a, b in combinations(ATTRS_ORDER, 2):
                summary[model][(a, b)].append({
                    "site": site,
                    "pearson":  pear.loc[a, b],
                    "spearman": spear.loc[a, b],
                })

            out_png = OUT / f"{site}_{model}_pairs.png"
            _pair_plot(df, f"{site} — {model}",
                       COLOURS.get(model, "#1b7837"), out_png)
            print(f"  {model} → {out_png.name}")

    # Cross-site summary of correlations per model
    for model in MODELS:
        rows = []
        for (a, b), recs in summary[model].items():
            d = pd.DataFrame(recs)
            rows.append({
                "model": model, "attr_x": a, "attr_y": b,
                "pearson_mean":  d["pearson"].mean(),
                "pearson_sd":    d["pearson"].std(),
                "spearman_mean": d["spearman"].mean(),
                "spearman_sd":   d["spearman"].std(),
                "n_sites": len(d),
            })
        pd.DataFrame(rows).to_csv(
            OUT / f"all_sites_{model}_corr_summary.csv", index=False)
    print("[03] done.")


if __name__ == "__main__":
    main()