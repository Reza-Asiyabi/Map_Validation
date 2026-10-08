"""
11 — Year-over-year change agreement: corr(Δmodel, ΔGEDI).

A model that *tracks real change* well doesn't just predict the right level
in any given year — it predicts the right inter-annual differences. We compute
per-pixel change between consecutive years for both the model and GEDI, then
correlate them.

For each site, attribute (Height, Cover), model, and consecutive year pair
(2019→2020, 2020→2021, 2021→2022):
    Δmodel = model_{y+1} - model_{y}
    ΔGEDI  = GEDI_{y+1}  - GEDI_{y}
    metrics(Δmodel, ΔGEDI)  via the standard error_metrics function

Crucially we use the JOINT GEDI mask: only pixels where BOTH years have a
valid GEDI observation are included. With GEDI's sparse coverage this is
typically a smaller set than either year alone.

CAVEAT: GEDI in 2019 and 2020 will sample different physical pixels (GEDI is
a footprint sensor), so ΔGEDI is noisier than ΔLang or ΔHansen would be. We
report the joint-pixel count alongside the correlation so weak Δ signals are
visible. Sites with few co-valid pixels will show high-variance correlations.

Outputs (under OUTPUT_DIR/11_year_over_year_change_agreement/):
    delta_metrics.csv              per (site, attr, model, year_pair)
    delta_summary.csv              cross-site mean ± sd
    delta_<attr>_corr.png          correlation timeline per model
    scatter/<site>_<attr>_<model>_<y1>to<y2>.png   ΔGEDI vs Δmodel scatter
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import SITES, YEARS, MODELS, ATTRIBUTES, OUTPUT_DIR, COLOURS
from io_utils import load_site_year, gedi_mask, density_scatter
from metrics import error_metrics


OUT = OUTPUT_DIR / "11_year_over_year_change_agreement"
OUT_SCAT = OUT / "scatter"
OUT.mkdir(parents=True, exist_ok=True)
OUT_SCAT.mkdir(parents=True, exist_ok=True)


TARGETS = {
    "Height": "GEDI_RH98",
    "Cover":  "GEDI_Cover",
}


def _year_pairs(years):
    return [(years[i], years[i+1]) for i in range(len(years)-1)]


def main():
    rows = []
    print("[11] computing year-over-year deltas ...")
    for site in SITES:
        # Load all years up-front so we only hit disk once per site
        bundles = {}
        for year in YEARS:
            try:
                bundles[year] = load_site_year(site, year)
            except FileNotFoundError as e:
                print(f"  [WARN] {site} {year}: {e}")
        if len(bundles) < 2:
            continue

        for attr, ref_key in TARGETS.items():
            for y1, y2 in _year_pairs(YEARS):
                if y1 not in bundles or y2 not in bundles:
                    continue
                b1, b2 = bundles[y1], bundles[y2]

                ref1 = b1["arrays"][ref_key]
                ref2 = b2["arrays"][ref_key]
                # joint GEDI mask: BOTH years must have valid GEDI at that pixel
                mask = np.isfinite(ref1) & np.isfinite(ref2)
                if mask.sum() < 50:
                    continue
                d_gedi = (ref2 - ref1)[mask]

                for model in MODELS:
                    p1 = b1["arrays"][f"{model}_{attr}"]
                    p2 = b2["arrays"][f"{model}_{attr}"]
                    m = mask & np.isfinite(p1) & np.isfinite(p2)
                    if m.sum() < 50:
                        continue
                    d_model = (p2 - p1)[m]
                    d_ref   = (ref2 - ref1)[m]

                    em = error_metrics(d_model, d_ref)
                    row = {
                        "site": site, "attribute": attr, "model": model,
                        "year_from": y1, "year_to": y2,
                        "n_pixels": int(m.sum()),
                        "mean_dmodel": float(d_model.mean()),
                        "mean_dgedi":  float(d_ref.mean()),
                        **em,
                    }
                    rows.append(row)

                    # ---- ΔGEDI vs Δmodel scatter ---------------------------
                    fig, ax = plt.subplots(figsize=(5, 4.5))
                    density_scatter(ax, d_ref, d_model, bins=150,
                                    cmap="viridis", s=3)
                    lim = max(abs(d_ref.min()), abs(d_ref.max()),
                              abs(d_model.min()), abs(d_model.max()), 1e-6)
                    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7, alpha=0.6)
                    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
                    ax.axvline(0, color="k", lw=0.4, alpha=0.4)
                    unit = ATTRIBUTES[attr]["unit"]
                    ax.set_xlabel(f"ΔGEDI {y1}→{y2} [{unit}]")
                    ax.set_ylabel(f"Δ{model} {y1}→{y2} [{unit}]")
                    ax.set_title(f"{site} {attr}: r={em['r']:+.2f}  "
                                 f"RMSE={em['RMSE']:.2f}  n={m.sum():,}")
                    fig.tight_layout()
                    fig.savefig(
                        OUT_SCAT / f"{site}_{attr}_{model}_{y1}to{y2}.png",
                        dpi=150, bbox_inches="tight")
                    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "delta_metrics.csv", index=False)
    print(f"[11] wrote {OUT / 'delta_metrics.csv'}  ({len(df)} rows)")

    # ---- Cross-site summary --------------------------------------------------
    if df.empty:
        return
    agg = (df.groupby(["attribute", "model", "year_from", "year_to"])
             .agg(n_sites=("site", "nunique"),
                  r_mean=("r", "mean"), r_sd=("r", "std"),
                  rmse_mean=("RMSE", "mean"), rmse_sd=("RMSE", "std"),
                  bias_mean=("bias", "mean"), bias_sd=("bias", "std"))
             .reset_index())
    agg.to_csv(OUT / "delta_summary.csv", index=False)

    # ---- Plot: correlation timeline per model -------------------------------
    for attr in TARGETS:
        sub = df[df.attribute == attr]
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for model in MODELS:
            s = sub[sub.model == model]
            if s.empty: continue
            # x = midpoint of year pair
            s = s.copy()
            s["pair_label"] = s["year_from"].astype(str) + "→" + s["year_to"].astype(str)
            agg_m = (s.groupby("pair_label")["r"]
                       .agg(mean="mean", sd="std", n="count").reset_index())
            agg_m["pair_order"] = agg_m["pair_label"].apply(
                lambda p: YEARS.index(int(p.split("→")[0])))
            agg_m = agg_m.sort_values("pair_order")
            x = np.arange(len(agg_m))
            c = COLOURS.get(model, "#444")
            ax.errorbar(x, agg_m["mean"], yerr=agg_m["sd"], marker="o", lw=2,
                        capsize=4, color=c, label=model)
            ax.set_xticks(x); ax.set_xticklabels(agg_m["pair_label"])
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Year pair"); ax.set_ylabel("Pearson r (Δmodel, ΔGEDI)")
        ax.set_title(f"{attr}: change agreement across years (mean ± sd over sites)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"delta_{attr}_corr.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
    print("[11] done.")


if __name__ == "__main__":
    main()
