"""
05 — Distribution comparison: histograms, Q-Q, and Kolmogorov–Smirnov tests.

Compare the pixel-value distribution of each candidate map against the GEDI
reference (where it exists) or against the other sources (where it does not).
This catches the most common failure mode of regression models: tail
compression / saturation (predictions clip the extremes of the true range).

Outputs (under OUTPUT_DIR/05_distributions/):
    <site>_<attr>_hist_qq.png     two-panel figure: overlaid CDFs + Q-Q
    ks_test_results.csv           D-statistic and p-value per (site, attr, source)
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import (SITES, ATTRIBUTES, OUTPUT_DIR, COLOURS,
                    MAX_PIXELS_FOR_PLOTS, RANDOM_SEED)
from io_utils import load_site, gedi_mask, joint_valid_mask


OUT = OUTPUT_DIR / "05_distributions"
OUT.mkdir(parents=True, exist_ok=True)


def _sources_for(attr: str):
    """Reference (or anchor) + candidates."""
    cfg = ATTRIBUTES[attr]
    candidates = [f"PG-CBM_{attr}", f"StruMPL_{attr}"] + cfg["externals"]
    return cfg["ref"], candidates


def _qq_plot(ax, ref_vals, src_vals, color, label):
    """Empirical Q-Q on shared quantiles (1st to 99th percentile)."""
    qs = np.linspace(0.01, 0.99, 99)
    rq = np.quantile(ref_vals, qs)
    sq = np.quantile(src_vals, qs)
    ax.plot(rq, sq, marker="o", ms=2, lw=0.8, color=color, label=label)


def main():
    ks_rows = []

    for site in SITES:
        print(f"[05] loading {site} ...")
        bundle = load_site(site)

        for attr in ATTRIBUTES:
            ref_name, candidates = _sources_for(attr)
            unit = ATTRIBUTES[attr]["unit"]

            if ref_name is not None:
                m = gedi_mask(bundle, attr)
                anchor = bundle["arrays"][ref_name][m]
                anchor_label = ref_name
                anchor_color = COLOURS.get(ref_name, "#000000")
                cand_arrays = {c: bundle["arrays"][c][m] for c in candidates
                               if c in bundle["arrays"]}
            else:
                # No reference: use joint-valid pixels across candidates,
                # and treat PG-CBM as the anchor for the QQ.
                avail = [c for c in candidates if c in bundle["arrays"]]
                if len(avail) < 2:
                    continue
                m = joint_valid_mask(bundle, avail)
                anchor = bundle["arrays"][avail[0]][m]
                anchor_label = avail[0]
                anchor_color = COLOURS.get("PG-CBM", "#1b7837")
                cand_arrays = {c: bundle["arrays"][c][m]
                               for c in avail[1:]}

            if anchor.size < 100:
                continue
            anchor = anchor[np.isfinite(anchor)]

            fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(11, 4.2))

            # ---- Empirical CDFs --------------------------------------------
            def _cdf(a):
                a = np.sort(a[np.isfinite(a)])
                return a, np.linspace(0, 1, a.size, endpoint=False)

            x, y = _cdf(anchor)
            ax_h.plot(x, y, color=anchor_color, lw=1.8,
                      label=f"{anchor_label} (anchor)")
            for name, vals in cand_arrays.items():
                v = vals[np.isfinite(vals)]
                if v.size < 100:
                    continue
                xx, yy = _cdf(v)
                short = name.split("_")[0]
                c = COLOURS.get(short if short in COLOURS else name, "#888")
                ax_h.plot(xx, yy, color=c, lw=1.3, label=name, alpha=0.85)

                # KS test
                D, p = stats.ks_2samp(anchor, v, mode="asymp")
                ks_rows.append({
                    "site": site, "attribute": attr,
                    "anchor": anchor_label, "source": name,
                    "ks_D": float(D), "ks_p": float(p),
                    "n_anchor": int(anchor.size), "n_source": int(v.size),
                })

            ax_h.set_xlabel(f"{attr} value [{unit}]")
            ax_h.set_ylabel("Empirical CDF")
            ax_h.set_title(f"{site} — {attr}: distribution")
            ax_h.legend(fontsize=8, loc="lower right")
            ax_h.grid(alpha=0.3)

            # ---- Q-Q plot --------------------------------------------------
            for name, vals in cand_arrays.items():
                v = vals[np.isfinite(vals)]
                if v.size < 100:
                    continue
                short = name.split("_")[0]
                c = COLOURS.get(short if short in COLOURS else name, "#888")
                _qq_plot(ax_q, anchor, v, c, name)
            lim_lo = min(anchor.min(),
                         *(a.min() for a in cand_arrays.values() if a.size))
            lim_hi = max(anchor.max(),
                         *(a.max() for a in cand_arrays.values() if a.size))
            ax_q.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                      "k--", lw=0.7, alpha=0.6, label="1:1")
            ax_q.set_xlabel(f"{anchor_label} quantile")
            ax_q.set_ylabel("Source quantile")
            ax_q.set_title(f"Q-Q (1st–99th percentile)")
            ax_q.legend(fontsize=8, loc="best")
            ax_q.grid(alpha=0.3)

            fig.tight_layout()
            out_png = OUT / f"{site}_{attr}_hist_qq.png"
            fig.savefig(out_png, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"  {attr} → {out_png.name}")

    pd.DataFrame(ks_rows).to_csv(OUT / "ks_test_results.csv", index=False)
    print(f"[05] wrote {OUT / 'ks_test_results.csv'}")
    print("[05] done.")


if __name__ == "__main__":
    main()
