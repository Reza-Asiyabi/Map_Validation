"""
01 — Quantitative metrics against GEDI ground truth.

For Height and Cover (the two attributes with a GEDI reference), compute
bias / MAE / RMSE / rRMSE / R^2 / r / rho between each candidate map and the
GEDI reference, per site and pooled. Also reports bootstrap CIs over sites
(cross-site uncertainty) — this is more honest than pixel-level bootstrap
because adjacent pixels are correlated.

Outputs (under OUTPUT_DIR/01_quantitative_metrics/):
    per_site_metrics.csv        long-format: site × attribute × source × metric
    pooled_metrics.csv          metrics over all pixels of all sites
    site_bootstrap_ci.csv       95% CIs across the 10 sites (resampled w/ replacement)
    summary_<attr>.png          bar chart per attribute comparing sources
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (SITES, ATTRIBUTES, OUTPUT_DIR, COLOURS, MODELS,
                    N_BOOTSTRAP, RANDOM_SEED)
from io_utils import load_site, gedi_mask
from metrics import error_metrics


OUT = OUTPUT_DIR / "01_quantitative_metrics"
OUT.mkdir(parents=True, exist_ok=True)


# Attributes with a GEDI reference, and the candidate sources to compare.
TARGETS = {
    "Height": {
        "reference": "GEDI_RH98",
        "sources":   ["PG-CBM_Height", "StruMPL_Height", "Lang_Height"],
    },
    "Cover": {
        "reference": "GEDI_Cover",
        "sources":   ["PG-CBM_Cover", "StruMPL_Cover", "Hansen_Cover"],
    },
}


def main():
    per_site_rows = []
    pooled_buf = {attr: {src: ([], []) for src in cfg["sources"]}
                  for attr, cfg in TARGETS.items()}

    for site in SITES:
        print(f"[01] loading {site} ...")
        bundle = load_site(site)

        for attr, cfg in TARGETS.items():
            mask = gedi_mask(bundle, attr)
            if mask is None or mask.sum() == 0:
                print(f"  {attr}: no GEDI pixels — skipping site")
                continue
            ref = bundle["arrays"][cfg["reference"]][mask]

            for src in cfg["sources"]:
                pred = bundle["arrays"][src][mask]
                m = error_metrics(pred, ref)
                row = {"site": site, "attribute": attr, "source": src, **m}
                per_site_rows.append(row)

                # Accumulate for pooled metrics
                pooled_buf[attr][src][0].append(pred)
                pooled_buf[attr][src][1].append(ref)

    per_site = pd.DataFrame(per_site_rows)
    per_site.to_csv(OUT / "per_site_metrics.csv", index=False)
    print(f"[01] wrote {OUT / 'per_site_metrics.csv'}")

    # ---- Pooled metrics (concatenate all sites) ------------------------------
    pooled_rows = []
    for attr, cfg in TARGETS.items():
        for src in cfg["sources"]:
            preds, refs = pooled_buf[attr][src]
            if not preds:
                continue
            preds = np.concatenate(preds)
            refs  = np.concatenate(refs)
            m = error_metrics(preds, refs)
            pooled_rows.append({"attribute": attr, "source": src, **m})
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT / "pooled_metrics.csv", index=False)
    print(f"[01] wrote {OUT / 'pooled_metrics.csv'}")

    # ---- Bootstrap CI across sites (proper spatial-cluster bootstrap) --------
    rng = np.random.default_rng(RANDOM_SEED)
    site_boot_rows = []
    for attr, cfg in TARGETS.items():
        sub = per_site[per_site["attribute"] == attr]
        if sub.empty:
            continue
        for src in cfg["sources"]:
            ssub = sub[sub["source"] == src]
            if ssub.empty:
                continue
            metrics_to_ci = ["bias", "MAE", "RMSE", "rRMSE", "R2", "r"]
            boot = {k: [] for k in metrics_to_ci}
            arr = {k: ssub[k].to_numpy(dtype=np.float64) for k in metrics_to_ci}
            n = len(ssub)
            for _ in range(N_BOOTSTRAP):
                idx = rng.integers(0, n, size=n)
                for k in metrics_to_ci:
                    vals = arr[k][idx]
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        boot[k].append(vals.mean())
            row = {"attribute": attr, "source": src, "n_sites": n}
            for k in metrics_to_ci:
                v = np.asarray(boot[k])
                if v.size == 0:
                    row[f"{k}_mean"] = row[f"{k}_lo"] = row[f"{k}_hi"] = np.nan
                else:
                    row[f"{k}_mean"] = float(v.mean())
                    row[f"{k}_lo"]   = float(np.percentile(v, 2.5))
                    row[f"{k}_hi"]   = float(np.percentile(v, 97.5))
            site_boot_rows.append(row)
    site_boot = pd.DataFrame(site_boot_rows)
    site_boot.to_csv(OUT / "site_bootstrap_ci.csv", index=False)
    print(f"[01] wrote {OUT / 'site_bootstrap_ci.csv'}")

    # ---- Bar charts: RMSE and bias per source, per attribute -----------------
    for attr in TARGETS:
        sub = site_boot[site_boot["attribute"] == attr]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, metric, title in zip(
            axes, ["RMSE", "bias"],
            [f"RMSE [{ATTRIBUTES[attr]['unit']}]",
             f"Bias [{ATTRIBUTES[attr]['unit']}]"]):
            x = np.arange(len(sub))
            means = sub[f"{metric}_mean"].to_numpy()
            los   = sub[f"{metric}_lo"].to_numpy()
            his   = sub[f"{metric}_hi"].to_numpy()
            err   = np.vstack([means - los, his - means])
            colors = [COLOURS.get(s.split("_")[0] if s.startswith(tuple(MODELS))
                                  else s, "#777") for s in sub["source"]]
            ax.bar(x, means, yerr=err, color=colors, capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["source"], rotation=20, ha="right")
            ax.set_ylabel(title)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{attr} — {metric} (cross-site bootstrap 95% CI)")
        fig.tight_layout()
        out_png = OUT / f"summary_{attr}.png"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[01] wrote {out_png}")

    print("[01] done.")


if __name__ == "__main__":
    main()
