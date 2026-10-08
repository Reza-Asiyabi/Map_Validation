"""
08 — Allometric consistency check.

A model that's internally consistent should reproduce the well-known
allometric couplings between forest attributes:
    AGB grows monotonically with Height
    AGB grows monotonically with Cover
    AGB grows (more weakly) with Stem density
    Height × Cover × Stem jointly explain most of AGB variability

For each model, we:
  1. Bin AGB by a predictor (Height / Cover / Stem) and report the response
     curve.  A flat or decreasing curve is a red flag.
  2. Fit an OLS  AGB ~ a*H + b*C + c*S + d  per site, report R^2 and
     coefficient signs.  Coefficients should be positive.
  3. Compare PG-CBM vs StruMPL allometric curves on the same axes.

Coefficient scaling: because the three predictors have very different
native ranges (Height ~ 0-25 m, Cover ~ 0-1 or 0-100 per COVER_UNITS, Stem ~ 0-1000 stems/ha),
raw regression coefficients are on incommensurable scales and cannot be
compared directly ("which predictor matters most?" is unanswerable from
the raw values). We therefore fit the OLS on GLOBALLY z-scored predictors:
each predictor is standardised using its mean and SD computed over ALL
joint-valid pixels across ALL sites (one global pass), then per-site OLS
is fitted in these units. The reported coefficients then read as "one
standard deviation of that predictor across the whole study region
translates into this many Mg/ha of AGB". Because the standardisation is
global, coefficients are directly comparable BOTH between predictors at a
given site AND between sites. R^2 is unchanged by this rescaling.

For a physically-interpretable complement, we also report per-site "range
contributions": for each predictor and each site, the coefficient
multiplied by that predictor's SD across the WHOLE study region gives the
change in AGB associated with the natural range of that predictor at that
site — how much AGB the predictor accounts for at each site in physical
Mg/ha units.

Outputs (under OUTPUT_DIR/08_allometric/):
    allometric_curves_<predictor>.png        cross-site mean curve per model
    allometric_ols_per_site.csv              standardised R^2 + coefficients per site/model
    allometric_curves_per_site/<site>.png    site-by-site detail
    allometric_global_scaling.csv            global means/SDs used for standardisation
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import SITES, MODELS, ATTRIBUTES, OUTPUT_DIR, COLOURS, COVER_SCALE
from io_utils import load_site, joint_valid_mask


OUT = OUTPUT_DIR / "08_allometric"
OUT.mkdir(parents=True, exist_ok=True)
DETAIL_DIR = OUT / "allometric_curves_per_site"
DETAIL_DIR.mkdir(exist_ok=True)


PREDICTORS = {
    "Height": np.linspace(0, 25, 26),
    "Cover":  np.linspace(0, 1 * COVER_SCALE, 21),
    "Stem":   np.linspace(0, 1000, 21),
}


def _binned_curve(x, y, edges, min_n=30):
    """Return (centres, mean_y, sd_y) per bin where bin has >= min_n samples."""
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1,
                  0, edges.size - 2)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means, sds = [], []
    keep_centres = []
    for i in range(edges.size - 1):
        sel = idx == i
        if sel.sum() < min_n:
            continue
        means.append(float(np.mean(y[sel])))
        sds.append(float(np.std(y[sel])))
        keep_centres.append(centres[i])
    return np.array(keep_centres), np.array(means), np.array(sds)


def _ols_fit_standardised(H, C, S, AGB,
                           g_means: dict, g_sds: dict):
    """OLS on GLOBALLY z-scored predictors. Returns (coeffs, R^2, obs_ranges)
    where coeffs are the standardised beta coefficients and obs_ranges is a
    dict giving each predictor's observed range (max - min) at this site,
    used later for the per-site range-contribution report.

    Standardisation:
        H_std = (H - g_means['Height']) / g_sds['Height']
        C_std = (C - g_means['Cover'])  / g_sds['Cover']
        S_std = (S - g_means['Stem'])   / g_sds['Stem']

    A returned coef of, say, 15.0 for Height means: "a 1-SD change in Height
    (as measured across the whole study region) is associated with a
    +15 Mg/ha change in AGB at this site". This makes the three coefficients
    directly comparable at a given site AND across sites.

    R^2 is invariant under linear rescaling of the predictors so it matches
    what a raw-scale fit would give.
    """
    H_std = (H - g_means['Height']) / g_sds['Height']
    C_std = (C - g_means['Cover'])  / g_sds['Cover']
    S_std = (S - g_means['Stem'])   / g_sds['Stem']

    X = np.column_stack([H_std, C_std, S_std, np.ones_like(H_std)])
    coef, *_ = np.linalg.lstsq(X, AGB, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((AGB - pred) ** 2))
    ss_tot = float(np.sum((AGB - AGB.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    obs_ranges = {
        'Height': float(np.max(H) - np.min(H)),
        'Cover':  float(np.max(C) - np.min(C)),
        'Stem':   float(np.max(S) - np.min(S)),
    }
    return coef, r2, obs_ranges


def _compute_global_scaling() -> dict[str, dict[str, dict[str, float]]]:
    """First pass across all sites: for each model, pool the joint-valid
    Height / Cover / Stem pixels and compute the global mean and SD.
    Standardisation is per-MODEL because PG-CBM and StruMPL predict on
    different overall scales and we want each model's coefficients
    interpreted in that model's own units.

    Returns:
        {model: {'Height': {'mean': ..., 'sd': ...}, 'Cover': ..., 'Stem': ...}}
    """
    scaling: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        print(f"[08] first pass: computing global scaling for {model}")
        pooled = {'Height': [], 'Cover': [], 'Stem': []}
        for site in SITES:
            try:
                bundle = load_site(site)
            except Exception as e:
                print(f"  [WARN] {site}: {e}")
                continue
            keys = [f"{model}_AGB", f"{model}_Height",
                    f"{model}_Cover", f"{model}_Stem"]
            m = joint_valid_mask(bundle, keys)
            if m.sum() < 200:
                continue
            pooled['Height'].append(bundle["arrays"][f"{model}_Height"][m])
            pooled['Cover'].append (bundle["arrays"][f"{model}_Cover"][m])
            pooled['Stem'].append  (bundle["arrays"][f"{model}_Stem"][m])
        entry = {}
        for pname, chunks in pooled.items():
            if not chunks:
                entry[pname] = {'mean': np.nan, 'sd': np.nan, 'n': 0}
                continue
            allv = np.concatenate(chunks)
            entry[pname] = {
                'mean': float(np.mean(allv)),
                'sd':   float(np.std(allv, ddof=1)),
                'n':    int(allv.size),
            }
        scaling[model] = entry
    return scaling


def main():
    # ---- First pass: compute global scaling per model -----------------------
    scaling = _compute_global_scaling()

    # Persist the scaling constants for reproducibility / paper table
    scaling_rows = []
    for model, m_scale in scaling.items():
        for pname, stats in m_scale.items():
            scaling_rows.append({
                "model": model, "predictor": pname,
                "global_mean": stats["mean"],
                "global_sd":   stats["sd"],
                "n_pixels":    stats.get("n", 0),
            })
    pd.DataFrame(scaling_rows).to_csv(
        OUT / "allometric_global_scaling.csv", index=False)
    print(f"[08] wrote allometric_global_scaling.csv")

    ols_rows = []
    # accumulate curves per model × predictor across sites
    accum = {(m, p): [] for m in MODELS for p in PREDICTORS}

    for site in SITES:
        print(f"[08] loading {site} ...")
        bundle = load_site(site)

        site_fig, site_axes = plt.subplots(1, 3, figsize=(14, 4))

        for model in MODELS:
            keys = [f"{model}_AGB", f"{model}_Height",
                    f"{model}_Cover", f"{model}_Stem"]
            m = joint_valid_mask(bundle, keys)
            if m.sum() < 200:
                continue
            agb = bundle["arrays"][f"{model}_AGB"][m]
            h   = bundle["arrays"][f"{model}_Height"][m]
            c   = bundle["arrays"][f"{model}_Cover"][m]
            s   = bundle["arrays"][f"{model}_Stem"][m]

            # ---- Standardised OLS fit --------------------------------------
            # Predictors are z-scored using the GLOBAL mean/SD computed above
            # so coefficients are comparable between predictors and sites.
            g_means_flat = {k: v["mean"] for k, v in scaling[model].items()}
            g_sds_flat   = {k: v["sd"]   for k, v in scaling[model].items()}
            coef, r2, obs_ranges = _ols_fit_standardised(
                h, c, s, agb, g_means_flat, g_sds_flat,
            )
            # coef_std_Height, coef_std_Cover, coef_std_Stem, std_intercept
            # Range contributions: coef_std × (site_range / global_sd) gives
            # the AGB change (Mg/ha) associated with the observed range of
            # that predictor at this specific site. This is the physically-
            # interpretable "how much AGB does the observed range of this
            # predictor account for at this site" quantity.
            range_h = obs_ranges["Height"] / scaling[model]["Height"]["sd"] * coef[0]
            range_c = obs_ranges["Cover"]  / scaling[model]["Cover"]["sd"]  * coef[1]
            range_s = obs_ranges["Stem"]   / scaling[model]["Stem"]["sd"]   * coef[2]

            ols_rows.append({
                "site":  site, "model": model,
                # Standardised beta coefficients (Mg/ha per one global SD
                # of the predictor). Directly comparable between predictors
                # and between sites.
                "coef_std_Height": coef[0],
                "coef_std_Cover":  coef[1],
                "coef_std_Stem":   coef[2],
                "std_intercept":   coef[3],
                "R2":              r2,
                # Per-site range contributions (Mg/ha across the observed
                # range of the predictor at this specific site). Physically
                # interpretable.
                "range_contrib_Height_Mgha": range_h,
                "range_contrib_Cover_Mgha":  range_c,
                "range_contrib_Stem_Mgha":   range_s,
                # Sample size and observed ranges for reference
                "n_pixels":                  int(m.sum()),
                "obs_range_Height":          obs_ranges["Height"],
                "obs_range_Cover":           obs_ranges["Cover"],
                "obs_range_Stem":            obs_ranges["Stem"],
            })

            # ---- Curves -----------------------------------------------------
            for ax, (pname, edges) in zip(site_axes, PREDICTORS.items()):
                x = {"Height": h, "Cover": c, "Stem": s}[pname]
                cx, my, sd = _binned_curve(x, agb, edges)
                if cx.size > 0:
                    color = COLOURS[model]
                    ax.plot(cx, my, color=color, lw=1.7, label=model)
                    ax.fill_between(cx, my - sd, my + sd,
                                    color=color, alpha=0.15)
                    accum[(model, pname)].append((cx, my))

        for ax, pname in zip(site_axes, PREDICTORS):
            ax.set_xlabel(f"{pname} [{ATTRIBUTES[pname]['unit']}]")
            ax.set_ylabel("AGB [Mg/ha]")
            ax.set_title(f"{site}: AGB vs {pname}")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
        site_fig.tight_layout()
        site_fig.savefig(DETAIL_DIR / f"{site}.png",
                         dpi=160, bbox_inches="tight")
        plt.close(site_fig)

    pd.DataFrame(ols_rows).to_csv(
        OUT / "allometric_ols_per_site.csv", index=False)

    # ---- Cross-site mean curves ---------------------------------------------
    for pname in PREDICTORS:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for model in MODELS:
            curves = accum[(model, pname)]
            if not curves:
                continue
            # Interpolate each site's curve onto common x-grid, then average
            edges = PREDICTORS[pname]
            xg = 0.5 * (edges[:-1] + edges[1:])
            mat = np.full((len(curves), xg.size), np.nan)
            for i, (cx, my) in enumerate(curves):
                mat[i] = np.interp(xg, cx, my, left=np.nan, right=np.nan)
            mean = np.nanmean(mat, axis=0)
            sd   = np.nanstd(mat, axis=0)
            color = COLOURS[model]
            ax.plot(xg, mean, color=color, lw=2, label=model)
            ax.fill_between(xg, mean - sd, mean + sd,
                            color=color, alpha=0.2)
        ax.set_xlabel(f"{pname} [{ATTRIBUTES[pname]['unit']}]")
        ax.set_ylabel("AGB [Mg/ha]")
        ax.set_title(f"Allometric curve: AGB vs {pname} (mean ± sd across sites)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"allometric_curves_{pname}.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("[08] done.")


if __name__ == "__main__":
    main()