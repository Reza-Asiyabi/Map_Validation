"""
14 — Height vs Cover allometric relationship: model vs GEDI.

Unlike script 08 (which fits within-model H/C/S → AGBD and has no external
reference), the H↔C relationship CAN be checked against GEDI: GEDI measures
both RH98 (height) and Cover at the same footprint, so we have a true
reference curve.

Two complementary views per site:

  (A) GEDI-masked (headline figure):
      For each (model, year), restrict to pixels where BOTH GEDI Cover and
      GEDI RH98 are valid, then compute the model's H-vs-C and C-vs-H curves
      on those exact pixels. Compare against GEDI's own curve over the same
      pixels. This is the apples-to-apples accuracy check: "Does the model
      reproduce the H-C coupling that GEDI sees?"

      A model whose H-vs-C curve hugs GEDI's tracks reality. A model that
      diverges has learned a physically inconsistent coupling — even if its
      per-pixel RMSE for H and C individually is acceptable.

  (B) Wall-to-wall (supplementary check):
      For each (model, year), use ALL pixels where the model has both H and C
      finite. Plotted on a SEPARATE figure against the same model's
      GEDI-masked curve. Tells us whether the model's internal H-C coupling
      is consistent everywhere it predicts, not just at GEDI footprints.
      If the wall-to-wall and GEDI-masked curves diverge, the model behaves
      differently in regions GEDI doesn't sample (a sampling-bias warning).

Both directions are computed:
  - C-binned-by-H: at each height, what cover does this model predict?
  - H-binned-by-C: at each cover, what height does this model predict?

Both are physically informative — neither is "the" predictor — so they go
side-by-side as two panels per figure. Pearson r and Spearman ρ are also
computed on the joint pixels, plus a linear OLS slope as a one-number summary
(with the caveat that the relationship is generally non-linear and the slope
is a local linear approximation).

Outputs (under OUTPUT_DIR/14_height_cover_allometry/):
    gedi_masked/<site>_HC_gedi.png          # 2-panel headline figure
    wallcheck/<site>_HC_wallcheck.png       # supplementary wall-to-wall check
    hc_metrics.csv                          # r, rho, OLS slope per (site, year, model, mask, direction)
    hc_curves.csv                           # binned curve points (long format)
    summary_<direction>.png                 # cross-site mean curve, model vs GEDI
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import (SITES, YEARS, MODELS, ATTRIBUTES, OUTPUT_DIR, COLOURS,
                    COVER_SCALE, COVER_UNIT_LABEL)
from io_utils import load_site_year, gedi_mask


OUT = OUTPUT_DIR / "14_height_cover_allometry_new"
OUT_GEDI = OUT / "gedi_masked"
OUT_WALL = OUT / "wallcheck"
OUT.mkdir(parents=True, exist_ok=True)
OUT_GEDI.mkdir(exist_ok=True)
OUT_WALL.mkdir(exist_ok=True)


# ----- Configuration: which models to draw in the figures --------------------
# The full validation still runs for every model in config.MODELS (per-year
# curves, per-year metrics, wallcheck data), so the CSVs remain complete for
# downstream analysis. This constant only controls what appears in the PLOTS.
#
# For a StruMPL-only paper figure, set to ["StruMPL"]. To compare StruMPL and
# PG-CBM on one figure, set to ["StruMPL", "PG-CBM"]. Any model in config.MODELS
# is valid here.
MODELS_TO_PLOT = ["StruMPL"] #, "PG-CBM"


# Bin edges for the two directions of the H↔C curve
# Height: 0 to 25 m (covers African dryland forest range) in 1-m bins
# Cover:  0 to 1 (fraction) or 0 to 100 (percent) in 20 equal bins
HEIGHT_BINS = np.linspace(0, 25, 26)
COVER_BINS  = np.linspace(0, 1 * COVER_SCALE, 21)

MIN_PIX_PER_BIN = 15        # bin retention floor; lower than script 08 because
                            # GEDI-masked pixels are sparse


def _binned_curve(x, y, edges, min_n=MIN_PIX_PER_BIN):
    """Mean y per bin of x, with the per-bin standard deviation.

    Returns (centres_kept, mean_kept, sd_kept, n_kept). Bins with fewer than
    `min_n` samples are dropped (treated as too noisy to plot).
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.empty(0), np.empty(0), np.empty(0), np.empty(0)
    x, y = x[ok], y[ok]
    nb = edges.size - 1
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    out_c, out_m, out_s, out_n = [], [], [], []
    for i in range(nb):
        sel = idx == i
        n = int(sel.sum())
        if n < min_n:
            continue
        out_c.append(float(centres[i]))
        out_m.append(float(y[sel].mean()))
        out_s.append(float(y[sel].std()))
        out_n.append(n)
    return (np.asarray(out_c), np.asarray(out_m),
            np.asarray(out_s), np.asarray(out_n, dtype=int))


def _scalar_metrics(x, y):
    """Pearson r, Spearman rho, OLS slope (y = slope*x + intercept), and n.

    Uses scipy.stats.linregress instead of np.polyfit:
      - returns a clean slope/intercept without raising RankWarning
      - also returns the slope's standard error so we can flag unreliable fits
    Slope is set to NaN if the data has insufficient spread (near-singular fit),
    which avoids reporting a meaningless number for degenerate bins.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(x.size)
    null = dict(r=np.nan, rho=np.nan, slope=np.nan, intercept=np.nan,
                slope_stderr=np.nan, n=n)
    if n < 30:
        return null
    sx, sy = float(x.std()), float(y.std())
    # Treat near-zero spread as degenerate. The threshold is tiny — only
    # catches the genuinely pathological case (all-equal x or y).
    if sx < 1e-6 or sy < 1e-6:
        return null
    r, _   = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    lr = stats.linregress(x, y)
    return dict(r=float(r), rho=float(rho),
                slope=float(lr.slope), intercept=float(lr.intercept),
                slope_stderr=float(lr.stderr), n=n)


def _year_alpha(year_idx, n_years):
    """Light-to-dark line alpha for earlier-to-later years (matches script 10)."""
    return 0.45 + 0.55 * (year_idx / max(1, n_years - 1))


def _plot_headline_figure(site, curves, scalars, out_path):
    """Two-panel figure: C-vs-H (left), H-vs-C (right). GEDI-masked.

    Each source (GEDI and every model in MODELS_TO_PLOT) shown as ONE line
    with a shaded within-bin SD band — symmetric style. The band measures
    the spread of pixel values within each bin, so it's directly
    comparable between GEDI and the model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    direction_specs = [
        ("C_vs_H", axes[0], "Height [m]", f"Cover [{COVER_UNIT_LABEL}]",
         "Cover as a function of Height (GEDI-masked)"),
        ("H_vs_C", axes[1], f"Cover [{COVER_UNIT_LABEL}]", "Height [m]",
         "Height as a function of Cover (GEDI-masked)"),
    ]
    for direction, ax, xlabel, ylabel, title in direction_specs:
        # GEDI reference: solid thick black line + shaded within-bin SD band
        if ("GEDI", None) in curves[direction]:
            cx, my, sd, _ = curves[direction][("GEDI", None)]
            if cx.size:
                ax.plot(cx, my, color="black", lw=2.4, label="GEDI",
                        zorder=10)
                ax.fill_between(cx, my - sd, my + sd, color="black",
                                alpha=0.15, zorder=1)

        # Each configured model: one pooled line + matching shaded band
        for model in MODELS_TO_PLOT:
            if model not in MODELS:
                # Guard against a typo in MODELS_TO_PLOT
                continue
            key = (model, "pooled")
            if key not in curves[direction]:
                continue
            cx, my, sd, _ = curves[direction][key]
            if cx.size == 0:
                continue
            base = COLOURS.get(model, "#444")
            ax.plot(cx, my, color=base, lw=2.2, label=model, zorder=9)
            ax.fill_between(cx, my - sd, my + sd, color=base, alpha=0.18,
                            zorder=1)

        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(alpha=0.3)

    axes[1].legend(fontsize=9, loc="center left",
                   bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    fig.suptitle(f"{site} — Height ↔ Cover allometry vs GEDI\n"
                 "(line = binned mean; shaded band = ±1 SD of pixel values within bin)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_wallcheck_figure(site, curves_gedi, curves_wall, out_path):
    """One row per model in MODELS_TO_PLOT, two columns (directions).
    Each cell shows:
      - GEDI reference: black solid line + shaded band (only in row 0, or
        on every panel for clarity — we choose every panel so each cell is
        self-explanatory).
      - Model pooled-across-years GEDI-masked: coloured solid + band
      - Model pooled-across-years wall-to-wall:   coloured dashed + band
    If solid and dashed overlap, the model's H-C coupling is the same at
    GEDI footprints and everywhere else — a sampling-bias check.
    """
    models_here = [m for m in MODELS_TO_PLOT if m in MODELS]
    if not models_here:
        return
    nrows, ncols = len(models_here), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5*nrows),
                             squeeze=False)
    direction_specs = [
        ("C_vs_H", 0, "Height [m]", f"Cover [{COVER_UNIT_LABEL}]"),
        ("H_vs_C", 1, f"Cover [{COVER_UNIT_LABEL}]", "Height [m]"),
    ]
    for ri, model in enumerate(models_here):
        base = COLOURS.get(model, "#444")
        for direction, ci, xlabel, ylabel in direction_specs:
            ax = axes[ri, ci]
            # GEDI reference in every cell for clarity
            if ("GEDI", None) in curves_gedi[direction]:
                cx, my, sd, _ = curves_gedi[direction][("GEDI", None)]
                if cx.size:
                    ax.plot(cx, my, color="black", lw=2, label="GEDI",
                            zorder=10)
                    ax.fill_between(cx, my - sd, my + sd, color="black",
                                    alpha=0.12, zorder=1)
            # Model GEDI-masked (pooled)
            key = (model, "pooled")
            if key in curves_gedi[direction]:
                cx, my, sd, _ = curves_gedi[direction][key]
                if cx.size:
                    ax.plot(cx, my, color=base, lw=2, ls="-",
                            label=f"{model} GEDI-mask", zorder=9)
                    ax.fill_between(cx, my - sd, my + sd, color=base,
                                    alpha=0.18, zorder=1)
            # Model wall-to-wall (pooled)
            if key in curves_wall[direction]:
                cx, my, sd, _ = curves_wall[direction][key]
                if cx.size:
                    ax.plot(cx, my, color=base, lw=1.8, ls="--",
                            label=f"{model} wall-to-wall", zorder=9)
                    ax.fill_between(cx, my - sd, my + sd, color=base,
                                    alpha=0.10, zorder=1)

            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(
                f"{model} — "
                f"{'Cover vs Height' if ci == 0 else 'Height vs Cover'}"
            )
            ax.grid(alpha=0.3)

    # One legend, top-left
    axes[0, 0].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(f"{site} — wall-to-wall vs GEDI-masked H-C curves\n"
                 "(solid = GEDI-masked pool; dashed = wall-to-wall pool; "
                 "shaded band = ±1 SD within bin)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    metric_rows = []     # scalar metrics per (site, year, model, mask, direction)
    curve_rows  = []     # binned curve points (long format) for re-plotting
    # cross-site accumulators for the summary figure.
    # We keep per-year keys (populated in the second pass) for backward
    # compatibility with any downstream reader, and add per-model "pooled"
    # keys (populated in the third pass) that drive the summary plot.
    accum = {
        d: {("GEDI", None): []}
        for d in ("C_vs_H", "H_vs_C")
    }
    for d in ("C_vs_H", "H_vs_C"):
        for m in MODELS:
            for y in YEARS:
                accum[d][(m, y)] = []
            accum[d][(m, "pooled")] = []

    for site in SITES:
        # Per-site curve dicts, populated below.
        curves_gedi = {"C_vs_H": {}, "H_vs_C": {}}
        curves_wall = {"C_vs_H": {}, "H_vs_C": {}}

        # ---- First pass: pool GEDI pixels across ALL available years -------
        # The H-C physical relationship doesn't change year-to-year; only the
        # set of footprints GEDI happens to sample changes. Pooling gives a
        # denser, smoother reference curve and naturally handles years where
        # GEDI is missing (e.g. Niger 2019).
        # We cache loaded bundles to avoid re-reading them in the second pass.
        bundles = {}
        g_h_pool, g_c_pool = [], []
        years_with_gedi = []
        for year in YEARS:
            try:
                bundles[year] = load_site_year(site, year)
            except FileNotFoundError as e:
                print(f"  [WARN] {site} {year}: {e}")
                continue
            gh = bundles[year]["arrays"]["GEDI_RH98"]
            gc = bundles[year]["arrays"]["GEDI_Cover"]
            joint = np.isfinite(gh) & np.isfinite(gc)
            if joint.sum() == 0:
                continue
            g_h_pool.append(gh[joint])
            g_c_pool.append(gc[joint])
            years_with_gedi.append(year)

        if g_h_pool:
            g_h_all = np.concatenate(g_h_pool)
            g_c_all = np.concatenate(g_c_pool)
            # Cover-vs-Height curve
            cx, my, sd, n = _binned_curve(g_h_all, g_c_all, HEIGHT_BINS)
            curves_gedi["C_vs_H"][("GEDI", None)] = (cx, my, sd, n)
            for c, m, s, nn in zip(cx, my, sd, n):
                curve_rows.append(dict(site=site,
                                       year="pooled_" + "_".join(map(str, years_with_gedi)),
                                       model="GEDI", mask="gedi",
                                       direction="C_vs_H",
                                       bin_centre=c, mean=m, sd=s, n=nn))
                accum["C_vs_H"][("GEDI", None)].append((c, m))
            # Height-vs-Cover curve
            cx, my, sd, n = _binned_curve(g_c_all, g_h_all, COVER_BINS)
            curves_gedi["H_vs_C"][("GEDI", None)] = (cx, my, sd, n)
            for c, m, s, nn in zip(cx, my, sd, n):
                curve_rows.append(dict(site=site,
                                       year="pooled_" + "_".join(map(str, years_with_gedi)),
                                       model="GEDI", mask="gedi",
                                       direction="H_vs_C",
                                       bin_centre=c, mean=m, sd=s, n=nn))
                accum["H_vs_C"][("GEDI", None)].append((c, m))
            # GEDI scalar metrics on the pooled pixels
            gm = _scalar_metrics(g_h_all, g_c_all)
            metric_rows.append(dict(site=site,
                                    year="pooled_" + "_".join(map(str, years_with_gedi)),
                                    model="GEDI", mask="gedi",
                                    direction="HC_joint", **gm))
        else:
            print(f"  [WARN] {site}: GEDI absent in every year — no reference curve")

        # ---- Second pass: per-year model curves ---------------------------
        for year in YEARS:
            if year not in bundles:
                # Already warned in the first pass — silently skip here.
                continue
            bundle = bundles[year]

            # GEDI joint mask for THIS year (may be empty for some years)
            gedi_h_full = bundle["arrays"]["GEDI_RH98"]
            gedi_c_full = bundle["arrays"]["GEDI_Cover"]
            gedi_joint  = np.isfinite(gedi_h_full) & np.isfinite(gedi_c_full)

            for model in MODELS:
                pred_h = bundle["arrays"][f"{model}_Height"]
                pred_c = bundle["arrays"][f"{model}_Cover"]

                # ----- GEDI-masked: same pixels where both GEDI H & C valid
                m_gedi = gedi_joint & np.isfinite(pred_h) & np.isfinite(pred_c)
                if m_gedi.sum() >= 50:
                    h, c = pred_h[m_gedi], pred_c[m_gedi]
                    cx, my, sd, n = _binned_curve(h, c, HEIGHT_BINS)
                    curves_gedi["C_vs_H"][(model, year)] = (cx, my, sd, n)
                    for cc, mm, ss, nn in zip(cx, my, sd, n):
                        curve_rows.append(dict(site=site, year=year, model=model,
                                               mask="gedi", direction="C_vs_H",
                                               bin_centre=cc, mean=mm, sd=ss, n=nn))
                        accum["C_vs_H"][(model, year)].append((cc, mm))
                    cx, my, sd, n = _binned_curve(c, h, COVER_BINS)
                    curves_gedi["H_vs_C"][(model, year)] = (cx, my, sd, n)
                    for cc, mm, ss, nn in zip(cx, my, sd, n):
                        curve_rows.append(dict(site=site, year=year, model=model,
                                               mask="gedi", direction="H_vs_C",
                                               bin_centre=cc, mean=mm, sd=ss, n=nn))
                        accum["H_vs_C"][(model, year)].append((cc, mm))
                    gm = _scalar_metrics(h, c)
                    metric_rows.append(dict(site=site, year=year, model=model,
                                            mask="gedi", direction="HC_joint",
                                            **gm))

                # ----- Wall-to-wall: ALL pixels where model H and C are finite
                m_wall = np.isfinite(pred_h) & np.isfinite(pred_c)
                if m_wall.sum() >= 200:
                    h, c = pred_h[m_wall], pred_c[m_wall]
                    cx, my, sd, n = _binned_curve(h, c, HEIGHT_BINS)
                    curves_wall["C_vs_H"][(model, year)] = (cx, my, sd, n)
                    for cc, mm, ss, nn in zip(cx, my, sd, n):
                        curve_rows.append(dict(site=site, year=year, model=model,
                                               mask="wall", direction="C_vs_H",
                                               bin_centre=cc, mean=mm, sd=ss, n=nn))
                    cx, my, sd, n = _binned_curve(c, h, COVER_BINS)
                    curves_wall["H_vs_C"][(model, year)] = (cx, my, sd, n)
                    for cc, mm, ss, nn in zip(cx, my, sd, n):
                        curve_rows.append(dict(site=site, year=year, model=model,
                                               mask="wall", direction="H_vs_C",
                                               bin_centre=cc, mean=mm, sd=ss, n=nn))
                    wm = _scalar_metrics(h, c)
                    metric_rows.append(dict(site=site, year=year, model=model,
                                            mask="wall", direction="HC_joint",
                                            **wm))

        # ---- Third pass: pool each model's pixels across all years, per site.
        # This is symmetric with GEDI's treatment (pooled across years,
        # one binned curve per model per site) and gives StruMPL a
        # within-bin SD band that measures the same thing as GEDI's band:
        # spread of PIXEL VALUES within each bin, not year drift.
        #
        # The plotting code uses ONLY these pooled entries. Per-year entries
        # are still kept in the CSVs (curve_rows) for downstream analysis.
        for model in MODELS:
            # ---- GEDI-masked pool ------------------------------------------
            g_h_stack, g_c_stack = [], []
            for year in YEARS:
                if year not in bundles:
                    continue
                b = bundles[year]
                gh, gc = b["arrays"]["GEDI_RH98"], b["arrays"]["GEDI_Cover"]
                gedi_joint = np.isfinite(gh) & np.isfinite(gc)
                pred_h = b["arrays"][f"{model}_Height"]
                pred_c = b["arrays"][f"{model}_Cover"]
                m_gedi = gedi_joint & np.isfinite(pred_h) & np.isfinite(pred_c)
                if m_gedi.sum() < 50:
                    continue
                g_h_stack.append(pred_h[m_gedi])
                g_c_stack.append(pred_c[m_gedi])
            if g_h_stack:
                h_all = np.concatenate(g_h_stack)
                c_all = np.concatenate(g_c_stack)
                cx, my, sd, n = _binned_curve(h_all, c_all, HEIGHT_BINS)
                curves_gedi["C_vs_H"][(model, "pooled")] = (cx, my, sd, n)
                for cc, mm in zip(cx, my):
                    accum["C_vs_H"][(model, "pooled")].append((cc, mm))
                cx, my, sd, n = _binned_curve(c_all, h_all, COVER_BINS)
                curves_gedi["H_vs_C"][(model, "pooled")] = (cx, my, sd, n)
                for cc, mm in zip(cx, my):
                    accum["H_vs_C"][(model, "pooled")].append((cc, mm))

            # ---- Wall-to-wall pool ------------------------------------------
            w_h_stack, w_c_stack = [], []
            for year in YEARS:
                if year not in bundles:
                    continue
                b = bundles[year]
                pred_h = b["arrays"][f"{model}_Height"]
                pred_c = b["arrays"][f"{model}_Cover"]
                m_wall = np.isfinite(pred_h) & np.isfinite(pred_c)
                if m_wall.sum() < 200:
                    continue
                w_h_stack.append(pred_h[m_wall])
                w_c_stack.append(pred_c[m_wall])
            if w_h_stack:
                h_all = np.concatenate(w_h_stack)
                c_all = np.concatenate(w_c_stack)
                cx, my, sd, n = _binned_curve(h_all, c_all, HEIGHT_BINS)
                curves_wall["C_vs_H"][(model, "pooled")] = (cx, my, sd, n)
                cx, my, sd, n = _binned_curve(c_all, h_all, COVER_BINS)
                curves_wall["H_vs_C"][(model, "pooled")] = (cx, my, sd, n)

        # ---- Headline figure: GEDI-masked, two panels ------------------------
        n_gedi_lines = sum(len(curves_gedi[d]) for d in ("C_vs_H","H_vs_C"))
        if n_gedi_lines == 0:
            print(f"  [WARN] {site}: no GEDI-masked curves — skipping headline figure")
        else:
            _plot_headline_figure(site, curves_gedi, None,
                                  OUT_GEDI / f"{site}_HC_gedi.png")

        # ---- Wall-to-wall supplementary check -------------------------------
        n_wall_lines = sum(len(curves_wall[d]) for d in ("C_vs_H","H_vs_C"))
        if n_wall_lines == 0:
            print(f"  [WARN] {site}: no wall-to-wall curves — skipping wallcheck figure")
        else:
            _plot_wallcheck_figure(site, curves_gedi, curves_wall,
                                   OUT_WALL / f"{site}_HC_wallcheck.png")

    # ---- Persist tables -----------------------------------------------------
    pd.DataFrame(metric_rows).to_csv(OUT / "hc_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "hc_curves.csv", index=False)
    print(f"[14] wrote {OUT / 'hc_metrics.csv'} and hc_curves.csv")

    # ---- Cross-site summary curves (GEDI-masked only) -----------------------
    # Uses per-site pooled-across-years curves as inputs. GEDI and each model
    # in MODELS_TO_PLOT get symmetric treatment: line + shaded band. The band
    # here is cross-site SD at each x value (not within-bin SD as in the
    # per-site figures) — it measures site-to-site variation in the mean
    # curve, which is the honest cross-site uncertainty band.
    for direction, edges, xlabel, ylabel in [
        ("C_vs_H", HEIGHT_BINS, "Height [m]",        f"Cover [{COVER_UNIT_LABEL}]"),
        ("H_vs_C", COVER_BINS,  f"Cover [{COVER_UNIT_LABEL}]",  "Height [m]"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))

        # GEDI summary line + band (cross-site SD around the mean)
        rec = accum[direction][("GEDI", None)]
        if rec:
            d = (pd.DataFrame(rec, columns=["x", "y"])
                   .groupby("x")["y"].agg(["mean", "std"]).reset_index())
            ax.plot(d["x"], d["mean"], color="black", lw=2.4, label="GEDI",
                    zorder=10)
            ax.fill_between(d["x"], d["mean"] - d["std"],
                            d["mean"] + d["std"], color="black",
                            alpha=0.15, zorder=1)

        # Each configured model: pooled-per-site curve, cross-site mean + SD
        for model in MODELS_TO_PLOT:
            if model not in MODELS:
                continue
            rec = accum[direction].get((model, "pooled"), [])
            if not rec:
                continue
            d = (pd.DataFrame(rec, columns=["x", "y"])
                   .groupby("x")["y"].agg(["mean", "std"]).reset_index())
            base = COLOURS.get(model, "#444")
            ax.plot(d["x"], d["mean"], color=base, lw=2.2, label=model,
                    zorder=9)
            ax.fill_between(d["x"], d["mean"] - d["std"],
                            d["mean"] + d["std"], color=base, alpha=0.18,
                            zorder=1)

        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(
            f"Cross-site mean H-C curve "
            f"({'Cover vs Height' if direction == 'C_vs_H' else 'Height vs Cover'})\n"
            "GEDI-masked pixels, all sites — band = ±1 SD across sites"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(OUT / f"summary_{direction}.png", dpi=180,
                    bbox_inches="tight")
        plt.close(fig)

    print("[14] done.")


if __name__ == "__main__":
    main()