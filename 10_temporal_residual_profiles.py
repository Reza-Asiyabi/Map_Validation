"""
10 — Temporal residual profiles + per-year RMSE/bias trajectories.

For Height and Cover (the attributes with a GEDI reference):

  (A) Residual longitude profiles per site:
      For each site, plot residual = (model − GEDI) binned by longitude (and
      latitude). One line per (model × year). Eight lines per panel for
      2 models × 4 years. A model that is *temporally consistent* shows years
      stacked on top of each other; a model that drifts shows years fanning out.

  (B) RMSE & bias trajectories across years:
      Per-site RMSE and bias for each (model × year), aggregated across sites
      with bootstrap CIs. Shows whether either model is degrading or
      improving over time.

Outputs (under OUTPUT_DIR/10_temporal_residual_profiles/):
    residual_profiles/<site>_<attr>_<axis>.png
    residual_profiles/<site>_<attr>_<axis>.csv
    trajectories/<attr>_rmse.png
    trajectories/<attr>_bias.png
    per_year_metrics.csv             site × year × model × attribute
    trajectory_bootstrap_ci.csv      cross-site mean ± 95% CI per (year, model, attribute)
    boxplots/temporal_<attr>.png     3-panel figure (bias, |bias|, RMSE) with
                                     year on x-axis and one box per model per year,
                                     showing the distribution ACROSS 10 sites
                                     within each (model, year) cell
    boxplots/headline_<attr>.png     3-panel figure (bias, |bias|, RMSE) with
                                     one box per model, aggregating ALL site-year
                                     observations. The paper-ready "which model wins
                                     overall" summary
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import (SITES, YEARS, MODELS, ATTRIBUTES, OUTPUT_DIR, COLOURS,
                    PROFILE_N_BINS, PROFILE_MIN_PIX_PER_BIN,
                    N_BOOTSTRAP, RANDOM_SEED)
from io_utils import load_site_year, gedi_mask
from metrics import error_metrics


OUT = OUTPUT_DIR / "10_temporal_residual_profiles"
OUT_PROF = OUT / "residual_profiles"
OUT_TRAJ = OUT / "trajectories"
OUT_BOXES = OUT / "boxplots"
OUT_PROF.mkdir(parents=True, exist_ok=True)
OUT_TRAJ.mkdir(parents=True, exist_ok=True)
OUT_BOXES.mkdir(parents=True, exist_ok=True)


TARGETS = {
    "Height": "GEDI_RH98",
    "Cover":  "GEDI_Cover",
}


def _bin_residuals_on_edges(resid, coords, edges):
    """Mean residual per bin, with counts. Bins below min_pix get NaN."""
    finite = np.isfinite(resid) & np.isfinite(coords)
    nb = edges.size - 1
    if finite.sum() == 0:
        return np.full(nb, np.nan), np.zeros(nb, dtype=int)
    r = resid[finite]; c = coords[finite]
    idx = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, nb - 1)
    sums = np.bincount(idx, weights=r, minlength=nb)
    cnts = np.bincount(idx,            minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(cnts > 0, sums / cnts, np.nan)
    return means, cnts


def _make_edges(coords, n_bins):
    c = coords[np.isfinite(coords)]
    if c.size == 0:
        return None
    lo, hi = float(c.min()), float(c.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return np.linspace(lo, hi, n_bins + 1)


def _bootstrap_ci(values, n_boot, seed):
    """Mean and 95% percentile CI of `values` (drops NaN). Returns (mu, lo, hi)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size < 3:
        return float(v.mean()), float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = v[rng.integers(0, v.size, size=v.size)].mean()
    return float(v.mean()), float(np.percentile(means, 2.5)), \
           float(np.percentile(means, 97.5))


def main():
    per_year_rows = []
    # Cache profile data per (site, attr, axis) so we can render one figure per site
    profile_cache = {}

    print("[10] computing per-year residuals across years and sites ...")
    for site in SITES:
        for year in YEARS:
            try:
                bundle = load_site_year(site, year)
            except FileNotFoundError as e:
                print(f"  [WARN] {site} {year}: {e}")
                continue
            for attr, ref_key in TARGETS.items():
                mask = gedi_mask(bundle, attr)
                if mask is None or mask.sum() == 0:
                    continue
                ref = bundle["arrays"][ref_key]
                for model in MODELS:
                    src_key = f"{model}_{attr}"
                    pred = bundle["arrays"][src_key]
                    m = mask & np.isfinite(pred) & np.isfinite(ref)
                    if m.sum() < 50:
                        continue
                    resid = (pred - ref).astype(np.float32)

                    # ---- Per-cell pixel-level metrics (for trajectory) -------
                    pmetrics = error_metrics(pred[m], ref[m])
                    per_year_rows.append({
                        "site": site, "year": year, "model": model,
                        "attribute": attr, **pmetrics,
                    })

                    # ---- Binned residual profiles (lon & lat) ----------------
                    for axis in ("lon", "lat"):
                        coord = bundle[axis][m]
                        # Use the GEDI reference's coordinate range so all
                        # (model, year) lines share the same x grid
                        edges_key = (site, attr, axis)
                        if edges_key not in profile_cache:
                            edges = _make_edges(coord, PROFILE_N_BINS)
                            if edges is None:
                                continue
                            profile_cache[edges_key] = {
                                "edges": edges,
                                "centres": 0.5*(edges[:-1] + edges[1:]),
                                "data": [],   # list of (year, model, means, cnts)
                            }
                        edges = profile_cache[edges_key]["edges"]
                        means, cnts = _bin_residuals_on_edges(
                            resid[m], coord, edges)
                        profile_cache[edges_key]["data"].append(
                            (year, model, means, cnts))

    # ---- Save per-year metrics ----------------------------------------------
    per_year = pd.DataFrame(per_year_rows)
    per_year.to_csv(OUT / "per_year_metrics.csv", index=False)
    print(f"[10] wrote {OUT / 'per_year_metrics.csv'}  ({len(per_year)} rows)")

    # ---- Render residual profile figures, one per (site, attr, axis) --------
    n_figs = 0
    n_skipped = 0
    for (site, attr, axis), payload in profile_cache.items():
        centres = payload["centres"]
        unit = ATTRIBUTES[attr]["unit"]

        # Persist the binned data to CSV for reuse (always, even if plot empty)
        csv = {"bin_centre": centres}
        for year, model, means, _ in payload["data"]:
            csv[f"{model}_{year}"] = means
        pd.DataFrame(csv).to_csv(
            OUT_PROF / f"{site}_{attr}_{axis}.csv", index=False)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        lines_drawn = 0
        for year, model, means, cnts in payload["data"]:
            base_colour = COLOURS.get(model, "#444")
            year_idx = YEARS.index(year)
            alpha = 0.45 + 0.55 * (year_idx / max(1, len(YEARS)-1))
            lw = 1.0 + 0.4 * year_idx
            # Mask sparse bins for cleanliness
            ok = cnts >= PROFILE_MIN_PIX_PER_BIN
            if ok.sum() == 0:
                continue
            ax.plot(centres[ok], means[ok], color=base_colour, alpha=alpha,
                    lw=lw, label=f"{model} {year}")
            lines_drawn += 1
        if lines_drawn == 0:
            plt.close(fig)
            n_skipped += 1
            print(f"  [WARN] no bins met threshold for {site}/{attr}/{axis} "
                  f"— skipping figure (raise data density or lower "
                  f"PROFILE_MIN_PIX_PER_BIN)")
            continue

        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("Longitude (°)" if axis == "lon" else "Latitude (°)")
        ax.set_ylabel(f"Residual ({attr} pred − GEDI) [{unit}]")
        ax.set_title(f"{site} — {attr} residual profile by {axis}")
        ax.legend(fontsize=7, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_PROF / f"{site}_{attr}_{axis}.png",
                    dpi=170, bbox_inches="tight")
        plt.close(fig)
        n_figs += 1
    print(f"[10] wrote {n_figs} residual profile figures"
          + (f" ({n_skipped} skipped — empty)" if n_skipped else ""))

    # ---- Trajectories: per-(year, model) cross-site CI ----------------------
    if per_year.empty:
        print("[10] no per-year metrics computed — skipping trajectories")
        return

    boot_rows = []
    rng_seed = RANDOM_SEED
    for attr in TARGETS:
        for model in MODELS:
            for year in YEARS:
                sub = per_year[(per_year.attribute == attr)
                               & (per_year.model == model)
                               & (per_year.year == year)]
                if sub.empty:
                    continue
                for metric in ("RMSE", "bias", "MAE", "R2"):
                    mu, lo, hi = _bootstrap_ci(sub[metric].to_numpy(),
                                               N_BOOTSTRAP, rng_seed)
                    boot_rows.append({
                        "attribute": attr, "year": year, "model": model,
                        "metric": metric, "n_sites": len(sub),
                        "mean": mu, "lo": lo, "hi": hi,
                    })
                    rng_seed += 1
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "trajectory_bootstrap_ci.csv", index=False)
    print(f"[10] wrote {OUT / 'trajectory_bootstrap_ci.csv'}")

    # ---- Trajectory plots ---------------------------------------------------
    for attr in TARGETS:
        for metric, ylabel in [("RMSE", f"RMSE [{ATTRIBUTES[attr]['unit']}]"),
                               ("bias", f"Bias [{ATTRIBUTES[attr]['unit']}]")]:
            sub = boot[(boot.attribute == attr) & (boot.metric == metric)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for model in MODELS:
                s = sub[sub.model == model].sort_values("year")
                if s.empty:
                    continue
                c = COLOURS.get(model, "#444")
                ax.plot(s["year"], s["mean"], marker="o", lw=2,
                        color=c, label=model)
                ax.fill_between(s["year"], s["lo"], s["hi"],
                                color=c, alpha=0.18)
            if metric == "bias":
                ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(YEARS)
            ax.set_xlabel("Year"); ax.set_ylabel(ylabel)
            ax.set_title(f"{attr}: per-year {metric} (cross-site mean ± 95% CI)")
            ax.grid(alpha=0.3); ax.legend()
            fig.tight_layout()
            fig.savefig(OUT_TRAJ / f"{attr}_{metric.lower()}.png",
                        dpi=180, bbox_inches="tight")
            plt.close(fig)

    # ---- Box plots: temporal (per-year distribution across sites) -----------
    # For each attribute, a 3-panel figure (bias, |bias|, RMSE) with year on
    # x-axis, one box per model within each year. Boxes show the distribution
    # of per-site metrics across the 10 sites for that (model, year). Overlaid
    # site-year points make individual sites visible — with n=10 sites per
    # year, hiding the raw data behind a summary box would be misleading.
    per_year_ext = per_year.copy()
    per_year_ext["abs_bias"] = per_year_ext["bias"].abs()

    BOX_METRICS = [
        ("bias",     "Bias",     True),
        ("abs_bias", "|Bias|",   False),
        ("RMSE",     "RMSE",     False),
    ]

    for attr in TARGETS:
        sub_attr = per_year_ext[per_year_ext["attribute"] == attr]
        if sub_attr.empty:
            continue
        unit = ATTRIBUTES[attr]["unit"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (metric, title, signed) in zip(axes, BOX_METRICS):
            # Build box positions: within each year, one box per model
            n_models = len(MODELS)
            box_w = 0.7 / n_models
            for mi, model in enumerate(MODELS):
                s = sub_attr[sub_attr["model"] == model]
                data_by_year = []
                positions = []
                for yi, year in enumerate(YEARS):
                    vals = s[s["year"] == year][metric].to_numpy()
                    vals = vals[np.isfinite(vals)]
                    # vv = vals.copy()
                    # for nn in range(len(vals)):
                    #     if vals[nn] > 7.8:
                    #         vv = np.delete(vals, nn)
                    # vals = vv
                    data_by_year.append(vals)
                    positions.append(yi + (mi - (n_models - 1) / 2) * box_w)
                colour = COLOURS.get(model, "#444")
                bp = ax.boxplot(
                    data_by_year, positions=positions, widths=box_w * 0.9,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.2),
                    boxprops=dict(facecolor=colour, alpha=0.55,
                                  edgecolor="black", lw=0.6),
                    whiskerprops=dict(color="black", lw=0.6),
                    capprops=dict(color="black", lw=0.6),
                )
                # Overlay individual site points with light jitter
                jrng = np.random.default_rng(RANDOM_SEED + mi)
                for xi, vals in enumerate(data_by_year):
                    if vals.size == 0:
                        continue
                    jitter = jrng.uniform(-box_w * 0.18, box_w * 0.18,
                                          size=vals.size)
                    ax.scatter(np.full(vals.size, positions[xi]) + jitter,
                               vals, s=10, color=colour, alpha=0.7,
                               edgecolors="black", linewidths=0.3, zorder=3)
                # Register the model in the legend once
                ax.plot([], [], color=colour, marker="s", linestyle="",
                        markersize=8, label=model, alpha=0.7,
                        markeredgecolor="black")

            ax.set_xticks(np.arange(len(YEARS)))
            ax.set_xticklabels(YEARS)
            ax.set_xlabel("Year")
            ax.set_ylabel(f"{title} [{unit}]")
            ax.set_title(f"{attr} — {title}")
            ax.grid(alpha=0.3, axis="y")
            if signed:
                ax.axhline(0, color="k", lw=0.5)

        axes[-1].legend(loc="best", fontsize=9, framealpha=0.9)
        fig.suptitle(
            f"{attr}: per-year distribution of per-site metrics across {len(SITES)} sites",
            y=1.02, fontsize=12,
        )
        fig.tight_layout()
        out_png = OUT_BOXES / f"temporal_{attr}.png"
        fig.savefig(out_png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[10] wrote {out_png}")

    # ---- Box plots: headline model comparison (all site-years pooled) --------
    # For each attribute, a 3-panel figure (bias, |bias|, RMSE) with one box
    # per model. Each box aggregates ALL site-year observations (10 × 4 = 40)
    # per model. This is the "which model wins overall" summary — the multi-
    # year visual equivalent of script 09's paired Wilcoxon claim.
    for attr in TARGETS:
        sub_attr = per_year_ext[per_year_ext["attribute"] == attr]
        if sub_attr.empty:
            continue
        unit = ATTRIBUTES[attr]["unit"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (metric, title, signed) in zip(axes, BOX_METRICS):
            positions = np.arange(len(MODELS))
            data_per_model = []
            for model in MODELS:
                vals = sub_attr[sub_attr["model"] == model][metric].to_numpy()
                vals = vals[np.isfinite(vals)]
                data_per_model.append(vals)

            box_colours = [COLOURS.get(m, "#444") for m in MODELS]
            bp = ax.boxplot(
                data_per_model, positions=positions, widths=0.55,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.2),
                whiskerprops=dict(color="black", lw=0.6),
                capprops=dict(color="black", lw=0.6),
            )
            for patch, col in zip(bp["boxes"], box_colours):
                patch.set_facecolor(col)
                patch.set_alpha(0.55)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.6)

            # Overlay all site-year points with jitter
            jrng = np.random.default_rng(RANDOM_SEED + 999)
            for xi, (vals, colour) in enumerate(zip(data_per_model, box_colours)):
                if vals.size == 0:
                    continue
                jitter = jrng.uniform(-0.16, 0.16, size=vals.size)
                ax.scatter(np.full(vals.size, positions[xi]) + jitter, vals,
                           s=14, color=colour, alpha=0.6,
                           edgecolors="black", linewidths=0.3, zorder=3)

            ax.set_xticks(positions)
            ax.set_xticklabels(MODELS)
            ax.set_ylabel(f"{title} [{unit}]")
            ax.set_title(f"{attr} — {title}")
            ax.grid(alpha=0.3, axis="y")
            if signed:
                ax.axhline(0, color="k", lw=0.5)

        # Note the n annotation once, so the reader knows what each dot is
        n_obs = len(sub_attr[sub_attr["model"] == MODELS[0]])
        fig.suptitle(
            f"{attr}: cross-site × cross-year distribution\n"
            f"(each dot = one site-year; n = {n_obs} observations per model)",
            y=1.03, fontsize=12,
        )
        fig.tight_layout()
        out_png = OUT_BOXES / f"headline_{attr}.png"
        fig.savefig(out_png, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[10] wrote {out_png}")

    print("[10] done.")


if __name__ == "__main__":
    main()