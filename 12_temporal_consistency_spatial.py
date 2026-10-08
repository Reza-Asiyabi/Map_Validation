"""
12 — Spatial-pattern temporal consistency.

Three complementary diagnostics that ask whether the spatial pattern of model
behaviour is stable across years:

  (A) Moran's I of residuals per (site, year, model, attr):
      Are residuals spatially clustered? A model whose Moran's I changes a lot
      year-to-year has unstable spatial error structure.

  (B) Residual-map similarity across years:
      For each (site, attribute, model), the Pearson correlation between every
      pair of years' residual maps over their common valid mask. Plotted as a
      4x4 correlation matrix. Stable spatial error structure -> values near +1.
      (This is the mathematically equivalent quantity to an RV coefficient for
      a pair of 1-D arrays, and is the standard inter-map similarity measure
      in geosciences.)

  (C) Per-pixel temporal z-anomaly:
      For each pixel, compute z = (value_y - mean_over_years) / sd_over_years.
      The MAX over years of |z| highlights pixels where the model jumps
      around unphysically. Saved as a GeoTIFF + PNG per (site, attr, model).

Outputs (under OUTPUT_DIR/12_temporal_consistency_spatial/):
    morans_i_per_year.csv
    map_similarity_<site>_<attr>_<model>.png       4x4 Pearson r matrix
    map_similarity_summary.csv                     pairwise correlations
    z_anomaly/<site>_<attr>_<model>_maxabs_z.tif   per-site z-anomaly raster
    z_anomaly/<site>_<attr>_<model>_maxabs_z.png   coloured map
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

sys.path.insert(0, str(Path(__file__).parent))
from config import SITES, YEARS, MODELS, ATTRIBUTES, OUTPUT_DIR, COLOURS, RANDOM_SEED
from io_utils import load_site_year, gedi_mask


OUT = OUTPUT_DIR / "12_temporal_consistency_spatial"
OUT_Z = OUT / "z_anomaly"
OUT.mkdir(parents=True, exist_ok=True)
OUT_Z.mkdir(parents=True, exist_ok=True)


# Attributes for which we have a per-pixel reference (residual = pred - ref)
RESID_TARGETS = {
    "Height": "GEDI_RH98",
    "Cover":  "GEDI_Cover",
}


def _morans_i(values, coords, k=8, n_perm=199, max_n=4000, seed=RANDOM_SEED):
    """Same KNN-row-standardised Moran's I as script 06. Sub-samples for speed."""
    rng = np.random.default_rng(seed)
    n = values.size
    if n < 100:
        return np.nan, np.nan, 0
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        values = values[idx]; coords = coords[idx]
        n = values.size

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, nn = tree.query(coords, k=k+1)
    nn = nn[:, 1:]

    z = values - values.mean()
    s2 = (z**2).sum()
    if s2 == 0:
        return 0.0, 1.0, n
    z_nb = z[nn]
    I = (z[:, None] * z_nb).sum() / k / s2
    perm = np.empty(n_perm)
    for p in range(n_perm):
        zp = rng.permutation(z)
        perm[p] = (zp[:, None] * zp[nn]).sum() / k / s2
    p_two = (np.sum(np.abs(perm) >= abs(I)) + 1) / (n_perm + 1)
    return float(I), float(p_two), int(n)


def _residual_array(bundle, attr, model):
    """Return per-pixel residual = (model - GEDI) where both are finite, else NaN."""
    ref_key = RESID_TARGETS[attr]
    ref = bundle["arrays"][ref_key]
    pred = bundle["arrays"][f"{model}_{attr}"]
    out = pred - ref
    bad = ~(np.isfinite(ref) & np.isfinite(pred))
    out = out.astype(np.float32)
    out[bad] = np.nan
    return out


def main():
    # ============================================================
    # (A) Moran's I of residuals per (site, year, model, attribute)
    # ============================================================
    morans_rows = []
    # We need ALL per-year residual rasters in memory at once for parts (B) & (C),
    # but loading bundles repeatedly is expensive. Cache one site at a time.
    print("[12] processing sites ...")

    sim_rows = []

    for site in SITES:
        bundles = {}
        for y in YEARS:
            try:
                bundles[y] = load_site_year(site, y)
            except FileNotFoundError as e:
                print(f"  [WARN] {site} {y}: {e}")
        if len(bundles) < 2:
            continue
        # use the latest available bundle's coords/profile as the reference
        any_year = max(bundles.keys())
        lon = bundles[any_year]["lon"]; lat = bundles[any_year]["lat"]
        profile = bundles[any_year]["profile"]

        for attr in RESID_TARGETS:
            for model in MODELS:
                # ---- Per-year residual rasters ---------------------------
                per_year_resid = {}
                for y, b in bundles.items():
                    per_year_resid[y] = _residual_array(b, attr, model)

                # ---- (A) Moran's I each year ----------------------------
                for y, R in per_year_resid.items():
                    m = np.isfinite(R)
                    if m.sum() < 100:
                        morans_rows.append({
                            "site": site, "year": y, "attribute": attr,
                            "model": model, "morans_I": np.nan,
                            "p_value": np.nan, "n_pixels": int(m.sum())})
                        continue
                    coords = np.column_stack([lon[m], lat[m]])
                    I, p, n = _morans_i(R[m], coords)
                    morans_rows.append({
                        "site": site, "year": y, "attribute": attr,
                        "model": model, "morans_I": I,
                        "p_value": p, "n_pixels": int(m.sum()),
                        "n_subsampled": n})

                # ---- (B) Pairwise residual-map similarity ---------------
                years_ord = sorted(per_year_resid.keys())
                K = len(years_ord)
                R_mat = np.full((K, K), np.nan)
                for i, y1 in enumerate(years_ord):
                    for j, y2 in enumerate(years_ord):
                        a, b = per_year_resid[y1], per_year_resid[y2]
                        ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() < 50:
                            continue
                        x = a[ok]; z = b[ok]
                        if x.std() == 0 or z.std() == 0:
                            continue
                        r = float(np.corrcoef(x, z)[0, 1])
                        R_mat[i, j] = r
                        if i < j:
                            sim_rows.append({
                                "site": site, "attribute": attr, "model": model,
                                "year_x": y1, "year_y": y2,
                                "pearson_r": r, "n_pixels": int(ok.sum())})

                # Heatmap figure
                fig, ax = plt.subplots(figsize=(4.5, 4))
                im = ax.imshow(R_mat, vmin=-1, vmax=1, cmap="RdBu_r")
                ax.set_xticks(range(K)); ax.set_yticks(range(K))
                ax.set_xticklabels(years_ord); ax.set_yticklabels(years_ord)
                for i in range(K):
                    for j in range(K):
                        if np.isfinite(R_mat[i, j]):
                            ax.text(j, i, f"{R_mat[i,j]:.2f}", ha="center",
                                    va="center", fontsize=9,
                                    color="white" if abs(R_mat[i,j])>0.5 else "black")
                fig.colorbar(im, ax=ax, label="Pearson r")
                ax.set_title(f"{site} — {attr} {model}\nresidual-map similarity")
                fig.tight_layout()
                fig.savefig(
                    OUT / f"map_similarity_{site}_{attr}_{model}.png",
                    dpi=150, bbox_inches="tight")
                plt.close(fig)

                # ---- (C) Per-pixel max|z| anomaly across years ----------
                stack = np.stack([per_year_resid[y] for y in years_ord], axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mu = np.nanmean(stack, axis=0)
                    sd = np.nanstd(stack, axis=0)
                    z = (stack - mu[None, ...]) / np.where(sd > 1e-6, sd, np.nan)
                    maxabs = np.nanmax(np.abs(z), axis=0).astype(np.float32)

                # Write GeoTIFF
                tif = OUT_Z / f"{site}_{attr}_{model}_maxabs_z.tif"
                prof_out = profile.copy()
                prof_out.update(dtype="float32", count=1, nodata=np.nan,
                                compress="deflate")
                with rasterio.open(tif, "w", **prof_out) as dst:
                    dst.write(maxabs, 1)

                # Plot map
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                vmax = float(np.nanpercentile(maxabs, 98))
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1.0
                im = ax.imshow(maxabs, cmap="magma", vmin=0, vmax=vmax,
                               extent=[lon.min(), lon.max(),
                                       lat.min(), lat.max()],
                               origin="upper", interpolation="nearest")
                ax.set_title(f"{site} — {attr} {model}\nmax|z| across years")
                ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                cb = fig.colorbar(im, ax=ax, shrink=0.8)
                cb.set_label("max |z-score|")
                fig.tight_layout()
                fig.savefig(OUT_Z / f"{site}_{attr}_{model}_maxabs_z.png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)

    pd.DataFrame(morans_rows).to_csv(OUT / "morans_i_per_year.csv", index=False)
    pd.DataFrame(sim_rows   ).to_csv(OUT / "map_similarity_summary.csv", index=False)
    print(f"[12] wrote morans_i_per_year.csv and map_similarity_summary.csv")
    print("[12] done.")


if __name__ == "__main__":
    main()
