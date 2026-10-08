"""
06 — Spatial residual maps and Moran's I.

For Height and Cover (the attributes with a GEDI reference) and for AGBD
(against CCI as a comparator, *not* a reference), compute per-pixel residuals
(candidate − reference), save a residual GeoTIFF, plot the residual map, and
compute global Moran's I to test whether residuals are spatially clustered.
Clustered residuals indicate structured (non-random) error — usually missing
covariates or a regional bias.

Moran's I is computed on a sub-sampled set of pixels with rook-style queen
neighbours via libpysal's KNN (k=8) — pure-numpy fallback if libpysal is
unavailable.

Outputs (under OUTPUT_DIR/06_spatial_residuals/):
    <site>_<attr>_<src>_residual.tif    full-resolution residual raster
    <site>_<attr>_<src>_residual.png    coloured residual map
    morans_i.csv                        I-statistic + permutation p-value
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

sys.path.insert(0, str(Path(__file__).parent))
from config import (SITES, ATTRIBUTES, OUTPUT_DIR, MODELS, RANDOM_SEED)
from io_utils import load_site, gedi_mask


OUT = OUTPUT_DIR / "06_spatial_residuals"
OUT.mkdir(parents=True, exist_ok=True)


# (attribute, reference key) — "reference" loosely here; for AGBD it's a comparator
TARGETS = [
    ("Height", "GEDI_RH98"),
    ("Cover",  "GEDI_Cover"),
    ("AGBD",    "CCI_AGBD"),       # treat as comparator, not truth
]


def _morans_i(values: np.ndarray, coords: np.ndarray,
              k: int = 8, n_perm: int = 199,
              max_n: int = 5000, seed: int = RANDOM_SEED):
    """
    Global Moran's I with KNN-row-standardised weights, permutation p-value.

    coords: (N, 2) array of (row, col) or (lon, lat) — only distances matter.
    Sub-samples to `max_n` points for tractability on large rasters.
    """
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
    # k+1 because the first neighbour is the point itself
    _, nn = tree.query(coords, k=k + 1)
    nn = nn[:, 1:]                         # drop self

    z = values - values.mean()
    s2 = (z ** 2).sum()
    if s2 == 0:
        return 0.0, 1.0, n

    # row-standardised weights: each row sums to 1, equal weights 1/k
    z_nb = z[nn]                           # (n, k)
    numer = (z[:, None] * z_nb).sum() / k  # = sum_i z_i * mean(z_j over neighbours)
    I = (n / s2) * numer / n               # = numer * n / s2 (n cancels: numer/s2 in the right form)
    # Simpler / canonical form: I = sum_i sum_j w_ij z_i z_j / sum_i z_i^2
    # with row-standardised w (sum_j w_ij = 1), so:
    I = (z[:, None] * z_nb).sum() / k / s2

    # Permutation test: shuffle values, keep neighbour structure fixed
    perm_I = np.empty(n_perm)
    for p in range(n_perm):
        zp = rng.permutation(z)
        perm_I[p] = (zp[:, None] * zp[nn]).sum() / k / s2
    p_two = (np.sum(np.abs(perm_I) >= abs(I)) + 1) / (n_perm + 1)
    return float(I), float(p_two), int(n)


def main():
    moran_rows = []

    for site in SITES:
        print(f"[06] loading {site} ...")
        bundle = load_site(site)
        prof = bundle["profile"]
        lon = bundle["lon"]; lat = bundle["lat"]

        for attr, ref_key in TARGETS:
            ref = bundle["arrays"].get(ref_key)
            if ref is None:
                continue

            # Mask: GEDI mask where applicable, else joint-valid
            if ATTRIBUTES[attr]["ref"] is not None:
                m_base = gedi_mask(bundle, attr)
            else:
                m_base = np.isfinite(ref)
            if m_base is None or m_base.sum() == 0:
                continue

            for model in MODELS:
                src = f"{model}_{attr}"
                pred = bundle["arrays"][src]
                m = m_base & np.isfinite(pred) & np.isfinite(ref)
                if m.sum() < 100:
                    continue

                residual = np.full(pred.shape, np.nan, dtype=np.float32)
                residual[m] = (pred[m] - ref[m]).astype(np.float32)

                # ---- Write residual GeoTIFF -----------------------------------
                tif_out = OUT / f"{site}_{attr}_{model}_residual.tif"
                prof_out = prof.copy()
                prof_out.update(dtype="float32", count=1, nodata=np.nan,
                                compress="deflate")
                with rasterio.open(tif_out, "w", **prof_out) as dst:
                    dst.write(residual, 1)

                # ---- Plot map -------------------------------------------------
                vmax = float(np.nanpercentile(np.abs(residual), 98))
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1.0
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(residual, cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax,
                               extent=[lon.min(), lon.max(),
                                       lat.min(), lat.max()],
                               origin="upper", interpolation="nearest")
                ax.set_title(f"{site} — {attr}: {model} − {ref_key}")
                ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
                cb = fig.colorbar(im, ax=ax, shrink=0.8)
                cb.set_label(f"Residual [{ATTRIBUTES[attr]['unit']}]")
                fig.tight_layout()
                fig.savefig(OUT / f"{site}_{attr}_{model}_residual.png",
                            dpi=160, bbox_inches="tight")
                plt.close(fig)

                # ---- Moran's I -------------------------------------------------
                vals   = residual[m]
                coords = np.column_stack([lon[m], lat[m]])
                I, p_val, n_used = _morans_i(vals, coords)
                moran_rows.append({
                    "site": site, "attribute": attr, "model": model,
                    "reference": ref_key,
                    "morans_I": I, "p_value": p_val, "n_subsampled": n_used,
                    "n_valid_pixels": int(m.sum()),
                })
                print(f"  {model}/{attr}: I={I:.3f}  p={p_val:.3f}")

    pd.DataFrame(moran_rows).to_csv(OUT / "morans_i.csv", index=False)
    print(f"[06] wrote {OUT / 'morans_i.csv'}")
    print("[06] done.")


if __name__ == "__main__":
    main()
