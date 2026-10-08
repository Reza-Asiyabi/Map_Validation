"""
aggregate_04_agb_agreement.py — StruMPL AGB agreement against external
products, per site x year and aggregated across the 40 site-year cells.

Produces the exact numbers needed for the paper's "AGB Agreement Against
External Products" table and paragraph.

For each of the ten test sites and each of the four years (2019-2022), and
for each of the three comparators (CCI Biomass, GEDI L4B, PG-CBM), we
compute the pixel-level Pearson correlation and RMSE between StruMPL AGB
and the comparator on the joint-valid pixel mask (pixels finite for both).
Then we aggregate across the 40 site-year cells to produce the cross-site
x cross-year mean +/- SD per pair.

This is spatial *agreement*, not accuracy: none of CCI, L4B, or PG-CBM is
a per-pixel ground reference. The correlation and RMSE reflect the
combined uncertainty of both products in each pair.

Also computed for the last paragraph of the subsection:
  - site-mean AGB per site per year (drives the high/low biomass split)
  - a wide-format "per-cell disagreement" table with StruMPL and each of
    the three comparators, so you can inspect which sites drive the
    largest disagreements.

Outputs (under OUTPUT_DIR/04_agb_agreement/):
    per_cell_metrics.csv        one row per (site, year, comparator) with
                                Pearson r, RMSE, and n_pixels
    site_mean_agb.csv           one row per (site, year) with the site
                                mean of StruMPL AGB and each comparator
    agreement_summary.csv       one row per comparator with cross-site x
                                cross-year mean and SD of r and RMSE.
                                THESE ARE THE TABLE NUMBERS.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import SITES, YEARS, OUTPUT_DIR
from io_utils import load_site_year


OUT = OUTPUT_DIR / "04_agb_agreement"
OUT.mkdir(parents=True, exist_ok=True)


# Comparators are keyed by the bundle array name (left) and given a short
# label used in output tables (right). Adjust these if your bundle uses
# different keys.
COMPARATORS: dict[str, str] = {
    "CCI_AGB":      "CCI",
    "GEDI_L4B_AGB": "L4B",
    "PG-CBM_AGB":   "PG-CBM",
}

# Minimum finite-pixel count in a site-year cell for its metrics to be
# reported. Below this the estimates are too noisy to average.
MIN_PIXELS_PER_CELL = 500


def _pair_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Pearson r and RMSE between two 1-D arrays over their joint-finite mask.
    Returns NaN when the joint mask has fewer than MIN_PIXELS_PER_CELL pixels,
    or when either array has zero variance."""
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < MIN_PIXELS_PER_CELL:
        return dict(n_pixels=n, r=np.nan, rmse=np.nan)
    x, y = a[ok], b[ok]
    if x.std() < 1e-9 or y.std() < 1e-9:
        return dict(n_pixels=n, r=np.nan, rmse=np.nan)
    r    = float(stats.pearsonr(x, y)[0])
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    return dict(n_pixels=n, r=r, rmse=rmse)


def _compute_per_cell() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (site, year, comparator) metrics AND per (site, year) site-mean
    AGB for every source. Returns (per_cell, site_means)."""
    per_cell_rows = []
    site_mean_rows = []
    for site in SITES:
        for year in YEARS:
            try:
                bundle = load_site_year(site, year)
            except FileNotFoundError as e:
                print(f"  [WARN] {site} {year}: {e}")
                continue

            arrays = bundle["arrays"]
            if "StruMPL_AGB" not in arrays:
                print(f"  [WARN] {site} {year}: StruMPL_AGB missing")
                continue
            strumpl = arrays["StruMPL_AGB"].ravel()

            # Site-mean AGB for each source that is present. The mean is
            # over finite pixels only; a source absent from the bundle
            # gets NaN.
            row = {"site": site, "year": int(year),
                   "StruMPL_mean": float(np.nanmean(strumpl))}
            for key, label in COMPARATORS.items():
                if key in arrays:
                    row[f"{label}_mean"] = float(np.nanmean(arrays[key]))
                else:
                    row[f"{label}_mean"] = np.nan
            site_mean_rows.append(row)

            # Pairwise metrics between StruMPL and each comparator.
            for key, label in COMPARATORS.items():
                if key not in arrays:
                    per_cell_rows.append({
                        "site": site, "year": int(year), "comparator": label,
                        "n_pixels": 0, "r": np.nan, "rmse": np.nan,
                    })
                    continue
                comp = arrays[key].ravel()
                m = _pair_metrics(strumpl, comp)
                per_cell_rows.append({
                    "site": site, "year": int(year), "comparator": label,
                    **m,
                })

    return pd.DataFrame(per_cell_rows), pd.DataFrame(site_mean_rows)


def _aggregate_across_cells(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Cross-site x cross-year mean and SD per comparator. Sample-size
    weighting is NOT applied — each site-year cell counts equally, which
    is consistent with treating sites (not pixels) as the unit of replication.

    ddof=1 is used for the SD, which is the unbiased sample estimate.
    """
    rows = []
    for comp, grp in per_cell.groupby("comparator"):
        # Drop cells with missing metrics (NaN r or RMSE)
        valid = grp.dropna(subset=["r", "rmse"])
        n_cells = len(valid)
        if n_cells == 0:
            rows.append({
                "comparator": comp,
                "n_site_years": 0,
                "r_mean":    np.nan, "r_sd":    np.nan,
                "rmse_mean": np.nan, "rmse_sd": np.nan,
            })
            continue
        rows.append({
            "comparator":   comp,
            "n_site_years": n_cells,
            "r_mean":    float(valid["r"].mean()),
            "r_sd":      float(valid["r"].std(ddof=1)) if n_cells > 1 else np.nan,
            "rmse_mean": float(valid["rmse"].mean()),
            "rmse_sd":   float(valid["rmse"].std(ddof=1)) if n_cells > 1 else np.nan,
        })

    # Enforce a stable order matching the paper's table
    order = ["CCI", "L4B", "PG-CBM"]
    df = pd.DataFrame(rows)
    df["_ord"] = df["comparator"].map(lambda c: order.index(c) if c in order else 99)
    return df.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)


def main():
    print(f"[agg04-agb] computing per-cell metrics across "
          f"{len(SITES)} sites x {len(YEARS)} years x {len(COMPARATORS)} comparators")
    per_cell, site_means = _compute_per_cell()
    per_cell_path = OUT / "per_cell_metrics.csv"
    site_mean_path = OUT / "site_mean_agb.csv"
    per_cell.to_csv(per_cell_path, index=False)
    site_means.to_csv(site_mean_path, index=False)
    print(f"[agg04-agb] wrote {per_cell_path.name}  ({len(per_cell)} rows)")
    print(f"[agg04-agb] wrote {site_mean_path.name}  ({len(site_means)} rows)")

    summary = _aggregate_across_cells(per_cell)
    summary_path = OUT / "agreement_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[agg04-agb] wrote {summary_path.name}")

    # Print a paper-ready summary directly to the console for convenience.
    print("\n=== Paper table values ===")
    print(f"{'Comparator':>10}  {'n cells':>8}  "
          f"{'r (mean +/- SD)':>22}  {'RMSE Mg/ha (mean +/- SD)':>28}")
    for _, row in summary.iterrows():
        print(
            f"{row['comparator']:>10}  {int(row['n_site_years']):>8}  "
            f"{row['r_mean']:>+10.3f} +/- {row['r_sd']:>7.3f}  "
            f"{row['rmse_mean']:>+15.2f} +/- {row['rmse_sd']:>10.2f}"
        )

    print("[agg04-agb] done.")


if __name__ == "__main__":
    main()
