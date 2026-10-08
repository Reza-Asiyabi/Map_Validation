"""
09 — Cross-site summary tables.

Reads the CSV outputs from the other scripts (which must have been run first)
and produces a compact summary suitable for paper tables / supplementary
material.

Per-attribute summary table contains, for each (attribute, source):
    n_sites, mean ± sd of {bias, MAE, RMSE, rRMSE, R^2, r}
    median per-site and best/worst site

A separate 'model_comparison.csv' tells you whether PG-CBM beats StruMPL
(or vice versa) per attribute, including a paired Wilcoxon signed-rank test
on per-site RMSE differences (10 sites → small-sample but interpretable).

Outputs (under OUTPUT_DIR/09_summary/):
    summary_<attr>.csv
    model_comparison.csv
    overview.txt                    plain-text human-readable summary
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, ATTRIBUTES, MODELS


OUT = OUTPUT_DIR / "09_summary"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    in_csv = OUTPUT_DIR / "01_quantitative_metrics" / "per_site_metrics.csv"
    if not in_csv.exists():
        raise FileNotFoundError(
            f"{in_csv} not found. Run 01_quantitative_metrics.py first."
        )
    df = pd.read_csv(in_csv)

    # ---- Per-attribute summary ----------------------------------------------
    summary_lines = ["FOREST ATTRIBUTE VALIDATION — CROSS-SITE SUMMARY", "=" * 60, ""]
    for attr in df["attribute"].unique():
        sub = df[df["attribute"] == attr]
        rows = []
        for src, g in sub.groupby("source"):
            row = {"attribute": attr, "source": src, "n_sites": len(g)}
            for metric in ["bias", "MAE", "RMSE", "rRMSE", "R2", "r"]:
                vals = g[metric].dropna()
                if vals.empty:
                    row.update({f"{metric}_mean": np.nan,
                                f"{metric}_sd":   np.nan,
                                f"{metric}_median": np.nan,
                                f"{metric}_min":  np.nan,
                                f"{metric}_max":  np.nan})
                else:
                    row.update({
                        f"{metric}_mean":   float(vals.mean()),
                        f"{metric}_sd":     float(vals.std()),
                        f"{metric}_median": float(vals.median()),
                        f"{metric}_min":    float(vals.min()),
                        f"{metric}_max":    float(vals.max()),
                    })
            rows.append(row)
        s = pd.DataFrame(rows).sort_values("RMSE_mean")
        s.to_csv(OUT / f"summary_{attr}.csv", index=False)

        summary_lines.append(f"--- {attr} ---")
        for _, r in s.iterrows():
            unit = ATTRIBUTES[attr]["unit"]
            summary_lines.append(
                f"  {r['source']:<25}  "
                f"RMSE = {r['RMSE_mean']:.2f} ± {r['RMSE_sd']:.2f} {unit}   "
                f"bias = {r['bias_mean']:+.2f}   "
                f"R² = {r['R2_mean']:.2f}   "
                f"(n_sites = {int(r['n_sites'])})"
            )
        summary_lines.append("")

    # ---- Head-to-head model comparison ---------------------------------------
    comp_rows = []
    for attr in ["Height", "Cover"]:        # we only have refs for these
        pgcbm = df[(df.attribute == attr) & (df.source == f"PG-CBM_{attr}")]
        strumpl = df[(df.attribute == attr) & (df.source == f"StruMPL_{attr}")]
        merged = pd.merge(pgcbm, strumpl, on="site",
                          suffixes=("_pgcbm", "_strumpl"))
        if len(merged) < 3:
            continue
        for metric in ["RMSE", "MAE", "bias", "R2", "r"]:
            d_pg = merged[f"{metric}_pgcbm"].to_numpy()
            d_st = merged[f"{metric}_strumpl"].to_numpy()
            ok = np.isfinite(d_pg) & np.isfinite(d_st)
            if ok.sum() < 3:
                continue
            d_pg, d_st = d_pg[ok], d_st[ok]
            try:
                w_stat, w_p = stats.wilcoxon(d_pg, d_st)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            comp_rows.append({
                "attribute": attr, "metric": metric,
                "pg_cbm_mean": float(np.mean(d_pg)),
                "strumpl_mean": float(np.mean(d_st)),
                "diff_mean":  float(np.mean(d_pg - d_st)),
                "wilcoxon_W": float(w_stat) if w_stat is not None else np.nan,
                "wilcoxon_p": float(w_p) if w_p is not None else np.nan,
                "n_sites": int(ok.sum()),
            })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / "model_comparison.csv", index=False)

    summary_lines.append("HEAD-TO-HEAD: PG-CBM vs StruMPL (paired Wilcoxon)")
    summary_lines.append("-" * 60)
    for _, r in comp.iterrows():
        winner = "PG-CBM" if (r["diff_mean"] < 0 and r["metric"] in ("RMSE", "MAE")
                              or r["diff_mean"] > 0 and r["metric"] in ("R2", "r")) \
                          else "StruMPL"
        if r["metric"] == "bias":
            winner = "smaller |bias|: " + (
                "PG-CBM" if abs(r["pg_cbm_mean"]) < abs(r["strumpl_mean"])
                else "StruMPL"
            )
        summary_lines.append(
            f"  {r['attribute']:<7} {r['metric']:<5}  "
            f"PG-CBM={r['pg_cbm_mean']:+.3f}  "
            f"StruMPL={r['strumpl_mean']:+.3f}  "
            f"p={r['wilcoxon_p']:.3f}   →  {winner}"
        )

    (OUT / "overview.txt").write_text("\n".join(summary_lines),
                                      encoding="utf-8")
    print(f"[09] wrote {OUT / 'overview.txt'}")
    print("[09] done.")


if __name__ == "__main__":
    main()