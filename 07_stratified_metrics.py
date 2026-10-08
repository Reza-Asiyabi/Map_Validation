"""
07 — Stratified metrics.

Pooled metrics can hide where a model fails. Two stratifications:

  (A) By reference-value bin: split GEDI Height (and Cover) into bins and
      compute per-bin RMSE/bias. Reveals saturation — the canonical failure
      mode where models plateau at the high end of the value range.

  (B) By cover class: bin pixels by GEDI Cover (or, where unavailable, by
      Hansen Cover) into Sparse / Open / Closed forest. Useful for AGB
      especially: cover acts as a proxy for ecological zone when no
      ecoregion raster is supplied.

If you later add an ecoregion raster per site, add it as a third strat;
the script structure makes that straightforward.

Outputs (under OUTPUT_DIR/07_stratified/):
    stratified_by_refbin_<attr>.csv
    stratified_by_cover_<attr>.csv
    saturation_<attr>.png           bias vs ref-bin per source (the diagnostic)
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import SITES, ATTRIBUTES, OUTPUT_DIR, COLOURS, COVER_SCALE
from io_utils import load_site, gedi_mask, joint_valid_mask
from metrics import error_metrics


OUT = OUTPUT_DIR / "07_stratified"
OUT.mkdir(parents=True, exist_ok=True)


REF_BINS = {
    "Height": np.array([0, 3, 6, 9, 12, 15, 20, 30]),
    "Cover":  np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]) * COVER_SCALE,
    "AGB":    np.array([0, 10, 25, 50, 100, 200, 400]),
}

COVER_CLASSES = {
    "Sparse": (0.0 * COVER_SCALE, 0.10 * COVER_SCALE),
    "Open":   (0.10 * COVER_SCALE, 0.40 * COVER_SCALE),
    "Closed": (0.40 * COVER_SCALE, 1.01 * COVER_SCALE),
}


def _stratify_by_refbin(pred, ref, edges):
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (ref >= lo) & (ref < hi)
        if sel.sum() < 30:
            continue
        m = error_metrics(pred[sel], ref[sel])
        rows.append({"bin_lo": lo, "bin_hi": hi,
                     "bin_centre": 0.5 * (lo + hi), **m})
    return rows


def _cover_class_array(bundle):
    """Pick GEDI Cover if available (any pixel), else fall back to Hansen."""
    g = bundle["arrays"].get("GEDI_Cover")
    h = bundle["arrays"].get("Hansen_Cover")
    # Prefer Hansen as a continuous map (GEDI is sparse, classes would be empty
    # in most pixels). We use Hansen for the cover-class stratification but
    # leave GEDI as the value-bin reference where applicable.
    if h is not None:
        return h, "Hansen_Cover"
    return g, "GEDI_Cover"


def main():
    rb_rows = {a: [] for a in ["Height", "Cover", "AGB"]}
    cc_rows = {a: [] for a in ATTRIBUTES}

    for site in SITES:
        print(f"[07] loading {site} ...")
        bundle = load_site(site)
        cover_arr, cover_name = _cover_class_array(bundle)

        # ---- (A) By GEDI reference-value bin (Height, Cover only) -----------
        for attr in ["Height", "Cover"]:
            mask = gedi_mask(bundle, attr)
            if mask is None or mask.sum() == 0:
                continue
            ref_key = ATTRIBUTES[attr]["ref"]
            ref = bundle["arrays"][ref_key][mask]
            for src in [f"PG-CBM_{attr}", f"StruMPL_{attr}",
                        *ATTRIBUTES[attr]["externals"]]:
                pred = bundle["arrays"][src][mask]
                for r in _stratify_by_refbin(pred, ref, REF_BINS[attr]):
                    rb_rows[attr].append(
                        {"site": site, "source": src, **r})

        # ---- (A') AGB: stratify by CCI bin since we have no truth ------------
        if "CCI_AGB" in bundle["arrays"]:
            cci = bundle["arrays"]["CCI_AGB"]
            m = np.isfinite(cci)
            ref = cci[m]
            for src in [f"PG-CBM_AGB", f"StruMPL_AGB", "GEDI_L4B_AGB"]:
                if src not in bundle["arrays"]:
                    continue
                pred = bundle["arrays"][src][m]
                ok = np.isfinite(pred) & np.isfinite(ref)
                if ok.sum() < 30:
                    continue
                for r in _stratify_by_refbin(pred[ok], ref[ok],
                                             REF_BINS["AGB"]):
                    rb_rows["AGB"].append(
                        {"site": site, "source": src,
                         "comparator": "CCI_AGB", **r})

        # ---- (B) By cover class -------------------------------------------------
        if cover_arr is None:
            continue
        for cname, (lo, hi) in COVER_CLASSES.items():
            class_mask = np.isfinite(cover_arr) & (cover_arr >= lo) & (cover_arr < hi)
            if class_mask.sum() < 100:
                continue
            for attr in ATTRIBUTES:
                ref_key = ATTRIBUTES[attr]["ref"]
                # For Height/Cover use GEDI; for AGB use CCI; for Stem skip
                if ref_key is not None:
                    g_mask = gedi_mask(bundle, attr)
                    m = class_mask & g_mask
                    if m.sum() < 30:
                        continue
                    ref = bundle["arrays"][ref_key][m]
                elif attr == "AGB":
                    cci = bundle["arrays"].get("CCI_AGB")
                    if cci is None: continue
                    m = class_mask & np.isfinite(cci)
                    if m.sum() < 30:
                        continue
                    ref = cci[m]
                    ref_key = "CCI_AGB"
                else:
                    continue
                for src in [f"PG-CBM_{attr}", f"StruMPL_{attr}",
                            *ATTRIBUTES[attr]["externals"]]:
                    if src not in bundle["arrays"]: continue
                    pred = bundle["arrays"][src][m]
                    metrics = error_metrics(pred, ref)
                    cc_rows[attr].append({
                        "site": site, "source": src, "cover_class": cname,
                        "reference": ref_key, **metrics,
                    })

    # ---- Save per-attribute tables -------------------------------------------
    for attr, rows in rb_rows.items():
        if rows:
            pd.DataFrame(rows).to_csv(
                OUT / f"stratified_by_refbin_{attr}.csv", index=False)
    for attr, rows in cc_rows.items():
        if rows:
            pd.DataFrame(rows).to_csv(
                OUT / f"stratified_by_cover_{attr}.csv", index=False)

    # ---- Saturation diagnostic plots: bias vs ref-bin centre, averaged across sites -----
    for attr in ["Height", "Cover", "AGB"]:
        rows = rb_rows[attr]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(7, 4))
        for src, group in df.groupby("source"):
            agg = (group.groupby("bin_centre")
                        .agg(bias_mean=("bias", "mean"),
                             bias_sd=("bias", "std"))
                        .reset_index())
            short = src.split("_")[0]
            color = COLOURS.get(short if short in COLOURS else src, "#888")
            ax.errorbar(agg["bin_centre"], agg["bias_mean"],
                        yerr=agg["bias_sd"], marker="o", lw=1.5,
                        color=color, label=src, capsize=3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f"Reference {attr} [{ATTRIBUTES[attr]['unit']}]")
        ax.set_ylabel(f"Bias (pred − ref) [{ATTRIBUTES[attr]['unit']}]")
        ax.set_title(f"{attr}: bias vs reference value (saturation diagnostic)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"saturation_{attr}.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("[07] done.")


if __name__ == "__main__":
    main()
