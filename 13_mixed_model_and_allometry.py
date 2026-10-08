"""
13 — Mixed-model variance decomposition + allometric stability across years.

Two complementary temporal analyses:

  (A) Mixed-effects models on site-year-model cell statistics.

      We aggregate residuals to ONE row per (site, year, model, attribute)
      cell — using cell means and cell RMSE as separate responses — and fit:

          bias ~ model * year + (1 | site)         # mixed model on cell bias
          RMSE ~ model * year + (1 | site)         # mixed model on cell RMSE

      Why aggregated, not pixel-level? Pixels are spatially autocorrelated;
      using millions raw gives spuriously tight p-values and meaningless
      inference. Aggregating to cell stats matches the actual unit of
      replication (sites are the independent replicates).

      Why year as fixed, not random? With only 4 years a random-effect
      variance component is poorly estimated; treating year as a fixed factor
      lets us test year-on-year differences directly.

      The model * year interaction is the scientifically interesting term:
      it asks whether the gap BETWEEN models changes across years.
      If it's non-significant we report the additive form (model + year)
      as a more parsimonious fit.

      We also report a variance decomposition: how much of the total
      residual-bias variance is attributable to (model, year, site, residual)?

  (B) Allometric stability across years.

      For each (model, year, site), fit the same H+C+S -> AGBD OLS as in
      script 08, then check whether the coefficients (a, b, c) are stable
      across years. A *physically consistent* model has coefficients that
      barely move year to year. We report the mean and within-site sd of
      each coefficient across years.

Outputs (under OUTPUT_DIR/13_mixed_model_and_allometry/):
    cell_stats.csv                    one row per (site, year, model, attr)
    mixed_model_bias_<attr>.txt       statsmodels MixedLM summary
    mixed_model_rmse_<attr>.txt
    variance_decomposition.csv        % of variance attributable to each factor
    allometric_year_fits.csv          OLS coefficients per (model, year, site)
    allometric_stability.csv          per-(site, model): mean+sd of coefs across years
    allometric_stability.png         summary plot
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import SITES, YEARS, MODELS, ATTRIBUTES, OUTPUT_DIR, COLOURS
from io_utils import load_site_year, gedi_mask, joint_valid_mask
from metrics import error_metrics


OUT = OUTPUT_DIR / "13_mixed_model_and_allometry"
OUT.mkdir(parents=True, exist_ok=True)


REF_TARGETS = {"Height": "GEDI_RH98", "Cover": "GEDI_Cover"}


def _ols_h_c_s(H, C, S, AGBD):
    """OLS  AGBD = a*H + b*C + c*S + d.  Returns dict {coef_H, coef_C, coef_S, intercept, R2, n}."""
    ok = np.isfinite(H) & np.isfinite(C) & np.isfinite(S) & np.isfinite(AGBD)
    H, C, S, AGBD = H[ok], C[ok], S[ok], AGBD[ok]
    if H.size < 30:
        return None
    X = np.column_stack([H, C, S, np.ones_like(H)])
    coef, *_ = np.linalg.lstsq(X, AGBD, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((AGBD - pred)**2))
    ss_tot = float(np.sum((AGBD - AGBD.mean())**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return dict(coef_H=float(coef[0]), coef_C=float(coef[1]),
                coef_S=float(coef[2]), intercept=float(coef[3]),
                R2=r2, n=int(H.size))


def main():
    # ============================================================
    # (1) Build the cell statistics table
    # ============================================================
    cell_rows = []
    allo_rows = []
    print("[13] computing cell statistics across sites × years × models ...")
    for site in SITES:
        for year in YEARS:
            try:
                b = load_site_year(site, year)
            except FileNotFoundError as e:
                print(f"  [WARN] {site} {year}: {e}")
                continue

            # ---- Cell stats for Height & Cover -----------------------------
            for attr, ref_key in REF_TARGETS.items():
                mask = gedi_mask(b, attr)
                if mask is None or mask.sum() == 0:
                    continue
                ref = b["arrays"][ref_key]
                for model in MODELS:
                    pred = b["arrays"][f"{model}_{attr}"]
                    m = mask & np.isfinite(pred) & np.isfinite(ref)
                    if m.sum() < 50:
                        continue
                    em = error_metrics(pred[m], ref[m])
                    cell_rows.append({
                        "site": site, "year": year, "model": model,
                        "attribute": attr, **em,
                    })

            # ---- Allometric OLS per (year, model) site ---------------------
            for model in MODELS:
                keys = [f"{model}_{a}" for a in ["AGBD","Height","Cover","Stem"]]
                m = joint_valid_mask(b, keys)
                if m.sum() < 200:
                    continue
                fit = _ols_h_c_s(
                    b["arrays"][f"{model}_Height"][m],
                    b["arrays"][f"{model}_Cover"][m],
                    b["arrays"][f"{model}_Stem"][m],
                    b["arrays"][f"{model}_AGBD"][m],
                )
                if fit is None: continue
                allo_rows.append({
                    "site": site, "year": year, "model": model, **fit,
                })

    cell = pd.DataFrame(cell_rows)
    cell.to_csv(OUT / "cell_stats.csv", index=False)
    print(f"[13] wrote cell_stats.csv  ({len(cell)} rows)")

    allo = pd.DataFrame(allo_rows)
    allo.to_csv(OUT / "allometric_year_fits.csv", index=False)
    print(f"[13] wrote allometric_year_fits.csv  ({len(allo)} rows)")

    # ============================================================
    # (2) Mixed-effects models per attribute
    # ============================================================
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("[13] statsmodels not installed — skipping mixed models")
        smf = None

    if smf is not None and not cell.empty:
        for attr in REF_TARGETS:
            sub = cell[cell.attribute == attr].copy()
            if sub.empty: continue
            # Encode factors as strings so statsmodels treats them categorical
            sub["year"]  = sub["year"].astype(str)
            sub["model"] = sub["model"].astype("category")
            sub["site"]  = sub["site"].astype("category")

            for response in ("bias", "RMSE"):
                # Drop NaN responses
                d = sub.dropna(subset=[response])
                if d["site"].nunique() < 2 or len(d) < 8:
                    print(f"[13]  {attr}/{response}: too few rows, skipping")
                    continue
                # Fit interactive first; fall back to additive if singular
                txt_lines = [f"Mixed model: {response} ~ model * year + (1|site)",
                             f"Attribute: {attr}",
                             f"n_obs = {len(d)},  n_sites = {d['site'].nunique()}",
                             "="*72, ""]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m_int = smf.mixedlm(f"{response} ~ C(model) * C(year)",
                                             data=d, groups=d["site"]).fit(
                                                 reml=True, method="lbfgs")
                    txt_lines.append(m_int.summary().as_text())
                except Exception as e:
                    txt_lines.append(f"[interaction model failed: {e}]")
                    m_int = None

                txt_lines += ["", "-"*72, "", "Additive form: " + response +
                              " ~ C(model) + C(year) + (1|site)", ""]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m_add = smf.mixedlm(f"{response} ~ C(model) + C(year)",
                                             data=d, groups=d["site"]).fit(
                                                 reml=True, method="lbfgs")
                    txt_lines.append(m_add.summary().as_text())
                except Exception as e:
                    txt_lines.append(f"[additive model failed: {e}]")

                (OUT / f"mixed_model_{response.lower()}_{attr}.txt").write_text(
                    "\n".join(txt_lines), encoding="utf-8")

    # ============================================================
    # (3) Variance decomposition via ANOVA-style sums of squares
    # ============================================================
    # Use a fixed-effects OLS to partition Type-III SS into model/year/site/interaction.
    # This is a quick descriptive decomposition (NOT inferential — for inference
    # see the mixed model summaries above).
    if smf is not None and not cell.empty:
        try:
            import statsmodels.api as sm
            vd_rows = []
            for attr in REF_TARGETS:
                sub = cell[cell.attribute == attr].dropna(subset=["bias"]).copy()
                if sub.empty: continue
                sub["year"]  = sub["year"].astype(str)
                sub["model"] = sub["model"].astype("category")
                sub["site"]  = sub["site"].astype("category")
                for response in ("bias", "RMSE"):
                    d = sub.dropna(subset=[response])
                    if len(d) < 10: continue
                    f = f"{response} ~ C(model) + C(year) + C(site) + C(model):C(year)"
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ols = smf.ols(f, data=d).fit()
                            anova = sm.stats.anova_lm(ols, typ=2)
                    except Exception as e:
                        print(f"  [WARN] ANOVA {attr}/{response}: {e}")
                        continue
                    total_ss = float(anova["sum_sq"].sum())
                    for term, row in anova.iterrows():
                        vd_rows.append({
                            "attribute": attr, "response": response,
                            "term": term,
                            "sum_sq": float(row["sum_sq"]),
                            "pct_of_total": (100.0 * float(row["sum_sq"])
                                              / total_ss) if total_ss else np.nan,
                            "df": float(row["df"]),
                            "F": float(row.get("F", np.nan)),
                            "p": float(row.get("PR(>F)", np.nan)),
                        })
            pd.DataFrame(vd_rows).to_csv(
                OUT / "variance_decomposition.csv", index=False)
            print("[13] wrote variance_decomposition.csv")
        except Exception as e:
            print(f"[13] variance decomposition failed: {e}")

    # ============================================================
    # (4) Allometric stability summary
    # ============================================================
    if not allo.empty:
        # Per (site, model): mean & sd of each coefficient across years
        agg = (allo.groupby(["site", "model"])
                   .agg(coef_H_mean=("coef_H", "mean"),
                        coef_H_sd  =("coef_H", "std"),
                        coef_C_mean=("coef_C", "mean"),
                        coef_C_sd  =("coef_C", "std"),
                        coef_S_mean=("coef_S", "mean"),
                        coef_S_sd  =("coef_S", "std"),
                        R2_mean   =("R2",     "mean"),
                        R2_sd     =("R2",     "std"),
                        n_years   =("year",   "nunique"))
                   .reset_index())
        agg.to_csv(OUT / "allometric_stability.csv", index=False)

        # Summary plot: bar per coefficient showing within-site sd across years,
        # averaged over sites — smaller = more temporally consistent
        coefs = ["coef_H", "coef_C", "coef_S"]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(coefs))
        width = 0.35
        for i, model in enumerate(MODELS):
            sub = agg[agg.model == model]
            if sub.empty: continue
            vals = [sub[f"{c}_sd"].mean() for c in coefs]
            ax.bar(x + (i-0.5)*width, vals, width,
                   color=COLOURS.get(model,"#444"), label=model)
        ax.set_xticks(x); ax.set_xticklabels(
            ["a (Height)", "b (Cover)", "c (Stem)"])
        ax.set_ylabel("Mean across sites of (within-site sd of coef across years)")
        ax.set_title("Allometric coefficient stability across years\n"
                     "(lower = more temporally consistent)")
        ax.grid(alpha=0.3, axis="y"); ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "allometric_stability.png",
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("[13] wrote allometric_stability.csv and .png")

    print("[13] done.")


if __name__ == "__main__":
    main()
