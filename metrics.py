"""
Error and agreement metrics, with optional block-bootstrap confidence intervals.

For metrics computed against a reference (GEDI), use `error_metrics`.
For inter-product comparisons (e.g. PG-CBM vs CCI), use the same function but
interpret outputs as 'agreement', not 'accuracy'.
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Dict


def _safe(a: np.ndarray, b: np.ndarray):
    """Drop pairs where either is NaN. Returns aligned 1-D arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def error_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """
    All standard error metrics treating `ref` as truth.
        bias  = mean(pred - ref)
        MAE   = mean(|pred - ref|)
        RMSE  = sqrt(mean((pred-ref)^2))
        rRMSE = RMSE / mean(ref)         (only if mean(ref) > 0)
        R2    = 1 - SS_res / SS_tot      (coefficient of determination)
        r     = Pearson correlation
        rho   = Spearman correlation
        n     = sample size
    """
    p, r = _safe(pred, ref)
    n = p.size
    if n < 2:
        return {k: np.nan for k in
                ["bias", "MAE", "RMSE", "rRMSE", "R2", "r", "rho", "n"]} | {"n": n}

    resid = p - r
    bias  = float(np.mean(resid))
    mae   = float(np.mean(np.abs(resid)))
    rmse  = float(np.sqrt(np.mean(resid ** 2)))
    mu_r  = float(np.mean(r))
    rrmse = float(rmse / mu_r) if mu_r > 1e-9 else np.nan
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((r - mu_r) ** 2))
    r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    pearson = float(stats.pearsonr(p, r)[0]) if n >= 3 else np.nan
    spearman = float(stats.spearmanr(p, r)[0]) if n >= 3 else np.nan

    return {
        "bias":  bias,
        "MAE":   mae,
        "RMSE":  rmse,
        "rRMSE": rrmse,
        "R2":    r2,
        "r":     pearson,
        "rho":   spearman,
        "n":     int(n),
    }


def bootstrap_metrics(pred: np.ndarray,
                      ref: np.ndarray,
                      n_boot: int = 1000,
                      block_size: int | None = None,
                      seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap CIs for all error metrics. If block_size is given, uses a moving
    block bootstrap on the flat index — a rough way to respect spatial
    autocorrelation. Otherwise iid resampling at pixel level (will UNDERSTATE
    uncertainty for spatially clustered data).

    Returns:
        {metric: {'mean': float, 'lo': 2.5pct, 'hi': 97.5pct, 'sd': float}}
    """
    p, r = _safe(pred, ref)
    n = p.size
    if n < 10:
        return {}

    rng = np.random.default_rng(seed)
    results = {k: [] for k in ["bias", "MAE", "RMSE", "rRMSE", "R2", "r", "rho"]}

    if block_size is None or block_size <= 1:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            m = error_metrics(p[idx], r[idx])
            for k in results:
                results[k].append(m[k])
    else:
        # Moving block bootstrap (1-D — pixels are flattened, so this is a
        # coarse proxy for spatial blocks; good enough for a sanity-check CI).
        n_blocks = int(np.ceil(n / block_size))
        for _ in range(n_boot):
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
            m = error_metrics(p[idx], r[idx])
            for k in results:
                results[k].append(m[k])

    out = {}
    for k, vals in results.items():
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[k] = {"mean": np.nan, "lo": np.nan, "hi": np.nan, "sd": np.nan}
        else:
            out[k] = {
                "mean": float(np.mean(vals)),
                "lo":   float(np.percentile(vals, 2.5)),
                "hi":   float(np.percentile(vals, 97.5)),
                "sd":   float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
            }
    return out
