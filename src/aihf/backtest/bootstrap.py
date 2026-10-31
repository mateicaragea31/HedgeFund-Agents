"""Confidence intervals by the stationary bootstrap (Politis & Romano, 1994).

Daily returns are resampled in blocks of random length (geometric, mean `block`), wrapping
around the end, so short-run dependence in the returns is kept. Each resample gives a CR and a
Sharpe; the 2.5 % and 97.5 % percentiles are the 95 % interval. With ~60 daily returns the
intervals are wide: that is the honest size of what one 3-month window can show.
"""

import math

import numpy as np
import pandas as pd

from aihf.backtest.metrics import TRADING_DAYS


def stationary_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """n indices into 0..n-1 made of blocks with geometric lengths (mean `block`)."""
    p = 1.0 / block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def bootstrap_ci(returns: pd.Series, rf_daily: float = 0.0, *, resamples: int = 5000, block: float = 5.0,
                 seed: int = 0, level: float = 0.95) -> dict:
    """95 % intervals for CR and Sharpe of a daily-return series."""
    r = returns.dropna().to_numpy()
    if len(r) < 2:
        return {"CR_ci": [None, None], "Sharpe_ci": [None, None]}
    rng = np.random.default_rng(seed)
    crs, sharpes = np.empty(resamples), np.empty(resamples)
    for b in range(resamples):
        sample = r[stationary_indices(len(r), block, rng)]
        crs[b] = np.prod(1 + sample) - 1
        sd = sample.std(ddof=1)
        sharpes[b] = math.sqrt(TRADING_DAYS) * (sample.mean() - rf_daily) / sd if sd > 0 else np.nan
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    sharpe_ci = ([None, None] if np.isnan(sharpes).all()  # e.g. a strategy that stayed in cash
                 else [float(np.nanpercentile(sharpes, lo)), float(np.nanpercentile(sharpes, hi))])
    return {"CR_ci": [float(np.percentile(crs, lo)), float(np.percentile(crs, hi))], "Sharpe_ci": sharpe_ci}
