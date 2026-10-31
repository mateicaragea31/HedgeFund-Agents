"""Performance metrics of a value curve (CLAUDE.md §6.4). Every formula is stated here.

    CR      cumulative return       = V_T / V_0 - 1
    AR      annualised return       = (1 + CR) ** (252 / N) - 1, N = number of daily returns
    Sharpe  annualised Sharpe ratio = sqrt(252) * mean(r_t - rf_t) / std(r_t)   (std with ddof=1)
    MDD     maximum drawdown        = max over t of (1 - V_t / max_{s <= t} V_s), as a positive fraction
    hit rate     share of days with a position held whose return was positive
    turnover     total traded amount / initial capital
    avg exposure mean of |exposure| at the closes

For the paper audit (CLAUDE.md §2.2 item 1): the paper's "ARR" column equals V_T / $100k - 1,
i.e. our CR, while its "CR" column follows neither formula. `paper_style_arr` reports that number.
"""

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cumulative_return(values: pd.Series) -> float:
    return float(values.iloc[-1] / values.iloc[0] - 1)


def annualised_return(cr: float, n_returns: int) -> float:
    return float((1 + cr) ** (TRADING_DAYS / n_returns) - 1) if n_returns else float("nan")


def sharpe(returns: pd.Series, rf: pd.Series | float = 0.0) -> float:
    """rf: the daily risk-free return, as a series aligned with `returns` or one number."""
    r = returns.dropna()
    excess = r - (rf.reindex(r.index).fillna(0.0) if isinstance(rf, pd.Series) else rf)
    sd = r.std(ddof=1)
    return float(math.sqrt(TRADING_DAYS) * excess.mean() / sd) if sd and sd > 0 else float("nan")


def max_drawdown(values: pd.Series) -> float:
    peak = values.cummax()
    return float((1 - values / peak).max())


def evaluate(sim: pd.DataFrame, initial_cash: float, rf_daily: pd.Series | float = 0.0) -> dict:
    """All metrics of one simulation (simulator.simulate's output)."""
    values = sim["value"].reset_index(drop=True)
    returns = sim["daily_return"].reset_index(drop=True)
    n = int(returns.notna().sum())
    cr = cumulative_return(values)
    rf = rf_daily if not isinstance(rf_daily, pd.Series) else pd.Series(rf_daily.to_numpy(), index=returns.index)
    held = sim["exposure"].shift(1).fillna(0.0).reset_index(drop=True) != 0  # position held during the day
    invested = returns[held & returns.notna()]
    return {
        "CR": cr,
        "AR": annualised_return(cr, n),
        "Sharpe": sharpe(returns, rf),
        "MDD": max_drawdown(values),
        "hit_rate": float((invested > 0).mean()) if len(invested) else float("nan"),
        "turnover": float(sim["traded"].sum() / initial_cash),
        "avg_exposure": float(sim["exposure"].abs().mean()),
        "final_value": float(values.iloc[-1]),
        "paper_style_arr": float(values.iloc[-1] / initial_cash - 1),
        "n_returns": n,
    }


def daily_risk_free(dtb3_percent: pd.Series) -> pd.Series:
    """DTB3 (3-month T-bill, annual %, a discount rate) → a daily return: (1 + y) ** (1/252) - 1."""
    return (1 + dtb3_percent.astype(float) / 100) ** (1 / TRADING_DAYS) - 1


def nan_to_none(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}
