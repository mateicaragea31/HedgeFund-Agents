"""Settling past decisions: once a decision's holding window has traded, score it against SPY and
write a reflection on it into the decision log.

Follows upstream `graph/settlement.py` (commit 35543d0). Differences:
- Prices come from our archive (data/prices.py), not a fresh Yahoo download. Returns use
  `adj_close`, like upstream's adjusted closes, so a dividend paid inside the window counts.
  The return is a ratio, never shown as a price, so DEVIATIONS D6 is untouched.
- settle_pending settles every pending entry (all tickers) in one go: the backtest controller
  calls it before each wave. Upstream settles one ticker at the start of that ticker's next run.
- A ticker or benchmark missing from the archive is an error. Upstream leaves the entry
  pending and retries, which in a backtest would quietly switch memory off.

Point in time: the outcome of a decision on day d is known at the close of d + holding_days
trading days (`resolution_date`); get_past_context shows it only to cells on or after that day.
"""

import logging

import pandas as pd

from aihf.data.prices import coverage, price_file
from aihf.memory.decision_log import DecisionLog
from aihf.memory.reflection import reflect_on_final_decision

logger = logging.getLogger(__name__)


def _closes_from(archive_root, ticker: str, trade_date: str) -> pd.Series:
    """Adjusted closes from `trade_date` on (inclusive), indexed by date."""
    if coverage(archive_root, ticker) is None:
        raise FileNotFoundError(f"{ticker} is not in the price archive under {archive_root}; "
                                "collect it with aihf.scripts.collect_prices")
    bars = pd.read_parquet(price_file(archive_root, ticker), columns=["date", "adj_close"])
    bars = bars[bars["date"] >= pd.Timestamp(trade_date)].dropna().sort_values("date")
    return bars.set_index("date")["adj_close"]


def fetch_returns(
    archive_root, ticker: str, trade_date: str, holding_days: int = 5, benchmark: str = "SPY"
) -> tuple[float, float, int, str] | tuple[None, None, None, None]:
    """(raw return, alpha vs benchmark, holding days, resolution date), or all None if the
    holding window has not traded yet (the archive ends too soon).

    raw = close[t + holding_days] / close[t] - 1, in trading days, from the trade date's close.
    """
    stock = _closes_from(archive_root, ticker, trade_date)
    bench = _closes_from(archive_root, benchmark, trade_date)
    if len(stock) <= holding_days or len(bench) <= holding_days:
        return None, None, None, None
    raw = float((stock.iloc[holding_days] - stock.iloc[0]) / stock.iloc[0])
    bench_ret = float((bench.iloc[holding_days] - bench.iloc[0]) / bench.iloc[0])
    resolution_date = stock.index[holding_days].strftime("%Y-%m-%d")
    return raw, raw - bench_ret, holding_days, resolution_date


def settle_pending(
    log: DecisionLog, llm, archive_root, holding_days: int = 5, benchmark: str = "SPY"
) -> int:
    """Settle every pending entry whose holding window has traded; returns how many were settled.

    One reflection call (quick model) per settled entry, then one write for all of them.
    An entry whose reflection call fails stays pending for the next call (as upstream).
    """
    updates = []
    for entry in log.get_pending_entries():
        raw, alpha, days, resolution_date = fetch_returns(
            archive_root, entry["ticker"], entry["date"], holding_days, benchmark)
        if raw is None:
            continue  # window not traded yet
        try:
            reflection = reflect_on_final_decision(
                llm, entry.get("decision", ""), raw, alpha, benchmark_name=benchmark, holding_days=days)
        except Exception as exc:
            logger.warning("Reflection failed for %s on %s: %s", entry["ticker"], entry["date"], exc)
            continue
        updates.append({
            "ticker": entry["ticker"],
            "trade_date": entry["date"],
            "raw_return": raw,
            "alpha_return": alpha,
            "holding_days": days,
            "reflection": reflection,
            "resolution_date": resolution_date,
        })
    log.batch_update_with_outcomes(updates)
    return len(updates)
