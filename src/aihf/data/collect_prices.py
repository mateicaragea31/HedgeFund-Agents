"""Download daily prices once from Yahoo Finance, check them, and save them to the price archive.

Runs on the FEP login node (or the Mac), never inside a backtest. For each ticker:
  1. download [start, end] with yfinance, prices as traded (auto_adjust=False);
  2. stamp each bar with available_at = 16:00 New York on its day (data/pit.py);
  3. check the data, and refuse to save if anything is wrong (see `check`);
  4. write <root>/prices/<TICKER>.parquet and <TICKER>.meta.json (format: data/prices.py).

Note: Yahoo's "as traded" prices are still adjusted for stock splits, even with
auto_adjust=False. So a split after the test window would rewrite the prices the agents see;
`check` refuses any ticker with a split on or after `no_split_after` (CLAUDE.md rule 4).
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from aihf.data.pit import bar_available_at
from aihf.data.prices import PRICE_COLUMNS, price_file

logger = logging.getLogger(__name__)

# Trading-day counts we know for sure, used to catch missing days. The paper window:
# 2024-01-02 .. 2024-03-28 has 61 trading days (CLAUDE.md §2.1).
KNOWN_TRADING_DAYS = {("2024-01-02", "2024-03-28"): 61}

# yfinance column → our column
YAHOO_COLUMNS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
    "Volume": "volume", "Dividends": "dividends", "Stock Splits": "stock_splits",
}

# A fetcher returns (daily history like yfinance's, every split in the ticker's history).
Fetcher = Callable[[str, str, str], tuple[pd.DataFrame, pd.Series]]


class PriceDataError(Exception):
    """The downloaded data failed a check; nothing was saved."""


def yahoo_fetch(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """Daily bars from start to end (both included), as traded, plus the full split history."""
    import yfinance as yf  # only the collector needs it; agents and tests never import it

    end_exclusive = f"{date.fromisoformat(end) + timedelta(days=1)}"  # yfinance's end is exclusive
    t = yf.Ticker(ticker)
    for attempt in range(3):
        try:
            history = t.history(start=start, end=end_exclusive, auto_adjust=False, actions=True)
            if not history.empty:
                return history, t.splits
            problem = "empty result"
        except Exception as exc:  # network hiccup, rate limit, ...
            problem = exc
        logger.warning("%s: download attempt %d failed (%s)", ticker, attempt + 1, problem)
        time.sleep(2 * (attempt + 1))
    raise PriceDataError(f"{ticker}: no data from Yahoo after 3 attempts")


def to_archive(history: pd.DataFrame) -> pd.DataFrame:
    """yfinance's table → our archive table (PRICE_COLUMNS), one row per trading day."""
    bars = history.rename(columns=YAHOO_COLUMNS)[list(YAHOO_COLUMNS.values())].copy()
    # Yahoo's dates are midnight New York time; keep just the day.
    bars.insert(0, "date", pd.DatetimeIndex(history.index).tz_localize(None).normalize())
    bars["volume"] = bars["volume"].astype("int64")
    bars["available_at"] = [bar_available_at(d.date()) for d in bars["date"]]
    return bars.reset_index(drop=True)[PRICE_COLUMNS]


def check(bars: pd.DataFrame, splits: pd.Series, ticker: str, start: str, end: str,
          no_split_after: str) -> None:
    """Raise PriceDataError listing every problem found; return quietly if the data is fine."""
    problems = []
    prices = bars[["open", "high", "low", "close"]]
    if prices.isna().any().any():
        problems.append(f"{int(prices.isna().any(axis=1).sum())} day(s) with an empty price")
    bad_range = (bars["low"] > bars[["open", "close"]].min(axis=1)) | \
                (bars["high"] < bars[["open", "close"]].max(axis=1))
    if bad_range.any():
        problems.append(f"{int(bad_range.sum())} day(s) where open/close is outside low..high, "
                        f"first {bars.loc[bad_range, 'date'].iloc[0]:%Y-%m-%d}")
    if bars["date"].duplicated().any():
        problems.append("a date appears twice")
    if not bars["date"].is_monotonic_increasing:
        problems.append("dates are not in order")
    if (bars["date"].min() - pd.Timestamp(start)).days > 5:
        problems.append(f"starts on {bars['date'].min():%Y-%m-%d}, asked for {start}")
    if (pd.Timestamp(end) - bars["date"].max()).days > 5:
        problems.append(f"ends on {bars['date'].max():%Y-%m-%d}, asked for {end}")
    for (first, last), expected in KNOWN_TRADING_DAYS.items():
        if start <= first and last <= end:
            n = int(bars["date"].between(first, last).sum())
            if n != expected:
                problems.append(f"{n} trading days in {first}..{last}, expected {expected}")
    late = [f"{d:%Y-%m-%d} ({r:g}:1)" for d, r in splits.items()
            if d.date() >= date.fromisoformat(no_split_after)]
    if late:
        problems.append(f"stock split on or after {no_split_after}: {', '.join(late)}; "
                        "Yahoo has rewritten the earlier prices, so agents would see wrong ones")
    if problems:
        raise PriceDataError(f"{ticker}: " + "; ".join(problems))


def save(bars: pd.DataFrame, root: Path, ticker: str, meta: dict) -> Path:
    """Write the parquet file and its meta.json. Each is written to a temporary name first and
    then renamed, so a crash never leaves a half-written file behind."""
    path = price_file(root, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    bars.to_parquet(tmp, index=False)
    meta = {**meta, "sha256": hashlib.sha256(tmp.read_bytes()).hexdigest()}
    tmp.replace(path)
    meta_path = path.with_suffix(".meta.json")
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    tmp_meta.replace(meta_path)
    return path


def collect(tickers: list[str], start: str, end: str, root: Path, *, no_split_after: str,
            force: bool = False, dry_run: bool = False, fetch: Fetcher = yahoo_fetch) -> dict[str, str]:
    """Collect every ticker. Returns {ticker: what happened}; raises if any ticker failed."""
    outcome, failed = {}, []
    for ticker in (t.upper() for t in tickers):
        path = price_file(root, ticker)
        if path.exists() and not force:
            outcome[ticker] = f"skipped: {path} already exists (use --force to download again)"
        elif dry_run:
            outcome[ticker] = f"dry run: would download {start}..{end} into {path}"
        else:
            try:
                history, splits = fetch(ticker, start, end)
                bars = to_archive(history)
                check(bars, splits, ticker, start, end, no_split_after)
                save(bars, root, ticker, _meta(bars, start, end, no_split_after))
                outcome[ticker] = (f"saved {len(bars)} days, {bars['date'].min():%Y-%m-%d}.."
                                   f"{bars['date'].max():%Y-%m-%d}, into {path}")
            except PriceDataError as exc:
                outcome[ticker] = f"FAILED: {exc}"
                failed.append(ticker)
        logger.info("%s: %s", ticker, outcome[ticker])
    if failed:
        raise PriceDataError(f"failed: {', '.join(failed)} (nothing was saved for them)")
    return outcome


def _meta(bars: pd.DataFrame, start: str, end: str, no_split_after: str) -> dict:
    """The label stored next to each price file."""
    try:
        from importlib.metadata import version
        yf_version = version("yfinance")
    except Exception:
        yf_version = "unknown"
    return {
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf_version,
        "prices": "as traded (auto_adjust=False; split-adjusted by Yahoo, not dividend-adjusted)",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "requested_start": start,
        "requested_end": end,
        "first_date": f"{bars['date'].min():%Y-%m-%d}",
        "last_date": f"{bars['date'].max():%Y-%m-%d}",
        "rows": len(bars),
        "checked_no_split_after": no_split_after,
    }
