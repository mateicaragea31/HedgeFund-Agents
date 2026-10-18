"""The daily price archive: what the files look like, and reading only what was known at a cutoff.

One file per ticker, written by the price collector (Phase 1 step 3c):

    <root>/prices/<TICKER>.parquet     one row per trading day, columns PRICE_COLUMNS
    <root>/prices/<TICKER>.meta.json   source, versions, fetch time, first/last date covered

open/high/low/close are the prices as traded that day: NOT adjusted for dividends paid later
(Yahoo's adjusted prices change history after the fact; DEVIATIONS D6). adj_close is kept for
the simulator and audits, never shown to agents.
"""

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PRICE_COLUMNS = [
    "date",  # trading day (datetime64, midnight)
    "open", "high", "low", "close",  # as traded
    "adj_close",  # Yahoo's dividend- and split-adjusted close, as of the download day
    "volume",
    "dividends",  # cash dividend per share with its ex-date on this day, else 0
    "stock_splits",  # split ratio on this day, else 0
    "available_at",  # when the bar became known: 16:00 New York on `date`, in UTC
]


@dataclass(frozen=True)
class Coverage:
    """The days a price file covers, from its meta.json."""

    first: date
    last: date


def price_file(root: Path, ticker: str) -> Path:
    return Path(root) / "prices" / f"{ticker.upper()}.parquet"


def coverage(root: Path, ticker: str) -> Coverage | None:
    """First and last day the ticker's file covers, or None if there is no file."""
    meta = price_file(root, ticker).with_suffix(".meta.json")
    if not meta.exists() or not price_file(root, ticker).exists():
        return None
    info = json.loads(meta.read_text())
    return Coverage(date.fromisoformat(info["first_date"]), date.fromisoformat(info["last_date"]))


def load_bars(root: Path, ticker: str, cutoff: datetime) -> pd.DataFrame:
    """Every bar that was known at `cutoff` (available_at <= cutoff), oldest first.

    This is the only way tools read prices, so no tool can see a bar from after the cutoff,
    whatever dates it was asked for.
    """
    bars = pd.read_parquet(price_file(root, ticker), columns=PRICE_COLUMNS)
    bars = bars[bars["available_at"] <= pd.Timestamp(cutoff)]
    return bars.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
