"""What a run consists of: its id, its trade dates, and the waves they are run in.

Waves (CLAUDE.md §4.4): a decision made on trade date d is settled after `holding` trading days,
and a cell on date t may only read lessons settled by t. So `holding` consecutive trade dates
never depend on each other's decisions and can run at the same time, for all tickers. Without
memory, every cell is independent and the whole run is one wave.
"""

import hashlib
from pathlib import Path

import pandas as pd

from aihf.data.prices import price_file


def run_id(experiment_yaml: Path, model_yaml: Path, repeat: int, git_commit: str) -> str:
    """12 hex digits from the experiment file, the model file, the repeat and our commit.

    Same inputs → same id → same results folder, so starting a run again continues it.
    """
    h = hashlib.sha256()
    for part in (Path(experiment_yaml).read_bytes(), Path(model_yaml).read_bytes(),
                 f"repeat={repeat}".encode(), f"commit={git_commit}".encode()):
        h.update(part)
        h.update(b"\0")
    return h.hexdigest()[:12]


def trade_dates(archive_root: Path, ticker: str, start: str, end: str) -> list[str]:
    """The trading days in [start, end]: the days the ticker's price file has a bar for."""
    bars = pd.read_parquet(price_file(archive_root, ticker), columns=["date"])
    days = [f"{d:%Y-%m-%d}" for d in sorted(bars["date"])]
    return [d for d in days if start <= d <= end]


def waves(dates: list[str], holding: int, memory: bool) -> list[list[str]]:
    """Consecutive blocks of `holding` trade dates (memory on), or one block of all (memory off)."""
    if not memory:
        return [list(dates)] if dates else []
    return [dates[i:i + holding] for i in range(0, len(dates), holding)]
