"""The DataSource that reads our point-in-time archive.

For now it serves prices (the Market Analyst's three tools); every other method is still a
gap note inherited from GapSource. Phase 1.1 adds news, Reddit and EDGAR filings here.
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd

from aihf.data.pit import decision_cutoff
from aihf.data.prices import coverage, load_bars
from aihf.tools.market import indicator_window, stock_data_csv, verified_snapshot
from aihf.tools.source import GapSource, gap


class ArchiveSource(GapSource):
    """Reads <root>/prices/... (see data/prices.py). Dates arrive already clamped by the tools."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def from_env(cls) -> "ArchiveSource":
        """The archive under $AIHF_DATA/archive."""
        return cls(Path(os.environ["AIHF_DATA"]) / "archive")

    def _bars(self, symbol: str, curr_date: str) -> pd.DataFrame | str:
        """The bars known at the decision cutoff of `curr_date`, or a gap note saying why not.

        A date past the end of the file is a gap, never "the latest bars we have": serving
        older prices as if they were current is the stale-data trap upstream also guards against.
        """
        cov = coverage(self.root, symbol)
        if cov is None:
            return gap("Price data", f"up to {curr_date}", f"trading in {symbol}",
                       reason=f"{symbol.upper()} is not in our price archive")
        if date.fromisoformat(curr_date) > cov.last:
            return gap("Price data", f"up to {curr_date}", f"trading in {symbol}",
                       reason=f"our price archive for {symbol.upper()} ends on {cov.last}")
        return load_bars(self.root, symbol, decision_cutoff(curr_date))

    def get_stock_data(self, symbol, start_date, end_date):
        bars = self._bars(symbol, end_date)
        return bars if isinstance(bars, str) else stock_data_csv(bars, symbol, start_date, end_date)

    def get_indicators(self, symbol, indicator, curr_date, look_back_days):
        bars = self._bars(symbol, curr_date)
        return bars if isinstance(bars, str) else indicator_window(bars, indicator, curr_date, look_back_days)

    def get_verified_market_snapshot(self, symbol, curr_date, look_back_days):
        bars = self._bars(symbol, curr_date)
        return bars if isinstance(bars, str) else verified_snapshot(bars, symbol, curr_date, look_back_days)
