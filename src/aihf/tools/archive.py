"""The DataSource that reads our point-in-time archive.

It serves prices (the Market Analyst's three tools) and SEC filings (the Fundamentals Analyst's
five tools); every other method is still a gap note inherited from GapSource. The rest of
Phase 1.1 adds news, Reddit and macro series here.
"""

import os
from datetime import date
from pathlib import Path

import pandas as pd

from aihf.data.pit import decision_cutoff
from aihf.data.prices import coverage, load_bars
from aihf.data.sec import days_before, facts_known_at, form4_known_at, read_meta
from aihf.tools.market import indicator_window, stock_data_csv, verified_snapshot
from aihf.tools.sec import fundamentals_overview, insider_table, statement_table
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

    # --- SEC filings (data/sec.py, tools/sec.py) ---

    INSIDER_LOOK_BACK_DAYS = 180  # insider trades shown: the last ~6 months before the trade date

    def _facts(self, ticker: str, curr_date: str, what: str):
        """Facts public at the cutoff of `curr_date`, or a gap note saying why there are none."""
        if read_meta(self.root, ticker) is None:
            return gap(what, f"as of {curr_date}", f"{ticker} filings",
                       reason=f"{ticker.upper()} is not in our SEC archive")
        return facts_known_at(self.root, ticker, decision_cutoff(curr_date))

    def _statement(self, kind, title, ticker, freq, curr_date):
        known = self._facts(ticker, curr_date, f"{title} ({freq})")
        if isinstance(known, str):
            return known
        text = statement_table(known, kind, ticker, freq, curr_date, title)
        return text or f"No {freq} {title.lower()} for {ticker.upper()} filed by {curr_date} (SEC EDGAR)"

    def get_balance_sheet(self, ticker, freq, curr_date):
        return self._statement("balance_sheet", "Balance Sheet", ticker, freq, curr_date)

    def get_income_statement(self, ticker, freq, curr_date):
        return self._statement("income_statement", "Income Statement", ticker, freq, curr_date)

    def get_cashflow(self, ticker, freq, curr_date):
        return self._statement("cashflow", "Cash Flow Statement", ticker, freq, curr_date)

    def get_fundamentals(self, ticker, curr_date):
        known = self._facts(ticker, curr_date, "Fundamentals")
        if isinstance(known, str):
            return known
        bars = self._bars(ticker, curr_date)
        if isinstance(bars, str):  # no prices: the overview leaves out the price-based fields
            bars = pd.DataFrame(columns=["date", "close", "high", "low", "dividends"])
        text = fundamentals_overview(known, bars, ticker, curr_date)
        return text or f"No fundamentals for {ticker.upper()} filed by {curr_date} (SEC EDGAR)"

    def get_insider_transactions(self, ticker, curr_date):
        meta = read_meta(self.root, ticker)
        since = days_before(curr_date, self.INSIDER_LOOK_BACK_DAYS)
        window = f"{since}..{curr_date}"
        if meta is None:
            return gap("Insider transactions", window, f"{ticker} insider trading",
                       reason=f"{ticker.upper()} is not in our SEC archive")
        if since < meta.form4_start or date.fromisoformat(curr_date) > meta.form4_end:
            return gap("Insider transactions", window, f"{ticker} insider trading",
                       reason=f"our Form 4 archive covers {meta.form4_start}..{meta.form4_end} only")
        trades = form4_known_at(self.root, ticker, decision_cutoff(curr_date), since)
        return insider_table(trades, ticker, curr_date, since)
