"""Where the tools get their data: the `DataSource` interface, and a source that has nothing yet.

Tools never fetch anything themselves. They ask a DataSource, and every DataSource answers
with text in the shape upstream's tools return, so the copied prompts fit. Dates reaching a
DataSource are already clamped to the trade date (see tools/dates.py).

GapSource answers every question with an explicit "not available" note instead of an empty
string: an empty answer would read as "nothing happened", which is a claim nobody checked.
Real sources (prices in Phase 1 step 3b, the archive in 6.1) extend it and override what they
can serve; whatever they don't override stays a gap.
"""

from typing import Protocol


class DataSource(Protocol):
    """Everything the four analysts can ask for. Each method returns text for the model."""

    # Market Analyst
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> str: ...
    def get_indicators(self, symbol: str, indicator: str, curr_date: str, look_back_days: int) -> str: ...
    def get_verified_market_snapshot(self, symbol: str, curr_date: str, look_back_days: int) -> str: ...

    # Fundamentals Analyst
    def get_fundamentals(self, ticker: str, curr_date: str) -> str: ...
    def get_balance_sheet(self, ticker: str, freq: str, curr_date: str) -> str: ...
    def get_cashflow(self, ticker: str, freq: str, curr_date: str) -> str: ...
    def get_income_statement(self, ticker: str, freq: str, curr_date: str) -> str: ...
    def get_insider_transactions(self, ticker: str, curr_date: str) -> str: ...

    # News Analyst (get_news is also read by the Sentiment Analyst)
    def get_news(self, ticker: str, start_date: str, end_date: str) -> str: ...
    def get_global_news(self, curr_date: str, look_back_days: int | None, limit: int | None) -> str: ...
    def get_macro_indicators(self, indicator: str, curr_date: str, look_back_days: int | None) -> str: ...

    # Sentiment Analyst
    def get_stocktwits_messages(self, ticker: str, start_date: str, end_date: str) -> str: ...
    def get_reddit_posts(self, ticker: str, start_date: str, end_date: str) -> str: ...


def gap(source: str, window: str, subject: str, reason: str = "not in our point-in-time archive yet") -> str:
    """The note a tool returns for data we don't have. Same shape as upstream's coverage_gap."""
    return f"<{source} unavailable for {window}: {reason}, so this is not an absence of {subject}>"


class GapSource:
    """A DataSource with no data at all: every answer is a gap note."""

    def get_stock_data(self, symbol, start_date, end_date):
        return gap("Price data", f"{start_date}..{end_date}", f"trading in {symbol}")

    def get_indicators(self, symbol, indicator, curr_date, look_back_days):
        return gap(f"Indicator {indicator}", f"{look_back_days} days to {curr_date}", f"{symbol} price history")

    def get_verified_market_snapshot(self, symbol, curr_date, look_back_days):
        return gap("Market snapshot", curr_date, f"{symbol} price history")

    def get_fundamentals(self, ticker, curr_date):
        return gap("Fundamentals", f"as of {curr_date}", f"{ticker} financial data")

    def get_balance_sheet(self, ticker, freq, curr_date):
        return gap(f"Balance sheet ({freq})", f"as of {curr_date}", f"{ticker} filings")

    def get_cashflow(self, ticker, freq, curr_date):
        return gap(f"Cash flow statement ({freq})", f"as of {curr_date}", f"{ticker} filings")

    def get_income_statement(self, ticker, freq, curr_date):
        return gap(f"Income statement ({freq})", f"as of {curr_date}", f"{ticker} filings")

    def get_insider_transactions(self, ticker, curr_date):
        return gap("Insider transactions", f"as of {curr_date}", f"{ticker} insider trading")

    def get_news(self, ticker, start_date, end_date):
        return gap("Company news", f"{start_date}..{end_date}", f"news about {ticker}")

    def get_global_news(self, curr_date, look_back_days, limit):
        window = f"{look_back_days} days to {curr_date}" if look_back_days else f"the days to {curr_date}"
        return gap("Global news", window, "global news")

    def get_macro_indicators(self, indicator, curr_date, look_back_days):
        window = f"{look_back_days} days to {curr_date}" if look_back_days else f"the year to {curr_date}"
        return gap(f"Macro series {indicator}", window, "macro data")

    def get_stocktwits_messages(self, ticker, start_date, end_date):
        # Permanent: StockTwits has no free historical access (DEVIATIONS D4).
        return gap("StockTwits", f"{start_date}..{end_date}", f"StockTwits posts about {ticker}",
                   reason="no free historical source exists")

    def get_reddit_posts(self, ticker, start_date, end_date):
        return gap("Reddit", f"{start_date}..{end_date}", f"Reddit posts about {ticker}")
