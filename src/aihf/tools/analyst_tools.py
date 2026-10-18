"""The tools the analysts can call, with upstream's names, arguments and docstrings.

Copied from upstream `agents/tools.py` (commit 35543d0, Apache-2.0, see
LICENSES/TradingAgents-LICENSE). The model reads a tool's name, argument descriptions and
docstring, so those are kept word for word. What changed is where the data comes from:
upstream sends each call to a globally configured "vendor"; ours asks the DataSource passed to
build_tools(). get_prediction_markets is not built (DEVIATIONS D3).

`trade_date: Annotated[str, InjectedState("trade_date")]` is filled by LangGraph from the
state, and the model never sees it: whatever date the model asks for, the tool clamps it to
the trade date before asking the DataSource.
"""

from typing import Annotated

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from aihf.tools.dates import as_of, as_of_window
from aihf.tools.source import DataSource


def build_tools(data: DataSource) -> dict[str, BaseTool]:
    """Make every analyst tool, reading from `data`. Returns {tool name: tool}."""

    @tool
    def get_stock_data(
        symbol: Annotated[str, "ticker symbol of the company"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve stock price data (OHLCV) for a given ticker symbol.
        Uses the configured core_stock_apis vendor.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
        """
        start_date, end_date = as_of_window(start_date, end_date, trade_date)
        return data.get_stock_data(symbol, start_date, end_date)

    @tool
    def get_indicators(
        symbol: Annotated[str, "ticker symbol of the company"],
        indicator: Annotated[str, "technical indicator to get the analysis and report of"],
        curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
        look_back_days: Annotated[int, "how many days to look back"] = 30,
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve a single technical indicator for a given ticker symbol.
        Uses the configured technical_indicators vendor.
        Args:
            symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
            indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
            curr_date (str): The current trading date you are trading on, YYYY-mm-dd
            look_back_days (int): How many days to look back, default is 30
        Returns:
            str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
        """
        # Like upstream: models sometimes pass "rsi, macd" in one call, so each is served in turn.
        curr_date = as_of(curr_date, trade_date)
        indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
        results = []
        for ind in indicators:
            try:
                results.append(data.get_indicators(symbol, ind, curr_date, look_back_days))
            except ValueError as e:  # e.g. an indicator name we don't support
                results.append(str(e))
        return "\n\n".join(results)

    @tool
    def get_verified_market_snapshot(
        symbol: Annotated[str, "ticker symbol of the company"],
        curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
        look_back_days: Annotated[
            int, "number of recent trading rows to include for sanity-checking"
        ] = 30,
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """Deterministic verification snapshot for exact market-data claims.

    Returns the latest OHLCV row on or before curr_date, common technical
    indicators, and recent closes. Call this before making exact claims about
    price levels, Bollinger bands, RSI, MACD, moving averages, support /
    resistance, or historical comparisons, and treat it as the source of truth.
    """
        # The docstring above is indented by 4, not 8, on purpose: a docstring whose first line
        # sits next to the quotes keeps its indentation in the tool description the model reads,
        # and upstream's (a top-level function) is indented by 4.
        return data.get_verified_market_snapshot(symbol, as_of(curr_date, trade_date), look_back_days)

    @tool
    def get_fundamentals(
        ticker: Annotated[str, "ticker symbol"],
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve comprehensive fundamental data for a given ticker symbol.
        Uses the configured fundamental_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
            curr_date (str): Current date you are trading at, yyyy-mm-dd
        Returns:
            str: A formatted report containing comprehensive fundamental data
        """
        return data.get_fundamentals(ticker, as_of(curr_date, trade_date))

    @tool
    def get_balance_sheet(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve balance sheet data for a given ticker symbol.
        Uses the configured fundamental_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
            freq (str): Reporting frequency: annual/quarterly (default quarterly)
            curr_date (str): Current date you are trading at, yyyy-mm-dd
        Returns:
            str: A formatted report containing balance sheet data
        """
        return data.get_balance_sheet(ticker, freq, as_of(curr_date, trade_date))

    @tool
    def get_cashflow(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve cash flow statement data for a given ticker symbol.
        Uses the configured fundamental_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
            freq (str): Reporting frequency: annual/quarterly (default quarterly)
            curr_date (str): Current date you are trading at, yyyy-mm-dd
        Returns:
            str: A formatted report containing cash flow statement data
        """
        return data.get_cashflow(ticker, freq, as_of(curr_date, trade_date))

    @tool
    def get_income_statement(
        ticker: Annotated[str, "ticker symbol"],
        freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
        curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve income statement data for a given ticker symbol.
        Uses the configured fundamental_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
            freq (str): Reporting frequency: annual/quarterly (default quarterly)
            curr_date (str): Current date you are trading at, yyyy-mm-dd
        Returns:
            str: A formatted report containing income statement data
        """
        return data.get_income_statement(ticker, freq, as_of(curr_date, trade_date))

    @tool
    def get_news(
        ticker: Annotated[str, "Ticker symbol"],
        start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
        end_date: Annotated[str, "End date in yyyy-mm-dd format"],
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve news data for a given ticker symbol.
        Uses the configured news_data vendor.
        Args:
            ticker (str): Ticker symbol
            start_date (str): Start date in yyyy-mm-dd format
            end_date (str): End date in yyyy-mm-dd format
        Returns:
            str: A formatted string containing news data
        """
        start_date, end_date = as_of_window(start_date, end_date, trade_date)
        return data.get_news(ticker, start_date, end_date)

    @tool
    def get_global_news(
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
        look_back_days: Annotated[int | None, "Days to look back; omit to use the configured default"] = None,
        limit: Annotated[int | None, "Max articles to return; omit to use the configured default"] = None,
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve global news data.
        Uses the configured news_data vendor. Defaults for look_back_days and
        limit come from DEFAULT_CONFIG (global_news_lookback_days,
        global_news_article_limit); pass explicit values to override.

        Args:
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): Number of days to look back; omit to inherit config
            limit (int): Maximum number of articles to return; omit to inherit config

        Returns:
            str: A formatted string containing global news data
        """
        return data.get_global_news(as_of(curr_date, trade_date), look_back_days, limit)

    @tool
    def get_insider_transactions(
        ticker: Annotated[str, "ticker symbol"],
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve insider transaction information about a company.
        Uses the configured news_data vendor.
        Args:
            ticker (str): Ticker symbol of the company
        Returns:
            str: A report of insider transaction data
        """
        return data.get_insider_transactions(ticker, trade_date)

    @tool
    def get_macro_indicators(
        indicator: Annotated[
            str,
            "Macro indicator: a friendly alias such as 'cpi', 'core_pce', "
            "'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve', "
            "'real_gdp', 'vix', or a raw FRED series ID such as 'CPIAUCSL'.",
        ],
        curr_date: Annotated[str, "Current date in yyyy-mm-dd format; the end of the window"],
        look_back_days: Annotated[
            int | None, "Trailing window length in days; omit for a 1-year window"
        ] = None,
        trade_date: Annotated[str, InjectedState("trade_date")] = "",
    ) -> str:
        """
        Retrieve a macroeconomic indicator time series from FRED (Federal Reserve
        Economic Data): policy rates, Treasury yields, inflation, labor, and growth.
        Returns the series title, units, frequency, the latest value, the change
        over the window, and a recent observation table. Uses the configured
        macro_data vendor.

        Args:
            indicator (str): Friendly alias or raw FRED series ID
            curr_date (str): Current date in yyyy-mm-dd format
            look_back_days (int): Trailing window length; omit for a 1-year window

        Returns:
            str: A formatted markdown report of the macro series
        """
        return data.get_macro_indicators(indicator, as_of(curr_date, trade_date), look_back_days)

    tools = [
        get_stock_data, get_indicators, get_verified_market_snapshot,
        get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement,
        get_insider_transactions, get_news, get_global_news, get_macro_indicators,
    ]
    return {t.name: t for t in tools}
