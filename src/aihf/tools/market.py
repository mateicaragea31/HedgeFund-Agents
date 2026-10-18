"""The Market Analyst's tool outputs: price CSV, one indicator over a window, verified snapshot.

The text is shaped exactly like upstream's (`dataflows/vendors/yahoo/market.py` and
`snapshot.py`, commit 35543d0, Apache-2.0, see LICENSES/TradingAgents-LICENSE), so the copied
Market Analyst prompt fits. Each function receives bars already cut at the decision cutoff
(data/prices.py: load_bars) and never looks at anything later.

Indicators come from stockstats, computed on the as-traded prices (DEVIATIONS D6).
"""

from datetime import datetime, timedelta

import pandas as pd
from stockstats import wrap

# Upstream's best_ind_params (market.py), word for word: the indicators get_indicators serves,
# and the description appended under each one's values.
INDICATOR_DESCRIPTIONS = {
    # Moving Averages
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. "
        "Usage: Identify trend direction and serve as dynamic support/resistance. "
        "Tips: It lags price; combine with faster indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. "
        "Usage: Confirm overall market trend and identify golden/death cross setups. "
        "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. "
        "Usage: Capture quick shifts in momentum and potential entry points. "
        "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
    ),
    # MACD Related
    "macd": (
        "MACD: Computes momentum via differences of EMAs. "
        "Usage: Look for crossovers and divergence as signals of trend changes. "
        "Tips: Confirm with other indicators in low-volatility or sideways markets."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. "
        "Usage: Use crossovers with the MACD line to trigger trades. "
        "Tips: Should be part of a broader strategy to avoid false positives."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. "
        "Usage: Visualize momentum strength and spot divergence early. "
        "Tips: Can be volatile; complement with additional filters in fast-moving markets."
    ),
    # Momentum Indicators
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. "
        "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
        "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
    ),
    # Volatility Indicators
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
        "Usage: Acts as a dynamic benchmark for price movement. "
        "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
        "Usage: Signals potential overbought conditions and breakout zones. "
        "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
        "Usage: Indicates potential oversold conditions. "
        "Tips: Use additional analysis to avoid false reversal signals."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. "
        "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
        "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
    ),
    # Volume-Based Indicators
    "vwma": (
        "VWMA: A moving average weighted by volume. "
        "Usage: Confirm trends by integrating price action with volume data. "
        "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
    ),
    "mfi": (
        "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
        "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
        "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
    ),
}

# Upstream's DEFAULT_SNAPSHOT_INDICATORS (snapshot.py): the same set every run.
SNAPSHOT_INDICATORS = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _ohlcv(bars: pd.DataFrame) -> pd.DataFrame:
    """Our archive columns renamed to Yahoo's, as upstream's text shows them."""
    return bars.rename(columns={
        "date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "dividends": "Dividends", "stock_splits": "Stock Splits",
    })[["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]]


def _with_indicators(bars: pd.DataFrame, names) -> pd.DataFrame:
    """stockstats over the whole history up to the cutoff; one column per indicator."""
    stock = wrap(bars[["date", "open", "high", "low", "close", "volume"]].copy())
    for name in names:
        stock[name]  # reading a column makes stockstats compute it
    return stock


def stock_data_csv(bars: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> str:
    """Upstream's get_YFin_data_online text: a 2-line header, then the bars as CSV."""
    rows = bars[(bars["date"] >= pd.Timestamp(start_date)) & (bars["date"] <= pd.Timestamp(end_date))]
    if rows.empty:
        return f"No trading days for {symbol.upper()} between {start_date} and {end_date}."
    data = _ohlcv(rows).set_index("Date")
    for col in ["Open", "High", "Low", "Close"]:
        data[col] = data[col].round(2)
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n\n"
    return header + data.to_csv()


def indicator_window(bars: pd.DataFrame, indicator: str, curr_date: str, look_back_days: int) -> str:
    """Upstream's get_stock_stats_indicators_window text: one line per calendar day, newest first.

    Raises ValueError for an indicator we don't serve (the tool shows the message to the model).
    """
    if indicator not in INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(INDICATOR_DESCRIPTIONS.keys())}"
        )
    stock = _with_indicators(bars, [indicator])
    values = {
        day.strftime("%Y-%m-%d"): ("N/A" if pd.isna(v) else str(v))
        for day, v in zip(bars["date"], stock[indicator], strict=True)
    }
    curr = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr - timedelta(days=look_back_days)
    lines = ""
    day = curr
    while day >= before:
        key = day.strftime("%Y-%m-%d")
        lines += f"{key}: {values.get(key, 'N/A: Not a trading day (weekend or holiday)')}\n"
        day -= timedelta(days=1)
    return (
        f"## {indicator} values from {before:%Y-%m-%d} to {curr_date}:\n\n"
        + lines
        + "\n\n"
        + INDICATOR_DESCRIPTIONS[indicator]
    )


def _fmt(value) -> str:
    """Upstream's snapshot number format: floats with 2 decimals, dates as YYYY-MM-DD."""
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def verified_snapshot(bars: pd.DataFrame, symbol: str, curr_date: str, look_back_days: int) -> str:
    """Upstream's build_verified_market_snapshot text: latest row, 11 indicators, recent closes."""
    if bars.empty:
        return f"No OHLCV rows on or before {curr_date} for {symbol}."
    stock = _with_indicators(bars, SNAPSHOT_INDICATORS)
    df = _ohlcv(bars)
    latest = df.iloc[-1]
    recent = df.tail(max(1, min(int(look_back_days), 30)))

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest trading row used: {_fmt(latest['Date'])}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest[field])} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name in SNAPSHOT_INDICATORS:
        lines.append(f"| {name} | {_fmt(stock[name].iloc[-1])} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row['Close'])} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)
