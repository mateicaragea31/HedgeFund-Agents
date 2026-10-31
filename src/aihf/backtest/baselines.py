"""The paper's rule-based baselines, with standard parameters (the paper gives none; CLAUDE.md §6.4).

Each turns the closes known on day t (as traded, like the agents saw them) into a target for the
next open: +1 long, -1 short, 0 cash, None = keep. They start from cash and act only on their
signals. Indicators are computed on the whole price history before the window, so they are
already valid on the first trade date; nothing after day t is ever used (each signal on day t
looks only at rows up to t).

- Buy & Hold: long on the first day, then keep.
- MACD(12, 26, 9): MACD = EMA12 - EMA26, signal = EMA9 of MACD. MACD crossing above the signal
  → long; crossing below → short.
- KDJ(9, 3, 3) + RSI(14, 30/70): RSV = (C - L9) / (H9 - L9) · 100, K = 2/3 K' + 1/3 RSV,
  D = 2/3 D' + 1/3 K (K, D start at 50); RSI = Wilder's RSI(14). Long when K crosses above D
  while K < 20, or RSI < 30 (oversold); short when K crosses below D while K > 80, or RSI > 70.
- ZMR (z-score mean reversion): z = (C - SMA20) / SD20. Long when z < -1, short when z > +1;
  close the position (cash) when z crosses back through 0. Tracked from the window's first day.
- SMA(20/50): SMA20 crossing above SMA50 → long; crossing below → short.
"""

import pandas as pd


def _crosses_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _crosses_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def macd_signals(bars: pd.DataFrame) -> pd.Series:
    c = bars["close"]
    macd = _ema(c, 12) - _ema(c, 26)
    signal = _ema(macd, 9)
    out = pd.Series(None, index=bars.index, dtype=object)
    out[_crosses_up(macd, signal)] = 1.0
    out[_crosses_down(macd, signal)] = -1.0
    return out


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def kdj(bars: pd.DataFrame, n: int = 9) -> tuple[pd.Series, pd.Series]:
    low, high = bars["low"].rolling(n).min(), bars["high"].rolling(n).max()
    rsv = ((bars["close"] - low) / (high - low) * 100).fillna(50)
    k, d = [], []
    kp = dp = 50.0
    for v in rsv:
        kp = 2 / 3 * kp + 1 / 3 * v
        dp = 2 / 3 * dp + 1 / 3 * kp
        k.append(kp)
        d.append(dp)
    return pd.Series(k, index=bars.index), pd.Series(d, index=bars.index)


def kdj_rsi_signals(bars: pd.DataFrame) -> pd.Series:
    k, d = kdj(bars)
    r = rsi(bars["close"])
    out = pd.Series(None, index=bars.index, dtype=object)
    out[(_crosses_up(k, d) & (k < 20)) | (r < 30)] = 1.0
    out[(_crosses_down(k, d) & (k > 80)) | (r > 70)] = -1.0
    return out


def zmr_signals(bars: pd.DataFrame, window: int = 20, entry: float = 1.0, start: int = 0) -> pd.Series:
    """z from the whole history; the position is tracked from row `start` on (it starts in cash)."""
    c = bars["close"]
    z = (c - c.rolling(window).mean()) / c.rolling(window).std(ddof=1)
    out = pd.Series(None, index=bars.index, dtype=object)
    position = 0.0
    for i in range(start, len(z)):
        zi = z.iloc[i]
        if pd.isna(zi):
            continue
        if (position > 0 and zi >= 0) or (position < 0 and zi <= 0):
            position = 0.0
            out.iloc[i] = 0.0
        if position == 0 and zi < -entry:
            position = 1.0
            out.iloc[i] = 1.0
        elif position == 0 and zi > entry:
            position = -1.0
            out.iloc[i] = -1.0
    return out


def sma_signals(bars: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    c = bars["close"]
    f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
    out = pd.Series(None, index=bars.index, dtype=object)
    out[_crosses_up(f, s)] = 1.0
    out[_crosses_down(f, s)] = -1.0
    return out


SIGNALS = {"MACD": macd_signals, "KDJ&RSI": kdj_rsi_signals, "ZMR": zmr_signals, "SMA": sma_signals}


def baseline_targets(bars: pd.DataFrame, dates: list[str]) -> dict[str, dict[str, float | None]]:
    """{baseline name: {trade date: target}} for the window's dates, indicators from the full history.

    Every baseline starts the window in cash (as the account does) and acts on its own signals.
    """
    bars = bars.sort_values("date").reset_index(drop=True)
    day = bars["date"].dt.strftime("%Y-%m-%d")
    window = set(dates)
    first = int(day[day == dates[0]].index[0])
    out = {"Buy&Hold": {dates[0]: 1.0}}
    for name, fn in SIGNALS.items():
        signals = fn(bars, start=first) if name == "ZMR" else fn(bars)
        out[name] = {d: (None if pd.isna(s) else float(s)) for d, s in zip(day, signals, strict=True)
                     if d in window}
    return out
