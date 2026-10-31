"""A single-asset account: daily targets in, trades at the next open, a value curve out (CLAUDE.md §6.4).

Conventions (the paper gives none):
- $100,000 cash at the close of the window's first trade date; one account per ticker.
- A decision made at the close of day t sets a target exposure (+1 = 100 % long, -1 = 100 %
  short, 0 = cash, None = keep the current one). A change of target is traded at the open of day
  t+1, the first price anyone could trade at: the position is resized to target × the account's
  value at that open. Between changes the shares are kept (the exposure drifts with the price,
  as with real orders and in backtrader); repeating the current target trades nothing. The
  account is valued at every close.
- The decision of the window's last day would trade after the window and is not used: a
  61-day window gives 60 daily returns.
- Profit and loss use dividend-adjusted prices (adjusted open = open × adj_close / close), so a
  dividend counts as return, as in any total-return comparison. (The agents saw prices as
  traded, DEVIATIONS D6: that is about what they read, not about what the account earns.)
- Fractional shares; costs = cost_bps of the traded amount, per side; no borrowing cost for
  shorts (a stated limitation).
"""

from dataclasses import dataclass

import pandas as pd

# Rating → target exposure. None = keep the current position. REVIEW (no readable rating) is a
# hold, and is counted separately by the evaluation, never silently turned into a Hold.
MAPPINGS = {
    "paper3": {"Buy": 1.0, "Overweight": 1.0, "Hold": None, "Underweight": -1.0, "Sell": -1.0, "REVIEW": None},
    "graded5": {"Buy": 1.0, "Overweight": 0.5, "Hold": None, "Underweight": -0.5, "Sell": -1.0, "REVIEW": None},
}


@dataclass(frozen=True)
class SimConfig:
    initial_cash: float = 100_000.0
    cost_bps: float = 0.0  # per side, on the traded amount
    short_mode: str = "short"  # "short": negative targets go short; "flat": they mean cash

    def clip(self, target: float | None) -> float | None:
        if target is None or self.short_mode == "short":
            return target
        if self.short_mode == "flat":
            return max(target, 0.0)
        raise ValueError(f"short_mode must be 'short' or 'flat', got {self.short_mode!r}")


DEFAULT = SimConfig()  # $100k, no costs, shorts allowed


def targets_from_ratings(ratings: dict[str, str], mapping: str) -> dict[str, float | None]:
    """{trade date: rating} → {trade date: target exposure or None (keep)}."""
    table = MAPPINGS[mapping]
    unknown = {r for r in ratings.values() if r not in table}
    if unknown:
        raise ValueError(f"ratings not in mapping {mapping}: {unknown}")
    return {day: table[r] for day, r in ratings.items()}


def adjusted(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars with adjusted open/close: date, open, close (dividend- and split-adjusted)."""
    factor = bars["adj_close"] / bars["close"]
    return pd.DataFrame({"date": pd.to_datetime(bars["date"]), "open": bars["open"] * factor,
                         "close": bars["adj_close"]}).sort_values("date").reset_index(drop=True)


def simulate(bars: pd.DataFrame, targets: dict[str, float | None], dates: list[str],
             config: SimConfig = DEFAULT) -> pd.DataFrame:
    """One row per trade date: the value at its close and what happened that day.

    bars: adjusted bars (see `adjusted`) covering `dates`. targets: decision per date (missing =
    keep). Columns: date, open, close, target (decided at this close), exposure (held from this
    open), shares, cash, value, daily_return, traded, cost.
    """
    px = adjusted(bars).set_index("date")
    days = [pd.Timestamp(d) for d in dates]
    missing = [d for d in days if d not in px.index]
    if missing:
        raise ValueError(f"no price for {missing[:3]}")

    cash, shares, target = config.initial_cash, 0.0, 0.0
    pending: float | None = None  # a new target decided at the last close, traded at this open
    rows = []
    for i, day in enumerate(days):
        o, c = float(px.at[day, "open"]), float(px.at[day, "close"])
        traded = cost = 0.0
        if pending is not None:
            value_at_open = cash + shares * o
            wanted = pending * value_at_open / o
            traded = abs(wanted - shares) * o
            cost = traded * config.cost_bps / 1e4
            cash -= (wanted - shares) * o + cost
            shares, target, pending = wanted, pending, None
        value = cash + shares * c
        decided = config.clip(targets.get(f"{day:%Y-%m-%d}")) if i < len(days) - 1 else None
        if decided is not None and decided != target:
            pending = decided
        rows.append({"date": day, "open": o, "close": c, "target": decided,
                     "exposure": shares * c / value if value else 0.0, "shares": shares, "cash": cash,
                     "value": value, "traded": traded, "cost": cost})
    out = pd.DataFrame(rows)
    out["daily_return"] = out["value"].pct_change()
    return out
