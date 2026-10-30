"""Coverage report: for every ticker and trade date, what each data tool would give an agent.

It calls the same DataSource methods the agents call, with the arguments they typically use
(the Sentiment Analyst's 7-day window, upstream's default look-backs), and records whether
each answer is real data or a gap note, plus how many items it holds. One row per
(ticker, trade date); `summarize` turns it into per-tool totals for DATA_CARD.
"""

import re
from datetime import date, timedelta

import pandas as pd

from aihf.tools.logged import is_gap


def _lines(pattern: str):
    """An item counter: how many lines of the answer match `pattern`."""
    compiled = re.compile(pattern, re.MULTILINE)
    return lambda text: len(compiled.findall(text))


def _records(text: str) -> int:
    match = re.search(r"Total records: (\d+)", text)
    return int(match.group(1)) if match else 0


# (column, how to call the DataSource, how to count items in the answer)
CHECKS = [
    ("prices", lambda d, t, day: d.get_stock_data(t, _back(day, 30), day), _records),
    ("indicators", lambda d, t, day: d.get_indicators(t, "rsi", day, 30), None),
    ("snapshot", lambda d, t, day: d.get_verified_market_snapshot(t, day, 30), None),
    ("fundamentals", lambda d, t, day: d.get_fundamentals(t, day), _lines(r"^[A-Z][^#\n]*: ")),
    ("balance_sheet", lambda d, t, day: d.get_balance_sheet(t, "quarterly", day), None),
    ("income_statement", lambda d, t, day: d.get_income_statement(t, "quarterly", day), None),
    ("cashflow", lambda d, t, day: d.get_cashflow(t, "quarterly", day), None),
    ("insider", lambda d, t, day: d.get_insider_transactions(t, day), _lines(r"^\d{4}-\d{2}-\d{2},")),
    ("news", lambda d, t, day: d.get_news(t, _back(day, 7), day), _lines(r"^### ")),
    ("global_news", lambda d, t, day: d.get_global_news(day, None, None), _lines(r"^### ")),
    ("macro_cpi", lambda d, t, day: d.get_macro_indicators("cpi", day, None), _lines(r"^\| \d{4}-")),
    ("macro_10y", lambda d, t, day: d.get_macro_indicators("10y_treasury", day, None), _lines(r"^\| \d{4}-")),
    ("reddit", lambda d, t, day: d.get_reddit_posts(t, _back(day, 7), day), _lines(r"^  \[\d{4}")),
    ("stocktwits", lambda d, t, day: d.get_stocktwits_messages(t, _back(day, 7), day), None),
]


def _back(day: str, days: int) -> str:
    return f"{date.fromisoformat(day) - timedelta(days=days)}"


def coverage_table(data, tickers, trade_dates) -> pd.DataFrame:
    """One row per (ticker, trade date): for each check, "gap" or the number of items."""
    rows = []
    for ticker in tickers:
        for day in trade_dates:
            row = {"ticker": ticker, "trade_date": day}
            for name, call, count in CHECKS:
                text = call(data, ticker, day)
                if is_gap(text) or text.startswith(("No ", "FRED:")):
                    row[name] = "gap" if is_gap(text) else "empty"
                else:
                    row[name] = count(text) if count else "ok"
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Per check: ticker-days with data, gaps, empties, and min / median / max items."""
    out = []
    for name, _, count in CHECKS:
        col = table[name]
        numbers = pd.to_numeric(col, errors="coerce").dropna()
        out.append({
            "check": name, "with_data": int((~col.isin(["gap", "empty"])).sum()),
            "gap": int((col == "gap").sum()), "empty": int((col == "empty").sum()),
            "min": int(numbers.min()) if count and len(numbers) else None,
            "median": float(numbers.median()) if count and len(numbers) else None,
            "max": int(numbers.max()) if count and len(numbers) else None,
        })
    return pd.DataFrame(out)
