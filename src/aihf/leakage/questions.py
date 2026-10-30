"""Questions with known answers, built from our archive, for the leakage probe (CLAUDE.md §6.2).

Each question is about one window: the test window (the backtest's, after the model's knowledge
cutoff) or a control window (before the cutoff, which the model should know). Every question
carries its right answer and a naive baseline answer, so a model can be compared with guessing.

Kinds:
- close: a closing price on a date (as traded). Naive: the last close before the window.
- direction: month-end close higher or lower than the previous month-end. Chance: 50 %.
- bucket: the month's move in one of four bands (A > +5 %, B 0..+5 %, C -5..0 %, D < -5 %). Chance: 25 %.
- revenue / eps: a figure from a 10-Q or 10-K accepted in the window. Naive: the same period a
  year earlier.
- macro: a FRED value (as first published in the window, or as known at the start of our
  vintages for the control window). Naive: the last value before the window.
"""

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import pandas as pd

from aihf.data.macro import series_file
from aihf.data.prices import price_file
from aihf.data.sec import load_facts

BUCKETS = {"A": "rose more than 5%", "B": "rose between 0% and 5%", "C": "fell between 0% and 5%",
           "D": "fell more than 5%"}
REVENUE_TAGS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues")
MACRO = {  # series → (question wording, unit text); rates are scored in points, CPI in %
    "UNRATE": ("the US unemployment rate", "in percent"),
    "FEDFUNDS": ("the effective federal funds rate (monthly average)", "in percent"),
    "CPIAUCSL": ("the US CPI-U index level (seasonally adjusted, 1982-84=100)", "as an index level"),
}
DAILY_MACRO = {"DGS10": ("the 10-year US Treasury constant-maturity yield", "in percent")}


@dataclass
class Question:
    id: str
    window: str  # "test" or "control"
    kind: str  # close, direction, bucket, revenue, eps, macro
    subject: str  # ticker or FRED series
    text: str  # what the model is asked
    answer_format: str  # "number" or "higher/lower" or "letter"
    truth: float | str
    naive: float | str | None  # the baseline guess (None for direction/bucket: chance)
    error_unit: str = "pct"  # "pct" relative error, "abs" absolute error in the answer's unit

    def to_dict(self) -> dict:
        return asdict(self)


def _bars(root, ticker) -> pd.DataFrame:
    df = pd.read_parquet(price_file(root, ticker), columns=["date", "close"])
    return df.sort_values("date").reset_index(drop=True)


def _month_ends(bars: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    """The last trading day of each month touching [start, end], plus the one before."""
    inside = bars[(bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))]
    ends = inside.groupby(inside["date"].dt.to_period("M"))["date"].max().tolist()
    before = bars[bars["date"] < pd.Timestamp(start)]["date"].max()
    return [before, *ends]


def price_questions(root, ticker: str, company: str, window: str, start: str, end: str,
                    n_dates: int = 6) -> list[Question]:
    bars = _bars(root, ticker)
    inside = bars[(bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))]
    naive = float(bars[bars["date"] < pd.Timestamp(start)]["close"].iloc[-1])
    step = max(len(inside) // n_dates, 1)
    picked = inside.iloc[step // 2::step].head(n_dates)
    who = f"{company} ({ticker})" if company else ticker
    qs = [Question(f"{window}-close-{ticker}-{d:%Y-%m-%d}", window, "close", ticker,
                   f"What was the closing price of {who} stock on {d:%Y-%m-%d}? "
                   "Answer with the price in US dollars.", "number", round(float(c), 2), round(naive, 2))
          for d, c in zip(picked["date"], picked["close"], strict=True)]

    ends = _month_ends(bars, start, end)
    close = bars.set_index("date")["close"]
    for prev, cur in zip(ends, ends[1:], strict=False):
        change = close[cur] / close[prev] - 1
        qs.append(Question(f"{window}-direction-{ticker}-{cur:%Y-%m}", window, "direction", ticker,
                           f"Was the closing price of {who} stock on {cur:%Y-%m-%d} higher or lower than "
                           f"on {prev:%Y-%m-%d}? Answer with one word: higher or lower.", "higher/lower",
                           "higher" if change > 0 else "lower", None))
        letter = "A" if change > 0.05 else "B" if change > 0 else "C" if change > -0.05 else "D"
        options = " ".join(f"{k}) {v}." for k, v in BUCKETS.items())
        qs.append(Question(f"{window}-bucket-{ticker}-{cur:%Y-%m}", window, "bucket", ticker,
                           f"From the close on {prev:%Y-%m-%d} to the close on {cur:%Y-%m-%d}, {who} stock: "
                           f"{options} Answer with one letter.", "letter", letter, None))
    return qs


def _span(row) -> int:
    return (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days


def filing_questions(root, ticker: str, company: str, window: str, start: str, end: str) -> list[Question]:
    """Revenue and diluted EPS of the latest 10-Q/10-K period whose filing was accepted in the window."""
    facts = load_facts(root, ticker)
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    qs = []
    for kind, tags, unit, scale, unit_text in [
        ("revenue", REVENUE_TAGS, "USD", 1e9, "in billions of US dollars"),
        ("eps", ("EarningsPerShareDiluted",), "USD/shares", 1.0, "in US dollars per share"),
    ]:
        rows = facts[facts["tag"].isin(tags) & (facts["unit"] == unit) & facts["start"].notna()
                     & facts["form"].str.startswith(("10-Q", "10-K"))]
        filed = rows[(rows["available_at"] >= lo) & (rows["available_at"] < hi)]
        if not len(filed):
            continue
        spans = filed.apply(_span, axis=1)
        current = filed[spans.between(60, 115) | spans.between(300, 400)]
        current = current[current["end"] == current["end"].max()]
        if not len(current):
            continue
        row = current.sort_values("available_at").iloc[0]  # its first filing
        span = _span(row)
        prior_end = date.fromisoformat(row["end"]) - timedelta(days=365)
        prior = rows[rows.apply(lambda r, s=span, p=prior_end: abs(_span(r) - s) <= 10
                                and abs((date.fromisoformat(r["end"]) - p).days) <= 10, axis=1)]
        # Last year's figure as shown in the same filing (restated after a stock split), else
        # the latest version known by then: a pre-split EPS would be a meaningless baseline.
        same = prior[prior["accn"] == row["accn"]]
        known = prior[prior["available_at"] <= row["available_at"]].sort_values("available_at")
        base = same if len(same) else known.tail(1)
        naive = float(base.iloc[0]["val"]) / scale if len(base) else None
        period = "fiscal quarter" if span < 200 else "fiscal year"
        qs.append(Question(f"{window}-{kind}-{ticker}-{row['end']}", window, kind, ticker,
                           f"What {'total revenue' if kind == 'revenue' else 'diluted earnings per share'} did "
                           f"{company} report for the {period} from {row['start']} to {row['end']}? "
                           f"Answer with a number {unit_text}.", "number",
                           round(float(row["val"]) / scale, 4), None if naive is None else round(naive, 4)))
    return qs


def macro_questions(root, window: str, start: str, end: str, first_release: bool) -> list[Question]:
    """FRED questions: monthly series for the months in the window, DGS10 on three dates.

    first_release (the test window): each value as first published, for observations first
    published inside the window. Otherwise (the control window, whose own vintages we do not
    archive): the values as known at our earliest vintage. Naive: the last observation before
    the window, as known at the window's start (or our earliest vintage).
    """
    qs = []
    for series, (what, unit_text) in {**MACRO, **DAILY_MACRO}.items():
        path = series_file(root, series)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[~df["value"].isin([".", ""])]
        as_of = max(start, df["realtime_start"].min())
        known = df[(df["realtime_start"] <= as_of) & (df["realtime_end"] >= as_of)].sort_values("date")
        month_start = start[:8] + "01"  # monthly observations are dated the 1st: January is "2023-01-01"
        before = known[known["date"] < month_start]
        if not len(before):
            continue
        naive = float(before["value"].iloc[-1])
        if first_release:
            firsts = df.sort_values("realtime_start").drop_duplicates("date")  # first version of each date
            rows = firsts[(firsts["realtime_start"] >= start) & (firsts["realtime_start"] <= end)
                          & (firsts["date"] <= end)]
        else:
            rows = known[(known["date"] >= month_start) & (known["date"] <= end)]
        rows = rows.sort_values("date")
        if series in DAILY_MACRO:
            rows = rows.iloc[len(rows) // 6::max(len(rows) // 3, 1)].head(3)
        for _, r in rows.iterrows():
            when = f"on {r['date']}" if series in DAILY_MACRO else f"for {r['date'][:7]}"
            if first_release and series not in DAILY_MACRO:
                when += " (as first reported)"
            qs.append(Question(f"{window}-macro-{series}-{r['date']}", window, "macro", series,
                               f"What was {what} {when}? Answer with a number {unit_text}.", "number",
                               float(r["value"]), naive, error_unit="pct" if series == "CPIAUCSL" else "abs"))
    return qs


def build_questions(root, instruments: dict, tickers, windows: dict[str, tuple[str, str]]) -> list[Question]:
    """All questions for these tickers and windows; windows = {"test": (start, end), "control": ...}."""
    qs = []
    for window, (start, end) in windows.items():
        for ticker in tickers:
            if not price_file(root, ticker).exists():
                continue  # not archived: no questions about it
            company = (instruments.get(ticker) or {}).get("company", "")
            qs += price_questions(root, ticker, company, window, start, end)
            if (instruments.get(ticker) or {}).get("cik"):
                qs += filing_questions(root, ticker, company, window, start, end)
        qs += macro_questions(root, window, start, end, first_release=(window == "test"))
    return qs
