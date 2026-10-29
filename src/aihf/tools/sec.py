"""Text for the Fundamentals Analyst's tools, built from SEC facts known at the decision cutoff.

- statement_table: balance sheet / income statement / cash flow, copied from upstream
  `dataflows/vendors/sec_edgar.py` (commit 35543d0, Apache-2.0): same line items, tags, spans
  and output text. Only the clock differs: a fact counts from its filing's acceptance time
  instead of its filing date (DEVIATIONS D11).
- fundamentals_overview: upstream's yfinance overview fields, computed from filings and our
  archived prices as they stood on the trade date. Upstream returns a gap for past dates,
  because its only source is today's Yahoo profile (DEVIATIONS D12).
- insider_table: Form 4 trades, dated by when they became public.

All take facts/trades already cut at the decision cutoff (data/sec.py); they never look further.
"""

from datetime import date, timedelta

import pandas as pd

# --- statements (upstream's sec_edgar vendor) ------------------------------------------------

# Line items, each with the tags filers use for it, best first. First match wins
# and values are never summed across tags: a company reporting revenue under two
# tags would otherwise be counted twice.
_STATEMENTS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "balance_sheet": [
        ("Total Assets", ("Assets",)),
        ("Current Assets", ("AssetsCurrent",)),
        ("Cash and Equivalents", ("CashAndCashEquivalentsAtCarryingValue",)),
        ("Total Liabilities", ("Liabilities",)),
        ("Current Liabilities", ("LiabilitiesCurrent",)),
        ("Stockholders Equity", ("StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")),
    ],
    "income_statement": [
        ("Revenue", ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                     "SalesRevenueNet")),
        ("Cost of Revenue", ("CostOfRevenue", "CostOfGoodsAndServicesSold")),
        ("Gross Profit", ("GrossProfit",)),
        ("Operating Income", ("OperatingIncomeLoss",)),
        ("Net Income", ("NetIncomeLoss",)),
        ("Diluted EPS", ("EarningsPerShareDiluted",)),
    ],
    "cashflow": [
        ("Operating Cash Flow", ("NetCashProvidedByUsedInOperatingActivities",
                                 "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
        ("Investing Cash Flow", ("NetCashProvidedByUsedInInvestingActivities",)),
        ("Financing Cash Flow", ("NetCashProvidedByUsedInFinancingActivities",)),
        ("Capital Expenditure", ("PaymentsToAcquirePropertyPlantAndEquipment",
                                 "PaymentsToAcquireProductiveAssets")),
    ],
}

# A statement's figures cover a span: a quarter is about 90 days, a year about
# 365. One filing reports both the quarter and the year to date under the same
# end date, so a match on the end date alone can report half a year as a quarter.
_SPANS = {"quarterly": (60, 115), "annual": (300, 400)}

# A fiscal year is a period an annual report covers. A 10-Q balance has no span
# to reject, and some filers' 10-Qs report twelve-month totals that pass the span
# check, so either would read as a fiscal year.
_ANNUAL_FORMS = ("10-K", "20-F", "40-F")

# Ours: the most recent periods shown per table. Upstream's SEC table lists every period since
# 2008 (60+ columns); its default statements (yfinance) show about 5 quarters or 4 years.
_MAX_PERIODS = {"quarterly": 8, "annual": 5}

# Ours: figures older than this are left out of the overview. A filer that stopped using a tag
# (Amazon's GrossProfit) would otherwise report a figure from years ago as current.
_STALE_INSTANT_DAYS = 400  # a balance: about a year after its quarter ended
_STALE_YEAR_DAYS = 460  # a fiscal year: about 15 months after it ended


def _span_days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _as_of(known: pd.DataFrame, tags: tuple[str, ...], span: tuple[int, int],
           forms: tuple[str, ...] = ()) -> tuple[dict, str]:
    """({period end: value}, unit) for the first tag the filer reports, as known at the cutoff.

    Upstream's `_as_of`, with `available_at` (acceptance time) where upstream compares the
    filing date. A period reported more than once takes its latest filing known at the cutoff,
    so an amendment counts from the moment it was accepted and not before.
    """
    low, high = span
    values: dict[str, float] = {}
    chosen_unit = "USD"
    us_gaap = known[known["taxonomy"] == "us-gaap"]
    for tag in tags:
        tag_rows = us_gaap[us_gaap["tag"] == tag]
        for unit, unit_rows in tag_rows.groupby("unit", sort=False):
            latest: dict[str, dict] = {}
            covered: set[str] = set()  # period ends a filing of `forms` reports
            for fact in unit_rows.to_dict("records"):
                if fact["end"] in values:
                    continue
                # A duration fact (revenue, cash flow) must cover the span asked for.
                # An instant fact (a balance) has no span and serves both.
                if isinstance(fact["start"], str) and not low <= _span_days(fact["start"], fact["end"]) <= high:
                    continue
                if not forms or fact["form"].startswith(forms):
                    covered.add(fact["end"])
                seen = latest.get(fact["end"])
                if seen is None or fact["available_at"] >= seen["available_at"]:
                    latest[fact["end"]] = fact
            latest = {end: fact for end, fact in latest.items() if end in covered}
            if latest:
                chosen_unit = unit
                values.update({end: fact["val"] for end, fact in latest.items()})
    return dict(sorted(values.items())), chosen_unit


def statement_table(known: pd.DataFrame, kind: str, ticker: str, freq: str, curr_date: str,
                    title: str) -> str | None:
    """Upstream's statement text, or None when nothing of that kind was public yet."""
    quarterly = freq.lower() == "quarterly"
    span = _SPANS["quarterly" if quarterly else "annual"]
    forms = () if quarterly else _ANNUAL_FORMS
    lines = {label: _as_of(known, tags, span, forms) for label, tags in _STATEMENTS[kind]}
    periods = sorted({end for values, _ in lines.values() for end in values})
    if not periods:
        return None
    periods = periods[-_MAX_PERIODS["quarterly" if quarterly else "annual"]:]

    header = (
        f"# {title} for {ticker.upper()} ({freq}), USD in millions unless the row says otherwise\n"
        f"# SEC EDGAR facts filed on or before {curr_date}, at the values filed then\n\n"
    )
    rows = [",".join([""] + periods)]
    for label, (values, unit) in lines.items():
        # Every row spans the same columns, or a reader lines the table up wrong.
        if not values:
            rows.append(",".join([label] + ["unavailable (not tagged by this filer)"] * len(periods)))
            continue
        name = label if unit == "USD" else f"{label} ({unit})"
        # Plain numbers: a thousands separator would split the CSV field.
        cells = [
            (f"{values[p] / 1e6:.0f}" if unit == "USD" else f"{values[p]:.2f}")
            if p in values else "" for p in periods
        ]
        rows.append(",".join([name] + cells))
    return header + "\n".join(rows) + "\n"


# --- fundamentals overview -------------------------------------------------------------------

def _latest_by_period(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (start, end): the latest filing's value known at the cutoff."""
    rows = rows.sort_values("available_at", kind="stable")
    return rows.drop_duplicates(["start", "end"], keep="last")


def _recent(column: pd.Series, as_of: date, days: int) -> pd.Series:
    return pd.to_datetime(column) >= pd.Timestamp(as_of) - pd.Timedelta(days=days)


def _latest_instant(known: pd.DataFrame, tags: tuple[str, ...], as_of: date) -> float | None:
    """The most recent balance (instant fact) for the first tag that has a recent one."""
    for tag in tags:
        rows = known[(known["taxonomy"] == "us-gaap") & (known["tag"] == tag) & known["start"].isna()]
        rows = rows[_recent(rows["end"], as_of, _STALE_INSTANT_DAYS)]
        if len(rows):
            rows = _latest_by_period(rows)
            return float(rows.loc[rows["end"] == rows["end"].max(), "val"].iloc[-1])
    return None


def _within(column: pd.Series, day: date, days: int) -> pd.Series:
    """Which dates (yyyy-mm-dd strings) lie within `days` days of `day`."""
    return (pd.to_datetime(column) - pd.Timestamp(day)).abs() <= pd.Timedelta(days=days)


def _ttm(known: pd.DataFrame, tags: tuple[str, ...], as_of: date) -> float | None:
    """Trailing twelve months for a flow (revenue, net income, EPS, cash flow), from filings only.

    TTM = last fiscal year (10-K) + this year to date − the same span a year earlier, both
    from the latest 10-Q after that fiscal year. Right after a 10-K it is that year's figure.
    """
    for tag in tags:
        rows = known[(known["taxonomy"] == "us-gaap") & (known["tag"] == tag) & known["start"].notna()]
        if not len(rows):
            continue
        rows = _latest_by_period(rows)
        spans = (pd.to_datetime(rows["end"]) - pd.to_datetime(rows["start"])).dt.days
        annual = rows[spans.between(*_SPANS["annual"]) & rows["form"].str.startswith(_ANNUAL_FORMS)
                      & _recent(rows["end"], as_of, _STALE_YEAR_DAYS)]
        if not len(annual):
            continue
        fy = annual.loc[annual["end"].idxmax()]
        year_start = date.fromisoformat(fy["end"]) + timedelta(days=1)
        ytd = rows[_within(rows["start"], year_start, 7) & (rows["end"] > fy["end"])]
        if not len(ytd):
            return float(fy["val"])
        cur = ytd.loc[ytd["end"].idxmax()]
        a_year = timedelta(days=365)
        prev = rows[_within(rows["start"], date.fromisoformat(cur["start"]) - a_year, 10)
                    & _within(rows["end"], date.fromisoformat(cur["end"]) - a_year, 10)]
        if not len(prev):
            return None
        return float(fy["val"] + cur["val"] - prev.iloc[-1]["val"])
    return None


def _shares_outstanding(known: pd.DataFrame, as_of: date) -> float | None:
    """Shares outstanding: the cover-page count (dei), else the balance-sheet count.

    Alphabet reports no dei count in its company facts (three share classes), but does report
    the total of all classes as CommonStockSharesOutstanding.
    """
    dei = known[(known["taxonomy"] == "dei") & (known["tag"] == "EntityCommonStockSharesOutstanding")]
    dei = dei[_recent(dei["end"], as_of, _STALE_INSTANT_DAYS)]
    if len(dei):
        latest = dei[dei["end"] == dei["end"].max()]
        latest = latest[latest["available_at"] == latest["available_at"].max()]
        return float(latest.drop_duplicates("val")["val"].sum())  # several classes: their total
    return _latest_instant(known, ("CommonStockSharesOutstanding",), as_of)


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b not in (None, 0) else None


def fundamentals_overview(known: pd.DataFrame, bars: pd.DataFrame, ticker: str, curr_date: str) -> str | None:
    """Upstream's overview fields that can be computed point in time; None if no filings yet.

    `bars` are the price bars known at the cutoff (as traded, data/prices.py).
    """
    as_of = date.fromisoformat(curr_date)

    def ttm(tags):
        return _ttm(known, tags, as_of)

    def balance(tags):
        return _latest_instant(known, tags, as_of)

    revenue = ttm(_STATEMENTS["income_statement"][0][1])
    net_income = ttm(("NetIncomeLoss",))
    if revenue is None and net_income is None:
        return None
    gross = ttm(("GrossProfit",))
    operating = ttm(("OperatingIncomeLoss",))
    eps = ttm(("EarningsPerShareDiluted",))
    ocf = ttm(_STATEMENTS["cashflow"][0][1])
    capex = ttm(_STATEMENTS["cashflow"][3][1])
    equity = balance(_STATEMENTS["balance_sheet"][5][1])
    assets = balance(("Assets",))
    liabilities = balance(("Liabilities",))
    if liabilities is None:  # Amazon tags only "liabilities and equity"; the difference is the same
        total = balance(("LiabilitiesAndStockholdersEquity",))
        liabilities = total - equity if total is not None and equity is not None else None
    current_ratio = _ratio(balance(("AssetsCurrent",)), balance(("LiabilitiesCurrent",)))
    shares = _shares_outstanding(known, as_of)

    close = close_day = None
    high52 = low52 = avg50 = avg200 = dividend_yield = None
    if len(bars):
        close, close_day = float(bars["close"].iloc[-1]), bars["date"].iloc[-1]
        year = bars.tail(252)
        high52, low52 = float(year["high"].max()), float(year["low"].min())
        avg50, avg200 = float(bars["close"].tail(50).mean()), float(bars["close"].tail(200).mean())
        paid = bars.loc[bars["date"] > pd.Timestamp(curr_date) - timedelta(days=365), "dividends"].sum()
        dividend_yield = float(paid) / close
    market_cap = close * shares if close is not None and shares is not None else None

    fields = [
        ("Market Cap", market_cap, "{:.0f}"),
        ("PE Ratio (TTM)", _ratio(close, eps), "{:.2f}"),
        ("Price to Book", _ratio(market_cap, equity), "{:.2f}"),
        ("EPS (TTM)", eps, "{:.2f}"),
        ("Dividend Yield", dividend_yield, "{:.4f}"),
        ("52 Week High", high52, "{:.2f}"),
        ("52 Week Low", low52, "{:.2f}"),
        ("50 Day Average", avg50, "{:.2f}"),
        ("200 Day Average", avg200, "{:.2f}"),
        ("Revenue (TTM)", revenue, "{:.0f}"),
        ("Gross Profit (TTM)", gross, "{:.0f}"),
        ("Operating Income (TTM)", operating, "{:.0f}"),
        ("Net Income (TTM)", net_income, "{:.0f}"),
        ("Profit Margin", _ratio(net_income, revenue), "{:.4f}"),
        ("Operating Margin", _ratio(operating, revenue), "{:.4f}"),
        ("Return on Equity", _ratio(net_income, equity), "{:.4f}"),
        ("Return on Assets", _ratio(net_income, assets), "{:.4f}"),
        ("Total Liabilities to Equity", _ratio(liabilities, equity), "{:.2f}"),
        ("Current Ratio", current_ratio, "{:.2f}"),
        ("Book Value per Share", _ratio(equity, shares), "{:.2f}"),
        ("Free Cash Flow (TTM)", ocf - capex if ocf is not None and capex is not None else None, "{:.0f}"),
        ("Shares Outstanding", shares, "{:.0f}"),
    ]
    price_note = f"the {close_day:%Y-%m-%d} close" if close_day is not None else "no price (none archived)"
    header = (
        f"# Company Fundamentals for {ticker.upper()}\n"
        f"# Computed from SEC EDGAR filings accepted by 16:00 New York on {curr_date} and {price_note}. "
        "TTM = last fiscal year + this year to date - the same span a year earlier. "
        "USD; ratios are fractions (0.25 = 25%).\n\n"
    )
    return header + "\n".join(f"{label}: {fmt.format(v)}" for label, v, fmt in fields if v is not None)


# --- insider transactions --------------------------------------------------------------------

TRANSACTION_CODES = {
    "P": "Purchase", "S": "Sale", "A": "Grant/award", "M": "Option exercise",
    "F": "Tax withholding", "G": "Gift", "C": "Conversion", "D": "Disposition to issuer",
    "X": "Option exercise", "J": "Other",
}


def insider_table(trades: pd.DataFrame, ticker: str, curr_date: str, since: date) -> str:
    """Form 4 trades as CSV, newest first, each with the day it became public."""
    ticker = ticker.upper()
    if not len(trades):
        return f"No insider transactions reported for {ticker} between {since} and {curr_date} (SEC Form 4)"
    header = (
        f"# Insider transactions for {ticker} from SEC Form 4 filings accepted by 16:00 New York on "
        f"{curr_date}, trades since {since}\n"
        "# Public = the day the Form 4 was accepted by SEC; Traded = the trade date\n\n"
    )
    lines = ["Public,Traded,Insider,Position,Transaction,Shares,Price,Value,Owned After,Ownership"]
    for t in trades.itertuples():
        price = "" if pd.isna(t.price) else f"{t.price:.2f}"
        value = "" if pd.isna(t.price) or pd.isna(t.shares) else f"{t.price * t.shares:.0f}"
        shares = "" if pd.isna(t.shares) else f"{t.shares:.0f}"
        owned = "" if pd.isna(t.owned_after) else f"{t.owned_after:.0f}"
        action = f"{t.code} - {TRANSACTION_CODES.get(t.code, 'Other')}"
        position = (t.position or "").replace(",", ";")
        lines.append(",".join([
            f"{pd.Timestamp(t.accepted_at).tz_convert('America/New_York'):%Y-%m-%d}",
            f"{t.transaction_date:%Y-%m-%d}", t.insider.replace(",", " "), position, action,
            shares, price, value, owned, t.ownership or "",
        ]))
    return header + "\n".join(lines) + "\n"
