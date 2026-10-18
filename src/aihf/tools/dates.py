"""Keep the model's requested dates from reaching past the trade date.

Same rules as upstream's `dataflows/date_window.py` (`as_of`, `as_of_window`, commit 35543d0):
a model may ask for today's date, or omit the date; the tool then serves the trade date instead.
"""

from datetime import datetime, timedelta


def _parse(day: str | None) -> datetime | None:
    try:
        return datetime.strptime(day, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def as_of(requested: str | None, trade_date: str) -> str | None:
    """The date a tool serves: the model's date, but never later than the trade date.

    as_of("2024-01-05", "2024-01-10") -> "2024-01-05"
    as_of("2026-10-18", "2024-01-10") -> "2024-01-10"   (too late: clamped)
    as_of(None, "2024-01-10")         -> "2024-01-10"   (missing or unreadable: trade date)
    An empty trade_date (a tool called outside a graph run) passes the request through.
    """
    if not trade_date:
        return requested
    parsed = _parse(requested)
    return requested if parsed is not None and parsed <= _parse(trade_date) else trade_date


def as_of_window(start_date: str, end_date: str, trade_date: str) -> tuple[str, str]:
    """[start, end] with the end clamped to the trade date.

    A window lying wholly after the trade date keeps its length and moves back to end there:
    as_of_window("2024-06-01", "2024-06-08", "2024-01-10") -> ("2024-01-03", "2024-01-10")
    """
    end = as_of(end_date, trade_date)
    start, old_end = _parse(start_date), _parse(end_date)
    if end == end_date or start is None or start <= _parse(end):
        return start_date, end
    span = (old_end - start) if old_end is not None and old_end >= start else timedelta(0)
    return f"{_parse(end) - span:%Y-%m-%d}", end


def days_back(day: str, days: int) -> str:
    """The date `days` calendar days before `day`: days_back("2024-01-10", 7) -> "2024-01-03"."""
    return f"{_parse(day) - timedelta(days=days):%Y-%m-%d}"
