"""Point-in-time rules (CLAUDE.md §3 rule 2): when a decision is made, and when data became known.

All times are stored in UTC. New York's clock changes twice a year (EST = UTC-5 in winter,
EDT = UTC-4 in summer), so "16:00 New York" is 21:00 UTC on 2024-03-08 but 20:00 UTC on
2024-03-11. zoneinfo knows those rules; we never add a fixed offset by hand.
"""

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
MARKET_CLOSE = time(16, 0)


def _at_close(day: date | str) -> datetime:
    if isinstance(day, str):
        day = date.fromisoformat(day)
    return datetime.combine(day, MARKET_CLOSE, tzinfo=NEW_YORK).astimezone(UTC)


def decision_cutoff(trade_date: date | str) -> datetime:
    """The moment a decision for `trade_date` is made: 16:00 New York that day, in UTC.

    Agents may only see records with available_at <= this moment.
    """
    return _at_close(trade_date)


def bar_available_at(day: date | str) -> datetime:
    """When a daily price bar became known: at the close, 16:00 New York on its day, in UTC."""
    return _at_close(day)
