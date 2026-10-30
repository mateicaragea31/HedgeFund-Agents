"""The macro archive: FRED series with every published version ("vintage"), read as of a date.

One file per series, written by the macro collector (data/collect_macro.py):

    <root>/macro/fred/<SERIES>.parquet     one row per (observation date, vintage): date, value,
                                           realtime_start, realtime_end
    <root>/macro/fred/<SERIES>.meta.json   title, units, frequency, seasonal adjustment, and the
                                           observation and vintage ranges collected

FRED keeps each value with the period it was the published one: CPI for 2023-12 first read
306.746 from 2024-01-11 (its release day) and changed when revised. A run on a trade date sees
the version valid that day (realtime_start <= trade date <= realtime_end), as upstream's live
calls do with realtime_start = realtime_end = trade date.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

OBS_COLUMNS = ["date", "value", "realtime_start", "realtime_end"]  # all strings, as FRED sends them


def series_file(root: Path, series_id: str) -> Path:
    return Path(root) / "macro" / "fred" / f"{series_id.upper()}.parquet"


@dataclass(frozen=True)
class SeriesInfo:
    title: str
    units: str
    frequency: str
    seasonal: str
    vintages_from: date  # the collected vintage range: trade dates inside it can be served
    vintages_to: date


def read_info(root: Path, series_id: str) -> SeriesInfo | None:
    meta = series_file(root, series_id).with_suffix(".meta.json")
    if not meta.exists() or not series_file(root, series_id).exists():
        return None
    m = json.loads(meta.read_text())
    return SeriesInfo(m["title"], m["units"], m["frequency"], m["seasonal"],
                      date.fromisoformat(m["realtime_start"]), date.fromisoformat(m["realtime_end"]))


def observations_as_of(root: Path, series_id: str, as_of: str, start: str, end: str) -> list[tuple[str, str]]:
    """(date, value) for observation dates in [start, end], at the vintage valid on `as_of`.

    FRED's "." (no value) is left out, as upstream does.
    """
    df = pd.read_parquet(series_file(root, series_id), columns=OBS_COLUMNS)
    df = df[(df["realtime_start"] <= as_of) & (df["realtime_end"] >= as_of)
            & (df["date"] >= start) & (df["date"] <= end) & ~df["value"].isin([".", ""])]
    return list(df.sort_values("date")[["date", "value"]].itertuples(index=False, name=None))
