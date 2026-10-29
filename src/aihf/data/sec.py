"""The SEC archive: what the files look like, and reading only what was public at a cutoff.

One folder per ticker, written by the SEC collector (data/collect_sec.py):

    <root>/sec/<TICKER>/companyfacts.json   every number the company reported in XBRL, each with
                                            the accession number of the filing it came from
    <root>/sec/<TICKER>/filings.parquet     every filing: accession, form, dates, accepted_at
    <root>/sec/<TICKER>/form4.parquet       insider trades from Form 4 filings, one row each
    <root>/sec/<TICKER>/meta.json           source, download time, counts, sha256 of each file

Point in time: a filing is public from the moment SEC accepted it (`accepted_at`, UTC). A fact
or a trade is visible to a decision only if its filing was accepted by the decision cutoff
(16:00 New York on the trade date). Upstream uses the filing *date*, which also lets in
filings accepted after the close on that day (DEVIATIONS D11).
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from aihf.data.pit import NEW_YORK, UTC

FILING_COLUMNS = ["accession", "form", "filing_date", "report_date", "accepted_at", "primary_document"]
FORM4_COLUMNS = [
    "accession", "accepted_at",  # the filing, and when it became public (UTC)
    "insider", "position",  # who traded, e.g. "KONDO CHRIS", "Principal Accounting Officer"
    "transaction_date", "code",  # when they traded, and the SEC transaction code (S, P, M, ...)
    "shares", "price",  # shares traded and price per share (NaN when not reported, e.g. gifts)
    "acquired_disposed",  # "A" acquired or "D" disposed
    "owned_after",  # shares held after the trade
    "ownership",  # "D" direct or "I" indirect
]


def sec_dir(root: Path, ticker: str) -> Path:
    return Path(root) / "sec" / ticker.upper()


@dataclass(frozen=True)
class SecMeta:
    """What the collector wrote into meta.json that the readers need."""

    fetched_at: datetime  # when companyfacts/filings were downloaded (UTC)
    form4_start: date  # Form 4 filings are archived for this filing-date range only
    form4_end: date


def read_meta(root: Path, ticker: str) -> SecMeta | None:
    """None if the ticker has no (complete) SEC folder."""
    folder = sec_dir(root, ticker)
    names = ("meta.json", "companyfacts.json", "filings.parquet", "form4.parquet")
    if not all((folder / n).exists() for n in names):
        return None
    info = json.loads((folder / "meta.json").read_text())
    return SecMeta(datetime.fromisoformat(info["fetched_at"]),
                   date.fromisoformat(info["form4_start"]), date.fromisoformat(info["form4_end"]))


def end_of_filing_day(filing_date: str) -> datetime:
    """For a fact whose filing is missing from filings.parquet: 23:59 New York on its filing
    date, in UTC. Later than any cutoff on that day, so it is only seen from the next day on."""
    return datetime.combine(date.fromisoformat(filing_date), time(23, 59), tzinfo=NEW_YORK).astimezone(UTC)


@lru_cache(maxsize=16)
def load_facts(root: Path, ticker: str) -> pd.DataFrame:
    """All us-gaap and dei facts as rows, each with the moment its filing became public.

    Columns: taxonomy, tag, unit, start (None for balances), end, val, accn, form, filed,
    available_at (UTC). Cached: the agents ask several times per cell.
    """
    folder = sec_dir(root, ticker)
    facts = json.loads((folder / "companyfacts.json").read_text())["facts"]
    filings = pd.read_parquet(folder / "filings.parquet", columns=["accession", "accepted_at"])
    accepted = dict(zip(filings["accession"], filings["accepted_at"], strict=True))

    rows = []
    for taxonomy in ("us-gaap", "dei"):
        for tag, body in (facts.get(taxonomy) or {}).items():
            for unit, values in body.get("units", {}).items():
                for f in values:
                    rows.append((taxonomy, tag, unit, f.get("start"), f["end"], f["val"], f["accn"],
                                 f.get("form", ""), f["filed"]))
    df = pd.DataFrame(rows, columns=["taxonomy", "tag", "unit", "start", "end", "val", "accn", "form", "filed"])
    known = df["accn"].map(accepted)
    fallback = df.loc[known.isna(), "filed"].map(end_of_filing_day)
    df["available_at"] = pd.to_datetime(known.fillna(fallback), utc=True)
    return df


def facts_known_at(root: Path, ticker: str, cutoff: datetime) -> pd.DataFrame:
    """The facts whose filing was public at `cutoff`."""
    df = load_facts(Path(root), ticker.upper())
    return df[df["available_at"] <= pd.Timestamp(cutoff)]


def form4_known_at(root: Path, ticker: str, cutoff: datetime, since: date) -> pd.DataFrame:
    """Insider trades public at `cutoff` and traded on or after `since`, newest first."""
    df = pd.read_parquet(sec_dir(root, ticker) / "form4.parquet", columns=FORM4_COLUMNS)
    df = df[(df["accepted_at"] <= pd.Timestamp(cutoff)) & (df["transaction_date"] >= pd.Timestamp(since))]
    return df.sort_values(["transaction_date", "accepted_at"], ascending=False).reset_index(drop=True)


def days_before(day: str, days: int) -> date:
    return date.fromisoformat(day) - timedelta(days=days)
