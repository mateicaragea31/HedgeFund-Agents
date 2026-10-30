"""The news archive: what the files look like, and reading the articles of a time window.

Two feeds, one file per feed key and month, written by the news collector (data/collect_news.py):

    <root>/news/eodhd/<TICKER>/<YYYY-MM>.parquet      company news from EODHD (articles tagged with
                                                      the ticker)
    <root>/news/guardian/<query-slug>/<YYYY-MM>.parquet   global news: the Guardian's results for one
                                                      of upstream's five macro searches
    <root>/news/<feed>/<key>/meta.json                months completed, counts, fetch time

We keep the headline, a short summary (at most SUMMARY_CHARS) and the link, never the full
article. Point in time: an article is public from `published_at` (UTC).
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from aihf.data.reddit import months_between

NEWS_COLUMNS = [
    "published_at",  # UTC: when the article was published, the moment it became public
    "title",
    "summary",  # the feed's summary, or the start of the article, at most SUMMARY_CHARS
    "link",
    "source",  # the website, e.g. "finance.yahoo.com" (EODHD has no publisher field)
    "symbols",  # tickers the feed tagged, comma-separated ("" for the Guardian)
]
SUMMARY_CHARS = 500

# Upstream's global news searches (default_config.py `global_news_queries`, commit 35543d0).
GLOBAL_NEWS_QUERIES = (
    "Federal Reserve interest rates inflation",
    "S&P 500 earnings GDP economic outlook",
    "geopolitical risk trade war sanctions",
    "ECB Bank of England BOJ central bank policy",
    "oil commodities supply chain energy",
)


def slug(text: str) -> str:
    """A folder name for a search: "S&P 500 earnings ..." → "s-p-500-earnings-..."."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def feed_dir(root: Path, feed: str, key: str) -> Path:
    return Path(root) / "news" / feed / key


def month_file(root: Path, feed: str, key: str, month: str) -> Path:
    return feed_dir(root, feed, key) / f"{month}.parquet"


def completed_months(root: Path, feed: str, key: str) -> set[str]:
    meta = feed_dir(root, feed, key) / "meta.json"
    return set(json.loads(meta.read_text())["months"]) if meta.exists() else set()


def missing_months(root: Path, feed: str, keys, start: datetime, end: datetime) -> list[str]:
    wanted = months_between(start.date(), end.date())
    return [f"{feed}/{k} {m}" for k in keys for m in wanted if m not in completed_months(root, feed, k)]


def load_articles(root: Path, feed: str, key: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Articles of one feed key published in [start, end] (UTC datetimes), newest first."""
    frames = [pd.read_parquet(p, columns=NEWS_COLUMNS)
              for m in months_between(start.date(), end.date())
              if (p := month_file(root, feed, key, m)).exists()]
    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["published_at"] >= pd.Timestamp(start)) & (df["published_at"] <= pd.Timestamp(end))]
    return df.sort_values("published_at", ascending=False, kind="stable").reset_index(drop=True)

