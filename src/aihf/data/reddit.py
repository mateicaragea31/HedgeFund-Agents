"""The Reddit archive: what the files look like, and reading the posts made in a time window.

One file per subreddit and month, written by the Reddit collector (data/collect_reddit.py):

    <root>/reddit/<subreddit>/<YYYY-MM>.parquet   every post of that month, columns POST_COLUMNS
    <root>/reddit/<subreddit>/meta.json           source, fetch time, the months completed

All posts are kept, not only those naming our tickers, so the matching rule can change without
downloading again. Point in time: a post is public from `created_at` (UTC).
"""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

SUBREDDITS = ("wallstreetbets", "stocks", "investing")  # upstream's DEFAULT_SUBREDDITS

POST_COLUMNS = [
    "id", "subreddit",
    "created_at",  # when the post was made (UTC): the moment it became public
    "title", "selftext",  # as the archive first captured them (usually within a day)
    "score", "num_comments",  # counted when the archive captured the post, not on the trade date: never shown
    "retrieved_on",  # when the archive captured it (UTC)
]


def sub_dir(root: Path, subreddit: str) -> Path:
    return Path(root) / "reddit" / subreddit.lower()


def month_file(root: Path, subreddit: str, month: str) -> Path:
    return sub_dir(root, subreddit) / f"{month}.parquet"


def months_between(start: date, end: date) -> list[str]:
    """["2023-12", "2024-01", ...] for every month touching [start, end]."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def completed_months(root: Path, subreddit: str) -> set[str]:
    meta = sub_dir(root, subreddit) / "meta.json"
    return set(json.loads(meta.read_text())["months"]) if meta.exists() else set()


def missing_months(root: Path, subreddits, start: datetime, end: datetime) -> list[str]:
    """"r/sub YYYY-MM" for each month of the window not (completely) in the archive."""
    wanted = months_between(start.date(), end.date())
    return [f"r/{s} {m}" for s in subreddits for m in wanted if m not in completed_months(root, s)]


def load_posts(root: Path, subreddits, start: datetime, end: datetime) -> pd.DataFrame:
    """Posts made in [start, end] (UTC datetimes) in these subreddits, newest first."""
    frames = []
    for sub in subreddits:
        for month in months_between(start.date(), end.date()):
            path = month_file(root, sub, month)
            if path.exists():
                frames.append(pd.read_parquet(path, columns=POST_COLUMNS))
    if not frames:
        return pd.DataFrame(columns=POST_COLUMNS)
    posts = pd.concat(frames, ignore_index=True)
    posts = posts[(posts["created_at"] >= pd.Timestamp(start)) & (posts["created_at"] <= pd.Timestamp(end))]
    return posts.sort_values("created_at", ascending=False).reset_index(drop=True)
