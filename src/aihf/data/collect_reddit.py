"""Download every post of some subreddits, month by month, from the Arctic Shift Reddit archive.

Runs on the FEP login node, never inside a backtest. Arctic Shift (arctic-shift.photon-reddit.com)
is a free, keyless archive of Reddit. Its text search does not work on very busy subreddits
such as r/wallstreetbets, so we download all posts of each month and match tickers ourselves
(tools/reddit.py). A month is written only when it is complete, so an interrupted run resumes
at the first missing month.
"""

import json
import logging
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

import certifi
import pandas as pd

from aihf.data.reddit import POST_COLUMNS, completed_months, month_file, months_between, sub_dir

logger = logging.getLogger(__name__)

SEARCH_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
FIELDS = "id,subreddit,created_utc,title,selftext,score,num_comments,retrieved_on"
PAGE = 100  # the API's maximum page size
REQUEST_GAP_SECONDS = 1.0  # "a couple of requests per second" is fine; we stay below that
_TLS = ssl.create_default_context(cafile=certifi.where())

# Fetches one page of posts (dict of query parameters → list of post dicts). Faked in tests.
Getter = Callable[[dict], list[dict]]


class RedditDataError(Exception):
    """A month could not be downloaded completely; nothing was saved for it."""


def arctic_shift_get(params: dict) -> list[dict]:
    """One page of posts, with backoff when the server asks us to slow down."""
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "aihf-research (dissertation data collector)"})
    problem = None
    for attempt in range(6):
        time.sleep(REQUEST_GAP_SECONDS)
        try:
            with urllib.request.urlopen(request, timeout=120, context=_TLS) as response:
                body = json.load(response)
            if body.get("data") is not None:
                return body["data"]
            problem = body.get("error")  # e.g. "Timeout. Maybe slow down a bit"
        except urllib.error.HTTPError as exc:
            problem = f"HTTP {exc.code}"
            reset = exc.headers.get("X-RateLimit-Reset") if exc.headers else None
            if reset and reset.isdigit():
                time.sleep(int(reset))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            problem = exc
        time.sleep(10 * (attempt + 1))
    raise RedditDataError(f"Arctic Shift kept failing ({problem}) for {params}")


def fetch_month(subreddit: str, month: str, get: Getter) -> pd.DataFrame:
    """Every post of r/<subreddit> made in `month` ("2024-01"), oldest first, without duplicates.

    Pages are requested oldest first. The next page starts one second before the last post seen,
    because several posts can share a second; posts already seen are dropped. If a page brings
    nothing new (it was all one second), the next one starts after that second, so every request
    moves forward. Only posts beyond the first 100 within one single second could be missed.
    """
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    after, end_ts = int(start.timestamp()) - 1, int(end.timestamp())
    seen: dict[str, dict] = {}
    while True:
        page = get({"subreddit": subreddit, "after": after, "before": end_ts, "limit": PAGE,
                    "sort": "asc", "fields": FIELDS})
        new = [p for p in page if p["id"] not in seen]
        seen.update((p["id"], p) for p in new)
        if len(page) < PAGE:
            break
        last = max(p["created_utc"] for p in page)
        after = last - 1 if new and last - 1 > after else last
    posts = pd.DataFrame(list(seen.values()), columns=["id", "subreddit", "created_utc", "title", "selftext",
                                                       "score", "num_comments", "retrieved_on"])
    posts["created_at"] = pd.to_datetime(posts["created_utc"], unit="s", utc=True)
    posts["retrieved_on"] = pd.to_datetime(posts["retrieved_on"], unit="s", utc=True)
    posts = posts[(posts["created_at"] >= start) & (posts["created_at"] < end)]
    posts["subreddit"] = subreddit.lower()
    posts["selftext"] = posts["selftext"].fillna("")
    return posts[POST_COLUMNS].sort_values("created_at").reset_index(drop=True)


def collect(subreddits, start: str, end: str, root, *, force: bool = False, dry_run: bool = False,
            get: Getter = arctic_shift_get) -> None:
    """Download every month touching [start, end] for each subreddit; skips completed months."""
    for sub in (s.lower() for s in subreddits):
        done = set() if force else completed_months(root, sub)
        todo = [m for m in months_between(pd.Timestamp(start).date(), pd.Timestamp(end).date()) if m not in done]
        if not todo:
            logger.info("r/%s: all months already collected", sub)
            continue
        if dry_run:
            logger.info("r/%s: would download %s", sub, ", ".join(todo))
            continue
        folder = sub_dir(root, sub)
        folder.mkdir(parents=True, exist_ok=True)
        for month in todo:
            posts = fetch_month(sub, month, get)
            path = month_file(root, sub, month)
            tmp = path.with_suffix(".parquet.tmp")
            posts.to_parquet(tmp, index=False)
            tmp.replace(path)
            done.add(month)
            meta = {"source": "Arctic Shift (arctic-shift.photon-reddit.com), posts search API",
                    "subreddit": sub, "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "months": sorted(done),
                    "counts": {**_counts(folder), month: len(posts)}}
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))
            logger.info("r/%s %s: saved %d posts", sub, month, len(posts))


def _counts(folder) -> dict:
    meta = folder / "meta.json"
    return json.loads(meta.read_text()).get("counts", {}) if meta.exists() else {}
