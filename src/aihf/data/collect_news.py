"""Download news once into the archive: company news from EODHD, global news from the Guardian.

Runs on the FEP login node, never inside a backtest. Both feeds are free with a key
(EODHD_API_KEY, GUARDIAN_API_KEY in ~/.bashrc). Months are written only when complete, so a run
that hits a daily limit stops cleanly and the next run resumes at the first missing month.

- EODHD: every article tagged with the ticker, one calendar month per window (their forward
  paging, direction=1, allows at most 31 days), up to 1000 articles per request. Each request
  costs 5 of the free plan's 20 daily calls (plus a one-time bonus of 500).
- Guardian: for each of upstream's five global-news searches, every matching article of the
  month, 200 per page, at most 1 request per second (free key: 500 per day).
"""

import html
import json
import logging
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

import certifi
import pandas as pd

from aihf.data.news import (
    GLOBAL_NEWS_QUERIES,
    NEWS_COLUMNS,
    SUMMARY_CHARS,
    completed_months,
    feed_dir,
    month_file,
    slug,
)
from aihf.data.reddit import months_between

logger = logging.getLogger(__name__)

EODHD_URL = "https://eodhd.com/api/news"
GUARDIAN_URL = "https://content.guardianapis.com/search"
EODHD_PAGE = 1000
GUARDIAN_PAGE = 200
REQUEST_GAP_SECONDS = 1.1
_TLS = ssl.create_default_context(cafile=certifi.where())

# Fetches a URL (with its query parameters) and returns the decoded JSON. Faked in tests.
Getter = Callable[[str, dict], object]


class NewsDataError(Exception):
    """A request failed for good; nothing was saved for that month."""


class LimitReached(NewsDataError):
    """The feed's daily limit is used up: run again tomorrow, it resumes where it stopped."""


def http_get_json(url: str, params: dict) -> object:
    request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}",
                                     headers={"User-Agent": "aihf-research (dissertation data collector)"})
    problem = None
    for attempt in range(4):
        time.sleep(REQUEST_GAP_SECONDS)
        try:
            with urllib.request.urlopen(request, timeout=120, context=_TLS) as response:
                body = response.read().decode("utf-8")
            if "Rate Limit Exceeded" in body or "limit exceeded" in body.lower():
                raise LimitReached(f"daily limit reached at {url}")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            if exc.code in (402, 429):
                raise LimitReached(f"daily limit reached at {url} (HTTP {exc.code})") from exc
            if exc.code in (401, 403):
                raise NewsDataError(f"{url} refused the key (HTTP {exc.code})") from exc
            problem = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            problem = exc
        time.sleep(10 * (attempt + 1))
    raise NewsDataError(f"{url} kept failing: {problem}")


def _key(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise NewsDataError(f"set {name} in ~/.bashrc")
    return value


def _summary(text: str) -> str:
    """Plain text, one line, at most SUMMARY_CHARS (cut at a word)."""
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text or "")).split())
    if len(text) <= SUMMARY_CHARS:
        return text
    return text[:SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"


def _site(link: str) -> str:
    host = urllib.parse.urlparse(link or "").netloc.lower()
    return host.removeprefix("www.") or "unknown"


def _month_bounds(month: str) -> tuple[str, str]:
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthEnd(0)
    return f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"


def fetch_eodhd_month(ticker: str, month: str, get: Getter, token: str) -> pd.DataFrame:
    """Every EODHD article tagged with <ticker>.US published in `month`."""
    start, end = _month_bounds(month)
    rows, offset = [], 0
    while True:
        page = get(EODHD_URL, {"s": f"{ticker.upper()}.US", "from": start, "to": end, "limit": EODHD_PAGE,
                               "offset": offset, "direction": 1, "fmt": "json", "api_token": token})
        if not isinstance(page, list):
            raise NewsDataError(f"EODHD answered {str(page)[:200]}")
        rows += [{"published_at": a.get("date"), "title": (a.get("title") or "").strip(),
                  "summary": _summary(a.get("content")), "link": a.get("link") or "",
                  "source": _site(a.get("link")), "symbols": ",".join(a.get("symbols") or [])}
                 for a in page]
        if len(page) < EODHD_PAGE:
            break
        offset += EODHD_PAGE
    return _frame(rows, month)


def fetch_guardian_month(query: str, month: str, get: Getter, key: str) -> pd.DataFrame:
    """Every Guardian article matching `query` (the Guardian's search) published in `month`."""
    start, end = _month_bounds(month)
    rows, page = [], 1
    while True:
        body = get(GUARDIAN_URL, {"q": query, "from-date": start, "to-date": end, "page": page,
                                  "page-size": GUARDIAN_PAGE, "order-by": "oldest",
                                  "show-fields": "trailText", "api-key": key})
        response = body.get("response", {}) if isinstance(body, dict) else {}
        if response.get("status") != "ok":
            raise NewsDataError(f"Guardian answered {str(body)[:200]}")
        rows += [{"published_at": r["webPublicationDate"], "title": r.get("webTitle", "").strip(),
                  "summary": _summary((r.get("fields") or {}).get("trailText")), "link": r.get("webUrl", ""),
                  "source": "theguardian.com", "symbols": ""}
                 for r in response.get("results", [])]
        if page >= response.get("pages", 0):
            break
        page += 1
    return _frame(rows, month)


def _frame(rows: list[dict], month: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    first = pd.Timestamp(f"{month}-01", tz="UTC")
    df = df[(df["published_at"] >= first) & (df["published_at"] < first + pd.offsets.MonthBegin(1))]
    df = df.drop_duplicates(["published_at", "title", "link"])
    return df.sort_values("published_at").reset_index(drop=True)


def _save(root, feed: str, key: str, month: str, df: pd.DataFrame, source: str) -> None:
    folder = feed_dir(root, feed, key)
    folder.mkdir(parents=True, exist_ok=True)
    path = month_file(root, feed, key, month)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"months": [], "counts": {}}
    meta.update(source=source, key=key, updated_at=datetime.now(UTC).isoformat(timespec="seconds"))
    meta["months"] = sorted(set(meta["months"]) | {month})
    meta["counts"][month] = len(df)
    meta_path.write_text(json.dumps(meta, indent=2))


def collect(tickers, start: str, end: str, root, *, queries=GLOBAL_NEWS_QUERIES, force: bool = False,
            dry_run: bool = False, get: Getter = http_get_json) -> None:
    """Company news for each ticker, then global news for each query, month by month.

    Stops at the first daily limit (LimitReached is re-raised after logging what is done).
    """
    months = months_between(pd.Timestamp(start).date(), pd.Timestamp(end).date())
    jobs = [("eodhd", t.upper(), t) for t in tickers] + [("guardian", slug(q), q) for q in queries]
    keys = {}
    for feed, key, what in jobs:
        done = set() if force else completed_months(root, feed, key)
        todo = [m for m in months if m not in done]
        if not todo:
            logger.info("%s/%s: all months already collected", feed, key)
            continue
        if dry_run:
            logger.info("%s/%s: would download %s", feed, key, ", ".join(todo))
            continue
        env = "EODHD_API_KEY" if feed == "eodhd" else "GUARDIAN_API_KEY"
        keys.setdefault(env, _key(env))
        for month in todo:
            if feed == "eodhd":
                df = fetch_eodhd_month(what, month, get, keys[env])
                source = "EODHD financial news API (eodhd.com/api/news), articles tagged with the ticker"
            else:
                df = fetch_guardian_month(what, month, get, keys[env])
                source = f"The Guardian Open Platform search, q={what!r}"
            _save(root, feed, key, month, df, source)
            logger.info("%s/%s %s: saved %d articles", feed, key, month, len(df))
