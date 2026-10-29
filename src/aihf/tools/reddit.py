"""The Reddit block of the Sentiment Analyst's prompt, from posts already cut at the cutoff.

Same text as upstream's `fetch_reddit_posts` (`dataflows/vendors/reddit.py`, commit 35543d0):
per subreddit, up to 5 newest posts mentioning the ticker, each "[date] title" with a
240-character body excerpt. Two differences (DEVIATIONS D13): posts come from our archive
instead of Reddit's live search (which only covers the last week), and a post matches on the
ticker or the company's name, not on Reddit's search for the ticker.
"""

import html
import re

import pandas as pd

LIMIT_PER_SUB = 5  # upstream's limit_per_sub
_EXCERPT_CHARS = 240


def mention_pattern(ticker: str, aliases: list[str]) -> re.Pattern:
    """Whole-word, case-sensitive: "AAPL", "$AAPL", "Apple", "Apple's" match; "apple pie",
    "Applebee's" and "SNAAPL" don't."""
    names = [re.escape(n) for n in [ticker.upper(), *aliases]]
    return re.compile(r"(?<![A-Za-z0-9])\$?(?:" + "|".join(names) + r")(?![A-Za-z0-9])")


def _clean(text: str) -> str:
    """Plain text on one line: decode HTML entities ("&amp;", "&#x200B;"), drop zero-width
    spaces, collapse whitespace. Upstream's feed text gets the same treatment (html.unescape)."""
    return " ".join(html.unescape(text or "").replace("\u200b", " ").split())


def visible(posts: pd.DataFrame) -> pd.DataFrame:
    """Drop posts the moderators removed: removed posts were hidden from readers on Reddit."""
    return posts[posts["selftext"] != "[removed]"]


def reddit_block(posts: pd.DataFrame, ticker: str, aliases: list[str], subreddits) -> str:
    """Upstream's text for the posts mentioning the ticker; `posts` are already cut to the window."""
    ticker = ticker.upper()
    pattern = mention_pattern(ticker, aliases)
    posts = visible(posts)
    text = posts["title"].fillna("") + "\n" + posts["selftext"].fillna("")
    posts = posts[text.map(lambda t: bool(pattern.search(t)))]

    blocks = []
    for sub in subreddits:
        sub_posts = posts[posts["subreddit"] == sub.lower()].head(LIMIT_PER_SUB)  # newest first
        if not len(sub_posts):
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker}>")
            continue
        lines = [f"r/{sub} — {len(sub_posts)} recent posts mentioning {ticker}:"]
        for p in sub_posts.itertuples():
            title = _clean(p.title)
            body = "" if p.selftext in ("[deleted]", "[removed]") else _clean(p.selftext)
            if len(body) > _EXCERPT_CHARS:
                body = body[:_EXCERPT_CHARS] + "…"
            lines.append(f"  [{p.created_at:%Y-%m-%d}] {title}" + (f"\n    body excerpt: {body}" if body else ""))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
