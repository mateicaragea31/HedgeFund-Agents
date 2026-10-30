"""News text for the News and Sentiment Analysts, from articles already cut at the cutoff.

Same text as upstream's Yahoo news vendor (`dataflows/vendors/yahoo/news.py`, commit 35543d0):

    ## AAPL News, from 2024-01-03 to 2024-01-10:

    ### <title> (source: <publisher>)
    <summary>
    Link: <url>

Company news: the newest `NEWS_ARTICLE_LIMIT` articles. Global news: upstream's five searches
in order, articles in the window, titles shown once, until `GLOBAL_NEWS_ARTICLE_LIMIT`.
Limits and look-back are upstream's defaults (default_config.py).

Global news ranking (ours, DEVIATIONS D14): the Guardian's search returns every article with
*any* word of the query (6,000+ a month for "ECB Bank of England BOJ central bank policy"), so
we rank them ourselves: score = how many different query words appear in the headline or
summary; at least MIN_QUERY_WORDS; best score first, then newest. Yahoo's search, which
upstream uses, also ranks by relevance; this is a simple, transparent version of that.
"""

import re

import pandas as pd

NEWS_ARTICLE_LIMIT = 20  # news_article_limit
GLOBAL_NEWS_ARTICLE_LIMIT = 10  # global_news_article_limit
GLOBAL_NEWS_LOOKBACK_DAYS = 7  # global_news_lookback_days
MIN_QUERY_WORDS = 2
_STOPWORDS = {"of", "the", "and", "a", "in"}


def query_words(query: str) -> list[str]:
    """ "ECB Bank of England BOJ central bank policy" → ["ecb", "bank", "england", "boj", "central", "policy"]."""
    words = [w for w in query.lower().split() if w not in _STOPWORDS]
    return list(dict.fromkeys(words))  # unique, in order


def relevance(query: str, articles: pd.DataFrame) -> pd.Series:
    """How many different query words each article's headline + summary contains (whole words)."""
    text = (articles["title"].fillna("") + " " + articles["summary"].fillna("")).str.lower()
    score = pd.Series(0, index=articles.index)
    for word in query_words(query):
        pattern = r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])"
        score += text.str.contains(pattern, regex=True).astype(int)
    return score


def rank_for_query(query: str, articles: pd.DataFrame) -> pd.DataFrame:
    """The query's articles with at least MIN_QUERY_WORDS words, best score first, then newest."""
    if not len(articles):
        return articles
    ranked = articles.assign(_score=relevance(query, articles))
    ranked = ranked[ranked["_score"] >= MIN_QUERY_WORDS]
    return ranked.sort_values(["_score", "published_at"], ascending=[False, False], kind="stable")


def _article(a) -> str:
    text = f"### {a.title} (source: {a.source})\n"
    if a.summary:
        text += f"{a.summary}\n"
    if a.link:
        text += f"Link: {a.link}\n"
    return text + "\n"


def company_news(articles: pd.DataFrame, ticker: str, start_date: str, end_date: str) -> str:
    """`articles`: the ticker's articles in the window, newest first."""
    articles = articles.drop_duplicates("title").head(NEWS_ARTICLE_LIMIT)
    if not len(articles):
        return f"No news found for {ticker} between {start_date} and {end_date}"
    body = "".join(_article(a) for a in articles.itertuples())
    return f"## {ticker} News, from {start_date} to {end_date}:\n\n{body}"


def global_news(per_query: list[tuple[str, pd.DataFrame]], start_date: str, curr_date: str, limit: int) -> str:
    """`per_query`: (search, its articles in the window), in upstream's search order."""
    seen, picked = set(), []
    for query, articles in per_query:
        for a in rank_for_query(query, articles).itertuples():
            if a.title and a.title not in seen:
                seen.add(a.title)
                picked.append(a)
        if len(picked) >= limit:
            break
    if not picked:
        return f"No global news found between {start_date} and {curr_date}"
    body = "".join(_article(a) for a in picked[:limit])
    return f"## Global Market News, from {start_date} to {curr_date}:\n\n{body}"
