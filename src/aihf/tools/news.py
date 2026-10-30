"""News text for the News and Sentiment Analysts, from articles already cut at the cutoff.

Same text as upstream's Yahoo news vendor (`dataflows/vendors/yahoo/news.py`, commit 35543d0):

    ## AAPL News, from 2024-01-03 to 2024-01-10:

    ### <title> (source: <publisher>)
    <summary>
    Link: <url>

Company news: the newest `NEWS_ARTICLE_LIMIT` articles. Global news: upstream's five searches
in order, articles in the window, titles shown once, until `GLOBAL_NEWS_ARTICLE_LIMIT`.
Limits and look-back are upstream's defaults (default_config.py).
"""

import pandas as pd

NEWS_ARTICLE_LIMIT = 20  # news_article_limit
GLOBAL_NEWS_ARTICLE_LIMIT = 10  # global_news_article_limit
GLOBAL_NEWS_LOOKBACK_DAYS = 7  # global_news_lookback_days


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


def global_news(per_query: list[pd.DataFrame], start_date: str, curr_date: str, limit: int) -> str:
    """`per_query`: each search's articles in the window, newest first, in upstream's order."""
    seen, picked = set(), []
    for articles in per_query:
        for a in articles.itertuples():
            if a.title and a.title not in seen:
                seen.add(a.title)
                picked.append(a)
        if len(picked) >= limit:
            break
    if not picked:
        return f"No global news found between {start_date} and {curr_date}"
    body = "".join(_article(a) for a in picked[:limit])
    return f"## Global Market News, from {start_date} to {curr_date}:\n\n{body}"
