"""Reading the 5-tier rating (Buy / Overweight / Hold / Underweight / Sell) out of a decision.

Copied from upstream `agents/rating.py` (commit 35543d0, Apache-2.0, see
LICENSES/TradingAgents-LICENSE).

When no rating can be read, the answer is REVIEW, never Hold: a decision nobody can read is
not a decision to hold. The backtest counts REVIEWs and reports them (CLAUDE.md §6.3).
"""

import re
import unicodedata

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

# The signal for a decision without a recognizable rating. Not a tradeable position.
RATING_REVIEW = "REVIEW"

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating — **X**" — tolerates markdown
# bold wrappers and any dash or colon a model writes as the separator.
_RATING_LABEL_RE = re.compile(r"rating\b[^:\-‐-―]*[:\-‐-―][\s*]*(\w+)",
                              re.IGNORECASE)

# A line presenting the scale rather than a decision ("Rating Scale: Buy, ...").
_RATING_SCALE_RE = re.compile(r"rating\s*(scale|options|legend)", re.IGNORECASE)

# Ours (DEVIATIONS D16): "Rating" written as a heading, with the rating alone on the next line —
# "## Rating\nHold", "**Rating**\n**Hold**". Upstream's label pattern needs both on one line.
_RATING_HEADING_RE = re.compile(r"^[#*\s]*rating[*:\s]*$", re.IGNORECASE)
_RATING_ALONE_RE = re.compile(r"^[\s*_\-]*(" + "|".join(RATINGS_5_TIER) + r")[\s*_.]*$", re.IGNORECASE)

# Standalone 5-tier word anywhere (word boundaries so "Buyer"/"Holding" don't match).
_RATING_WORD_RE = re.compile(
    r"\b(" + "|".join(RATINGS_5_TIER) + r")\b", re.IGNORECASE
)


def extract_rating(text: str) -> str | None:
    """The 5-tier rating in `text`, or None if there is none.

    Passes on the NFKC-normalized text (so fullwidth punctuation like "Rating：Overweight"
    reads the same as ASCII):
    1. An explicit "Rating: X" label (the last one, skipping lines that list the scale).
    2. Ours: a "Rating" heading with the rating alone on the next line (the last such pair).
    3. Otherwise, the rating word, if exactly one of the five appears in the text.
    """
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)

    # The labelled rating, taking the last one written: a decision states its
    # rating after discussing the alternatives. Lines presenting the scale
    # itself are a legend the model echoed, not a call.
    labelled = None
    for line in norm.splitlines():
        if _RATING_SCALE_RE.search(line):
            continue
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            labelled = m.group(1).capitalize()
    if labelled:
        return labelled

    # A "Rating" heading followed by a line holding only the rating (the next non-empty line).
    lines = [line for line in norm.splitlines() if line.strip()]
    for heading, following in zip(lines, lines[1:], strict=False):
        if _RATING_HEADING_RE.match(heading):
            m = _RATING_ALONE_RE.match(following)
            if m:
                labelled = m.group(1).capitalize()
    if labelled:
        return labelled

    # No label. A single rating word in the text is the call; several are an
    # argument, and picking one of them reports a direction nobody decided --
    # prose that rejects a Buy before concluding Underweight read as Buy.
    named = {m.group(1).capitalize() for m in _RATING_WORD_RE.finditer(norm)}
    return named.pop() if len(named) == 1 else None


def parse_rating(text: str, default: str = RATING_REVIEW) -> str:
    """The 5-tier rating in `text`, or REVIEW when it has none."""
    rating = extract_rating(text)
    return rating if rating is not None else default


def is_review(signal: str) -> bool:
    """Whether a signal is the non-tradeable REVIEW marker."""
    return signal == RATING_REVIEW
