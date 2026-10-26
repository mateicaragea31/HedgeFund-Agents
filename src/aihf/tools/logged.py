"""A DataSource wrapper that writes down every data request and whether it came back as a gap.

It sits between the agents and the real DataSource, so it sees both the analysts' tool calls
and the Sentiment Analyst's direct fetches. The cell report lists these requests, so a run
shows plainly which inputs were real data and which were "unavailable" notes.
"""

from aihf.tools.source import DataSource


def is_gap(text: str) -> bool:
    """Whether a data answer is a gap note (the shape made by tools/source.py `gap`)."""
    return text.startswith("<") and " unavailable for " in text


class LoggedSource:
    """Pass every get_* call on to `inner`, and record it in `self.requests`."""

    def __init__(self, inner: DataSource):
        self.inner = inner
        self.requests: list[dict] = []

    def __getattr__(self, name):
        # Only called for names this class doesn't have, i.e. the DataSource methods.
        method = getattr(self.inner, name)
        if not name.startswith("get_"):
            return method

        def logged(*args):
            record = {"method": name, "args": [str(a) for a in args]}
            self.requests.append(record)
            try:
                result = method(*args)
            except Exception as exc:
                record.update(gap=False, chars=0, error=f"{type(exc).__name__}: {exc}")
                raise
            record.update(gap=is_gap(result), chars=len(result), error=None)
            return result

        return logged
