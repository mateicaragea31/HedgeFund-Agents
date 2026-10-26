"""The decision log: a markdown file of past decisions, their outcomes and the lessons drawn.

Copied from upstream `decision_log.py` (TradingMemoryLog, commit 35543d0, Apache-2.0, see
LICENSES/TradingAgents-LICENSE); same file format, so our logs read like upstream's. An entry
goes through two stages:

    [2024-01-10 | AAPL | Overweight | pending]                        ← store_decision
    DECISION:
    ...
    [2024-01-10 | AAPL | Overweight | +2.1% | +0.8% | 5d | resolved:2024-01-18]   ← settled
    DECISION:
    ...
    REFLECTION:
    ...

Differences from upstream:
- get_past_context always filters by the trade date (`as_of`); upstream turns the filter off for
  runs dated today, and all of ours are historical.
- Writes hold a file lock (CLAUDE.md §4.4), on top of upstream's write-then-rename.
- Upstream's optional cap on the number of entries (off by default) is left out.
"""

import fcntl
import re
from contextlib import contextmanager
from pathlib import Path

from aihf.agents.rating import parse_rating


class DecisionLog:
    """Append-only markdown log of decisions and reflections, one file per run."""

    # HTML comment: cannot appear in LLM prose output, safe as a hard delimiter
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        """Only one process at a time reads-and-rewrites the log (a lock on a side file)."""
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    # --- write: a new decision ---

    def store_decision(self, ticker: str, trade_date: str, final_trade_decision: str) -> None:
        """Append a pending entry. No model call. A second entry for the same ticker and date is
        ignored, pending or settled, so a re-run cell is never counted twice."""
        with self._locked():
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if line.startswith(f"[{trade_date} | {ticker} |") and line.endswith("]"):
                        return
            rating = parse_rating(final_trade_decision)  # REVIEW when unreadable
            tag = f"[{trade_date} | {ticker} | {rating} | pending]"
            entry = f"{tag}\n\nDECISION:\n{final_trade_decision}{self._SEPARATOR}"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(entry)

    # --- read ---

    def load_entries(self) -> list[dict]:
        """Every entry, oldest first, as a dict (date, ticker, rating, pending, raw, alpha, ...)."""
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]
        return [p for p in (self._parse_entry(raw) for raw in raw_entries) if p]

    def get_pending_entries(self) -> list[dict]:
        """Entries whose outcome is not known yet."""
        return [e for e in self.load_entries() if e.get("pending")]

    def get_past_context(self, ticker: str, as_of: str, n_same: int = 5, n_cross: int = 3) -> str:
        """The lessons a cell on `as_of` (yyyy-mm-dd) may read, for the Portfolio Manager's prompt.

        Point in time: only entries settled on or before `as_of` (resolved:<date> <= as_of).
        Up to `n_same` full entries for this ticker and `n_cross` reflections from other
        tickers, most recent first. "" when there is nothing yet.
        """
        entries = [e for e in self.load_entries()
                   if not e.get("pending") and e.get("resolved") and e["resolved"] <= as_of]
        if not entries:
            return ""

        same, cross = [], []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["ticker"] == ticker and len(same) < n_same:
                same.append(e)
            elif e["ticker"] != ticker and len(cross) < n_cross:
                cross.append(e)

        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    # --- write: outcomes ---

    def batch_update_with_outcomes(self, updates: list[dict]) -> None:
        """Settle several pending entries in one read and one atomic write.

        Each update has: ticker, trade_date, raw_return, alpha_return, holding_days,
        reflection, resolution_date. The new text goes to a temporary file that then replaces
        the log, so a crash mid-write never leaves half a log.
        """
        if not updates:
            return
        with self._locked():
            if not self.path.exists():
                return
            blocks = self.path.read_text(encoding="utf-8").split(self._SEPARATOR)
            update_map = {(u["trade_date"], u["ticker"]): u for u in updates}

            new_blocks = []
            for block in blocks:
                stripped = block.strip()
                if not stripped:
                    new_blocks.append(block)
                    continue
                lines = stripped.splitlines()
                tag_line = lines[0].strip()

                matched = False
                for (trade_date, ticker), upd in list(update_map.items()):
                    if tag_line.startswith(f"[{trade_date} | {ticker} |") and tag_line.endswith("| pending]"):
                        rating = [f.strip() for f in tag_line[1:-1].split("|")][2]
                        new_tag = self._resolved_tag(
                            trade_date, ticker, rating, f"{upd['raw_return']:+.1%}",
                            f"{upd['alpha_return']:+.1%}", upd["holding_days"],
                            upd.get("resolution_date"),
                        )
                        rest = "\n".join(lines[1:])
                        new_blocks.append(f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{upd['reflection']}")
                        del update_map[(trade_date, ticker)]
                        matched = True
                        break
                if not matched:
                    new_blocks.append(block)

            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(self._SEPARATOR.join(new_blocks), encoding="utf-8")
            tmp_path.replace(self.path)

    # --- helpers ---

    @staticmethod
    def _resolved_tag(trade_date, ticker, rating, raw_pct, alpha_pct, holding_days, resolution_date) -> str:
        """The settled tag. resolution_date (the last price bar used) is when the outcome became
        known: get_past_context filters on it."""
        tag = f"[{trade_date} | {ticker} | {rating} | {raw_pct} | {alpha_pct} | {holding_days}d"
        if resolution_date:
            tag += f" | resolved:{resolution_date}"
        return tag + "]"

    def _parse_entry(self, raw: str) -> dict | None:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        tag_line = lines[0].strip()
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None
        fields = [f.strip() for f in tag_line[1:-1].split("|")]
        if len(fields) < 4:
            return None
        resolved = None
        for f in fields[6:]:
            if f.startswith("resolved:"):
                resolved = f[len("resolved:"):].strip()
        entry = {
            "date": fields[0],
            "ticker": fields[1],
            "rating": fields[2],
            "pending": fields[3] == "pending",
            "raw": fields[3] if fields[3] != "pending" else None,
            "alpha": fields[4] if len(fields) > 4 else None,
            "holding": fields[5] if len(fields) > 5 else None,
            "resolved": resolved,
        }
        body = "\n".join(lines[1:]).strip()
        decision_match = self._DECISION_RE.search(body)
        reflection_match = self._REFLECTION_RE.search(body)
        entry["decision"] = decision_match.group(1).strip() if decision_match else ""
        entry["reflection"] = reflection_match.group(1).strip() if reflection_match else ""
        return entry

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"
