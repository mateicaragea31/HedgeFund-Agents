"""Reading the model's answers and scoring them against the truth and the naive baseline.

Per question: the parsed answer, whether it parsed, and its error (relative or absolute) or
whether it is correct. Per (window, kind): how often answers parsed, the median error of the
model and of the naive guess, how often the model beat the naive guess, or the accuracy and the
chance level. Leakage shows up as the test window looking like the control window: errors well
below the naive guess, or accuracy well above chance.
"""

import re

import pandas as pd

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
CHANCE = {"direction": 0.5, "bucket": 0.25}


def parse_answer(reply: str, answer_format: str):
    """The answer in the asked format, or None if it can't be read.

    "$185.64" → 185.64; "1,234.5" → 1234.5; "Higher." → "higher"; "B) rose…" → "B".
    For a number, the first one in the reply counts.
    """
    text = (reply or "").strip()
    if answer_format == "number":
        match = _NUMBER.search(text.replace("$", ""))
        if not match:
            return None
        try:
            return float(match.group().replace(",", ""))
        except ValueError:
            return None
    if answer_format == "higher/lower":
        found = re.findall(r"\b(higher|lower)\b", text.lower())
        return found[0] if found else None
    if answer_format == "letter":
        match = re.match(r"\W*([ABCD])\b", text.upper())
        return match.group(1) if match else None
    raise ValueError(f"unknown answer format {answer_format!r}")


def _error(value, truth, kind: str) -> float | None:
    if value is None or truth is None:
        return None
    return abs(value - truth) / abs(truth) if kind == "pct" else abs(value - truth)


def score_rows(rows: list[dict]) -> pd.DataFrame:
    """rows: question dicts plus "reply". Adds parsed, error, naive_error, beats_naive, correct."""
    out = []
    for r in rows:
        parsed = parse_answer(r["reply"], r["answer_format"])
        rec = {**r, "parsed": parsed, "parsed_ok": parsed is not None}
        if r["answer_format"] == "number":
            rec["error"] = _error(parsed, r["truth"], r["error_unit"])
            rec["naive_error"] = _error(r["naive"], r["truth"], r["error_unit"])
            rec["beats_naive"] = (rec["error"] < rec["naive_error"]
                                  if rec["error"] is not None and rec["naive_error"] is not None else None)
        else:
            rec["correct"] = parsed == r["truth"]
        out.append(rec)
    return pd.DataFrame(out)


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (window, kind)."""
    rows = []
    for (window, kind), g in scored.groupby(["window", "kind"], sort=True):
        row = {"window": window, "kind": kind, "questions": len(g), "parsed": int(g["parsed_ok"].sum())}
        if kind in CHANCE:
            row.update(accuracy=round(float(g["correct"].mean()), 3), chance=CHANCE[kind])
        else:
            ok = g[g["error"].notna()]
            row.update(error_unit="%" if (g["error_unit"] == "pct").all() else "mixed/points",
                       median_error=round(float(ok["error"].median()), 4) if len(ok) else None,
                       median_naive_error=round(float(g["naive_error"].median()), 4)
                       if g["naive_error"].notna().any() else None,
                       beats_naive=round(float(ok["beats_naive"].dropna().mean()), 3)
                       if ok["beats_naive"].notna().any() else None)
        rows.append(row)
    return pd.DataFrame(rows)
