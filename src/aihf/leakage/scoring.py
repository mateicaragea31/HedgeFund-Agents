"""Reading the model's answers and scoring them against the truth and the naive baseline.

Per question: the parsed answer, whether it parsed, and its error (relative or absolute) or
whether it is correct. Per (window, kind): how often answers parsed, the median error of the
model and of the naive guess, how often the model beat the naive guess, or the accuracy and the
chance level. Leakage shows up as the test window looking like the control window: errors well
below the naive guess, or accuracy well above chance.
"""

import json
import re

import pandas as pd

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
CHANCE = {"direction": 0.5, "bucket": 0.25}


_DATES = re.compile(
    r"\d{4}-\d{2}(-\d{2})?"  # 2024-01-09, 2024-01
    r"|(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}"  # January 9, 2024
    r"|\b(19|20)\d{2}\b")  # a bare year


def _json_answer(text: str):
    """The "answer" of a JSON form reply ({"answer": 185.6}), or None if the reply isn't one."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value.get("answer") if isinstance(value, dict) else None


def parse_answer(reply: str, answer_format: str):
    """The answer in the asked format, or None if it can't be read.

    A JSON form reply ({"answer": ...}) is read directly. Free text, the fallback:
    "$185.64" → 185.64; "1,234.5 billion" → 1234.5; "Higher." → "higher"; "B) rose…" → "B".
    Dates and years are skipped ("…on 2024-01-09 was $185" → 185), and a refusal with no other
    number ("I'm not aware of the price on 2024-01-09") gives None.
    """
    text = (reply or "").strip()
    form = _json_answer(text)
    if form is not None:
        text = str(form)
    if answer_format == "number":
        if isinstance(form, int | float):
            return float(form)
        cleaned = _DATES.sub(" ", text)
        dollar = re.search(r"\$\s*(-?\d[\d,]*\.?\d*)", cleaned)
        match = dollar.group(1) if dollar else (m.group() if (m := _NUMBER.search(cleaned)) else None)
        if match is None:
            return None
        try:
            return float(match.replace(",", ""))
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
            # EPS can be near zero or negative (Amazon 2022: -0.27), where a relative error explodes:
            # score it in dollars, like the rates are scored in points.
            unit = "abs" if r["kind"] == "eps" else r["error_unit"]
            rec["error_unit"] = unit
            rec["error"] = _error(parsed, r["truth"], unit)
            rec["naive_error"] = _error(r["naive"], r["truth"], unit)
            rec["beats_naive"] = (rec["error"] < rec["naive_error"]
                                  if rec["error"] is not None and rec["naive_error"] is not None else None)
            latest = r.get("naive_latest")
            rec["latest_error"] = _error(latest, r["truth"], unit) if latest is not None and latest == latest else None
            rec["beats_latest"] = (rec["error"] < rec["latest_error"]
                                   if rec["error"] is not None and rec["latest_error"] is not None else None)
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
            row.update(error_unit="%" if (g["error_unit"] == "pct").all() else "abs",
                       median_error=round(float(ok["error"].median()), 4) if len(ok) else None,
                       median_naive_error=round(float(g["naive_error"].median()), 4)
                       if g["naive_error"].notna().any() else None,
                       beats_naive=round(float(ok["beats_naive"].dropna().mean()), 3)
                       if ok["beats_naive"].notna().any() else None)
            if "latest_error" in g and g["latest_error"].notna().any():
                row.update(median_latest_error=round(float(g["latest_error"].median()), 4),
                           beats_latest=round(float(ok["beats_latest"].dropna().mean()), 3))
        rows.append(row)
    return pd.DataFrame(rows)
