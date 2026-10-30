"""Leakage probe: ask the model, with no tools, questions whose answers lie in the test window
(after its knowledge cutoff) and in a control window (before it), and score the answers.

    python -m aihf.scripts.leakage_probe --model configs/models/llama31_8b_dev.yaml --dry-run
    python -m aihf.scripts.leakage_probe --model configs/models/llama31_8b_dev.yaml \\
        --backend-url http://127.0.0.1:PORT/v1

Questions come from $AIHF_DATA/archive (prices, SEC filings, FRED vintages), so every right
answer is exact. Writes results/leakage/<model name>/: questions.jsonl, answers.jsonl (with
parsed answers and errors), llm_log.jsonl, summary.csv; prints the summary. Temperature 0.
"""

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from aihf.agents.context import load_instruments
from aihf.config import REPO, load_model
from aihf.leakage.questions import Question, build_questions
from aihf.leakage.scoring import score_rows, summarize
from aihf.llm import JsonlCallLog, make_chat_model
from aihf.llm.structured import response_format

logger = logging.getLogger("leakage_probe")

SYSTEM = ("You are a financial analyst. Answer from your own knowledge only: you have no tools and "
          "no internet access. If you are not sure, give your single best guess; never refuse and "
          "never explain. Reply with only the answer, in the requested format.")
TICKERS = ["AAPL", "GOOGL", "AMZN", "SPY"]


# The answer forms: vLLM's guided decoding makes the model fill one, so it always gives a
# best guess and can't refuse (a model that doesn't know guesses no better than the baseline).
class NumberAnswer(BaseModel):
    answer: float


class DirectionAnswer(BaseModel):
    answer: Literal["higher", "lower"]


class LetterAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D"]


FORMS = {"number": NumberAnswer, "higher/lower": DirectionAnswer, "letter": LetterAnswer}


def ask_all(llm, questions) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        form = response_format(FORMS[q.answer_format])
        reply = llm.bind(response_format=form).invoke([("system", SYSTEM), ("human", q.text)]).content
        rows.append({**q.to_dict(), "reply": reply})
        if i % 25 == 0:
            logger.info("%d / %d answered", i, len(questions))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="configs/models/<name>.yaml")
    p.add_argument("--backend-url", help="vLLM server, e.g. http://127.0.0.1:28152/v1")
    p.add_argument("--test-window", nargs=2, default=["2024-01-02", "2024-03-28"])
    p.add_argument("--control-window", nargs=2, default=["2023-01-03", "2023-03-31"])
    p.add_argument("--out", type=Path, help="default: results/leakage/<model name>")
    p.add_argument("--dry-run", action="store_true", help="build and show the questions, ask nothing")
    p.add_argument("--rescore", action="store_true",
                   help="re-read the saved answers.jsonl and recompute the scores (no model needed)")
    args = p.parse_args(argv)
    if not os.environ.get("AIHF_DATA"):
        p.error("set AIHF_DATA (e.g. in ~/.bashrc)")

    (REPO / "logs").mkdir(exist_ok=True)
    log_file = REPO / "logs" / f"leakage_probe-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])

    model = load_model(REPO / args.model if not args.model.is_absolute() else args.model)
    out = args.out or REPO / "results" / "leakage" / model.name
    if args.rescore:
        saved = pd.read_json(out / "answers.jsonl", lines=True)
        keep = [c for c in saved.columns if c in {*Question.__dataclass_fields__, "reply"}]
        scored = score_rows(saved[keep].to_dict("records"))
        scored.to_json(out / "answers.jsonl", orient="records", lines=True)
        summary = summarize(scored)
        summary.to_csv(out / "summary.csv", index=False)
        logger.info("rescored %s\n%s", out, summary.to_string(index=False))
        return 0
    root = Path(os.environ["AIHF_DATA"]) / "archive"
    windows = {"test": tuple(args.test_window), "control": tuple(args.control_window)}
    questions = build_questions(root, load_instruments(), TICKERS, windows)
    counts = {}
    for q in questions:
        counts[(q.window, q.kind)] = counts.get((q.window, q.kind), 0) + 1
    logger.info("model %s (cutoff %s); %d questions: %s", model.name, model.knowledge_cutoff, len(questions),
                ", ".join(f"{w}/{k} {n}" for (w, k), n in sorted(counts.items())))
    for q in questions[:3]:
        logger.info("example: %s | truth %s | naive %s", q.text, q.truth, q.naive)
    if args.dry_run:
        return 0
    if not args.backend_url:
        p.error("--backend-url is required (except with --dry-run)")

    out.mkdir(parents=True, exist_ok=True)
    (out / "questions.jsonl").write_text("".join(json.dumps(q.to_dict()) + "\n" for q in questions))
    (out / "llm_log.jsonl").unlink(missing_ok=True)
    llm = make_chat_model(args.backend_url, model.name, temperature=0, max_tokens=60,
                          callbacks=[JsonlCallLog(out / "llm_log.jsonl")])
    scored = score_rows(ask_all(llm, questions))
    scored.to_json(out / "answers.jsonl", orient="records", lines=True)
    summary = summarize(scored)
    summary.to_csv(out / "summary.csv", index=False)
    meta = {"model": model.name, "revision": model.revision, "knowledge_cutoff": str(model.knowledge_cutoff),
            "windows": windows, "questions": len(questions), "finished_at": datetime.now(UTC).isoformat(),
            "git_commit": os.environ.get("AIHF_GIT_COMMIT", "unknown")}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("results in %s\n%s", out, summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
