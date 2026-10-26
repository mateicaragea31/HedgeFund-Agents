"""Run one cell (one ticker on one trade date) through the whole pipeline against our vLLM server.

    python -m aihf.scripts.run_cell --experiment configs/experiments/e0_dev.yaml \\
        --ticker AAPL --date 2024-01-10 --backend-url http://127.0.0.1:PORT/v1

Writes <out>/<TICKER>_<DATE>/ with:
    state.json      the full final state (reports, both debates, the decision)
    llm_log.jsonl   every model call: prompt, reply, tokens, time
    report.json     rating, wall time, LLM calls and tokens per agent, every data request and
                    whether it was a gap, git commit, models and the resolved config

Memory is only read here (lessons known on the trade date, from <out>/memory.md); storing and
settling decisions is the backtest controller's job (CLAUDE.md §4.4). Prices come from
$AIHF_DATA/archive. Exits with code 1 if the cell fails (report.json then holds the error).
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

from aihf.agents.context import build_instrument_context, load_instruments
from aihf.config import REPO, ExperimentConfig, ModelConfig, load_experiment
from aihf.data.prices import coverage
from aihf.graph.cell import run_cell
from aihf.graph.setup import build_graph
from aihf.llm import JsonlCallLog, make_chat_model
from aihf.memory import DecisionLog
from aihf.tools.archive import ArchiveSource
from aihf.tools.logged import LoggedSource

logger = logging.getLogger("run_cell")


class _WarningCounter(logging.Handler):
    """Counts the warnings of one logger: form → plain-text fallbacks (aihf.llm.structured) and
    dropped extra tool calls (aihf.agents.analysts)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def git_info() -> dict:
    """Our commit and whether the checkout is clean. The container has no git, so the sbatch
    script computes both outside it and passes them in AIHF_GIT_COMMIT / AIHF_GIT_CLEAN."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "clean": not dirty}
    except (OSError, subprocess.CalledProcessError):
        clean = os.environ.get("AIHF_GIT_CLEAN")
        return {"commit": os.environ.get("AIHF_GIT_COMMIT", "unknown"),
                "clean": None if clean is None else clean == "1"}


def summarize_llm_log(path: Path) -> dict:
    """Calls and tokens, in total and per agent, from the call log."""
    total = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}
    by_agent: dict[str, dict] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            agent = by_agent[rec.get("agent") or "(none)"]
            for d in (total, agent):
                d["calls"] += 1
                d["input_tokens"] += rec.get("input_tokens") or 0
                d["output_tokens"] += rec.get("output_tokens") or 0
            total["errors"] += rec.get("error") is not None
    return {**total, "by_agent": dict(by_agent)}


def _jsonable(obj):
    """For json.dumps: messages and other LangChain objects as plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def run(
    experiment: ExperimentConfig,
    model: ModelConfig,
    ticker: str,
    trade_date: str,
    *,
    quick_llm,
    deep_llm,
    data,
    cell_dir: Path,
    memory_path: Path,
    extra: dict | None = None,
) -> dict:
    """Run the cell and write state.json + report.json into cell_dir. Returns the report.

    The models (with their call log already attached) and the DataSource are passed in, so
    tests can run this with a fake model and a fake archive.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    logged = LoggedSource(data)
    counters = {"structured_fallbacks": ("aihf.llm.structured", _WarningCounter()),
                "tool_calls_dropped_turns": ("aihf.agents.analysts", _WarningCounter())}
    for name, counter in counters.values():
        logging.getLogger(name).addHandler(counter)
    past_context = DecisionLog(memory_path).get_past_context(ticker, as_of=trade_date) if experiment.memory else ""
    report = {
        "ticker": ticker,
        "trade_date": trade_date,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "experiment": experiment.model_dump(mode="json"),
        "model": {"name": model.name, "profile": model.profile, "revision": model.revision},
        "past_context_chars": len(past_context),
        **(extra or {}),
    }
    t0 = time.monotonic()
    try:
        graph = build_graph(quick_llm, deep_llm, logged, experiment.analysts,
                            experiment.max_debate_rounds, experiment.max_risk_discuss_rounds)
        identity = load_instruments().get(ticker)
        final_state, rating = run_cell(graph, ticker, trade_date,
                                       instrument_context=build_instrument_context(ticker, identity),
                                       past_context=past_context)
        (cell_dir / "state.json").write_text(
            json.dumps(final_state, default=_jsonable, ensure_ascii=False, indent=2), encoding="utf-8")
        report.update(rating=rating, error=None)
    except Exception as exc:
        logger.exception("cell failed")
        report.update(rating=None, error=f"{type(exc).__name__}: {exc}")
    finally:
        for name, counter in counters.values():
            logging.getLogger(name).removeHandler(counter)

    report["wall_seconds"] = round(time.monotonic() - t0, 1)
    report["llm"] = summarize_llm_log(cell_dir / "llm_log.jsonl")
    for key, (_, counter) in counters.items():
        report["llm"][key] = counter.count
    report["data_requests"] = logged.requests
    report["gaps"] = sorted({r["method"] for r in logged.requests if r.get("gap")})
    report["real_data"] = sorted({r["method"] for r in logged.requests
                                  if r.get("gap") is False and not r.get("error")})
    (cell_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", type=Path, required=True, help="configs/experiments/<name>.yaml")
    p.add_argument("--ticker", required=True)
    p.add_argument("--date", required=True, help="trade date, YYYY-MM-DD")
    p.add_argument("--backend-url", help="vLLM server, e.g. http://127.0.0.1:28152/v1")
    p.add_argument("--out", type=Path, default=REPO / "results" / "dev", help="results folder")
    p.add_argument("--force", action="store_true", help="run again even if the cell has a state.json")
    p.add_argument("--dry-run", action="store_true", help="check the config and the data, run nothing")
    args = p.parse_args(argv)

    (REPO / "logs").mkdir(exist_ok=True)
    log_file = REPO / "logs" / f"run_cell-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logger.info("run_cell %s (log: %s)", " ".join(argv or sys.argv[1:]), log_file)

    experiment, model = load_experiment(args.experiment)  # refuses a too-late knowledge cutoff
    ticker = args.ticker.upper()
    if ticker not in experiment.tickers:
        p.error(f"{ticker} is not in the experiment's tickers {experiment.tickers}")
    if not experiment.window.start <= date.fromisoformat(args.date) <= experiment.window.end:
        p.error(f"{args.date} is outside the window {experiment.window.start}..{experiment.window.end}")
    if not os.environ.get("AIHF_DATA"):
        p.error("set AIHF_DATA (e.g. in ~/.bashrc)")
    archive = ArchiveSource.from_env()
    cov = coverage(archive.root, ticker)
    if cov is None:
        p.error(f"{ticker} has no price file under {archive.root}")
    cell_dir = args.out / f"{ticker}_{args.date}"
    if (cell_dir / "state.json").exists() and not args.force:
        logger.info("%s already done (state.json exists); use --force to run it again", cell_dir)
        return 0

    logger.info("experiment %s, model %s (%s), quick=%s deep=%s, analysts=%s, rounds=%d/%d, memory=%s",
                experiment.name, model.name, model.revision[:8], experiment.quick_model,
                experiment.deep_model, experiment.analysts, experiment.max_debate_rounds,
                experiment.max_risk_discuss_rounds, experiment.memory)
    logger.info("prices for %s: %s..%s; output: %s", ticker, cov.first, cov.last, cell_dir)
    if args.dry_run:
        logger.info("dry run: nothing executed")
        return 0
    if not args.backend_url:
        p.error("--backend-url is required (except with --dry-run)")

    (cell_dir / "llm_log.jsonl").unlink(missing_ok=True)  # a re-run starts a fresh log
    call_log = JsonlCallLog(cell_dir / "llm_log.jsonl")
    quick = make_chat_model(args.backend_url, experiment.quick_model,
                            temperature=experiment.temperature, callbacks=[call_log])
    deep = make_chat_model(args.backend_url, experiment.deep_model,
                           temperature=experiment.temperature, callbacks=[call_log])
    report = run(experiment, model, ticker, args.date, quick_llm=quick, deep_llm=deep, data=archive,
                 cell_dir=cell_dir, memory_path=args.out / "memory.md",
                 extra={"backend_url": args.backend_url})

    llm = report["llm"]
    logger.info("rating %s in %.0f s; %d LLM calls, %d tokens in, %d out, %d form fallbacks, "
                "%d turns with extra tool calls dropped", report["rating"], report["wall_seconds"],
                llm["calls"], llm["input_tokens"], llm["output_tokens"], llm["structured_fallbacks"],
                llm["tool_calls_dropped_turns"])
    logger.info("gaps: %s; real data: %s", report["gaps"], report["real_data"])
    if report["error"]:
        logger.error("cell failed: %s", report["error"])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
