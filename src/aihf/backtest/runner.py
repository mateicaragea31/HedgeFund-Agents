"""The wave loop: run every cell of a run, at most `max_workers` at a time, with memory between waves.

For each wave (backtest/plan.py):
  1. settle the earlier decisions whose holding window has traded (memory on);
  2. for each cell not done yet, read the lessons known on its date and start it;
  3. as each cell finishes, store its decision in the memory log (only this process writes it);
  4. a cell that fails or runs past `cell_timeout` is started again, at most `retries` more times,
     then recorded as failed; the run goes on.
Before starting a cell, the deadline is checked: no new cell starts in the last `launch_margin`
seconds, cells still running at the deadline are stopped, and the run reports itself unfinished
(start it again to continue: finished cells are skipped).

How a cell actually runs is passed in (`start`): in a child process for real runs
(backtest/cell_worker.py), in-process with a fake model in tests.
"""

import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellJob:
    ticker: str
    trade_date: str
    cell_dir: Path
    past_context: str = ""


class Handle(Protocol):
    """A started cell: is it finished, did it succeed, stop it."""

    def done(self) -> bool: ...
    def ok(self) -> bool: ...
    def kill(self) -> None: ...


Starter = Callable[[CellJob], Handle]


def cell_dir(run_dir: Path, ticker: str, trade_date: str) -> Path:
    return Path(run_dir) / "cells" / f"{ticker}_{trade_date}"


def read_report(path: Path) -> dict | None:
    report = Path(path) / "report.json"
    return json.loads(report.read_text()) if report.exists() else None


def is_done(path: Path) -> bool:
    """A cell is done when it saved its final state and its report has no error."""
    report = read_report(path)
    return bool(report) and report.get("error") is None and (Path(path) / "state.json").exists()


def final_decision(path: Path) -> str:
    state = Path(path) / "state.json"
    return json.loads(state.read_text()).get("final_trade_decision", "") if state.exists() else ""


@dataclass
class WaveStats:
    index: int
    dates: list[str]
    cells: int = 0  # cells run in this wave (not skipped)
    ok: int = 0
    failed: int = 0
    attempts: int = 0
    wall_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def output_tokens_per_second(self) -> float:
        return round(self.output_tokens / self.wall_seconds, 1) if self.wall_seconds else 0.0


@dataclass
class RunResult:
    finished: bool  # every cell done (no deadline stop)
    failed: list[str] = field(default_factory=list)  # cells that failed after all retries
    waves: list[WaveStats] = field(default_factory=list)


class Runner:
    def __init__(self, run_dir: Path, tickers: list[str], waves: list[list[str]], *, start: Starter,
                 memory_log=None, settle: Callable[[], int] | None = None, max_workers: int = 15,
                 cell_timeout: float = 1800, retries: int = 2, deadline: float | None = None,
                 launch_margin: float = 1200, poll: float = 2.0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.run_dir = Path(run_dir)
        self.tickers, self.wave_dates = tickers, waves
        self.start, self.memory_log, self.settle = start, memory_log, settle
        self.max_workers, self.cell_timeout, self.retries = max_workers, cell_timeout, retries
        self.deadline, self.launch_margin = deadline, launch_margin
        self.poll, self.clock, self.sleep = poll, clock, sleep

    # --- deadline ---

    def _may_launch(self) -> bool:
        return self.deadline is None or self.deadline - self.clock() > self.launch_margin

    def _past_deadline(self) -> bool:
        return self.deadline is not None and self.clock() >= self.deadline

    # --- one wave ---

    def _jobs(self, dates: list[str]) -> list[CellJob]:
        jobs = []
        for day in dates:
            for ticker in self.tickers:
                path = cell_dir(self.run_dir, ticker, day)
                if is_done(path):
                    continue
                lessons = self.memory_log.get_past_context(ticker, as_of=day) if self.memory_log else ""
                jobs.append(CellJob(ticker, day, path, lessons))
        return jobs

    def _finish(self, job: CellJob, ok: bool, attempts: int, stats: WaveStats, failed: list[str]) -> None:
        report = read_report(job.cell_dir) or {}
        if ok:
            stats.ok += 1
            llm = report.get("llm") or {}
            stats.input_tokens += llm.get("input_tokens", 0) or 0
            stats.output_tokens += llm.get("output_tokens", 0) or 0
            if self.memory_log is not None:
                decision = final_decision(job.cell_dir)
                if decision:
                    self.memory_log.store_decision(job.ticker, job.trade_date, decision)
        else:
            stats.failed += 1
            failed.append(f"{job.ticker}_{job.trade_date}")
            report.setdefault("ticker", job.ticker)
            report.setdefault("trade_date", job.trade_date)
            report["error"] = report.get("error") or f"no result after {attempts} attempts (failed or timed out)"
            job.cell_dir.mkdir(parents=True, exist_ok=True)
            (job.cell_dir / "report.json").write_text(json.dumps(report, indent=2))
        logger.info("%s %s: %s after %d attempt(s), rating %s", job.ticker, job.trade_date,
                    "ok" if ok else "FAILED", attempts, report.get("rating"))

    def _run_wave(self, jobs: list[CellJob], stats: WaveStats, failed: list[str]) -> bool:
        """Run the jobs; False if the deadline stopped the wave before all of them finished."""
        queue = [(job, 1) for job in jobs]  # (job, attempt number)
        running: list[tuple[CellJob, int, Handle, float]] = []
        while queue or running:
            while queue and len(running) < self.max_workers and self._may_launch():
                job, attempt = queue.pop(0)
                running.append((job, attempt, self.start(job), self.clock()))
                stats.attempts += 1
            if not running:  # nothing running and the deadline forbids starting more
                return False
            self.sleep(self.poll)
            still = []
            for job, attempt, handle, started in running:
                timed_out = self.clock() - started > self.cell_timeout
                if not handle.done() and not timed_out and not self._past_deadline():
                    still.append((job, attempt, handle, started))
                    continue
                if not handle.done():
                    handle.kill()
                    if self._past_deadline():  # stopped by the deadline: the next run redoes it
                        logger.warning("%s %s stopped at the deadline", job.ticker, job.trade_date)
                        continue
                    logger.warning("%s %s timed out (attempt %d)", job.ticker, job.trade_date, attempt)
                if handle.done() and handle.ok() and is_done(job.cell_dir):
                    self._finish(job, True, attempt, stats, failed)
                elif attempt <= self.retries:
                    queue.append((job, attempt + 1))
                else:
                    self._finish(job, False, attempt, stats, failed)
            running = still
        return True

    # --- the run ---

    def run(self) -> RunResult:
        result = RunResult(finished=True)
        for i, dates in enumerate(self.wave_dates, 1):
            if self.settle is not None:
                settled = self.settle()
                logger.info("wave %d: settled %d earlier decision(s)", i, settled)
            jobs = self._jobs(dates)
            stats = WaveStats(i, dates, cells=len(jobs))
            if jobs and not self._may_launch():
                result.finished = False
                break
            t0 = self.clock()
            complete = self._run_wave(jobs, stats, result.failed)
            stats.wall_seconds = round(self.clock() - t0, 1)
            if jobs:
                result.waves.append(stats)
                logger.info("wave %d (%s..%s): %d ok, %d failed, %.0f s, %.1f output tokens/s",
                            i, dates[0], dates[-1], stats.ok, stats.failed, stats.wall_seconds,
                            stats.output_tokens_per_second)
            if not complete:
                result.finished = False
                break
        return result


def summarize(run_dir: Path, tickers: list[str], dates: list[str], result: RunResult) -> dict:
    """summary.json: cells done/failed/missing, ratings (REVIEW counted, never read as Hold),
    tokens, and per-wave timing and throughput."""
    ratings, done, missing = Counter(), 0, []
    tokens_in = tokens_out = 0
    for day in dates:
        for ticker in tickers:
            path = cell_dir(run_dir, ticker, day)
            report = read_report(path)
            if is_done(path):
                done += 1
                ratings[report.get("rating") or "none"] += 1
                tokens_in += (report.get("llm") or {}).get("input_tokens", 0) or 0
                tokens_out += (report.get("llm") or {}).get("output_tokens", 0) or 0
            elif report is None:
                missing.append(f"{ticker}_{day}")
    waves = [{**asdict(w), "output_tokens_per_second": w.output_tokens_per_second} for w in result.waves]
    return {"cells_total": len(tickers) * len(dates), "cells_done": done, "cells_failed": result.failed,
            "cells_missing": missing, "finished": result.finished and not missing and not result.failed,
            "ratings": dict(ratings), "review": ratings.get("REVIEW", 0),
            "input_tokens": tokens_in, "output_tokens": tokens_out, "waves_this_session": waves}
