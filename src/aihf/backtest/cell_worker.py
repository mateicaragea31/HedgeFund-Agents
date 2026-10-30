"""One cell in its own process, for the backtest runner.

Each cell runs in a child process so a stuck cell can be stopped (a thread can't be) and cells
share nothing but the vLLM server. The child builds its own models, data source and graph, runs
the cell with the lessons the runner passed in, and writes state.json, report.json and
llm_log.jsonl into the cell folder (scripts/run_cell.py `run`). Exit code 0 = no error.
"""

import json
import logging
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from aihf.backtest.runner import CellJob

_SPAWN = mp.get_context("spawn")  # a fresh interpreter per cell: no state shared with the runner


@dataclass(frozen=True)
class WorkerConfig:
    experiment_path: str
    model_path: str
    backend_url: str
    archive_root: str


def work(cfg: WorkerConfig, job: CellJob) -> None:
    """The child process: run one cell, exit 1 if it failed."""
    from aihf.config import load_experiment
    from aihf.llm import JsonlCallLog, make_chat_model
    from aihf.scripts.run_cell import run
    from aihf.tools.archive import ArchiveSource

    job.cell_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.FileHandler(job.cell_dir / "worker.log", mode="w")])
    experiment, model = load_experiment(Path(cfg.experiment_path))
    (job.cell_dir / "llm_log.jsonl").unlink(missing_ok=True)  # an attempt starts a fresh log
    log = JsonlCallLog(job.cell_dir / "llm_log.jsonl")
    quick = make_chat_model(cfg.backend_url, experiment.quick_model, temperature=experiment.temperature,
                            max_tokens=experiment.max_tokens, callbacks=[log])
    deep = make_chat_model(cfg.backend_url, experiment.deep_model, temperature=experiment.temperature,
                           max_tokens=experiment.max_tokens, callbacks=[log])
    report = run(experiment, model, job.ticker, job.trade_date, quick_llm=quick, deep_llm=deep,
                 data=ArchiveSource(Path(cfg.archive_root)), cell_dir=job.cell_dir,
                 past_context=job.past_context, extra={"backend_url": cfg.backend_url})
    sys.exit(0 if report["error"] is None else 1)


class ProcessHandle:
    def __init__(self, process):
        self.process = process

    def done(self) -> bool:
        return not self.process.is_alive()

    def ok(self) -> bool:
        return self.process.exitcode == 0

    def kill(self) -> None:
        self.process.terminate()
        self.process.join(10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(5)


class ProcessStarter:
    """The runner's `start`: each call launches `target(cfg, job)` in a new process."""

    def __init__(self, cfg: WorkerConfig, target=work):
        self.cfg, self.target = cfg, target

    def __call__(self, job: CellJob) -> ProcessHandle:
        process = _SPAWN.Process(target=self.target, args=(self.cfg, job), daemon=False)
        process.start()
        return ProcessHandle(process)


def _selftest_target(cfg: WorkerConfig, job: CellJob) -> None:
    """For tests of the process machinery: behaves as told by the ticker name.
    "OK…" writes a finished cell, "FAIL…" exits 1, "HANG…" sleeps forever."""
    if job.ticker.startswith("HANG"):
        time.sleep(3600)
    if job.ticker.startswith("FAIL"):
        sys.exit(1)
    job.cell_dir.mkdir(parents=True, exist_ok=True)
    (job.cell_dir / "state.json").write_text(json.dumps({"final_trade_decision": "**Rating**: Hold"}))
    (job.cell_dir / "report.json").write_text(json.dumps({"rating": "Hold", "error": None,
                                                          "past_context_chars": len(job.past_context)}))
