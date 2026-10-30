"""Run every cell of an experiment (one repeat) against our vLLM server, in waves with memory.

    python -m aihf.scripts.run_backtest --experiment configs/experiments/e1_llama70b.yaml \\
        --repeat 0 --dry-run
    python -m aihf.scripts.run_backtest --experiment configs/experiments/e1_llama70b.yaml \\
        --repeat 0 --backend-url http://127.0.0.1:PORT/v1 --deadline-minutes 690

Results go to results/<run_id>/ (run_id = hash of the experiment and model files, the repeat and
our git commit): run.json, cells/<TICKER>_<DATE>/{state.json, report.json, llm_log.jsonl},
memory.md, reflections.jsonl, summary.json. Starting the same command again continues the run
(finished cells are skipped). Exit codes: 0 all cells done; 1 some cells failed after retries;
2 bad configuration; 3 stopped at the deadline (submit again to continue).
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from aihf.backtest.cell_worker import ProcessStarter, WorkerConfig
from aihf.backtest.plan import run_id, trade_dates, waves
from aihf.backtest.runner import Runner, summarize
from aihf.config import REPO, load_experiment
from aihf.data.prices import price_file
from aihf.llm import JsonlCallLog, make_chat_model
from aihf.memory import DecisionLog, settle_pending
from aihf.scripts.run_cell import git_info

logger = logging.getLogger("run_backtest")


def vllm_version(backend_url: str) -> str:
    try:
        with urllib.request.urlopen(backend_url.removesuffix("/v1") + "/version", timeout=10) as r:
            return json.load(r).get("version", "unknown")
    except Exception:  # an older server or no /version: not worth failing the run for
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--backend-url", help="vLLM server, e.g. http://127.0.0.1:28152/v1")
    p.add_argument("--dates", type=int, help="run only the first N trade dates (test runs); same run folder")
    p.add_argument("--max-workers", type=int, default=15, help="cells at the same time")
    p.add_argument("--cell-timeout-minutes", type=float, default=60)
    p.add_argument("--retries", type=int, default=2, help="extra attempts for a failed cell")
    p.add_argument("--deadline-minutes", type=float, help="stop starting cells after this long (12 h job: 690)")
    p.add_argument("--run-id", help="continue this run folder instead of the one the inputs hash to")
    p.add_argument("--dry-run", action="store_true", help="show the run id, waves and cells; run nothing")
    args = p.parse_args(argv)
    started = time.monotonic()

    if not os.environ.get("AIHF_DATA"):
        p.error("set AIHF_DATA (e.g. in ~/.bashrc)")
    experiment_path = args.experiment if args.experiment.is_absolute() else REPO / args.experiment
    experiment, model = load_experiment(experiment_path)  # refuses a too-late knowledge cutoff
    if not 0 <= args.repeat < experiment.repeats:
        p.error(f"--repeat must be in 0..{experiment.repeats - 1} for {experiment.name}")
    archive = Path(os.environ["AIHF_DATA"]) / "archive"
    missing = [t for t in experiment.tickers if not price_file(archive, t).exists()]
    if missing:
        p.error(f"no price file for {missing} under {archive}")
    git = git_info()
    model_path = REPO / experiment.model_profile
    rid = args.run_id or run_id(experiment_path, model_path, args.repeat, git["commit"])
    run_dir = Path(os.environ.get("AIHF_RESULTS", REPO / "results")) / rid
    dates = trade_dates(archive, experiment.tickers[0], str(experiment.window.start), str(experiment.window.end))
    if args.dates:
        dates = dates[: args.dates]
    plan = waves(dates, experiment.holding_period_days, experiment.memory)

    run_dir.mkdir(parents=True, exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)
    log_file = REPO / "logs" / f"run_backtest-{rid}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logger.info("run %s: %s repeat %d, model %s (%s), commit %s%s", rid, experiment.name, args.repeat,
                model.name, model.revision[:8], git["commit"][:7], "" if git["clean"] else " (NOT clean)")
    logger.info("%d trade dates %s..%s × %d tickers = %d cells in %d wave(s) of up to %d dates; memory %s",
                len(dates), dates[0], dates[-1], len(experiment.tickers), len(dates) * len(experiment.tickers),
                len(plan), experiment.holding_period_days if experiment.memory else len(dates),
                "on" if experiment.memory else "off")
    if args.dry_run:
        for i, w in enumerate(plan, 1):
            logger.info("wave %d: %s", i, ", ".join(w))
        return 0
    if not args.backend_url:
        p.error("--backend-url is required (except with --dry-run)")
    if not git["clean"]:
        logger.warning("the checkout has uncommitted changes: results may not be reproducible from the commit")

    run_json = run_dir / "run.json"
    meta = json.loads(run_json.read_text()) if run_json.exists() else {"sessions": []}
    meta.update(run_id=rid, experiment=experiment.model_dump(mode="json"), repeat=args.repeat,
                model={"name": model.name, "revision": model.revision, "path": model.path,
                       "hardware": model.hardware.model_dump(), "vllm": model.vllm},
                git=git, upstream_prompts_commit="35543d0")
    meta["sessions"].append({"started_at": datetime.now(UTC).isoformat(timespec="seconds"),
                             "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
                             "partition": os.environ.get("SLURM_JOB_PARTITION"),
                             "serving_image_sha256": os.environ.get("AIHF_SERVE_IMAGE_SHA", "unknown"),
                             "vllm_version": vllm_version(args.backend_url), "dates": len(dates)})
    run_json.write_text(json.dumps(meta, indent=2))

    memory = DecisionLog(run_dir / "memory.md") if experiment.memory else None
    settle = None
    if memory is not None:
        reflector = make_chat_model(args.backend_url, experiment.quick_model, temperature=experiment.temperature,
                                    max_tokens=experiment.max_tokens,
                                    callbacks=[JsonlCallLog(run_dir / "reflections.jsonl")])

        def settle() -> int:
            return settle_pending(memory, reflector, archive, holding_days=experiment.holding_period_days)

    deadline = started + args.deadline_minutes * 60 if args.deadline_minutes else None
    starter = ProcessStarter(WorkerConfig(str(experiment_path), str(model_path), args.backend_url, str(archive)))
    runner = Runner(run_dir, experiment.tickers, plan, start=starter, memory_log=memory, settle=settle,
                    max_workers=args.max_workers, cell_timeout=args.cell_timeout_minutes * 60,
                    retries=args.retries, deadline=deadline)
    result = runner.run()

    summary = summarize(run_dir, experiment.tickers, dates, result)
    summary.update(run_id=rid, finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
                   session_wall_seconds=round(time.monotonic() - started, 1))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    meta["sessions"][-1]["finished_at"] = summary["finished_at"]
    run_json.write_text(json.dumps(meta, indent=2))
    logger.info("done: %d / %d cells, %d failed, ratings %s (REVIEW %d), %d input / %d output tokens; %s",
                summary["cells_done"], summary["cells_total"], len(summary["cells_failed"]), summary["ratings"],
                summary["review"], summary["input_tokens"], summary["output_tokens"], run_dir / "summary.json")
    if not result.finished:
        logger.warning("stopped at the deadline: submit the same command again to continue")
        return 3
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
