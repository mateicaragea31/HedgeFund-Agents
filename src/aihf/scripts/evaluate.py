"""Evaluate backtest runs: simulate their ratings and the baselines, print a Table-1-shaped table.

    python -m aihf.scripts.evaluate --runs d7a3dd4bcfe6 fe41aeeed439 59b59224556a
    python -m aihf.scripts.evaluate --runs <ids> --mapping graded5 --short-mode flat --cost-bps 10

Reads results/<run_id>/ (run.json for the experiment; cells/*/report.json for the ratings) and the
price archive ($AIHF_DATA/archive; DTB3 for the risk-free rate if collected). Writes
results/evaluation/<label>/: table.csv (mean ± std over the runs, baselines alongside),
per_run.csv (every run, with bootstrap 95 % intervals), curves.csv (daily values),
evaluation.json (settings, REVIEW counts). See src/aihf/backtest/ for every convention.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from aihf.backtest.evaluation import METRICS, evaluate_runs, save
from aihf.backtest.plan import trade_dates
from aihf.backtest.simulator import MAPPINGS, SimConfig
from aihf.config import REPO

logger = logging.getLogger("evaluate")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run ids (the repeats of one experiment)")
    p.add_argument("--mapping", choices=sorted(MAPPINGS), default="paper3")
    p.add_argument("--short-mode", choices=["short", "flat"], default="short")
    p.add_argument("--cost-bps", type=float, default=0.0, help="per side")
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--label", help="output folder name (default: from the experiment and settings)")
    p.add_argument("--dry-run", action="store_true", help="check the runs and print the plan only")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not os.environ.get("AIHF_DATA"):
        p.error("set AIHF_DATA (e.g. in ~/.bashrc)")

    results = Path(os.environ.get("AIHF_RESULTS", REPO / "results"))
    runs = {rid: results / rid for rid in args.runs}
    metas = {}
    for rid, path in runs.items():
        if not (path / "run.json").exists():
            p.error(f"no run.json in {path}")
        metas[rid] = json.loads((path / "run.json").read_text())
    names = {m["experiment"]["name"] for m in metas.values()}
    if len(names) != 1:
        p.error(f"the runs belong to different experiments: {names}")
    experiment = next(iter(metas.values()))["experiment"]
    archive = Path(os.environ["AIHF_DATA"]) / "archive"
    tickers = experiment["tickers"]
    dates = trade_dates(archive, tickers[0], experiment["window"]["start"], experiment["window"]["end"])
    label = args.label or (f"{experiment['name']}_{args.mapping}_{args.short_mode}_{args.cost_bps:g}bps")
    logger.info("%s: %d runs (repeats %s), %s, %d trade dates %s..%s; %s, short_mode %s, %g bps → results/evaluation/%s",
                experiment["name"], len(runs), [m.get("repeat") for m in metas.values()], tickers, len(dates),
                dates[0], dates[-1], args.mapping, args.short_mode, args.cost_bps, label)
    if args.dry_run:
        return 0

    config = SimConfig(initial_cash=args.initial_cash, cost_bps=args.cost_bps, short_mode=args.short_mode)
    result = evaluate_runs(runs, tickers, dates, archive, mapping=args.mapping, config=config)
    out = results / "evaluation" / label
    save(result, out, {"experiment": experiment, "runs": {r: m.get("repeat") for r, m in metas.items()},
                       "mapping": args.mapping, "short_mode": args.short_mode, "cost_bps": args.cost_bps,
                       "initial_cash": args.initial_cash, "dates": [dates[0], dates[-1], len(dates)]})
    t = result["table"].copy()
    for m in METRICS:
        scale = 100 if m in ("CR", "AR", "MDD") else 1
        t[m] = [f"{v * scale:.2f}" + (f" ± {s * scale:.2f}" if s == s and s is not None else "")
                for v, s in zip(t[m], t[f"{m}_std"], strict=True)]
    logger.info("CR, AR, MDD in %%; Sharpe annualised. REVIEW counts: %s\n%s\n(per-run values and "
                "bootstrap 95%% intervals: %s)", result["review_counts"],
                t[["ticker", "strategy", "repeats", *METRICS]].to_string(index=False), out / "per_run.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
