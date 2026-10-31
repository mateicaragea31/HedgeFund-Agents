"""Plot cumulative returns (the paper's Fig. 7 style) from an evaluation's curves.csv.

    python -m aihf.scripts.plot_evaluation --label e1_llama70b_paper_window_paper3_short_0bps

Reads results/evaluation/<label>/curves.csv and evaluation.json (written by aihf.scripts.evaluate)
and writes results/evaluation/<label>/plots/: <TICKER>.png (one per ticker) and all.png (side by
side). Needs matplotlib: pip install -e ".[plots]".
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from aihf.backtest.plots import plot_all, plot_ticker
from aihf.config import REPO

logger = logging.getLogger("plot_evaluation")


def subtitle(meta: dict) -> str:
    exp = meta.get("experiment", {})
    return (f"{exp.get('name', '?')} · mapping {meta.get('mapping')} · shorts {meta.get('short_mode')} · "
            f"{meta.get('cost_bps', 0):g} bps · fill at next open · ${meta.get('initial_cash', 0):,.0f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="folder under results/evaluation/")
    p.add_argument("--dry-run", action="store_true", help="check the inputs and print the plan only")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    folder = Path(os.environ.get("AIHF_RESULTS", REPO / "results")) / "evaluation" / args.label
    if not (folder / "curves.csv").exists():
        p.error(f"no curves.csv in {folder} (run aihf.scripts.evaluate first)")
    curves = pd.read_csv(folder / "curves.csv")
    meta_path = folder / "evaluation.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    tickers = list(dict.fromkeys(curves["ticker"]))
    out = folder / "plots"
    logger.info("%s: tickers %s, strategies %s → %s", args.label, tickers,
                sorted(curves["strategy"].unique()), out)
    if args.dry_run:
        return 0

    out.mkdir(exist_ok=True)
    text = subtitle(meta)
    for ticker in tickers:
        plot_ticker(curves, ticker, out / f"{ticker}.png", text)
    plot_all(curves, tickers, out / "all.png", text)
    logger.info("wrote %d plots to %s", len(tickers) + 1, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
