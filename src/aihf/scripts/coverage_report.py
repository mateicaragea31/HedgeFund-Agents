"""Coverage report: what every data tool gives an agent on each trade date, per ticker.

Run on the FEP login node after the collectors (a few minutes; reads the archive only):
    python -m aihf.scripts.coverage_report --tickers AAPL GOOGL AMZN --dry-run
    python -m aihf.scripts.coverage_report --tickers AAPL GOOGL AMZN

Trade dates = the trading days in [--start, --end] of the first ticker's price file.
Writes results/coverage/coverage_<start>_<end>.csv (one row per ticker-day: item counts, or
"gap" / "empty") and ..._summary.csv (per tool), and prints the summary. Exits with code 1 if
a tool other than StockTwits (DEVIATIONS D4) returns a gap on any ticker-day.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from aihf.data.coverage import coverage_table, summarize
from aihf.data.prices import price_file
from aihf.tools.archive import ArchiveSource

REPO = Path(__file__).resolve().parents[3]
EXPECTED_GAPS = {"stocktwits"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start", default="2024-01-02", help="first trade date (default: paper window)")
    p.add_argument("--end", default="2024-03-28", help="last trade date (default: paper window)")
    p.add_argument("--out", type=Path, default=REPO / "results" / "coverage")
    p.add_argument("--dry-run", action="store_true", help="list the trade dates, check nothing")
    args = p.parse_args(argv)
    if not os.environ.get("AIHF_DATA"):
        p.error("set AIHF_DATA (e.g. in ~/.bashrc)")

    (REPO / "logs").mkdir(exist_ok=True)
    log_file = REPO / "logs" / f"coverage_report-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])

    data = ArchiveSource.from_env()
    bars = pd.read_parquet(price_file(data.root, args.tickers[0]), columns=["date"])
    days = [f"{d:%Y-%m-%d}" for d in bars["date"] if args.start <= f"{d:%Y-%m-%d}" <= args.end]
    logging.info("%d trade dates %s..%s, tickers %s", len(days), days[0], days[-1], args.tickers)
    if args.dry_run:
        return 0

    table = coverage_table(data, [t.upper() for t in args.tickers], days)
    summary = summarize(table)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"coverage_{args.start}_{args.end}"
    table.to_csv(f"{stem}.csv", index=False)
    summary.to_csv(f"{stem}_summary.csv", index=False)
    logging.info("per ticker-day: %s.csv\n%s", stem, summary.to_string(index=False))

    unexpected = summary[(summary["gap"] > 0) & ~summary["check"].isin(EXPECTED_GAPS)]
    if len(unexpected):
        logging.error("unexpected gaps in: %s", ", ".join(unexpected["check"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
