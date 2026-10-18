"""Download daily prices from Yahoo Finance into the price archive ($AIHF_DATA/archive/prices/).

Run on the FEP login node (light work, a few seconds per ticker):
    python -m aihf.scripts.collect_prices --tickers AAPL GOOGL AMZN --dry-run
    python -m aihf.scripts.collect_prices --tickers AAPL GOOGL AMZN

Tickers already saved are skipped unless --force. Exits with code 1 if any ticker fails its
checks (nothing is saved for that ticker). A log is written to logs/collect_prices-<time>.log.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from aihf.data.collect_prices import PriceDataError, collect

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL GOOGL AMZN")
    p.add_argument("--start", default="2019-01-02", help="first day (default: 5 years before the paper window)")
    p.add_argument("--end", default="2024-06-28", help="last day (default: 3 months after the paper window)")
    p.add_argument("--no-split-after", default="2024-01-01",
                   help="refuse a ticker with a stock split on or after this day (default: window start)")
    p.add_argument("--root", type=Path, help="archive folder (default: $AIHF_DATA/archive)")
    p.add_argument("--force", action="store_true", help="download again even if the file exists")
    p.add_argument("--dry-run", action="store_true", help="only print what would be done")
    args = p.parse_args(argv)

    if args.root is None:
        if not os.environ.get("AIHF_DATA"):
            p.error("set AIHF_DATA (e.g. in ~/.bashrc) or pass --root")
        args.root = Path(os.environ["AIHF_DATA"]) / "archive"

    (REPO / "logs").mkdir(exist_ok=True)
    log_file = REPO / "logs" / f"collect_prices-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logging.info("collect_prices %s -> %s (log: %s)", " ".join(argv or sys.argv[1:]), args.root, log_file)

    try:
        collect(args.tickers, args.start, args.end, args.root, no_split_after=args.no_split_after,
                force=args.force, dry_run=args.dry_run)
    except PriceDataError as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
