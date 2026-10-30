"""Download company news (EODHD) and global news (the Guardian) into the archive, month by month.

Run on the FEP login node; needs EODHD_API_KEY and GUARDIAN_API_KEY in ~/.bashrc:
    python -m aihf.scripts.collect_news --tickers AAPL GOOGL AMZN --dry-run
    python -m aihf.scripts.collect_news --tickers AAPL GOOGL AMZN

Writes $AIHF_DATA/archive/news/{eodhd,guardian}/... (format: src/aihf/data/news.py). Completed
months are skipped. If a feed's daily limit is reached, the run stops with exit code 2: run the
same command again the next day and it resumes. Other failures exit with code 1. A log is
written to logs/collect_news-<time>.log.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from aihf.data.collect_news import LimitReached, NewsDataError, collect

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL GOOGL AMZN")
    p.add_argument("--start", default="2023-12-01", help="first day (default: a month before the paper window)")
    p.add_argument("--end", default="2024-03-31", help="last day (default: end of the paper window's month)")
    p.add_argument("--root", type=Path, help="archive folder (default: $AIHF_DATA/archive)")
    p.add_argument("--force", action="store_true", help="download completed months again")
    p.add_argument("--dry-run", action="store_true", help="only print what would be done")
    args = p.parse_args(argv)

    if args.root is None:
        if not os.environ.get("AIHF_DATA"):
            p.error("set AIHF_DATA (e.g. in ~/.bashrc) or pass --root")
        args.root = Path(os.environ["AIHF_DATA"]) / "archive"

    (REPO / "logs").mkdir(exist_ok=True)
    log_file = REPO / "logs" / f"collect_news-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logging.info("collect_news %s -> %s (log: %s)", " ".join(argv or sys.argv[1:]), args.root, log_file)

    try:
        collect(args.tickers, args.start, args.end, args.root, force=args.force, dry_run=args.dry_run)
    except LimitReached as exc:
        logging.warning("%s. Completed months are saved; run the same command again tomorrow.", exc)
        return 2
    except NewsDataError as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
