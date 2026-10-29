"""Download every post of r/wallstreetbets, r/stocks and r/investing, month by month (Arctic Shift).

Run on the FEP login node (about 20 minutes for Dec 2023 - Mar 2024; no key needed):
    python -m aihf.scripts.collect_reddit --dry-run
    python -m aihf.scripts.collect_reddit

Writes $AIHF_DATA/archive/reddit/<subreddit>/<YYYY-MM>.parquet (format: src/aihf/data/reddit.py).
Completed months are skipped, so an interrupted run resumes where it stopped. Exits with
code 1 if a month cannot be downloaded. A log is written to logs/collect_reddit-<time>.log.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from aihf.data.collect_reddit import RedditDataError, collect
from aihf.data.reddit import SUBREDDITS

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subreddits", nargs="+", default=list(SUBREDDITS), help="default: upstream's three")
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
    log_file = REPO / "logs" / f"collect_reddit-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logging.info("collect_reddit %s -> %s (log: %s)", " ".join(argv or sys.argv[1:]), args.root, log_file)

    try:
        collect(args.subreddits, args.start, args.end, args.root, force=args.force, dry_run=args.dry_run)
    except RedditDataError as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
