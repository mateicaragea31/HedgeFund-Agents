"""Download SEC filings (filing list, XBRL company facts, Form 4 insider trades) into the archive.

Run on the FEP login node (about a minute per ticker); needs SEC_EDGAR_USER_AGENT="Name email":
    python -m aihf.scripts.collect_sec --tickers AAPL GOOGL AMZN --dry-run
    python -m aihf.scripts.collect_sec --tickers AAPL GOOGL AMZN

Writes $AIHF_DATA/archive/sec/<TICKER>/ (format: src/aihf/data/sec.py). Tickers already saved
are skipped unless --force. Exits with code 1 if any ticker fails (nothing saved for it).
A log is written to logs/collect_sec-<time>.log.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from aihf.agents.context import load_instruments
from aihf.data.collect_sec import SecDataError, collect, user_agent

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL GOOGL AMZN (CIKs in configs/instruments.yaml)")
    p.add_argument("--form4-start", default="2023-01-01", help="first Form 4 filing date to archive")
    p.add_argument("--form4-end", default="2024-06-28", help="last Form 4 filing date to archive")
    p.add_argument("--root", type=Path, help="archive folder (default: $AIHF_DATA/archive)")
    p.add_argument("--force", action="store_true", help="download again even if the folder exists")
    p.add_argument("--dry-run", action="store_true", help="only print what would be done")
    args = p.parse_args(argv)

    if args.root is None:
        if not os.environ.get("AIHF_DATA"):
            p.error("set AIHF_DATA (e.g. in ~/.bashrc) or pass --root")
        args.root = Path(os.environ["AIHF_DATA"]) / "archive"
    try:
        user_agent()
    except SecDataError as exc:
        p.error(str(exc))

    (REPO / "logs").mkdir(exist_ok=True)
    log_file = REPO / "logs" / f"collect_sec-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logging.info("collect_sec %s -> %s (log: %s)", " ".join(argv or sys.argv[1:]), args.root, log_file)

    try:
        collect(load_instruments(), args.tickers, args.root, form4_start=args.form4_start,
                form4_end=args.form4_end, force=args.force, dry_run=args.dry_run)
    except SecDataError as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
