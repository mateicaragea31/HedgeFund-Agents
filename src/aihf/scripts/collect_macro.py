"""Download the FRED series behind upstream's macro aliases, with all their vintages.

Run on the FEP login node (about a minute); needs FRED_API_KEY in ~/.bashrc:
    python -m aihf.scripts.collect_macro --dry-run
    python -m aihf.scripts.collect_macro

Writes $AIHF_DATA/archive/macro/fred/<SERIES>.parquet (format: src/aihf/data/macro.py).
Observations from 2019 (so a several-year look-back still works), versions published from
2023-12-01 to 2024-03-31 (the trade dates of the paper window). Series already saved are
skipped unless --force. Exits with code 1 if any series fails. Log: logs/collect_macro-<time>.log.
"""

import argparse
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from aihf.data.collect_macro import MacroDataError, collect
from aihf.tools.macro import MACRO_SERIES

REPO = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--series", nargs="+", default=sorted(set(MACRO_SERIES.values())),
                   help="FRED series IDs (default: every series behind upstream's aliases)")
    p.add_argument("--obs-start", default="2019-01-01", help="first observation date")
    p.add_argument("--obs-end", default="2024-03-31", help="last observation date")
    p.add_argument("--vintage-start", default="2023-12-01", help="first vintage (realtime) date")
    p.add_argument("--vintage-end", default="2024-03-31", help="last vintage (realtime) date")
    p.add_argument("--root", type=Path, help="archive folder (default: $AIHF_DATA/archive)")
    p.add_argument("--force", action="store_true", help="download again even if saved")
    p.add_argument("--dry-run", action="store_true", help="only print what would be done")
    args = p.parse_args(argv)

    if args.root is None:
        if not os.environ.get("AIHF_DATA"):
            p.error("set AIHF_DATA (e.g. in ~/.bashrc) or pass --root")
        args.root = Path(os.environ["AIHF_DATA"]) / "archive"

    (REPO / "logs").mkdir(exist_ok=True)
    log_file = REPO / "logs" / f"collect_macro-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logging.info("collect_macro %s -> %s (log: %s)", " ".join(argv or sys.argv[1:]), args.root, log_file)

    try:
        collect(args.series, args.root, obs_start=args.obs_start, obs_end=args.obs_end,
                vintage_start=args.vintage_start, vintage_end=args.vintage_end,
                force=args.force, dry_run=args.dry_run)
    except MacroDataError as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
