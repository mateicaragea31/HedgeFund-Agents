"""Download each company's SEC filings once, check them, and save them to the SEC archive.

Runs on the FEP login node, never inside a backtest. For each ticker (CIK from
configs/instruments.yaml):
  1. the filing list (submissions API, all pages), with each filing's acceptance time;
  2. company facts (every XBRL number, each tagged with its filing's accession number);
  3. the Form 4 filings in [form4_start, form4_end] (by filing date), parsed into trades;
  4. checks (see `check_filings`), then <root>/sec/<TICKER>/... (format: data/sec.py).

SEC needs no key but requires a User-Agent naming a person and an email
(SEC_EDGAR_USER_AGENT, set in ~/.bashrc). Requests are spaced to stay well under SEC's limit
of 10 per second.
"""

import hashlib
import io
import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import certifi
import pandas as pd

from aihf.data.pit import NEW_YORK
from aihf.data.sec import FILING_COLUMNS, FORM4_COLUMNS, sec_dir

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
REQUEST_GAP_SECONDS = 0.2  # at most 5 requests per second
# certifi's certificate list, so HTTPS works the same on the Mac and in the slim container
# (a Mac python.org build ships without system certificates).
_TLS = ssl.create_default_context(cafile=certifi.where())

# Fetches a URL, returns its body. Swapped for a fake in tests.
Getter = Callable[[str], bytes]


class SecDataError(Exception):
    """The downloaded data failed a check; nothing was saved."""


def user_agent() -> str:
    agent = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if "@" not in agent:
        raise SecDataError("set SEC_EDGAR_USER_AGENT to your name and email (SEC requires it), "
                           'e.g. export SEC_EDGAR_USER_AGENT="Jane Doe jane@example.com"')
    return agent


def http_get(url: str) -> bytes:
    """GET with SEC's User-Agent, spaced requests and 3 tries."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent(), "Accept-Encoding": "identity"})
    for attempt in range(3):
        time.sleep(REQUEST_GAP_SECONDS)
        try:
            with urllib.request.urlopen(request, timeout=60, context=_TLS) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            problem = exc
            time.sleep(5 * (attempt + 1))  # SEC asks clients to back off after a refusal
    raise SecDataError(f"could not download {url}: {problem}")


def _filings_frame(block: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "accession": block["accessionNumber"],
        "form": block["form"],
        "filing_date": block["filingDate"],
        "report_date": block["reportDate"],
        "accepted_at": block["acceptanceDateTime"],
        "primary_document": block["primaryDocument"],
    })


def fetch_filings(cik: int, get: Getter) -> pd.DataFrame:
    """Every filing of the company, with accepted_at as UTC (SEC writes it with a 'Z')."""
    first = json.loads(get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json")))
    frames = [_filings_frame(first["filings"]["recent"])]
    for page in first["filings"].get("files", []):
        frames.append(_filings_frame(json.loads(get(SUBMISSIONS_URL.format(name=page["name"])))))
    filings = pd.concat(frames, ignore_index=True).drop_duplicates("accession")
    filings["accepted_at"] = pd.to_datetime(filings["accepted_at"], utc=True)
    return filings[FILING_COLUMNS].sort_values("accepted_at").reset_index(drop=True)


def _text(node: ET.Element | None, path: str) -> str | None:
    found = node.find(path) if node is not None else None
    return found.text.strip() if found is not None and found.text and found.text.strip() else None


def _number(node: ET.Element, path: str) -> float:
    value = _text(node, path)
    try:
        return float(value) if value is not None else float("nan")
    except ValueError:
        return float("nan")


def parse_form4(xml: bytes) -> list[dict]:
    """The non-derivative trades of one Form 4 (shares bought, sold, granted, ...).

    Derivative rows (options, units) are left out: the shares they turn into appear as
    non-derivative rows when exercised (code M).
    """
    root = ET.fromstring(xml)
    owner = root.find("reportingOwner")
    relation = owner.find("reportingOwnerRelationship") if owner is not None else None
    roles = [_text(relation, "officerTitle")]
    if _text(relation, "isDirector") in ("1", "true"):
        roles.append("Director")
    if _text(relation, "isTenPercentOwner") in ("1", "true"):
        roles.append("10% Owner")
    position = "; ".join(r for r in roles if r) or (_text(relation, "otherText") or "")
    insider = _text(owner, "reportingOwnerId/rptOwnerName") or "unknown"

    trades = []
    for t in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        trades.append({
            "insider": insider,
            "position": position,
            "transaction_date": _text(t, "transactionDate/value"),
            "code": _text(t, "transactionCoding/transactionCode") or "",
            "shares": _number(t, "transactionAmounts/transactionShares/value"),
            "price": _number(t, "transactionAmounts/transactionPricePerShare/value"),
            "acquired_disposed": _text(t, "transactionAmounts/transactionAcquiredDisposedCode/value") or "",
            "owned_after": _number(t, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
            "ownership": _text(t, "ownershipNature/directOrIndirectOwnership/value") or "",
        })
    return trades


def fetch_form4(cik: int, filings: pd.DataFrame, start: str, end: str, get: Getter) -> pd.DataFrame:
    """Trades from every Form 4 (and amendment) filed in [start, end]."""
    wanted = filings[filings["form"].isin(["4", "4/A"])
                     & (filings["filing_date"] >= start) & (filings["filing_date"] <= end)]
    rows = []
    for f in wanted.itertuples():
        document = f.primary_document.split("/")[-1]  # "xslF345X05/form4.xml" is the rendered view
        url = ARCHIVE_URL.format(cik=cik, accession=f.accession.replace("-", ""), document=document)
        for trade in parse_form4(get(url)):
            rows.append({"accession": f.accession, "accepted_at": f.accepted_at, **trade})
    trades = pd.DataFrame(rows, columns=FORM4_COLUMNS)
    trades["transaction_date"] = pd.to_datetime(trades["transaction_date"])
    return trades


def check_filings(filings: pd.DataFrame, ticker: str) -> list[str]:
    """Problems that mean the filing list can't be trusted. Empty list = all good.

    The clock check: SEC dates a filing accepted after 17:30 New York on the next business day.
    So a filing whose date is later than its acceptance day must have been accepted late (or
    on a weekend or holiday). If accepted_at were New York time mislabelled as UTC, many
    after-close filings would appear accepted in the afternoon and break this.
    """
    problems = []
    if not filings["accession"].is_unique:
        problems.append("duplicate accession numbers")
    if filings["accepted_at"].isna().any():
        problems.append("filings without an acceptance time")
    for form in ("10-K", "10-Q"):
        if not (filings["form"] == form).any():
            problems.append(f"no {form} filings")
    local = filings["accepted_at"].dt.tz_convert(NEW_YORK)
    later = filings["filing_date"] > local.dt.strftime("%Y-%m-%d")
    weekday = local.dt.dayofweek < 5
    early = (local.dt.hour * 60 + local.dt.minute) < 17 * 60 + 25
    odd = filings[later & weekday & early]
    if len(odd) > max(5, 0.02 * len(filings)):  # a few federal holidays are expected
        problems.append(f"{len(odd)} filings dated after an early acceptance: is accepted_at really UTC?")
    return problems


def _write(path: Path, data: bytes) -> str:
    """Write via a temporary file and rename; returns the sha256."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return hashlib.sha256(data).hexdigest()


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def collect(
    instruments: dict, tickers: list[str], root: Path, *, form4_start: str, form4_end: str,
    force: bool = False, dry_run: bool = False, get: Getter = http_get,
) -> None:
    """Collect every ticker; raises SecDataError at the end if any ticker failed."""
    failed = []
    for ticker in (t.upper() for t in tickers):
        folder = sec_dir(root, ticker)
        cik = (instruments.get(ticker) or {}).get("cik")
        if cik is None:
            logger.error("%s: no cik in configs/instruments.yaml", ticker)
            failed.append(ticker)
            continue
        if (folder / "meta.json").exists() and not force:
            logger.info("%s: already collected (%s); use --force to download again", ticker, folder)
            continue
        if dry_run:
            logger.info("%s: would download filings, company facts and Form 4s %s..%s (CIK %d) into %s",
                        ticker, form4_start, form4_end, cik, folder)
            continue
        try:
            fetched_at = datetime.now(UTC)
            filings = fetch_filings(cik, get)
            problems = check_filings(filings, ticker)
            if problems:
                raise SecDataError("; ".join(problems))
            facts = get(FACTS_URL.format(cik=cik))
            if not json.loads(facts).get("facts", {}).get("us-gaap"):
                raise SecDataError("company facts have no us-gaap section")
            trades = fetch_form4(cik, filings, form4_start, form4_end, get)
        except SecDataError as exc:
            logger.error("%s: %s; nothing saved", ticker, exc)
            failed.append(ticker)
            continue

        folder.mkdir(parents=True, exist_ok=True)
        hashes = {
            "companyfacts.json": _write(folder / "companyfacts.json", facts),
            "filings.parquet": _write(folder / "filings.parquet", _parquet_bytes(filings)),
            "form4.parquet": _write(folder / "form4.parquet", _parquet_bytes(trades)),
        }
        meta = {
            "source": "SEC EDGAR (submissions API, XBRL company facts, Form 4 XML)",
            "ticker": ticker, "cik": cik,
            "fetched_at": fetched_at.isoformat(timespec="seconds"),
            "form4_start": form4_start, "form4_end": form4_end,
            "filings": len(filings), "first_filing": f"{filings['accepted_at'].min():%Y-%m-%d}",
            "form4_filings": int(trades["accession"].nunique()), "form4_trades": len(trades),
            "sha256": hashes,
        }
        _write(folder / "meta.json", json.dumps(meta, indent=2).encode())
        logger.info("%s: saved %d filings, company facts, %d Form 4 trades from %d filings into %s",
                    ticker, len(filings), len(trades), meta["form4_filings"], folder)
    if failed:
        raise SecDataError(f"failed: {', '.join(failed)}")

