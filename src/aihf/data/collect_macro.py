"""Download FRED series with all their vintages once into the macro archive.

Runs on the FEP login node, never inside a backtest; needs FRED_API_KEY (free). For each series
behind upstream's macro aliases (tools/macro.py MACRO_SERIES): its description, and every
observation from `obs_start` with every version published between `vintage_start` and
`vintage_end` (FRED's realtime period). Two requests per series; FRED allows 120 a minute.
"""

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

import certifi
import pandas as pd

from aihf.data.macro import OBS_COLUMNS, series_file

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org/fred"
REQUEST_GAP_SECONDS = 0.6
_TLS = ssl.create_default_context(cafile=certifi.where())

# Gets a FRED endpoint path ("series", "series/observations") with parameters → decoded JSON.
Getter = Callable[[str, dict], dict]


class MacroDataError(Exception):
    """A FRED request failed; nothing was saved for that series."""


def fred_get(path: str, params: dict) -> dict:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise MacroDataError("set FRED_API_KEY in ~/.bashrc (free key: fred.stlouisfed.org)")
    query = urllib.parse.urlencode({**params, "api_key": key, "file_type": "json"})
    problem = None
    for attempt in range(4):
        time.sleep(REQUEST_GAP_SECONDS)
        try:
            with urllib.request.urlopen(f"{FRED_API_BASE}/{path}?{query}", timeout=120, context=_TLS) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code == 400:  # unknown series or bad parameter: retrying will not help
                raise MacroDataError(f"FRED {path} {params.get('series_id')}: {body[:300]}") from exc
            problem = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            problem = exc
        time.sleep(5 * (attempt + 1))
    raise MacroDataError(f"FRED {path} kept failing: {problem}")


def fetch_series(series_id: str, obs_start: str, obs_end: str, vintage_start: str, vintage_end: str,
                 get: Getter) -> tuple[dict, pd.DataFrame]:
    """(description, observations with their vintages) for one series."""
    realtime = {"realtime_start": vintage_start, "realtime_end": vintage_end}
    seriess = get("series", {"series_id": series_id, **realtime}).get("seriess") or []
    if not seriess:
        raise MacroDataError(f"FRED has no series {series_id}")
    info = seriess[-1]  # the description valid at the end of the range
    rows = get("series/observations", {"series_id": series_id, "observation_start": obs_start,
                                       "observation_end": obs_end, "sort_order": "asc", **realtime})
    obs = pd.DataFrame(rows.get("observations", []), columns=OBS_COLUMNS).astype(str)
    if not len(obs):
        raise MacroDataError(f"FRED returned no observations for {series_id}")
    meta = {"title": info.get("title", series_id), "units": info.get("units_short") or info.get("units", ""),
            "frequency": info.get("frequency", ""), "seasonal": info.get("seasonal_adjustment_short", "")}
    return meta, obs


def collect(series_ids, root, *, obs_start: str, obs_end: str, vintage_start: str, vintage_end: str,
            force: bool = False, dry_run: bool = False, get: Getter = fred_get) -> None:
    """Every series; raises MacroDataError at the end if any failed (the others are saved)."""
    failed = []
    for series_id in sorted({s.upper() for s in series_ids}):
        path = series_file(root, series_id)
        if path.exists() and not force:
            logger.info("%s: already collected", series_id)
            continue
        if dry_run:
            logger.info("%s: would download observations %s..%s with vintages %s..%s",
                        series_id, obs_start, obs_end, vintage_start, vintage_end)
            continue
        try:
            meta, obs = fetch_series(series_id, obs_start, obs_end, vintage_start, vintage_end, get)
        except MacroDataError as exc:
            logger.error("%s: %s", series_id, exc)
            failed.append(series_id)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        obs.to_parquet(tmp, index=False)
        tmp.replace(path)
        meta.update(source="FRED/ALFRED API (api.stlouisfed.org), series + series/observations",
                    series_id=series_id, fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    observation_start=obs_start, observation_end=obs_end,
                    realtime_start=vintage_start, realtime_end=vintage_end, rows=len(obs))
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("%s: saved %d rows (%s)", series_id, len(obs), meta["title"])
    if failed:
        raise MacroDataError(f"failed: {', '.join(failed)}")
