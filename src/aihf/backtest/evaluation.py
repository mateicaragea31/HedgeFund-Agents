"""Evaluate backtest runs: their ratings through the simulator, next to the baselines (CLAUDE.md §6.4).

For each run (one repeat) and ticker: ratings → targets (mapping) → simulate → metrics +
bootstrap intervals. The baselines go through the same simulator with the same fills and costs.
The table has one row per (ticker, strategy): for the agents, the mean and standard deviation
over the repeats; for the deterministic baselines, their single value.
"""

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from aihf.backtest.baselines import baseline_targets
from aihf.backtest.bootstrap import bootstrap_ci
from aihf.backtest.metrics import daily_risk_free, evaluate, nan_to_none
from aihf.backtest.runner import cell_dir, read_report
from aihf.backtest.simulator import DEFAULT, SimConfig, simulate, targets_from_ratings
from aihf.data.macro import series_file
from aihf.data.prices import price_file

logger = logging.getLogger(__name__)
METRICS = ["CR", "AR", "Sharpe", "MDD"]


def run_ratings(run_dir: Path, ticker: str, dates: list[str]) -> dict[str, str]:
    """{date: rating} of one run's cells; a cell without a rating counts as REVIEW."""
    out = {}
    for day in dates:
        report = read_report(cell_dir(run_dir, ticker, day)) or {}
        out[day] = report.get("rating") or "REVIEW"
    return out


def load_bars(archive: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(price_file(archive, ticker), columns=["date", "open", "high", "low", "close", "adj_close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def risk_free(archive: Path, dates: list[str]) -> pd.Series:
    """Daily risk-free return per trade date from FRED DTB3 (latest known values), or zeros."""
    path = series_file(archive, "DTB3")
    if not path.exists():
        logger.warning("no DTB3 in the archive: Sharpe uses a zero risk-free rate")
        return pd.Series(0.0, index=dates)
    df = pd.read_parquet(path)
    df = df[~df["value"].isin([".", ""])].sort_values("realtime_start").drop_duplicates("date", keep="last")
    rates = df.set_index("date")["value"].astype(float).sort_index()
    rates = rates.reindex(sorted(set(rates.index) | set(dates))).ffill().reindex(dates)
    return daily_risk_free(rates.fillna(0.0))


def _one(sim: pd.DataFrame, cfg: SimConfig, rf: pd.Series, seed: int) -> dict:
    rf_series = pd.Series(rf.to_numpy(), index=sim.index)
    m = evaluate(sim, cfg.initial_cash, rf_series)
    m.update(bootstrap_ci(sim["daily_return"], float(rf.mean()), seed=seed))
    return nan_to_none(m)


def evaluate_runs(runs: dict[str, Path], tickers: list[str], dates: list[str], archive: Path, *,
                  mapping: str = "paper3", config: SimConfig = DEFAULT) -> dict:
    """Per-run, per-ticker metrics for the agents and the baselines, plus the curves."""
    rf = risk_free(archive, dates)
    rows, curves, review = [], [], Counter()
    for ticker in tickers:
        bars = load_bars(archive, ticker)
        for run_id, run_dir in runs.items():
            ratings = run_ratings(run_dir, ticker, dates)
            review[run_id] += sum(1 for d, r in ratings.items() if r == "REVIEW" and d != dates[-1])
            sim = simulate(bars, targets_from_ratings(ratings, mapping), dates, config)
            rows.append({"ticker": ticker, "strategy": "Agents", "run": run_id, **_one(sim, config, rf, 0)})
            curves.append(sim.assign(ticker=ticker, strategy="Agents", run=run_id))
        for name, targets in baseline_targets(bars, dates).items():
            sim = simulate(bars, targets, dates, config)
            rows.append({"ticker": ticker, "strategy": name, "run": "", **_one(sim, config, rf, 0)})
            curves.append(sim.assign(ticker=ticker, strategy=name, run=""))
    per_run = pd.DataFrame(rows)
    return {"per_run": per_run, "table": table(per_run), "curves": pd.concat(curves, ignore_index=True),
            "review_counts": dict(review)}


def table(per_run: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, strategy): mean and std of the metrics over the repeats."""
    out = []
    for (ticker, strategy), g in per_run.groupby(["ticker", "strategy"], sort=False):
        row = {"ticker": ticker, "strategy": strategy, "repeats": len(g)}
        for m in METRICS:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[m] = round(float(vals.mean()), 4)
            row[f"{m}_std"] = round(float(vals.std(ddof=1)), 4) if len(g) > 1 else None
        out.append(row)
    return pd.DataFrame(out)


def save(result: dict, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result["per_run"].to_csv(out_dir / "per_run.csv", index=False)
    result["table"].to_csv(out_dir / "table.csv", index=False)
    result["curves"].to_csv(out_dir / "curves.csv", index=False)
    (out_dir / "evaluation.json").write_text(json.dumps({**meta, "review_counts": result["review_counts"]},
                                                         indent=2, default=str))
