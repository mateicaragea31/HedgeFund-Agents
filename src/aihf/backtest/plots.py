"""Cumulative-return plots in the style of the paper's Fig. 7 (CLAUDE.md §6.5).

Input: the curves of an evaluation (curves.csv: one row per date × ticker × strategy × run, with
the account value at each close). Each curve is divided by its first value, so every line starts
at 1.00, as in the paper. The agents have one curve per repeat: the plot draws their mean as a
bold line and the range between the lowest and highest repeat as a shaded band. The baselines are
deterministic and get one line each.

matplotlib is imported inside the functions: it is only needed by the plotting script
(`pip install -e .[plots]`), not by the backtest jobs.
"""

import pandas as pd

AGENTS = "Agents"
BASELINES = ["Buy&Hold", "MACD", "KDJ&RSI", "ZMR", "SMA"]
COLORS = {"Buy&Hold": "tab:blue", "MACD": "tab:orange", "KDJ&RSI": "tab:green", "ZMR": "tab:red",
          "SMA": "tab:purple", AGENTS: "black"}


def growth(curves: pd.DataFrame) -> pd.DataFrame:
    """Rows: dates; columns: (strategy, run); values: account value / first value."""
    df = curves.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Baselines have no run id; "" as a column label would make pandas return a Series below.
    df["run"] = df["run"].fillna("-").replace("", "-").astype(str)
    wide = df.pivot_table(index="date", columns=["strategy", "run"], values="value", aggfunc="first")
    return wide / wide.iloc[0]


def agents_band(g: pd.DataFrame) -> pd.DataFrame:
    """Mean, min and max over the agents' repeats, per date."""
    runs = g[AGENTS]
    return pd.DataFrame({"mean": runs.mean(axis=1), "low": runs.min(axis=1), "high": runs.max(axis=1)})


def draw_ticker(ax, curves: pd.DataFrame, ticker: str) -> None:
    """One panel: the baselines and the agents for one ticker."""
    g = growth(curves[curves["ticker"] == ticker])
    strategies = g.columns.get_level_values("strategy")
    for name in BASELINES:
        if name in strategies:
            ax.plot(g.index, g[name].iloc[:, 0], color=COLORS[name], lw=1.2, label=name)
    if AGENTS in strategies:
        band = agents_band(g)
        n = g[AGENTS].shape[1]
        ax.fill_between(band.index, band["low"], band["high"], color=COLORS[AGENTS], alpha=0.15, lw=0,
                        label=f"Agents: range of {n} repeats" if n > 1 else None)
        ax.plot(band.index, band["mean"], color=COLORS[AGENTS], lw=2.2,
                label=f"Agents (mean of {n})" if n > 1 else "Agents")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_title(f"Cumulative returns for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value / initial value")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", labelrotation=30)


def plot_ticker(curves: pd.DataFrame, ticker: str, path, subtitle: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")  # no screen on FEP
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    draw_ticker(ax, curves, ticker)
    ax.legend(loc="upper left", fontsize=8)
    if subtitle:
        fig.text(0.99, 0.01, subtitle, ha="right", va="bottom", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(curves: pd.DataFrame, tickers: list[str], path, subtitle: str = "") -> None:
    """All tickers side by side, one shared legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tickers), figsize=(6 * len(tickers), 5), squeeze=False)
    for ax, ticker in zip(axes[0], tickers, strict=True):
        draw_ticker(ax, curves, ticker)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8, frameon=False)
    if subtitle:
        fig.text(0.99, 0.01, subtitle, ha="right", va="bottom", fontsize=7, color="grey")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
