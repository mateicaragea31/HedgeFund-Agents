"""Running one cell (one ticker on one trade date) through the graph.

Follows upstream's TradingAgentsGraph.propagate (commit 35543d0): build the starting state, run
the graph, read the 5-tier rating from the Portfolio Manager's decision.
"""

from aihf.agents.rating import parse_rating
from aihf.graph.state import initial_state

# Upstream's max_recur_limit: the most graph steps one run may take. It stops a model that
# keeps asking for tools forever; a normal 4-analyst run takes well under half of it.
RECURSION_LIMIT = 100


def run_cell(
    graph,
    ticker: str,
    trade_date: str,
    *,
    instrument_context: str,
    past_context: str = "",
    recursion_limit: int = RECURSION_LIMIT,
) -> tuple[dict, str]:
    """Return (final state, rating). The rating is Buy ... Sell, or REVIEW if none can be read."""
    state = initial_state(ticker, trade_date, instrument_context=instrument_context,
                          past_context=past_context)
    final_state = graph.invoke(state, config={"recursion_limit": recursion_limit})
    return final_state, parse_rating(final_state.get("final_trade_decision", ""))
