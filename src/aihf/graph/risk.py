"""The decision section of the graph: Trader, then the risk debate, then the Portfolio Manager.

Follows upstream `graph/conditional_logic.py` (should_continue_risk_analysis) and the end of
`graph/setup.py` (commit 35543d0). With max_risk_discuss_rounds = 1 (the default):

    Trader → Aggressive → Conservative → Neutral → Portfolio Manager → next_node

One round = one speech from each of the three, so the debate stops when the counter reaches
3 × max_risk_discuss_rounds.
"""

from langgraph.graph import StateGraph

from aihf.agents.risk import (
    create_aggressive_debator,
    create_conservative_debator,
    create_neutral_debator,
    create_portfolio_manager,
)
from aihf.agents.trader import create_trader

TRADER = "Trader"
AGGRESSIVE, CONSERVATIVE, NEUTRAL = "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"
JUDGE = "Portfolio Manager"

# Every place the "who speaks next?" rule can send the graph (upstream's RISK_ANALYSIS_PATH_MAP).
RISK_PATH_MAP = {AGGRESSIVE: AGGRESSIVE, CONSERVATIVE: CONSERVATIVE, NEUTRAL: NEUTRAL, JUDGE: JUDGE}


def next_risk_speaker(max_risk_discuss_rounds: int):
    """The rule after each speech: Aggressive → Conservative → Neutral → Aggressive ..., or the judge."""

    def route(state) -> str:
        debate = state["risk_debate_state"]
        if debate["count"] >= 3 * max_risk_discuss_rounds:
            return JUDGE
        if debate["latest_speaker"].startswith("Aggressive"):
            return CONSERVATIVE
        if debate["latest_speaker"].startswith("Conservative"):
            return NEUTRAL
        return AGGRESSIVE

    return route


def add_trader_and_risk_debate(
    workflow: StateGraph, quick_llm, deep_llm, max_risk_discuss_rounds: int, next_node: str
) -> str:
    """Add the Trader and the risk debaters (quick model) and the Portfolio Manager (deep model).

    Returns the name of the first node (the Trader), for the Research Manager to hand over to.
    The Portfolio Manager hands over to `next_node` (END in the full graph).
    """
    if max_risk_discuss_rounds < 1:
        raise ValueError("max_risk_discuss_rounds must be at least 1")
    workflow.add_node(TRADER, create_trader(quick_llm))
    workflow.add_node(AGGRESSIVE, create_aggressive_debator(quick_llm))
    workflow.add_node(CONSERVATIVE, create_conservative_debator(quick_llm))
    workflow.add_node(NEUTRAL, create_neutral_debator(quick_llm))
    workflow.add_node(JUDGE, create_portfolio_manager(deep_llm))
    workflow.add_edge(TRADER, AGGRESSIVE)
    route = next_risk_speaker(max_risk_discuss_rounds)
    for debater in (AGGRESSIVE, CONSERVATIVE, NEUTRAL):
        workflow.add_conditional_edges(debater, route, RISK_PATH_MAP)
    workflow.add_edge(JUDGE, next_node)
    return TRADER
