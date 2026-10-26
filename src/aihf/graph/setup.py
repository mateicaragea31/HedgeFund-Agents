"""The whole TradingAgents graph, built from its three sections.

Follows upstream `graph/setup.py` (setup_graph, commit 35543d0):

    START → analysts (in the order given) → Bull ⇄ Bear → Research Manager
          → Trader → Aggressive → Conservative → Neutral → Portfolio Manager → END

Each section is added back to front, so every section knows the name of the node it hands
over to.
"""

from collections.abc import Sequence

from langgraph.graph import END, StateGraph

from aihf.graph.analysts import add_analysts
from aihf.graph.debate import add_research_debate
from aihf.graph.risk import add_trader_and_risk_debate
from aihf.graph.state import AgentState
from aihf.tools import DataSource

DEFAULT_ANALYSTS = ("market", "social", "news", "fundamentals")  # upstream's default and order


def build_graph(
    quick_llm,
    deep_llm,
    data: DataSource,
    analysts: Sequence[str] = DEFAULT_ANALYSTS,
    max_debate_rounds: int = 1,
    max_risk_discuss_rounds: int = 1,
):
    """The compiled graph, ready for `.invoke(state)` (see graph/cell.py).

    quick_llm: analysts, Bull/Bear, Trader, risk debaters. deep_llm: Research Manager and
    Portfolio Manager (upstream's split; CLAUDE.md §2.2 item 6). They may be the same object.
    """
    workflow = StateGraph(AgentState)
    trader = add_trader_and_risk_debate(workflow, quick_llm, deep_llm, max_risk_discuss_rounds,
                                        next_node=END)
    bull = add_research_debate(workflow, quick_llm, deep_llm, max_debate_rounds, next_node=trader)
    add_analysts(workflow, analysts, quick_llm, data, next_node=bull)
    return workflow.compile()
