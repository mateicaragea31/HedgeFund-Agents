"""The research debate section of the graph: Bull ⇄ Bear for n rounds, then the Research Manager.

Follows upstream `graph/conditional_logic.py` (should_continue_debate) and the debate part of
`graph/setup.py` (commit 35543d0). With max_debate_rounds = 1 (the default):

    Bull Researcher → Bear Researcher → Research Manager → next_node

One round = one speech from each side, so the debate stops when the counter reaches
2 × max_debate_rounds.
"""

from langgraph.graph import StateGraph

from aihf.agents.researchers import (
    create_bear_researcher,
    create_bull_researcher,
    create_research_manager,
)

BULL, BEAR, JUDGE = "Bull Researcher", "Bear Researcher", "Research Manager"

# Every place the "who speaks next?" rule can send the graph. Both debaters' edges list all of
# them, so an unexpected answer can never hit a missing edge (upstream's DEBATE_PATH_MAP).
DEBATE_PATH_MAP = {BULL: BULL, BEAR: BEAR, JUDGE: JUDGE}


def next_debate_speaker(max_debate_rounds: int):
    """The rule after each speech: the other side speaks, or the judge if the rounds are used up."""

    def route(state) -> str:
        debate = state["investment_debate_state"]
        if debate["count"] >= 2 * max_debate_rounds:
            return JUDGE
        if debate["current_response"].startswith("Bull"):
            return BEAR
        return BULL

    return route


def add_research_debate(
    workflow: StateGraph, quick_llm, deep_llm, max_debate_rounds: int, next_node: str
) -> str:
    """Add Bull, Bear (quick model) and the Research Manager (deep model) to `workflow`.

    Returns the name of the first node (the Bull), for the analysts to hand over to.
    The Research Manager hands over to `next_node` (the Trader, or END in tests).
    """
    if max_debate_rounds < 1:
        raise ValueError("max_debate_rounds must be at least 1")
    workflow.add_node(BULL, create_bull_researcher(quick_llm))
    workflow.add_node(BEAR, create_bear_researcher(quick_llm))
    workflow.add_node(JUDGE, create_research_manager(deep_llm))
    route = next_debate_speaker(max_debate_rounds)
    for debater in (BULL, BEAR):
        workflow.add_conditional_edges(debater, route, DEBATE_PATH_MAP)
    workflow.add_edge(JUDGE, next_node)
    return BULL
