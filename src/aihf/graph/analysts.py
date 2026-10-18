"""The analyst section of the graph: which analysts run, in what order, and how they connect.

Follows upstream `graph/analyst_execution.py` and the analyst part of `graph/setup.py`
(commit 35543d0). For each selected analyst, in order:

    Analyst ──(asked for tools)──> tools_<key> ──> back to Analyst      (tool loop)
       └────(wrote its report)───> Msg Clear <Name> ──> next analyst, or `next_node`

The Sentiment Analyst has no tools, so it goes straight to its Msg Clear node.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from aihf.agents.analysts import (
    FUNDAMENTALS_TOOLS,
    MARKET_TOOLS,
    NEWS_TOOLS,
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_sentiment_analyst,
)
from aihf.agents.context import create_msg_delete
from aihf.tools import DataSource, build_tools


@dataclass(frozen=True)
class AnalystSpec:
    key: str  # the name used in configs: market, social, news, fundamentals
    agent_node: str
    clear_node: str
    report_key: str
    tool_names: tuple[str, ...] = ()

    @property
    def tool_node(self) -> str | None:
        """The node that runs this analyst's tool calls; None when it has no tools."""
        return f"tools_{self.key}" if self.tool_names else None


# Upstream's node names, so our saved states and call logs read the same.
ANALYST_SPECS: dict[str, AnalystSpec] = {
    "market": AnalystSpec("market", "Market Analyst", "Msg Clear Market", "market_report",
                          MARKET_TOOLS),
    # "social" in configs, like upstream; it fetches its sources itself, so it has no tools.
    "social": AnalystSpec("social", "Sentiment Analyst", "Msg Clear Sentiment",
                          "sentiment_report"),
    "news": AnalystSpec("news", "News Analyst", "Msg Clear News", "news_report", NEWS_TOOLS),
    "fundamentals": AnalystSpec("fundamentals", "Fundamentals Analyst",
                                "Msg Clear Fundamentals", "fundamentals_report",
                                FUNDAMENTALS_TOOLS),
}


def _tools_or_clear(spec: AnalystSpec):
    """The conditional edge after an analyst: run its tool calls, or finish its report."""

    def route(state) -> str:
        return spec.tool_node if state["messages"][-1].tool_calls else spec.clear_node

    return route


def add_analysts(
    workflow: StateGraph, selected: Iterable[str], llm, data: DataSource, next_node: str
) -> list[AnalystSpec]:
    """Add the selected analysts to `workflow`, from START up to `next_node`.

    `selected` keeps its order, e.g. ["market", "social", "news", "fundamentals"].
    `next_node` is where the last analyst hands over (the Bull Researcher, or END in tests).
    Returns the specs used, in order.
    """
    specs = []
    for key in selected:
        if key not in ANALYST_SPECS:
            raise ValueError(f"unknown analyst {key!r}; choose from {list(ANALYST_SPECS)}")
        specs.append(ANALYST_SPECS[key])
    if not specs:
        raise ValueError("at least one analyst must be selected")

    tools = build_tools(data)
    factories = {
        "market": lambda: create_market_analyst(llm, tools),
        "social": lambda: create_sentiment_analyst(llm, data),
        "news": lambda: create_news_analyst(llm, tools),
        "fundamentals": lambda: create_fundamentals_analyst(llm, tools),
    }

    workflow.add_edge(START, specs[0].agent_node)
    for i, spec in enumerate(specs):
        workflow.add_node(spec.agent_node, factories[spec.key]())
        workflow.add_node(spec.clear_node, create_msg_delete())
        if spec.tool_names:
            workflow.add_node(spec.tool_node, ToolNode([tools[n] for n in spec.tool_names]))
            workflow.add_conditional_edges(
                spec.agent_node, _tools_or_clear(spec), [spec.tool_node, spec.clear_node]
            )
            workflow.add_edge(spec.tool_node, spec.agent_node)
        else:
            workflow.add_edge(spec.agent_node, spec.clear_node)
        following = specs[i + 1].agent_node if i + 1 < len(specs) else next_node
        workflow.add_edge(spec.clear_node, following)
    return specs
