"""[TEACHING] A tiny TradingAgents in LangGraph (CLAUDE.md Phase 0.8). Not used for results.

    Analyst (+ 1 tool)  ->  Bull  ->  Bear  ->  (debate again?)  ->  Judge (fills a form)

Run it:
    python learning_lab/toy_graph.py                       # fake model, on the Mac, instant
    python learning_lab/toy_graph.py --rounds 2            # longer debate
    python learning_lab/toy_graph.py --llm vllm --url http://127.0.0.1:PORT/v1   # real model on FEP

The comments marked CONCEPT 1..6 are the six ideas to learn. The real pipeline in Phase 1
uses exactly these six, only with more agents.
"""

import argparse
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------------------------
# CONCEPT 1 - STATE: one shared "notebook" that every agent reads and writes.
# Each agent (node) receives the whole notebook and returns only the fields it wants to change.
# ---------------------------------------------------------------------------------------------
class State(TypedDict):
    ticker: str
    # CONCEPT 2 - REDUCERS: how a field is updated when a node returns a new value.
    #   No reducer (e.g. `report`): the new value REPLACES the old one.
    #   `add_messages`: new messages are APPENDED to the conversation (used by the tool loop).
    #   `operator.add`: lists are concatenated, so each debate turn is APPENDED.
    messages: Annotated[list, add_messages]
    report: str
    debate: Annotated[list[str], operator.add]
    rounds: int
    decision: dict


# The analyst's one tool. Toy data, no internet: the real tools in Phase 1 read our archive.
@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest closing price and 5-day change for a stock ticker, e.g. AAPL."""
    return f"{ticker.upper()}: close 185.20 USD, 5-day change +3.1%"


class Decision(BaseModel):
    """The form the judge must fill in (structured output)."""

    rating: Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]
    reason: str = Field(description="One sentence")


# ---------------------------------------------------------------------------------------------
# CONCEPT 3 - NODES: each agent is a plain Python function  state -> {fields to update}.
# `make_nodes` builds them around a given LLM, so the same graph runs on a fake or a real model.
# ---------------------------------------------------------------------------------------------
def make_nodes(llm, max_rounds: int):
    analyst_llm = llm.bind_tools([get_stock_price])  # the analyst may ask for this tool
    judge_llm = llm.with_structured_output(Decision, method="json_schema")

    def analyst(state: State) -> dict:
        system = SystemMessage(
            "You are a market analyst. Use get_stock_price to look up the stock, then write a "
            "3-sentence report on its recent price action."
        )
        reply = analyst_llm.invoke([system, *state["messages"]])
        return {"messages": [reply]}  # appended, thanks to add_messages

    def write_report(state: State) -> dict:
        return {"report": state["messages"][-1].content}  # the analyst's final text

    def bull(state: State) -> dict:
        reply = llm.invoke([
            SystemMessage("You are a bull researcher. In 2 sentences, argue FOR buying."),
            HumanMessage(f"Analyst report:\n{state['report']}\n\nDebate so far:\n" + "\n".join(state["debate"])),
        ])
        return {"debate": [f"Bull: {reply.content}"]}  # appended, thanks to operator.add

    def bear(state: State) -> dict:
        reply = llm.invoke([
            SystemMessage("You are a bear researcher. In 2 sentences, argue AGAINST buying."),
            HumanMessage(f"Analyst report:\n{state['report']}\n\nDebate so far:\n" + "\n".join(state["debate"])),
        ])
        return {"debate": [f"Bear: {reply.content}"], "rounds": state["rounds"] + 1}

    def judge(state: State) -> dict:
        decision = judge_llm.invoke([
            SystemMessage("You are the judge. Read the debate and give a rating and a one-sentence reason."),
            HumanMessage("\n".join(state["debate"])),
        ])
        return {"decision": decision.model_dump()}

    # -----------------------------------------------------------------------------------------
    # CONCEPT 5 - CONDITIONAL EDGES: a function that looks at the state and names the next node.
    # -----------------------------------------------------------------------------------------
    def after_analyst(state: State) -> str:
        # CONCEPT 6 - TOOL LOOP: if the analyst asked for a tool, run it and come back;
        # otherwise the analyst is finished.
        return "tools" if state["messages"][-1].tool_calls else "write_report"

    def after_bear(state: State) -> str:
        return "bull" if state["rounds"] < max_rounds else "judge"

    return analyst, write_report, bull, bear, judge, after_analyst, after_bear


# ---------------------------------------------------------------------------------------------
# CONCEPT 4 - EDGES: the arrows. They say which node runs after which.
# ---------------------------------------------------------------------------------------------
def build_graph(llm, max_rounds: int = 1):
    analyst, write_report, bull, bear, judge, after_analyst, after_bear = make_nodes(llm, max_rounds)

    g = StateGraph(State)
    g.add_node("analyst", analyst)
    g.add_node("tools", ToolNode([get_stock_price]))  # runs the tool the analyst asked for
    g.add_node("write_report", write_report)
    g.add_node("bull", bull)
    g.add_node("bear", bear)
    g.add_node("judge", judge)

    g.add_edge(START, "analyst")
    g.add_conditional_edges("analyst", after_analyst, ["tools", "write_report"])
    g.add_edge("tools", "analyst")  # back to the analyst with the tool's answer
    g.add_edge("write_report", "bull")
    g.add_edge("bull", "bear")
    g.add_conditional_edges("bear", after_bear, ["bull", "judge"])
    g.add_edge("judge", END)
    return g.compile()


class FakeLLM:
    """Scripted stand-in for a real model, so the graph runs on the Mac with no GPU.

    It answers by looking at who is asking (the system prompt), like an actor reading a script.
    """

    turns = 0  # how many debate turns it has played

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, **kwargs):
        fake = self

        class Form:
            def invoke(self, messages):
                return schema(rating="Hold", reason=f"(fake) weighed {fake.turns} debate turns")

        return Form()

    def invoke(self, messages):
        role = messages[0].content
        if "market analyst" in role:
            results = [m for m in messages if isinstance(m, ToolMessage)]
            if not results:  # first visit: ask for the tool
                return AIMessage("", tool_calls=[{"name": "get_stock_price", "args": {"ticker": "AAPL"}, "id": "call_1"}])
            return AIMessage(f"(fake report) Tool said: {results[-1].content}. Momentum looks positive.")
        self.turns += 1
        side = "bull" if "bull" in role else "bear"
        return AIMessage(f"(fake {side} argument #{self.turns})")


def make_llm(kind: str, url: str | None, model: str):
    if kind == "fake":
        return FakeLLM()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(base_url=url, api_key="EMPTY", model=model, temperature=0, max_tokens=300)


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", choices=["fake", "vllm"], default="fake")
    p.add_argument("--url", help="vLLM endpoint, e.g. http://127.0.0.1:28152/v1")
    p.add_argument("--model", default="llama-3.1-8b")
    p.add_argument("--rounds", type=int, default=1, help="bull/bear exchanges before the judge")
    args = p.parse_args(argv)

    graph = build_graph(make_llm(args.llm, args.url, args.model), args.rounds)
    start = {"ticker": "AAPL", "messages": [HumanMessage("Analyse AAPL.")], "report": "",
             "debate": [], "rounds": 0, "decision": {}}

    # stream() runs the graph step by step. "updates" = what each node returned;
    # "values" = the whole notebook after the reducers applied that update.
    final = start
    for mode, chunk in graph.stream(start, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, update in chunk.items():
                print(f"--> {node:13s} wrote {sorted(update)}")
        else:
            final = chunk
    print("\nDebate:\n  " + "\n  ".join(final["debate"]))
    print(f"\nDecision: {final['decision']}")
    return final


if __name__ == "__main__":
    main()
