"""The research debate: Bull and Bear researchers argue, the Research Manager judges.

Follows upstream `agents/researchers/{bull,bear}_researcher.py` and
`agents/managers/research_manager.py` (commit 35543d0). Bull and Bear are the same agent with a
different prompt and opponent, so they share one function here (`_create_researcher`).

Each speech is added to the debate text in the notebook (`investment_debate_state`), and the
turn counter goes up by one; graph/debate.py uses the counter to decide when to stop.
"""

from aihf.agents.context import (
    get_instrument_context_from_state,
    opponent_argument_or_opening,
    report_or_absent,
)
from aihf.agents.schemas import ResearchPlan, render_research_plan
from aihf.llm import invoke_structured
from aihf.prompts.researchers import BEAR_PROMPT, BULL_PROMPT, RESEARCH_MANAGER_PROMPT


def _create_researcher(llm, prompt_template: str, speaker: str, own_history: str, opponent: str):
    """One debater. speaker = "Bull Analyst", own_history = "bull_history", opponent = "bear analyst"."""

    def researcher_node(state) -> dict:
        debate = state["investment_debate_state"]
        is_stock = state.get("asset_type", "stock") == "stock"
        prompt = prompt_template.format(
            target_label="stock" if is_stock else "asset",
            instrument_context=get_instrument_context_from_state(state),
            market_research_report=report_or_absent(state["market_report"], "market"),
            sentiment_report=report_or_absent(state["sentiment_report"], "sentiment"),
            news_report=report_or_absent(state["news_report"], "news"),
            fundamentals_label=("Company fundamentals report" if is_stock
                                else "Asset fundamentals report (may be unavailable for crypto)"),
            fundamentals_report=report_or_absent(state["fundamentals_report"], "fundamentals"),
            history=debate.get("history", ""),
            current_response=opponent_argument_or_opening(debate.get("current_response", ""), opponent),
        )
        argument = f"{speaker}: {llm.invoke(prompt).content}"

        # The debate dict has no reducer, so we return all of it (see graph/state.py).
        new_debate = dict(debate)
        new_debate["history"] = debate.get("history", "") + "\n" + argument
        new_debate[own_history] = debate.get(own_history, "") + "\n" + argument
        new_debate["current_response"] = argument
        new_debate["count"] = debate["count"] + 1
        return {"investment_debate_state": new_debate}

    return researcher_node


def create_bull_researcher(llm):
    return _create_researcher(llm, BULL_PROMPT, "Bull Analyst", "bull_history", "bear analyst")


def create_bear_researcher(llm):
    return _create_researcher(llm, BEAR_PROMPT, "Bear Analyst", "bear_history", "bull analyst")


def create_research_manager(llm):
    """The judge: reads the whole debate, fills in a ResearchPlan (rating, rationale, actions).

    If the form fails, the plain-text answer becomes the plan (as upstream); the rating is then
    read from the text later, or marked REVIEW.
    """

    def research_manager_node(state) -> dict:
        debate = state["investment_debate_state"]
        prompt = RESEARCH_MANAGER_PROMPT.format(
            instrument_context=get_instrument_context_from_state(state),
            history=debate.get("history", ""),
        )
        form, text = invoke_structured(llm, ResearchPlan, prompt, agent="Research Manager")
        plan = render_research_plan(form) if form is not None else text

        new_debate = dict(debate)
        new_debate["judge_decision"] = plan
        new_debate["current_response"] = plan
        return {"investment_debate_state": new_debate, "investment_plan": plan}

    return research_manager_node
