"""The risk debate: Aggressive, Conservative and Neutral analysts argue, the Portfolio Manager decides.

Follows upstream `agents/risk_mgmt/*_debator.py` and `agents/managers/portfolio_manager.py`
(commit 35543d0). The three debaters are the same agent with a different prompt, so they share
one function here (`_create_risk_debater`).

Each speech is added to the debate text in the notebook (`risk_debate_state`), `latest_speaker`
says who spoke, and the turn counter goes up by one; graph/risk.py uses both to pick the next
speaker and to decide when to stop.
"""

from aihf.agents.context import (
    get_instrument_context_from_state,
    get_portfolio_context_from_state,
    opponent_argument_or_opening,
    report_or_absent,
)
from aihf.agents.schemas import PortfolioDecision, render_pm_decision
from aihf.llm import invoke_structured
from aihf.prompts.risk import (
    AGGRESSIVE_PROMPT,
    CONSERVATIVE_PROMPT,
    LESSONS_LINE,
    NEUTRAL_PROMPT,
    PORTFOLIO_MANAGER_PROMPT,
)

STANCES = ("Aggressive", "Conservative", "Neutral")


def _create_risk_debater(llm, prompt_template: str, stance: str):
    """One debater. stance = "Aggressive", "Conservative" or "Neutral"."""
    own = stance.lower()

    def risk_debater_node(state) -> dict:
        debate = state["risk_debate_state"]
        # Each prompt uses the last answers of the other two; str.format ignores the third.
        last_answers = {
            f"current_{s.lower()}_response": opponent_argument_or_opening(
                debate.get(f"current_{s.lower()}_response", ""), f"{s.lower()} analyst")
            for s in STANCES
        }
        prompt = prompt_template.format(
            trader_decision=state["trader_investment_plan"],
            instrument_context=get_instrument_context_from_state(state),
            portfolio_context=get_portfolio_context_from_state(state),
            market_research_report=report_or_absent(state["market_report"], "market"),
            sentiment_report=report_or_absent(state["sentiment_report"], "sentiment"),
            news_report=report_or_absent(state["news_report"], "news"),
            fundamentals_report=report_or_absent(state["fundamentals_report"], "fundamentals"),
            history=debate.get("history", ""),
            **last_answers,
        )
        argument = f"{stance} Analyst: {llm.invoke(prompt).content}"

        # The debate dict has no reducer, so we return all of it (see graph/state.py).
        new_debate = dict(debate)
        new_debate["history"] = debate.get("history", "") + "\n" + argument
        new_debate[f"{own}_history"] = debate.get(f"{own}_history", "") + "\n" + argument
        new_debate["latest_speaker"] = stance
        new_debate[f"current_{own}_response"] = argument
        new_debate["count"] = debate["count"] + 1
        return {"risk_debate_state": new_debate}

    return risk_debater_node


def create_aggressive_debator(llm):
    return _create_risk_debater(llm, AGGRESSIVE_PROMPT, "Aggressive")


def create_conservative_debator(llm):
    return _create_risk_debater(llm, CONSERVATIVE_PROMPT, "Conservative")


def create_neutral_debator(llm):
    return _create_risk_debater(llm, NEUTRAL_PROMPT, "Neutral")


def create_portfolio_manager(llm):
    """The final judge: reads the plans and the risk debate, fills in a PortfolioDecision.

    If the form fails, the plain-text answer becomes the decision (as upstream); the rating is
    then read from the text by agents/rating.py, or marked REVIEW.
    """

    def portfolio_manager_node(state) -> dict:
        debate = state["risk_debate_state"]
        past_context = state.get("past_context", "")
        prompt = PORTFOLIO_MANAGER_PROMPT.format(
            instrument_context=get_instrument_context_from_state(state),
            portfolio_context=get_portfolio_context_from_state(state),
            research_plan=state["investment_plan"],
            trader_plan=state["trader_investment_plan"],
            lessons_line=LESSONS_LINE.format(past_context=past_context) if past_context else "",
            history=debate["history"],
        )
        form, text = invoke_structured(llm, PortfolioDecision, prompt, agent="Portfolio Manager")
        decision = render_pm_decision(form) if form is not None else text

        new_debate = dict(debate)
        new_debate["judge_decision"] = decision
        new_debate["latest_speaker"] = "Judge"
        return {"risk_debate_state": new_debate, "final_trade_decision": decision}

    return portfolio_manager_node
