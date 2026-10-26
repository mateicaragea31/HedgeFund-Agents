"""The Trader: turns the Research Manager's plan into a concrete trade (Buy / Hold / Sell).

Follows upstream `agents/trader/trader.py` (commit 35543d0). The Trader also reads the market
report, so its entry and stop-loss prices come from real price levels; that part of the prompt
is only added when a market report exists.
"""

from langchain_core.messages import AIMessage

from aihf.agents.context import get_instrument_context_from_state, get_portfolio_context_from_state
from aihf.agents.schemas import TraderProposal, render_trader_proposal
from aihf.llm import invoke_structured
from aihf.prompts.trader import (
    TRADER_GROUNDING,
    TRADER_REPORT_SECTION,
    TRADER_SYSTEM_CLOSING,
    TRADER_SYSTEM_OPENING,
    TRADER_USER,
)


def create_trader(llm):
    """Fill in a TraderProposal; if the form fails, the plain-text answer becomes the plan."""

    def trader_node(state) -> dict:
        market_report = (state["market_report"] or "").strip()
        if market_report:
            grounding = TRADER_GROUNDING
            report_section = TRADER_REPORT_SECTION.format(market_report=market_report)
        else:
            grounding = report_section = ""

        messages = [
            {"role": "system", "content": TRADER_SYSTEM_OPENING + grounding + TRADER_SYSTEM_CLOSING},
            {"role": "user", "content": TRADER_USER.format(
                company_name=state["company_of_interest"],
                instrument_context=get_instrument_context_from_state(state),
                report_section=report_section,
                portfolio_context=get_portfolio_context_from_state(state),
                investment_plan=state["investment_plan"],
            )},
        ]
        form, text = invoke_structured(llm, TraderProposal, messages, agent="Trader")
        plan = render_trader_proposal(form) if form is not None else text

        # `messages` and `sender` as upstream; nothing downstream reads them.
        return {
            "messages": [AIMessage(content=plan)],
            "trader_investment_plan": plan,
            "sender": "Trader",
        }

    return trader_node
