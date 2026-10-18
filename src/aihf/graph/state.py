"""The shared state of one cell (one ticker on one trade date): what every agent reads and writes.

Field names match upstream exactly (`tradingagents/agents/state.py` and
`tradingagents/graph/propagation.py`, commit 35543d0), so the copied prompts and our saved
final states line up with upstream's, field for field.

How LangGraph merges what a node returns into this state:
- `messages` has a reducer (`add_messages`, from MessagesState): returned messages are APPENDED.
- Every other field has no reducer: a returned value REPLACES the old one. This includes the two
  debate dicts, so a debate node must return the whole dict, not just the keys it changed.
"""

import re
from typing import TypedDict

from langgraph.graph import MessagesState

# Upstream writes `Annotated[str, "description"]`. We use comments instead: in LangGraph a
# second argument to Annotated normally means "reducer", and a string there is silently ignored.


class InvestDebateState(TypedDict):
    """Bull vs Bear researchers, then the Research Manager's verdict."""

    bull_history: str  # everything the Bull said
    bear_history: str  # everything the Bear said
    history: str  # the whole debate, both sides, in order
    current_response: str  # the last thing said (the other side answers it)
    judge_decision: str  # the Research Manager's verdict
    count: int  # turns so far; the debate stops at 2 × max_debate_rounds


class RiskDebateState(TypedDict):
    """Aggressive / conservative / neutral risk debaters, then the Portfolio Manager's verdict."""

    aggressive_history: str
    conservative_history: str
    neutral_history: str
    history: str  # the whole debate, all three, in order
    latest_speaker: str  # who spoke last (decides who speaks next)
    current_aggressive_response: str
    current_conservative_response: str
    current_neutral_response: str
    judge_decision: str  # the Portfolio Manager's verdict
    count: int  # turns so far; the debate stops at 3 × max_risk_discuss_rounds


class AgentState(MessagesState):
    """Everything one cell knows. MessagesState adds `messages` (the analysts' tool loops)."""

    company_of_interest: str  # the ticker, e.g. "AAPL"
    asset_type: str  # always "stock" for us; kept for parity with upstream (it also does crypto)
    instrument_context: str  # who the company is, pasted into prompts; built in tools/
    trade_date: str  # "YYYY-MM-DD"

    sender: str  # upstream: only the Trader writes it and nothing reads it; kept for parity

    # analyst team
    market_report: str
    sentiment_report: str
    news_report: str
    fundamentals_report: str

    # researcher team
    investment_debate_state: InvestDebateState
    investment_plan: str  # the Research Manager's plan

    trader_investment_plan: str  # the Trader's plan

    # risk team
    risk_debate_state: RiskDebateState
    final_trade_decision: str  # the Portfolio Manager's decision; the rating is parsed from it

    past_context: str  # memory lessons available on the trade date (filled by the controller)
    portfolio_context: str  # current position; empty in paper-fidelity mode


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def initial_state(
    ticker: str,
    trade_date: str,
    *,
    instrument_context: str,
    asset_type: str = "stock",
    past_context: str = "",
    portfolio_context: str = "",
) -> dict:
    """The state a cell starts from, with the same keys and values as upstream's Propagator.

    Like upstream, `sender`, `investment_plan`, `trader_investment_plan` and
    `final_trade_decision` are not set here: the agents that own them fill them in.
    """
    if not _DATE.fullmatch(trade_date):
        raise ValueError(f"trade_date must be 'YYYY-MM-DD', got {trade_date!r}")
    return {
        "messages": [("human", ticker)],
        "company_of_interest": ticker,
        "asset_type": asset_type,
        "instrument_context": instrument_context,
        "trade_date": trade_date,
        "past_context": past_context,
        "portfolio_context": portfolio_context,
        "investment_debate_state": InvestDebateState(
            bull_history="",
            bear_history="",
            history="",
            current_response="",
            judge_decision="",
            count=0,
        ),
        "risk_debate_state": RiskDebateState(
            aggressive_history="",
            conservative_history="",
            neutral_history="",
            history="",
            latest_speaker="",
            current_aggressive_response="",
            current_conservative_response="",
            current_neutral_response="",
            judge_decision="",
            count=0,
        ),
        "market_report": "",
        "fundamentals_report": "",
        "sentiment_report": "",
        "news_report": "",
    }
