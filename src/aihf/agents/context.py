"""Who the company is (instrument context), and the node that clears messages between analysts.

Follows upstream `agents/context.py` (commit 35543d0). One difference (DEVIATIONS D2): upstream
looks the company up in today's yfinance profile; we read a fixed table, configs/instruments.yaml,
written for the test window, so no present-day information reaches a past trade date.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, RemoveMessage

from aihf.prompts.analysts import MSG_CLEAR_NOTE

INSTRUMENTS_FILE = Path(__file__).resolve().parents[3] / "configs" / "instruments.yaml"


def load_instruments(path: Path = INSTRUMENTS_FILE) -> dict[str, dict[str, str]]:
    """The ticker → identity table, e.g. {"AAPL": {"company": "Apple Inc.", "exchange": "NASDAQ"}}."""
    return yaml.safe_load(path.read_text())


def build_instrument_context(ticker: str, identity: Mapping[str, str] | None = None) -> str:
    """The sentence every prompt carries so agents analyse the right company. Upstream's wording.

    Upstream also adds a warning that its identity "is how the vendor describes the instrument
    today"; ours describes the company during the test window, so that warning is left out.
    """
    context = (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )
    details = []
    if identity and identity.get("company"):
        details.append(f"Company: {identity['company']}")
    if identity and identity.get("exchange"):
        details.append(f"Exchange: {identity['exchange']}")
    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """The context stored in the state, or a ticker-only one if it's empty (as upstream)."""
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_instrument_context(str(state["company_of_interest"]))


def create_msg_delete():
    """A node that removes every message and leaves one note naming the company and date.

    Runs after each analyst, so the next one starts clean instead of reading the previous
    analyst's tool calls. Upstream's note is deliberately not a bare "Continue": some models
    take that literally as the task.
    """

    def delete_messages(state):
        removals = [RemoveMessage(id=m.id) for m in state["messages"]]
        note = HumanMessage(content=MSG_CLEAR_NOTE.format(
            instrument_context=get_instrument_context_from_state(state),
            trade_date=state.get("trade_date", "the requested date"),
        ))
        return {"messages": removals + [note]}

    return delete_messages
