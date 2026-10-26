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


def get_portfolio_context_from_state(state: Mapping[str, Any]) -> str:
    """The current position, or upstream's note that none was given.

    In paper-fidelity mode there is no portfolio context, so the agents read this note: without
    it they would size as if the account held nothing, which nobody told them.
    """
    context = state.get("portfolio_context")
    if isinstance(context, str) and context.strip():
        return context
    return (
        "Portfolio context: not provided. You do not know the caller's current "
        "holdings or cash, so do not assume a flat book; give direction and "
        "sizing guidance in terms the caller can apply to their own position."
    )


def report_or_absent(text: str, source: str) -> str:
    """An analyst's report, or a note saying it was never written (upstream's wording).

    A blank in the prompt reads like "the analyst found nothing", and the reading agent then
    fills it in from nothing; the note says plainly that the report doesn't exist.
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(No {source} report in this run: it is not available, not an empty finding.)"


def opponent_argument_or_opening(text: str, opponent: str) -> str:
    """The other side's last argument, or a note that they haven't spoken yet (upstream's wording).

    Without the note, the first speaker is asked to "refute" an empty argument and tends to
    invent one.
    """
    text = (text or "").strip()
    if text:
        return text
    return f"(The {opponent} has not spoken yet — open the debate with your own case.)"


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
