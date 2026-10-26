"""The agents of the TradingAgents pipeline, as LangGraph nodes."""

from aihf.agents.analysts import (
    create_fundamentals_analyst,
    create_market_analyst,
    create_news_analyst,
    create_sentiment_analyst,
)
from aihf.agents.context import build_instrument_context, create_msg_delete, load_instruments
from aihf.agents.researchers import (
    create_bear_researcher,
    create_bull_researcher,
    create_research_manager,
)
from aihf.agents.risk import (
    create_aggressive_debator,
    create_conservative_debator,
    create_neutral_debator,
    create_portfolio_manager,
)
from aihf.agents.trader import create_trader

__all__ = [
    "build_instrument_context",
    "create_aggressive_debator",
    "create_bear_researcher",
    "create_bull_researcher",
    "create_conservative_debator",
    "create_fundamentals_analyst",
    "create_market_analyst",
    "create_msg_delete",
    "create_neutral_debator",
    "create_news_analyst",
    "create_portfolio_manager",
    "create_research_manager",
    "create_sentiment_analyst",
    "create_trader",
    "load_instruments",
]
