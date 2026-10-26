"""The Trader's prompt, copied verbatim from upstream `agents/trader/trader.py`.

Source: github.com/TauricResearch/TradingAgents, commit 35543d0 (v0.5.1), Apache-2.0
(see LICENSES/TradingAgents-LICENSE).

Upstream builds the two messages from pieces: the grounding sentence and the market report are
only added when a market report exists. agents/trader.py joins these pieces the same way.
The only change: `get_language_instruction()` is left out (it returns "" for English).
"""

from aihf.prompts.analysts import NO_EXTERNAL_TOOLS

TRADER_SYSTEM_OPENING = (
    "You are a trading agent analyzing market data to make investment decisions. "
    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
)

# Added to the system message only when there is a market report.
TRADER_GROUNDING = (
    "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
    "market report's price structure -- current price, support/resistance, ATR, and "
    "volatility -- and use the research plan for direction and strategy. "
)

TRADER_SYSTEM_CLOSING = (
    "State entry price and stop-loss as absolute price levels in the "
    "instrument's quote currency (for example 189.5), never a percentage "
    "or a range; convert a percentage distance to the price level it "
    "implies, or omit the field if you cannot state a number. "
    + NO_EXTERNAL_TOOLS
)

# Filled with the market report; {report_section} is "" when there is none.
TRADER_REPORT_SECTION = "Technical Market Report:\n{market_report}\n\n"

TRADER_USER = (
    "Here is the research team's investment plan for {company_name}. "
    "{instrument_context}\n\n"
    "{report_section}"
    "{portfolio_context}\n\n"
    "Proposed Investment Plan:\n{investment_plan}\n\n"
    "Make an informed, strategic trading decision.\n\n"
    "## Output\n\n"
    "Write these sections, in this order, starting with the action "
    "on its own line:\n\n"
    "- **Action**: exactly one of Buy / Hold / Sell. A research "
    "recommendation of Overweight is a Buy and Underweight is a Sell, "
    "sized by how strong the case is; conflict alone is not a Hold.\n"
    "- **Reasoning**: why, against the plan and the price structure\n"
    "- **Entry Price**, **Stop Loss**, **Position Sizing**: when you can state them"
)
