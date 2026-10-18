"""The four analysts. Each is a graph node: it reads the state and returns its updates.

Follows upstream `agents/analysts/*.py` (commit 35543d0). Market, News and Fundamentals share
one shape, so they share one function here (`_create_tool_analyst`); only the prompt, the tools
and the report field differ. The Sentiment Analyst has no tools: it fetches its sources first,
pastes them into the prompt, and fills in a SentimentReport form.
"""

from collections.abc import Callable, Mapping

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from aihf.agents.context import get_instrument_context_from_state
from aihf.agents.schemas import SentimentReport, render_sentiment_report
from aihf.llm import invoke_structured
from aihf.prompts.analysts import (
    FUNDAMENTALS_SYSTEM,
    MARKET_SYSTEM,
    NEWS_SYSTEM,
    SENTIMENT_HEADER,
    SENTIMENT_SYSTEM,
    TOOL_ANALYST_HEADER,
)
from aihf.tools.dates import days_back
from aihf.tools.source import DataSource

# Which tools each analyst is offered (upstream's TOOLS tuples, by name).
MARKET_TOOLS = ("get_stock_data", "get_indicators", "get_verified_market_snapshot")
NEWS_TOOLS = ("get_news", "get_global_news", "get_macro_indicators")
FUNDAMENTALS_TOOLS = (
    "get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement",
    "get_insider_transactions",
)


def _create_tool_analyst(
    llm, tools: list[BaseTool], system_message: Callable[[Mapping], str], report_key: str
):
    """A node that either asks for tools or writes its report.

    Each visit makes one model call. If the reply asks for tools, the graph runs them and
    comes back here (the tool loop); the report stays "" until a reply without tool calls.
    """

    def analyst_node(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", TOOL_ANALYST_HEADER),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(
            system_message=system_message(state),
            tool_names=", ".join(t.name for t in tools),
            current_date=state["trade_date"],
            instrument_context=get_instrument_context_from_state(state),
        )
        result = (prompt | llm.bind_tools(tools)).invoke({"messages": state["messages"]})
        report = "" if result.tool_calls else result.content
        return {"messages": [result], report_key: report}

    return analyst_node


def _pick(tools: Mapping[str, BaseTool], names: tuple[str, ...]) -> list[BaseTool]:
    return [tools[name] for name in names]


def create_market_analyst(llm, tools: Mapping[str, BaseTool]):
    return _create_tool_analyst(llm, _pick(tools, MARKET_TOOLS), lambda s: MARKET_SYSTEM,
                                "market_report")


def create_news_analyst(llm, tools: Mapping[str, BaseTool]):
    def system_message(state):
        asset_label = "company" if state.get("asset_type", "stock") == "stock" else "asset"
        return NEWS_SYSTEM.format(asset_label=asset_label)

    return _create_tool_analyst(llm, _pick(tools, NEWS_TOOLS), system_message, "news_report")


def create_fundamentals_analyst(llm, tools: Mapping[str, BaseTool]):
    return _create_tool_analyst(llm, _pick(tools, FUNDAMENTALS_TOOLS),
                                lambda s: FUNDAMENTALS_SYSTEM, "fundamentals_report")


def create_sentiment_analyst(llm, data: DataSource):
    """Fetch news, StockTwits and Reddit for the past 7 days, then fill in a SentimentReport.

    If the form fails, the plain-text answer becomes the report (as upstream).
    """

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = days_back(end_date, 7)

        system_message = SENTIMENT_SYSTEM.format(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=data.get_news(ticker, start_date, end_date),
            stocktwits_block=data.get_stocktwits_messages(ticker, start_date, end_date),
            reddit_block=data.get_reddit_posts(ticker, start_date, end_date),
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SENTIMENT_HEADER),
            MessagesPlaceholder(variable_name="messages"),
        ]).partial(
            system_message=system_message,
            current_date=end_date,
            instrument_context=get_instrument_context_from_state(state),
        )
        messages = prompt.format_messages(messages=state["messages"])

        form, text = invoke_structured(llm, SentimentReport, messages, agent="Sentiment Analyst")
        report = render_sentiment_report(form) if form is not None else text
        return {"messages": [AIMessage(content=report)], "sentiment_report": report}

    return sentiment_analyst_node
