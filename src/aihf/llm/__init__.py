"""Talking to our served models: connection, <think> cleaning, forms, call log."""

from aihf.llm.calllog import JsonlCallLog
from aihf.llm.client import AihfChatOpenAI, make_chat_model
from aihf.llm.structured import invoke_structured
from aihf.llm.think import split_think

__all__ = ["AihfChatOpenAI", "JsonlCallLog", "invoke_structured", "make_chat_model", "split_think"]
