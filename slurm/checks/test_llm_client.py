"""Our own client (aihf.llm) against a running vLLM server: chat, tool call, form, call log.

Runs inside a GPU job, next to test_vllm_smoke.py (same environment variables, see there).
The call log is written to logs/llm_check-<job id>.jsonl so it can be opened after the job.
"""

import json
import os
from pathlib import Path
from typing import Literal

import pytest
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from aihf.llm import JsonlCallLog, invoke_structured, make_chat_model

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "logs" / f"llm_check-{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl"


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is not set; run these tests through slurm/smoke_dev.sbatch")
    return value


@pytest.fixture(scope="module")
def llm():
    cfg = yaml.safe_load((REPO / _env("AIHF_MODEL_CONFIG")).read_text())
    LOG.unlink(missing_ok=True)
    # temperature 0 so the answers are as repeatable as possible for a test
    return make_chat_model(_env("AIHF_VLLM_URL").rstrip("/"), cfg["name"], temperature=0,
                           max_tokens=256, timeout=120, max_retries=0,
                           callbacks=[JsonlCallLog(LOG)])


@tool
def get_stock_price(ticker: str) -> float:
    """Get the latest closing price for a stock ticker, e.g. AAPL."""
    return 123.45


class Decision(BaseModel):
    """A trading decision, the same kind of form our Portfolio Manager fills in."""

    rating: Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]
    reason: str = Field(description="One sentence explaining the rating")


def test_chat(llm):
    reply = llm.invoke([SystemMessage("Answer with just the number."), HumanMessage("2 + 2?")])
    assert "4" in reply.content


def test_tool_call(llm):
    reply = llm.bind_tools([get_stock_price], tool_choice="auto").invoke(
        "What is the latest closing price of AAPL? Use the tool.")
    assert reply.tool_calls, f"no tool call; the model answered: {reply.content!r}"
    assert reply.tool_calls[0]["name"] == "get_stock_price"


def test_form(llm):
    form, text = invoke_structured(
        llm, Decision,
        "A company just beat earnings expectations by 20% and raised guidance. "
        "Give a trading rating and a one-sentence reason.",
        agent="check")
    print(f"form: {form!r}")
    assert form is not None, f"form failed, plain-text fallback: {text!r}"


def test_call_log_has_every_call_with_real_token_counts(llm):
    lines = [json.loads(line) for line in LOG.read_text().splitlines()]
    print(f"call log: {LOG} ({len(lines)} lines)")
    assert len(lines) == 3  # chat, tool call, form (no fallback)
    for line in lines:
        assert line["error"] is None
        assert line["input_tokens"] > 0 and line["output_tokens"] > 0
        assert line["latency_ms"] > 0
    assert lines[1]["tools"] == ["get_stock_price"]
