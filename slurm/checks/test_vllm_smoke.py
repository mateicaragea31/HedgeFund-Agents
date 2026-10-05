"""Phase 0.5: smoke test of a running vLLM server (CLAUDE.md §5).

Runs only inside a GPU job (`slurm/smoke_dev.sbatch`), which starts the server and sets:
  AIHF_VLLM_URL       e.g. http://127.0.0.1:20123/v1
  AIHF_MODEL_CONFIG   e.g. configs/models/llama31_8b_dev.yaml

The four checks are what our agents rely on: the server is up, it chats, it calls tools,
and it fills a fixed answer form (structured output).
"""

import os
from pathlib import Path
from typing import Literal

import httpx
import pytest
import yaml
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is not set; run these tests through slurm/smoke_dev.sbatch")
    return value


@pytest.fixture(scope="module")
def base_url() -> str:
    return _env("AIHF_VLLM_URL").rstrip("/")


@pytest.fixture(scope="module")
def model_cfg() -> dict:
    return yaml.safe_load((REPO / _env("AIHF_MODEL_CONFIG")).read_text())


@pytest.fixture(scope="module")
def llm(base_url, model_cfg) -> ChatOpenAI:
    # temperature 0 so the answers are as repeatable as possible for a test
    return ChatOpenAI(
        base_url=base_url,
        api_key="EMPTY",  # our own server: no key needed
        model=model_cfg["name"],
        temperature=0,
        max_tokens=256,
        timeout=120,
        max_retries=0,
    )


def test_health_and_version(base_url, model_cfg):
    root = base_url.removesuffix("/v1")
    assert httpx.get(f"{root}/health", timeout=10).status_code == 200

    served = [m["id"] for m in httpx.get(f"{base_url}/models", timeout=10).json()["data"]]
    assert model_cfg["name"] in served

    images = yaml.safe_load((REPO / "containers/images.yaml").read_text())
    version = httpx.get(f"{root}/version", timeout=10).json()["version"]
    print(f"vLLM version: {version}")
    assert version.startswith(images["serving"]["versions"]["vllm"])


def test_plain_chat(llm):
    reply = llm.invoke("What is 2 + 2? Answer with just the number.")
    print(f"reply: {reply.content!r}")
    assert "4" in reply.content


@tool
def get_stock_price(ticker: str) -> float:
    """Get the latest closing price for a stock ticker, e.g. AAPL."""
    return 123.45


def test_tool_call(llm):
    reply = llm.bind_tools([get_stock_price], tool_choice="auto").invoke(
        "What is the latest closing price of AAPL? Use the tool."
    )
    print(f"tool_calls: {reply.tool_calls}")
    assert reply.tool_calls, f"no tool call; the model answered: {reply.content!r}"
    call = reply.tool_calls[0]
    assert call["name"] == "get_stock_price"
    assert call["args"].get("ticker", "").upper() == "AAPL"


class Decision(BaseModel):
    """A trading decision, the same kind of form our Portfolio Manager fills in."""

    rating: Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]
    reason: str = Field(description="One sentence explaining the rating")


def test_structured_output(llm):
    form = llm.with_structured_output(Decision, method="json_schema")
    decision = form.invoke(
        "A company just beat earnings expectations by 20% and raised guidance. "
        "Give a trading rating and a one-sentence reason."
    )
    print(f"decision: {decision!r}")
    assert isinstance(decision, Decision)
    assert decision.reason.strip()
