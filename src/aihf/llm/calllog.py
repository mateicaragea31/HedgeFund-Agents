"""The recorder: one JSON line per model call (prompt, reply, tokens, time). CLAUDE.md §3.5.

It is a LangChain *callback*: LangChain calls `on_chat_model_start` before every model call and
`on_llm_end` (or `on_llm_error`) after it, so every call is logged without the agents doing
anything. Give it to the model once: `make_chat_model(..., callbacks=[JsonlCallLog(path)])`.
"""

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

# Sampling settings worth keeping with every call, for reproducibility.
_PARAMS = ("temperature", "max_tokens", "max_completion_tokens", "seed", "reasoning_effort")


def _message(m: BaseMessage) -> dict:
    """A message as plain JSON: its role, text and (if any) tool requests or tool-call id."""
    out: dict[str, Any] = {"role": m.type, "content": m.content}  # role: system/human/ai/tool
    if getattr(m, "tool_calls", None):
        out["tool_calls"] = m.tool_calls
    if getattr(m, "tool_call_id", None):
        out["tool_call_id"] = m.tool_call_id
    return out


class JsonlCallLog(BaseCallbackHandler):
    """Append one line per model call to a .jsonl file (one file per cell)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started: dict[UUID, dict] = {}  # calls in progress, by LangChain's run id
        self._lock = threading.Lock()  # in case LangChain calls us from another thread

    def on_chat_model_start(
        self, serialized: dict, messages: list[list[BaseMessage]], *, run_id: UUID,
        metadata: dict | None = None, **kwargs: Any,
    ) -> None:
        params = kwargs.get("invocation_params") or {}
        extra = params.get("extra_body") or {}
        self._started[run_id] = {
            "t0": time.monotonic(),
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            # LangGraph tells every call which node made it, e.g. "Bull Researcher".
            "agent": (metadata or {}).get("langgraph_node", ""),
            "model": params.get("model") or params.get("model_name", ""),
            "params": {k: v for k, v in {**params, **extra}.items() if k in _PARAMS},
            "tools": [t["function"]["name"] for t in params.get("tools", []) if "function" in t],
            "messages": [_message(m) for m in messages[0]],
        }

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None) or {}
        self._write(run_id, {
            "response": message.content,
            "tool_calls": message.tool_calls,
            "reasoning": message.additional_kwargs.get("reasoning", ""),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "error": None,
        })

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._write(run_id, {"response": None, "error": f"{type(error).__name__}: {error}"})

    def _write(self, run_id: UUID, result: dict) -> None:
        record = self._started.pop(run_id, {})
        t0 = record.pop("t0", None)
        record["latency_ms"] = round((time.monotonic() - t0) * 1000) if t0 is not None else None
        record.update(result)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
