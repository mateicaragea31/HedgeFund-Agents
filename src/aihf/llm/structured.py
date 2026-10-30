"""Ask the model to fill in a form (structured output), with plain text as plan B.

Follows upstream's `agents/structured.py` (`invoke_structured_or_freetext`): a failed form must
never stop the pipeline. One difference, logged in docs/DEVIATIONS.md (D1): vLLM forces the reply
to match the form token by token ("guided decoding"); upstream asks for the form as a tool call,
which the model may ignore.

The form goes to vLLM as `structured_outputs: {"json": <schema>}`. Guided JSON lets the model write
whitespace between tokens, and the 70B sometimes writes it without end: in the first 70B
backtest (job 288964, 2026-10-30) 12 of 15 Portfolio Manager forms ran on until killed. The cure
is vLLM's server setting `--structured-outputs-config '{"backend": "xgrammar",
"disable_any_whitespace": true}'` (in the slurm scripts). The per-request field of the same name
is still sent, but vLLM 0.23's engines ignore it (only the server setting is read; job 289033
showed a runaway despite it). The answer-length cap (make_chat_model max_tokens) stops any other
runaway, and a cut-off form falls back to plain text like any broken one.

We check the reply ourselves (pydantic) instead of letting the openai SDK do it, so the
model's raw reply always reaches the call log, even when it doesn't fit the form.
"""

import logging
from typing import Any

from openai.lib._pydantic import to_strict_json_schema  # the openai SDK's form → JSON schema
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def response_format(schema: type[BaseModel]) -> dict:
    """The request field that tells vLLM: reply with JSON matching this form.

    Identical to what LangChain sends for `with_structured_output(schema, method="json_schema")`
    (the call our vLLM smoke test checked).
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": to_strict_json_schema(schema), "strict": True},
    }


def structured_outputs(llm: Any, schema: type[BaseModel]) -> dict:
    """The request's extra body: vLLM's guided JSON for `schema`, no whitespace allowed between
    tokens. Keeps whatever extra body the model already sends (e.g. gpt-oss's reasoning_effort)."""
    base = dict(getattr(llm, "extra_body", None) or {})
    base["structured_outputs"] = {"json": to_strict_json_schema(schema), "disable_any_whitespace": True}
    return base


# `[T: BaseModel]` (Python 3.12) says: T is some form class, and the function returns a form of
# that same class. So invoke_structured(llm, Decision, ...) gives back a Decision.
def invoke_structured[T: BaseModel](
    llm: Any, schema: type[T], messages: Any, agent: str
) -> tuple[T | None, str]:
    """Return (filled form, the JSON text), or (None, a plain-text answer) if the form failed.

    Plan A: the model fills in `schema` (e.g. Decision).
    Plan B: one plain call with the same messages; the caller decides what to do with the text
    (e.g. the rating parser marks it REVIEW if no rating can be read from it).
    """
    try:
        reply = llm.bind(extra_body=structured_outputs(llm, schema)).invoke(messages)
        return schema.model_validate_json(reply.content), reply.content
    except Exception as exc:  # server error, invalid JSON, a value not allowed by the form, ...
        problem = exc

    logger.warning("%s: structured output failed (%s); retrying once as plain text", agent, problem)
    reply = llm.invoke(messages)
    return None, reply.content
