"""The connection to our vLLM server (OpenAI-compatible API).

Agents never build a model themselves: they receive the object made by `make_chat_model` and
call `.invoke()`, `.bind_tools()` or `.with_structured_output()` on it.
"""

from typing import Any

from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from aihf.llm.think import split_think


class AihfChatOpenAI(ChatOpenAI):
    """LangChain's ChatOpenAI, plus one change: every reply is cleaned of <think> blocks.

    The answer stays in `.content`; the thinking moves to `.additional_kwargs["reasoning"]`,
    where the call log picks it up. (Upstream subclasses ChatOpenAI in the same way, see
    `llm_clients/openai_client.py`.)
    """

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> ChatResult:
        # Called for every reply from the server: plain, tool-calling and structured calls.
        result = super()._create_chat_result(response, generation_info)
        cleaned = []
        for gen in result.generations:
            message = gen.message
            if isinstance(message.content, str) and "think>" in message.content:
                answer, reasoning = split_think(message.content)
                message.content = answer
                message.additional_kwargs["reasoning"] = reasoning
            cleaned.append(ChatGeneration(message=message, generation_info=gen.generation_info))
        return ChatResult(generations=cleaned, llm_output=result.llm_output)


def make_chat_model(
    base_url: str,
    model: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    timeout: float = 600,
    max_retries: int = 2,
    callbacks: list | None = None,
    http_client: Any = None,
) -> AihfChatOpenAI:
    """Build the model object the agents talk to.

    base_url          e.g. "http://localhost:28152/v1" (the job sets the port)
    model             the --served-model-name, e.g. "llama-3.1-8b"
    temperature       None = send nothing, so vLLM uses the model's own defaults from
                      generation_config.json (CLAUDE.md §3.5). 0 for the reference run.
    reasoning_effort  only for gpt-oss ("low" = quick, "high" = deep); sent in the request body
    callbacks         e.g. [JsonlCallLog(path)]: called on every model call
    http_client       only for tests (a fake server); leave None
    """
    return AihfChatOpenAI(
        base_url=base_url,
        api_key="EMPTY",  # our own server: no key needed
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort else None,
        timeout=timeout,
        max_retries=max_retries,
        callbacks=callbacks,
        http_client=http_client,
    )
