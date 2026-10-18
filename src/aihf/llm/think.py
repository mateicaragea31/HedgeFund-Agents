"""Separate a reasoning model's "scratch work" (<think>...</think>) from its answer.

Reasoning models such as DeepSeek-R1-Distill write their thinking first:

    <think>The price rose 3%, but the P/E is high...</think>I recommend Hold.

Agents must only see the answer; the thinking is kept for the call log.
"""

import re

_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)  # DOTALL: "." also matches newlines


def split_think(text: str) -> tuple[str, str]:
    """Return (answer, reasoning). Text without <think> tags comes back unchanged, with "" reasoning."""
    reasoning = [part.strip() for part in _BLOCK.findall(text)]
    answer = _BLOCK.sub("", text)

    # Some chat templates open <think> inside the prompt, so the reply starts mid-thought and
    # only contains the closing tag: "...thinking...</think>answer".
    if "</think>" in answer:
        before, answer = answer.split("</think>", 1)
        reasoning.insert(0, before.strip())

    # A reply cut off by max_tokens can open <think> and never close it: all of it is thinking.
    if "<think>" in answer:
        answer, rest = answer.split("<think>", 1)
        reasoning.append(rest.strip())

    return answer.strip(), "\n\n".join(part for part in reasoning if part)
