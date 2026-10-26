"""Reflection: the quick model looks back at a settled decision and writes a 2-4 sentence lesson.

Follows upstream `graph/reflection.py` (Reflector.reflect_on_final_decision, commit 35543d0).
"""

from aihf.prompts.reflection import REFLECTION_SYSTEM, REFLECTION_USER


def reflect_on_final_decision(
    llm,
    final_decision: str,
    raw_return: float,
    alpha_return: float,
    benchmark_name: str = "SPY",
    holding_days: int = 5,
) -> str:
    """One model call: the decision and its outcome in, the lesson text out.

    The final decision already sums up every analyst's view, so nothing else is sent.
    """
    messages = [
        ("system", REFLECTION_SYSTEM.format(holding_days=holding_days)),
        ("human", REFLECTION_USER.format(
            holding_days=holding_days,
            raw_return=raw_return,
            alpha_return=alpha_return,
            benchmark_name=benchmark_name,
            final_decision=final_decision,
        )),
    ]
    return llm.invoke(messages).content
