"""The reflection prompt (a past decision, now that its outcome is known), copied verbatim.

Source: github.com/TauricResearch/TradingAgents, commit 35543d0 (v0.5.1), Apache-2.0
(see LICENSES/TradingAgents-LICENSE), `graph/reflection.py` (Reflector). Upstream builds both
messages with f-strings; ours are templates filled with str.format().
"""

# Reflector._system_prompt
REFLECTION_SYSTEM = (
    "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
    "The outcome covers {holding_days} trading days after the analysis date, "
    "which may be shorter than the horizon the decision was written for.\n"
    "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
    "Cover in order:\n"
    "1. What the {holding_days}-day alpha shows about the directional call (cite the figure), "
    "and say so plainly if the window is too short to judge the thesis.\n"
    "2. Which part of the investment thesis this window supports or undercuts.\n"
    "3. One concrete lesson to apply to the next similar analysis.\n\n"
    "Be specific and terse. Your output will be stored verbatim in a decision log "
    "and re-read by future analysts, so every word must earn its place."
)

# Reflector.reflect_on_final_decision (the human message)
REFLECTION_USER = (
    "Raw return over {holding_days} trading days: {raw_return:+.1%}\n"
    "Alpha vs {benchmark_name}: {alpha_return:+.1%}\n\n"
    "Final Decision:\n{final_decision}"
)
