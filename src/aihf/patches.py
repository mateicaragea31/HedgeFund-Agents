"""The only place that changes `tradingagents` behaviour.

Everything that alters upstream — vendor registration in
`tradingagents.dataflows.router`, module-level function overrides, subclassed
agents — is applied from here, so the difference between "the paper's system"
and "our changes" stays in one reviewable file. Each patch must name its
`docs/DEVIATIONS.md` entry.

Nothing is patched yet.
"""
