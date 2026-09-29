"""Phase 0.2: both packages import, and upstream is the pinned version."""

import socket

import pytest


def test_imports():
    import tradingagents
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    import aihf

    assert aihf.__version__
    assert tradingagents is not None
    assert TradingAgentsGraph is not None


def test_upstream_is_pinned_version():
    from importlib.metadata import version

    assert version("tradingagents") == "0.5.1"


def test_network_is_blocked():
    with pytest.raises(OSError, match="network"):
        socket.create_connection(("example.com", 80), timeout=1)
