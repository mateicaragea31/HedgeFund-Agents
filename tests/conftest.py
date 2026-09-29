"""Shared fixtures. Tests never touch the network (CLAUDE.md §9)."""

import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(request, monkeypatch):
    """Refuse outgoing connections, except in tests marked `gpu` (they talk to vLLM)."""
    if request.node.get_closest_marker("gpu"):
        return

    def refuse(self, address):
        raise OSError(f"test tried to reach the network: {address}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
