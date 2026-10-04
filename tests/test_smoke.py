"""Phase 0.2: the package imports, and it never depends on upstream `tradingagents`."""

import ast
import socket
from pathlib import Path

import pytest

import aihf

SRC = Path(aihf.__file__).parent


def test_imports():
    assert aihf.__version__


def test_never_imports_tradingagents():
    """`reference/` is reading material only (CLAUDE.md §9): no module may import it."""
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "tradingagents" or n.startswith("tradingagents.") for n in names):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"imports of tradingagents: {offenders}"


def test_network_is_blocked():
    with pytest.raises(OSError, match="network"):
        socket.create_connection(("example.com", 80), timeout=1)
