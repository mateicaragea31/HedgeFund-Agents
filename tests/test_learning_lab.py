"""The [TEACHING] toy graph keeps running with the fake model (Phase 0.8)."""

import importlib.util
from pathlib import Path

TOY = Path(__file__).resolve().parents[1] / "learning_lab" / "toy_graph.py"


def _load_toy():
    spec = importlib.util.spec_from_file_location("toy_graph", TOY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_toy_graph_runs_with_fake_llm():
    final = _load_toy().main(["--rounds", "2"])

    assert final["report"].startswith("(fake report) Tool said: AAPL: close")  # tool loop ran
    assert [turn.split(":")[0] for turn in final["debate"]] == ["Bull", "Bear", "Bull", "Bear"]
    assert final["rounds"] == 2
    assert final["decision"]["rating"] == "Hold"
