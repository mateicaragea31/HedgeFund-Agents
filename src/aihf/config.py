"""Experiment and model configs: yaml files checked by pydantic (CLAUDE.md §9).

A misspelled field is an error (extra="forbid"), never silently ignored. Paths in yaml may use
${AIHF_LAB}-style variables; they are filled from the environment when the file is read.
"""

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from aihf.graph.analysts import ANALYST_SPECS

REPO = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    # protected_namespaces=(): allow field names starting with "model_" (model_profile)
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelHardware(_Strict):
    partition: str
    gpus: int
    precision: str


class ModelConfig(_Strict):
    """configs/models/<name>.yaml: one served model."""

    name: str  # the vLLM --served-model-name
    profile: str
    hf_repo: str
    revision: str
    path: str
    knowledge_cutoff: date
    hardware: ModelHardware
    vllm: dict
    chat_template_source: dict | None = None


class Window(_Strict):
    start: date
    end: date


class Sampling(_Strict):
    temperature: Literal["model_default"] | float = "model_default"


class ExperimentConfig(_Strict):
    """configs/experiments/<name>.yaml. The simulator's fields come in §6.4."""

    name: str
    phase: int
    tickers: list[str]
    window: Window
    model_profile: Path  # relative to the repo root
    quick_model: str
    deep_model: str
    analysts: list[str]
    max_debate_rounds: int = 1
    max_risk_discuss_rounds: int = 1
    memory: bool = True
    holding_period_days: int = 5
    sampling: Sampling = Sampling()
    repeats: int = 1  # independent runs of the whole experiment (CLAUDE.md §3.5: at least 3)

    @model_validator(mode="after")
    def _check(self):
        unknown = [a for a in self.analysts if a not in ANALYST_SPECS]
        if unknown or not self.analysts:
            raise ValueError(f"analysts must be a non-empty list from {list(ANALYST_SPECS)}, got {unknown}")
        if self.window.end < self.window.start:
            raise ValueError("window.end is before window.start")
        if self.max_debate_rounds < 1 or self.max_risk_discuss_rounds < 1:
            raise ValueError("debate and risk rounds must be at least 1")
        return self

    @property
    def temperature(self) -> float | None:
        """None = send nothing, so vLLM uses the model's own defaults (CLAUDE.md §3.5)."""
        t = self.sampling.temperature
        return None if t == "model_default" else float(t)


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(os.path.expandvars(Path(path).read_text()))


def load_model(path: Path) -> ModelConfig:
    return ModelConfig.model_validate(_read_yaml(path))


def load_experiment(path: Path) -> tuple[ExperimentConfig, ModelConfig]:
    """The experiment and its model profile, after the knowledge-cutoff check (CLAUDE.md §3.3)."""
    experiment = ExperimentConfig.model_validate(_read_yaml(path))
    model = load_model(REPO / experiment.model_profile)
    check_knowledge_cutoff(experiment, model)
    return experiment, model


def check_knowledge_cutoff(experiment: ExperimentConfig, model: ModelConfig) -> None:
    """Refuse a model that may already know the test window: its cutoff must come before it."""
    if model.knowledge_cutoff >= experiment.window.start:
        raise ValueError(
            f"{model.name} has knowledge cutoff {model.knowledge_cutoff}, not before the window "
            f"start {experiment.window.start}: it may already know what happened (CLAUDE.md §3.3)")
