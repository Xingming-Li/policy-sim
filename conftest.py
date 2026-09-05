"""Shared test fixtures

Living at repo root, the file puts the project root on `sys.path`
for pytest's default import, so tests import the packages directly.

Every fixture here is offline: no test in this suite requires
`OPENAI_API_KEY` or network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.base_agent import Agent
from environment.forest_policy import ForestPolicyEnvironment
from models.context import RoundContext
from simulation.config import AgentSpec, SimulationConfig
from utils.fake_llm import FakeLLM

TEST_AGENTS = (
    AgentSpec(name="Alpha", role="Policymaker", goal="Balance", preferred_position=0.5),
    AgentSpec(name="Beta", role="Industry", goal="Profit", preferred_position=0.8),
    AgentSpec(name="Gamma", role="NGO", goal="Protect", preferred_position=0.1),
)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee the suite don't need a real key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def config(tmp_path: Path) -> SimulationConfig:
    return SimulationConfig(
        rounds=2,
        model="test-model",
        temperature=0.0,
        sensitivity=0.2,
        protocol="simultaneous",
        max_retries=2,
        output_dir=tmp_path / "runs",
        agents=TEST_AGENTS,
    )


@pytest.fixture
def environment() -> ForestPolicyEnvironment:
    return ForestPolicyEnvironment(sensitivity=0.2)


@pytest.fixture
def agent_spec() -> AgentSpec:
    return TEST_AGENTS[0]


def make_agent(
    spec: AgentSpec,
    script: list,
    *,
    max_retries: int = 2,
    loop: bool = False,
) -> Agent:
    return Agent(spec, FakeLLM(script, loop=loop), max_retries=max_retries)


def make_context(
    environment: ForestPolicyEnvironment,
    *,
    round_id: int = 1,
    total_rounds: int = 5,
    messages: tuple = (),
) -> RoundContext:
    return RoundContext(
        round_id=round_id,
        total_rounds=total_rounds,
        latest_messages=messages,
        environment=environment,
        valid_targets=tuple(spec.name for spec in TEST_AGENTS),
        group_average=0.47,
        spread=0.7,
    )
