"""Configuration for a single simulation run

Everything that changes the behaviour of a run lives here, so that
`resolved_config.json` in `runs/run_id` is a complete record of what
used. The objects are frozen: once a run starts, params can't drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# "simultaneous": every agent in a round sees the same frozen context
# (previous rounds' messages, start-of-round group statistics).
# "sequential": context is rebuilt before each agent acts, so later
# agents see what earlier agents said in the same round.
Protocol = Literal["simultaneous", "sequential"]


class AgentSpec(BaseModel):
    """Definition of one negotiating agent."""
    # `extra="forbid"` means a typo in a config file is a startup
    # error rather than a silently ignored setting.
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    preferred_position: float = Field(ge=0.0, le=1.0)


DEFAULT_AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="Government",
        role="Policymaker",
        goal="Balance economy and conservation",
        preferred_position=0.5,
    ),
    AgentSpec(
        name="Industry",
        role="Forestry company",
        goal="Increase logging profits",
        preferred_position=0.8,
    ),
    AgentSpec(
        name="NGO",
        role="Environmental organization",
        goal="Protect forests",
        preferred_position=0.1,
    ),
    AgentSpec(
        name="Scientist",
        role="Ecologist",
        goal="Provide evidence-based advice",
        preferred_position=0.4,
    ),
)


class SimulationConfig(BaseModel):
    """Resolved parameters for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rounds: int = Field(default=5, ge=1, le=100)
    model: str = Field(default="gpt-4o-mini", min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # Environment sensitivity: how strongly the collective policy moves the environment per round.
    sensitivity: float = Field(default=0.20, ge=0.0, le=1.0)

    protocol: Protocol = "simultaneous"

    # Retries per agent turn on API failures and malformed responses.
    # Total attempts per turn = max_retries + 1.
    max_retries: int = Field(default=2, ge=0, le=5)

    output_dir: Path = Field(default=Path("runs"))

    agents: tuple[AgentSpec, ...] = Field(default=DEFAULT_AGENTS, min_length=2)

    def model_post_init(self, __context: Any) -> None:
        names = [a.name for a in self.agents]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate agent names: {sorted(duplicates)}")

    @property
    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents]

    @property
    def attempts_per_turn(self) -> int:
        return self.max_retries + 1

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe dict for `resolved_config.json`.

        Contains no credentials: the API key is read from the
        environment by the client and never enters the config object.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_file(cls, path: str | Path) -> SimulationConfig:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"config file must contain a JSON object: {path}")
        return cls.model_validate(raw)

    def with_overrides(self, **overrides: Any) -> SimulationConfig:
        """Return a validated copy with non-None overrides applied.

        Validates instead of using `model_copy`, which bypasses field
        constraints and passes invalid CLI values (e.g. `--rounds 0`).
        """
        applied = {k: v for k, v in overrides.items() if v is not None}
        if not applied:
            return self
        return type(self).model_validate({**self.model_dump(), **applied})
