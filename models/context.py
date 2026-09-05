"""Per-turn context handed to an agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from models.message import Message


@runtime_checkable
class EnvironmentView(Protocol):
    """The environment surface an agent is allowed to observe."""

    forest_health: float
    economic_output: float


@dataclass(frozen=True)
class RoundContext:
    """Everything an agent sees when deciding its action for one turn."""

    round_id: int
    total_rounds: int
    latest_messages: Sequence[Message]
    environment: EnvironmentView
    valid_targets: Sequence[str]
    group_average: float
    spread: float
