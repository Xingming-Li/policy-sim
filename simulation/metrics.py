"""Convergence metrics over a transcript.

Each function takes the message list and reduces it to the latest stated
position per agent. Empty input returns a vacuous value instead of raising,
so a run that failed early still produces a readable summary.
"""

from __future__ import annotations

from typing import Sequence

from models.message import Message


def final_positions(transcript: Sequence[Message]) -> dict[str, float]:
    """Latest stated position per agent, in order of first appearance."""
    latest: dict[str, float] = {}

    for msg in transcript:
        latest[msg.sender] = msg.policy_position

    return latest


def position_spread(transcript: Sequence[Message]) -> float:
    """Gap between the most extreme positions. Vacuously 0.0 when empty."""
    positions = list(final_positions(transcript).values())

    if not positions:
        return 0.0

    return max(positions) - min(positions)


def agreement_score(transcript: Sequence[Message]) -> float:
    """1 - spread, so higher means more consensus. Vacuously 1.0 when empty."""
    return round(1 - position_spread(transcript), 2)


def average_policy(transcript: Sequence[Message]) -> float:
    """Mean of the latest positions. Vacuously 0.0 when empty."""
    positions = list(final_positions(transcript).values())

    if not positions:
        return 0.0

    return round(sum(positions) / len(positions), 2)
