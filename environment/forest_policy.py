"""Forest policy environment.

The dynamics are unchanged from the prototype: a linear, symmetric response to
the collective policy, clipped to [0, 1]. `SENSITIVITY` is now the default for
a constructor argument so that runs can record and vary it, and `state()`
exposes the values for logging. The update rule itself is untouched, and is
scheduled for replacement in a later phase.
"""

from __future__ import annotations


class ForestPolicyEnvironment:

    SENSITIVITY = 0.20

    def __init__(self, sensitivity: float | None = None):

        self.forest_health = 0.70

        self.economic_output = 0.70

        # Instance value shadows the class default; the formula is unchanged.
        self.sensitivity = (
            self.SENSITIVITY if sensitivity is None else float(sensitivity)
        )

    def state(self) -> dict[str, float]:
        """Snapshot for the run artifacts."""

        return {
            "forest_health": self.forest_health,
            "economic_output": self.economic_output,
        }

    def apply_policy(
        self,
        policy_position
    ):

        self.forest_health += (
            0.5 - policy_position
        ) * self.sensitivity

        self.economic_output += (
            policy_position - 0.5
        ) * self.sensitivity

        self.forest_health = max(
            0,
            min(1, self.forest_health)
        )

        self.economic_output = max(
            0,
            min(1, self.economic_output)
        )
