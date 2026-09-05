"""Environment dynamics. Behaviour is unchanged from the prototype."""

from __future__ import annotations

import pytest

from environment.forest_policy import ForestPolicyEnvironment


def test_policy():
    env = ForestPolicyEnvironment()

    old_value = env.economic_output

    env.apply_policy(0.9)

    assert env.economic_output > old_value


def test_default_sensitivity_matches_the_class_constant():
    assert ForestPolicyEnvironment().sensitivity == ForestPolicyEnvironment.SENSITIVITY


def test_configured_sensitivity_scales_the_response():
    weak = ForestPolicyEnvironment(sensitivity=0.05)
    strong = ForestPolicyEnvironment(sensitivity=0.5)

    weak.apply_policy(1.0)
    strong.apply_policy(1.0)

    assert weak.economic_output == pytest.approx(0.725)
    assert strong.economic_output == pytest.approx(0.95)


def test_conservation_and_logging_move_state_in_opposite_directions():
    env = ForestPolicyEnvironment(sensitivity=0.2)

    env.apply_policy(0.0)

    assert env.forest_health == pytest.approx(0.80)
    assert env.economic_output == pytest.approx(0.60)


def test_balanced_policy_is_a_no_op():
    env = ForestPolicyEnvironment(sensitivity=0.2)

    env.apply_policy(0.5)

    assert env.forest_health == pytest.approx(0.70)
    assert env.economic_output == pytest.approx(0.70)


def test_state_is_clipped_to_the_unit_interval():
    env = ForestPolicyEnvironment(sensitivity=0.5)

    for _ in range(10):
        env.apply_policy(1.0)

    assert env.forest_health == 0
    assert env.economic_output == 1


def test_state_snapshot_reports_both_variables():
    env = ForestPolicyEnvironment(sensitivity=0.2)

    assert env.state() == {"forest_health": 0.70, "economic_output": 0.70}
