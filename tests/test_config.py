"""Configuration resolution and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import main
from simulation.config import DEFAULT_AGENTS, AgentSpec, SimulationConfig


def test_defaults_match_the_documented_prototype_behaviour():
    config = SimulationConfig()

    assert config.rounds == 5
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.7
    assert config.sensitivity == 0.20
    assert config.protocol == "simultaneous"
    assert config.max_retries == 2
    assert config.output_dir == Path("runs")
    assert config.agent_names == ["Government", "Industry", "NGO", "Scientist"]
    assert config.attempts_per_turn == 3


def test_config_is_frozen():
    config = SimulationConfig()
    with pytest.raises(ValidationError):
        config.rounds = 9


@pytest.mark.parametrize(
    "field, value",
    [
        ("rounds", 0),
        ("temperature", -1.0),
        ("temperature", 3.0),
        ("sensitivity", 1.5),
        ("max_retries", -1),
        ("max_retries", 99),
        ("protocol", "round-robin"),
    ],
)
def test_out_of_range_values_are_rejected(field, value):
    with pytest.raises(ValidationError):
        SimulationConfig(**{field: value})


def test_unknown_field_is_rejected():
    """A typo in a config file must fail at startup, not be ignored."""
    with pytest.raises(ValidationError):
        SimulationConfig(round=3)


def test_duplicate_agent_names_are_rejected():
    duplicate = DEFAULT_AGENTS[0]
    with pytest.raises(ValidationError):
        SimulationConfig(agents=(duplicate, duplicate))


def test_at_least_two_agents_are_required():
    with pytest.raises(ValidationError):
        SimulationConfig(agents=(DEFAULT_AGENTS[0],))


def test_agent_preferred_position_is_bounded():
    with pytest.raises(ValidationError):
        AgentSpec(name="X", role="r", goal="g", preferred_position=1.4)


def test_from_file_reads_json(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps(
            {
                "rounds": 3,
                "model": "custom-model",
                "protocol": "sequential",
                "agents": [
                    {"name": "A", "role": "r", "goal": "g", "preferred_position": 0.2},
                    {"name": "B", "role": "r", "goal": "g", "preferred_position": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )

    config = SimulationConfig.from_file(path)

    assert config.rounds == 3
    assert config.model == "custom-model"
    assert config.protocol == "sequential"
    assert config.agent_names == ["A", "B"]
    # Unspecified fields keep their defaults.
    assert config.temperature == 0.7


def test_from_file_rejects_a_non_object(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        SimulationConfig.from_file(path)


def test_with_overrides_ignores_none():
    config = SimulationConfig()
    updated = config.with_overrides(rounds=7, model=None)

    assert updated.rounds == 7
    assert updated.model == config.model
    assert config.rounds == 5  # original untouched


@pytest.mark.parametrize(
    "field, value",
    [("rounds", 0), ("temperature", 5.0), ("max_retries", -1), ("sensitivity", 2.0)],
)
def test_with_overrides_revalidates(field, value):
    """Regression: model_copy would have let invalid CLI values through."""
    with pytest.raises(ValidationError):
        SimulationConfig().with_overrides(**{field: value})


def test_with_overrides_preserves_the_agent_roster():
    updated = SimulationConfig().with_overrides(rounds=2)
    assert updated.agents == DEFAULT_AGENTS


def test_to_json_dict_is_serialisable_and_complete():
    payload = SimulationConfig().to_json_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["output_dir"] == "runs"
    assert len(payload["agents"]) == 4


def test_to_json_dict_contains_no_credential_fields():
    payload = SimulationConfig().to_json_dict()
    joined = json.dumps(payload).lower()

    assert "api_key" not in joined
    assert "openai_api_key" not in joined
    assert "sk-" not in joined


# ------------------------------------------------------------------ CLI


def test_cli_defaults_resolve_without_arguments():
    args = main.build_arg_parser().parse_args([])
    config = main.resolve_config(args)

    assert config == SimulationConfig()


def test_cli_flags_override_defaults(tmp_path):
    args = main.build_arg_parser().parse_args(
        [
            "--rounds", "4",
            "--model", "m",
            "--temperature", "0.1",
            "--sensitivity", "0.05",
            "--protocol", "sequential",
            "--max-retries", "1",
            "--output-dir", str(tmp_path),
        ]
    )
    config = main.resolve_config(args)

    assert config.rounds == 4
    assert config.model == "m"
    assert config.temperature == 0.1
    assert config.sensitivity == 0.05
    assert config.protocol == "sequential"
    assert config.max_retries == 1
    assert config.output_dir == tmp_path


def test_cli_flags_win_over_the_config_file(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"rounds": 3, "model": "from-file"}), encoding="utf-8")

    args = main.build_arg_parser().parse_args(
        ["--config", str(path), "--rounds", "8"]
    )
    config = main.resolve_config(args)

    assert config.rounds == 8
    assert config.model == "from-file"
