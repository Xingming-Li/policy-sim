"""End-to-end runs and the artifacts they produce. No network, no API key."""

from __future__ import annotations

import json

import pytest

from agents.base_agent import Agent
from conftest import TEST_AGENTS
from environment.forest_policy import ForestPolicyEnvironment
from models.message import FALLBACK_MESSAGE_TYPE
from simulation.runlog import (
    CONFIG_FILENAME,
    MESSAGES_FILENAME,
    ROUNDS_FILENAME,
    SUMMARY_FILENAME,
    read_jsonl,
)
from simulation.simulation import Simulation
from utils.fake_llm import FakeApiError, FakeLLM, action

MESSAGE_ROW_FIELDS = {
    "round_id",
    "sender",
    "target",
    "message_type",
    "policy_position",
    "content",
    "timestamp",
    "status",
    "fallback_used",
    "attempts",
    "retries",
    "latency_ms",
    "raw_response",
    "error",
    "attempt_log",
    "prompt_tokens",
    "completion_tokens",
}

ROUND_ROW_FIELDS = {
    "round_id",
    "agent_order",
    "positions",
    "context_group_average",
    "context_spread",
    "environment_before",
    "environment_after",
    "applied_policy",
    "average_policy",
    "spread",
    "agreement_score",
    "fallback_count",
    "failed_attempt_count",
    "timestamp",
}


def build_simulation(config, llm, environment=None):
    environment = environment or ForestPolicyEnvironment(
        sensitivity=config.sensitivity
    )
    agents = [Agent(spec, llm, max_retries=config.max_retries) for spec in config.agents]
    return Simulation(agents, environment, config)


def converging_llm(positions=(0.5, 0.6, 0.4)):
    """One valid response per agent, cycled across rounds."""
    return FakeLLM(
        [action(policy_position=p, content=f"Proposing {p}.") for p in positions],
        loop=True,
    )


# ----------------------------------------------------------- happy path


def test_multi_round_run_completes(config, tmp_path):
    result = build_simulation(config, converging_llm()).run()

    assert result.summary is not None
    assert result.summary.rounds_completed == config.rounds
    assert len(result.rounds) == config.rounds
    # One turn per agent per round.
    assert len(result.transcript) == config.rounds * len(config.agents)
    assert len(result.records) == len(result.transcript)
    assert result.summary.fallback_count == 0
    assert result.summary.turn_status_counts.ok == len(result.records)


def test_round_ids_and_speakers_are_ordered(config):
    result = build_simulation(config, converging_llm()).run()

    names = [spec.name for spec in config.agents]
    assert [m.round_id for m in result.transcript] == [1, 1, 1, 2, 2, 2]
    assert [m.sender for m in result.transcript] == names * config.rounds
    assert all(r.agent_order == tuple(names) for r in result.rounds)


def test_environment_advances_with_the_applied_policy(config):
    result = build_simulation(config, converging_llm((0.9, 0.9, 0.9))).run()

    first = result.rounds[0]
    assert first.applied_policy == pytest.approx(0.9)
    assert first.environment_before["forest_health"] == pytest.approx(0.70)
    # health += (0.5 - 0.9) * 0.2 = -0.08
    assert first.environment_after["forest_health"] == pytest.approx(0.62)
    assert first.environment_after["economic_output"] == pytest.approx(0.78)
    assert result.summary.final_environment == result.rounds[-1].environment_after


def test_metrics_are_recorded_per_round(config):
    result = build_simulation(config, converging_llm((0.4, 0.6, 0.5))).run()

    first = result.rounds[0]
    assert first.positions == {"Alpha": 0.4, "Beta": 0.6, "Gamma": 0.5}
    assert first.spread == pytest.approx(0.2)
    assert first.agreement_score == pytest.approx(0.8)
    assert first.average_policy == pytest.approx(0.5)


def test_agents_hold_configured_starting_positions(config):
    """Round 1 group statistics come from the configured preferences."""
    result = build_simulation(config, converging_llm()).run()

    first = result.rounds[0]
    expected = [spec.preferred_position for spec in TEST_AGENTS]
    assert first.context_group_average == pytest.approx(sum(expected) / len(expected))
    assert first.context_spread == pytest.approx(max(expected) - min(expected))


# ------------------------------------------------------------ protocols


def test_simultaneous_protocol_freezes_context_within_a_round(config):
    llm = converging_llm()
    build_simulation(config, llm).run()

    round_one = llm.prompts[: len(config.agents)]
    # Same frozen group statistics for every agent in the round.
    assert all("current average position is\n0.47" in p for p in round_one)
    # No agent in round 1 sees another agent's round-1 message.
    assert all("(no discussion yet)" in p for p in round_one)


def test_sequential_protocol_exposes_earlier_turns_in_the_same_round(config):
    sequential = config.model_copy(update={"protocol": "sequential"})
    llm = converging_llm()
    build_simulation(sequential, llm).run()

    first, second = llm.prompts[0], llm.prompts[1]
    assert "(no discussion yet)" in first
    # The second agent sees what the first just said.
    assert "Alpha: Proposing 0.5." in second


def test_protocol_is_recorded_in_the_summary(config):
    result = build_simulation(config, converging_llm()).run()
    assert result.summary.protocol == "simultaneous"


def test_summary_distinguishes_the_configured_model_from_the_actual_client(config):
    """An offline run must not look like a real model run."""
    result = build_simulation(config, converging_llm()).run()

    assert result.summary.model == "test-model"  # what was configured
    assert result.summary.llm_client == "FakeLLM"  # what actually answered
    assert result.summary.llm_model == "fake-llm"


# ------------------------------------------------- failures in a run


def test_retries_and_fallbacks_are_counted_in_the_summary(config):
    # Alpha succeeds, Beta recovers after one malformed reply, Gamma exhausts.
    llm = FakeLLM(
        responder=lambda prompt, index: (
            action(policy_position=0.5)
            if "You are Alpha." in prompt
            else "{broken"
            if "You are Gamma." in prompt
            else ("{broken" if index % 2 == 1 else action(policy_position=0.7))
        )
    )
    result = build_simulation(config, llm).run()

    summary = result.summary
    assert summary.fallback_count == config.rounds  # Gamma, once per round
    assert summary.total_agent_turns == config.rounds * 3
    assert summary.fallback_rate == pytest.approx(1 / 3)
    assert summary.total_retries > 0
    assert summary.attempt_status_counts.invalid_json > 0
    assert summary.attempt_status_counts.ok > 0
    assert summary.turn_status_counts.ok == config.rounds * 2


def test_fallback_turns_are_marked_in_the_transcript(config):
    llm = FakeLLM(
        responder=lambda prompt, index: (
            "{broken" if "You are Gamma." in prompt else action(policy_position=0.5)
        )
    )
    result = build_simulation(config, llm).run()

    gamma = [m for m in result.transcript if m.sender == "Gamma"]
    others = [m for m in result.transcript if m.sender != "Gamma"]
    assert all(m.message_type == FALLBACK_MESSAGE_TYPE for m in gamma)
    assert all(m.message_type != FALLBACK_MESSAGE_TYPE for m in others)
    # A held position is still a position, so the round completes.
    assert all(m.policy_position == 0.1 for m in gamma)


def test_run_with_every_turn_failing_still_produces_artifacts(config):
    llm = FakeLLM([FakeApiError(retryable=True)], loop=True)
    result = build_simulation(config, llm).run()

    assert result.summary.fallback_rate == 1.0
    assert result.summary.turn_status_counts.ok == 0
    assert (result.output_dir / SUMMARY_FILENAME).exists()
    assert len(read_jsonl(result.output_dir / MESSAGES_FILENAME)) == 6


# ------------------------------------------------------------ artifacts


def test_run_directory_contains_all_four_artifacts(config):
    result = build_simulation(config, converging_llm()).run()

    for filename in (
        CONFIG_FILENAME,
        MESSAGES_FILENAME,
        ROUNDS_FILENAME,
        SUMMARY_FILENAME,
    ):
        assert (result.output_dir / filename).is_file(), filename

    assert result.output_dir.name == result.run_id
    assert result.output_dir.parent == config.output_dir


def test_run_ids_are_unique(config):
    first = build_simulation(config, converging_llm()).run()
    second = build_simulation(config, converging_llm()).run()

    assert first.run_id != second.run_id
    assert first.output_dir != second.output_dir


def test_resolved_config_round_trips(config):
    result = build_simulation(config, converging_llm()).run()

    saved = json.loads((result.output_dir / CONFIG_FILENAME).read_text("utf-8"))
    assert saved["rounds"] == config.rounds
    assert saved["model"] == "test-model"
    assert saved["protocol"] == "simultaneous"
    assert saved["sensitivity"] == config.sensitivity
    assert saved["max_retries"] == config.max_retries
    assert [a["name"] for a in saved["agents"]] == [s.name for s in config.agents]
    # Every field is present, so the file alone reproduces the run.
    assert set(saved) == set(config.model_dump())


def test_messages_jsonl_has_the_documented_fields(config):
    result = build_simulation(config, converging_llm()).run()

    rows = read_jsonl(result.output_dir / MESSAGES_FILENAME)
    assert len(rows) == 6
    for row in rows:
        assert set(row) == MESSAGE_ROW_FIELDS
    first = rows[0]
    assert first["sender"] == "Alpha"
    assert first["round_id"] == 1
    assert first["status"] == "ok"
    assert first["fallback_used"] is False
    assert first["timestamp"].endswith("Z") or "+00:00" in first["timestamp"]
    assert json.loads(first["raw_response"])["policy_position"] == 0.5


def test_message_rows_include_the_full_attempt_log(config):
    llm = FakeLLM(["{broken", action(policy_position=0.5)], loop=True)
    config = config.model_copy(update={"rounds": 1})
    result = build_simulation(config, llm).run()

    rows = read_jsonl(result.output_dir / MESSAGES_FILENAME)
    attempts = rows[0]["attempt_log"]
    assert [a["status"] for a in attempts] == ["invalid_json", "ok"]
    assert attempts[0]["raw_response"] == "{broken"
    assert rows[0]["retries"] == 1


def test_rounds_jsonl_has_the_documented_fields(config):
    result = build_simulation(config, converging_llm()).run()

    rows = read_jsonl(result.output_dir / ROUNDS_FILENAME)
    assert len(rows) == config.rounds
    for row in rows:
        assert set(row) == ROUND_ROW_FIELDS
    assert [row["round_id"] for row in rows] == [1, 2]
    assert rows[0]["agent_order"] == ["Alpha", "Beta", "Gamma"]
    assert set(rows[0]["positions"]) == {"Alpha", "Beta", "Gamma"}
    assert set(rows[0]["environment_after"]) == {"forest_health", "economic_output"}


def test_summary_json_reports_metrics_and_counters(config):
    result = build_simulation(config, converging_llm()).run()

    summary = json.loads((result.output_dir / SUMMARY_FILENAME).read_text("utf-8"))
    assert summary["run_id"] == result.run_id
    assert summary["rounds_completed"] == config.rounds
    assert summary["fallback_count"] == 0
    assert summary["total_llm_calls"] == 6
    assert summary["total_retries"] == 0
    assert set(summary["final_positions"]) == {"Alpha", "Beta", "Gamma"}
    assert summary["turn_status_counts"] == {
        "ok": 6,
        "api_error": 0,
        "invalid_json": 0,
        "schema_error": 0,
    }
    assert summary["runtime"]["python_version"]
    assert summary["duration_seconds"] >= 0.0


def test_artifacts_are_written_incrementally(config, monkeypatch):
    """An interrupted run must leave the already-decided turns on disk."""

    class Boom(RuntimeError):
        pass

    original = Agent.decide
    calls = {"n": 0}

    def flaky(self, context):
        calls["n"] += 1
        if calls["n"] == 5:  # part-way through round 2
            raise Boom("interrupted")
        return original(self, context)

    monkeypatch.setattr(Agent, "decide", flaky)

    simulation = build_simulation(config, converging_llm())
    with pytest.raises(Boom):
        simulation.run()

    run_dir = config.output_dir / simulation.run_id
    assert len(read_jsonl(run_dir / MESSAGES_FILENAME)) == 4
    assert len(read_jsonl(run_dir / ROUNDS_FILENAME)) == 1
    # No summary: the run never finished, and nothing pretends it did.
    assert not (run_dir / SUMMARY_FILENAME).exists()


# --------------------------------------------------------- no secrets


def test_artifacts_contain_no_credentials(config, monkeypatch):
    sentinel = "sk-do-not-log-this-key"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)

    result = build_simulation(config, converging_llm()).run()

    for path in result.output_dir.iterdir():
        assert sentinel not in path.read_text("utf-8"), path.name


def test_run_needs_no_api_key(config):
    """The whole suite runs with OPENAI_API_KEY unset (see conftest)."""
    import os

    assert "OPENAI_API_KEY" not in os.environ
    result = build_simulation(config, converging_llm()).run()
    assert result.summary.turn_status_counts.ok == 6
