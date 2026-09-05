"""Agent turn execution: validation, classification, retries, fallback."""

from __future__ import annotations

import pytest

from conftest import make_agent, make_context
from models.message import FALLBACK_MESSAGE_TYPE
from models.records import ResponseStatus
from utils.fake_llm import FakeApiError, action

# ---------------------------------------------------------------- valid


def test_valid_response_is_accepted(agent_spec, environment):
    agent = make_agent(
        agent_spec,
        [action(target="Beta", message_type="proposal", policy_position=0.55,
                content="Meet near the middle.")],
    )

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.OK
    assert record.fallback_used is False
    assert record.attempts == 1
    assert record.retries == 0
    assert record.message.policy_position == 0.55
    assert record.message.target == "Beta"
    assert record.message.message_type == "proposal"
    assert record.message.content == "Meet near the middle."
    assert record.message.round_id == 1
    # The agent's own state advances to the accepted position.
    assert agent.current_position == 0.55


def test_valid_response_records_raw_text_and_latency(agent_spec, environment):
    agent = make_agent(agent_spec, [action(policy_position=0.4)])

    record = agent.decide(make_context(environment))

    assert record.raw_response is not None
    assert '"policy_position": 0.4' in record.raw_response
    assert record.latency_ms >= 0.0
    assert len(record.attempt_log) == 1
    assert record.attempt_log[0].status is ResponseStatus.OK


def test_timestamp_is_timezone_aware(agent_spec, environment):
    agent = make_agent(agent_spec, [action()])

    record = agent.decide(make_context(environment))

    assert record.message.timestamp.tzinfo is not None
    assert record.message.timestamp.utcoffset().total_seconds() == 0


def test_null_target_is_valid(agent_spec, environment):
    agent = make_agent(agent_spec, [action(target=None)])

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.OK
    assert record.message.target is None


def test_target_and_type_are_case_normalised(agent_spec, environment):
    agent = make_agent(
        agent_spec,
        [action(target="  beta ", message_type="OPPOSE")],
    )

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.OK
    assert record.message.target == "Beta"
    assert record.message.message_type == "oppose"


# ------------------------------------------------------------ malformed


def test_malformed_json_is_classified_and_retried(agent_spec, environment):
    agent = make_agent(agent_spec, ["{not valid json,", action(policy_position=0.6)])

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.OK
    assert record.fallback_used is False
    assert record.attempts == 2
    assert record.retries == 1
    assert record.attempt_log[0].status is ResponseStatus.INVALID_JSON
    assert "json decode failed" in record.attempt_log[0].error
    # The unusable text is preserved for auditing.
    assert record.attempt_log[0].raw_response == "{not valid json,"
    assert record.message.policy_position == 0.6


def test_empty_response_body_is_invalid_json(agent_spec, environment):
    agent = make_agent(agent_spec, [None, action()])

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.INVALID_JSON
    assert record.attempt_log[0].error == "empty response body"
    assert record.status is ResponseStatus.OK


def test_valid_json_that_is_not_an_object_is_a_schema_error(agent_spec, environment):
    agent = make_agent(agent_spec, ["[1, 2, 3]", action()])

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.SCHEMA_ERROR
    assert "expected a JSON object" in record.attempt_log[0].error


# ------------------------------------------------------- invalid fields


@pytest.mark.parametrize("bad_position", [1.5, -0.2, 42])
def test_out_of_range_position_is_a_schema_error(agent_spec, environment, bad_position):
    """Out-of-range values are reported, not silently clamped."""
    agent = make_agent(
        agent_spec,
        [action(policy_position=bad_position), action(policy_position=0.5)],
    )

    record = agent.decide(make_context(environment))

    first = record.attempt_log[0]
    assert first.status is ResponseStatus.SCHEMA_ERROR
    assert "policy_position" in first.error
    assert record.status is ResponseStatus.OK
    assert record.message.policy_position == 0.5


def test_non_numeric_position_is_a_schema_error(agent_spec, environment):
    agent = make_agent(agent_spec, [action(policy_position="not-a-number"), action()])

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.SCHEMA_ERROR
    assert "policy_position" in record.attempt_log[0].error


@pytest.mark.parametrize(
    "payload, expected_field",
    [
        ({"message_type": "proposal", "content": "hi"}, "policy_position"),
        ({"policy_position": 0.5, "content": "hi"}, "message_type"),
        ({"message_type": "proposal", "policy_position": 0.5}, "content"),
    ],
)
def test_missing_required_fields_are_schema_errors(
    agent_spec, environment, payload, expected_field
):
    agent = make_agent(agent_spec, [payload, action()])

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.SCHEMA_ERROR
    assert expected_field in record.attempt_log[0].error


def test_blank_content_is_a_schema_error(agent_spec, environment):
    agent = make_agent(agent_spec, [action(content="   "), action()])

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.SCHEMA_ERROR


def test_unknown_message_type_is_a_schema_error(agent_spec, environment):
    """Includes the case where the model echoes the prompt's placeholder."""
    agent = make_agent(
        agent_spec,
        [action(message_type="proposal|support|oppose|evidence"), action()],
    )

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.SCHEMA_ERROR
    assert "message_type" in record.attempt_log[0].error


def test_unknown_target_is_a_schema_error(agent_spec, environment):
    agent = make_agent(agent_spec, [action(target="Lobbyist"), action()])

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.SCHEMA_ERROR
    assert "Lobbyist" in record.attempt_log[0].error


def test_self_target_is_a_schema_error(agent_spec, environment):
    agent = make_agent(agent_spec, [action(target="Alpha"), action()])

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.SCHEMA_ERROR
    assert "must not be the sender" in record.attempt_log[0].error


def test_fallback_type_cannot_be_produced_by_the_model(agent_spec, environment):
    agent = make_agent(
        agent_spec, [action(message_type=FALLBACK_MESSAGE_TYPE), action()]
    )

    record = agent.decide(make_context(environment))

    assert record.attempt_log[0].status is ResponseStatus.SCHEMA_ERROR


# ------------------------------------------------------------- retries


def test_transient_api_error_is_retried(agent_spec, environment):
    agent = make_agent(
        agent_spec,
        [FakeApiError("rate limited", retryable=True), action(policy_position=0.45)],
    )

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.OK
    assert record.attempts == 2
    assert record.attempt_log[0].status is ResponseStatus.API_ERROR
    assert record.attempt_log[0].retryable is True
    assert record.attempt_log[0].raw_response is None


def test_non_retryable_api_error_stops_immediately(agent_spec, environment):
    agent = make_agent(
        agent_spec,
        [
            FakeApiError("invalid api key", error_type="AuthenticationError",
                         retryable=False),
            action(),
        ],
    )

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.API_ERROR
    assert record.fallback_used is True
    # Only one call: a permanent error is not retried.
    assert record.attempts == 1
    assert agent.llm.call_count == 1


def test_retry_budget_is_bounded_by_max_retries(agent_spec, environment):
    agent = make_agent(
        agent_spec,
        [FakeApiError(retryable=True)],
        max_retries=2,
        loop=True,
    )

    record = agent.decide(make_context(environment))

    assert agent.llm.call_count == 3  # 1 initial + 2 retries
    assert record.attempts == 3
    assert record.retries == 2


def test_zero_retries_means_a_single_attempt(agent_spec, environment):
    agent = make_agent(agent_spec, ["nonsense"], max_retries=0)

    record = agent.decide(make_context(environment))

    assert agent.llm.call_count == 1
    assert record.fallback_used is True
    assert record.status is ResponseStatus.INVALID_JSON


# ------------------------------------------------------------ fallback


def test_exhausted_retries_produce_a_clear_failure_result(agent_spec, environment):
    agent = make_agent(agent_spec, ["still not json"], max_retries=2, loop=True)
    agent.current_position = 0.5

    record = agent.decide(make_context(environment))

    assert record.fallback_used is True
    assert record.status is ResponseStatus.INVALID_JSON
    assert record.attempts == 3
    assert record.retries == 2
    # The message is unmistakably not a model action.
    assert record.message.message_type == FALLBACK_MESSAGE_TYPE
    assert record.message.is_fallback is True
    assert "NO VALID MODEL RESPONSE" in record.message.content
    assert "invalid_json" in record.message.content
    assert record.message.target is None
    # Position is held, not invented.
    assert record.message.policy_position == 0.5
    assert agent.current_position == 0.5
    # Every failed attempt is on record.
    assert len(record.attempt_log) == 3
    assert all(a.status is ResponseStatus.INVALID_JSON for a in record.attempt_log)


def test_fallback_after_api_errors_reports_api_error_status(agent_spec, environment):
    agent = make_agent(
        agent_spec, [FakeApiError(retryable=True)], max_retries=1, loop=True
    )

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.API_ERROR
    assert "api_error" in record.message.content


def test_mixed_failures_then_success_records_each_attempt(agent_spec, environment):
    agent = make_agent(
        agent_spec,
        [
            FakeApiError(retryable=True),
            "{broken",
            action(policy_position=0.3),
        ],
        max_retries=2,
    )

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.OK
    assert record.fallback_used is False
    assert [a.status for a in record.attempt_log] == [
        ResponseStatus.API_ERROR,
        ResponseStatus.INVALID_JSON,
        ResponseStatus.OK,
    ]
    assert [a.attempt for a in record.attempt_log] == [1, 2, 3]


# -------------------------------------------------------------- prompt


def test_prompt_is_rendered_once_and_reused_across_retries(agent_spec, environment):
    agent = make_agent(agent_spec, ["{bad", "{bad", action()])

    agent.decide(make_context(environment))

    assert len(set(agent.llm.prompts)) == 1


def test_prompt_contains_role_state_and_roster(agent_spec, environment):
    agent = make_agent(agent_spec, [action()])
    agent.current_position = 0.42

    prompt = agent.build_prompt(make_context(environment, round_id=3, total_rounds=7))

    assert "You are Alpha." in prompt
    assert "Your current position: 0.42" in prompt
    assert "This is round 3 of 7." in prompt
    assert "forest_health = 0.70" in prompt
    # Roster excludes the agent itself.
    assert "exactly one of: Beta, Gamma" in prompt
