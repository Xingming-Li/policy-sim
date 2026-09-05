"""Schema constraints, metrics, and the fake client itself."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models.message import AgentAction, Message, utcnow
from models.records import ResponseStatus, StatusCounts
from simulation.metrics import (
    agreement_score,
    average_policy,
    final_positions,
    position_spread,
)
from utils.fake_llm import FakeApiError, FakeLLM, ScriptExhausted, action
from utils.llm import classify_retryable


def message(sender: str, position: float, round_id: int = 1) -> Message:
    return Message(
        round_id=round_id,
        sender=sender,
        message_type="proposal",
        policy_position=position,
        content="text",
    )


# ----------------------------------------------------------- schemas


@pytest.mark.parametrize("position", [-0.01, 1.01, 2.0])
def test_agent_action_rejects_out_of_range_positions(position):
    with pytest.raises(ValidationError):
        AgentAction(message_type="proposal", policy_position=position, content="x")


@pytest.mark.parametrize("position", [0.0, 0.5, 1.0])
def test_agent_action_accepts_boundary_positions(position):
    assert AgentAction(
        message_type="proposal", policy_position=position, content="x"
    ).policy_position == position


def test_agent_action_strips_content():
    assert AgentAction(
        message_type="proposal", policy_position=0.5, content="  hi  "
    ).content == "hi"


def test_agent_action_ignores_unknown_keys():
    parsed = AgentAction(
        message_type="proposal", policy_position=0.5, content="x", confidence=0.9
    )
    assert not hasattr(parsed, "confidence")


def test_agent_action_cannot_be_a_fallback():
    with pytest.raises(ValidationError):
        AgentAction(message_type="fallback", policy_position=0.5, content="x")


def test_message_rejects_out_of_range_position():
    with pytest.raises(ValidationError):
        message("A", 1.4)


def test_message_requires_a_round_id():
    with pytest.raises(ValidationError):
        Message(
            sender="A", message_type="proposal", policy_position=0.5, content="x"
        )


def test_message_rejects_naive_timestamps():
    with pytest.raises(ValidationError):
        Message(
            round_id=1,
            sender="A",
            message_type="proposal",
            policy_position=0.5,
            content="x",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )


def test_message_accepts_aware_timestamps():
    stamped = Message(
        round_id=1,
        sender="A",
        message_type="proposal",
        policy_position=0.5,
        content="x",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert stamped.timestamp.tzinfo is not None


def test_utcnow_is_aware_and_utc():
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset().total_seconds() == 0


def test_status_counts_from_statuses():
    counts = StatusCounts.from_statuses(
        [
            ResponseStatus.OK,
            ResponseStatus.OK,
            ResponseStatus.INVALID_JSON,
            ResponseStatus.API_ERROR,
        ]
    )
    assert counts.ok == 2
    assert counts.invalid_json == 1
    assert counts.api_error == 1
    assert counts.schema_error == 0


# ----------------------------------------------------------- metrics


def test_final_positions_takes_the_latest_per_agent():
    transcript = [
        message("A", 0.2, 1),
        message("B", 0.8, 1),
        message("A", 0.4, 2),
    ]
    assert final_positions(transcript) == {"A": 0.4, "B": 0.8}


def test_spread_and_agreement():
    transcript = [message("A", 0.2), message("B", 0.8)]
    assert position_spread(transcript) == pytest.approx(0.6)
    assert agreement_score(transcript) == pytest.approx(0.4)


def test_full_consensus_scores_one():
    transcript = [message("A", 0.5), message("B", 0.5)]
    assert agreement_score(transcript) == 1.0


def test_average_policy():
    transcript = [message("A", 0.2), message("B", 0.8), message("C", 0.5)]
    assert average_policy(transcript) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "metric, expected",
    [(agreement_score, 1.0), (average_policy, 0.0), (position_spread, 0.0)],
)
def test_metrics_are_safe_on_an_empty_transcript(metric, expected):
    """Previously raised ValueError / ZeroDivisionError."""
    assert metric([]) == expected


def test_metrics_handle_a_single_agent():
    transcript = [message("A", 0.3)]
    assert agreement_score(transcript) == 1.0
    assert average_policy(transcript) == pytest.approx(0.3)


# --------------------------------------------------------- fake client


def test_fake_llm_replays_its_script_in_order():
    llm = FakeLLM([action(policy_position=0.1), action(policy_position=0.2)])

    first = llm.generate("p1")
    second = llm.generate("p2")

    assert '"policy_position": 0.1' in first.text
    assert '"policy_position": 0.2' in second.text
    assert llm.prompts == ["p1", "p2"]
    assert llm.call_count == 2


def test_fake_llm_reports_api_errors_without_raising():
    llm = FakeLLM([FakeApiError("boom", retryable=False)])

    response = llm.generate("p")

    assert response.ok is False
    assert response.retryable is False
    assert response.text is None
    assert "boom" in response.error_message


def test_fake_llm_loops_when_asked():
    llm = FakeLLM([action(policy_position=0.3)], loop=True)
    for _ in range(5):
        assert llm.generate("p").ok
    assert llm.call_count == 5


def test_fake_llm_exhaustion_is_loud():
    llm = FakeLLM([action()])
    llm.generate("p")
    with pytest.raises(ScriptExhausted):
        llm.generate("p")


def test_fake_llm_requires_a_script_or_responder():
    with pytest.raises(ValueError):
        FakeLLM()


def test_fake_llm_satisfies_the_client_interface():
    from utils.llm import LLMClient

    assert isinstance(FakeLLM([action()]), LLMClient)


# ----------------------------------------------------- error classification


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_transient_status_codes_are_retryable(status):
    exc = RuntimeError("x")
    exc.status_code = status
    assert classify_retryable(exc) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    exc = RuntimeError("x")
    exc.status_code = status
    assert classify_retryable(exc) is False


def test_transient_exception_names_are_retryable():
    class RateLimitError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    assert classify_retryable(RateLimitError()) is True
    assert classify_retryable(AuthenticationError()) is False
