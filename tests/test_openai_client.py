"""The production client, exercised against a stubbed SDK.

`openai` need not be installed for these to run: a fake module is injected
into `sys.modules`. No network call is ever made.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from models.records import ResponseStatus
from simulation.config import AgentSpec
from utils.llm import OpenAIClient


@dataclass
class StubMessage:
    content: str | None


@dataclass
class StubChoice:
    message: StubMessage


@dataclass
class StubUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class StubCompletion:
    choices: list
    usage: StubUsage | None


class StubCompletions:
    def __init__(self, outcome):
        self._outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class StubOpenAI:
    last_instance = None

    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        outcome = StubOpenAI.outcome
        self.chat = types.SimpleNamespace(completions=StubCompletions(outcome))
        StubOpenAI.last_instance = self


@pytest.fixture
def stub_sdk(monkeypatch):
    """Install a fake `openai` module and control what it returns."""

    def install(outcome):
        StubOpenAI.outcome = outcome
        module = types.ModuleType("openai")
        module.OpenAI = StubOpenAI
        monkeypatch.setitem(sys.modules, "openai", module)
        return StubOpenAI

    return install


def ok_completion(text="{}", prompt_tokens=11, completion_tokens=7):
    return StubCompletion(
        choices=[StubChoice(StubMessage(text))],
        usage=StubUsage(prompt_tokens, completion_tokens),
    )


def test_successful_call_returns_text_and_usage(stub_sdk):
    stub_sdk(ok_completion('{"policy_position": 0.5}'))
    client = OpenAIClient(model="m", temperature=0.3, load_env=False)

    response = client.generate("hello")

    assert response.ok is True
    assert response.text == '{"policy_position": 0.5}'
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7
    assert response.latency_ms >= 0.0


def test_request_uses_configured_model_and_json_mode(stub_sdk):
    stub = stub_sdk(ok_completion())
    OpenAIClient(model="my-model", temperature=0.25, load_env=False).generate("hi")

    kwargs = stub.last_instance.chat.completions.calls[0]
    assert kwargs["model"] == "my-model"
    assert kwargs["temperature"] == 0.25
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_missing_usage_is_tolerated(stub_sdk):
    stub_sdk(StubCompletion(choices=[StubChoice(StubMessage("{}"))], usage=None))
    response = OpenAIClient(load_env=False).generate("hi")

    assert response.ok is True
    assert response.prompt_tokens is None


def test_transient_exception_is_reported_as_retryable(stub_sdk):
    error = RuntimeError("rate limited")
    error.status_code = 429
    stub_sdk(error)

    response = OpenAIClient(load_env=False).generate("hi")

    assert response.ok is False
    assert response.retryable is True
    assert response.error_type == "RuntimeError"
    assert "rate limited" in response.error_message
    assert response.text is None


def test_auth_exception_is_reported_as_permanent(stub_sdk):
    error = RuntimeError("invalid key")
    error.status_code = 401
    stub_sdk(error)

    response = OpenAIClient(load_env=False).generate("hi")

    assert response.ok is False
    assert response.retryable is False


def test_client_never_raises_on_api_failure(stub_sdk):
    stub_sdk(RuntimeError("total collapse"))
    # No exception escapes: the agent decides what to do about it.
    assert OpenAIClient(load_env=False).generate("hi").ok is False


def test_error_message_is_truncated(stub_sdk):
    stub_sdk(RuntimeError("x" * 5000))
    response = OpenAIClient(load_env=False).generate("hi")
    assert len(response.error_message) <= 500


def test_agent_consumes_the_production_client(stub_sdk, environment):
    """The real client satisfies the interface the agent depends on."""
    from agents.base_agent import Agent
    from conftest import make_context

    stub_sdk(
        ok_completion(
            '{"target": "Beta", "message_type": "support", '
            '"policy_position": 0.42, "content": "Agreed."}'
        )
    )
    client = OpenAIClient(model="m", load_env=False)
    agent = Agent(
        AgentSpec(name="Alpha", role="r", goal="g", preferred_position=0.5),
        client,
    )

    record = agent.decide(make_context(environment))

    assert record.status is ResponseStatus.OK
    assert record.message.policy_position == 0.42
    assert record.message.target == "Beta"


def test_explicit_api_key_is_passed_to_the_sdk_and_not_stored(stub_sdk):
    stub = stub_sdk(ok_completion())
    client = OpenAIClient(api_key="sk-test-key", load_env=False)

    assert stub.last_instance.api_key == "sk-test-key"
    # The key is not retained on our own object.
    assert "sk-test-key" not in repr(vars(client))


def test_missing_sdk_raises_a_helpful_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(RuntimeError, match="--fake-llm"):
        OpenAIClient(load_env=False)
