"""Offline LLM clients for tests and smoke runs.

Nothing in this module touches the network, reads credentials, or imports the
OpenAI SDK. `FakeLLM` replays a script; `make_demo_llm` builds a responder that
produces a plausible converging negotiation with injected faults, so
`python main.py --fake-llm` yields sample artifacts that exercise the retry and
fallback paths.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from utils.llm import LLMResponse


@dataclass(frozen=True)
class FakeApiError:
    """Script entry that simulates a transport failure."""

    message: str = "simulated API failure"
    error_type: str = "SimulatedAPIError"
    retryable: bool = True


class ScriptExhausted(AssertionError):
    """Raised when a non-looping script runs out of entries.

    An assertion rather than a normal error: it means the test under-specified
    its script, which should fail loudly instead of being retried.
    """


# A script entry is a dict (JSON-encoded), a raw string (returned verbatim),
# a FakeApiError, None (empty body), or a callable producing one of those.
ScriptEntry = dict[str, Any] | str | FakeApiError | None | Callable[[str, int], Any]


class FakeLLM:
    """Deterministic `LLMClient` implementation.

    Records every prompt it is given in `self.prompts`, so tests can assert on
    what the agent actually asked.
    """

    def __init__(
        self,
        script: Sequence[ScriptEntry] | None = None,
        *,
        responder: Callable[[str, int], Any] | None = None,
        loop: bool = False,
        model: str = "fake-llm",
        latency_ms: float = 0.0,
    ) -> None:
        if script is None and responder is None:
            raise ValueError("FakeLLM needs either a script or a responder")
        self._script: list[ScriptEntry] = list(script or [])
        self._responder = responder
        self._loop = loop
        self.model = model
        self.latency_ms = latency_ms
        self.prompts: list[str] = []
        self.call_count = 0

    def _next_entry(self, prompt: str, index: int) -> Any:
        if self._responder is not None:
            return self._responder(prompt, index)
        if not self._script:
            raise ScriptExhausted("FakeLLM script is empty")
        if index >= len(self._script) and not self._loop:
            raise ScriptExhausted(
                f"FakeLLM script exhausted after {len(self._script)} entries "
                f"(call {index + 1} requested)"
            )
        return self._script[index % len(self._script)]

    def generate(self, prompt: str) -> LLMResponse:
        index = self.call_count
        self.call_count += 1
        self.prompts.append(prompt)

        entry = self._next_entry(prompt, index)
        if callable(entry) and not isinstance(entry, FakeApiError):
            entry = entry(prompt, index)

        if isinstance(entry, FakeApiError):
            return LLMResponse.failure(
                error_type=entry.error_type,
                error_message=entry.message,
                latency_ms=self.latency_ms,
                retryable=entry.retryable,
            )

        if entry is None:
            return LLMResponse.success(None, self.latency_ms)

        text = entry if isinstance(entry, str) else json.dumps(entry)
        return LLMResponse.success(
            text,
            self.latency_ms,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
        )


def action(
    *,
    target: str | None = None,
    message_type: str = "proposal",
    policy_position: float = 0.5,
    content: str = "Proposing a balanced position.",
) -> dict[str, Any]:
    """Build a well-formed action dict for use in scripts."""
    return {
        "target": target,
        "message_type": message_type,
        "policy_position": policy_position,
        "content": content,
    }


_SELF_NAME = re.compile(r"^You are (.+?)\.\s*$", re.MULTILINE)
_CURRENT = re.compile(r"Your current position:\s*([0-9]*\.?[0-9]+)")
_GROUP_AVERAGE = re.compile(r"average position is\s+([0-9]*\.?[0-9]+)")
_TARGETS = re.compile(r"exactly one of:\s*(.+)")


def _demo_action(prompt: str) -> dict[str, Any]:
    """Derive a plausible concession from the rendered prompt.

    Reads the agent's own position and the group average out of the prompt and
    steps 20% of the way toward the average, which gives a readable converging
    trajectory in the sample artifacts.
    """
    name_match = _SELF_NAME.search(prompt)
    name = name_match.group(1) if name_match else "unknown"

    current = float(_CURRENT.search(prompt).group(1)) if _CURRENT.search(prompt) else 0.5
    average_match = _GROUP_AVERAGE.search(prompt)
    average = float(average_match.group(1)) if average_match else current

    position = round(min(1.0, max(0.0, current + 0.2 * (average - current))), 2)

    targets_match = _TARGETS.search(prompt)
    targets = (
        [t.strip() for t in targets_match.group(1).split(",") if t.strip()]
        if targets_match
        else []
    )

    return action(
        target=targets[0] if targets else None,
        message_type="proposal" if position != current else "oppose",
        policy_position=position,
        content=f"{name} moves to {position:.2f} to close the gap.",
    )


def make_demo_llm(
    *,
    malformed_calls: Iterable[int] = (2,),
    api_error_calls: Iterable[int] = (11, 12, 13),
) -> FakeLLM:
    """An offline client that mimics a real run, including failures.

    `malformed_calls` return unparseable text (recovered by retry).
    `api_error_calls` fail transport; when they cover every attempt of a turn,
    that turn ends in a recorded fallback.
    """
    malformed = set(malformed_calls)
    api_errors = set(api_error_calls)

    def responder(prompt: str, index: int) -> Any:
        if index in api_errors:
            return FakeApiError(f"simulated transport failure on call {index}")
        if index in malformed:
            return "{'not': valid json,"
        return _demo_action(prompt)

    return FakeLLM(responder=responder, model="fake-llm-demo")
