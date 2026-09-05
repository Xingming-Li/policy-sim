"""LLM transport layer

`LLMClient` is the interface the simulation depends on. `OpenAIClient` is the
production implementation. `utils.fake_llm.FakeLLM` is the offline one used by
the tests. Neither raises on API failure: both return an `LLMResponse` whose
`ok` flag and `retryable` flag let the caller decide whether to retry.

The OpenAI SDK and python-dotenv are imported lazily inside `OpenAIClient`, so
this module (and therefore the whole test suite) imports cleanly without those
packages installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# Exception class names the OpenAI SDK uses for conditions worth retrying.
_TRANSIENT_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ConflictError",
    }
)

_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429})


@dataclass(frozen=True)
class LLMResponse:
    """Result of one attempted generation.

    `ok=True` means transport succeeded and `text` holds the model
    output. It says nothing about whether that text is valid JSON.
    """

    ok: bool
    text: str | None
    latency_ms: float
    retryable: bool = False
    error_type: str | None = None
    error_message: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @classmethod
    def success(
        cls,
        text: str | None,
        latency_ms: float,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> LLMResponse:
        return cls(
            ok=True,
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @classmethod
    def failure(
        cls,
        error_type: str,
        error_message: str,
        latency_ms: float,
        *,
        retryable: bool,
    ) -> LLMResponse:
        return cls(
            ok=False,
            text=None,
            latency_ms=latency_ms,
            retryable=retryable,
            error_type=error_type,
            error_message=error_message,
        )


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract the simulation needs from a language model."""

    def generate(self, prompt: str) -> LLMResponse:
        """Produce one completion. Must not raise on API failure."""
        ...


def classify_retryable(exc: BaseException) -> bool:
    """Decide whether an SDK exception is worth retrying.

    Status-code driven where available (rate limits, timeouts),
    otherwise by exception class name. Auth errors and malformed
    requests are permanent and are not retried.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS_CODES or status >= 500
    return type(exc).__name__ in _TRANSIENT_EXCEPTION_NAMES


class OpenAIClient:
    """OpenAI chat-completions client in JSON mode."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        *,
        api_key: str | None = None,
        load_env: bool = True,
    ) -> None:
        if load_env:
            try:
                from dotenv import load_dotenv
            except ImportError:
                pass
            else:
                load_dotenv()

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "the 'openai' package is required for OpenAIClient; "
                "install it with 'pip install -r requirements.txt', or run "
                "with --fake-llm for an offline run"
            ) from exc

        self.model = model
        self.temperature = temperature
        # Key resolution is delegated to the SDK's own env lookup when not
        # passed explicitly, so the key never lands in our own state or logs.
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def generate(self, prompt: str) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )
        except Exception as exc:
            # Only the exception type and its message are recorded. Request
            # bodies and headers (which carry the key) are never logged.
            return LLMResponse.failure(
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                retryable=classify_retryable(exc),
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = getattr(response, "usage", None)
        return LLMResponse.success(
            response.choices[0].message.content,
            latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
