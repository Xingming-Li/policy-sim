"""Audit schemas: what gets written to the run directory.

These types are the contract for `messages.jsonl`, `rounds.jsonl` and
`summary.json`. They deliberately keep provenance (status, retries, latency,
raw response) next to the resulting message, so no reader can mistake a
fallback for a real model action.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from models.message import Message, utcnow


class ResponseStatus(str, Enum):
    """Outcome of a single LLM attempt, or of an agent turn overall."""

    OK = "ok"
    API_ERROR = "api_error"
    INVALID_JSON = "invalid_json"
    SCHEMA_ERROR = "schema_error"


FAILURE_STATUSES: frozenset[ResponseStatus] = frozenset(
    {
        ResponseStatus.API_ERROR,
        ResponseStatus.INVALID_JSON,
        ResponseStatus.SCHEMA_ERROR,
    }
)


class AttemptRecord(BaseModel):
    """One LLM call: what came back and why it was or wasn't usable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt: int = Field(ge=1)
    status: ResponseStatus
    latency_ms: float = Field(ge=0.0)
    retryable: bool = False
    raw_response: str | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class MessageRecord(BaseModel):
    """An agent turn: the message plus its full provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: Message
    status: ResponseStatus
    fallback_used: bool
    attempts: int = Field(ge=1)
    retries: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    raw_response: str | None = None
    error: str | None = None
    attempt_log: tuple[AttemptRecord, ...] = ()
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def to_row(self) -> dict[str, Any]:
        """Flat JSONL row: message fields first, then provenance."""
        message = self.message.model_dump(mode="json")
        provenance = self.model_dump(mode="json", exclude={"message"})
        return {**message, **provenance}


class RoundRecord(BaseModel):
    """Aggregate state for one negotiation round."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_id: int = Field(ge=1)
    agent_order: tuple[str, ...]
    positions: dict[str, float]

    # Group statistics that were shown to agents this round (the prompt input).
    context_group_average: float
    context_spread: float

    environment_before: dict[str, float]
    environment_after: dict[str, float]

    # Mean of this round's positions; this is the value applied to the environment.
    applied_policy: float

    # Metrics over the transcript at the end of this round.
    average_policy: float
    spread: float
    agreement_score: float

    fallback_count: int = Field(ge=0)
    failed_attempt_count: int = Field(ge=0)
    timestamp: AwareDatetime = Field(default_factory=utcnow)

    def to_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class StatusCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: int = 0
    api_error: int = 0
    invalid_json: int = 0
    schema_error: int = 0

    @classmethod
    def from_statuses(cls, statuses: list[ResponseStatus]) -> StatusCounts:
        return cls(
            ok=sum(s is ResponseStatus.OK for s in statuses),
            api_error=sum(s is ResponseStatus.API_ERROR for s in statuses),
            invalid_json=sum(s is ResponseStatus.INVALID_JSON for s in statuses),
            schema_error=sum(s is ResponseStatus.SCHEMA_ERROR for s in statuses),
        )


class RunSummary(BaseModel):
    """Final metrics and reliability counters for a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    started_at: AwareDatetime
    finished_at: AwareDatetime
    duration_seconds: float = Field(ge=0.0)

    protocol: str

    # `model` is the configured value; `llm_client` / `llm_model` describe the
    # client that actually served the run. They differ for offline runs, so a
    # fake run can never be mistaken for a real one.
    model: str
    llm_client: str
    llm_model: str

    rounds_requested: int
    rounds_completed: int

    final_positions: dict[str, float]
    average_policy: float
    agreement_score: float
    spread: float
    final_environment: dict[str, float]

    total_agent_turns: int = Field(ge=0)
    total_llm_calls: int = Field(ge=0)
    total_retries: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    fallback_rate: float = Field(ge=0.0, le=1.0)

    # Counts over final turn outcomes and over every individual attempt.
    turn_status_counts: StatusCounts
    attempt_status_counts: StatusCounts

    total_latency_ms: float = Field(ge=0.0)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    # Interpreter and library versions. Never environment variables.
    runtime: dict[str, str] = Field(default_factory=dict)
