"""Domain schemas for agent actions and negotiation messages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

# What an agent is allowed to propose.
ProposedMessageType = Literal[
    "proposal",
    "support",
    "oppose",
    "evidence",
    "question",
    "commitment",
]

# What can appear in a transcript. "fallback" is reserved for turns where no
# valid model response was obtained; it can never come from the model itself,
# so a fallback is always distinguishable from a real action.
MessageType = Literal[
    "proposal",
    "support",
    "oppose",
    "evidence",
    "question",
    "commitment",
    "fallback",
]

PROPOSED_MESSAGE_TYPES: frozenset[str] = frozenset(get_args(ProposedMessageType))
MESSAGE_TYPES: frozenset[str] = frozenset(get_args(MessageType))

FALLBACK_MESSAGE_TYPE = "fallback"


def utcnow() -> datetime:
    """Timezone-aware UTC now. Used everywhere instead of `datetime.now()`."""
    return datetime.now(timezone.utc)


class AgentAction(BaseModel):
    """A validated action parsed from a model response.

    Anything that fails these constraints is a schema error, not something to
    be silently repaired.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    target: str | None = None
    message_type: ProposedMessageType
    policy_position: float = Field(ge=0.0, le=1.0)
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("content must not be blank")
        return stripped


class Message(BaseModel):
    """One utterance in the negotiation transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_id: int = Field(ge=1)
    sender: str = Field(min_length=1)
    target: str | None = None
    message_type: MessageType
    policy_position: float = Field(ge=0.0, le=1.0)
    content: str = Field(min_length=1)
    timestamp: AwareDatetime = Field(default_factory=utcnow)

    @property
    def is_fallback(self) -> bool:
        return self.message_type == FALLBACK_MESSAGE_TYPE
