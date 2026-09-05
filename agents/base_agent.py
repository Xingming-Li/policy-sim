"""The negotiating agent.

`decide()` performs one turn: render the prompt, call the model, and validate
the response, retrying a bounded number of times on transient API failures and
malformed output. It always returns a `MessageRecord`, so a turn's outcome is
never ambiguous: `status` says what happened and `fallback_used` says whether
the resulting message came from the model or from us.

The prompt text is unchanged from the prototype; only the surrounding
plumbing is new.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from memory.memory import Memory
from models.context import RoundContext
from models.message import (
    FALLBACK_MESSAGE_TYPE,
    AgentAction,
    Message,
    utcnow,
)
from models.records import AttemptRecord, MessageRecord, ResponseStatus
from simulation.config import AgentSpec
from utils.llm import LLMClient

logger = logging.getLogger(__name__)


class SchemaViolation(ValueError):
    """Response was valid JSON but not a usable action."""


class Agent:
    """One stakeholder in the negotiation."""

    def __init__(
        self,
        spec: AgentSpec,
        llm: LLMClient,
        max_retries: int = 2,
    ) -> None:
        self.spec = spec
        self.name = spec.name
        self.role = spec.role
        self.goal = spec.goal
        self.preferred_position = spec.preferred_position
        self.current_position = spec.preferred_position
        self.max_retries = max_retries
        self.memory = Memory()
        self.llm = llm

    def observe(self, message: Message) -> None:
        self.memory.add_message(message)

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def build_prompt(self, context: RoundContext) -> str:
        """Render the negotiation prompt. Text preserved verbatim."""
        environment = context.environment

        recent_context = "\n".join(
            [
                f"{m.sender}: "
                f"{m.content} "
                f"(position={m.policy_position})"
                for m in context.latest_messages[-5:]
            ]
        ) or "(no discussion yet)"

        others = ", ".join(
            t for t in context.valid_targets
            if t != self.name
        )

        return f"""
You are {self.name}.

Role:
{self.role}

Goal:
{self.goal}

Policy scale:

0.0 = full conservation

0.5 = balanced

1.0 = aggressive logging

Your preferred position: {self.preferred_position}
Your current position: {self.current_position:.2f}

Current environment:

forest_health = {environment.forest_health:.2f} (below 0.40 = ecological collapse)

economic_output = {environment.economic_output:.2f} (below 0.40 = economic recession)

Recent discussion:

{recent_context}

Negotiation status:

This is round {context.round_id} of {context.total_rounds}. A single shared policy must be
agreed before the final round ends. The group's current average position is
{context.group_average:.2f}, and the gap between the most extreme participants is
{context.spread:.2f}.

You are negotiating with the others toward that shared policy. You want an
outcome that serves your goal, but reaching agreement is valuable and
deadlock is the worst result.

If the gap above is larger than 0.15, the group is still in disagreement and
time is running out. Move your position a meaningful step (typically
0.05-0.15) toward the group average this round, UNLESS doing so would
directly betray your core goal -- in which case hold, but say why.

Verbal agreement is not enough: your "policy_position" number must reflect
your actual concession. Do NOT simply repeat your current position unless you
have a strong, stated reason to hold.

Respond to the most recent proposal.

Return ONLY valid JSON. "target" must be exactly one of: {others}

{{
  "target": "{others.split(", ")[0] if others else "null"}",
  "message_type": "proposal|support|oppose|evidence",
  "policy_position": {self.current_position:.2f},
  "content": "under 25 words"
}}
"""

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def decide(self, context: RoundContext) -> MessageRecord:
        """Run one turn, with bounded retries. Never raises on model failure."""
        prompt = self.build_prompt(context)
        attempts: list[AttemptRecord] = []

        for attempt_number in range(1, self.max_retries + 2):
            response = self.llm.generate(prompt)

            if not response.ok:
                attempts.append(
                    AttemptRecord(
                        attempt=attempt_number,
                        status=ResponseStatus.API_ERROR,
                        latency_ms=response.latency_ms,
                        retryable=response.retryable,
                        error=f"{response.error_type}: {response.error_message}",
                    )
                )
                logger.warning(
                    "%s round %s attempt %s: api_error (%s, retryable=%s)",
                    self.name,
                    context.round_id,
                    attempt_number,
                    response.error_type,
                    response.retryable,
                )
                if not response.retryable:
                    break
                continue

            status, action, error = self._parse(response.text, context.valid_targets)

            attempts.append(
                AttemptRecord(
                    attempt=attempt_number,
                    status=status,
                    latency_ms=response.latency_ms,
                    retryable=status is not ResponseStatus.OK,
                    raw_response=response.text,
                    error=error,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )
            )

            if status is ResponseStatus.OK and action is not None:
                self.current_position = action.policy_position
                return self._record(
                    context=context,
                    attempts=attempts,
                    status=status,
                    action=action,
                )

            logger.warning(
                "%s round %s attempt %s: %s (%s)",
                self.name,
                context.round_id,
                attempt_number,
                status.value,
                error,
            )

        return self._fallback_record(context, attempts)

    def _parse(
        self,
        text: str | None,
        valid_targets: "list[str] | tuple[str, ...]",
    ) -> tuple[ResponseStatus, AgentAction | None, str | None]:
        """Classify a raw response as OK, invalid JSON, or a schema error."""
        if text is None or not text.strip():
            return ResponseStatus.INVALID_JSON, None, "empty response body"

        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            return ResponseStatus.INVALID_JSON, None, f"json decode failed: {exc}"

        try:
            normalized = self._normalize(payload, valid_targets)
            return ResponseStatus.OK, AgentAction.model_validate(normalized), None
        except SchemaViolation as exc:
            return ResponseStatus.SCHEMA_ERROR, None, str(exc)
        except ValidationError as exc:
            return ResponseStatus.SCHEMA_ERROR, None, _compact_validation_error(exc)

    def _normalize(
        self,
        payload: object,
        valid_targets: "list[str] | tuple[str, ...]",
    ) -> dict[str, object]:
        """Whitespace/case normalisation only.

        Deliberately does not repair semantics: an out-of-range position or an
        unknown target is reported as a schema error rather than clamped, so
        the artifacts show what the model actually produced.
        """
        if not isinstance(payload, dict):
            raise SchemaViolation(
                f"expected a JSON object, got {type(payload).__name__}"
            )

        normalized = dict(payload)

        message_type = normalized.get("message_type")
        if isinstance(message_type, str):
            normalized["message_type"] = message_type.strip().lower()

        target = normalized.get("target")
        if isinstance(target, str):
            candidate = target.strip()
            if candidate.lower() in {"", "null", "none"}:
                normalized["target"] = None
            else:
                match = next(
                    (t for t in valid_targets if t.lower() == candidate.lower()),
                    None,
                )
                if match is None:
                    raise SchemaViolation(
                        f"target {candidate!r} is not one of {list(valid_targets)}"
                    )
                if match == self.name:
                    raise SchemaViolation("target must not be the sender")
                normalized["target"] = match

        return normalized

    def _record(
        self,
        *,
        context: RoundContext,
        attempts: list[AttemptRecord],
        status: ResponseStatus,
        action: AgentAction,
    ) -> MessageRecord:
        message = Message(
            round_id=context.round_id,
            sender=self.name,
            target=action.target,
            message_type=action.message_type,
            policy_position=action.policy_position,
            content=action.content,
            timestamp=utcnow(),
        )
        return self._wrap(message, attempts, status, fallback_used=False)

    def _fallback_record(
        self,
        context: RoundContext,
        attempts: list[AttemptRecord],
    ) -> MessageRecord:
        """Build the explicit failure result after retries are exhausted.

        The position is held, but the message is typed `fallback` and its
        content says so, so downstream analysis cannot mistake it for a
        negotiated action.
        """
        status = attempts[-1].status if attempts else ResponseStatus.API_ERROR
        logger.error(
            "%s round %s: no valid response after %s attempt(s); "
            "recording fallback (status=%s)",
            self.name,
            context.round_id,
            len(attempts),
            status.value,
        )
        message = Message(
            round_id=context.round_id,
            sender=self.name,
            target=None,
            message_type=FALLBACK_MESSAGE_TYPE,
            policy_position=self.current_position,
            content=(
                f"[NO VALID MODEL RESPONSE: {status.value} after "
                f"{len(attempts)} attempt(s)] Position held at "
                f"{self.current_position:.2f}."
            ),
            timestamp=utcnow(),
        )
        return self._wrap(message, attempts, status, fallback_used=True)

    @staticmethod
    def _wrap(
        message: Message,
        attempts: list[AttemptRecord],
        status: ResponseStatus,
        *,
        fallback_used: bool,
    ) -> MessageRecord:
        prompt_tokens = [
            a.prompt_tokens for a in attempts if a.prompt_tokens is not None
        ]
        completion_tokens = [
            a.completion_tokens for a in attempts if a.completion_tokens is not None
        ]
        return MessageRecord(
            message=message,
            status=status,
            fallback_used=fallback_used,
            attempts=max(len(attempts), 1),
            retries=max(len(attempts) - 1, 0),
            latency_ms=sum(a.latency_ms for a in attempts),
            raw_response=attempts[-1].raw_response if attempts else None,
            error=attempts[-1].error if attempts else None,
            attempt_log=tuple(attempts),
            prompt_tokens=sum(prompt_tokens) if prompt_tokens else None,
            completion_tokens=sum(completion_tokens) if completion_tokens else None,
        )


def _compact_validation_error(exc: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    ]
    return "; ".join(parts)[:500]
