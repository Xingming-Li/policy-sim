"""The round loop.

`Simulation.run()` executes the configured number of rounds and returns a
`RunResult` holding the transcript, the per-turn provenance records, the
per-round records, and the summary. Artifacts are written as the run proceeds.

Two interaction protocols are supported, both explicit in the config so the
choice is recorded rather than implied:

  simultaneous  Context (recent messages, group average, spread) is frozen at
                the start of the round; every agent in the round sees the same
                state. This reproduces the prototype's behaviour exactly.
  sequential    Context is rebuilt before each agent acts, so later agents in
                a round see what earlier agents just said.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

from agents.base_agent import Agent
from environment.forest_policy import ForestPolicyEnvironment
from models.context import RoundContext
from models.message import Message, utcnow
from models.records import (
    FAILURE_STATUSES,
    MessageRecord,
    ResponseStatus,
    RoundRecord,
    RunSummary,
    StatusCounts,
)
from simulation.config import SimulationConfig
from simulation.metrics import (
    agreement_score,
    average_policy,
    final_positions,
    position_spread,
)
from simulation.runlog import RunWriter, new_run_id, runtime_info

logger = logging.getLogger(__name__)

CONTEXT_WINDOW = 10


@dataclass
class RunResult:
    """Everything one run produced, in memory and on disk."""

    run_id: str
    output_dir: "object"
    config: SimulationConfig
    transcript: list[Message] = field(default_factory=list)
    records: list[MessageRecord] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    summary: RunSummary | None = None


class Simulation:

    def __init__(
        self,
        agents: Sequence[Agent],
        environment: ForestPolicyEnvironment,
        config: SimulationConfig,
        run_id: str | None = None,
    ) -> None:
        self.agents = list(agents)
        self.environment = environment
        self.config = config
        self.run_id = run_id or new_run_id()
        self.transcript: list[Message] = []
        self.records: list[MessageRecord] = []
        self.rounds: list[RoundRecord] = []

    # ------------------------------------------------------------------

    def _group_statistics(self) -> tuple[float, float]:
        positions = [agent.current_position for agent in self.agents]
        average = sum(positions) / len(positions)
        return average, max(positions) - min(positions)

    def _build_context(self, round_id: int, valid_targets: list[str]) -> RoundContext:
        group_average, spread = self._group_statistics()
        return RoundContext(
            round_id=round_id,
            total_rounds=self.config.rounds,
            latest_messages=tuple(self.transcript[-CONTEXT_WINDOW:]),
            environment=self.environment,
            valid_targets=tuple(valid_targets),
            group_average=group_average,
            spread=spread,
        )

    # ------------------------------------------------------------------

    def run(self) -> RunResult:
        started_at = utcnow()
        started_perf = time.perf_counter()
        valid_targets = [agent.name for agent in self.agents]
        agent_order = tuple(valid_targets)

        result = RunResult(
            run_id=self.run_id,
            output_dir=None,
            config=self.config,
        )

        with RunWriter(self.config.output_dir, self.run_id) as writer:
            result.output_dir = writer.dir
            writer.write_config(self.config)
            logger.info(
                "run %s: %s round(s), model=%s, protocol=%s, artifacts=%s",
                self.run_id,
                self.config.rounds,
                self.config.model,
                self.config.protocol,
                writer.dir,
            )

            for round_index in range(self.config.rounds):
                round_id = round_index + 1
                self._run_round(round_id, valid_targets, agent_order, writer)

            summary = self._build_summary(
                started_at=started_at,
                duration_seconds=time.perf_counter() - started_perf,
            )
            writer.write_summary(summary)

        result.transcript = list(self.transcript)
        result.records = list(self.records)
        result.rounds = list(self.rounds)
        result.summary = summary
        return result

    def _run_round(
        self,
        round_id: int,
        valid_targets: list[str],
        agent_order: tuple[str, ...],
        writer: RunWriter,
    ) -> None:
        logger.info("round %s/%s", round_id, self.config.rounds)

        environment_before = self.environment.state()
        frozen_context = self._build_context(round_id, valid_targets)

        round_positions: list[float] = []
        positions: dict[str, float] = {}
        round_records: list[MessageRecord] = []

        for agent in self.agents:
            # Under "simultaneous" every agent sees the frozen start-of-round
            # context; under "sequential" it is rebuilt to include this round's
            # earlier turns.
            context = (
                frozen_context
                if self.config.protocol == "simultaneous"
                else self._build_context(round_id, valid_targets)
            )

            record = agent.decide(context)

            message = record.message
            agent.current_position = message.policy_position
            positions[agent.name] = message.policy_position
            round_positions.append(message.policy_position)

            self.transcript.append(message)
            self.records.append(record)
            round_records.append(record)
            writer.append_message(record)

            logger.info(
                "  %-10s position=%.2f type=%-9s target=%-10s status=%s%s",
                message.sender,
                message.policy_position,
                message.message_type,
                message.target,
                record.status.value,
                " FALLBACK" if record.fallback_used else "",
            )
            logger.info("  %s: %s", message.sender, message.content)

        applied_policy = sum(round_positions) / len(round_positions)
        self.environment.apply_policy(applied_policy)

        round_record = RoundRecord(
            round_id=round_id,
            agent_order=agent_order,
            positions=positions,
            context_group_average=frozen_context.group_average,
            context_spread=frozen_context.spread,
            environment_before=environment_before,
            environment_after=self.environment.state(),
            applied_policy=applied_policy,
            average_policy=average_policy(self.transcript),
            spread=position_spread(self.transcript),
            agreement_score=agreement_score(self.transcript),
            fallback_count=sum(r.fallback_used for r in round_records),
            failed_attempt_count=sum(
                1
                for r in round_records
                for a in r.attempt_log
                if a.status in FAILURE_STATUSES
            ),
        )
        self.rounds.append(round_record)
        writer.append_round(round_record)

        logger.info(
            "  environment: forest_health=%.2f economic_output=%.2f "
            "| applied_policy=%.2f agreement=%.2f",
            self.environment.forest_health,
            self.environment.economic_output,
            applied_policy,
            round_record.agreement_score,
        )

    # ------------------------------------------------------------------

    def _client_description(self, attribute: str) -> str:
        """Identify the client that actually served the run.

        Reported alongside the configured model so an offline or mixed run is
        self-evident in the artifacts.
        """
        if attribute == "class":
            values = {type(agent.llm).__name__ for agent in self.agents}
        else:
            values = {
                str(getattr(agent.llm, "model", "unknown")) for agent in self.agents
            }
        return ",".join(sorted(values))

    def _build_summary(
        self,
        *,
        started_at,
        duration_seconds: float,
    ) -> RunSummary:
        turn_statuses = [r.status for r in self.records]
        attempt_statuses = [a.status for r in self.records for a in r.attempt_log]

        prompt_tokens = [
            r.prompt_tokens for r in self.records if r.prompt_tokens is not None
        ]
        completion_tokens = [
            r.completion_tokens for r in self.records if r.completion_tokens is not None
        ]

        fallback_count = sum(r.fallback_used for r in self.records)
        turns = len(self.records)

        return RunSummary(
            run_id=self.run_id,
            started_at=started_at,
            finished_at=utcnow(),
            duration_seconds=duration_seconds,
            protocol=self.config.protocol,
            model=self.config.model,
            llm_client=self._client_description("class"),
            llm_model=self._client_description("model"),
            rounds_requested=self.config.rounds,
            rounds_completed=len(self.rounds),
            final_positions=final_positions(self.transcript),
            average_policy=average_policy(self.transcript),
            agreement_score=agreement_score(self.transcript),
            spread=position_spread(self.transcript),
            final_environment=self.environment.state(),
            total_agent_turns=turns,
            total_llm_calls=sum(len(r.attempt_log) for r in self.records),
            total_retries=sum(r.retries for r in self.records),
            fallback_count=fallback_count,
            fallback_rate=(fallback_count / turns) if turns else 0.0,
            turn_status_counts=StatusCounts.from_statuses(turn_statuses),
            attempt_status_counts=StatusCounts.from_statuses(attempt_statuses),
            total_latency_ms=sum(r.latency_ms for r in self.records),
            prompt_tokens=sum(prompt_tokens) if prompt_tokens else None,
            completion_tokens=sum(completion_tokens) if completion_tokens else None,
            runtime=runtime_info(),
        )


__all__ = ["Simulation", "RunResult", "ResponseStatus"]
