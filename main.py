"""Entry point for a single simulation run

    python main.py                     # defaults, real model, needs OPENAI_API_KEY
    python main.py --rounds 3          # override any config field
    python main.py --config cfg.json   # load a config, CLI flags still win
    python main.py --fake-llm          # offline smoke run, no key, no network
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agents.base_agent import Agent
from environment.forest_policy import ForestPolicyEnvironment
from simulation.config import SimulationConfig
from simulation.simulation import RunResult, Simulation
from utils.fake_llm import make_demo_llm
from utils.llm import LLMClient, OpenAIClient

logger = logging.getLogger("policy_sim")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run one LLM multi-agent forest policy negotiation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config file; CLI flags override its values.",
    )
    parser.add_argument("--rounds", type=int, help="Number of negotiation rounds.")
    parser.add_argument("--model", help="Model identifier.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument(
        "--sensitivity",
        type=float,
        help="How strongly the collective policy moves the environment.",
    )
    parser.add_argument(
        "--protocol",
        choices=("simultaneous", "sequential"),
        help="Whether agents in a round see each other's turns.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        dest="max_retries",
        help="Retries per agent turn on API errors or malformed responses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        dest="output_dir",
        help="Directory that run subdirectories are created in.",
    )
    parser.add_argument(
        "--fake-llm",
        action="store_true",
        help="Use the offline scripted client (no API key, no network).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log level.",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> SimulationConfig:
    """Defaults, then optional config file, then CLI overrides."""
    base = (
        SimulationConfig.from_file(args.config)
        if args.config
        else SimulationConfig()
    )
    return base.with_overrides(
        rounds=args.rounds,
        model=args.model,
        temperature=args.temperature,
        sensitivity=args.sensitivity,
        protocol=args.protocol,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
    )


def build_llm(config: SimulationConfig, *, fake: bool) -> LLMClient:
    if fake:
        logger.info("using offline fake LLM (no API calls will be made)")
        return make_demo_llm()
    return OpenAIClient(model=config.model, temperature=config.temperature)


def build_agents(config: SimulationConfig, llm: LLMClient) -> list[Agent]:
    return [
        Agent(spec, llm, max_retries=config.max_retries) for spec in config.agents
    ]


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def report(result: RunResult) -> None:
    summary = result.summary
    assert summary is not None

    print("\n===== RESULTS =====")
    print(f"run_id: {summary.run_id}")
    print(f"artifacts: {result.output_dir}")

    print("\nFinal positions:")
    for agent, position in summary.final_positions.items():
        print(f"{agent}: {position:.2f}")

    print(f"\nAverage policy: {summary.average_policy:.2f}")
    print(f"Agreement score: {summary.agreement_score:.2f}")
    print(
        "Environment: "
        f"forest_health={summary.final_environment['forest_health']:.2f} "
        f"economic_output={summary.final_environment['economic_output']:.2f}"
    )

    print(
        f"\nLLM calls: {summary.total_llm_calls} "
        f"across {summary.total_agent_turns} turn(s)\n"
        f"Retries: {summary.total_retries}"
    )
    if summary.fallback_count:
        print(
            f"WARNING: {summary.fallback_count} of {summary.total_agent_turns} "
            f"turn(s) fell back after exhausting retries "
            f"({summary.fallback_rate:.0%}). Treat these results with caution."
        )
    else:
        print("Fallbacks: 0")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        config = resolve_config(args)
    except Exception as exc:
        logger.error("invalid configuration: %s", exc)
        return 2

    try:
        llm = build_llm(config, fake=args.fake_llm)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2

    environment = ForestPolicyEnvironment(sensitivity=config.sensitivity)
    simulation = Simulation(build_agents(config, llm), environment, config)

    result = simulation.run()
    report(result)

    # A run whose turns all fell back produced no model behaviour at all.
    if result.summary is not None and result.summary.fallback_rate == 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
