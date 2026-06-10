# Multi-Agent Policy Simulation

A multi-agent LLM simulation of a forest policy negotiation. Four agents with
conflicting goals (government, industry, NGO, and scientist) argue over rounds
about where to set policy on a single "conservation vs. logging" scale. Their
collective decision feeds back into a simulated environment, and the run reports
how they converge.

## How it works

- Each agent has a role, a goal, and a preferred position on the policy scale
  (`0.0` = full conservation, `0.5` = balanced, `1.0` = aggressive logging).
- Every round, each agent sees the recent discussion, the environment state, and
  the group's current average/spread, then returns a JSON action: a target,
  message type, an updated `policy_position`, and a short message.
- The round's average position is applied to the environment: pro-logging policy
  raises `economic_output` and lowers `forest_health`, while conservation does the
  reverse.
- Agents are pushed to move toward consensus as the deadline nears, but may hold
  firm when conceding would betray their core goal. So principled deadlock is a
  possible outcome.

## Metrics

Reported each round and in the final summary:

- **average_policy** — mean of agents' latest positions.
- **agreement_score** — `1 - spread` between the most extreme positions (higher =
  more consensus).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Run

```bash
python main.py
```

Each round prints every agent's action, the resulting environment state, and the
convergence metrics, followed by a final summary of positions.

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Connects the agents and environment, runs the simulation |
| `agents/base_agent.py` | `Agent` — builds the prompt, parses and validates the response |
| `environment/forest_policy.py` | `ForestPolicyEnvironment` — forest health and economic output |
| `simulation/simulation.py` | `Simulation.run()` — the round loop |
| `simulation/metrics.py` | `final_positions`, `agreement_score`, `average_policy` |
| `memory/memory.py` | Per-agent memory store (incomplete) |
| `models/message.py` | Pydantic `Message` schema |
| `utils/llm.py` | `LLMClient` — OpenAI chat wrapper (JSON mode) |
| `experiments/run_batch.py` | Runs in batches (incomplete) |

## Configuration

- **Rounds:** `Simulation.run(rounds=5)` in `main.py`.
- **Agents:** edit the list in `main.py` (name, role, goal, preferred position).
- **Environment sensitivity:** `ForestPolicyEnvironment.SENSITIVITY` controls how
  strongly policy moves the world each round.
- **Model:** set in `utils/llm.py` (currently `gpt-4o-mini`).
