# Multi-Agent Policy Simulation

A multi-agent LLM simulation of a forest policy negotiation. Four agents with
conflicting goals (government, industry, NGO, and scientist) argue over rounds
about where to set policy on a single "conservation vs. logging" scale. Their
collective decision feeds back into a simulated environment, and every run
writes a complete audit trail to disk.

## How it works

- Each agent has a role, a goal, and a preferred position on the policy scale
  (`0.0` = full conservation, `0.5` = balanced, `1.0` = aggressive logging).
- Every round, each agent sees recent discussion, the environment state, and
  the group's current average/spread, then returns a JSON action: a target,
  message type, an updated `policy_position`, and a short message.
- The round's average position is applied to the environment: pro-logging
  policy raises `economic_output` and lowers `forest_health`, while
  conservation does the reverse.
- The prompt instructs agents to step toward the group average whenever the
  spread exceeds 0.15, unless conceding would betray their core goal.

## Metrics

Reported each round (`rounds.jsonl`) and in the final summary (`summary.json`):

- **average_policy** — mean of agents' latest positions.
- **agreement_score** — `1 - spread` between the most extreme positions
  (higher = more consensus).
- **spread** — the raw gap between the most extreme positions.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10 or newer. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## Testing

```bash
python -m pytest
```

The whole suite runs offline: no test requires `OPENAI_API_KEY` or network
access. The production client is exercised against a stubbed SDK, and agent
and simulation behaviour against `utils/fake_llm.FakeLLM`.

## Run

```bash
python main.py                     # defaults: 5 rounds, gpt-4o-mini
python main.py --rounds 3          # override any config field
python main.py --config cfg.json   # load a config; CLI flags still win
python main.py --fake-llm          # offline run: no API key, no network
python main.py --help              # list every flag
```

`--fake-llm` uses a scripted local client and needs no credentials, which
makes it useful for smoke-testing changes and for inspecting the artifact
format. Such runs are identifiable in `summary.json` via `llm_client` and
`llm_model`, so an offline run can never be mistaken for a real one.

## Run artifacts

Each run gets a unique ID and its own directory under `--output-dir`
(default `runs/`):

| File | Contents |
| --- | --- |
| `resolved_config.json` | Every parameter that was executed, after file and CLI overrides |
| `messages.jsonl` | One row per agent turn: round, speaker, target, type, validated position, content, UTC timestamp, plus parse status, fallback flag, attempt count, retries, latency, token counts, and the raw model response |
| `rounds.jsonl` | One row per round: agent order, positions, environment state before and after, applied policy, average policy, spread, agreement score, fallback and failed-attempt counts |
| `summary.json` | Final metrics, environment state, reliability counters, and interpreter/library versions |

Rows are flushed as they are produced, so an interrupted run still leaves a
readable partial trail. Timestamps are timezone-aware UTC. No credential or
environment-variable value is ever written to an artifact.

## Failure handling

Model responses are validated, not repaired. Each attempt is classified as
`ok`, `api_error`, `invalid_json`, or `schema_error`. An out-of-range
`policy_position`, an unknown target, or a missing field is a schema error
rather than something silently corrected. Malformed responses and transient
API errors are retried up to `max_retries` times (permanent errors such as
authentication failures are not retried).

If every attempt for a turn fails, the agent holds its position and the
transcript records a message of type `fallback` whose content states the
failure. Fallbacks are counted in `summary.json` and reported on the console,
so a degraded run is visible rather than indistinguishable from a clean one.

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | `main()` — CLI, wiring, console report |
| `agents/base_agent.py` | `Agent` — builds the prompt, validates responses, retries |
| `environment/forest_policy.py` | `ForestPolicyEnvironment` — forest health and economic output |
| `simulation/config.py` | `SimulationConfig`, `AgentSpec` — run parameters |
| `simulation/simulation.py` | `Simulation.run()` — the round loop, returns a `RunResult` |
| `simulation/metrics.py` | `final_positions`, `agreement_score`, `average_policy`, `position_spread` |
| `simulation/runlog.py` | `RunWriter` — run IDs and artifact writing |
| `models/message.py` | `Message`, `AgentAction` — validated schemas |
| `models/records.py` | Audit schemas for the artifact files |
| `models/context.py` | `RoundContext` — what an agent sees on its turn |
| `utils/llm.py` | `LLMClient` protocol and `OpenAIClient` |
| `utils/fake_llm.py` | `FakeLLM` — offline scripted client |
| `memory/memory.py` | Per-agent memory store (defined but not yet used) |
| `experiments/run_batch.py` | Placeholder; does not yet run the simulation |
| `tests/` | Test suite |

## Configuration

All parameters live in `SimulationConfig` and can be set by CLI flag or JSON
file. Unknown keys and out-of-range values are startup errors.

| Field | Default | Meaning |
| --- | --- | --- |
| `rounds` | `5` | Negotiation rounds |
| `model` | `gpt-4o-mini` | Model identifier |
| `temperature` | `0.7` | Sampling temperature |
| `sensitivity` | `0.20` | How strongly policy moves the environment each round |
| `protocol` | `simultaneous` | Whether agents in a round see each other's turns |
| `max_retries` | `2` | Retries per turn on API or validation failure |
| `output_dir` | `runs` | Where run directories are created |
| `agents` | 4 stakeholders | Name, role, goal, preferred position |

### Interaction protocol

- `simultaneous` — context (recent messages, group average, spread) is frozen
  at the start of the round, so every agent sees the same state and no agent
  sees another's turn from the same round.
- `sequential` — context is rebuilt before each agent acts, so later agents in
  a round see what earlier agents just said.

## Example real-model run

A five-round illustrative run using default configuration completed 20 agent turns with no retries or fallbacks.

| Metric | Result |
| --- | ---: |
| Final average policy | 0.44 |
| Final agreement score | 0.45 |
| Forest health | 0.76 |
| Economic output | 0.64 |
| LLM calls | 20 |
| Retries | 0 |
| Fallbacks | 0 |

The agents moved somewhat closer but did not reach consensus. This is a single
nondeterministic demonstration run, not an empirical finding or a validated
prediction of stakeholder behaviour.

## Limitations

These are real and not yet addressed. They matter for interpreting any result
this code produces.

- **Convergence is partly instructed.** The prompt tells each agent to step
  toward the group average and shows it that average every round, so observed
  convergence cannot currently be separated from prompt compliance. There is
  no non-LLM baseline to compare against.
- **The environment is illustrative, not calibrated.** The update rule is
  linear, symmetric, and fully reversible, with no regrowth dynamics and no
  tipping point — the "0.40 = ecological collapse" figure shown to agents has
  no mechanical effect. `sensitivity` is not derived from data.
- **Deadline pressure is not time-dependent.** The concession instruction
  triggers on spread alone; the round number is shown to agents but does not
  change the pressure applied.
- **`target` is recorded but not useful.** Messages are broadcast to all agents.
  The target field does not route anything.
- **Agents have no memory** beyond the recent-message window. `Memory` exists
  but is not wired in, so commitments are not tracked across rounds.
- **Runs are auditable but not bit-reproducible.** Prompts, responses, config
  and versions are all recorded, but no seed is passed to the model and
  sampling is non-deterministic. Replication is statistical, not exact.
- **No batch experiment support.** `experiments/run_batch.py` is a
  placeholder. There is no parameter sweep, replicate handling, or aggregation.
