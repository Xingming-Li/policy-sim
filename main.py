from agents.base_agent import Agent

from utils.llm import LLMClient

from environment.forest_policy import ForestPolicyEnvironment

from simulation.simulation import Simulation

from simulation.metrics import average_policy, final_positions, agreement_score


llm = LLMClient()

agents = [

    Agent(
        "Government",
        "Policymaker",
        "Balance economy and conservation",
        0.5,
        llm
    ),

    Agent(
        "Industry",
        "Forestry company",
        "Increase logging profits",
        0.8,
        llm
    ),

    Agent(
        "NGO",
        "Environmental organization",
        "Protect forests",
        0.1,
        llm
    ),

    Agent(
        "Scientist",
        "Ecologist",
        "Provide evidence-based advice",
        0.4,
        llm
    )
]

environment = (
    ForestPolicyEnvironment()
)

simulation = (
    Simulation(
        agents,
        environment
    )
)

transcript = simulation.run(rounds=5)

print("\n===== RESULTS =====")

positions = final_positions(transcript)

print("\nFinal positions:")

for agent, position in positions.items():

    print(
        f"{agent}: {position:.2f}"
    )

print(
    f"\nAverage policy: {average_policy(transcript):.2f}"
)

print(
    f"Agreement score: {agreement_score(transcript):.2f}"
)