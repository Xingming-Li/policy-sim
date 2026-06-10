from agents.base_agent import Agent
from utils.llm import LLMClient
from simulation.run import run_simulation

llm = LLMClient()

agents = [
    Agent("Gov", "Government policymaker", "Balance economy and environment", llm),
    Agent("Industry", "Forestry company", "Maximize logging profit", llm),
    Agent("NGO", "Environmental NGO", "Protect forests", llm),
    Agent("Scientist", "Ecologist", "Provide neutral evidence-based input", llm),
]

result = run_simulation(agents)

for r in result:
    print(r)