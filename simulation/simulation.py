from datetime import datetime

from models.message import Message


class Simulation:

    def __init__(
        self,
        agents,
        environment
    ):

        self.agents = agents

        self.environment = environment

        self.transcript = []

    def run(
        self,
        rounds=3
    ):

        conversation = ""

        for _ in range(rounds):

            for agent in self.agents:

                response = (
                    agent.generate_response(
                        conversation
                    )
                )

                msg = Message(
                    sender=agent.name,
                    message_type="proposal",
                    policy_position=0.5,
                    content=response,
                    timestamp=datetime.now()
                )

                self.transcript.append(
                    msg
                )

                conversation += (
                    f"\n{agent.name}: "
                    f"{response}"
                )

        return self.transcript