from datetime import datetime

from models.message import Message

from simulation.metrics import (
    agreement_score,
    average_policy
)


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
        rounds=5
    ):

        valid_targets = [
            agent.name
            for agent in self.agents
        ]

        for round_id in range(rounds):

            print(
                f"\n===== ROUND {round_id + 1} ====="
            )

            latest_messages = (
                self.transcript[-10:]
            )

            current_positions = [
                agent.current_position
                for agent in self.agents
            ]

            group_average = (
                sum(current_positions)
                /
                len(current_positions)
            )

            spread = (
                max(current_positions)
                -
                min(current_positions)
            )

            round_positions = []

            for agent in self.agents:

                action = (
                    agent.generate_response(
                        latest_messages,
                        self.environment,
                        valid_targets,
                        round_id + 1,
                        rounds,
                        group_average,
                        spread
                    )
                )

                position = float(
                    action[
                        "policy_position"
                    ]
                )

                agent.current_position = position

                round_positions.append(
                    position
                )

                message = Message(
                    sender=agent.name,
                    target=action.get(
                        "target"
                    ),
                    message_type=action[
                        "message_type"
                    ],
                    policy_position=position,
                    content=action[
                        "content"
                    ],
                    timestamp=datetime.now()
                )

                self.transcript.append(
                    message
                )

                print(
                    f"\n{message.sender}"
                )

                print(
                    f"target={message.target}"
                )

                print(
                    f"type={message.message_type}"
                )

                print(
                    f"position={message.policy_position}"
                )

                print(
                    message.content
                )

            collective_policy = (
                sum(round_positions)
                /
                len(round_positions)
            )

            self.environment.apply_policy(
                collective_policy
            )

            print(
                "\nEnvironment"
            )

            print(
                "forest_health:",
                round(
                    self.environment.forest_health,
                    2
                )
            )

            print(
                "economic_output:",
                round(
                    self.environment.economic_output,
                    2
                )
            )

            print(
                "agreement_score:",
                agreement_score(self.transcript)
            )

            print(
                "average_policy:",
                average_policy(self.transcript)
            )

        return self.transcript