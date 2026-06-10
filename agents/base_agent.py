from memory.memory import Memory


class Agent:

    def __init__(
        self,
        name,
        role,
        goal,
        llm
    ):

        self.name = name

        self.role = role

        self.goal = goal

        self.memory = Memory()

        self.llm = llm

    def observe(
        self,
        message
    ):

        self.memory.add_message(
            message
        )

    def generate_response(
        self,
        conversation
    ):

        prompt = f"""
You are {self.name}.

Role:
{self.role}

Goal:
{self.goal}

Current policy scale:

0.0 = full conservation

0.5 = balanced

1.0 = aggressive logging

Conversation:

{conversation}

Provide one policy argument.
"""

        return self.llm.generate(
            prompt
        )