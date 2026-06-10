class Memory:

    def __init__(self):

        self.short_term = []

        self.commitments = []

        self.agent_models = {}

    def add_message(self, message):

        self.short_term.append(message)

    def add_commitment(self, commitment):

        self.commitments.append(commitment)

    def update_model(
        self,
        agent_name,
        trust,
        stance
    ):

        self.agent_models[agent_name] = {
            "trust": trust,
            "stance": stance
        }