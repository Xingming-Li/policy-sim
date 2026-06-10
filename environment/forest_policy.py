class ForestPolicyEnvironment:

    def __init__(self):

        self.forest_health = 0.7

        self.economic_output = 0.7

        self.public_support = 0.7

    def apply_policy(
        self,
        policy_position
    ):

        self.forest_health += (
            0.5 - policy_position
        ) * 0.1

        self.economic_output += (
            policy_position - 0.5
        ) * 0.1

        self.forest_health = max(
            0,
            min(1, self.forest_health)
        )

        self.economic_output = max(
            0,
            min(1, self.economic_output)
        )