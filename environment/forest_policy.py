class ForestPolicyEnvironment:

    def __init__(self):

        self.forest_health = 0.70

        self.economic_output = 0.70

    SENSITIVITY = 0.20

    def apply_policy(self, policy_position):

        self.forest_health += (0.5 - policy_position) * self.SENSITIVITY

        self.economic_output += (policy_position - 0.5) * self.SENSITIVITY

        self.forest_health = max(0, min(1, self.forest_health))

        self.economic_output = max(0, min(1, self.economic_output))
        