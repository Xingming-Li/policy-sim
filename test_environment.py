from environment.forest_policy import ForestPolicyEnvironment


def test_policy():

    env = ForestPolicyEnvironment()

    old_value = env.economic_output

    env.apply_policy(0.9)

    assert env.economic_output > old_value
    