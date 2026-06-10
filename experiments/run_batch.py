import random


results = []

for run_id in range(20):

    outcome = {
        "run": run_id,
        "policy": round(
            random.uniform(
                0,
                1
            ),
            2
        )
    }

    results.append(
        outcome
    )

print(results)