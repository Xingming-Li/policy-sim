def final_positions(
    transcript
):

    latest = {}

    for msg in transcript:

        latest[msg.sender] = (
            msg.policy_position
        )

    return latest


def agreement_score(
    transcript
):

    positions = list(
        final_positions(
            transcript
        ).values()
    )

    spread = (
        max(positions)
        -
        min(positions)
    )

    return round(
        1 - spread,
        2
    )


def average_policy(
    transcript
):

    positions = list(
        final_positions(
            transcript
        ).values()
    )

    return round(
        sum(positions)
        /
        len(positions),
        2
    )