def agreement_score(
    positions
):
    if not positions:
        return 0

    spread = (
        max(positions)
        -
        min(positions)
    )

    return 1 - spread