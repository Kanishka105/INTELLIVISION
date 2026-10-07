def calculate_direction(
    previous_center,
    current_center,
    threshold=5
):
    """
    Determine movement direction from
    change in center position.
    """

    if (
        previous_center is None
        or current_center is None
    ):
        return "stationary"


    previous_x, previous_y = (
        previous_center
    )

    current_x, current_y = (
        current_center
    )


    dx = current_x - previous_x
    dy = current_y - previous_y


    if abs(dx) < threshold and abs(dy) < threshold:
        return "stationary"


    if abs(dx) > abs(dy):

        if dx > 0:
            return "right"

        return "left"


    if dy > 0:
        return "down"

    return "up"