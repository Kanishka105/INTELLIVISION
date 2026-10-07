import math


def calculate_displacement(
    previous_center,
    current_center
):

    if (
        previous_center is None
        or current_center is None
    ):
        return 0.0

    x1, y1 = previous_center

    x2, y2 = current_center

    return math.sqrt(
        (x2 - x1) ** 2
        +
        (y2 - y1) ** 2
    )