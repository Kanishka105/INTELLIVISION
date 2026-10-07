import cv2


def high_boost_filter(
    image,
    A=2.0,
    sigma=1.0
):

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigma
    )

    high_boost = cv2.addWeighted(
        image,
        A,
        blurred,
        -(A - 1),
        0
    )

    return high_boost