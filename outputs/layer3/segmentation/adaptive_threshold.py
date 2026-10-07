import cv2


def adaptive_threshold(
    image,
    block_size=11,
    c=2
):
    """
    Apply adaptive Gaussian thresholding.
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    if block_size % 2 == 0:
        raise ValueError(
            "block_size must be odd"
        )

    result = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )

    return result