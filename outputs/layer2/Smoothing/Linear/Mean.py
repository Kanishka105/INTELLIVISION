import cv2


def mean_filter(
    image,
    kernel_size=(3, 3)
):
    """
    Apply a linear mean/average filter.

    Each output pixel is the average
    of its neighborhood.
    """

    return cv2.blur(
        image,
        kernel_size
    )