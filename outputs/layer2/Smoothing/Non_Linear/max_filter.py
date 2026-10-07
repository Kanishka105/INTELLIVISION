import cv2
import numpy as np


def max_filter(
    image,
    kernel_size=3
):
    """
    Apply a maximum filter.

    The output pixel becomes the maximum
    value in its neighborhood.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8
    )

    return cv2.dilate(
        image,
        kernel
    )