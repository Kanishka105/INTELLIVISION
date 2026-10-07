import cv2
import numpy as np


def min_filter(
    image,
    kernel_size=3
):
    """
    Apply a minimum filter.

    The output pixel becomes the minimum
    value in its neighborhood.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8
    )

    return cv2.erode(
        image,
        kernel
    )