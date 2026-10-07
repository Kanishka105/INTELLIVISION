import cv2
import numpy as np


def color_segmentation(
    image,
    lower=(0, 0, 0),
    upper=(180, 255, 255)
):
    """
    Segment image based on HSV color range.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        lower,
        dtype=np.uint8
    )

    upper = np.array(
        upper,
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    return mask