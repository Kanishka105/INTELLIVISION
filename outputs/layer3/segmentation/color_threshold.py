import cv2
import numpy as np


def color_threshold(
    image,
    lower,
    upper
):

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

    return cv2.inRange(
        hsv,
        lower,
        upper
    )