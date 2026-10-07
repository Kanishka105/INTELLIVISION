import cv2
import numpy as np


def adjust_brightness(image, value=30):
    """
    Increase or decrease image brightness.

    value > 0  -> brighter
    value < 0  -> darker
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Change brightness
    v = np.clip(
        v.astype(np.int16) + value,
        0,
        255
    ).astype(np.uint8)

    enhanced_hsv = cv2.merge([h, s, v])

    result = cv2.cvtColor(
        enhanced_hsv,
        cv2.COLOR_HSV2BGR
    )

    return result