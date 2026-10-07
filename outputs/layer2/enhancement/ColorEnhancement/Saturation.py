import cv2
import numpy as np


def adjust_saturation(image, factor=1.5):
    """
    Adjust color saturation.

    factor = 1.0 → original
    factor > 1.0 → more colorful
    factor < 1.0 → less colorful
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Change saturation
    s = np.clip(
        s.astype(np.float32) * factor,
        0,
        255
    ).astype(np.uint8)

    enhanced_hsv = cv2.merge([h, s, v])

    result = cv2.cvtColor(
        enhanced_hsv,
        cv2.COLOR_HSV2BGR
    )

    return result