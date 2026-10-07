import cv2
import numpy as np


def hsv_enhancement(
    image,
    brightness=0,
    saturation=1.0,
    hue_shift=0
):
    """
    HSV-based image enhancement.

    brightness:
        positive → brighter
        negative → darker

    saturation:
        1.0 → original
        >1.0 → more colorful
        <1.0 → less colorful

    hue_shift:
        changes the color tone
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    # Brightness
    v = np.clip(
        v.astype(np.int16) + brightness,
        0,
        255
    ).astype(np.uint8)

    # Saturation
    s = np.clip(
        s.astype(np.float32) * saturation,
        0,
        255
    ).astype(np.uint8)

    # Hue
    h = (
        h.astype(np.int16) + hue_shift
    ) % 180

    h = h.astype(np.uint8)

    enhanced_hsv = cv2.merge([
        h,
        s,
        v
    ])

    result = cv2.cvtColor(
        enhanced_hsv,
        cv2.COLOR_HSV2BGR
    )

    return result