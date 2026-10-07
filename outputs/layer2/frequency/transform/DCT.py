import cv2
import numpy as np


def dct_transform(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    dct = cv2.dct(
        gray
    )

    dct_display = np.log(
        np.abs(dct) + 1
    )

    dct_display = cv2.normalize(
        dct_display,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(
        dct_display
    )