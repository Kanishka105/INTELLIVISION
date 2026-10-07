import cv2
import numpy as np


def inverse_dct(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    dct = cv2.dct(
        gray
    )

    result = cv2.idct(
        dct
    )

    result = np.clip(
        result,
        0,
        255
    )

    return np.uint8(
        result
    )