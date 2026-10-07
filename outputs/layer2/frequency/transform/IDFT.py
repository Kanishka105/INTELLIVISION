import cv2
import numpy as np


def inverse_dft(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    dft = cv2.dft(
        gray,
        flags=cv2.DFT_COMPLEX_OUTPUT
    )

    inverse = cv2.idft(
        dft
    )

    result = cv2.magnitude(
        inverse[:, :, 0],
        inverse[:, :, 1]
    )

    result = cv2.normalize(
        result,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(result)