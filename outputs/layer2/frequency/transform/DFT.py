import cv2
import numpy as np


def dft_transform(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    dft = cv2.dft(
        gray,
        flags=cv2.DFT_COMPLEX_OUTPUT
    )

    dft_shift = np.fft.fftshift(
        dft
    )

    magnitude = cv2.magnitude(
        dft_shift[:, :, 0],
        dft_shift[:, :, 1]
    )

    magnitude = np.log(
        magnitude + 1
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return np.uint8(magnitude)