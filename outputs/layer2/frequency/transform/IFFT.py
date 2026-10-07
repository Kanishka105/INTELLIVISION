import cv2
import numpy as np


def inverse_fft(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    fft = np.fft.fft2(
        gray
    )

    result = np.fft.ifft2(
        fft
    )

    result = np.abs(
        result
    )

    return np.uint8(
        np.clip(result, 0, 255)
    )