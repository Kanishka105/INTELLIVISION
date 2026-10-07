import cv2
import numpy as np


def fft_transform(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    fft = np.fft.fft2(
        gray
    )

    fft_shift = np.fft.fftshift(
        fft
    )

    magnitude = np.abs(
        fft_shift
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