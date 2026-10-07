import cv2
import numpy as np


def gamma_correction(image, gamma=1.0):

    normalized = image / 255.0

    corrected = np.power(
        normalized,
        gamma
    )

    corrected = corrected * 255

    return np.uint8(
        np.clip(corrected, 0, 255)
    )