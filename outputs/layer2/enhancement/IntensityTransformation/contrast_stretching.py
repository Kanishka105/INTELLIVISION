import cv2
import numpy as np


def contrast_stretching(image):

    image_float = np.float32(image)

    min_value = np.min(image_float)
    max_value = np.max(image_float)

    stretched = (
        (image_float - min_value)
        /
        (max_value - min_value)
    ) * 255

    stretched = np.uint8(
        np.clip(stretched, 0, 255)
    )

    return stretched