import cv2
import numpy as np


def log_transform(image):

    image_float = np.float32(image)

    c = 255 / np.log(1 + np.max(image_float))

    log_image = c * np.log(
        1 + image_float
    )

    log_image = np.uint8(
        np.clip(log_image, 0, 255)
    )

    return log_image