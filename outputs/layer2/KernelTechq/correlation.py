import cv2
import numpy as np


def correlation(image, kernel):

    kernel = np.asarray(kernel, dtype=np.float32)

    return cv2.filter2D(
        image,
        -1,
        kernel
    )