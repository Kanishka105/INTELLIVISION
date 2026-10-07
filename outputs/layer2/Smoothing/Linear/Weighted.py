import cv2
import numpy as np


def weighted_mean_filter(image):

    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32)

    kernel = kernel / kernel.sum()

    return cv2.filter2D(
        image,
        -1,
        kernel
    )