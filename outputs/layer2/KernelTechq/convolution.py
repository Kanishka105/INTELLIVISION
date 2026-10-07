import cv2
import numpy as np


def convolution(image, kernel):

    kernel = np.asarray(
        kernel,
        dtype=np.float32
    )

    flipped_kernel = cv2.flip(
        kernel,
        -1
    )

    return cv2.filter2D(
        image,
        -1,
        flipped_kernel
    )