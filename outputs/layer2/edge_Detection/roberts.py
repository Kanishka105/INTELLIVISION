import cv2
import numpy as np


def roberts_edge_detection(image):

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Roberts kernels
    kernel_x = np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [0, 1],
        [-1, 0]
    ], dtype=np.float32)

    # Apply kernels
    roberts_x = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_x
    )

    roberts_y = cv2.filter2D(
        gray,
        cv2.CV_32F,
        kernel_y
    )

    # Convert to absolute values
    roberts_x = cv2.convertScaleAbs(
        roberts_x
    )

    roberts_y = cv2.convertScaleAbs(
        roberts_y
    )

    # Combine
    result = cv2.addWeighted(
        roberts_x,
        0.5,
        roberts_y,
        0.5,
        0
    )

    return result