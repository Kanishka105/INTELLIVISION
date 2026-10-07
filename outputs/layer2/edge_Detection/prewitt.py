import cv2
import numpy as np


def prewitt_edge_detection(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ], dtype=np.float32)

    prewitt_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    prewitt_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)

    prewitt_x = cv2.convertScaleAbs(prewitt_x)
    prewitt_y = cv2.convertScaleAbs(prewitt_y)

    return cv2.addWeighted(
        prewitt_x,
        0.5,
        prewitt_y,
        0.5,
        0
    )