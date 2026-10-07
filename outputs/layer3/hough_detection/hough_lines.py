import cv2
import numpy as np


def hough_lines(
    image,
    threshold=100
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLines(
        edges,
        1,
        np.pi / 180,
        threshold
    )

    return lines