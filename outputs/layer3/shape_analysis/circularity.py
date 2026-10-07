import cv2
import math


def circularity(contour):
    """
    Circularity = 4 * pi * Area / Perimeter^2
    """

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        return 0.0

    return (
        4 * math.pi * area
        / (perimeter ** 2)
    )