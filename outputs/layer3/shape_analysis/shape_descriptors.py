import cv2
import math


def calculate_shape_descriptors(
    contour
):

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, width, height = cv2.boundingRect(
        contour
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    if height != 0:
        aspect_ratio_value = width / height
    else:
        aspect_ratio_value = 0.0

    if perimeter != 0:
        circularity_value = (
            4 * math.pi * area
            / (perimeter ** 2)
        )
    else:
        circularity_value = 0.0

    if hull_area != 0:
        solidity_value = (
            area / hull_area
        )
    else:
        solidity_value = 0.0

    box_area = width * height

    if box_area != 0:
        extent_value = (
            area / box_area
        )
    else:
        extent_value = 0.0

    return {

        "area": float(area),

        "perimeter": float(perimeter),

        "x": int(x),

        "y": int(y),

        "width": int(width),

        "height": int(height),

        "aspect_ratio":
            float(aspect_ratio_value),

        "circularity":
            float(circularity_value),

        "solidity":
            float(solidity_value),

        "extent":
            float(extent_value)
    }