import cv2


def solidity(contour):

    area = cv2.contourArea(
        contour
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = cv2.contourArea(
        hull
    )

    if hull_area == 0:
        return 0.0

    return area / hull_area