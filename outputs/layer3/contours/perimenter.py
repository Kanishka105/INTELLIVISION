import cv2


def contour_perimeter(
    contour,
    closed=True
):

    return cv2.arcLength(
        contour,
        closed
    )