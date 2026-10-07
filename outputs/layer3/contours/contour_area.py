import cv2


def contour_area(contour):

    return cv2.contourArea(
        contour
    )