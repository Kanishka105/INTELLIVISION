import cv2


def convex_hull(contour):

    return cv2.convexHull(
        contour
    )