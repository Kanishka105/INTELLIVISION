import cv2
import numpy as np


def hough_circles(
    image,
    dp=1,
    min_dist=20,
    param1=50,
    param2=30,
    min_radius=5,
    max_radius=100
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        1
    )

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:

        circles = np.round(
            circles[0]
        ).astype(int)

    return circles