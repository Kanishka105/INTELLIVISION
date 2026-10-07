import cv2
import numpy as np


def sobel_edge_detection(image):
    """
    Detect edges using the Sobel operator.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = image

    gray = np.asarray(
        gray,
        dtype=np.uint8
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx.astype(np.float32),
        gy.astype(np.float32)
    )

    magnitude = cv2.convertScaleAbs(
        magnitude
    )

    return magnitude