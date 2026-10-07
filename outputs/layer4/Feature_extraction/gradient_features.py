import cv2
import numpy as np


def extract_gradient_features(
    image,
    bins=16
):

    if image is None or image.size == 0:

        return np.zeros(
            bins + 3,
            dtype=np.float32
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = cv2.magnitude(
        gx,
        gy
    )

    mean_gradient = np.mean(
        magnitude
    )

    std_gradient = np.std(
        magnitude
    )

    max_gradient = np.max(
        magnitude
    )

    histogram = cv2.calcHist(
        [magnitude],
        [0],
        None,
        [bins],
        [0, np.max(magnitude) + 1]
    )

    histogram = cv2.normalize(
        histogram,
        histogram
    ).flatten()

    return np.concatenate(
        [
            np.array(
                [
                    mean_gradient,
                    std_gradient,
                    max_gradient
                ],
                dtype=np.float32
            ),
            histogram.astype(
                np.float32
            )
        ]
    )