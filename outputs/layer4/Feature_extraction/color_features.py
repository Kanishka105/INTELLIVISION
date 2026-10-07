import cv2
import numpy as np


def extract_color_features(
    image,
    bins=8
):
    """
    Extract color histogram features
    from B, G and R channels.

    8 bins per channel:
    8 + 8 + 8 = 24 features
    """

    if image is None or image.size == 0:

        return np.zeros(
            bins * 3,
            dtype=np.float32
        )

    features = []

    channels = cv2.split(image)

    for channel in channels:

        histogram = cv2.calcHist(
            [channel],
            [0],
            None,
            [bins],
            [0, 256]
        )

        histogram = cv2.normalize(
            histogram,
            histogram
        )

        features.extend(
            histogram.flatten()
        )

    return np.asarray(
        features,
        dtype=np.float32
    )
# How much dark color?
# How much bright color?
# Which color ranges dominate?
# 8 B + 8 G + 8 R = 24 features