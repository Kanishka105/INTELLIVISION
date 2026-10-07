import cv2
import numpy as np


def calculate_entropy(gray):

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    total = histogram.sum()

    if total == 0:
        return 0.0

    probability = histogram / total

    probability = probability[
        probability > 0
    ]

    return float(
        -np.sum(
            probability *
            np.log2(probability)
        )
    )


def extract_texture_features(image):

    if image is None or image.size == 0:
        return np.zeros(
            3,
            dtype=np.float32
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mean_value = np.mean(gray)
    std_value = np.std(gray)
    entropy_value = calculate_entropy(gray)

    return np.array(
        [
            mean_value,
            std_value,
            entropy_value
        ],
        dtype=np.float32
    )