import cv2
import numpy as np


def filter_regions(
    binary_image,
    min_area=100
):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_image,
            connectivity=8
        )
    )

    result = np.zeros_like(
        binary_image
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:

            result[
                labels == label
            ] = 255

    return result