import cv2
import numpy as np


def extract_sift_features(image):

    output_size = 256

    if image is None or image.size == 0:

        return np.zeros(
            output_size,
            dtype=np.float32
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    if descriptors is None:

        return np.zeros(
            output_size,
            dtype=np.float32
        )

    mean_descriptor = np.mean(
        descriptors,
        axis=0
    )

    std_descriptor = np.std(
        descriptors,
        axis=0
    )

    feature_vector = np.concatenate(
        [
            mean_descriptor,
            std_descriptor
        ]
    )

    return feature_vector.astype(
        np.float32
    )