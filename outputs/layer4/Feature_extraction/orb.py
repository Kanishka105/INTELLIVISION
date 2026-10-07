import cv2
import numpy as np


def extract_orb_features(
    image,
    max_features=500
):

    output_size = 64

    if image is None or image.size == 0:

        return np.zeros(
            output_size,
            dtype=np.float32
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(
        nfeatures=max_features
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    if descriptors is None:

        return np.zeros(
            output_size,
            dtype=np.float32
        )

    descriptors = descriptors.astype(
        np.float32
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