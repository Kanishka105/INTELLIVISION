import numpy as np


def create_feature_vector(
    descriptors
):

    feature_vector = np.array(
        [
            descriptors["area"],
            descriptors["perimeter"],
            descriptors["width"],
            descriptors["height"],
            descriptors["aspect_ratio"],
            descriptors["circularity"],
            descriptors["solidity"],
            descriptors["extent"]
        ],
        dtype=np.float32
    )

    return feature_vector