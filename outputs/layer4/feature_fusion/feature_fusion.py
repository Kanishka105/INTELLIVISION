import numpy as np


def fuse_features(
    shape_features,
    color_features,
    texture_features,
    gradient_features,
    sift_features,
    orb_features,
    hog_features
):

    shape_features = np.asarray(
        shape_features,
        dtype=np.float32
    ).flatten()

    color_features = np.asarray(
        color_features,
        dtype=np.float32
    ).flatten()

    texture_features = np.asarray(
        texture_features,
        dtype=np.float32
    ).flatten()

    gradient_features = np.asarray(
        gradient_features,
        dtype=np.float32
    ).flatten()

    sift_features = np.asarray(
        sift_features,
        dtype=np.float32
    ).flatten()

    orb_features = np.asarray(
        orb_features,
        dtype=np.float32
    ).flatten()

    hog_features = np.asarray(
        hog_features,
        dtype=np.float32
    ).flatten()

    final_vector = np.concatenate(
        [
            shape_features,
            color_features,
            texture_features,
            gradient_features,
            sift_features,
            orb_features,
            hog_features
        ]
    )

    return final_vector.astype(
        np.float32
    )