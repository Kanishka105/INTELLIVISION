import cv2
import numpy as np


def extract_hog_features(
    image,
    image_size=(64, 64),
    cell_size=8,
    block_size=2,
    bins=9
):
    """
    Extract Histogram of Oriented Gradients (HOG)
    features manually using NumPy.

    64x64 image
    8x8 cells
    2x2 cells per block
    9 orientation bins

    Final feature length:
    7 x 7 x 2 x 2 x 9 = 1764
    """

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if image is None or image.size == 0:
        return np.zeros(
            1764,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    resized = cv2.resize(
        image,
        image_size
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype(
        np.float32
    )

    # --------------------------------------------------------
    # GRADIENTS
    # --------------------------------------------------------

    gx = np.zeros_like(
        gray,
        dtype=np.float32
    )

    gy = np.zeros_like(
        gray,
        dtype=np.float32
    )

    gx[:, 1:-1] = (
        gray[:, 2:] -
        gray[:, :-2]
    )

    gy[1:-1, :] = (
        gray[2:, :] -
        gray[:-2, :]
    )

    # --------------------------------------------------------
    # MAGNITUDE + ANGLE
    # --------------------------------------------------------

    magnitude = np.sqrt(
        gx ** 2 +
        gy ** 2
    )

    angle = (
        np.degrees(
            np.arctan2(
                gy,
                gx
            )
        ) % 180
    )

    # --------------------------------------------------------
    # CELL HISTOGRAMS
    # --------------------------------------------------------

    height, width = gray.shape

    cells_y = height // cell_size
    cells_x = width // cell_size

    cell_histograms = np.zeros(
        (
            cells_y,
            cells_x,
            bins
        ),
        dtype=np.float32
    )

    bin_width = 180 / bins

    for cy in range(cells_y):

        for cx in range(cells_x):

            y_start = cy * cell_size
            y_end = y_start + cell_size

            x_start = cx * cell_size
            x_end = x_start + cell_size

            cell_magnitude = magnitude[
                y_start:y_end,
                x_start:x_end
            ]

            cell_angle = angle[
                y_start:y_end,
                x_start:x_end
            ]

            # Determine orientation bins
            bin_indices = (
                cell_angle /
                bin_width
            ).astype(int)

            bin_indices = np.clip(
                bin_indices,
                0,
                bins - 1
            )

            for b in range(bins):

                mask = (
                    bin_indices == b
                )

                cell_histograms[
                    cy,
                    cx,
                    b
                ] = np.sum(
                    cell_magnitude[mask]
                )

    # --------------------------------------------------------
    # BLOCK NORMALIZATION
    # --------------------------------------------------------

    features = []

    blocks_y = cells_y - block_size + 1
    blocks_x = cells_x - block_size + 1

    for by in range(blocks_y):

        for bx in range(blocks_x):

            block = cell_histograms[
                by:by + block_size,
                bx:bx + block_size,
                :
            ]

            block_vector = block.flatten()

            # L2 normalization
            norm = np.sqrt(
                np.sum(
                    block_vector ** 2
                ) + 1e-6
            )

            normalized_block = (
                block_vector / norm
            )

            features.extend(
                normalized_block
            )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    return features
# Histogram of Oriented Gradients (HOG) features are used to describe the shape and structure of objects in an image. HOG captures the distribution of gradient orientations in localized regions, making it effective for object detection and recognition tasks.
# For vehicle recognition, HOG is useful because object shape is strongly related to edge structure.