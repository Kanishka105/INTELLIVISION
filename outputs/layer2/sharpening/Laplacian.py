import cv2
import numpy as np


def laplacian_sharpen(
    image,
    alpha=1.0
):
    """
    Sharpen an image using the Laplacian operator.
    """

    if image is None:
        raise ValueError("Input image is None.")

    # Convert to grayscale only for edge calculation
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

    # Laplacian
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    # Sharpen
    if len(image.shape) == 3:

        result = cv2.addWeighted(
            image,
            1.0,
            cv2.cvtColor(
                laplacian,
                cv2.COLOR_GRAY2BGR
            ),
            -alpha,
            0
        )

    else:

        result = cv2.addWeighted(
            image,
            1.0,
            laplacian,
            -alpha,
            0
        )

    return result