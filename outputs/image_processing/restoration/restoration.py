import cv2


def restore_image(
    image,
    method="bilateral"
):
    """
    Basic image restoration/denoising.
    """

    if image is None:
        raise ValueError("Image is None.")

    if method == "bilateral":

        return cv2.bilateralFilter(
            image,
            9,
            75,
            75
        )

    if method == "median":

        return cv2.medianBlur(
            image,
            5
        )

    if method == "gaussian":

        return cv2.GaussianBlur(
            image,
            (5, 5),
            0
        )

    if method == "nlm":

        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            10,
            10,
            7,
            21
        )

    raise ValueError(
        f"Unknown restoration method: {method}"
    )