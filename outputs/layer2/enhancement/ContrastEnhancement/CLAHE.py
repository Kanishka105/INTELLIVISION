import cv2


def clahe_enhancement(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Apply CLAHE contrast enhancement.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if len(image.shape) == 3:

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(
            lab
        )

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        l_channel = clahe.apply(
            l_channel
        )

        enhanced_lab = cv2.merge(
            [
                l_channel,
                a_channel,
                b_channel
            ]
        )

        return cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)