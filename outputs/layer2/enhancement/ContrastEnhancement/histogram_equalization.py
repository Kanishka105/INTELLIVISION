import cv2


def histogram_equalization(image):
    """
    Apply histogram equalization.

    For a color image, equalization is applied
    only to the luminance channel to preserve color.
    """

    if image is None:
        raise ValueError("Input image is None.")

    # Grayscale image
    if len(image.shape) == 2:
        return cv2.equalizeHist(image)

    # Color image
    ycrcb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2YCrCb
    )

    y_channel, cr, cb = cv2.split(
        ycrcb
    )

    y_channel = cv2.equalizeHist(
        y_channel
    )

    enhanced = cv2.merge(
        [
            y_channel,
            cr,
            cb
        ]
    )

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_YCrCb2BGR
    )