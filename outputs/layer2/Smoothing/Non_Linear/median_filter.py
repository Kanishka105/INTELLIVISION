import cv2


def median_filter(
    image,
    kernel_size=5
):
    """
    Apply median filtering to remove
    salt-and-pepper noise.
    """

    if image is None:
        raise ValueError("Input image is None.")

    if kernel_size < 3:
        kernel_size = 3

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.medianBlur(
        image,
        kernel_size
    )