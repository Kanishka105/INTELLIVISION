import cv2


def gaussian_filter(image, kernel_size=5, sigma=0):
    """
    Apply Gaussian smoothing to an image/frame.
    """

    if image is None:
        raise ValueError("Input image/frame is None.")

    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma
    )