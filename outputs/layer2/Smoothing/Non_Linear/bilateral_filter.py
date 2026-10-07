import cv2


def bilateral_filter(
    image,
    diameter=9,
    sigma_color=75,
    sigma_space=75
):
    """
    Apply bilateral filtering to an image.

    Parameters:
        image: Input BGR image
        diameter: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Filtered image
    """

    if image is None:
        raise ValueError("Input image is None.")

    return cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space
    )