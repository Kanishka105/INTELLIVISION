import cv2
import numpy as np


def create_structuring_element(
    shape="ellipse",
    kernel_size=(5, 5)
):
    """
    Create a morphological structuring element.

    shape:
        "rect"    -> rectangular kernel
        "ellipse" -> elliptical kernel
        "cross"   -> cross-shaped kernel

    kernel_size:
        (height, width)
    """

    shape = shape.lower()

    if shape == "rect":
        kernel_shape = cv2.MORPH_RECT

    elif shape == "ellipse":
        kernel_shape = cv2.MORPH_ELLIPSE

    elif shape == "cross":
        kernel_shape = cv2.MORPH_CROSS

    else:
        raise ValueError(
            "Shape must be 'rect', 'ellipse', or 'cross'"
        )

    kernel = cv2.getStructuringElement(
        kernel_shape,
        kernel_size
    )

    return kernel


if __name__ == "__main__":

    kernel = create_structuring_element(
        shape="ellipse",
        kernel_size=(5, 5)
    )

    print("Structuring Element:")
    print(kernel)