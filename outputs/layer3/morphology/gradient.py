import cv2

from structuring_element import create_structuring_element


def morphological_gradient(
    image,
    kernel_size=(5, 5),
    shape="ellipse"
):

    kernel = create_structuring_element(
        shape,
        kernel_size
    )

    result = cv2.morphologyEx(
        image,
        cv2.MORPH_GRADIENT,
        kernel
    )

    return result