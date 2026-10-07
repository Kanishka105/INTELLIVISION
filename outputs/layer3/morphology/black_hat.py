import cv2

from structuring_element import create_structuring_element


def black_hat(
    image,
    kernel_size=(9, 9),
    shape="ellipse"
):

    kernel = create_structuring_element(
        shape,
        kernel_size
    )

    result = cv2.morphologyEx(
        image,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    return result