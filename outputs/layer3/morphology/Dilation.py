import cv2

from structuring_element import create_structuring_element


def dilation(
    image,
    kernel_size=(5, 5),
    shape="ellipse",
    iterations=1
):

    kernel = create_structuring_element(
        shape,
        kernel_size
    )

    result = cv2.dilate(
        image,
        kernel,
        iterations=iterations
    )

    return result