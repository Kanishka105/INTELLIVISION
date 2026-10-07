import cv2


def find_contours(
    binary_image,
    mode=cv2.RETR_EXTERNAL,
    method=cv2.CHAIN_APPROX_SIMPLE
):

    contours, hierarchy = cv2.findContours(
        binary_image,
        mode,
        method
    )

    return contours, hierarchy