import cv2


def bounding_box(contour):

    x, y, width, height = cv2.boundingRect(
        contour
    )

    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }