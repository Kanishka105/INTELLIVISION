import cv2


def extent(contour):
    """
    Extent = contour area / bounding box area
    """

    area = cv2.contourArea(
        contour
    )

    x, y, width, height = cv2.boundingRect(
        contour
    )

    bounding_box_area = (
        width * height
    )

    if bounding_box_area == 0:
        return 0.0

    return (
        area / bounding_box_area
    )