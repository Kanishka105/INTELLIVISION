import cv2


def crop_detection(image, detection):
    """
    Crop one detected object from an image.

    detection must contain:
        detection["bbox"] = [x1, y1, x2, y2]
    """

    height, width = image.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        detection["bbox"]
    )

    # Keep coordinates inside image
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    return crop