def crop_plate(image, bbox):
    """
    Crop license plate from image.

    bbox = [x1, y1, x2, y2]
    """

    if image is None:
        return None

    h, w = image.shape[:2]

    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))

    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]