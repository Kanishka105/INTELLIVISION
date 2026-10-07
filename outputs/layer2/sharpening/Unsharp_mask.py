import cv2


def unsharp_mask(
    image,
    sigma=1.0,
    amount=1.5
):

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigma
    )

    sharpened = cv2.addWeighted(
        image,
        1 + amount,
        blurred,
        -amount,
        0
    )

    return sharpened