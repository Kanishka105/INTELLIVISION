import cv2
import numpy as np


def notch_filter(
    image,
    notch_points=None,
    notch_radius=5
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    if notch_points is None:
        notch_points = []

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = gray.shape

    crow = rows // 2
    ccol = cols // 2

    mask = np.ones(
        (rows, cols),
        dtype=np.uint8
    )

    for x, y in notch_points:

        x1 = ccol + x
        y1 = crow + y

        x2 = ccol - x
        y2 = crow - y

        cv2.circle(
            mask,
            (x1, y1),
            notch_radius,
            0,
            -1
        )

        cv2.circle(
            mask,
            (x2, y2),
            notch_radius,
            0,
            -1
        )

    filtered = fft_shift * mask

    inverse_shift = np.fft.ifftshift(
        filtered
    )

    result = np.fft.ifft2(
        inverse_shift
    )

    result = np.abs(result)

    return np.uint8(
        np.clip(result, 0, 255)
    )