import cv2
import numpy as np


def band_reject_filter(image, low_radius=20, high_radius=60):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = gray.shape

    crow = rows // 2
    ccol = cols // 2

    y, x = np.ogrid[:rows, :cols]

    distance = np.sqrt(
        (x - ccol) ** 2 +
        (y - crow) ** 2
    )

    mask = np.ones(
        (rows, cols),
        dtype=np.uint8
    )

    mask[
        (distance >= low_radius) &
        (distance <= high_radius)
    ] = 0

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