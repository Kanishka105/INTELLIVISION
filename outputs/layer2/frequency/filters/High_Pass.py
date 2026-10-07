import cv2
import numpy as np


def high_pass_filter(image, radius=30):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Start with everything kept
    mask = np.ones((rows, cols), np.uint8)

    # Remove low frequencies
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    # Apply filter
    filtered = fshift * mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(filtered)
    result = np.fft.ifft2(f_ishift)

    result = np.abs(result)

    return np.uint8(np.clip(result, 0, 255))