import cv2


def laplacian_edge_detection(image):

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Laplacian
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    # Convert to displayable image
    result = cv2.convertScaleAbs(
        laplacian
    )

    return result