import cv2


def scharr_edge_detection(image):

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Scharr X
    scharr_x = cv2.Scharr(
        gray,
        cv2.CV_64F,
        1,
        0
    )

    # Scharr Y
    scharr_y = cv2.Scharr(
        gray,
        cv2.CV_64F,
        0,
        1
    )

    # Convert to absolute values
    scharr_x = cv2.convertScaleAbs(
        scharr_x
    )

    scharr_y = cv2.convertScaleAbs(
        scharr_y
    )

    # Combine
    result = cv2.addWeighted(
        scharr_x,
        0.5,
        scharr_y,
        0.5,
        0
    )

    return result