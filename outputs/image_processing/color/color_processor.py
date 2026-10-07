import cv2


def adjust_brightness(
    image,
    value=0
):
    """
    Increase or decrease image brightness.
    """

    if image is None:
        raise ValueError("Image is None.")

    result = image.astype("int16")

    result = result + value

    result = result.clip(
        0,
        255
    ).astype("uint8")

    return result


def adjust_saturation(
    image,
    factor=1.0
):
    """
    Adjust saturation using HSV.
    factor = 1.0 means unchanged.
    """

    if image is None:
        raise ValueError("Image is None.")

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = (
        hsv[:, :, 1]
        .astype("float32")
    )

    saturation *= factor

    hsv[:, :, 1] = saturation.clip(
        0,
        255
    ).astype("uint8")

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


def hsv_process(
    image,
    hue_shift=0
):
    """
    Shift the hue channel.
    """

    if image is None:
        raise ValueError("Image is None.")

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hue = (
        hsv[:, :, 0].astype("int16")
    )

    hue = (hue + hue_shift) % 180

    hsv[:, :, 0] = hue.astype("uint8")

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


def process_color(
    image,
    brightness=0,
    saturation=1.0,
    hue_shift=0
):
    """
    Complete color processing.
    """

    image = adjust_brightness(
        image,
        brightness
    )

    image = adjust_saturation(
        image,
        saturation
    )

    image = hsv_process(
        image,
        hue_shift
    )

    return image